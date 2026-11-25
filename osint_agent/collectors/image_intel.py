"""Image EXIF/GPS forensics, perceptual avatar matching, and OpenStreetMap Overpass/Nominatim geo recon."""

from __future__ import annotations

import io
import logging
import math
from typing import Any

import exifread
import httpx
from PIL import Image

from osint_agent.collectors.enrich import is_private_or_loopback_host
from osint_agent.collectors.tools import DEFAULT_USER_AGENT
from osint_agent.core.schema import EntityType, Hit

logger = logging.getLogger(__name__)


def _dms_to_decimal(dms_values: list[Any], ref: str) -> float | None:
    """Convert EXIF degrees/minutes/seconds Ratio list into signed decimal degrees."""
    try:
        d = float(dms_values[0].num) / float(dms_values[0].den)
        m = float(dms_values[1].num) / float(dms_values[1].den)
        s = float(dms_values[2].num) / float(dms_values[2].den)
        decimal = d + (m / 60.0) + (s / 3600.0)
        if str(ref).upper() in ("S", "W"):
            decimal = -decimal
        return round(decimal, 6)
    except Exception:
        return None


def extract_image_metadata(image_bytes: bytes) -> dict[str, Any]:
    """Extract EXIF metadata, camera info, author/artist, and GPS coordinates from image bytes."""
    meta: dict[str, Any] = {
        "width": None,
        "height": None,
        "format": None,
        "camera_make": None,
        "camera_model": None,
        "datetime": None,
        "software": None,
        "artist": None,
        "gps": None,
    }

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            meta["width"], meta["height"] = img.size
            meta["format"] = img.format
    except Exception as exc:
        logger.debug("Pillow failed to open image: %s", exc)

    try:
        tags = exifread.process_file(io.BytesIO(image_bytes), details=False)
        if "Image Make" in tags:
            meta["camera_make"] = str(tags["Image Make"]).strip()
        if "Image Model" in tags:
            meta["camera_model"] = str(tags["Image Model"]).strip()
        if "EXIF DateTimeOriginal" in tags:
            meta["datetime"] = str(tags["EXIF DateTimeOriginal"]).strip()
        elif "Image DateTime" in tags:
            meta["datetime"] = str(tags["Image DateTime"]).strip()
        if "Image Software" in tags:
            meta["software"] = str(tags["Image Software"]).strip()
        if "Image Artist" in tags:
            meta["artist"] = str(tags["Image Artist"]).strip()

        lat_tag = tags.get("GPS GPSLatitude")
        lat_ref = tags.get("GPS GPSLatitudeRef")
        lon_tag = tags.get("GPS GPSLongitude")
        lon_ref = tags.get("GPS GPSLongitudeRef")

        if lat_tag and lat_ref and lon_tag and lon_ref:
            lat = _dms_to_decimal(lat_tag.values, str(lat_ref))
            lon = _dms_to_decimal(lon_tag.values, str(lon_ref))
            if lat is not None and lon is not None:
                meta["gps"] = {"lat": lat, "lon": lon}
    except Exception as exc:
        logger.debug("EXIF parsing failed: %s", exc)

    return meta


def compute_dhash(image_bytes: bytes, hash_size: int = 8) -> int | None:
    """Compute a 64-bit difference hash (dHash) for fast perceptual image comparison."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = list(gray.get_flattened_data() if hasattr(gray, "get_flattened_data") else gray.getdata())
            diff_bits = 0
            bit_idx = 0
            for row in range(hash_size):
                row_start = row * (hash_size + 1)
                for col in range(hash_size):
                    if pixels[row_start + col] > pixels[row_start + col + 1]:
                        diff_bits |= 1 << bit_idx
                    bit_idx += 1
            return diff_bits
    except Exception:
        return None


def compute_color_histogram_sim(bytes_a: bytes, bytes_b: bytes) -> float:
    """Compute cosine similarity between normalized RGB histograms of two images."""
    try:
        with Image.open(io.BytesIO(bytes_a)) as img_a, Image.open(io.BytesIO(bytes_b)) as img_b:
            hist_a = img_a.convert("RGB").resize((64, 64)).histogram()
            hist_b = img_b.convert("RGB").resize((64, 64)).histogram()
            dot = sum(a * b for a, b in zip(hist_a, hist_b))
            norm_a = math.sqrt(sum(a * a for a in hist_a))
            norm_b = math.sqrt(sum(b * b for b in hist_b))
            if norm_a == 0.0 or norm_b == 0.0:
                return 0.0
            return round(max(0.0, min(1.0, dot / (norm_a * norm_b))), 4)
    except Exception:
        return 0.0


def compare_image_bytes(bytes_a: bytes, bytes_b: bytes) -> float:
    """Compare two images combining perceptual dHash similarity and RGB histogram cosine."""
    h1 = compute_dhash(bytes_a)
    h2 = compute_dhash(bytes_b)
    if h1 is None or h2 is None:
        return 0.0

    hamming = bin(h1 ^ h2).count("1")
    dhash_sim = max(0.0, 1.0 - (hamming / 64.0))
    hist_sim = compute_color_histogram_sim(bytes_a, bytes_b)
    return round(0.65 * dhash_sim + 0.35 * hist_sim, 4)


async def compare_reference_to_avatar_url(
    ref_bytes: bytes,
    avatar_url: str,
    client: httpx.AsyncClient,
) -> float:
    """Fetch a public profile avatar safely and score visual similarity against reference image."""
    from urllib.parse import urlparse

    host = urlparse(avatar_url).netloc.lower()
    if not host or is_private_or_loopback_host(host):
        return 0.0

    try:
        resp = await client.get(
            avatar_url,
            headers={"User-Agent": DEFAULT_USER_AGENT},
            follow_redirects=True,
            timeout=5.0,
        )
        if resp.status_code == 200 and len(resp.content) <= 2_000_000:
            return compare_image_bytes(ref_bytes, resp.content)
    except Exception:
        pass
    return 0.0


async def reverse_geocode_and_overpass(
    lat: float,
    lon: float,
    target_id: str,
    client: httpx.AsyncClient,
) -> list[Hit]:
    """Reverse-geocode GPS coordinates via Nominatim and query nearby landmarks via Overpass API."""
    hits: list[Hit] = []
    seen: set[str] = set()

    # 1. OpenStreetMap Nominatim Reverse Geocoding
    try:
        resp = await client.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": str(lat), "lon": str(lon), "format": "jsonv2", "addressdetails": "1"},
            headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"},
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            display_name = data.get("display_name")
            if display_name and display_name.lower() not in seen:
                seen.add(display_name.lower())
                hits.append(
                    Hit(
                        target_entity_id=target_id,
                        discovered_type=EntityType.PLACE,
                        discovered_value=display_name,
                        relation="gps_location",
                        source_tool="image_intel",
                        is_trusted=True,
                        raw_data={"lat": lat, "lon": lon, "address": data.get("address", {})},
                        metadata={"lat": lat, "lon": lon},
                    )
                )
            addr = data.get("address", {})
            for key in ("city", "town", "state", "country"):
                val = addr.get(key)
                if val and isinstance(val, str) and val.lower() not in seen:
                    seen.add(val.lower())
                    hits.append(
                        Hit(
                            target_entity_id=target_id,
                            discovered_type=EntityType.PLACE,
                            discovered_value=val.strip(),
                            relation="located_in",
                            source_tool="image_intel",
                            is_trusted=True,
                            raw_data={"lat": lat, "lon": lon, "admin_level": key},
                            metadata={"lat": lat, "lon": lon},
                        )
                    )
    except Exception as exc:
        logger.debug("Reverse geocode failed for (%s, %s): %s", lat, lon, exc)

    # 2. Overpass API nearby named POIs / landmarks (500m radius)
    overpass_query = (
        f'[out:json][timeout:6];'
        f'(node(around:500,{lat},{lon})["name"];way(around:500,{lat},{lon})["name"];);'
        f'out center 5;'
    )
    try:
        ov_resp = await client.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": overpass_query},
            headers={"User-Agent": DEFAULT_USER_AGENT},
            timeout=8.0,
        )
        if ov_resp.status_code == 200:
            elements = ov_resp.json().get("elements", [])
            for el in elements[:5]:
                tags = el.get("tags", {})
                poi_name = tags.get("name")
                if poi_name and poi_name.lower() not in seen:
                    seen.add(poi_name.lower())
                    hits.append(
                        Hit(
                            target_entity_id=target_id,
                            discovered_type=EntityType.PLACE,
                            discovered_value=poi_name.strip(),
                            relation="nearby_landmark",
                            source_tool="image_intel",
                            is_trusted=True,
                            raw_data={"tags": tags, "lat": lat, "lon": lon},
                            metadata={"poi_type": tags.get("amenity") or tags.get("building") or "landmark"},
                        )
                    )
    except Exception as exc:
        logger.debug("Overpass query failed for (%s, %s): %s", lat, lon, exc)

    return hits
