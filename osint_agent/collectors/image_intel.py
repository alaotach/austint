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
            info_clues: list[str] = []
            for k in ("Author", "Description", "Comment", "Copyright", "Source", "URL", "Title", "author", "description", "comment"):
                val = img.info.get(k)
                if isinstance(val, str) and val.strip():
                    info_clues.append(val.strip()[:250])
            if info_clues:
                meta["text_clues"] = info_clues
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
        for extra_tag in ("Image ImageDescription", "Image Copyright", "EXIF UserComment"):
            if extra_tag in tags:
                val = str(tags[extra_tag]).strip()
                if val and val not in meta.get("text_clues", []):
                    meta.setdefault("text_clues", []).append(val[:250])

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


def _descriptive_filename_query(filename: str | None) -> str:
    """Extract a clean human/entity query from an uploaded filename if it is not generic camera/hash junk."""
    import os
    import re

    if not filename:
        return ""
    stem = os.path.splitext(os.path.basename(filename))[0].strip()
    if not stem or len(stem) < 3:
        return ""
    # Skip generic camera / screenshot / UUID / hex names
    if re.match(
        r"^(?:img|dsc|pxl|vid|mvimg|screenshot|screen_shot|photo|image|media|download|untitled|whatsapp|snap)[_-]?\d*",
        stem,
        re.IGNORECASE,
    ):
        return ""
    if re.fullmatch(r"[0-9a-fA-F_-]{12,}", stem):
        return ""
    cleaned = re.sub(r"[_\-.]+", " ", stem).strip()
    return cleaned[:60] if len(cleaned) >= 3 else ""


async def analyze_image_vision(
    image_bytes: bytes,
    filename: str | None = None,
    meta: dict[str, Any] | None = None,
    llm_client: Any = None,
) -> dict[str, Any]:
    """Use multimodal Vision LLM (via base64 data URI) to identify people, text, handles, and search queries."""
    import asyncio
    import base64
    from osint_agent.collectors.profile_analyzer import _resolve_optional_llm_client

    active_llm = _resolve_optional_llm_client(llm_client)
    if active_llm is None or not hasattr(active_llm, "complete_multimodal_json"):
        return {}

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb = img.convert("RGB")
            rgb.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return {}

    data_uri = f"data:image/jpeg;base64,{b64}"
    system = (
        "You are an OSINT visual forensics & reverse-image analysis assistant. Inspect the image carefully and extract "
        "any visible text (OCR), usernames/handles, emails, domains, organizations, landmarks, or recognizable public figures. "
        "Suggest 1-3 concise web search queries to find the user or origin of this image online. "
        "Return valid JSON only: {\"visual_description\": str, \"visible_text\": str, \"detected_names\": [str], "
        "\"detected_usernames\": [str], \"detected_emails\": [str], \"detected_domains\": [str], "
        "\"detected_organizations\": [str], \"landmarks_or_locations\": [str], \"search_queries\": [str]}."
    )
    user = f"Analyze this uploaded image (filename={filename or 'unknown'}, exif_artist={(meta or {}).get('artist') or ''})."

    try:
        res = await asyncio.to_thread(
            active_llm.complete_multimodal_json, system, user, [data_uri], 2048
        )
        return res if isinstance(res, dict) else {}
    except Exception as exc:
        logger.debug("Vision image analysis skipped: %s", exc)
        return {}


async def reverse_image_search_yandex(
    image_bytes: bytes,
    filename: str | None,
    target_id: str,
    client: httpx.AsyncClient,
    max_results: int = 10,
) -> list[Hit]:
    """Upload image bytes to Yandex CBIR and parse matching web pages from CbirSites JSON."""
    import html as _html
    import json
    import re
    from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

    cbir_headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    def _clean_yandex_url(raw_url: str) -> str:
        p = urlparse(raw_url)
        kept = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if k not in ("utm_source", "utm_medium")]
        return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(kept), p.fragment))

    def _find_sites(obj: Any) -> list[dict[str, Any]]:
        if isinstance(obj, dict):
            if "sites" in obj and isinstance(obj["sites"], list):
                return [s for s in obj["sites"] if isinstance(s, dict)]
            for v in obj.values():
                res = _find_sites(v)
                if res:
                    return res
        elif isinstance(obj, list):
            for v in obj:
                res = _find_sites(v)
                if res:
                    return res
        return []

    hits: list[Hit] = []
    try:
        upload_resp = await client.post(
            'https://yandex.com/images/search?rpt=imageview&format=json&request=%7B%22blocks%22%3A%5B%7B%22block%22%3A%22b-page_type_search-by-image__link%22%7D%5D%7D',
            headers=cbir_headers,
            files={"upfile": (filename or "image.jpg", image_bytes, "image/jpeg")},
            follow_redirects=True,
            timeout=15.0,
        )
        if upload_resp.status_code != 200:
            return []
        up_data = upload_resp.json()
        cbir_id = ""
        for blk in up_data.get("blocks") or []:
            if isinstance(blk, dict):
                cbir_id = str((blk.get("params") or {}).get("cbirId") or "")
                if cbir_id:
                    break
        if not cbir_id:
            return []

        sites_resp = await client.get(
            f"https://yandex.com/images/search?rpt=imageview&cbir_id={quote(cbir_id, safe='/')}&cbir_page=sites",
            headers=cbir_headers,
            follow_redirects=True,
            timeout=15.0,
        )
        if sites_resp.status_code != 200 or "showcaptcha" in str(sites_resp.url).lower():
            return []

        sites: list[dict[str, Any]] = []
        for m in re.finditer(r'data-state="([^"]+)"', sites_resp.text):
            raw_state = _html.unescape(m.group(1))
            if "CbirSites" in raw_state or '"sites"' in raw_state:
                try:
                    parsed = json.loads(raw_state)
                    sites = _find_sites(parsed)
                    if sites:
                        break
                except Exception:
                    continue

        _SKIP_CBIR_DOMAINS = ("xhamster", "stripchat", "xlivesex", "chaturbate", "pornhub", "xvideos")
        _MIN_CBIR_SIM = 0.72
        seen_urls: set[str] = set()
        domain_counts: dict[str, int] = {}
        candidates: list[Hit] = []
        for s in sites:
            raw_u = str(s.get("url") or "").strip()
            if not raw_u.startswith(("http://", "https://")):
                continue
            clean_u = _clean_yandex_url(raw_u)
            if clean_u in seen_urls:
                continue
            dom = str(s.get("domain") or urlparse(clean_u).netloc.lower()).lower()
            if any(bad in dom for bad in _SKIP_CBIR_DOMAINS):
                continue
            if domain_counts.get(dom, 0) >= 2:
                continue
            seen_urls.add(clean_u)
            domain_counts[dom] = domain_counts.get(dom, 0) + 1

            title = str(s.get("title") or "").strip()
            desc = str(s.get("description") or "").strip()
            thumb_u = str(((s.get("thumb") or {}).get("url")) or "").strip()
            if thumb_u.startswith("//"):
                thumb_u = "https:" + thumb_u

            candidates.append(
                Hit(
                    target_entity_id=target_id,
                    discovered_type=EntityType.PROFILE,
                    discovered_value=clean_u,
                    relation="reverse_image_match",
                    source_tool="yandex_cbir",
                    is_trusted=False,
                    raw_data={
                        "site": dom,
                        "url": clean_u,
                        "title": title,
                        "description": f"{title} — {desc}".strip(" —")[:300],
                    },
                    metadata={
                        "site": dom,
                        "url": clean_u,
                        "title": title,
                        "bio": desc[:400] if desc else title,
                        **({"thumbnail_url": thumb_u} if thumb_u.startswith("http") else {}),
                    },
                )
            )
            if len(candidates) >= max_results * 2:
                break

        # Verify perceptual similarity against uploaded image so visually similar strangers are rejected
        if candidates:
            import asyncio as _asyncio

            async def _score_cbir(c_hit: Hit) -> Hit | None:
                tu = str(c_hit.metadata.get("thumbnail_url") or "")
                if not tu:
                    return None
                sim = await compare_reference_to_avatar_url(image_bytes, tu, client)
                if sim >= _MIN_CBIR_SIM:
                    c_hit.metadata["visual_similarity"] = sim
                    c_hit.metadata["avatar_match"] = sim >= 0.78
                    c_hit.is_trusted = sim >= 0.78
                    return c_hit
                return None

            scored = await _asyncio.gather(*(_score_cbir(ch) for ch in candidates), return_exceptions=True)
            for res in scored:
                if isinstance(res, Hit):
                    hits.append(res)
                    if len(hits) >= max_results:
                        break
    except Exception as exc:
        logger.debug("Yandex CBIR reverse image search failed: %s", exc)

    return hits


async def search_image_web(
    image_bytes: bytes,
    filename: str | None,
    meta: dict[str, Any],
    vision_info: dict[str, Any],
    target_id: str,
    client: httpx.AsyncClient,
) -> dict[str, Any]:
    """Run reverse image search (Yandex CBIR) + text web search (Exa + DDG + Wiki), scrape profiles, and score avatar similarity."""
    import asyncio
    import re
    from urllib.parse import quote, urlparse
    from osint_agent.collectors.extractor import extract_entities
    from osint_agent.collectors.profile_analyzer import analyze_profiles_batch
    from osint_agent.collectors.tools import collect_exa_search
    from osint_agent.core.schema import Entity

    hits: list[Hit] = []
    discovered_seeds: list[Entity] = []
    seen_seed_ids: set[str] = set()
    seen_urls: set[str] = set()

    def _add_seed(ent: Entity) -> None:
        if ent.id not in seen_seed_ids:
            seen_seed_ids.add(ent.id)
            discovered_seeds.append(ent)

    # 0. Real Reverse Image Search via Yandex CBIR (only high-similarity matches >= 0.72)
    cbir_hits = await reverse_image_search_yandex(image_bytes, filename, target_id, client, max_results=10)
    for ch in cbir_hits:
        if ch.discovered_value not in seen_urls:
            seen_urls.add(ch.discovered_value)
            hits.append(ch)
            # Only promote social handles from CBIR if visual similarity is high (>= 0.78)
            if float(ch.metadata.get("visual_similarity") or 0.0) >= 0.78:
                gh_m = re.match(r"^https?://(?:www\.)?github\.com/([a-zA-Z0-9_-]+)/?$", ch.discovered_value, re.I)
                if gh_m:
                    _add_seed(Entity.create(EntityType.USERNAME, gh_m.group(1), confidence=0.85, provenance=["yandex_cbir"]))

    # 1. Gather entity clues from EXIF, PNG text chunks, filename, and Vision LLM
    clue_texts: list[str] = []
    if meta.get("artist"):
        _add_seed(Entity.create(EntityType.USERNAME, str(meta["artist"]).strip(), confidence=0.9, provenance=["image_exif"]))
        clue_texts.append(str(meta["artist"]))
    for tc in meta.get("text_clues") or []:
        clue_texts.append(str(tc))

    for nm in vision_info.get("detected_names") or []:
        if isinstance(nm, str) and len(nm.strip()) >= 2:
            _add_seed(Entity.create(EntityType.PUBLIC_FIGURE, nm.strip(), confidence=0.8, provenance=["image_vision"]))
            clue_texts.append(nm.strip())
    for un in vision_info.get("detected_usernames") or []:
        if isinstance(un, str) and len(un.strip()) >= 2:
            _add_seed(Entity.create(EntityType.USERNAME, un.strip().lstrip("@"), confidence=0.85, provenance=["image_vision"]))
            clue_texts.append(un.strip())
    for em in vision_info.get("detected_emails") or []:
        if isinstance(em, str) and "@" in em:
            _add_seed(Entity.create(EntityType.EMAIL, em.strip().lower(), confidence=0.9, provenance=["image_vision"]))
    for dom in vision_info.get("detected_domains") or []:
        if isinstance(dom, str) and "." in dom:
            _add_seed(Entity.create(EntityType.DOMAIN, dom.strip().lower(), confidence=0.8, provenance=["image_vision"]))
    for org in vision_info.get("detected_organizations") or []:
        if isinstance(org, str) and len(org.strip()) >= 2:
            _add_seed(Entity.create(EntityType.ORG, org.strip(), confidence=0.8, provenance=["image_vision"]))

    if vision_info.get("visible_text"):
        clue_texts.append(str(vision_info["visible_text"]))
    for e in extract_entities(" ".join(clue_texts)):
        _add_seed(e)

    # Build deduplicated search queries
    queries: list[str] = []
    for q in vision_info.get("search_queries") or []:
        if isinstance(q, str) and len(q.strip()) >= 3 and q.strip() not in queries:
            queries.append(q.strip())
    if meta.get("artist") and str(meta["artist"]).strip() not in queries:
        queries.append(str(meta["artist"]).strip())
    fn_query = _descriptive_filename_query(filename)
    if fn_query and fn_query not in queries:
        queries.append(fn_query)
    for seed in discovered_seeds[:2]:
        if seed.value not in queries:
            queries.append(seed.value)

    # 2. Exa AI Web Search (honors 402/429 circuit breakers)
    for q in queries[:2]:
        q_entity = Entity.create(EntityType.PUBLIC_FIGURE, q, confidence=0.8, provenance=["image_web_search"])
        exa_hits = await collect_exa_search(q_entity, client)
        for eh in exa_hits:
            if eh.discovered_value not in seen_urls:
                seen_urls.add(eh.discovered_value)
                eh.target_entity_id = target_id
                eh.relation = "image_web_match"
                hits.append(eh)

    # 3. Free keyless DuckDuckGo Instant Answer + Wikipedia search for top queries
    for q in queries[:2]:
        try:
            ddg_resp = await client.get(
                f"https://api.duckduckgo.com/?q={quote(q)}&format=json&no_html=1&skip_disambig=1",
                headers={"User-Agent": DEFAULT_USER_AGENT},
                timeout=6.0,
            )
            if ddg_resp.status_code == 200:
                ddg = ddg_resp.json()
                abstract_url = str(ddg.get("AbstractURL") or "").strip()
                abstract_text = str(ddg.get("AbstractText") or "").strip()
                heading = str(ddg.get("Heading") or q).strip()
                if abstract_url.startswith("http") and abstract_url not in seen_urls:
                    seen_urls.add(abstract_url)
                    hits.append(
                        Hit(
                            target_entity_id=target_id,
                            discovered_type=EntityType.PROFILE,
                            discovered_value=abstract_url,
                            relation="image_web_match",
                            source_tool="image_web_search",
                            is_trusted=False,
                            raw_data={"site": urlparse(abstract_url).netloc.lower(), "title": heading, "description": abstract_text[:300]},
                            metadata={"site": urlparse(abstract_url).netloc.lower(), "url": abstract_url, "name": heading, "bio": abstract_text[:400]},
                        )
                    )
        except Exception:
            pass

        try:
            wiki_resp = await client.get(
                f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote(q)}&srlimit=2&format=json",
                headers={"User-Agent": DEFAULT_USER_AGENT},
                timeout=6.0,
            )
            if wiki_resp.status_code == 200:
                for item in (wiki_resp.json().get("query") or {}).get("search") or []:
                    title = str(item.get("title") or "").strip()
                    if not title:
                        continue
                    wiki_url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                    if wiki_url not in seen_urls:
                        seen_urls.add(wiki_url)
                        hits.append(
                            Hit(
                                target_entity_id=target_id,
                                discovered_type=EntityType.PROFILE,
                                discovered_value=wiki_url,
                                relation="image_web_match",
                                source_tool="image_web_search",
                                is_trusted=False,
                                raw_data={"site": "en.wikipedia.org", "title": title},
                                metadata={"site": "en.wikipedia.org", "url": wiki_url, "name": title},
                            )
                        )
        except Exception:
            pass

    # 4. Scrape discovered web profiles & compare their avatars visually against uploaded image
    if hits:
        await analyze_profiles_batch(hits, client, max_profiles=10, concurrency=6)
        avatar_checks: list[tuple[Hit, str]] = []
        for h in hits:
            av = str(h.metadata.get("avatar_url") or h.metadata.get("thumbnail_url") or "")
            if av.startswith("http"):
                avatar_checks.append((h, av))
        if avatar_checks:
            sims = await asyncio.gather(
                *(compare_reference_to_avatar_url(image_bytes, av, client) for _, av in avatar_checks),
                return_exceptions=True,
            )
            for (h, _av), sim in zip(avatar_checks, sims):
                if isinstance(sim, float) and sim > 0.0:
                    h.metadata["visual_similarity"] = sim
                    if sim >= 0.72:
                        h.metadata["avatar_match"] = True
                        h.is_trusted = True

    return {
        "hits": hits,
        "discovered_seeds": discovered_seeds,
        "queries": queries,
    }
