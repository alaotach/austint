"""OSINT-Link dataset builder: self-declared link crawling, local CLIP/pHash features, hard negatives, and person-group splits."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import numpy as np
import yaml
from PIL import Image

from osint_agent.collectors.enrich import can_fetch_robots, is_private_or_loopback_host
from osint_agent.core.schema import Entity, EntityType, Hit
from osint_agent.ml.features import TargetContext, extract_features

logger = logging.getLogger(__name__)

_CLIP_SINGLETON: tuple[Any, Any] | None = None
_CLIP_FAILED: bool = False
_REL_ME_RE = re.compile(
    r'<a[^>]+rel=["\'][^"\']*\bme\b[^"\']*["\'][^>]+href=["\'](https?://[^"\'>\s]+)["\']'
    r'|<a[^>]+href=["\'](https?://[^"\'>\s]+)["\'][^>]+rel=["\'][^"\']*\bme\b[^"\']*["\']',
    re.IGNORECASE,
)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_META_DESC_RE = re.compile(
    r'<meta[^>]+(?:name|property)=["\'](?:description|og:description)["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_OG_IMAGE_RE = re.compile(
    r'<meta[^>]+(?:property|name)=["\'](?:og:image|twitter:image)["\'][^>]+content=["\'](https?://[^"\'>\s]+)["\']',
    re.IGNORECASE,
)


def load_config(config_path: str | Path = "configs/osint_link.yaml") -> dict[str, Any]:
    """Load central YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def hash_person_id(raw_id: str, salt: str = "osint-link-public-research-v1") -> str:
    """Compute a deterministic salted SHA-256 person identifier."""
    digest = hashlib.sha256(f"{salt}:{raw_id.strip().lower()}".encode("utf-8")).hexdigest()
    return f"person_{digest[:16]}"


def _jaro_winkler(s1: str, s2: str) -> float:
    a, b = s1.lower().strip(), s2.lower().strip()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    max_dist = max(len(a), len(b)) // 2 - 1
    if max_dist < 0:
        max_dist = 0
    a_m = [False] * len(a)
    b_m = [False] * len(b)
    matches = 0
    for i, ch in enumerate(a):
        lo, hi = max(0, i - max_dist), min(i + max_dist + 1, len(b))
        for j in range(lo, hi):
            if not b_m[j] and ch == b[j]:
                a_m[i] = b_m[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    t = 0
    k = 0
    for i in range(len(a)):
        if a_m[i]:
            while not b_m[k]:
                k += 1
            if a[i] != b[k]:
                t += 1
            k += 1
    jaro = (matches / len(a) + matches / len(b) + (matches - t / 2.0) / matches) / 3.0
    prefix = 0
    for c1, c2 in zip(a[:4], b[:4]):
        if c1 == c2:
            prefix += 1
        else:
            break
    return round(jaro + 0.1 * prefix * (1.0 - jaro), 4)


def compute_phash(image_bytes: bytes) -> str:
    """Compute a 64-bit perceptual hash (pHash) via 2D FFT low-frequency energy; returns 16-char hex."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            arr = np.asarray(img.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
        freq = np.abs(np.fft.fft2(arr))[:8, :8]
        # Exclude DC component [0, 0] when computing median threshold
        flat = freq.flatten()
        med = float(np.median(flat[1:]))
        bits = 0
        for idx, val in enumerate(flat):
            if float(val) > med:
                bits |= 1 << idx
        return f"{bits:016x}"
    except Exception:
        return ""


def phash_similarity(hex_a: str, hex_b: str) -> float:
    """Normalized Hamming similarity [0, 1] between two 64-bit hex pHashes."""
    if not hex_a or not hex_b:
        return 0.0
    try:
        xor = int(hex_a, 16) ^ int(hex_b, 16)
        dist = bin(xor).count("1")
        return round(max(0.0, 1.0 - dist / 64.0), 4)
    except ValueError:
        return 0.0


def _fallback_visual_embedding(image_bytes: bytes, dim: int = 64) -> list[float]:
    """Deterministic lightweight RGB+gradient embedding used when CLIP is disabled or offline."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb = img.convert("RGB").resize((16, 16), Image.Resampling.LANCZOS)
            hist = np.asarray(rgb.histogram(), dtype=np.float32)  # 768 bins
            # Pool into `dim` bins and L2-normalize
            bins = np.array_split(hist, dim)
            vec = np.array([float(b.sum()) for b in bins], dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            return [round(float(x), 5) for x in vec]
    except Exception:
        return [0.0] * dim


def compute_clip_embedding(
    image_bytes: bytes,
    model_name: str = "openai/clip-vit-base-patch32",
    allow_download: bool = True,
) -> list[float]:
    """Compute L2-normalized CLIP image embedding locally (falls back offline/in tests without storing raw image)."""
    global _CLIP_SINGLETON, _CLIP_FAILED
    if not image_bytes:
        return []

    if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("OSINT_LINK_OFFLINE_EMBEDDINGS") == "1" or not allow_download:
        return _fallback_visual_embedding(image_bytes)

    if _CLIP_SINGLETON is None and not _CLIP_FAILED:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = CLIPModel.from_pretrained(model_name).to(device)
            processor = CLIPProcessor.from_pretrained(model_name)
            model.eval()
            _CLIP_SINGLETON = (model, processor, device)
        except Exception as exc:
            logger.warning("Could not load CLIP (%s); using lightweight visual fallback: %s", model_name, exc)
            _CLIP_FAILED = True

    if _CLIP_SINGLETON is None:
        return _fallback_visual_embedding(image_bytes)

    try:
        import torch

        model, processor, device = _CLIP_SINGLETON
        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb = img.convert("RGB")
            inputs = processor(images=rgb, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return [round(float(x), 5) for x in feats[0].cpu().tolist()]
    except Exception:
        return _fallback_visual_embedding(image_bytes)


def compute_bio_embedding(bio: str, dim: int = 64) -> list[float]:
    """Compute a compact normalized text embedding without storing raw bio text in exported datasets."""
    clean = (bio or "").strip()
    if not clean:
        return []
    if not os.getenv("PYTEST_CURRENT_TEST") and os.getenv("OSINT_LINK_OFFLINE_EMBEDDINGS") != "1":
        try:
            from osint_agent.ml.embeddings import get_embedding

            emb = get_embedding(clean)
            if emb is not None:
                return [round(float(x), 5) for x in emb.tolist()]
        except Exception:
            pass

    # Deterministic hashed character/word trigram projection fallback (offline)
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9]+", clean.lower())
    for tok in tokens:
        idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16) % dim
        vec[idx] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return [round(float(x), 5) for x in vec]


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return round(float(np.dot(va, vb) / (na * nb)), 4)


@dataclass
class ProfileRecord:
    """In-memory representation of a public profile during crawling (raw bio/photo discarded on export)."""

    person_id: str  # hashed person_id
    platform: str
    url: str
    username: str
    display_name: str = ""
    bio: str = ""  # used only to compute bio_emb & hand features; never exported raw
    location: str = ""
    company: str = ""
    links: list[str] = field(default_factory=list)
    label_source: str = ""
    phash: str = ""
    clip_emb: list[float] = field(default_factory=list)
    bio_emb: list[float] = field(default_factory=list)

    @property
    def link_domains(self) -> set[str]:
        doms: set[str] = set()
        for u in self.links:
            h = urlparse(u).netloc.lower().lstrip("www.")
            if h:
                doms.add(h)
        return doms


def _infer_platform(url: str) -> str:
    host = urlparse(url).netloc.lower().lstrip("www.")
    if "github.com" in host:
        return "github"
    if "gitlab.com" in host:
        return "gitlab"
    if "dev.to" in host:
        return "devto"
    if "huggingface.co" in host:
        return "huggingface"
    if "linkedin.com" in host:
        return "linkedin"
    if "x.com" in host or "twitter.com" in host:
        return "twitter"
    if "youtube.com" in host:
        return "youtube"
    if "instagram.com" in host:
        return "instagram"
    if "bsky.app" in host:
        return "bluesky"
    if "mastodon" in host or "fosstodon" in host or "hachyderm" in host or "/@" in url:
        return "mastodon"
    if "reddit.com" in host:
        return "reddit"
    return "personal_site"


def _extract_username_from_url(url: str) -> str:
    parts = [p for p in urlparse(url).path.split("/") if p]
    if not parts:
        return urlparse(url).netloc.lower().split(".")[0]
    return parts[-1].lstrip("@")


class _PoliteFetcher:
    """Async HTTP fetcher enforcing SSRF guard, robots.txt, response cap, and per-domain delay."""

    def __init__(self, delay_sec: float = 1.0, max_bytes: int = 204_800, timeout: float = 8.0) -> None:
        self.delay_sec = delay_sec
        self.max_bytes = max_bytes
        self.timeout = timeout
        self._domain_last: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._github_rate_limited: bool = False

    async def get(self, client: httpx.AsyncClient, url: str, check_robots: bool = True) -> httpx.Response | None:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if not host or is_private_or_loopback_host(host):
            return None

        if check_robots and not await can_fetch_robots(client, parsed):
            return None

        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            elapsed = time.monotonic() - self._domain_last.get(host, 0.0)
            if elapsed < self.delay_sec:
                await asyncio.sleep(self.delay_sec - elapsed)
            self._domain_last[host] = time.monotonic()

        try:
            resp = await client.get(url, follow_redirects=True, timeout=self.timeout)
            if resp.status_code in (403, 429) and "api.github.com" in url:
                self._github_rate_limited = True
                return None
            if resp.status_code == 200 and len(resp.content) <= self.max_bytes:
                return resp
        except Exception:
            return None
        return None


async def crawl_self_declared_profiles(
    config: dict[str, Any],
    client: httpx.AsyncClient | None = None,
) -> tuple[list[ProfileRecord], list[tuple[ProfileRecord, ProfileRecord, str]]]:
    """Crawl public GitHub & Mastodon self-declared link graphs and emit verified positive pairs."""
    crawl_cfg = config.get("crawl", {})
    feat_cfg = config.get("features", {})
    salt = os.getenv(config["dataset"].get("salt_env_var", "OSINT_LINK_HASH_SALT"), config["dataset"].get("default_salt", "osint-link-v1"))

    fetcher = _PoliteFetcher(
        delay_sec=float(crawl_cfg.get("per_domain_delay_sec", 1.0)),
        max_bytes=int(crawl_cfg.get("max_response_bytes", 204_800)),
        timeout=float(crawl_cfg.get("request_timeout_sec", 8.0)),
    )
    clip_model = str(feat_cfg.get("clip_model", "openai/clip-vit-base-patch32"))
    max_seed_users = int(crawl_cfg.get("max_seed_users", 1500))
    max_per_person = int(crawl_cfg.get("max_profiles_per_person", 6))

    gh_headers = {"User-Agent": "OSINT-Link-Builder/1.0 (Academic Research)", "Accept": "application/vnd.github+json"}
    gh_tok = os.getenv("GITHUB_TOKEN", "").strip()
    if gh_tok:
        gh_headers["Authorization"] = f"Bearer {gh_tok}"

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(headers=gh_headers, timeout=fetcher.timeout)

    profiles: list[ProfileRecord] = []
    positive_pairs: list[tuple[ProfileRecord, ProfileRecord, str]] = []

    try:
        # 1. Discover public OSS maintainers/members from seed GitHub orgs
        gh_logins: list[str] = []
        seen_logins: set[str] = set()
        for org in crawl_cfg.get("seed_orgs") or []:
            if len(gh_logins) >= max_seed_users:
                break
            r = await fetcher.get(client, f"https://api.github.com/orgs/{org}/public_members?per_page=100", check_robots=False)
            if r is not None:
                try:
                    for item in r.json():
                        lg = str(item.get("login") or "").strip()
                        if lg and lg.lower() not in seen_logins:
                            seen_logins.add(lg.lower())
                            gh_logins.append(lg)
                except Exception:
                    pass

        # 2. Fetch each GitHub user + social_accounts + verify outbound self-declared links
        sem = asyncio.Semaphore(int(crawl_cfg.get("concurrency", 8)))

        async def _process_github_user(login: str) -> None:
            if fetcher._github_rate_limited:
                return
            async with sem:
                if fetcher._github_rate_limited:
                    return
                u_resp = await fetcher.get(client, f"https://api.github.com/users/{login}", check_robots=False)
                if u_resp is None:
                    return
                try:
                    udata = u_resp.json()
                except Exception:
                    return
                if not isinstance(udata, dict) or udata.get("type") != "User":
                    return

                pid = hash_person_id(f"github:{login}", salt=salt)
                gh_url = f"https://github.com/{login}"
                blog = str(udata.get("blog") or "").strip()
                if blog and not blog.startswith("http"):
                    blog = f"https://{blog}"

                soc_links: list[str] = [blog] if blog.startswith("http") else []
                s_resp = await fetcher.get(client, f"https://api.github.com/users/{login}/social_accounts", check_robots=False)
                if s_resp is not None:
                    try:
                        for sc in s_resp.json():
                            su = str(sc.get("url") or "").strip()
                            if su.startswith("http") and su not in soc_links:
                                soc_links.append(su)
                    except Exception:
                        pass

                if not soc_links:
                    return

                # Compute avatar features for GitHub profile
                gh_phash, gh_clip = "", []
                av_url = str(udata.get("avatar_url") or "")
                if av_url.startswith("http"):
                    av_resp = await fetcher.get(client, av_url, check_robots=False)
                    if av_resp is not None:
                        gh_phash = compute_phash(av_resp.content)
                        gh_clip = compute_clip_embedding(av_resp.content, model_name=clip_model)

                gh_bio = str(udata.get("bio") or "")
                gh_rec = ProfileRecord(
                    person_id=pid,
                    platform="github",
                    url=gh_url,
                    username=login,
                    display_name=str(udata.get("name") or login),
                    bio=gh_bio,
                    location=str(udata.get("location") or ""),
                    company=str(udata.get("company") or ""),
                    links=soc_links,
                    label_source="github_social_accounts",
                    phash=gh_phash,
                    clip_emb=gh_clip,
                    bio_emb=compute_bio_embedding(gh_bio),
                )
                person_recs: list[ProfileRecord] = [gh_rec]

                for target_url in soc_links[: max_per_person - 1]:
                    plat = _infer_platform(target_url)
                    t_resp = await fetcher.get(client, target_url, check_robots=True)
                    if t_resp is None:
                        continue
                    html_txt = t_resp.text
                    # Verify self-declared link or bidirectional rel="me" / GitHub backlink
                    rel_urls = [u1 or u2 for u1, u2 in _REL_ME_RE.findall(html_txt)]
                    has_backlink = gh_url.lower() in html_txt.lower() or any(login.lower() in u.lower() for u in rel_urls)
                    label_src = "github_plus_bidirectional_rel_me" if has_backlink else "github_self_declared_social"

                    title_m = _TITLE_RE.search(html_txt)
                    desc_m = _META_DESC_RE.search(html_txt)
                    og_img_m = _OG_IMAGE_RE.search(html_txt)
                    t_bio = desc_m.group(1).strip() if desc_m else ""
                    t_name = title_m.group(1).strip()[:100] if title_m else _extract_username_from_url(target_url)

                    t_phash, t_clip = "", []
                    if og_img_m:
                        img_r = await fetcher.get(client, og_img_m.group(1), check_robots=False)
                        if img_r is not None:
                            t_phash = compute_phash(img_r.content)
                            t_clip = compute_clip_embedding(img_r.content, model_name=clip_model)

                    t_rec = ProfileRecord(
                        person_id=pid,
                        platform=plat,
                        url=target_url,
                        username=_extract_username_from_url(target_url),
                        display_name=t_name,
                        bio=t_bio,
                        links=rel_urls[:8],
                        label_source=label_src,
                        phash=t_phash,
                        clip_emb=t_clip,
                        bio_emb=compute_bio_embedding(t_bio),
                    )
                    person_recs.append(t_rec)

                if len(person_recs) >= 2:
                    profiles.extend(person_recs)
                    for other in person_recs[1:]:
                        positive_pairs.append((gh_rec, other, other.label_source))

        await asyncio.gather(*(_process_github_user(lg) for lg in gh_logins[:max_seed_users]), return_exceptions=True)

        # 3. Mastodon public directory verified rel="me" accounts
        for inst in crawl_cfg.get("mastodon_instances") or []:
            dir_resp = await fetcher.get(client, f"https://{inst}/api/v1/directory?limit=40&order=active", check_robots=False)
            if dir_resp is None:
                continue
            try:
                accts = dir_resp.json()
            except Exception:
                continue
            if not isinstance(accts, list):
                continue
            for acc in accts:
                if not isinstance(acc, dict):
                    continue
                acct_url = str(acc.get("url") or "")
                uname = str(acc.get("username") or "")
                if not acct_url or not uname:
                    continue
                verified_links = [
                    re.search(r'href=["\'](https?://[^"\'>\s]+)["\']', str(f.get("value") or "")).group(1)
                    for f in (acc.get("fields") or [])
                    if isinstance(f, dict)
                    and f.get("verified_at")
                    and re.search(r'href=["\'](https?://[^"\'>\s]+)["\']', str(f.get("value") or ""))
                ]
                if not verified_links:
                    continue
                pid = hash_person_id(f"mastodon:{acct_url}", salt=salt)
                m_bio = re.sub(r"<[^>]+>", " ", str(acc.get("note") or "")).strip()
                m_rec = ProfileRecord(
                    person_id=pid,
                    platform="mastodon",
                    url=acct_url,
                    username=uname,
                    display_name=str(acc.get("display_name") or uname),
                    bio=m_bio,
                    links=verified_links,
                    label_source="mastodon_verified_rel_me",
                    bio_emb=compute_bio_embedding(m_bio),
                )
                added = [m_rec]
                for vurl in verified_links[:3]:
                    v_rec = ProfileRecord(
                        person_id=pid,
                        platform=_infer_platform(vurl),
                        url=vurl,
                        username=_extract_username_from_url(vurl),
                        display_name=m_rec.display_name,
                        links=[acct_url],
                        label_source="mastodon_verified_rel_me",
                    )
                    added.append(v_rec)
                    positive_pairs.append((m_rec, v_rec, "mastodon_verified_rel_me"))
                profiles.extend(added)
    finally:
        if owns_client:
            await client.aclose()

    return profiles, positive_pairs


def _norm_link(url: str) -> str:
    p = urlparse((url or "").strip())
    host = p.netloc.lower().lstrip("www.")
    path = p.path.rstrip("/").lower()
    return f"{host}{path}"


def _mask_pair_formation_links(
    a: ProfileRecord,
    b: ProfileRecord,
) -> tuple[list[str], list[str], set[str], set[str]]:
    """Drop the self-declared link(s) connecting `a` and `b` to prevent label leakage."""
    norm_a = _norm_link(a.url)
    norm_b = _norm_link(b.url)
    host_a = urlparse(a.url).netloc.lower().lstrip("www.")
    host_b = urlparse(b.url).netloc.lower().lstrip("www.")
    site_platforms = {"personal_site", "blog", "website", "other"}

    def _keep(url: str, target_norm: str, target_host: str, target_plat: str) -> bool:
        nu = _norm_link(url)
        if not nu or nu == target_norm:
            return False
        hu = urlparse(url).netloc.lower().lstrip("www.")
        if target_plat in site_platforms and target_host and hu == target_host:
            return False
        return True

    masked_a = [u for u in a.links if _keep(u, norm_b, host_b, b.platform)]
    masked_b = [u for u in b.links if _keep(u, norm_a, host_a, a.platform)]
    doms_a = {urlparse(u).netloc.lower().lstrip("www.") for u in masked_a if urlparse(u).netloc}
    doms_b = {urlparse(u).netloc.lower().lstrip("www.") for u in masked_b if urlparse(u).netloc}
    return masked_a, masked_b, doms_a, doms_b


def _pair_to_exported_dict(
    a: ProfileRecord,
    b: ProfileRecord,
    label: int,
    label_source: str,
    neg_type: str = "",
) -> dict[str, Any]:
    """Serialize a pair to privacy-preserving features + hashed person IDs (zero raw bios or raw photos)."""
    masked_links_a, masked_links_b, doms_a, doms_b = _mask_pair_formation_links(a, b)
    ctx = TargetContext(
        names=[a.display_name] if a.display_name else [],
        usernames=[a.username] if a.username else [],
        texts=[a.bio] if a.bio else [],
        locations=[a.location] if a.location else [],
        domains=list(doms_a),
    )
    hit = Hit(
        target_entity_id=a.person_id,
        discovered_type=EntityType.PROFILE,
        discovered_value=b.url,
        relation="candidate_link",
        source_tool=b.platform,
        is_trusted=False,
        raw_data={"description": b.bio},
        metadata={
            "username": b.username,
            "name": b.display_name,
            "bio": b.bio,
            "location": b.location,
            "company": b.company,
            "linked_urls": masked_links_b,
        },
    )
    hand_feats = extract_features(ctx, hit)
    # Override avatar_sim with pHash/CLIP similarity when available
    p_sim = phash_similarity(a.phash, b.phash)
    c_sim = cosine_sim(a.clip_emb, b.clip_emb)
    if p_sim > 0 or c_sim > 0:
        hand_feats["avatar_sim"] = round(max(p_sim, c_sim), 4)

    union_doms = doms_a | doms_b
    shared_doms = doms_a & doms_b
    links_jaccard = round(len(shared_doms) / len(union_doms), 4) if union_doms else 0.0

    pair_key = hashlib.sha256(f"{a.url}|{b.url}|{label}".encode("utf-8")).hexdigest()[:16]
    return {
        "pair_id": f"pair_{pair_key}",
        "label": label,
        "neg_type": neg_type,
        "label_source": label_source,
        "person_id_a": a.person_id,
        "person_id_b": b.person_id,
        "platform_a": a.platform,
        "platform_b": b.platform,
        "username_a": a.username,
        "username_b": b.username,
        "display_name_a": a.display_name,
        "display_name_b": b.display_name,
        "location_a": a.location,
        "location_b": b.location,
        "bio_len_a": len(a.bio),
        "bio_len_b": len(b.bio),
        "bio_emb_a": a.bio_emb,
        "bio_emb_b": b.bio_emb,
        "bio_emb_sim": cosine_sim(a.bio_emb, b.bio_emb),
        "links_jaccard": links_jaccard,
        "shared_domain_count": len(shared_doms),
        "link_count_a": len(masked_links_a),
        "link_count_b": len(masked_links_b),
        "phash_a": a.phash,
        "phash_b": b.phash,
        "phash_sim": p_sim,
        "clip_emb_a": a.clip_emb,
        "clip_emb_b": b.clip_emb,
        "clip_sim": c_sim,
        "hand_features": hand_feats,
    }


def mine_hard_negatives(
    positives: list[tuple[ProfileRecord, ProfileRecord, str]],
    pool: list[ProfileRecord],
    negatives_per_positive: int = 2,
    seed: int = 42,
) -> list[tuple[ProfileRecord, ProfileRecord, str, str]]:
    """Mine 3 types of hard negatives strictly across DISTINCT person_ids within the given pool."""
    rng = random.Random(seed)
    negatives: list[tuple[ProfileRecord, ProfileRecord, str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    target_total = len(positives) * negatives_per_positive

    if len(pool) < 2 or not positives:
        return negatives

    for anchor, _pos, _src in positives:
        if len(negatives) >= target_total:
            break
        candidates_other = [p for p in pool if p.person_id != anchor.person_id]
        if not candidates_other:
            continue

        # Score candidates for each hard negative strategy
        scored_uname: list[tuple[float, ProfileRecord]] = []
        scored_name_loc: list[tuple[float, ProfileRecord]] = []
        scored_disp_links: list[tuple[float, ProfileRecord]] = []

        for cand in candidates_other[:250]:
            u_sim = _jaro_winkler(anchor.username, cand.username)
            n_sim = _jaro_winkler(anchor.display_name, cand.display_name)
            diff_loc = bool(anchor.location and cand.location and anchor.location.lower() != cand.location.lower())
            diff_org = bool(anchor.company and cand.company and anchor.company.lower() != cand.company.lower())
            no_link_overlap = len(anchor.link_domains & cand.link_domains) == 0

            scored_uname.append((u_sim, cand))
            if diff_loc or diff_org:
                scored_name_loc.append((n_sim + 0.25, cand))
            else:
                scored_name_loc.append((n_sim, cand))
            if no_link_overlap:
                scored_disp_links.append((max(u_sim, n_sim) + 0.15, cand))

        scored_uname.sort(key=lambda x: x[0], reverse=True)
        scored_name_loc.sort(key=lambda x: x[0], reverse=True)
        scored_disp_links.sort(key=lambda x: x[0], reverse=True)

        picks: list[tuple[ProfileRecord, str]] = []
        if scored_uname:
            picks.append((scored_uname[0][1], "similar_username_diff_person"))
        if scored_name_loc:
            picks.append((scored_name_loc[0][1], "same_name_diff_org_city"))
        if scored_disp_links:
            picks.append((scored_disp_links[0][1], "same_display_name_diff_links"))
        # Fallback random different-person negative if duplicates occurred
        rng.shuffle(candidates_other)
        picks.append((candidates_other[0], "diff_person_fallback"))

        added_for_anchor = 0
        for cand, ntype in picks:
            if added_for_anchor >= negatives_per_positive:
                break
            key = tuple(sorted((anchor.url, cand.url)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            negatives.append((anchor, cand, "hard_negative_mining", ntype))
            added_for_anchor += 1

    return negatives


def verify_no_person_leakage(
    splits: dict[str, list[dict[str, Any]]],
    ood_platforms: set[str] | None = None,
) -> None:
    """Assert zero person_id overlap across train/val/test and zero OOD platform leakage into train/val."""
    split_persons: dict[str, set[str]] = {}
    for sname in ("train", "val", "test"):
        pids: set[str] = set()
        for row in splits.get(sname, []):
            pids.add(row["person_id_a"])
            pids.add(row["person_id_b"])
        split_persons[sname] = pids

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_persons[a] & split_persons[b]
        if overlap:
            raise AssertionError(f"Person leakage detected between {a} and {b}: {len(overlap)} shared person_ids")

    if ood_platforms:
        for sname in ("train", "val"):
            for row in splits.get(sname, []):
                if row["platform_a"] in ood_platforms or row["platform_b"] in ood_platforms:
                    raise AssertionError(
                        f"OOD platform leakage in {sname}: {row['platform_a']} / {row['platform_b']}"
                    )


def split_and_export_dataset(
    profiles: list[ProfileRecord],
    positive_pairs: list[tuple[ProfileRecord, ProfileRecord, str]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Group-split by person_id into train/val/test + new-platform OOD split, mine hard negatives, and export JSONL."""
    seed = int(config.get("seed", 42))
    rng = random.Random(seed)
    split_cfg = config.get("splits", {})
    ds_cfg = config.get("dataset", {})

    ood_platforms = set(split_cfg.get("ood_platforms") or [])
    neg_per_pos = int(ds_cfg.get("negatives_per_positive", 2))
    out_dir = Path(ds_cfg.get("output_dir", "data/osint_link"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Separate OOD platform pairs first
    ood_pos: list[tuple[ProfileRecord, ProfileRecord, str]] = []
    id_pos: list[tuple[ProfileRecord, ProfileRecord, str]] = []
    for a, b, src in positive_pairs:
        if a.platform in ood_platforms or b.platform in ood_platforms:
            ood_pos.append((a, b, src))
        else:
            id_pos.append((a, b, src))

    # Group-split person_ids across train / val / test
    id_persons = sorted({p.person_id for a, b, _ in id_pos for p in (a, b)})
    rng.shuffle(id_persons)
    n = len(id_persons)
    n_train = max(1, int(n * float(split_cfg.get("train_ratio", 0.70)))) if n >= 3 else n
    n_val = max(1, int(n * float(split_cfg.get("val_ratio", 0.15)))) if n >= 3 else 0

    person_to_split: dict[str, str] = {}
    for i, pid in enumerate(id_persons):
        if i < n_train:
            person_to_split[pid] = "train"
        elif i < n_train + n_val:
            person_to_split[pid] = "val"
        else:
            person_to_split[pid] = "test"

    # Bucket positive pairs & profile pools strictly by person split (zero cross-split persons)
    pos_by_split: dict[str, list[tuple[ProfileRecord, ProfileRecord, str]]] = {"train": [], "val": [], "test": []}
    pool_by_split: dict[str, list[ProfileRecord]] = {"train": [], "val": [], "test": []}
    for p in profiles:
        if p.platform in ood_platforms:
            continue
        sp = person_to_split.get(p.person_id)
        if sp:
            pool_by_split[sp].append(p)
    for a, b, src in id_pos:
        sp = person_to_split.get(a.person_id)
        if sp and person_to_split.get(b.person_id) == sp:
            pos_by_split[sp].append((a, b, src))

    exported: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": [], "ood_platform_test": []}
    for sp in ("train", "val", "test"):
        negs = mine_hard_negatives(
            pos_by_split[sp],
            pool_by_split[sp],
            negatives_per_positive=neg_per_pos,
            seed=seed,
        )
        for a, b, src in pos_by_split[sp]:
            exported[sp].append(_pair_to_exported_dict(a, b, label=1, label_source=src))
        for a, b, src, ntype in negs:
            exported[sp].append(_pair_to_exported_dict(a, b, label=0, label_source=src, neg_type=ntype))
        rng.shuffle(exported[sp])

    # OOD platform test split (exclude persons already in train/val so OOD is also person-clean if possible)
    ood_pool = [p for p in profiles if p.platform in ood_platforms or p.person_id not in person_to_split or person_to_split[p.person_id] == "test"]
    ood_negs = mine_hard_negatives(ood_pos, ood_pool or profiles, negatives_per_positive=neg_per_pos, seed=seed + 1)
    for a, b, src in ood_pos:
        exported["ood_platform_test"].append(_pair_to_exported_dict(a, b, label=1, label_source=src))
    for a, b, src, ntype in ood_negs:
        exported["ood_platform_test"].append(_pair_to_exported_dict(a, b, label=0, label_source=src, neg_type=ntype))
    rng.shuffle(exported["ood_platform_test"])

    verify_no_person_leakage(exported, ood_platforms=ood_platforms)

    for sname, rows in exported.items():
        fpath = out_dir / f"{sname}.jsonl"
        with open(fpath, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": ds_cfg.get("name", "osint-link"),
        "seed": seed,
        "ood_platforms": sorted(ood_platforms),
        "counts": {
            sname: {
                "total": len(rows),
                "positives": sum(1 for r in rows if r["label"] == 1),
                "negatives": sum(1 for r in rows if r["label"] == 0),
            }
            for sname, rows in exported.items()
        },
        "total_profiles_crawled": len(profiles),
        "store_raw_bios": False,
        "store_raw_photos": False,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


async def build_and_export_osint_link(config_path: str | Path = "configs/osint_link.yaml") -> dict[str, Any]:
    """End-to-end Phase A runner: crawl self-declared links, compute features, mine hard negatives, and export splits."""
    cfg = load_config(config_path)
    profiles, positive_pairs = await crawl_self_declared_profiles(cfg)
    return split_and_export_dataset(profiles, positive_pairs, cfg)
