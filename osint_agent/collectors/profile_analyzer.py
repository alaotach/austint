"""Deep profile analyzer combining platform public APIs, JSON-LD, and HTML link extraction."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

import httpx

from osint_agent.collectors.enrich import (
    fetch_profile_meta,
    is_private_or_loopback_host,
    parse_html_metadata,
)
from osint_agent.core.schema import EntityType, Hit

logger = logging.getLogger(__name__)

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

SOCIAL_LINK_PATTERN = re.compile(
    r'href=["\'](https?://(?:www\.)?(?:github\.com|gitlab\.com|linkedin\.com/in|x\.com|twitter\.com|t\.me|youtube\.com|instagram\.com|bsky\.app|mastodon\.social|dev\.to|medium\.com|huggingface\.co)/[^"\'>\s]+)["\']',
    re.IGNORECASE,
)
REL_ME_PATTERN = re.compile(
    r'<a[^>]+rel=["\'][^"\']*me[^"\']*["\'][^>]+href=["\'](https?://[^"\'>\s]+)["\']',
    re.IGNORECASE,
)
JSON_LD_PATTERN = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.DOTALL | re.IGNORECASE,
)
HTML_TAG_STRIP = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", HTML_TAG_STRIP.sub(" ", text or "")).strip()


def _extract_username_from_url(url: str) -> str:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        return ""
    last = parts[-1].lstrip("@")
    if last in ("about", "overview", "u", "user", "in", "profile") and len(parts) >= 2:
        last = parts[-2].lstrip("@")
    return last


async def _analyze_github(username: str, client: httpx.AsyncClient) -> dict[str, Any]:
    """Fetch public profile, social accounts, and top repo languages from GitHub API."""
    info: dict[str, Any] = {}
    try:
        r = await client.get(f"https://api.github.com/users/{username}", timeout=5.0)
        if r.status_code == 200:
            d = r.json()
            if d.get("name"):
                info["name"] = d["name"]
            if d.get("bio"):
                info["bio"] = d["bio"]
            if d.get("location"):
                info["location"] = d["location"]
            if d.get("company"):
                info["company"] = d["company"]
            if d.get("blog"):
                info["website"] = d["blog"]
            if d.get("email"):
                info["email"] = d["email"]
            if d.get("twitter_username"):
                info["twitter"] = f"@{d['twitter_username']}"
            if d.get("followers") is not None:
                info["followers"] = d["followers"]
            if d.get("public_repos") is not None:
                info["public_repos"] = d["public_repos"]
            if d.get("avatar_url"):
                info["avatar_url"] = d["avatar_url"]

        rs = await client.get(f"https://api.github.com/users/{username}/social_accounts", timeout=4.0)
        if rs.status_code == 200 and isinstance(rs.json(), list):
            links = [item.get("url") for item in rs.json() if item.get("url")]
            if links:
                info["linked_urls"] = links
    except Exception:
        pass
    return info


async def _analyze_reddit(username: str, client: httpx.AsyncClient) -> dict[str, Any]:
    """Fetch public Reddit profile info via about.json."""
    info: dict[str, Any] = {}
    try:
        r = await client.get(
            f"https://www.reddit.com/user/{username}/about.json",
            headers=BROWSER_HEADERS,
            timeout=5.0,
        )
        if r.status_code == 200:
            data = r.json().get("data", {})
            sub = data.get("subreddit", {})
            if sub.get("title"):
                info["name"] = sub["title"]
            if sub.get("public_description"):
                info["bio"] = sub["public_description"]
            if data.get("total_karma") is not None:
                info["karma"] = data["total_karma"]
    except Exception:
        pass
    return info


async def _analyze_bluesky(handle: str, client: httpx.AsyncClient) -> dict[str, Any]:
    """Fetch public Bluesky profile via official ATProto public API."""
    info: dict[str, Any] = {}
    actor = handle if "." in handle else f"{handle}.bsky.social"
    try:
        r = await client.get(
            "https://public.api.bsky.app/xrpc/app.bsky.actor.getProfile",
            params={"actor": actor},
            timeout=5.0,
        )
        if r.status_code == 200:
            d = r.json()
            if d.get("displayName"):
                info["name"] = d["displayName"]
            if d.get("description"):
                info["bio"] = d["description"]
            if d.get("followersCount") is not None:
                info["followers"] = d["followersCount"]
            if d.get("postsCount") is not None:
                info["posts"] = d["postsCount"]
            if d.get("avatar"):
                info["avatar_url"] = d["avatar"]
    except Exception:
        pass
    return info


async def _analyze_mastodon(url: str, username: str, client: httpx.AsyncClient) -> dict[str, Any]:
    """Fetch public Mastodon profile via instance API v1 lookup."""
    info: dict[str, Any] = {}
    host = urlparse(url).netloc
    if not host or is_private_or_loopback_host(host):
        return info
    try:
        r = await client.get(
            f"https://{host}/api/v1/accounts/lookup",
            params={"acct": username},
            timeout=5.0,
        )
        if r.status_code == 200:
            d = r.json()
            if d.get("display_name"):
                info["name"] = d["display_name"]
            if d.get("note"):
                info["bio"] = _strip_html(d["note"])
            if d.get("followers_count") is not None:
                info["followers"] = d["followers_count"]
            links = []
            for fld in d.get("fields", []):
                val = _strip_html(fld.get("value", ""))
                if val.startswith("http"):
                    links.append(val)
            if links:
                info["linked_urls"] = links
    except Exception:
        pass
    return info


async def _analyze_devto(username: str, client: httpx.AsyncClient) -> dict[str, Any]:
    """Fetch public DEV Community profile via dev.to JSON API."""
    info: dict[str, Any] = {}
    try:
        r = await client.get(
            "https://dev.to/api/users/by_username",
            params={"url": username},
            timeout=5.0,
        )
        if r.status_code == 200:
            d = r.json()
            if d.get("name"):
                info["name"] = d["name"]
            if d.get("summary"):
                info["bio"] = d["summary"]
            if d.get("location"):
                info["location"] = d["location"]
            if d.get("website_url"):
                info["website"] = d["website_url"]
            if d.get("joined_at"):
                info["joined"] = d["joined_at"]
            links = []
            if d.get("github_username"):
                links.append(f"https://github.com/{d['github_username']}")
            if d.get("twitter_username"):
                links.append(f"https://x.com/{d['twitter_username']}")
            if links:
                info["linked_urls"] = links
    except Exception:
        pass
    return info


async def _analyze_huggingface(username: str, client: httpx.AsyncClient) -> dict[str, Any]:
    """Fetch public HuggingFace user overview."""
    info: dict[str, Any] = {}
    try:
        r = await client.get(f"https://huggingface.co/api/users/{username}/overview", timeout=5.0)
        if r.status_code == 200:
            d = r.json()
            if d.get("fullname"):
                info["name"] = d["fullname"]
            if d.get("details"):
                info["bio"] = d["details"]
            if d.get("numFollowers") is not None:
                info["followers"] = d["numFollowers"]
    except Exception:
        pass
    return info


async def _analyze_telegram(url: str, client: httpx.AsyncClient) -> dict[str, Any]:
    """Extract public Telegram preview card title, bio, and subscriber count."""
    info: dict[str, Any] = {}
    try:
        r = await client.get(url, headers=BROWSER_HEADERS, follow_redirects=True, timeout=5.0)
        if r.status_code == 200 and "text/html" in r.headers.get("content-type", "").lower():
            html = r.text[:100_000]
            title_m = re.search(r'<div class="tgme_page_title"[^>]*>(.*?)</div>', html, re.DOTALL)
            desc_m = re.search(r'<div class="tgme_page_description"[^>]*>(.*?)</div>', html, re.DOTALL)
            extra_m = re.search(r'<div class="tgme_page_extra"[^>]*>(.*?)</div>', html, re.DOTALL)
            if title_m:
                info["name"] = _strip_html(title_m.group(1))
            if desc_m:
                info["bio"] = _strip_html(desc_m.group(1))
            if extra_m:
                info["stats"] = _strip_html(extra_m.group(1))
    except Exception:
        pass
    return info


async def _analyze_generic_html(url: str, client: httpx.AsyncClient) -> dict[str, Any]:
    """Extract OpenGraph metadata, JSON-LD person fields, and rel=me/social links."""
    info: dict[str, Any] = {}
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if not host or is_private_or_loopback_host(host):
        return info

    # First run standard polite metadata fetcher (honors robots.txt & 200KB cap)
    meta = await fetch_profile_meta(client, url)
    if isinstance(meta, dict):
        if meta.get("title"):
            info["title"] = meta["title"]
        if meta.get("description"):
            info["bio"] = meta["description"]
    elif isinstance(meta, str) and meta:
        info["bio"] = meta

    # Also attempt lightweight HTML link & JSON-LD extraction if not already blocked
    try:
        resp = await client.get(url, headers=BROWSER_HEADERS, follow_redirects=True, timeout=5.0)
        if resp.status_code == 200 and "text/html" in resp.headers.get("content-type", "").lower():
            html = resp.text[:150_000]
            if not info.get("bio"):
                parsed_meta = parse_html_metadata(html)
                if isinstance(parsed_meta, dict):
                    if parsed_meta.get("title"):
                        info["title"] = parsed_meta["title"]
                    if parsed_meta.get("description"):
                        info["bio"] = parsed_meta["description"]
                elif isinstance(parsed_meta, str) and parsed_meta:
                    info["bio"] = parsed_meta

            # JSON-LD extraction
            for match in JSON_LD_PATTERN.finditer(html):
                try:
                    ld = json.loads(match.group(1))
                    if isinstance(ld, dict):
                        if ld.get("name") and not info.get("name"):
                            info["name"] = str(ld["name"])
                        if ld.get("description") and not info.get("bio"):
                            info["bio"] = str(ld["description"])
                        if ld.get("jobTitle"):
                            info["job_title"] = str(ld["jobTitle"])
                        same_as = ld.get("sameAs")
                        if isinstance(same_as, list):
                            info.setdefault("linked_urls", []).extend(str(u) for u in same_as if str(u).startswith("http"))
                except Exception:
                    pass

            # rel="me" & outbound social links
            rel_links = REL_ME_PATTERN.findall(html)
            social_links = SOCIAL_LINK_PATTERN.findall(html)
            all_links = list(dict.fromkeys(rel_links + social_links))
            # Filter out self-links
            clean_links = [lnk for lnk in all_links if urlparse(lnk).netloc.lower() != host][:8]
            if clean_links:
                existing = info.get("linked_urls", [])
                info["linked_urls"] = list(dict.fromkeys(existing + clean_links))
    except Exception:
        pass

    return info


async def analyze_profile_hit(hit: Hit, client: httpx.AsyncClient) -> dict[str, Any]:
    """Deeply analyze a discovered PROFILE hit using platform APIs or HTML extraction."""
    url = hit.metadata.get("url") or (
        hit.discovered_value if hit.discovered_value.startswith("http") else ""
    )
    if not url:
        return {}

    host = urlparse(url).netloc.lower()
    username = hit.metadata.get("username") or _extract_username_from_url(url)

    info: dict[str, Any] = {}
    if "github.com" in host and "gist.github.com" not in host and username:
        info = await _analyze_github(username, client)
    elif "reddit.com" in host and username:
        info = await _analyze_reddit(username, client)
    elif "bsky.app" in host and username:
        info = await _analyze_bluesky(username, client)
    elif ("mastodon." in host or "fosstodon." in host or "/@" in url) and "medium.com" not in host and "youtube.com" not in host and username:
        info = await _analyze_mastodon(url, username, client)
    elif "dev.to" in host and username:
        info = await _analyze_devto(username, client)
    elif "huggingface.co" in host and username:
        info = await _analyze_huggingface(username, client)
    elif "t.me" in host:
        info = await _analyze_telegram(url, client)

    if not info:
        info = await _analyze_generic_html(url, client)

    if not info:
        return {}

    # Merge extracted fields back into Hit's raw_data & metadata
    text_parts = []
    if info.get("name"):
        hit.metadata["name"] = info["name"]
        hit.raw_data["name"] = info["name"]
        text_parts.append(f"Name: {info['name']}")
    if info.get("bio"):
        hit.metadata["bio"] = info["bio"]
        hit.raw_data["bio"] = info["bio"]
        text_parts.append(info["bio"])
    elif info.get("title"):
        text_parts.append(info["title"])
    if info.get("location"):
        hit.metadata["location"] = info["location"]
        hit.raw_data["location"] = info["location"]
        text_parts.append(f"Location: {info['location']}")
    if info.get("company"):
        hit.metadata["company"] = info["company"]
        text_parts.append(f"Company: {info['company']}")
    if info.get("website"):
        hit.metadata["website"] = info["website"]
        text_parts.append(f"Website: {info['website']}")
    if info.get("email"):
        hit.metadata["email"] = info["email"]
        text_parts.append(f"Email: {info['email']}")
    if info.get("followers") is not None:
        hit.metadata["followers"] = info["followers"]
    if info.get("karma") is not None:
        hit.metadata["karma"] = info["karma"]
    if info.get("linked_urls"):
        hit.metadata["linked_urls"] = info["linked_urls"]
        text_parts.append("Links: " + " ".join(info["linked_urls"]))
    if info.get("avatar_url"):
        hit.metadata["avatar_url"] = info["avatar_url"]

    combined_desc = " | ".join(p for p in text_parts if p)
    if combined_desc:
        hit.raw_data["description"] = combined_desc

    return info


async def analyze_profiles_batch(
    hits: list[Hit],
    client: httpx.AsyncClient,
    max_profiles: int = 40,
    concurrency: int = 12,
) -> None:
    """Concurrently analyze all discovered PROFILE hits up to max_profiles."""
    profile_hits = [h for h in hits if h.discovered_type == EntityType.PROFILE][:max_profiles]
    if not profile_hits:
        return

    sem = asyncio.Semaphore(concurrency)

    async def _run_one(h: Hit) -> None:
        async with sem:
            await analyze_profile_hit(h, client)

    await asyncio.gather(*[_run_one(h) for h in profile_hits], return_exceptions=True)
