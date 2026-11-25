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
SCRIPT_STYLE_STRIP = re.compile(r"<(script|style|noscript|svg|nav|header|footer|aside)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
MAIN_CONTENT_PATTERN = re.compile(r"<(?:main|article)[^>]*>(.*?)</(?:main|article)>", re.DOTALL | re.IGNORECASE)
NEXT_DATA_PATTERN = re.compile(r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>', re.DOTALL | re.IGNORECASE)
OG_IMAGE_PATTERN = re.compile(
    r'<meta[^>]+(?:property|name)=["\'](?:og:image|twitter:image)["\'][^>]+content=["\'](https?://[^"\'>\s]+)["\']',
    re.IGNORECASE,
)
AVATAR_IMG_PATTERN = re.compile(
    r'<img[^>]+(?:class|alt|id)=["\'][^"\']*(?:avatar|profile|user)[^"\']*["\'][^>]+src=["\'](https?://[^"\'>\s]+)["\']',
    re.IGNORECASE,
)
MAILTO_PATTERN = re.compile(r'mailto:([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', re.IGNORECASE)
EXT_LINK_PATTERN = re.compile(r'href=["\'](https?://[^"\'>\s]+)["\']', re.IGNORECASE)
_IGNORED_EXT_DOMAINS = (
    "w3.org", "schema.org", "fonts.googleapis.com", "fonts.gstatic.com",
    "cdnjs.cloudflare.com", "googletagmanager.com", "google-analytics.com",
    "apple.com/app", "play.google.com/intl", "policies.google.com",
)


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


async def _analyze_wattpad(username: str, client: httpx.AsyncClient) -> dict[str, Any]:
    """Fetch public Wattpad user profile and published stories via Wattpad v3 API."""
    info: dict[str, Any] = {}
    try:
        u_resp, s_resp = await asyncio.gather(
            client.get(f"https://www.wattpad.com/api/v3/users/{username}", headers=BROWSER_HEADERS, timeout=6.0),
            client.get(f"https://www.wattpad.com/api/v3/users/{username}/stories", headers=BROWSER_HEADERS, timeout=6.0),
            return_exceptions=True,
        )
        if isinstance(u_resp, httpx.Response) and u_resp.status_code == 200:
            d = u_resp.json()
            if d.get("name"):
                info["name"] = d["name"]
            if d.get("description"):
                info["bio"] = d["description"]
            if d.get("location"):
                info["location"] = d["location"]
            if d.get("avatar"):
                info["avatar_url"] = d["avatar"]
            if d.get("numFollowers") is not None:
                info["followers"] = d["numFollowers"]
            if d.get("website"):
                info["website"] = d["website"]

        if isinstance(s_resp, httpx.Response) and s_resp.status_code == 200:
            stories = s_resp.json().get("stories") or []
            if stories:
                story_summaries = []
                for st in stories[:5]:
                    title = str(st.get("title") or "").strip()
                    desc = str(st.get("description") or "").strip()[:180]
                    votes = st.get("voteCount", 0)
                    if title:
                        part = f"{title} ({votes} votes)"
                        if desc:
                            part += f": {desc}"
                        story_summaries.append(part)
                if story_summaries:
                    info["stories"] = story_summaries
                    stories_text = "Stories: " + " | ".join(story_summaries)
                    info["bio"] = (info.get("bio") + " | " + stories_text) if info.get("bio") else stories_text
    except Exception:
        pass
    return info


async def _analyze_playstore(url: str, client: httpx.AsyncClient) -> dict[str, Any]:
    """Extract developer contact email, website, app titles, and office address from Google Play Store pages."""
    info: dict[str, Any] = {}
    try:
        r = await client.get(url, headers=BROWSER_HEADERS, follow_redirects=True, timeout=8.0)
        if r.status_code != 200:
            return info
        html = r.text
        parsed_meta = parse_html_metadata(html[:120_000])
        if isinstance(parsed_meta, dict):
            if parsed_meta.get("title"):
                info["title"] = parsed_meta["title"]
            if parsed_meta.get("description"):
                info["bio"] = parsed_meta["description"]

        og_m = OG_IMAGE_PATTERN.search(html[:60_000])
        if og_m:
            info["avatar_url"] = og_m.group(1)

        # Extract developer contact emails
        emails = list(dict.fromkeys(MAILTO_PATTERN.findall(html)))
        if not emails:
            raw_emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', html)
            emails = [e for e in dict.fromkeys(raw_emails) if not e.endswith((".png", ".jpg", ".webp", "google.com", "gstatic.com"))]
        if emails:
            info["email"] = emails[0]
            info["contact_emails"] = emails[:3]

        # Extract Play Store "App support" / developer address & website from visible text
        cleaned = _strip_html(SCRIPT_STYLE_STRIP.sub(" ", html))
        addr_m = re.search(
            r"(?:Address|Developer\s+address)\s*[:\-]?\s*([A-Za-z0-9 ,.\-/#]{10,160})",
            cleaned,
            re.IGNORECASE,
        )
        if addr_m:
            info["location"] = addr_m.group(1).strip()

        # Extract outbound developer website / privacy policy links
        ext_links = [
            lnk for lnk in re.findall(r'href=["\'](https?://[^"\'>\s]+)["\']', html)
            if not any(d in lnk.lower() for d in ("play.google.com", "google.com", "gstatic.com", "ggpht.com", "youtube.com"))
        ]
        if ext_links:
            info["linked_urls"] = list(dict.fromkeys(ext_links))[:6]
            info["website"] = info["linked_urls"][0]

        if cleaned:
            info["page_text_excerpt"] = cleaned[:2500]
    except Exception:
        pass
    return info


def _extract_hydration_fields(obj: Any, out: dict[str, Any], depth: int = 0) -> None:
    """Recursively harvest profile fields from embedded Next.js / JSON hydration blobs."""
    if depth > 7:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if isinstance(v, str) and v.strip():
                val = v.strip()
                if kl in ("bio", "about", "aboutme", "description", "summary") and not out.get("bio"):
                    out["bio"] = val[:600]
                elif kl in ("name", "fullname", "displayname", "display_name") and not out.get("name"):
                    out["name"] = val[:120]
                elif kl in ("location", "city", "country", "address") and not out.get("location"):
                    out["location"] = val[:160]
                elif kl in ("company", "organization", "org") and not out.get("company"):
                    out["company"] = val[:120]
                elif kl in ("website", "blog", "homepage", "website_url") and val.startswith("http") and not out.get("website"):
                    out["website"] = val
                elif kl in ("email", "contact_email") and "@" in val and not out.get("email"):
                    out["email"] = val
                elif kl in ("avatar", "avatar_url", "avatarurl", "profile_image") and val.startswith("http") and not out.get("avatar_url"):
                    out["avatar_url"] = val
            elif isinstance(v, (dict, list)):
                _extract_hydration_fields(v, out, depth + 1)
    elif isinstance(obj, list):
        for item in obj[:25]:
            _extract_hydration_fields(item, out, depth + 1)


async def _analyze_generic_html(url: str, client: httpx.AsyncClient) -> dict[str, Any]:
    """Extract OpenGraph metadata, JSON-LD, hydration JSON, images, emails, links, and cleaned page text."""
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

    try:
        resp = await client.get(url, headers=BROWSER_HEADERS, follow_redirects=True, timeout=6.0)
        if resp.status_code == 200 and "text/html" in resp.headers.get("content-type", "").lower():
            html = resp.text[:250_000]
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
                    _extract_hydration_fields(ld, info)
                    if isinstance(ld, dict):
                        if ld.get("jobTitle"):
                            info["job_title"] = str(ld["jobTitle"])
                        same_as = ld.get("sameAs")
                        if isinstance(same_as, list):
                            info.setdefault("linked_urls", []).extend(str(u) for u in same_as if str(u).startswith("http"))
                except Exception:
                    pass

            # Next.js / SPA hydration JSON extraction
            next_m = NEXT_DATA_PATTERN.search(html)
            if next_m:
                try:
                    _extract_hydration_fields(json.loads(next_m.group(1)), info)
                except Exception:
                    pass

            # og:image / twitter:image / avatar <img> & mailto contact emails
            og_m = OG_IMAGE_PATTERN.search(html)
            if og_m and not info.get("avatar_url"):
                info["avatar_url"] = og_m.group(1)
            else:
                av_m = AVATAR_IMG_PATTERN.search(html)
                if av_m and not info.get("avatar_url"):
                    info["avatar_url"] = av_m.group(1)

            mailto_emails = list(dict.fromkeys(MAILTO_PATTERN.findall(html)))
            if mailto_emails:
                info["email"] = mailto_emails[0]
                info["contact_emails"] = mailto_emails[:3]

            # Cleaned visible text excerpt (prioritize <main>/<article>, strip nav/header/footer/scripts)
            main_chunks = MAIN_CONTENT_PATTERN.findall(html)
            content_html = " ".join(main_chunks) if main_chunks else html
            visible_text = _strip_html(SCRIPT_STYLE_STRIP.sub(" ", content_html))
            if len(visible_text) > 80:
                info["page_text_excerpt"] = visible_text[:4000]

            # rel="me", social links, and general outbound profile/website links
            rel_links = REL_ME_PATTERN.findall(html)
            social_links = SOCIAL_LINK_PATTERN.findall(html)
            ext_links = [
                lnk for lnk in EXT_LINK_PATTERN.findall(html)
                if not any(ign in lnk.lower() for ign in _IGNORED_EXT_DOMAINS)
                and not lnk.lower().endswith((".css", ".js", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".woff2"))
            ]
            all_links = list(dict.fromkeys(rel_links + social_links + ext_links))
            clean_links = [lnk for lnk in all_links if urlparse(lnk).netloc.lower() != host][:10]
            if clean_links:
                existing = info.get("linked_urls", [])
                info["linked_urls"] = list(dict.fromkeys(existing + clean_links))

            # Opportunistically follow 1 high-value same-host subpage (/about, /stories, /projects, /contact)
            sub_candidates = [
                lnk for lnk in all_links
                if urlparse(lnk).netloc.lower() == host
                and any(seg in lnk.lower() for seg in ("/about", "/stories", "/projects", "/contact"))
                and lnk.rstrip("/") != url.rstrip("/")
            ]
            if sub_candidates:
                try:
                    sub_resp = await client.get(sub_candidates[0], headers=BROWSER_HEADERS, follow_redirects=True, timeout=4.0)
                    if sub_resp.status_code == 200 and "text/html" in sub_resp.headers.get("content-type", "").lower():
                        sub_text = _strip_html(SCRIPT_STYLE_STRIP.sub(" ", sub_resp.text[:120_000]))
                        if len(sub_text) > 80:
                            info["page_text_excerpt"] = (info.get("page_text_excerpt", "") + "\n" + sub_text[:1500]).strip()
                except Exception:
                    pass
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
    elif "wattpad.com" in host and username:
        info = await _analyze_wattpad(username, client)
    elif "play.google.com" in host:
        info = await _analyze_playstore(url, client)
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
    if info.get("stories"):
        hit.metadata["stories"] = info["stories"]
        hit.raw_data["stories"] = info["stories"]
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
    if info.get("page_text_excerpt"):
        hit.raw_data["_page_text_excerpt"] = info["page_text_excerpt"]

    combined_desc = " | ".join(p for p in text_parts if p)
    if combined_desc:
        hit.raw_data["description"] = combined_desc

    return info


def _resolve_optional_llm_client(llm_client: Any = None) -> Any:
    """Return provided LLM client or auto-load configured OpenRouter/Anthropic/Gemini client if key exists."""
    if llm_client is not None:
        return llm_client
    import os
    if os.getenv("PYTEST_CURRENT_TEST"):
        return None
    from osint_agent.llm.client import _load_dotenv, get_llm_client

    _load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "").lower()
    if provider in ("openrouter", "openai", "hackclub") and os.getenv("OPENROUTER_API_KEY"):
        return get_llm_client(provider="openrouter", call_budget=15)
    if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        return get_llm_client(provider="anthropic", call_budget=15)
    if provider == "gemini" and os.getenv("GEMINI_API_KEY"):
        return get_llm_client(provider="gemini", call_budget=15)
    if os.getenv("OPENROUTER_API_KEY"):
        return get_llm_client(provider="openrouter", call_budget=15)
    return None


async def _llm_enrich_batch(hits: list[Hit], llm_client: Any, batch_size: int = 6) -> None:
    """Use LLM (text + optional avatar/image URLs) in batches to extract structured OSINT details across ALL scraped profiles."""
    rich_hits = [
        h for h in hits
        if h.raw_data.get("_page_text_excerpt") or h.metadata.get("bio") or h.metadata.get("stories")
    ]
    if not rich_hits:
        return

    system = (
        "You are an OSINT profile extraction assistant. Given a JSON list of scraped public profile pages "
        "(and optional profile images), extract ONLY facts explicitly present for EACH profile URL. "
        "Return valid JSON: {\"profiles\": {\"<url>\": {\"real_name\": str, \"office_or_location\": str, "
        "\"contact_emails\": [str], \"projects_or_stories\": [str], \"organizations\": [str], "
        "\"linked_urls\": [str], \"summary\": str}}}."
    )

    for idx in range(0, len(rich_hits), batch_size):
        chunk = rich_hits[idx : idx + batch_size]
        payload_items = []
        image_urls: list[str] = []
        url_to_hit: dict[str, Hit] = {}

        for h in chunk:
            url = str(h.metadata.get("url") or h.discovered_value)
            url_to_hit[url] = h
            excerpt = str(h.raw_data.pop("_page_text_excerpt", ""))[:1800]
            avatar_url = str(h.metadata.get("avatar_url") or "")
            if avatar_url.startswith("http") and avatar_url not in image_urls:
                image_urls.append(avatar_url)
            payload_items.append(
                {
                    "url": url,
                    "site": h.metadata.get("site", ""),
                    "bio": str(h.metadata.get("bio", ""))[:400],
                    "stories": h.metadata.get("stories", [])[:3] if isinstance(h.metadata.get("stories"), list) else [],
                    "page_text": excerpt,
                }
            )

        user = json.dumps(payload_items, ensure_ascii=False)
        try:
            if hasattr(llm_client, "complete_multimodal_json") and image_urls:
                res = await asyncio.to_thread(
                    llm_client.complete_multimodal_json, system, user, image_urls[:3], 1200
                )
            else:
                res = await asyncio.to_thread(llm_client.complete_json, system, user, 1200)

            profiles_map = res.get("profiles") if isinstance(res, dict) else None
            if not isinstance(profiles_map, dict):
                continue

            for url, extracted in profiles_map.items():
                h = url_to_hit.get(str(url))
                if not h or not isinstance(extracted, dict):
                    continue
                if extracted.get("real_name") and not h.metadata.get("name"):
                    h.metadata["name"] = str(extracted["real_name"])
                if extracted.get("office_or_location") and not h.metadata.get("location"):
                    h.metadata["location"] = str(extracted["office_or_location"])
                if isinstance(extracted.get("contact_emails"), list):
                    emails = [str(e) for e in extracted["contact_emails"] if "@" in str(e)]
                    if emails and not h.metadata.get("email"):
                        h.metadata["email"] = emails[0]
                if isinstance(extracted.get("projects_or_stories"), list) and extracted["projects_or_stories"]:
                    h.metadata["projects_or_stories"] = [str(p) for p in extracted["projects_or_stories"][:6]]
                if isinstance(extracted.get("organizations"), list) and extracted["organizations"]:
                    h.metadata["organizations"] = [str(o) for o in extracted["organizations"][:5]]
                if isinstance(extracted.get("linked_urls"), list) and extracted["linked_urls"]:
                    existing_links = h.metadata.get("linked_urls") or []
                    new_links = [str(u) for u in extracted["linked_urls"] if str(u).startswith("http")]
                    h.metadata["linked_urls"] = list(dict.fromkeys(list(existing_links) + new_links))[:12]
                if extracted.get("summary"):
                    h.metadata["llm_summary"] = str(extracted["summary"])
                    existing_desc = h.raw_data.get("description", "")
                    h.raw_data["description"] = (
                        f"{existing_desc} | {extracted['summary']}" if existing_desc else str(extracted["summary"])
                    )
        except Exception as exc:
            logger.debug("LLM batch profile enrichment skipped for batch %d: %s", idx, exc)


async def analyze_profiles_batch(
    hits: list[Hit],
    client: httpx.AsyncClient,
    max_profiles: int = 40,
    concurrency: int = 12,
    llm_client: Any = None,
) -> None:
    """Concurrently analyze ALL discovered PROFILE hits up to max_profiles, with batched LLM extraction."""
    profile_hits = [h for h in hits if h.discovered_type == EntityType.PROFILE][:max_profiles]
    if not profile_hits:
        return

    sem = asyncio.Semaphore(concurrency)

    async def _run_one(h: Hit) -> None:
        async with sem:
            await analyze_profile_hit(h, client)

    await asyncio.gather(*[_run_one(h) for h in profile_hits], return_exceptions=True)

    active_llm = _resolve_optional_llm_client(llm_client)
    if active_llm is not None:
        await _llm_enrich_batch(profile_hits, active_llm, batch_size=6)
    for h in profile_hits:
        h.raw_data.pop("_page_text_excerpt", None)


async def synthesize_profile_relations(graph: Any, llm_client: Any = None) -> dict[str, Any] | None:
    """Use LLM to synthesize cross-profile relations, shared themes, and identity links across the graph."""
    active_llm = _resolve_optional_llm_client(llm_client)
    if active_llm is None:
        return None

    profile_summaries: list[dict[str, Any]] = []
    url_to_node_id: dict[str, str] = {}

    for node_id, data in graph.nodes(data=True):
        if data.get("entity_type") != "profile":
            continue
        attrs = data.get("attributes") or {}
        url = str(data.get("value") or "")
        url_to_node_id[url] = node_id
        # Include only profiles with meaningful scraped info
        if any(attrs.get(k) for k in ("name", "bio", "location", "company", "stories", "email", "linked_urls", "projects_or_stories", "llm_summary")):
            profile_summaries.append(
                {
                    "url": url,
                    "site": attrs.get("site", ""),
                    "name": attrs.get("name", ""),
                    "bio": str(attrs.get("bio", ""))[:300],
                    "location": attrs.get("location", ""),
                    "company": attrs.get("company", ""),
                    "email": attrs.get("email", ""),
                    "stories": attrs.get("stories", [])[:3] if isinstance(attrs.get("stories"), list) else [],
                    "linked_urls": attrs.get("linked_urls", [])[:5] if isinstance(attrs.get("linked_urls"), list) else [],
                }
            )

    if len(profile_summaries) < 2:
        return None

    system = (
        "You are an OSINT intelligence analyst synthesizing connections across discovered public profiles. "
        "Given JSON profiles, identify cross-profile relations, shared identities, projects, stories, "
        "locations/offices, and contact details. Return ONLY valid JSON with keys: "
        "summary (str), inferred_name (str), locations (list[str]), organizations (list[str]), "
        "relations (list of {source_url, target_url, relation, evidence})."
    )
    user = json.dumps(profile_summaries[:25], ensure_ascii=False)

    try:
        result = await asyncio.to_thread(active_llm.complete_json, system, user, 900)
        if not isinstance(result, dict):
            return None

        graph.graph["llm_synthesis"] = result
        for rel in result.get("relations") or []:
            if not isinstance(rel, dict):
                continue
            src = url_to_node_id.get(str(rel.get("source_url", "")))
            dst = url_to_node_id.get(str(rel.get("target_url", "")))
            if src and dst and src != dst and not graph.has_edge(src, dst):
                graph.add_edge(
                    src,
                    dst,
                    relation=str(rel.get("relation") or "related_to"),
                    source="llm_synthesis",
                    source_tool="llm_synthesis",
                    evidence=str(rel.get("evidence") or ""),
                    score=0.85,
                )
        return result
    except Exception as exc:
        logger.debug("LLM cross-profile synthesis skipped: %s", exc)
        return None
