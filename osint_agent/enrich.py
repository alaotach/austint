"""Profile enrichment via asynchronous HTML metadata extraction with rate limiting and SSRF protection."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
import urllib.parse
import urllib.robotparser
from typing import Any

import httpx

from osint_agent.schema import Hit
from osint_agent.tools import DEFAULT_USER_AGENT

logger = logging.getLogger(__name__)

# Global concurrency and per-domain rate limiting state
_concurrency_semaphore = asyncio.Semaphore(5)
_domain_locks: dict[str, asyncio.Lock] = {}
_domain_last_request: dict[str, float] = {}
_robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}
_global_lock = asyncio.Lock()

MAX_RESPONSE_BYTES = 200 * 1024  # 200 KB size cap
DEFAULT_DOMAIN_DELAY = 1.0


def is_private_or_loopback_host(hostname: str) -> bool:
    """Verify whether hostname resolves to private, loopback, or reserved IP (SSRF guard)."""
    clean_host = hostname.strip().lower()
    if clean_host in ("localhost", "loopback", "127.0.0.1", "::1"):
        return True

    # Check if host is direct IP literal
    try:
        ip = ipaddress.ip_address(clean_host)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
        )
    except ValueError:
        pass

    # Resolve hostname to check target IP
    try:
        addr_info = socket.getaddrinfo(clean_host, None)
        for _, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
            ):
                return True
    except (socket.gaierror, socket.herror):
        # Unresolvable in public/local DNS; cannot reach internal hosts via standard IP routing
        return False

    return False


async def can_fetch_robots(client: httpx.AsyncClient, parsed_url: urllib.parse.ParseResult) -> bool:
    """Check domain robots.txt cache or retrieve and parse policies."""
    domain_key = f"{parsed_url.scheme}://{parsed_url.netloc}"
    async with _global_lock:
        if domain_key in _robots_cache:
            rp = _robots_cache[domain_key]
            return rp.can_fetch(DEFAULT_USER_AGENT, parsed_url.geturl())

    # Fetch robots.txt
    robots_url = f"{domain_key}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    try:
        resp = await client.get(
            robots_url,
            headers={"User-Agent": DEFAULT_USER_AGENT},
            timeout=5.0,
            follow_redirects=True,
        )
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
        else:
            # If 404 or unavailable, default to allowing fetch
            rp.allow_all = True
    except Exception:
        rp.allow_all = True

    async with _global_lock:
        _robots_cache[domain_key] = rp

    return rp.can_fetch(DEFAULT_USER_AGENT, parsed_url.geturl())


def parse_html_metadata(html: str) -> dict[str, str]:
    """Extract og:title, og:description, meta description, and title from HTML string."""
    meta: dict[str, str] = {}

    # Extract <title>...</title>
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if title_match:
        meta["title"] = title_match.group(1).strip()

    # Extract <meta property="og:title" content="..."> or name="og:title"
    og_title_match = re.search(
        r'<meta[^>]+(?:property|name)=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
        html,
        re.IGNORECASE,
    ) or re.search(
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']og:title["\']',
        html,
        re.IGNORECASE,
    )
    if og_title_match:
        meta["og:title"] = og_title_match.group(1).strip()

    # Extract og:description
    og_desc_match = re.search(
        r'<meta[^>]+(?:property|name)=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']',
        html,
        re.IGNORECASE,
    ) or re.search(
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']og:description["\']',
        html,
        re.IGNORECASE,
    )
    if og_desc_match:
        meta["og:description"] = og_desc_match.group(1).strip()

    # Extract meta description
    desc_match = re.search(
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
        html,
        re.IGNORECASE,
    ) or re.search(
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']description["\']',
        html,
        re.IGNORECASE,
    )
    if desc_match:
        meta["description"] = desc_match.group(1).strip()

    return meta


async def fetch_profile_meta(
    client: httpx.AsyncClient,
    url: str,
    domain_delay: float = DEFAULT_DOMAIN_DELAY,
) -> str:
    """Safely fetch HTML metadata from profile URL subject to SSRF, size, robots, and rate limits."""
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return ""

    hostname = parsed.hostname or ""
    # 1. SSRF Guard
    if is_private_or_loopback_host(hostname):
        logger.warning("SSRF guard blocked fetch to private/loopback address: %s", hostname)
        return ""

    # 2. Robots.txt check
    if not await can_fetch_robots(client, parsed):
        logger.info("robots.txt disallowed crawling %s", url)
        return ""

    # 3. Domain rate limiting
    domain_key = hostname.lower()
    async with _global_lock:
        if domain_key not in _domain_locks:
            _domain_locks[domain_key] = asyncio.Lock()
        domain_lock = _domain_locks[domain_key]

    async with _concurrency_semaphore:
        async with domain_lock:
            loop = asyncio.get_event_loop()
            now = loop.time()
            last_time = _domain_last_request.get(domain_key, 0.0)
            wait_time = domain_delay - (now - last_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            _domain_last_request[domain_key] = loop.time()

            try:
                # Use streaming request to enforce 200 KB cap without loading unbounded responses
                headers = {
                    "User-Agent": DEFAULT_USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml",
                }
                async with client.stream(
                    "GET",
                    url,
                    headers=headers,
                    timeout=10.0,
                    follow_redirects=True,
                ) as response:
                    if response.status_code != 200:
                        return ""

                    content_chunks: list[bytes] = []
                    total_bytes = 0
                    async for chunk in response.aiter_bytes():
                        content_chunks.append(chunk)
                        total_bytes += len(chunk)
                        if total_bytes >= MAX_RESPONSE_BYTES:
                            break

                    raw_html = b"".join(content_chunks).decode("utf-8", errors="replace")
            except (httpx.HTTPError, httpx.TimeoutException, OSError) as err:
                logger.debug("Failed fetching profile meta for %s: %s", url, err)
                return ""

    meta = parse_html_metadata(raw_html)
    summary_parts = []
    if meta.get("og:title"):
        summary_parts.append(meta["og:title"])
    elif meta.get("title"):
        summary_parts.append(meta["title"])

    if meta.get("og:description"):
        summary_parts.append(meta["og:description"])
    elif meta.get("description"):
        summary_parts.append(meta["description"])

    return " - ".join(summary_parts).strip()


async def enrich_profile_hits(
    hits: list[Hit],
    client: httpx.AsyncClient,
    max_enrich: int = 15,
) -> list[Hit]:
    """Enrich at most max_enrich profile hits by extracting bio text and metadata."""
    targets = hits[:max_enrich]
    for hit in targets:
        url = hit.metadata.get("url") or (
            hit.discovered_value if hit.discovered_value.startswith("http") else ""
        )
        if url:
            meta_text = await fetch_profile_meta(client, url)
            if meta_text:
                hit.raw_data["description"] = meta_text
                hit.metadata["description"] = meta_text
                hit.metadata["bio"] = meta_text
    return hits
