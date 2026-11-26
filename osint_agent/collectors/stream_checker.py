"""Native high-concurrency streaming username checker using Maigret's local site + engine rules."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import httpx

from osint_agent.core.schema import EntityType, Hit

logger = logging.getLogger(__name__)

_CACHED_SITES: list[tuple[int, str, dict[str, Any]]] | None = None
_DEAD_HOSTS_CACHE: dict[str, float] | None = None
_FALSE_POSITIVE_HOSTS_CACHE: set[str] | None = None
_DEAD_HOSTS_TTL_SECONDS = 7 * 24 * 3600  # 7 days
_DEAD_HOSTS_FILE = Path(".osint_dead_hosts.json")
_FALSE_POSITIVE_SITES_FILE = Path(__file__).resolve().parent / "false_positive_sites.json"
_BASELINE_PROBE_USERNAME = "zzqwx991827364nonexistent"

_BOT_CHALLENGE_MARKERS = (
    "making sure you're not a bot",
    "making sure you&#39;re not a bot",
    "/.within.website/",
    "google.com/recaptcha/challengepage",
    "cf-browser-verification",
    "checking your browser before accessing",
    "<title>just a moment...</title>",
    "attention required! | cloudflare",
    "enable javascript and cookies to continue",
    "client challenge",
    "a required part of this site couldn't load",
    "a required part of this site couldn’t load",
    "verify you are human",
    "please enable cookies",
    "ddos-guard",
)

_SOFT_404_MARKERS = (
    "<title>404",
    "<title>error 404",
    "<title>page not found",
    "<title>not found",
    "<title>user not found",
    "<title>profile not found",
    "this page isn't available",
    "this page is not available",
    "page doesn't exist",
    "user does not exist",
    "account doesn't exist",
    "no such user",
    "user not found",
    "profile not found",
    "member not found",
    "user was not found",
    "no user found",
    "couldn't find that user",
    "could not find that user",
    "invalid username",
    "not a valid user",
    "this account has been suspended",
    "page cannot be found",
    "404 not found",
    "nothing here",
)

_BAD_FINAL_URL_MARKERS = (
    "/login",
    "/signin",
    "/sign_in",
    "/signup",
    "/register",
    "/notfound",
    "/not-found",
    "/404",
    "/error",
    "/dashboard",
    "/verify-human",
    "authwall",
    "/.within.website/",
    "challenge",
)


def load_false_positive_hosts() -> set[str]:
    """Load curated catch-all / soft-404 host blocklist from false_positive_sites.json."""
    global _FALSE_POSITIVE_HOSTS_CACHE
    if _FALSE_POSITIVE_HOSTS_CACHE is not None:
        return _FALSE_POSITIVE_HOSTS_CACHE
    hosts: set[str] = set()
    if _FALSE_POSITIVE_SITES_FILE.exists():
        try:
            with open(_FALSE_POSITIVE_SITES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for h in data.get("hosts") or []:
                if isinstance(h, str) and h.strip():
                    hosts.add(h.strip().lower())
        except Exception as exc:
            logger.debug("Could not load false_positive_sites.json: %s", exc)
    _FALSE_POSITIVE_HOSTS_CACHE = hosts
    return _FALSE_POSITIVE_HOSTS_CACHE

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _load_dead_hosts() -> dict[str, float]:
    """Load persisted dead-host negative cache (expired entries pruned)."""
    global _DEAD_HOSTS_CACHE
    if _DEAD_HOSTS_CACHE is not None:
        return _DEAD_HOSTS_CACHE

    now = time.time()
    loaded: dict[str, float] = {}
    if _DEAD_HOSTS_FILE.exists():
        try:
            with open(_DEAD_HOSTS_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                loaded = {
                    str(host): float(ts)
                    for host, ts in raw.items()
                    if isinstance(ts, (int, float)) and (now - float(ts)) < _DEAD_HOSTS_TTL_SECONDS
                }
        except Exception:
            loaded = {}

    _DEAD_HOSTS_CACHE = loaded
    return _DEAD_HOSTS_CACHE


def _save_dead_hosts(dead_hosts: dict[str, float]) -> None:
    """Persist dead-host negative cache to disk."""
    try:
        with open(_DEAD_HOSTS_FILE, "w", encoding="utf-8") as f:
            json.dump(dead_hosts, f)
    except Exception as exc:
        logger.debug("Could not persist dead hosts cache: %s", exc)


def load_maigret_sites() -> list[tuple[int, str, dict[str, Any]]]:
    """Load and cache all active, engine-resolved username site rules sorted by traffic rank."""
    global _CACHED_SITES
    if _CACHED_SITES is not None:
        return _CACHED_SITES

    try:
        import maigret  # type: ignore

        db_path = os.path.join(os.path.dirname(maigret.__file__), "resources", "data.json")
        if not os.path.exists(db_path):
            return []

        with open(db_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        raw_sites = raw_data.get("sites", {})
        engines = raw_data.get("engines", {})

        candidates: list[tuple[int, str, dict[str, Any]]] = []
        for name, cfg in raw_sites.items():
            if not isinstance(cfg, dict) or cfg.get("disabled"):
                continue
            if cfg.get("type", "username") != "username":
                continue
            if "(by id)" in name.lower():
                continue

            # Resolve engine inheritance (DiscourseJson, XenForo, uCoz, vBulletin, etc.)
            eng_name = cfg.get("engine")
            eng_site = engines.get(eng_name, {}).get("site", {}) if eng_name else {}
            merged = {**eng_site, **cfg}

            url_main = str(merged.get("urlMain") or "").rstrip("/")
            url_subpath = str(merged.get("urlSubpath") or "")

            url_tpl = str(merged.get("url") or "")
            if "{urlMain}" in url_tpl:
                url_tpl = url_tpl.replace("{urlMain}", url_main).replace("{urlSubpath}", url_subpath)

            probe_tpl = str(merged.get("urlProbe") or url_tpl)
            if "{urlMain}" in probe_tpl:
                probe_tpl = probe_tpl.replace("{urlMain}", url_main).replace("{urlSubpath}", url_subpath)

            if not url_tpl or "{username}" not in url_tpl or "{username}" not in probe_tpl:
                continue

            ctype = merged.get("checkType")
            if ctype not in ("status_code", "message", "response_url"):
                continue

            resolved_cfg = {
                "url": url_tpl,
                "urlProbe": probe_tpl,
                "checkType": ctype,
                "presenseStrs": merged.get("presenseStrs"),
                "absenceStrs": merged.get("absenceStrs"),
                "regexCheck": merged.get("regexCheck"),
                "host": urlparse(probe_tpl).netloc.lower(),
            }

            rank = int(merged.get("alexaRank") or 9_999_999)
            candidates.append((rank, name, resolved_cfg))

        candidates.sort(key=lambda x: x[0])
        _CACHED_SITES = candidates
        logger.info("Loaded %d active username site rules from Maigret DB.", len(candidates))
        return candidates
    except Exception as exc:
        logger.warning("Failed to load local Maigret site database: %s", exc)
        return []


def _is_bot_or_soft404_response(
    resp: httpx.Response,
    body: str,
    ctype: str | None,
    probe_url: str | None = None,
) -> bool:
    """Detect WAF/bot challenge pages, generic redirects, and soft-404 responses returning HTTP 200."""
    # Reject 202 Accepted (commonly used by Akamai/WAF challenges)
    if resp.status_code == 202:
        return True

    final_parsed = urlparse(str(resp.url).lower())
    final_url_lower = str(resp.url).lower()
    if any(marker in final_url_lower for marker in _BAD_FINAL_URL_MARKERS):
        return True

    # If probe_url had a specific profile path (e.g. /user/foo) and redirected to root homepage (/), it's a catch-all
    if probe_url:
        orig_path = urlparse(probe_url.lower()).path.strip("/")
        final_path = final_parsed.path.strip("/")
        if orig_path and not final_path:
            return True

    snippet = body[:65_536].lower()
    if any(marker in snippet for marker in _BOT_CHALLENGE_MARKERS):
        return True

    if any(marker in snippet for marker in _SOFT_404_MARKERS):
        return True

    if "application/json" in resp.headers.get("content-type", "").lower():
        compact = re.sub(r"\s+", "", body.strip()).lower()
        if (
            compact in ("{}", "[]", "null", "")
            or '"users":{}' in compact
            or '"users":[]' in compact
            or '"data":null' in compact
            or '"user":null' in compact
            or '"success":false' in compact
        ):
            return True

    return False


async def _probe_site(
    client: httpx.AsyncClient,
    name: str,
    cfg: dict[str, Any],
    probe_username: str,
    target_id: str,
    newly_dead: dict[str, float] | None = None,
) -> Hit | None:
    """Probe a single site rule for `probe_username` and return a Hit if matched."""
    probe_tpl = cfg.get("urlProbe") or cfg.get("url")
    display_tpl = cfg.get("url") or probe_tpl
    if not probe_tpl or not display_tpl:
        return None

    probe_url = probe_tpl.replace("{username}", probe_username)
    display_url = display_tpl.replace("{username}", probe_username)
    ctype = cfg.get("checkType")
    host = cfg.get("host") or urlparse(probe_url).netloc.lower()

    try:
        resp = await client.get(
            probe_url,
            follow_redirects=(ctype != "response_url"),
        )
        body = resp.text if resp.status_code < 400 else ""
        if _is_bot_or_soft404_response(resp, body, ctype, probe_url=probe_url):
            return None

        matched = False
        if ctype == "status_code":
            matched = resp.status_code == 200
        elif ctype == "message":
            if resp.status_code == 200:
                abs_strs = cfg.get("absenceStrs") or []
                pres_strs = cfg.get("presenseStrs") or []
                if not any(s in body for s in abs_strs):
                    if pres_strs:
                        matched = any(s in body for s in pres_strs)
                    elif abs_strs:
                        matched = True
        elif ctype == "response_url":
            matched = resp.status_code == 200

        if not matched:
            return None

        bio = ""
        location = ""
        full_name = probe_username

        if "application/json" in resp.headers.get("content-type", "").lower():
            try:
                jdata = resp.json()
                if isinstance(jdata, dict):
                    full_name = jdata.get("name") or full_name
                    bio = jdata.get("bio") or jdata.get("company") or ""
                    location = jdata.get("location") or ""
            except Exception:
                pass

        return Hit(
            target_entity_id=target_id,
            discovered_type=EntityType.PROFILE,
            discovered_value=display_url,
            relation="account_on",
            source_tool="maigret",
            is_trusted=False,
            raw_data={
                "site": name,
                "description": bio,
                "bio": bio,
                "location": location,
                "name": full_name,
                "username": probe_username,
            },
            metadata={
                "username": probe_username,
                "site": name,
                "url": display_url,
                "bio": bio,
                "location": location,
                "name": full_name,
            },
        )
    except httpx.ConnectError as exc:
        err_msg = str(exc).lower()
        if newly_dead is not None and host and any(
            msg in err_msg
            for msg in ("name or service not known", "nodename", "no address associated")
        ):
            newly_dead[host] = time.time()
        return None
    except Exception:
        return None


async def stream_username_check(
    username: str,
    target_id: str,
    max_sites: int = 11000,
    concurrency: int = 350,
    overall_timeout: float = 35.0,
    on_hit: Callable[[Hit], None] | None = None,
) -> list[Hit]:
    """Check up to all 10k+ active sites concurrently, pruning dead hosts and catch-all 200 false positives."""
    all_sites = load_maigret_sites()
    if not all_sites:
        return []

    dead_hosts = _load_dead_hosts()
    fp_hosts = load_false_positive_hosts()
    newly_dead: dict[str, float] = {}

    filtered: list[tuple[str, dict[str, Any]]] = []
    skipped_dead = 0

    for _, name, cfg in all_sites:
        host = cfg.get("host") or urlparse(cfg.get("urlProbe", "")).netloc.lower()
        if host and (host in dead_hosts or host in fp_hosts):
            skipped_dead += 1
            continue

        rx = cfg.get("regexCheck")
        if rx:
            try:
                if not re.match(rx, username):
                    continue
            except re.error:
                pass
        filtered.append((name, cfg))
        if len(filtered) >= max_sites:
            break

    if skipped_dead:
        logger.info("Skipped %d known dead/unreachable hosts from negative cache.", skipped_dead)

    sem = asyncio.Semaphore(concurrency)
    candidates: list[tuple[Hit, str, dict[str, Any]]] = []
    seen_urls: set[str] = set()

    # Generous 3s connect timeout prevents false drops; 4s read timeout
    timeout_cfg = httpx.Timeout(4.0, connect=3.0)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=100)

    queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
    for item in filtered:
        queue.put_nowait(item)

    async def worker(client: httpx.AsyncClient) -> None:
        while True:
            try:
                name, cfg = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                async with sem:
                    hit = await _probe_site(client, name, cfg, username, target_id, newly_dead)
                if hit and hit.discovered_value not in seen_urls:
                    seen_urls.add(hit.discovered_value)
                    candidates.append((hit, name, cfg))
            finally:
                queue.task_done()

    verified_hits: list[Hit] = []
    async with httpx.AsyncClient(headers=BROWSER_HEADERS, limits=limits, timeout=timeout_cfg) as client:
        num_workers = min(concurrency, max(1, len(filtered)))
        workers = [asyncio.create_task(worker(client)) for _ in range(num_workers)]
        try:
            await asyncio.wait_for(asyncio.gather(*workers, return_exceptions=True), timeout=overall_timeout)
        except asyncio.TimeoutError:
            pass
        finally:
            for w in workers:
                if not w.done():
                    w.cancel()

        # Baseline false-positive filter: probe each candidate site with a guaranteed-nonexistent username.
        # Any site that still matches for the nonexistent username returns 200/match for all usernames.
        if candidates:
            async def _verify(hit: Hit, name: str, cfg: dict[str, Any]) -> Hit | None:
                baseline_hit = await _probe_site(client, name, cfg, _BASELINE_PROBE_USERNAME, target_id)
                if baseline_hit is not None:
                    logger.debug("Filtered catch-all false positive site %s (%s)", name, hit.discovered_value)
                    bhost = cfg.get("host", "")
                    if bhost:
                        newly_dead[bhost] = time.time()
                    return None
                return hit

            results = await asyncio.gather(*(_verify(h, n, c) for h, n, c in candidates), return_exceptions=True)
            for res in results:
                if isinstance(res, Hit):
                    verified_hits.append(res)
                    if on_hit:
                        on_hit(res)

    if newly_dead:
        dead_hosts.update(newly_dead)
        _save_dead_hosts(dead_hosts)
        logger.info("Pruned %d newly unreachable hosts into dead-host cache.", len(newly_dead))

    return verified_hits
