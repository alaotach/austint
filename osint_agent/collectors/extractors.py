"""Normalized profile schema and 4-tier public extractor registry.

Tiers tried in order per platform:
  a. Official/public API or JSON endpoint (`source="api"`)
  b. Embedded structured data: JSON-LD, __NEXT_DATA__, __NUXT__, __sc_hydration, initial-state (`source="structured"`)
  c. Playwright-rendered DOM for JS-only public pages (`source="dom"`)
  d. ML/heuristic text fallback (`source="ml"`, lower confidence)
"""

from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from html import unescape
from typing import Any, Awaitable, Callable, Literal
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from osint_agent.collectors.enrich import can_fetch_robots, is_private_or_loopback_host


DEFAULT_UA = "IntelligentOSINT/1.0 (+https://github.com/alaotach/austint)"


class NormalizedProfile(BaseModel):
    """Canonical public profile schema rendered by UI cards and Inspector."""

    username: str = ""
    display_name: str = ""
    bio: str = ""
    location: str = ""
    country: str = ""
    joined_at: str = ""
    last_active: str = ""
    followers: int | str | None = None
    following: int | str | None = None
    post_count: dict[str, int] = Field(default_factory=dict)
    total_views: int | None = None
    links: list[str] = Field(default_factory=list)
    avatar_url: str = ""
    banner_url: str = ""
    verified: bool | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    field_sources: dict[str, Literal["api", "structured", "dom", "ml"]] = Field(default_factory=dict)
    field_confidence: dict[str, float] = Field(default_factory=dict)
    limited_public_data: bool = False
    status_note: str = ""

    def set_field(
        self,
        name: str,
        value: Any,
        source: Literal["api", "structured", "dom", "ml"],
        confidence: float | None = None,
    ) -> None:
        if value is None or value == "" or value == [] or value == {}:
            return
        setattr(self, name, value)
        self.field_sources[name] = source
        default_conf = {"api": 0.95, "structured": 0.85, "dom": 0.70, "ml": 0.45}[source]
        self.field_confidence[name] = round(float(confidence if confidence is not None else default_conf), 2)

    def to_metadata_dict(self) -> dict[str, Any]:
        """Convert NormalizedProfile into Hit.metadata-compatible dictionary."""
        out: dict[str, Any] = {
            "normalized_profile": self.model_dump(),
            "field_sources": dict(self.field_sources),
            "field_confidence": dict(self.field_confidence),
        }
        if self.username:
            out["username"] = self.username
        if self.display_name:
            out["name"] = self.display_name
        if self.bio:
            out["bio"] = self.bio
        if self.location:
            out["location"] = self.location
        if self.country:
            out["country"] = self.country
        if self.joined_at:
            out["joined"] = self.joined_at
        if self.last_active:
            out["activity_dates"] = self.last_active
        if self.followers is not None:
            out["followers"] = self.followers
        if self.following is not None:
            out["following"] = self.following
        if self.post_count:
            out["post_count"] = dict(self.post_count)
            for k, v in self.post_count.items():
                out[k] = v
        if self.total_views is not None:
            out["total_views"] = self.total_views
        if self.links:
            out["linked_urls"] = list(self.links)
        if self.avatar_url:
            out["avatar_url"] = self.avatar_url
        if self.banner_url:
            out["banner_url"] = self.banner_url
        if self.verified is not None:
            out["verified"] = self.verified
        if self.extra:
            out["extra_fields"] = dict(self.extra)
            if self.extra.get("languages"):
                out["languages"] = self.extra["languages"]
            if self.extra.get("orgs"):
                out["orgs"] = self.extra["orgs"]
            if self.extra.get("company"):
                out["company"] = self.extra["company"]
            if self.extra.get("website"):
                out["website"] = self.extra["website"]
            if self.extra.get("stories"):
                out["stories"] = self.extra["stories"]
            if self.extra.get("rating") is not None:
                out["rating"] = self.extra["rating"]
            if self.extra.get("rank"):
                out["rank"] = self.extra["rank"]
        if self.limited_public_data:
            out["limited_public_data"] = True
            out["status_note"] = self.status_note or "exists, limited public data"
        return out


ExtractorFn = Callable[[str, str, httpx.AsyncClient], Awaitable[NormalizedProfile]]
EXTRACTOR_REGISTRY: dict[str, ExtractorFn] = {}


def register_extractor(platform: str) -> Callable[[ExtractorFn], ExtractorFn]:
    def _decorator(fn: ExtractorFn) -> ExtractorFn:
        EXTRACTOR_REGISTRY[platform] = fn
        return fn

    return _decorator


def _emit_unavailable(tool: str, reason: str) -> None:
    try:
        from osint_agent.collectors.tools import log_audit_event

        log_audit_event("tool_unavailable", {"tool": tool, "reason": reason})
    except Exception:
        pass


@register_extractor("dockerhub")
async def extract_dockerhub(url: str, username: str, client: httpx.AsyncClient) -> NormalizedProfile:
    prof = NormalizedProfile(username=username)
    try:
        r = await client.get(f"https://hub.docker.com/v2/users/{username}/", headers={"User-Agent": DEFAULT_UA}, timeout=8.0)
        if r.status_code == 200:
            d = r.json()
            prof.set_field("display_name", d.get("full_name") or d.get("username"), "api")
            prof.set_field("location", d.get("location"), "api")
            prof.set_field("joined_at", str(d.get("date_joined") or "")[:10], "api")
            prof.set_field("avatar_url", d.get("avatar_url") or d.get("gravatar_url"), "api")
            if d.get("company"):
                prof.extra["company"] = d["company"]
            if d.get("profile_url"):
                prof.set_field("links", [d["profile_url"]], "api")
        rr = await client.get(f"https://hub.docker.com/v2/repositories/{username}/?page_size=10", headers={"User-Agent": DEFAULT_UA}, timeout=8.0)
        if rr.status_code == 200:
            rd = rr.json()
            cnt = rd.get("count")
            if cnt is not None:
                prof.set_field("post_count", {"repos": int(cnt)}, "api")
    except Exception:
        pass
    return prof


@register_extractor("soundcloud")
async def extract_soundcloud(url: str, username: str, client: httpx.AsyncClient) -> NormalizedProfile:
    prof = NormalizedProfile(username=username)
    try:
        r = await client.get(url, headers={"User-Agent": DEFAULT_UA}, follow_redirects=True, timeout=8.0)
        if r.status_code != 200:
            return prof
        html = r.text[:350_000]
        m = re.search(r"window\.__sc_hydration\s*=\s*(\[.*?\]);\s*</script>", html, re.DOTALL)
        if m:
            items = json.loads(m.group(1))
            for it in items:
                if isinstance(it, dict) and it.get("hydratable") == "user":
                    u = it.get("data") or {}
                    prof.set_field("display_name", u.get("full_name") or u.get("username"), "structured")
                    prof.set_field("bio", u.get("description"), "structured")
                    prof.set_field("country", u.get("country_code"), "structured")
                    prof.set_field("location", u.get("city"), "structured")
                    prof.set_field("joined_at", str(u.get("created_at") or "")[:10], "structured")
                    prof.set_field("followers", u.get("followers_count"), "structured")
                    prof.set_field("following", u.get("followings_count"), "structured")
                    prof.set_field("avatar_url", u.get("avatar_url"), "structured")
                    if u.get("track_count") is not None:
                        prof.set_field("post_count", {"tracks": int(u["track_count"])}, "structured")
                    visuals = ((u.get("visuals") or {}).get("visuals") or [])
                    if visuals and isinstance(visuals[0], dict) and visuals[0].get("visual_url"):
                        prof.set_field("banner_url", visuals[0]["visual_url"], "structured")
                    break
    except Exception:
        pass
    return prof


@register_extractor("youtube")
async def extract_youtube(url: str, username: str, client: httpx.AsyncClient) -> NormalizedProfile:
    from osint_agent.llm.client import _load_dotenv

    _load_dotenv()
    prof = NormalizedProfile(username=username)
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    handle = username.lstrip("@")
    if api_key and handle:
        try:
            r = await client.get(
                "https://www.googleapis.com/youtube/v3/channels",
                params={"part": "snippet,statistics,brandingSettings", "forHandle": f"@{handle}", "key": api_key},
                timeout=8.0,
            )
            if r.status_code == 200:
                items = r.json().get("items") or []
                if items:
                    ch = items[0]
                    sn = ch.get("snippet") or {}
                    st = ch.get("statistics") or {}
                    br = (ch.get("brandingSettings") or {}).get("image") or {}
                    prof.set_field("display_name", sn.get("title"), "api")
                    prof.set_field("bio", sn.get("description"), "api")
                    prof.set_field("country", sn.get("country"), "api")
                    prof.set_field("joined_at", str(sn.get("publishedAt") or "")[:10], "api")
                    thumb = ((sn.get("thumbnails") or {}).get("high") or {}).get("url")
                    prof.set_field("avatar_url", thumb, "api")
                    prof.set_field("banner_url", br.get("bannerExternalUrl"), "api")
                    if st.get("subscriberCount") is not None:
                        prof.set_field("followers", int(st["subscriberCount"]), "api")
                    if st.get("videoCount") is not None:
                        prof.set_field("post_count", {"videos": int(st["videoCount"])}, "api")
                    if st.get("viewCount") is not None:
                        prof.set_field("total_views", int(st["viewCount"]), "api")
                    return prof
        except Exception:
            pass
    elif not api_key:
        _emit_unavailable("youtube_api", "YOUTUBE_API_KEY not set; falling back to structured HTML tags.")

    # Tier 2 fallback: public og:* & ytInitialData metadata
    try:
        r = await client.get(url, headers={"User-Agent": DEFAULT_UA}, follow_redirects=True, timeout=8.0)
        if r.status_code == 200:
            html = r.text[:250_000]
            og_t = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
            og_d = re.search(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
            og_i = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
            if og_t:
                prof.set_field("display_name", unescape(og_t.group(1)), "structured")
            if og_d:
                prof.set_field("bio", unescape(og_d.group(1))[:400], "structured")
            if og_i:
                prof.set_field("avatar_url", og_i.group(1), "structured")
            sub_m = re.search(r'"subscriberCountText":\{"simpleText":"([^"]+)"', html)
            if sub_m:
                prof.extra["subscribers_text"] = sub_m.group(1)
            vid_m = re.search(r'"videosCountText":\{"runs":\[\{"text":"([^"]+)"', html)
            if vid_m:
                prof.extra["videos_text"] = vid_m.group(1)
    except Exception:
        pass
    return prof


@register_extractor("chess")
async def extract_chess(url: str, username: str, client: httpx.AsyncClient) -> NormalizedProfile:
    prof = NormalizedProfile(username=username)
    try:
        r = await client.get(f"https://api.chess.com/pub/player/{username.lower()}", headers={"User-Agent": DEFAULT_UA}, timeout=8.0)
        if r.status_code == 200:
            d = r.json()
            prof.set_field("display_name", d.get("name") or d.get("username"), "api")
            prof.set_field("avatar_url", d.get("avatar"), "api")
            prof.set_field("location", d.get("location"), "api")
            if d.get("country"):
                prof.set_field("country", str(d["country"]).rstrip("/").split("/")[-1], "api")
            if d.get("followers") is not None:
                prof.set_field("followers", int(d["followers"]), "api")
            if d.get("joined"):
                from datetime import datetime, timezone

                prof.set_field("joined_at", datetime.fromtimestamp(int(d["joined"]), tz=timezone.utc).strftime("%Y-%m-%d"), "api")
            if d.get("last_online"):
                from datetime import datetime, timezone

                prof.set_field("last_active", datetime.fromtimestamp(int(d["last_online"]), tz=timezone.utc).strftime("%Y-%m-%d"), "api")
            prof.set_field("verified", bool(d.get("verified")), "api")
        sr = await client.get(f"https://api.chess.com/pub/player/{username.lower()}/stats", headers={"User-Agent": DEFAULT_UA}, timeout=8.0)
        if sr.status_code == 200:
            sd = sr.json()
            ratings = {
                mode.replace("chess_", ""): sd[mode]["last"]["rating"]
                for mode in ("chess_blitz", "chess_rapid", "chess_bullet")
                if isinstance(sd.get(mode), dict) and isinstance(sd[mode].get("last"), dict)
            }
            if ratings:
                prof.extra["ratings"] = ratings
    except Exception:
        pass
    return prof


@register_extractor("codeforces")
async def extract_codeforces(url: str, username: str, client: httpx.AsyncClient) -> NormalizedProfile:
    prof = NormalizedProfile(username=username)
    try:
        r = await client.get(f"https://codeforces.com/api/user.info?handles={username}", headers={"User-Agent": DEFAULT_UA}, timeout=8.0)
        if r.status_code == 200 and r.json().get("status") == "OK":
            u = (r.json().get("result") or [{}])[0]
            name = " ".join(x for x in (u.get("firstName"), u.get("lastName")) if x).strip()
            prof.set_field("display_name", name or u.get("handle"), "api")
            prof.set_field("country", u.get("country"), "api")
            prof.set_field("location", u.get("city"), "api")
            prof.set_field("avatar_url", u.get("titlePhoto") or u.get("avatar"), "api")
            if u.get("organization"):
                prof.extra["orgs"] = [u["organization"]]
            if u.get("rating") is not None:
                prof.extra["rating"] = u["rating"]
                prof.extra["rank"] = u.get("rank")
            if u.get("friendOfCount") is not None:
                prof.set_field("followers", int(u["friendOfCount"]), "api")
            if u.get("registrationTimeSeconds"):
                from datetime import datetime, timezone

                prof.set_field("joined_at", datetime.fromtimestamp(int(u["registrationTimeSeconds"]), tz=timezone.utc).strftime("%Y-%m-%d"), "api")
            if u.get("lastOnlineTimeSeconds"):
                from datetime import datetime, timezone

                prof.set_field("last_active", datetime.fromtimestamp(int(u["lastOnlineTimeSeconds"]), tz=timezone.utc).strftime("%Y-%m-%d"), "api")
    except Exception:
        pass
    return prof


@register_extractor("npm")
async def extract_npm(url: str, username: str, client: httpx.AsyncClient) -> NormalizedProfile:
    prof = NormalizedProfile(username=username)
    try:
        r = await client.get(
            f"https://registry.npmjs.org/-/v1/search?text=maintainer:{username}&size=10",
            headers={"User-Agent": DEFAULT_UA},
            timeout=8.0,
        )
        if r.status_code == 200:
            data = r.json()
            objs = data.get("objects") or []
            total = int(data.get("total") or len(objs))
            prof.set_field("post_count", {"packages": total}, "api")
            links = [
                str((o.get("package") or {}).get("links", {}).get("npm") or "")
                for o in objs
                if isinstance(o, dict)
            ]
            links = [l for l in links if l]
            if links:
                prof.set_field("links", links[:6], "api")
            pkgs = [str((o.get("package") or {}).get("name") or "") for o in objs if isinstance(o, dict)]
            if pkgs:
                prof.extra["packages"] = [p for p in pkgs if p][:8]
    except Exception:
        pass
    return prof


@register_extractor("pypi")
async def extract_pypi(url: str, username: str, client: httpx.AsyncClient) -> NormalizedProfile:
    prof = NormalizedProfile(username=username)
    try:
        r = await client.get(url, headers={"User-Agent": DEFAULT_UA}, follow_redirects=True, timeout=8.0)
        if r.status_code == 200:
            html = r.text[:200_000]
            pkgs = re.findall(r'<h3[^>]*class="package-snippet__title"[^>]*>.*?<span[^>]*class="package-snippet__name"[^>]*>(.*?)</span>', html, re.DOTALL | re.I)
            clean_pkgs = [unescape(p).strip() for p in pkgs if p.strip()]
            if clean_pkgs:
                prof.set_field("post_count", {"packages": len(clean_pkgs)}, "structured")
                prof.extra["packages"] = clean_pkgs[:10]
            nm = re.search(r'<h1[^>]*class="author-profile__name"[^>]*>(.*?)</h1>', html, re.DOTALL | re.I)
            if nm:
                prof.set_field("display_name", unescape(re.sub(r"<[^>]+>", "", nm.group(1))).strip(), "structured")
            jm = re.search(r'<time[^>]+datetime="([^"]+)"', html, re.I)
            if jm:
                prof.set_field("joined_at", jm.group(1)[:10], "structured")
    except Exception:
        pass
    return prof


@register_extractor("steam")
async def extract_steam(url: str, username: str, client: httpx.AsyncClient) -> NormalizedProfile:
    prof = NormalizedProfile(username=username)
    api_key = os.getenv("STEAM_API_KEY", "").strip()
    if not api_key:
        _emit_unavailable("steam_api", "STEAM_API_KEY not set; falling back to public ?xml=1 profile feed.")
    try:
        xml_url = url.rstrip("/") + "/?xml=1"
        r = await client.get(xml_url, headers={"User-Agent": DEFAULT_UA}, follow_redirects=True, timeout=8.0)
        if r.status_code == 200 and "<profile>" in r.text[:500]:
            root = ET.fromstring(r.text)
            prof.set_field("display_name", (root.findtext("steamID") or "").strip(), "structured")
            prof.set_field("avatar_url", (root.findtext("avatarFull") or "").strip(), "structured")
            prof.set_field("location", (root.findtext("location") or "").strip(), "structured")
            prof.set_field("joined_at", (root.findtext("memberSince") or "").strip(), "structured")
            prof.set_field("bio", unescape(re.sub(r"<[^>]+>", " ", root.findtext("summary") or "")).strip()[:400], "structured")
    except Exception:
        pass
    return prof


def _extract_og_meta(html: str, prop: str) -> str:
    m = re.search(
        rf'<meta[^>]+property=["\']{re.escape(prop)}["\'][^>]+content=["\']([^"\']*)["\']',
        html,
        re.I,
    )
    if not m:
        m = re.search(
            rf'<meta[^>]+content=["\']([^"\']*)["\'][^>]+property=["\']{re.escape(prop)}["\']',
            html,
            re.I,
        )
    return unescape(m.group(1)).strip() if m else ""


@register_extractor("telegram")
async def extract_telegram(url: str, username: str, client: httpx.AsyncClient) -> NormalizedProfile:
    prof = NormalizedProfile(username=username)
    try:
        r = await client.get(url, headers={"User-Agent": DEFAULT_UA}, follow_redirects=True, timeout=8.0)
        if r.status_code == 200:
            html = r.text[:150_000]
            tm = re.search(r'<div[^>]*class="tgme_page_title"[^>]*>(.*?)</div>', html, re.DOTALL | re.I)
            dm = re.search(r'<div[^>]*class="tgme_page_description"[^>]*>(.*?)</div>', html, re.DOTALL | re.I)
            em = re.search(r'<div[^>]*class="tgme_page_extra"[^>]*>(.*?)</div>', html, re.DOTALL | re.I)
            im = re.search(r'<img[^>]*class="tgme_page_photo_image"[^>]*src="([^"]+)"', html, re.I)
            if tm:
                prof.set_field("display_name", unescape(re.sub(r"<[^>]+>", "", tm.group(1))).strip(), "structured")
            if not prof.display_name:
                prof.set_field("display_name", _extract_og_meta(html, "og:title"), "structured")
            if dm:
                prof.set_field("bio", unescape(re.sub(r"<[^>]+>", " ", dm.group(1))).strip()[:400], "structured")
            if not prof.bio:
                prof.set_field("bio", _extract_og_meta(html, "og:description")[:400], "structured")
            if im:
                prof.set_field("avatar_url", im.group(1), "structured")
            if not prof.avatar_url:
                prof.set_field("avatar_url", _extract_og_meta(html, "og:image"), "structured")
            if em:
                extra_txt = unescape(re.sub(r"<[^>]+>", " ", em.group(1))).strip()
                sub_m = re.search(r"([\d\s]+)\s+(?:subscribers|members)", extra_txt, re.I)
                if sub_m:
                    prof.set_field("followers", int(sub_m.group(1).replace(" ", "")), "structured")
    except Exception:
        pass
    return prof


@register_extractor("medium")
async def extract_medium(url: str, username: str, client: httpx.AsyncClient) -> NormalizedProfile:
    prof = NormalizedProfile(username=username)
    handle = username.lstrip("@")
    try:
        r = await client.get(f"https://medium.com/feed/@{handle}", headers={"User-Agent": DEFAULT_UA}, follow_redirects=True, timeout=8.0)
        if r.status_code == 200 and "<rss" in r.text[:500]:
            root = ET.fromstring(r.text)
            ch = root.find("channel")
            if ch is not None:
                prof.set_field("display_name", (ch.findtext("title") or "").replace("Stories by ", "").replace(" on Medium", "").strip(), "structured")
                prof.set_field("bio", (ch.findtext("description") or "").strip()[:400], "structured")
                img_u = ch.findtext("image/url")
                if img_u:
                    prof.set_field("avatar_url", img_u.strip(), "structured")
                items = ch.findall("item")
                if items:
                    prof.set_field("post_count", {"posts": len(items)}, "structured")
                    prof.extra["recent_posts"] = [it.findtext("title") for it in items[:5] if it.findtext("title")]
    except Exception:
        pass
    return prof


@register_extractor("github")
async def extract_github(url: str, username: str, client: httpx.AsyncClient) -> NormalizedProfile:
    from osint_agent.llm.client import _load_dotenv

    _load_dotenv()
    prof = NormalizedProfile(username=username)
    headers = {"User-Agent": DEFAULT_UA, "Accept": "application/vnd.github+json"}
    gh_tok = os.getenv("GITHUB_TOKEN", "").strip()
    if gh_tok:
        headers["Authorization"] = f"Bearer {gh_tok}"
    try:
        r = await client.get(f"https://api.github.com/users/{username}", headers=headers, timeout=8.0)
        if r.status_code == 200:
            d = r.json()
            prof.set_field("display_name", d.get("name") or d.get("login"), "api")
            prof.set_field("bio", d.get("bio"), "api")
            prof.set_field("location", d.get("location"), "api")
            prof.set_field("joined_at", str(d.get("created_at") or "")[:10], "api")
            prof.set_field("last_active", str(d.get("updated_at") or "")[:10], "api")
            prof.set_field("followers", d.get("followers"), "api")
            prof.set_field("following", d.get("following"), "api")
            prof.set_field("avatar_url", d.get("avatar_url"), "api")
            if d.get("public_repos") is not None:
                prof.set_field("post_count", {"repos": int(d["public_repos"])}, "api")
            if d.get("company"):
                prof.extra["company"] = str(d["company"]).strip()
            links = [u for u in [d.get("blog")] if isinstance(u, str) and u.startswith("http")]
            if links:
                prof.extra["website"] = links[0]
            rs = await client.get(f"https://api.github.com/users/{username}/social_accounts", headers=headers, timeout=8.0)
            if rs.status_code == 200 and isinstance(rs.json(), list):
                for s in rs.json():
                    if isinstance(s, dict) and isinstance(s.get("url"), str) and s["url"].startswith("http"):
                        links.append(s["url"])
            if links:
                prof.set_field("links", list(dict.fromkeys(links)), "api")
        rr = await client.get(f"https://api.github.com/users/{username}/repos?sort=updated&per_page=10", headers=headers, timeout=8.0)
        if rr.status_code == 200 and isinstance(rr.json(), list):
            langs = list(dict.fromkeys(str(repo.get("language")) for repo in rr.json() if repo.get("language")))
            if langs:
                prof.extra["languages"] = langs[:8]
        ro = await client.get(f"https://api.github.com/users/{username}/orgs", headers=headers, timeout=8.0)
        if ro.status_code == 200 and isinstance(ro.json(), list):
            orgs = [str(o.get("login")) for o in ro.json() if o.get("login")]
            if orgs:
                prof.extra["orgs"] = orgs[:8]
    except Exception:
        pass
    return prof


@register_extractor("devto")
async def extract_devto(url: str, username: str, client: httpx.AsyncClient) -> NormalizedProfile:
    prof = NormalizedProfile(username=username)
    try:
        r = await client.get(f"https://dev.to/api/users/by_username?url={username}", headers={"User-Agent": DEFAULT_UA}, timeout=8.0)
        if r.status_code == 200:
            d = r.json()
            prof.set_field("display_name", d.get("name") or d.get("username"), "api")
            prof.set_field("bio", d.get("summary"), "api")
            prof.set_field("location", d.get("location"), "api")
            prof.set_field("joined_at", str(d.get("joined_at") or "")[:20], "api")
            prof.set_field("avatar_url", d.get("profile_image"), "api")
            links = [u for u in [d.get("website_url")] if isinstance(u, str) and u.startswith("http")]
            if d.get("github_username"):
                links.append(f"https://github.com/{d['github_username']}")
            if d.get("twitter_username"):
                links.append(f"https://x.com/{d['twitter_username']}")
            if links:
                prof.set_field("links", list(dict.fromkeys(links)), "api")
    except Exception:
        pass
    return prof


@register_extractor("huggingface")
async def extract_huggingface(url: str, username: str, client: httpx.AsyncClient) -> NormalizedProfile:
    prof = NormalizedProfile(username=username)
    try:
        r = await client.get(f"https://huggingface.co/api/users/{username}/overview", headers={"User-Agent": DEFAULT_UA}, timeout=8.0)
        if r.status_code == 200:
            d = r.json()
            prof.set_field("display_name", d.get("fullname") or d.get("user"), "api")
            prof.set_field("bio", d.get("details"), "api")
            prof.set_field("avatar_url", d.get("avatarUrl"), "api")
            if d.get("numFollowers") is not None:
                prof.set_field("followers", int(d["numFollowers"]), "api")
            counts = {}
            for k, label in (("numModels", "models"), ("numDatasets", "datasets"), ("numSpaces", "spaces")):
                if d.get(k) is not None:
                    counts[label] = int(d[k])
            if counts:
                prof.set_field("post_count", counts, "api")
    except Exception:
        pass
    return prof


@register_extractor("wattpad")
async def extract_wattpad(url: str, username: str, client: httpx.AsyncClient) -> NormalizedProfile:
    prof = NormalizedProfile(username=username)
    try:
        r = await client.get(
            f"https://www.wattpad.com/api/v3/users/{username}?fields=username,name,description,avatar,numFollowers,numFollowing,numStoriesPublished,createDate,location,website",
            headers={"User-Agent": DEFAULT_UA},
            timeout=8.0,
        )
        if r.status_code == 200:
            d = r.json()
            prof.set_field("display_name", d.get("name") or d.get("username"), "api")
            prof.set_field("bio", d.get("description"), "api")
            prof.set_field("avatar_url", d.get("avatar"), "api")
            prof.set_field("location", d.get("location"), "api")
            if d.get("createDate"):
                prof.set_field("joined_at", str(d["createDate"])[:10], "api")
            if d.get("numFollowers") is not None:
                prof.set_field("followers", int(d["numFollowers"]), "api")
            if d.get("numFollowing") is not None:
                prof.set_field("following", int(d["numFollowing"]), "api")
            if d.get("numStoriesPublished") is not None:
                prof.set_field("post_count", {"stories": int(d["numStoriesPublished"])}, "api")
            if isinstance(d.get("website"), str) and d["website"].startswith("http"):
                prof.set_field("links", [d["website"]], "api")
        rs = await client.get(f"https://www.wattpad.com/api/v3/users/{username}/stories", headers={"User-Agent": DEFAULT_UA}, timeout=8.0)
        if rs.status_code == 200:
            stories = rs.json().get("stories") or []
            if stories:
                prof.extra["stories"] = [
                    f"{s.get('title', '')} — {s.get('description', '')}".strip(" —")
                    if isinstance(s, dict)
                    else str(s)
                    for s in stories[:10]
                ]
                if not prof.post_count:
                    prof.set_field("post_count", {"stories": len(stories)}, "api")
    except Exception:
        pass
    return prof


@register_extractor("kaggle")
async def extract_kaggle(url: str, username: str, client: httpx.AsyncClient) -> NormalizedProfile:
    prof = NormalizedProfile(username=username)
    try:
        r = await client.get(url, headers={"User-Agent": DEFAULT_UA}, follow_redirects=True, timeout=8.0)
        if r.status_code == 200:
            html = r.text[:250_000]
            og_t = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
            og_d = re.search(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
            og_i = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
            if og_t:
                prof.set_field("display_name", unescape(og_t.group(1)).split("|")[0].strip(), "structured")
            if og_d:
                prof.set_field("bio", unescape(og_d.group(1))[:400], "structured")
            if og_i:
                prof.set_field("avatar_url", og_i.group(1), "structured")
            st_m = re.search(r'Kaggle\.State\.push\((\{.*?\})\)', html, re.DOTALL)
            if st_m:
                try:
                    sd = json.loads(st_m.group(1))
                    if sd.get("displayName"):
                        prof.set_field("display_name", sd["displayName"], "structured")
                    if sd.get("bio"):
                        prof.set_field("bio", sd["bio"][:400], "structured")
                    if sd.get("city") or sd.get("country"):
                        prof.set_field("location", ", ".join(x for x in (sd.get("city"), sd.get("country")) if x), "structured")
                    if sd.get("userAvatarUrl"):
                        prof.set_field("avatar_url", sd["userAvatarUrl"], "structured")
                    if sd.get("userJoinDate"):
                        prof.set_field("joined_at", str(sd["userJoinDate"])[:10], "structured")
                    if sd.get("followersCount") is not None:
                        prof.set_field("followers", int(sd["followersCount"]), "structured")
                except Exception:
                    pass
    except Exception:
        pass
    return prof


@register_extractor("itchio")
async def extract_itchio(url: str, username: str, client: httpx.AsyncClient) -> NormalizedProfile:
    prof = NormalizedProfile(username=username)
    try:
        r = await client.get(url, headers={"User-Agent": DEFAULT_UA}, follow_redirects=True, timeout=8.0)
        if r.status_code == 200:
            html = r.text[:250_000]
            og_t = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
            og_d = re.search(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
            og_i = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
            if og_t:
                prof.set_field("display_name", unescape(og_t.group(1)).strip(), "structured")
            if og_d:
                prof.set_field("bio", unescape(og_d.group(1))[:400], "structured")
            if og_i:
                prof.set_field("avatar_url", og_i.group(1), "structured")
            games = re.findall(r'class="game_link"[^>]*href="([^"]+)"', html, re.I)
            if games:
                prof.set_field("post_count", {"games": len(set(games))}, "structured")
                prof.set_field("links", list(dict.fromkeys(games))[:6], "structured")
    except Exception:
        pass
    return prof


def ml_text_fallback_extract(text: str, base_url: str = "") -> NormalizedProfile:
    """Tier 4 ML/heuristic schema extractor for unknown public pages (`source='ml'`)."""
    prof = NormalizedProfile()
    clean = " ".join((text or "").split())[:2500]
    if not clean:
        return prof

    joined_m = re.search(r"(?:joined|member since|since)\s+([A-Za-z]+\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{4})", clean, re.I)
    if joined_m:
        prof.set_field("joined_at", joined_m.group(1), "ml")
    fol_m = re.search(r"([\d,.]+[KMB]?)\s+followers", clean, re.I)
    if fol_m:
        prof.set_field("followers", fol_m.group(1), "ml")
    loc_m = re.search(r"(?:location|based in|from)\s*[:\-]?\s*([A-Z][a-zA-Z\s,]{2,40})", clean)
    if loc_m:
        prof.set_field("location", loc_m.group(1).strip(" ,."), "ml")
    if len(clean) >= 24:
        prof.set_field("bio", clean[:280], "ml")
    return prof


async def extract_dom_fallback(url: str) -> NormalizedProfile:
    """Tier 3 Playwright DOM extractor for public JS-only pages (respects SSRF guard)."""
    prof = NormalizedProfile()
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host or is_private_or_loopback_host(host):
        return prof
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(user_agent=DEFAULT_UA)
            await page.goto(url, wait_until="domcontentloaded", timeout=8000)
            title = await page.title()
            text = await page.inner_text("body")
            await browser.close()
        if title:
            prof.set_field("display_name", title.strip()[:120], "dom")
        if text:
            prof.set_field("bio", " ".join(text.split())[:400], "dom")
    except Exception:
        pass
    return prof
