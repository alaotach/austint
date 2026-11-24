"""Collector tools registry and implementations for public data sources."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import httpx

from osint_agent.schema import Entity, EntityType, Hit

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = os.getenv(
    "OSINT_USER_AGENT",
    "IntelligentOSINT-Agent/1.0 (Educational Academic Project; contact: agent@osint-project.local)",
)

TOOL_REGISTRY: dict[str, ToolDefinition] = {}
_nominatim_lock = asyncio.Lock()
_last_nominatim_call = 0.0
_audit_callback: Callable[[str, dict[str, Any]], None] | None = None


def set_audit_callback(callback: Callable[[str, dict[str, Any]], None] | None) -> None:
    """Register active audit callback for collector events."""
    global _audit_callback
    _audit_callback = callback


def log_audit_event(event_type: str, details: dict[str, Any]) -> None:
    """Log an audit event via registered callback or fallback JSONL."""
    if _audit_callback is not None:
        try:
            _audit_callback(event_type, details)
            return
        except Exception as err:
            logger.debug("Failed calling audit callback: %s", err)

    try:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "data": details,
        }
        with open("audit.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


@dataclass
class ToolDefinition:
    """Metadata and execution pointer for a registered OSINT collector."""

    name: str
    description: str
    target_types: list[EntityType]
    func: Callable[[Entity, httpx.AsyncClient], Awaitable[list[Hit]]]

    async def run(self, entity: Entity, client: httpx.AsyncClient) -> list[Hit]:
        """Execute the collector safely with logging."""
        logger.info(
            "Executing collector %s for entity %s (%s)",
            self.name,
            entity.id,
            entity.entity_type.value,
        )
        return await self.func(entity, client)


def register_tool(
    name: str,
    target_types: list[EntityType],
    description: str,
):
    """Decorator to register a new collector tool into TOOL_REGISTRY."""

    def decorator(fn: Callable[[Entity, httpx.AsyncClient], Awaitable[list[Hit]]]):
        tool_def = ToolDefinition(
            name=name,
            description=description,
            target_types=target_types,
            func=fn,
        )
        TOOL_REGISTRY[name] = tool_def
        return fn

    return decorator


def get_tools_for_entity(entity_type: EntityType) -> list[ToolDefinition]:
    """Retrieve all collectors capable of processing a given entity type."""
    return [tool for tool in TOOL_REGISTRY.values() if entity_type in tool.target_types]


# Alias for planner compatibility
get_tools_for_type = get_tools_for_entity


@register_tool(
    name="crtsh",
    target_types=[EntityType.DOMAIN],
    description="Queries crt.sh Certificate Transparency logs (with HackerTarget fallback) for public subdomains.",
)
async def collect_crtsh(entity: Entity, client: httpx.AsyncClient) -> list[Hit]:
    """Extract public subdomains from Certificate Transparency logs with fast DNS fallback."""
    domain = entity.value.strip().lower()
    domain = re.sub(r"^https?://", "", domain)
    domain = re.sub(r"^\*\.", "", domain).split("/")[0]

    url = f"https://crt.sh/?q=%.{domain}&output=json"
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}

    hits: list[Hit] = []
    seen_values: set[str] = set()

    # Primary source: crt.sh Certificate Transparency
    try:
        response = await client.get(url, headers=headers, timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                for item in data:
                    name_value = item.get("name_value") or item.get("common_name")
                    if not name_value or not isinstance(name_value, str):
                        continue

                    for raw_name in name_value.split("\n"):
                        clean_name = raw_name.strip().lower()
                        clean_name = re.sub(r"^\*\.", "", clean_name)

                        if not clean_name or clean_name == domain:
                            continue
                        if not (clean_name.endswith(f".{domain}") or clean_name == domain):
                            continue
                        if clean_name in seen_values:
                            continue

                        seen_values.add(clean_name)
                        hits.append(
                            Hit(
                                target_entity_id=entity.id,
                                discovered_type=EntityType.DOMAIN,
                                discovered_value=clean_name,
                                relation="subdomain_of",
                                source_tool="crtsh",
                                is_trusted=True,
                                raw_data={
                                    "issuer_name": item.get("issuer_name"),
                                    "entry_timestamp": item.get("entry_timestamp"),
                                    "logged_name": raw_name,
                                },
                                metadata={"parent_domain": domain},
                            )
                        )
        else:
            logger.warning("crt.sh returned status %d for domain %s; trying fallback.", response.status_code, domain)
    except (httpx.HTTPError, httpx.TimeoutException, ValueError) as exc:
        logger.warning("crt.sh failed for %s (%s); trying fallback.", domain, exc)

    # Fallback source: HackerTarget HostSearch (fast, keyless public DNS records)
    if not hits:
        fallback_url = f"https://api.hackertarget.com/hostsearch/?q={domain}"
        try:
            resp = await client.get(fallback_url, headers={"User-Agent": DEFAULT_USER_AGENT}, timeout=8.0)
            if resp.status_code == 200 and "error" not in resp.text.lower()[:30] and "api count" not in resp.text.lower():
                for line in resp.text.splitlines():
                    parts = line.strip().split(",")
                    if not parts or not parts[0]:
                        continue
                    clean_name = parts[0].strip().lower()
                    ip_addr = parts[1].strip() if len(parts) > 1 else None
                    if not clean_name or clean_name == domain:
                        continue
                    if not clean_name.endswith(f".{domain}"):
                        continue
                    if clean_name in seen_values:
                        continue

                    seen_values.add(clean_name)
                    hits.append(
                        Hit(
                            target_entity_id=entity.id,
                            discovered_type=EntityType.DOMAIN,
                            discovered_value=clean_name,
                            relation="subdomain_of",
                            source_tool="crtsh",
                            is_trusted=True,
                            raw_data={
                                "source": "hackertarget_fallback",
                                "ip": ip_addr,
                            },
                            metadata={"parent_domain": domain},
                        )
                    )
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            logger.warning("HackerTarget fallback also failed for %s: %s", domain, exc)

    return hits


@register_tool(
    name="domain_intel",
    target_types=[EntityType.DOMAIN],
    description="Gathers RDAP registrar/nameservers, DNS A-records, HTTP server headers, and page metadata for domains.",
)
async def collect_domain_intel(entity: Entity, client: httpx.AsyncClient) -> list[Hit]:
    """Gather infrastructure, RDAP registration, hosting, and homepage metadata for a domain."""
    # Avoid wasting budget on nameserver domains
    if entity.depth > 0 and any("nameserver_for" in p for p in entity.provenance):
        return []

    domain = entity.value.strip().lower()
    domain = re.sub(r"^https?://", "", domain)
    domain = re.sub(r"^\*\.", "", domain).split("/")[0]

    hits: list[Hit] = []
    seen_values: set[str] = set()

    # 1. Resolve DNS A-records (IPv4) non-blockingly
    ips: list[str] = []
    try:
        import socket
        _, _, ips = await asyncio.to_thread(socket.gethostbyname_ex, domain)
    except Exception:
        pass

    # 2. Query RDAP for registrar, nameservers, and registration dates (root domains only)
    reg_date = ""
    exp_date = ""
    if domain.count(".") == 1:
        rdap_url = f"https://rdap.org/domain/{domain}"
        try:
            rdap_resp = await client.get(
                rdap_url,
                headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/rdap+json, application/json"},
                follow_redirects=True,
                timeout=8.0,
            )
            if rdap_resp.status_code == 200:
                rdap_data = rdap_resp.json()
                for ev in rdap_data.get("events", []):
                    action = ev.get("eventAction", "")
                    if action == "registration":
                        reg_date = str(ev.get("eventDate", ""))[:10]
                    elif action == "expiration":
                        exp_date = str(ev.get("eventDate", ""))[:10]

                # Registrar entity
                for ent in rdap_data.get("entities", []):
                    if "registrar" in ent.get("roles", []):
                        vcard = ent.get("vcardArray", [])
                        if len(vcard) >= 2 and isinstance(vcard[1], list):
                            for prop in vcard[1]:
                                if isinstance(prop, list) and len(prop) >= 4 and prop[0] == "fn":
                                    reg_name = str(prop[3]).strip()
                                    if reg_name and reg_name.lower() not in seen_values:
                                        seen_values.add(reg_name.lower())
                                        hits.append(
                                            Hit(
                                                target_entity_id=entity.id,
                                                discovered_type=EntityType.ORG,
                                                discovered_value=reg_name,
                                                relation="registered_with",
                                                source_tool="domain_intel",
                                                is_trusted=True,
                                                raw_data={"registered": reg_date, "expires": exp_date},
                                                metadata={"domain": domain},
                                            )
                                        )

                # Nameservers
                for ns in rdap_data.get("nameservers", [])[:4]:
                    ns_name = str(ns.get("ldhName", "")).strip().lower()
                    if ns_name and ns_name not in seen_values:
                        seen_values.add(ns_name)
                        hits.append(
                            Hit(
                                target_entity_id=entity.id,
                                discovered_type=EntityType.DOMAIN,
                                discovered_value=ns_name,
                                relation="nameserver_for",
                                source_tool="domain_intel",
                                is_trusted=True,
                                raw_data={"parent_domain": domain},
                                metadata={"is_nameserver": True},
                            )
                        )
        except (httpx.HTTPError, httpx.TimeoutException, ValueError) as exc:
            logger.debug("RDAP query skipped/failed for %s: %s", domain, exc)

    # 3. Live HTTP inspection for hosting/CDN headers and homepage metadata
    web_url = f"https://{domain}"
    try:
        from osint_agent.enrich import parse_html_metadata

        web_resp = await client.get(
            web_url,
            headers={"User-Agent": DEFAULT_USER_AGENT},
            follow_redirects=True,
            timeout=6.0,
        )
        server_hdr = web_resp.headers.get("server", "").strip()
        powered_by = web_resp.headers.get("x-powered-by", "").strip()
        cache_status = web_resp.headers.get("cache-status", "").strip()

        providers: list[str] = []
        if server_hdr:
            providers.append(server_hdr)
        if powered_by:
            providers.append(powered_by)
        if "netlify" in cache_status.lower():
            providers.append("Netlify Edge")
        elif web_resp.headers.get("x-vercel-id"):
            providers.append("Vercel")

        for prov in providers:
            clean_prov = prov.strip()
            if clean_prov and clean_prov.lower() not in seen_values:
                seen_values.add(clean_prov.lower())
                hits.append(
                    Hit(
                        target_entity_id=entity.id,
                        discovered_type=EntityType.ORG,
                        discovered_value=clean_prov,
                        relation="hosted_on",
                        source_tool="domain_intel",
                        is_trusted=True,
                        raw_data={"ips": ips, "http_status": web_resp.status_code},
                        metadata={"domain": domain},
                    )
                )

        # Parse homepage HTML metadata for lead extraction
        content_type = web_resp.headers.get("content-type", "").lower()
        if "text/html" in content_type and web_resp.text:
            meta_dict = parse_html_metadata(web_resp.text[:200_000])
            summary_parts = []
            if isinstance(meta_dict, dict):
                if meta_dict.get("title"):
                    summary_parts.append(f"Title: {meta_dict['title']}")
                if meta_dict.get("description"):
                    summary_parts.append(meta_dict["description"])
            elif isinstance(meta_dict, str) and meta_dict:
                summary_parts.append(meta_dict)
            if ips:
                summary_parts.append(f"IPs: {', '.join(ips)}")
            if reg_date:
                summary_parts.append(f"Registered: {reg_date}")
            if exp_date:
                summary_parts.append(f"Expires: {exp_date}")

            hits.append(
                Hit(
                    target_entity_id=entity.id,
                    discovered_type=EntityType.PROFILE,
                    discovered_value=str(web_resp.url),
                    relation="web_presence",
                    source_tool="domain_intel",
                    is_trusted=True,
                    raw_data={
                        "description": " | ".join(summary_parts),
                        "ips": ips,
                        "registered": reg_date,
                        "expires": exp_date,
                        "server": server_hdr,
                    },
                    metadata={"domain": domain, "url": str(web_resp.url)},
                )
            )
    except (httpx.HTTPError, httpx.TimeoutException, ValueError) as exc:
        logger.debug("HTTP metadata inspection skipped/failed for %s: %s", domain, exc)

    return hits


@register_tool(
    name="nominatim",
    target_types=[EntityType.PLACE],
    description="Queries OpenStreetMap Nominatim for geocoding and parent administrative divisions.",
)
async def collect_nominatim(entity: Entity, client: httpx.AsyncClient) -> list[Hit]:
    """Query OpenStreetMap Nominatim geocoder respecting polite rate limits."""
    global _last_nominatim_call
    query = entity.value.strip()
    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}
    params = {
        "q": query,
        "format": "jsonv2",
        "addressdetails": "1",
        "limit": "5",
    }

    hits: list[Hit] = []
    seen: set[str] = set()

    async with _nominatim_lock:
        now = asyncio.get_event_loop().time()
        delay = 1.0 - (now - _last_nominatim_call)
        if delay > 0:
            await asyncio.sleep(delay)
        _last_nominatim_call = asyncio.get_event_loop().time()

        try:
            response = await client.get(url, params=params, headers=headers, timeout=12.0)
            if response.status_code != 200:
                logger.warning("Nominatim returned status %d for query %s", response.status_code, query)
                return []
            results = response.json()
        except (httpx.HTTPError, httpx.TimeoutException, ValueError) as exc:
            logger.error("Error querying Nominatim for %s: %s", query, exc)
            return []

    if not isinstance(results, list):
        return []

    for item in results:
        display_name = item.get("display_name")
        if not display_name or not isinstance(display_name, str):
            continue

        clean_display = display_name.strip()
        if clean_display and clean_display.lower() not in seen:
            seen.add(clean_display.lower())
            hits.append(
                Hit(
                    target_entity_id=entity.id,
                    discovered_type=EntityType.PLACE,
                    discovered_value=clean_display,
                    relation="geographic_match",
                    source_tool="nominatim",
                    is_trusted=False,
                    raw_data={
                        "osm_type": item.get("osm_type"),
                        "category": item.get("category"),
                        "place_rank": item.get("place_rank"),
                        "lat": item.get("lat"),
                        "lon": item.get("lon"),
                    },
                    metadata={"query": query},
                )
            )

        address = item.get("address", {})
        if isinstance(address, dict):
            for rel_type in ("state", "country"):
                parent_val = address.get(rel_type)
                if parent_val and isinstance(parent_val, str) and parent_val.lower() not in seen:
                    seen.add(parent_val.lower())
                    hits.append(
                        Hit(
                            target_entity_id=entity.id,
                            discovered_type=EntityType.PLACE,
                            discovered_value=parent_val.strip(),
                            relation=f"located_in_{rel_type}",
                            source_tool="nominatim",
                            is_trusted=False,
                            raw_data={"level": rel_type},
                            metadata={"parent_of": query},
                        )
                    )

    return hits




@register_tool(
    name="wikidata",
    target_types=[
        EntityType.ORG,
        EntityType.PUBLIC_FIGURE,
        EntityType.CONSENTING_PERSON,
        EntityType.PLACE,
    ],
    description="Queries Wikidata entity search API for public organizations, figures, and places.",
)
async def collect_wikidata(entity: Entity, client: httpx.AsyncClient) -> list[Hit]:
    """Query Wikidata entity search API for matching public knowledge entities."""
    query = entity.value.strip()
    url = "https://www.wikidata.org/w/api.php"
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}
    params = {
        "action": "wbsearchentities",
        "search": query,
        "language": "en",
        "format": "json",
        "limit": "5",
        "type": "item",
    }

    hits: list[Hit] = []
    seen: set[str] = set()

    try:
        response = await client.get(url, params=params, headers=headers, timeout=10.0)
        if response.status_code != 200:
            logger.warning("Wikidata returned status %d for query %s", response.status_code, query)
            return []
        data = response.json()
    except (httpx.HTTPError, httpx.TimeoutException, ValueError) as exc:
        logger.error("Error querying Wikidata for %s: %s", query, exc)
        return []

    search_results = data.get("search", [])
    if not isinstance(search_results, list):
        return []

    for item in search_results:
        label = item.get("label")
        if not label or not isinstance(label, str):
            continue

        clean_label = label.strip()
        q_id = item.get("id", "")
        desc = item.get("description", "")
        key = f"{clean_label.lower()}:{q_id}"
        if key in seen:
            continue
        seen.add(key)

        hits.append(
            Hit(
                target_entity_id=entity.id,
                discovered_type=entity.entity_type,
                discovered_value=clean_label,
                relation="wikidata_reference",
                source_tool="wikidata",
                is_trusted=False,
                raw_data={
                    "wikidata_id": q_id,
                    "description": desc,
                    "concepturi": item.get("concepturi"),
                },
                metadata={"query": query},
            )
        )

    return hits




def parse_sherlock_output(stdout: str, username: str, target_id: str) -> list[Hit]:
    """Parse standard stdout lines from sherlock execution into normalized Hits."""
    hits: list[Hit] = []
    pattern = re.compile(r"^\[\+\]\s*([^:]+):\s*(https?://\S+)", re.MULTILINE)
    for match in pattern.finditer(stdout):
        site_name = match.group(1).strip()
        url = match.group(2).strip()
        hits.append(
            Hit(
                target_entity_id=target_id,
                discovered_type=EntityType.PROFILE,
                discovered_value=url,
                relation="account_on",
                source_tool="sherlock",
                is_trusted=False,
                raw_data={"site": site_name, "username": username},
                metadata={"username": username, "site": site_name, "url": url, "name": username},
            )
        )
    return hits




def parse_maigret_json(data: Any, username: str, target_id: str) -> list[Hit]:
    """Parse structured JSON report from maigret execution into normalized Hits."""
    hits: list[Hit] = []
    items: list[dict[str, Any]] = []

    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                item = dict(v)
                item["_site_key"] = k
                items.append(item)
    elif isinstance(data, list):
        items = [x for x in data if isinstance(x, dict)]

    for item in items:
        raw_status = item.get("status")
        if isinstance(raw_status, dict):
            status = str(raw_status.get("status", "")).lower()
            ids = raw_status.get("ids", {}) if isinstance(raw_status.get("ids"), dict) else {}
        else:
            status = str(raw_status or "").lower()
            ids = item.get("ids", {}) if isinstance(item.get("ids"), dict) else {}

        if "not found" in status or "error" in status:
            continue
        if "claimed" not in status and "found" not in status and status:
            continue

        url = item.get("url_user") or item.get("url")
        if not url and isinstance(raw_status, dict):
            url = raw_status.get("url")
        if not url or not isinstance(url, str):
            continue

        site = item.get("_site_key")
        if not site and isinstance(raw_status, dict):
            site = raw_status.get("site_name")
        if not site and isinstance(item.get("site"), str):
            site = item.get("site")
        site = str(site or "unknown")

        parsed_info = item.get("parsed_info", {}) if isinstance(item.get("parsed_info"), dict) else {}
        bio = ids.get("bio") or ids.get("company") or parsed_info.get("bio") or item.get("bio", "")
        location = ids.get("location") or parsed_info.get("location") or item.get("location", "")
        name = ids.get("fullname") or ids.get("name") or parsed_info.get("name") or item.get("name", username)

        hits.append(
            Hit(
                target_entity_id=target_id,
                discovered_type=EntityType.PROFILE,
                discovered_value=url,
                relation="account_on",
                source_tool="maigret",
                is_trusted=False,
                raw_data={
                    "site": site,
                    "description": bio,
                    "bio": bio,
                    "location": location,
                    "name": name,
                    "username": username,
                },
                metadata={
                    "username": username,
                    "site": site,
                    "url": url,
                    "bio": bio,
                    "location": location,
                    "name": name,
                },
            )
        )
    return hits




@register_tool(
    name="sherlock",
    target_types=[EntityType.USERNAME, EntityType.PUBLIC_FIGURE, EntityType.CONSENTING_PERSON],
    description="Searches social accounts across hundreds of sites via Sherlock CLI subprocess.",
)
async def collect_sherlock(entity: Entity, client: httpx.AsyncClient) -> list[Hit]:
    """Execute sherlock CLI in isolated temp directory to discover public profile accounts."""
    binary = shutil.which("sherlock") or shutil.which("sherlock", path=os.path.join(sys.prefix, "bin"))
    if not binary:
        logger.warning("sherlock binary is not installed on PATH.")
        log_audit_event("tool_unavailable", {"tool": "sherlock", "reason": "binary not found on PATH"})
        return []

    username = entity.value.strip().lstrip("@")
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            proc = await asyncio.create_subprocess_exec(
                binary,
                username,
                "--print-found",
                "--no-color",
                "--timeout",
                "5",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmpdir,
            )
            try:
                stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=45.0)
            except asyncio.TimeoutError:
                proc.terminate()
                stdout_bytes, _ = await proc.communicate()
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            return parse_sherlock_output(stdout, username, entity.id)
        except Exception as exc:
            logger.warning("Sherlock execution error for %s: %s", username, exc)
            return []


@register_tool(
    name="maigret",
    target_types=[EntityType.USERNAME, EntityType.PUBLIC_FIGURE, EntityType.CONSENTING_PERSON],
    description="Extracts detailed social profiles and metadata via Maigret CLI subprocess.",
)
async def collect_maigret(entity: Entity, client: httpx.AsyncClient) -> list[Hit]:
    """Execute maigret CLI in isolated temp directory to extract rich profile dossiers."""
    binary = shutil.which("maigret") or shutil.which("maigret", path=os.path.join(sys.prefix, "bin"))
    if not binary:
        logger.warning("maigret binary is not installed on PATH.")
        log_audit_event("tool_unavailable", {"tool": "maigret", "reason": "binary not found on PATH"})
        return []

    username = entity.value.strip().lstrip("@")
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            proc = await asyncio.create_subprocess_exec(
                binary,
                username,
                "--top-sites",
                "1500",
                "-n",
                "250",
                "--no-autoupdate",
                "--json",
                "simple",
                "--timeout",
                "4",
                "--no-progressbar",
                "--no-color",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmpdir,
            )
            try:
                stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=60.0)
            except asyncio.TimeoutError:
                proc.terminate()
                stdout_bytes, _ = await proc.communicate()

            # Discover all generated JSON reports in tmpdir or tmpdir/reports
            all_hits: list[Hit] = []
            seen_urls: set[str] = set()
            for root, _, files in os.walk(tmpdir):
                for f in files:
                    if f.endswith(".json"):
                        json_path = os.path.join(root, f)
                        try:
                            with open(json_path, "r", encoding="utf-8") as fp:
                                data = json.load(fp)
                            for h in parse_maigret_json(data, username, entity.id):
                                if h.discovered_value not in seen_urls:
                                    seen_urls.add(h.discovered_value)
                                    all_hits.append(h)
                        except Exception:
                            pass

            if all_hits:
                return all_hits

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            try:
                data = json.loads(stdout)
                return parse_maigret_json(data, username, entity.id)
            except Exception:
                return parse_sherlock_output(stdout, username, entity.id)
        except Exception as exc:
            logger.warning("Maigret execution error for %s: %s", username, exc)
            return []
