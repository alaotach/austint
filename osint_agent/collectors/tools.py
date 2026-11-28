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
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

import httpx

from osint_agent.core.schema import Entity, EntityType, Hit

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
    from osint_agent.core.audit import write_audit_event

    if _audit_callback is not None:
        try:
            _audit_callback(event_type, details)
            return
        except Exception as err:
            logger.debug("Failed calling audit callback: %s", err)

    write_audit_event(event_type, details)


def _normalize_domain(value: str) -> str:
    """Strip URL scheme, wildcards, and path fragments from a domain string."""
    domain = value.strip().lower()
    domain = re.sub(r"^https?://", "", domain)
    return re.sub(r"^\*\.", "", domain).split("/")[0]


def _find_cli_binary(name: str) -> str | None:
    """Locate an external CLI binary on PATH or inside the active Python environment."""
    return shutil.which(name) or shutil.which(name, path=os.path.join(sys.prefix, "bin"))


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
    domain = _normalize_domain(entity.value)

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

    domain = _normalize_domain(entity.value)

    hits: list[Hit] = []
    seen_values: set[str] = set()

    # 1. Resolve DNS A, MX, NS, and TXT records via dnspython (with socket fallback)
    ips: list[str] = []
    mx_hosts: list[str] = []
    txt_records: list[str] = []
    try:
        import dns.resolver  # type: ignore

        def _query_dns_sync() -> tuple[list[str], list[str], list[str]]:
            a_list, mx_list, txt_list = [], [], []
            res = dns.resolver.Resolver()
            res.lifetime = 3.0
            try:
                a_list = [str(r) for r in res.resolve(domain, "A")]
            except Exception:
                pass
            try:
                mx_list = [str(r.exchange).rstrip(".").lower() for r in res.resolve(domain, "MX")]
            except Exception:
                pass
            try:
                txt_list = [str(r).strip('"') for r in res.resolve(domain, "TXT")]
            except Exception:
                pass
            return a_list, mx_list, txt_list

        ips, mx_hosts, txt_records = await asyncio.to_thread(_query_dns_sync)
    except Exception:
        pass

    if not ips:
        try:
            import socket
            _, _, ips = await asyncio.to_thread(socket.gethostbyname_ex, domain)
        except Exception:
            pass

    for mx in mx_hosts[:3]:
        if mx and mx not in seen_values:
            seen_values.add(mx)
            hits.append(
                Hit(
                    target_entity_id=entity.id,
                    discovered_type=EntityType.DOMAIN,
                    discovered_value=mx,
                    relation="mail_server_for",
                    source_tool="domain_intel",
                    is_trusted=True,
                    raw_data={"parent_domain": domain},
                    metadata={"is_nameserver": True},
                )
            )

    # 1b. Query Shodan InternetDB (free, keyless) for open ports & CVEs on first IP
    shodan_ports: list[int] = []
    shodan_vulns: list[str] = []
    if ips and domain.count(".") == 1:
        try:
            shodan_resp = await client.get(
                f"https://internetdb.shodan.io/{ips[0]}",
                headers={"User-Agent": DEFAULT_USER_AGENT},
                timeout=3.0,
            )
            if shodan_resp.status_code == 200:
                sdata = shodan_resp.json()
                shodan_ports = list(sdata.get("ports") or [])
                shodan_vulns = list(sdata.get("vulns") or [])
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
        from osint_agent.collectors.enrich import parse_html_metadata

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

        # Parse homepage HTML metadata & tech stack fingerprints
        content_type = web_resp.headers.get("content-type", "").lower()
        tech_stack: list[str] = []
        if "text/html" in content_type and web_resp.text:
            html_head = web_resp.text[:200_000]
            html_lower = html_head.lower()
            tech_signatures = [
                ("Next.js", "/_next/static"),
                ("React", "data-reactroot"),
                ("Vite", "/assets/index-"),
                ("WordPress", "wp-content/"),
                ("Tailwind CSS", "tailwindcss"),
                ("Google Analytics", "googletagmanager.com/gtag"),
                ("Stripe", "js.stripe.com"),
                ("Cloudflare Turnstile", "challenges.cloudflare.com/turnstile"),
            ]
            for tech_name, marker in tech_signatures:
                if marker in html_lower:
                    tech_stack.append(tech_name)

            meta_dict = parse_html_metadata(html_head)
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
            if shodan_ports:
                summary_parts.append(f"Open Ports: {', '.join(str(p) for p in shodan_ports)}")
            if shodan_vulns:
                summary_parts.append(f"CVEs: {', '.join(shodan_vulns[:5])}")
            if tech_stack:
                summary_parts.append(f"Tech: {', '.join(tech_stack)}")
            if txt_records:
                summary_parts.append(f"TXT: {' ; '.join(txt_records[:3])}")
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
                        "ports": shodan_ports,
                        "vulns": shodan_vulns,
                        "tech_stack": tech_stack,
                        "txt_records": txt_records,
                        "registered": reg_date,
                        "expires": exp_date,
                        "server": server_hdr,
                    },
                    metadata={
                        "domain": domain,
                        "url": str(web_resp.url),
                        "ips": ", ".join(ips),
                        "tech_stack": ", ".join(tech_stack),
                        "ports": ", ".join(str(p) for p in shodan_ports),
                    },
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
    binary = _find_cli_binary("sherlock")
    if not binary:
        logger.warning("sherlock binary is not installed on PATH.")
        log_audit_event("tool_unavailable", {"tool": "sherlock", "reason": "binary not found on PATH"})
        return []

    username = entity.value.strip().lstrip("@")
    if not username or " " in username or not re.fullmatch(r"[A-Za-z0-9._-]{2,64}", username):
        logger.debug("Skipping sherlock for non-username value: %s", entity.value)
        return []
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
            raw_hits = parse_sherlock_output(stdout, username, entity.id)
            if not raw_hits:
                return []

            # Filter out catch-all sites that return HTTP 200 even for a nonexistent username
            from osint_agent.collectors.stream_checker import (
                _BASELINE_PROBE_USERNAME,
                _is_bot_or_soft404_response,
                load_false_positive_hosts,
            )

            fp_hosts = load_false_positive_hosts()

            async def _verify_sherlock_hit(hit: Hit) -> Hit | None:
                url = hit.discovered_value
                host = urlparse(url).netloc.lower()
                if host in fp_hosts:
                    return None
                if username.lower() not in url.lower():
                    return hit
                probe_url = re.sub(re.escape(username), _BASELINE_PROBE_USERNAME, url, count=1, flags=re.IGNORECASE)
                try:
                    probe_resp, real_resp = await asyncio.gather(
                        client.get(probe_url, timeout=4.5, follow_redirects=True),
                        client.get(url, timeout=4.5, follow_redirects=True),
                    )
                    # Real URL must be 200 (not 202 WAF, 403, 404) and not a bot/soft-404/redirect
                    if real_resp.status_code != 200:
                        return None
                    real_body = real_resp.text
                    if _is_bot_or_soft404_response(real_resp, real_body, "status_code", probe_url=url):
                        return None
                    # Baseline URL returning 200 without being caught as soft-404 means catch-all site
                    if probe_resp.status_code == 200:
                        probe_body = probe_resp.text
                        if not _is_bot_or_soft404_response(probe_resp, probe_body, "status_code", probe_url=probe_url):
                            return None
                        # Same title & near-identical body length -> custom soft-404 page
                        t1 = re.search(r"<title[^>]*>(.*?)</title>", real_body[:20_000], re.I | re.S)
                        t2 = re.search(r"<title[^>]*>(.*?)</title>", probe_body[:20_000], re.I | re.S)
                        if t1 and t2 and t1.group(1).strip().lower() == t2.group(1).strip().lower():
                            len_diff = abs(len(real_body) - len(probe_body)) / max(1, len(real_body))
                            if len_diff < 0.03:
                                return None
                except Exception:
                    return hit
                return hit

            checked = await asyncio.gather(*(_verify_sherlock_hit(h) for h in raw_hits), return_exceptions=True)
            return [h for h in checked if isinstance(h, Hit)]
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
    binary = _find_cli_binary("maigret")
    if not binary:
        logger.warning("maigret binary is not installed on PATH.")
        log_audit_event("tool_unavailable", {"tool": "maigret", "reason": "binary not found on PATH"})
        return []

    username = entity.value.strip().lstrip("@")
    if not username or " " in username or not re.fullmatch(r"[A-Za-z0-9._-]{2,64}", username):
        logger.debug("Skipping maigret for non-username value: %s", entity.value)
        return []

    # Fast native streaming checker using Maigret's local site rules
    try:
        from osint_agent.collectors.stream_checker import load_maigret_sites, stream_username_check

        if load_maigret_sites():
            def _on_live_hit(hit: Hit) -> None:
                log_audit_event(
                    "profile_discovered",
                    {
                        "site": hit.metadata.get("site", "unknown"),
                        "url": hit.discovered_value,
                        "username": username,
                    },
                )

            return await stream_username_check(
                username=username,
                target_id=entity.id,
                max_sites=11000,
                concurrency=200,
                overall_timeout=90.0,
                on_hit=_on_live_hit,
            )
    except Exception as exc:
        logger.warning("Native streaming checker failed (%s); falling back to Maigret CLI.", exc)

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


FREE_EMAIL_PROVIDERS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "live.com",
    "icloud.com", "me.com", "proton.me", "protonmail.com", "aol.com",
    "zoho.com", "yandex.com", "mail.com", "gmx.com", "tutanota.com",
}


def parse_holehe_output(stdout: str, email: str, target_id: str) -> list[Hit]:
    """Parse Holehe --only-used stdout lines into normalized PROFILE Hits."""
    hits: list[Hit] = []
    seen_sites: set[str] = set()
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("[+]"):
            continue
        if "email used" in stripped.lower():
            continue
        site = stripped[3:].strip().lower()
        if not site or "." not in site or " " in site:
            continue
        if site in seen_sites:
            continue
        seen_sites.add(site)

        url = f"https://{site}"
        hits.append(
            Hit(
                target_entity_id=target_id,
                discovered_type=EntityType.PROFILE,
                discovered_value=url,
                relation="registered_on",
                source_tool="email_intel",
                is_trusted=True,
                raw_data={"site": site, "email": email, "method": "holehe"},
                metadata={"site": site, "url": url, "email": email},
            )
        )
    return hits




@register_tool(
    name="email_intel",
    target_types=[EntityType.EMAIL],
    description=(
        "Checks email intelligence via GitHub commits/users, XposedOrNot breaches, Gravatar, "
        "Duolingo, OpenPGP/Ubuntu keyservers, Holehe, and custom domain linking."
    ),
)
async def collect_email_intel(entity: Entity, client: httpx.AsyncClient) -> list[Hit]:
    """Gather public GitHub activity, breach exposure, Gravatar, PGP keys, and registration leads for an email."""
    import hashlib
    import os
    from urllib.parse import quote, unquote

    email = entity.value.strip().lower()
    if "@" not in email:
        return []

    hits: list[Hit] = []
    seen_values: set[str] = set()

    def _add_hit(hit: Hit) -> None:
        if hit.discovered_value not in seen_values:
            seen_values.add(hit.discovered_value)
            hits.append(hit)

    # 1. Link non-free corporate/custom email domain
    domain = email.split("@", 1)[1].strip()
    if domain and domain not in FREE_EMAIL_PROVIDERS:
        _add_hit(
            Hit(
                target_entity_id=entity.id,
                discovered_type=EntityType.DOMAIN,
                discovered_value=domain,
                relation="email_domain_of",
                source_tool="email_intel",
                is_trusted=True,
                raw_data={"email": email},
                metadata={"domain": domain},
            )
        )

    email_hash = hashlib.md5(email.encode("utf-8")).hexdigest()
    gh_headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/vnd.github+json"}
    gh_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if gh_token:
        gh_headers["Authorization"] = f"Bearer {gh_token}"

    async def _check_gravatar() -> None:
        gravatar_url = f"https://en.gravatar.com/{email_hash}.json"
        found_json = False
        try:
            resp = await client.get(
                gravatar_url,
                headers={"User-Agent": DEFAULT_USER_AGENT},
                follow_redirects=True,
                timeout=6.0,
            )
            if resp.status_code == 200:
                entries = resp.json().get("entry", [])
                if isinstance(entries, list) and entries:
                    found_json = True
                    prof = entries[0]
                    pref_user = prof.get("preferredUsername")
                    if pref_user:
                        _add_hit(
                            Hit(
                                target_entity_id=entity.id,
                                discovered_type=EntityType.USERNAME,
                                discovered_value=pref_user,
                                relation="gravatar_handle",
                                source_tool="email_intel",
                                is_trusted=True,
                                raw_data={
                                    "name": prof.get("displayName"),
                                    "bio": prof.get("aboutMe"),
                                    "location": prof.get("currentLocation"),
                                },
                                metadata={
                                    "name": prof.get("displayName", ""),
                                    "bio": prof.get("aboutMe", ""),
                                    "location": prof.get("currentLocation", ""),
                                },
                            )
                        )
                    for acc in prof.get("accounts", []):
                        acc_url = acc.get("url")
                        if acc_url:
                            _add_hit(
                                Hit(
                                    target_entity_id=entity.id,
                                    discovered_type=EntityType.PROFILE,
                                    discovered_value=acc_url,
                                    relation="account_on",
                                    source_tool="email_intel",
                                    is_trusted=True,
                                    raw_data={"site": acc.get("shortname"), "method": "gravatar"},
                                    metadata={"site": acc.get("shortname", ""), "url": acc_url},
                                )
                            )
        except (httpx.HTTPError, httpx.TimeoutException, ValueError) as exc:
            logger.debug("Gravatar JSON lookup failed for %s: %s", email, exc)

        if not found_json:
            try:
                av_resp = await client.get(
                    f"https://www.gravatar.com/avatar/{email_hash}?d=404",
                    headers={"User-Agent": DEFAULT_USER_AGENT},
                    timeout=5.0,
                )
                if av_resp.status_code == 200:
                    _add_hit(
                        Hit(
                            target_entity_id=entity.id,
                            discovered_type=EntityType.PROFILE,
                            discovered_value=f"https://www.gravatar.com/{email_hash}",
                            relation="registered_on",
                            source_tool="email_intel",
                            is_trusted=True,
                            raw_data={"site": "gravatar", "method": "avatar_check"},
                            metadata={"site": "gravatar", "url": f"https://www.gravatar.com/{email_hash}"},
                        )
                    )
            except (httpx.HTTPError, httpx.TimeoutException):
                pass

    async def _check_github() -> None:
        encoded_email = quote(email, safe="")
        # 1) GitHub commit author email search
        try:
            c_resp = await client.get(
                f"https://api.github.com/search/commits?q=author-email:{encoded_email}&per_page=5",
                headers=gh_headers,
                timeout=8.0,
            )
            if c_resp.status_code == 200:
                cdata = c_resp.json()
                total_commits = int(cdata.get("total_count") or 0)
                for item in cdata.get("items") or []:
                    author = item.get("author") or {}
                    login = author.get("login")
                    commit_author = ((item.get("commit") or {}).get("author") or {})
                    full_name = commit_author.get("name") or ""
                    repo_name = (item.get("repository") or {}).get("full_name") or ""
                    if login:
                        _add_hit(
                            Hit(
                                target_entity_id=entity.id,
                                discovered_type=EntityType.USERNAME,
                                discovered_value=login,
                                relation="github_author_of",
                                source_tool="email_intel",
                                is_trusted=True,
                                raw_data={
                                    "name": full_name,
                                    "email": email,
                                    "commit_count": total_commits,
                                    "sample_repo": repo_name,
                                },
                                metadata={
                                    "username": login,
                                    "name": full_name,
                                    "site": "GitHub",
                                    "commit_count": str(total_commits),
                                    "sample_repo": repo_name,
                                },
                            )
                        )
                        gh_url = f"https://github.com/{login}"
                        _add_hit(
                            Hit(
                                target_entity_id=entity.id,
                                discovered_type=EntityType.PROFILE,
                                discovered_value=gh_url,
                                relation="account_on",
                                source_tool="email_intel",
                                is_trusted=True,
                                raw_data={
                                    "site": "GitHub",
                                    "name": full_name,
                                    "commit_count": total_commits,
                                    "sample_repo": repo_name,
                                },
                                metadata={
                                    "site": "GitHub",
                                    "url": gh_url,
                                    "username": login,
                                    "name": full_name,
                                    "commit_count": str(total_commits),
                                },
                            )
                        )
        except (httpx.HTTPError, httpx.TimeoutException, ValueError) as exc:
            logger.debug("GitHub commit email lookup failed for %s: %s", email, exc)

        # 2) GitHub public user email search
        try:
            u_resp = await client.get(
                f"https://api.github.com/search/users?q={encoded_email}+in:email&per_page=3",
                headers=gh_headers,
                timeout=8.0,
            )
            if u_resp.status_code == 200:
                for user_item in u_resp.json().get("items") or []:
                    login = user_item.get("login")
                    if login:
                        _add_hit(
                            Hit(
                                target_entity_id=entity.id,
                                discovered_type=EntityType.USERNAME,
                                discovered_value=login,
                                relation="github_account_for",
                                source_tool="email_intel",
                                is_trusted=True,
                                raw_data={"site": "GitHub", "email": email},
                                metadata={"username": login, "site": "GitHub"},
                            )
                        )
                        gh_url = user_item.get("html_url") or f"https://github.com/{login}"
                        _add_hit(
                            Hit(
                                target_entity_id=entity.id,
                                discovered_type=EntityType.PROFILE,
                                discovered_value=gh_url,
                                relation="account_on",
                                source_tool="email_intel",
                                is_trusted=True,
                                raw_data={"site": "GitHub", "username": login},
                                metadata={"site": "GitHub", "url": gh_url, "username": login},
                            )
                        )
        except (httpx.HTTPError, httpx.TimeoutException, ValueError) as exc:
            logger.debug("GitHub user email lookup failed for %s: %s", email, exc)

    async def _check_xposedornot() -> None:
        # 1. Basic breach name list
        try:
            resp = await client.get(
                f"https://api.xposedornot.com/v1/check-email/{quote(email, safe='')}",
                headers={"User-Agent": DEFAULT_USER_AGENT},
                timeout=8.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                breach_lists = data.get("breaches") or []
                breach_names: list[str] = []
                if breach_lists and isinstance(breach_lists[0], list):
                    breach_names = [str(b) for b in breach_lists[0] if b]
                elif isinstance(breach_lists, list):
                    breach_names = [str(b) for b in breach_lists if isinstance(b, str)]
                for breach_name in breach_names:
                    _add_hit(
                        Hit(
                            target_entity_id=entity.id,
                            discovered_type=EntityType.PROFILE,
                            discovered_value=f"breach:{breach_name}",
                            relation="breach_exposure",
                            source_tool="email_intel",
                            is_trusted=True,
                            raw_data={"breach": breach_name, "email": email, "source": "xposedornot"},
                            metadata={"site": breach_name, "breach": breach_name, "source": "xposedornot", "email": email},
                        )
                    )
        except (httpx.HTTPError, httpx.TimeoutException, ValueError) as exc:
            logger.debug("XposedOrNot check-email failed for %s: %s", email, exc)

        # 2. Detailed breach analytics: data types exposed, risk score, password strength
        try:
            analytics_resp = await client.get(
                f"https://api.xposedornot.com/v1/breach-analytics?email={quote(email, safe='')}",
                headers={"User-Agent": DEFAULT_USER_AGENT},
                timeout=8.0,
            )
            if analytics_resp.status_code == 200:
                adata = analytics_resp.json()
                metrics = adata.get("BreachMetrics") or {}

                risk_info = (metrics.get("risk") or [{}])[0] if metrics.get("risk") else {}
                risk_label = risk_info.get("risk_label", "")
                risk_score = risk_info.get("risk_score", 0)

                pwd_strength_list = metrics.get("passwords_strength") or [{}]
                pwd_strength = pwd_strength_list[0] if pwd_strength_list else {}
                plaintext_count = int(pwd_strength.get("PlainText") or 0)
                easy_crack_count = int(pwd_strength.get("EasyToCrack") or 0)
                strong_hash_count = int(pwd_strength.get("StrongHash") or 0)

                # Extract exposed data types from xposed_data tree
                exposed_data_types: list[str] = []
                for entry in metrics.get("xposed_data") or []:
                    for child in entry.get("children") or []:
                        for item in child.get("children") or []:
                            if item.get("value", 0) > 0:
                                name_raw = str(item.get("name") or "").replace("data_", "").strip()
                                if name_raw:
                                    exposed_data_types.append(name_raw)

                if risk_label or exposed_data_types:
                    summary_parts = []
                    if risk_label:
                        summary_parts.append(f"Breach Risk: {risk_label} (score {risk_score})")
                    if exposed_data_types:
                        summary_parts.append("Exposed data types: " + ", ".join(exposed_data_types[:8]))
                    if plaintext_count > 0:
                        summary_parts.append(f"⚠ {plaintext_count} plaintext password(s) exposed in breach data")
                    if easy_crack_count > 0:
                        summary_parts.append(f"⚠ {easy_crack_count} easy-to-crack password hash(es) in breach data")
                    if strong_hash_count > 0:
                        summary_parts.append(f"{strong_hash_count} strong hash(es) in breach data")

                    _add_hit(
                        Hit(
                            target_entity_id=entity.id,
                            discovered_type=EntityType.PROFILE,
                            discovered_value=f"breach-analytics:{email}",
                            relation="breach_analytics",
                            source_tool="email_intel",
                            is_trusted=True,
                            raw_data={
                                "source": "xposedornot_analytics",
                                "risk_label": risk_label,
                                "risk_score": risk_score,
                                "exposed_data_types": exposed_data_types,
                                "plaintext_passwords": plaintext_count,
                                "easy_crack_hashes": easy_crack_count,
                                "strong_hashes": strong_hash_count,
                                "description": " | ".join(summary_parts),
                            },
                            metadata={
                                "site": "XposedOrNot Analytics",
                                "risk_label": risk_label,
                                "risk_score": str(risk_score),
                                "exposed_data_types": exposed_data_types,
                                "password_exposure_note": (
                                    f"plaintext:{plaintext_count}, easy_crack:{easy_crack_count}, strong_hash:{strong_hash_count}"
                                ),
                                "bio": " | ".join(summary_parts),
                            },
                        )
                    )
        except (httpx.HTTPError, httpx.TimeoutException, ValueError) as exc:
            logger.debug("XposedOrNot analytics failed for %s: %s", email, exc)

    async def _check_duolingo() -> None:
        try:
            resp = await client.get(
                f"https://www.duolingo.com/2017-06-30/users?email={quote(email, safe='')}",
                headers={"User-Agent": DEFAULT_USER_AGENT},
                timeout=6.0,
            )
            if resp.status_code == 200:
                users = resp.json().get("users") or []
                if isinstance(users, list) and users:
                    user = users[0]
                    uname = user.get("username")
                    if uname:
                        _add_hit(
                            Hit(
                                target_entity_id=entity.id,
                                discovered_type=EntityType.USERNAME,
                                discovered_value=uname,
                                relation="duolingo_handle",
                                source_tool="email_intel",
                                is_trusted=True,
                                raw_data={"site": "Duolingo", "name": user.get("name", ""), "email": email},
                                metadata={"username": uname, "name": user.get("name", ""), "site": "Duolingo"},
                            )
                        )
                        duo_url = f"https://www.duolingo.com/profile/{uname}"
                        _add_hit(
                            Hit(
                                target_entity_id=entity.id,
                                discovered_type=EntityType.PROFILE,
                                discovered_value=duo_url,
                                relation="account_on",
                                source_tool="email_intel",
                                is_trusted=True,
                                raw_data={"site": "Duolingo", "name": user.get("name", "")},
                                metadata={"site": "Duolingo", "url": duo_url, "name": user.get("name", "")},
                            )
                        )
        except (httpx.HTTPError, httpx.TimeoutException, ValueError) as exc:
            logger.debug("Duolingo lookup failed for %s: %s", email, exc)

    async def _check_pgp_keyservers() -> None:
        encoded = quote(email, safe="")
        try:
            resp = await client.get(
                f"https://keys.openpgp.org/vks/v1/by-email/{encoded}",
                headers={"User-Agent": DEFAULT_USER_AGENT},
                timeout=6.0,
            )
            if resp.status_code == 200 and "BEGIN PGP PUBLIC KEY BLOCK" in resp.text:
                pgp_url = f"https://keys.openpgp.org/search?q={encoded}"
                _add_hit(
                    Hit(
                        target_entity_id=entity.id,
                        discovered_type=EntityType.PROFILE,
                        discovered_value=pgp_url,
                        relation="pgp_key_on",
                        source_tool="email_intel",
                        is_trusted=True,
                        raw_data={"site": "keys.openpgp.org", "email": email},
                        metadata={"site": "keys.openpgp.org", "url": pgp_url},
                    )
                )
        except (httpx.HTTPError, httpx.TimeoutException):
            pass

        try:
            u_resp = await client.get(
                f"https://keyserver.ubuntu.com/pks/lookup?op=index&options=mr&search={encoded}",
                headers={"User-Agent": DEFAULT_USER_AGENT},
                timeout=6.0,
            )
            if u_resp.status_code == 200 and "pub:" in u_resp.text:
                uid_name = ""
                for line in u_resp.text.splitlines():
                    if line.startswith("uid:"):
                        parts = line.split(":")
                        if len(parts) > 1 and parts[1]:
                            uid_name = unquote(parts[1])
                            break
                ub_url = f"https://keyserver.ubuntu.com/pks/lookup?search={encoded}&op=index"
                _add_hit(
                    Hit(
                        target_entity_id=entity.id,
                        discovered_type=EntityType.PROFILE,
                        discovered_value=ub_url,
                        relation="pgp_key_on",
                        source_tool="email_intel",
                        is_trusted=True,
                        raw_data={"site": "keyserver.ubuntu.com", "uid": uid_name, "email": email},
                        metadata={"site": "keyserver.ubuntu.com", "url": ub_url, "name": uid_name},
                    )
                )
        except (httpx.HTTPError, httpx.TimeoutException):
            pass

    async def _check_hudson_rock_email() -> None:
        """Check Hudson Rock Cavalier for infostealer compromise data tied to this email."""
        try:
            resp = await client.get(
                f"https://cavalier.hudsonrock.com/api/json/v2/osint-tools/search-by-email?email={quote(email, safe='')}",
                headers={"User-Agent": DEFAULT_USER_AGENT},
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                stealers = data.get("stealers") or []
                for stealer in stealers[:5]:
                    computer = stealer.get("computer_name", "")
                    os_name = stealer.get("operating_system", "")
                    stealer_family = stealer.get("stealer_family", "")
                    date_comp = stealer.get("date_compromised", "")
                    total_svcs = stealer.get("total_user_services", 0)
                    ip_addr = stealer.get("ip", "")
                    top_logins = stealer.get("top_logins") or []
                    top_passwords = stealer.get("top_passwords") or []
                    # Build summary — passwords are already redacted by Hudson Rock (e.g. "k********4")
                    summary_parts = []
                    if computer:
                        summary_parts.append(f"Computer: {computer}")
                    if os_name:
                        summary_parts.append(f"OS: {os_name}")
                    if stealer_family:
                        summary_parts.append(f"Stealer: {stealer_family}")
                    if date_comp:
                        summary_parts.append(f"Compromised: {date_comp}")
                    if total_svcs:
                        summary_parts.append(f"Services in stealer log: {total_svcs}")
                    if ip_addr:
                        summary_parts.append(f"IP (partial): {ip_addr}")
                    if top_logins:
                        summary_parts.append(f"Sample logins (redacted): {', '.join(str(l) for l in top_logins[:3])}")
                    if top_passwords:
                        summary_parts.append(f"Sample passwords (redacted by provider): {', '.join(str(p) for p in top_passwords[:3])}")
                    _add_hit(
                        Hit(
                            target_entity_id=entity.id,
                            discovered_type=EntityType.PROFILE,
                            discovered_value=f"infostealer:{email}:{date_comp or stealer_family or 'unknown'}",
                            relation="infostealer_compromise",
                            source_tool="email_intel",
                            is_trusted=True,
                            raw_data={
                                "source": "hudson_rock_cavalier",
                                "computer_name": computer,
                                "operating_system": os_name,
                                "stealer_type": stealer_family,
                                "date_compromised": date_comp,
                                "total_user_services": total_svcs,
                                "ip": ip_addr,
                                "top_logins_redacted": top_logins[:3],
                                "top_passwords_redacted": top_passwords[:3],
                                "description": " | ".join(summary_parts),
                            },
                            metadata={
                                "site": "Hudson Rock Cavalier",
                                "computer_name": computer,
                                "operating_system": os_name,
                                "stealer_type": stealer_family,
                                "date_compromised": date_comp,
                                "total_user_services": str(total_svcs),
                                "bio": " | ".join(summary_parts),
                                "note": "Infostealer compromise metadata only. Passwords shown as redacted patterns provided by Hudson Rock.",
                            },
                        )
                    )
        except (httpx.HTTPError, httpx.TimeoutException, ValueError) as exc:
            logger.debug("Hudson Rock email lookup failed for %s: %s", email, exc)

    async def _check_google_people() -> None:
        cid = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
        csec = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
        rtok = os.environ.get("GOOGLE_REFRESH_TOKEN", "").strip()
        if not (cid and csec and rtok):
            return
        try:
            tok_resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": cid,
                    "client_secret": csec,
                    "refresh_token": rtok,
                    "grant_type": "refresh_token",
                },
                timeout=8.0,
            )
            if tok_resp.status_code != 200:
                return
            access_token = tok_resp.json().get("access_token", "")
            if not access_token:
                return
            headers = {"Authorization": f"Bearer {access_token}"}
            for ep in (
                f"https://people.googleapis.com/v1/otherContacts:search?query={quote(email)}&readMask=names,photos,emailAddresses",
                f"https://people.googleapis.com/v1/people:searchContacts?query={quote(email)}&readMask=names,photos,emailAddresses",
            ):
                resp = await client.get(ep, headers=headers, timeout=8.0)
                if resp.status_code != 200:
                    continue
                for res_item in resp.json().get("results") or []:
                    person = res_item.get("person") if isinstance(res_item, dict) else None
                    if not isinstance(person, dict):
                        continue
                    emails_list = [str(e.get("value") or "").lower() for e in (person.get("emailAddresses") or []) if isinstance(e, dict)]
                    if emails_list and email not in emails_list:
                        continue
                    names_list = [str(n.get("displayName") or "").strip() for n in (person.get("names") or []) if isinstance(n, dict)]
                    photos_list = [str(p.get("url") or "").strip() for p in (person.get("photos") or []) if isinstance(p, dict) and p.get("url")]
                    display_name = next((n for n in names_list if n), "")
                    photo_url = next((u for u in photos_list if u.startswith("http")), "")
                    if display_name or photo_url:
                        _add_hit(
                            Hit(
                                target_entity_id=entity.id,
                                discovered_type=EntityType.PROFILE,
                                discovered_value=f"https://contacts.google.com/search/{quote(email)}",
                                relation="google_people_match",
                                source_tool="email_intel",
                                is_trusted=True,
                                raw_data={"site": "google", "name": display_name, "avatar_url": photo_url},
                                metadata={
                                    "site": "google",
                                    "email": email,
                                    **({"name": display_name} if display_name else {}),
                                    **({"avatar_url": photo_url, "images": [photo_url]} if photo_url else {}),
                                },
                            )
                        )
        except Exception as exc:
            logger.debug("Google People lookup failed for %s: %s", email, exc)

    tasks = [
        _check_gravatar(),
        _check_github(),
        _check_duolingo(),
        _check_pgp_keyservers(),
        _check_google_people(),
    ]
    if os.getenv("OSINT_ENABLE_BREACH_INTEL") == "1":
        tasks.extend([_check_xposedornot(), _check_hudson_rock_email()])

    await asyncio.gather(*tasks, return_exceptions=True)

    # 3. Run Holehe registration check (with -NP so no password reset emails are sent)
    binary = _find_cli_binary("holehe")
    if not binary:
        logger.warning("holehe binary is not installed; skipping holehe email registration check.")
        log_audit_event("tool_unavailable", {"tool": "holehe", "reason": "binary not found on PATH"})
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                proc = await asyncio.create_subprocess_exec(
                    binary,
                    email,
                    "--only-used",
                    "--no-color",
                    "--no-clear",
                    "-NP",
                    "-T",
                    "5",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=tmpdir,
                )
                try:
                    stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=20.0)
                except asyncio.TimeoutError:
                    proc.terminate()
                    stdout_bytes, _ = await proc.communicate()
                stdout = stdout_bytes.decode("utf-8", errors="replace")
                for h in parse_holehe_output(stdout, email, entity.id):
                    _add_hit(h)
            except Exception as exc:
                logger.warning("Holehe execution error for %s: %s", email, exc)

    for h in hits:
        if h.discovered_type == EntityType.PROFILE:
            log_audit_event(
                "profile_discovered",
                {
                    "site": h.metadata.get("site", "unknown"),
                    "url": h.discovered_value,
                    "username": email,
                },
            )

    return hits




@register_tool(
    name="web_archive_crawl",
    target_types=[EntityType.DOMAIN],
    description="Queries Wayback Machine CDX archives, sitemap.xml, and outbound social/email links on a domain.",
)
async def collect_web_archive_crawl(entity: Entity, client: httpx.AsyncClient) -> list[Hit]:
    """Query Internet Archive Wayback CDX API and politely crawl homepage/sitemap links."""
    if entity.depth > 0 and any(
        tag in p for p in entity.provenance for tag in ("nameserver_for", "mail_server_for")
    ):
        return []

    domain = _normalize_domain(entity.value)
    if domain.count(".") > 1 and entity.depth > 0:
        return []

    from urllib.parse import urlparse

    from osint_agent.collectors.enrich import can_fetch_robots, is_private_or_loopback_host
    from osint_agent.collectors.profile_analyzer import SOCIAL_LINK_PATTERN

    if is_private_or_loopback_host(domain):
        return []

    hits: list[Hit] = []
    seen_values: set[str] = set()
    wayback_urls: list[str] = []

    # 1. Query Wayback Machine CDX API (historical snapshots & endpoints)
    try:
        cdx_resp = await client.get(
            "https://web.archive.org/cdx/search/cdx",
            params={
                "url": domain,
                "matchType": "prefix",
                "output": "json",
                "limit": "10",
                "collapse": "urlkey",
                "fl": "timestamp,original,statuscode",
            },
            headers={"User-Agent": DEFAULT_USER_AGENT},
            timeout=6.0,
        )
        if cdx_resp.status_code == 200:
            rows = cdx_resp.json()
            if isinstance(rows, list) and len(rows) > 1:
                for row in rows[1:]:
                    if isinstance(row, list) and len(row) >= 2:
                        wayback_urls.append(f"{row[1]} ({str(row[0])[:8]})")
    except Exception as exc:
        logger.debug("Wayback CDX skipped/failed for %s: %s", domain, exc)

    # 2. Polite web crawl: check robots.txt, sitemap.xml, and homepage social/mailto links
    base_url = f"https://{domain}"
    sitemap_urls: list[str] = []
    try:
        if await can_fetch_robots(client, urlparse(base_url)):
            # Sitemap discovery
            sm_resp = await client.get(
                f"{base_url}/sitemap.xml",
                headers={"User-Agent": DEFAULT_USER_AGENT},
                follow_redirects=True,
                timeout=4.0,
            )
            if sm_resp.status_code == 200 and "<loc>" in sm_resp.text:
                sitemap_urls = re.findall(r"<loc>\s*(https?://[^<\s]+)\s*</loc>", sm_resp.text[:100_000])[:10]

            # Homepage link & mailto extraction
            home_resp = await client.get(
                base_url,
                headers={"User-Agent": DEFAULT_USER_AGENT},
                follow_redirects=True,
                timeout=5.0,
            )
            if home_resp.status_code == 200 and "text/html" in home_resp.headers.get("content-type", "").lower():
                html = home_resp.text[:200_000]

                # Mailto emails
                for email_match in re.findall(r'mailto:([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', html):
                    clean_email = email_match.strip().lower()
                    if clean_email not in seen_values:
                        seen_values.add(clean_email)
                        hits.append(
                            Hit(
                                target_entity_id=entity.id,
                                discovered_type=EntityType.EMAIL,
                                discovered_value=clean_email,
                                relation="contact_on_site",
                                source_tool="web_archive_crawl",
                                is_trusted=True,
                                raw_data={"domain": domain},
                                metadata={"source_url": str(home_resp.url)},
                            )
                        )

                # Outbound social profile links
                for soc_url in SOCIAL_LINK_PATTERN.findall(html)[:10]:
                    if soc_url not in seen_values:
                        seen_values.add(soc_url)
                        hits.append(
                            Hit(
                                target_entity_id=entity.id,
                                discovered_type=EntityType.PROFILE,
                                discovered_value=soc_url,
                                relation="linked_from_site",
                                source_tool="web_archive_crawl",
                                is_trusted=True,
                                raw_data={"domain": domain},
                                metadata={"url": soc_url},
                            )
                        )
    except Exception as exc:
        logger.debug("Web crawl skipped/failed for %s: %s", domain, exc)

    if wayback_urls or sitemap_urls:
        hits.append(
            Hit(
                target_entity_id=entity.id,
                discovered_type=EntityType.PROFILE,
                discovered_value=f"https://web.archive.org/web/*/{domain}",
                relation="archived_in",
                source_tool="web_archive_crawl",
                is_trusted=True,
                raw_data={
                    "wayback_snapshots": wayback_urls,
                    "sitemap_urls": sitemap_urls,
                    "description": f"Wayback snapshots: {len(wayback_urls)} | Sitemap URLs: {len(sitemap_urls)}",
                },
                metadata={
                    "wayback_count": len(wayback_urls),
                    "sitemap_count": len(sitemap_urls),
                },
            )
        )

    return hits


@register_tool(
    name="github_code_intel",
    target_types=[EntityType.USERNAME, EntityType.ORG],
    description="Discovers public GitHub repositories, languages, org memberships, project homepages, and commit emails.",
)
async def collect_github_code_intel(entity: Entity, client: httpx.AsyncClient) -> list[Hit]:
    """Gather public code repositories, org memberships, and commit metadata from GitHub API."""
    if entity.entity_type == EntityType.ORG and entity.depth > 0 and any(
        tag in p for p in entity.provenance for tag in ("registered_with", "hosted_on")
    ):
        return []

    slug = re.sub(r"[^a-zA-Z0-9-]", "", entity.value.strip().lstrip("@"))
    if not slug:
        return []

    import os

    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/vnd.github+json"}
    gh_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"
    hits: list[Hit] = []
    seen_values: set[str] = set()

    # 1. Public repositories
    repos_url = (
        f"https://api.github.com/orgs/{slug}/repos?sort=updated&per_page=8"
        if entity.entity_type == EntityType.ORG
        else f"https://api.github.com/users/{slug}/repos?sort=updated&per_page=8"
    )
    try:
        r_resp = await client.get(repos_url, headers=headers, timeout=6.0)
        if r_resp.status_code == 200 and isinstance(r_resp.json(), list):
            for repo in r_resp.json()[:8]:
                html_url = repo.get("html_url")
                if not html_url or html_url in seen_values:
                    continue
                seen_values.add(html_url)
                lang = repo.get("language") or "Unknown"
                stars = repo.get("stargazers_count", 0)
                desc = repo.get("description") or ""
                hits.append(
                    Hit(
                        target_entity_id=entity.id,
                        discovered_type=EntityType.PROFILE,
                        discovered_value=html_url,
                        relation="authored_repo",
                        source_tool="github_code_intel",
                        is_trusted=True,
                        raw_data={
                            "repo_name": repo.get("name"),
                            "language": lang,
                            "stars": stars,
                            "description": f"[{lang}, {stars}★] {desc}".strip(),
                        },
                        metadata={
                            "name": repo.get("name", ""),
                            "bio": f"[{lang}, {stars}★] {desc}".strip(),
                            "url": html_url,
                        },
                    )
                )

                # Extract custom project homepage domain
                homepage = (repo.get("homepage") or "").strip()
                if homepage.startswith("http"):
                    hp_domain = _normalize_domain(homepage)
                    if hp_domain and "github.com" not in hp_domain and hp_domain not in seen_values:
                        seen_values.add(hp_domain)
                        hits.append(
                            Hit(
                                target_entity_id=entity.id,
                                discovered_type=EntityType.DOMAIN,
                                discovered_value=hp_domain,
                                relation="project_homepage",
                                source_tool="github_code_intel",
                                is_trusted=True,
                                raw_data={"repo": repo.get("name")},
                                metadata={"domain": hp_domain},
                            )
                        )
    except Exception as exc:
        logger.debug("GitHub repos lookup failed for %s: %s", slug, exc)

    # 2. Organization memberships & public commit author emails (for USERNAME targets)
    if entity.entity_type == EntityType.USERNAME:
        try:
            o_resp = await client.get(f"https://api.github.com/users/{slug}/orgs", headers=headers, timeout=5.0)
            if o_resp.status_code == 200 and isinstance(o_resp.json(), list):
                for org in o_resp.json()[:5]:
                    org_login = org.get("login")
                    if org_login and org_login.lower() not in seen_values:
                        seen_values.add(org_login.lower())
                        hits.append(
                            Hit(
                                target_entity_id=entity.id,
                                discovered_type=EntityType.ORG,
                                discovered_value=org_login,
                                relation="member_of",
                                source_tool="github_code_intel",
                                is_trusted=True,
                                raw_data={"description": org.get("description") or ""},
                                metadata={"org": org_login},
                            )
                        )
        except Exception:
            pass

        try:
            e_resp = await client.get(
                f"https://api.github.com/users/{slug}/events/public?per_page=15",
                headers=headers,
                timeout=5.0,
            )
            if e_resp.status_code == 200 and isinstance(e_resp.json(), list):
                for ev in e_resp.json():
                    if ev.get("type") != "PushEvent":
                        continue
                    for commit in ev.get("payload", {}).get("commits", []):
                        author = commit.get("author", {})
                        email = (author.get("email") or "").strip().lower()
                        if (
                            email
                            and "@" in email
                            and "noreply.github.com" not in email
                            and email not in seen_values
                        ):
                            seen_values.add(email)
                            hits.append(
                                Hit(
                                    target_entity_id=entity.id,
                                    discovered_type=EntityType.EMAIL,
                                    discovered_value=email,
                                    relation="commit_author_email",
                                    source_tool="github_code_intel",
                                    is_trusted=True,
                                    raw_data={"author_name": author.get("name")},
                                    metadata={"email": email},
                                )
                            )
        except Exception:
            pass

    return hits


@register_tool(
    name="threat_intel",
    target_types=[EntityType.DOMAIN],
    description="Queries keyless URLScan.io public scans plus optional VirusTotal, AlienVault OTX, and AbuseIPDB feeds.",
)
async def collect_threat_intel(entity: Entity, client: httpx.AsyncClient) -> list[Hit]:
    """Gather threat intelligence, ASN info, and reputation signals for a domain."""
    if entity.depth > 0 and any(
        tag in p for p in entity.provenance for tag in ("nameserver_for", "mail_server_for")
    ):
        return []

    domain = _normalize_domain(entity.value)
    if domain.count(".") > 1 and entity.depth > 0:
        return []

    hits: list[Hit] = []
    seen_values: set[str] = set()
    summary_notes: list[str] = []

    # 1. URLScan.io Public Search API (100% free & keyless)
    try:
        us_resp = await client.get(
            "https://urlscan.io/api/v1/search/",
            params={"q": f"page.domain:{domain}", "size": "5"},
            headers={"User-Agent": DEFAULT_USER_AGENT},
            timeout=6.0,
        )
        if us_resp.status_code == 200:
            results = us_resp.json().get("results", [])
            if results:
                first_page = results[0].get("page", {})
                asn_name = (first_page.get("asnname") or "").strip()
                if asn_name and asn_name.lower() not in seen_values:
                    seen_values.add(asn_name.lower())
                    hits.append(
                        Hit(
                            target_entity_id=entity.id,
                            discovered_type=EntityType.ORG,
                            discovered_value=asn_name,
                            relation="asn_provider",
                            source_tool="threat_intel",
                            is_trusted=True,
                            raw_data={"asn": first_page.get("asn"), "country": first_page.get("country")},
                            metadata={"asn": first_page.get("asn", "")},
                        )
                    )
                age_days = first_page.get("domainAgeDays")
                tls_issuer = first_page.get("tlsIssuer")
                if age_days is not None:
                    summary_notes.append(f"DomainAge: {age_days}d")
                if tls_issuer:
                    summary_notes.append(f"TLS: {tls_issuer}")

                report_url = results[0].get("result")
                if report_url and report_url not in seen_values:
                    seen_values.add(report_url)
                    hits.append(
                        Hit(
                            target_entity_id=entity.id,
                            discovered_type=EntityType.PROFILE,
                            discovered_value=report_url,
                            relation="urlscan_report",
                            source_tool="threat_intel",
                            is_trusted=True,
                            raw_data={
                                "ip": first_page.get("ip"),
                                "asn": first_page.get("asn"),
                                "description": " | ".join(summary_notes) or "URLScan.io public scan",
                            },
                            metadata={"url": report_url, "scans_found": len(results)},
                        )
                    )
    except Exception as exc:
        logger.debug("URLScan.io lookup failed for %s: %s", domain, exc)

    # 2. Optional VirusTotal v3 API (if VIRUSTOTAL_API_KEY is set)
    vt_key = os.getenv("VIRUSTOTAL_API_KEY", "").strip()
    if vt_key:
        try:
            vt_resp = await client.get(
                f"https://www.virustotal.com/api/v3/domains/{domain}",
                headers={"x-apikey": vt_key, "User-Agent": DEFAULT_USER_AGENT},
                timeout=6.0,
            )
            if vt_resp.status_code == 200:
                stats = vt_resp.json().get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
                malicious = stats.get("malicious", 0)
                harmless = stats.get("harmless", 0)
                hits.append(
                    Hit(
                        target_entity_id=entity.id,
                        discovered_type=EntityType.PROFILE,
                        discovered_value=f"https://www.virustotal.com/gui/domain/{domain}",
                        relation="virustotal_report",
                        source_tool="threat_intel",
                        is_trusted=True,
                        raw_data={"stats": stats, "description": f"VT Malicious: {malicious} | Harmless: {harmless}"},
                        metadata={"malicious": malicious, "harmless": harmless},
                    )
                )
        except Exception as exc:
            logger.debug("VirusTotal lookup failed for %s: %s", domain, exc)

    # 3. Optional AlienVault OTX API (if OTX_API_KEY is set)
    otx_key = os.getenv("OTX_API_KEY", "").strip()
    if otx_key:
        try:
            otx_resp = await client.get(
                f"https://otx.alienvault.com/api/v1/indicators/domain/{domain}/general",
                headers={"X-OTX-API-KEY": otx_key, "User-Agent": DEFAULT_USER_AGENT},
                timeout=6.0,
            )
            if otx_resp.status_code == 200:
                pulses = otx_resp.json().get("pulse_info", {}).get("count", 0)
                hits.append(
                    Hit(
                        target_entity_id=entity.id,
                        discovered_type=EntityType.PROFILE,
                        discovered_value=f"https://otx.alienvault.com/indicator/domain/{domain}",
                        relation="otx_pulses",
                        source_tool="threat_intel",
                        is_trusted=True,
                        raw_data={"pulse_count": pulses, "description": f"OTX Threat Pulses: {pulses}"},
                        metadata={"pulse_count": pulses},
                    )
                )
        except Exception as exc:
            logger.debug("OTX lookup failed for %s: %s", domain, exc)

    return hits


@register_tool(
    name="open_data_intel",
    target_types=[EntityType.ORG, EntityType.PUBLIC_FIGURE, EntityType.PLACE],
    description="Queries Wikipedia summaries, OpenSky Network live flights over places, GDELT news, and optional OpenSanctions.",
)
async def collect_open_data_intel(entity: Entity, client: httpx.AsyncClient) -> list[Hit]:
    """Gather open-data intelligence from Wikipedia, OpenSky Network, GDELT, and OpenSanctions."""
    if entity.depth > 0 and any(
        tag in p for p in entity.provenance for tag in ("registered_with", "hosted_on", "asn_provider", "nearby_landmark")
    ):
        return []

    query = entity.value.strip()
    if not query:
        return []

    hits: list[Hit] = []
    seen_values: set[str] = set()

    # 1. OpenSky Network live flights over a PLACE (free & keyless)
    if entity.entity_type == EntityType.PLACE:
        lat = entity.attributes.get("lat")
        lon = entity.attributes.get("lon")
        if lat is None or lon is None:
            try:
                nom_resp = await client.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": query, "format": "jsonv2", "limit": "1"},
                    headers={"User-Agent": DEFAULT_USER_AGENT},
                    timeout=5.0,
                )
                if nom_resp.status_code == 200 and nom_resp.json():
                    lat = float(nom_resp.json()[0]["lat"])
                    lon = float(nom_resp.json()[0]["lon"])
            except Exception:
                pass

        if lat is not None and lon is not None:
            try:
                lat_f, lon_f = float(lat), float(lon)
                os_resp = await client.get(
                    "https://opensky-network.org/api/states/all",
                    params={
                        "lamin": f"{lat_f - 0.5:.4f}",
                        "lomin": f"{lon_f - 0.5:.4f}",
                        "lamax": f"{lat_f + 0.5:.4f}",
                        "lomax": f"{lon_f + 0.5:.4f}",
                    },
                    headers={"User-Agent": DEFAULT_USER_AGENT},
                    timeout=6.0,
                )
                if os_resp.status_code == 200:
                    states = os_resp.json().get("states") or []
                    for st in states[:5]:
                        if not isinstance(st, list) or len(st) < 10:
                            continue
                        callsign = (st[1] or st[0] or "").strip()
                        country = (st[2] or "Unknown").strip()
                        alt_m = st[7] or st[13] or 0
                        vel_ms = st[9] or 0
                        if callsign and callsign not in seen_values:
                            seen_values.add(callsign)
                            hits.append(
                                Hit(
                                    target_entity_id=entity.id,
                                    discovered_type=EntityType.PROFILE,
                                    discovered_value=f"Flight {callsign} ({country})",
                                    relation="live_flight_overhead",
                                    source_tool="open_data_intel",
                                    is_trusted=True,
                                    raw_data={
                                        "callsign": callsign,
                                        "origin_country": country,
                                        "altitude_m": alt_m,
                                        "velocity_ms": vel_ms,
                                        "description": f"Live flight {callsign} from {country} at {alt_m}m",
                                    },
                                    metadata={"callsign": callsign, "country": country},
                                )
                            )
            except Exception as exc:
                logger.debug("OpenSky lookup failed for %s: %s", query, exc)

    # 2. Wikipedia REST Summary API (100% free & keyless)
    if entity.entity_type in (EntityType.ORG, EntityType.PUBLIC_FIGURE, EntityType.PLACE):
        wiki_slug = query.replace(" ", "_")
        try:
            wk_resp = await client.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{wiki_slug}",
                headers={"User-Agent": DEFAULT_USER_AGENT},
                follow_redirects=True,
                timeout=5.0,
            )
            if wk_resp.status_code == 200:
                wdata = wk_resp.json()
                page_url = wdata.get("content_urls", {}).get("desktop", {}).get("page")
                extract = wdata.get("extract", "")
                if page_url and page_url not in seen_values:
                    seen_values.add(page_url)
                    hits.append(
                        Hit(
                            target_entity_id=entity.id,
                            discovered_type=EntityType.PROFILE,
                            discovered_value=page_url,
                            relation="wikipedia_summary",
                            source_tool="open_data_intel",
                            is_trusted=True,
                            raw_data={"description": extract[:500], "title": wdata.get("title")},
                            metadata={"url": page_url, "bio": extract[:300]},
                        )
                    )
        except Exception as exc:
            logger.debug("Wikipedia summary failed for %s: %s", query, exc)

    # 3. GDELT Global News Events (best-effort, free & keyless)
    if entity.entity_type in (EntityType.ORG, EntityType.PUBLIC_FIGURE):
        try:
            gd_resp = await client.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={"query": f'"{query}"', "mode": "ArtList", "maxrecords": "3", "format": "json"},
                headers={"User-Agent": DEFAULT_USER_AGENT},
                timeout=4.0,
            )
            if gd_resp.status_code == 200:
                for art in gd_resp.json().get("articles", [])[:3]:
                    art_url = art.get("url")
                    if art_url and art_url not in seen_values:
                        seen_values.add(art_url)
                        hits.append(
                            Hit(
                                target_entity_id=entity.id,
                                discovered_type=EntityType.PROFILE,
                                discovered_value=art_url,
                                relation="news_mention",
                                source_tool="open_data_intel",
                                is_trusted=True,
                                raw_data={"title": art.get("title"), "description": art.get("title", "")},
                                metadata={"url": art_url, "domain": art.get("domain", "")},
                            )
                        )
        except Exception as exc:
            logger.debug("GDELT lookup skipped/failed for %s: %s", query, exc)

    # 4. Optional OpenSanctions API (if OPENSANCTIONS_API_KEY is set)
    osanc_key = os.getenv("OPENSANCTIONS_API_KEY", "").strip()
    if osanc_key and entity.entity_type in (EntityType.ORG, EntityType.PUBLIC_FIGURE):
        try:
            s_resp = await client.get(
                "https://api.opensanctions.org/search/default",
                params={"q": query, "limit": "3"},
                headers={"Authorization": f"ApiKey {osanc_key}", "User-Agent": DEFAULT_USER_AGENT},
                timeout=5.0,
            )
            if s_resp.status_code == 200:
                for res in s_resp.json().get("results", [])[:3]:
                    caption = res.get("caption")
                    if caption and caption not in seen_values:
                        seen_values.add(caption)
                        hits.append(
                            Hit(
                                target_entity_id=entity.id,
                                discovered_type=EntityType.PROFILE,
                                discovered_value=f"https://www.opensanctions.org/entities/{res.get('id', '')}/",
                                relation="sanctions_match",
                                source_tool="open_data_intel",
                                is_trusted=False,
                                raw_data={"name": caption, "datasets": res.get("datasets", [])},
                                metadata={"name": caption, "schema": res.get("schema", "")},
                            )
                        )
        except Exception as exc:
            logger.debug("OpenSanctions lookup failed for %s: %s", query, exc)

    return hits


@register_tool(
    name="playstore_intel",
    target_types=[EntityType.USERNAME, EntityType.PUBLIC_FIGURE, EntityType.ORG, EntityType.CONSENTING_PERSON],
    description="Searches Google Play Store developer pages and apps to extract developer email, website, and office address.",
)
async def collect_playstore_intel(entity: Entity, client: httpx.AsyncClient) -> list[Hit]:
    """Discover Google Play Store developer pages and published apps for a handle, person, or org."""
    from urllib.parse import quote

    query = entity.value.strip().lstrip("@")
    if not query or len(query) < 3:
        return []

    hits: list[Hit] = []
    seen_urls: set[str] = set()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    dev_urls = [
        f"https://play.google.com/store/apps/developer?id={quote(query)}&hl=en",
        f"https://play.google.com/store/search?q={quote(query)}&c=apps&hl=en",
    ]
    query_norm = re.sub(r"[^a-z0-9]", "", query.lower())

    for idx, dev_url in enumerate(dev_urls):
        try:
            dev_resp = await client.get(dev_url, headers=headers, follow_redirects=True, timeout=8.0)
            if dev_resp.status_code != 200:
                continue
            html = dev_resp.text
            is_search = idx == 1

            # Extract matching developer pages
            dev_ids = list(dict.fromkeys(re.findall(r'/store/apps/(?:developer|dev)\?id=([^"&<>\\]+)', html)))
            for dev_id in dev_ids[:5]:
                from urllib.parse import unquote as _unquote
                dev_clean = _unquote(dev_id)
                if is_search and query_norm not in re.sub(r"[^a-z0-9]", "", dev_clean.lower()):
                    continue
                d_url = f"https://play.google.com/store/apps/developer?id={quote(dev_clean)}&hl=en"
                if d_url not in seen_urls:
                    seen_urls.add(d_url)
                    hits.append(
                        Hit(
                            target_entity_id=entity.id,
                            discovered_type=EntityType.PROFILE,
                            discovered_value=d_url,
                            relation="developer_on",
                            source_tool="playstore_intel",
                            is_trusted=True,
                            raw_data={"site": "Google Play Store", "developer": dev_clean},
                            metadata={"site": "Google Play Store", "url": d_url, "name": dev_clean},
                        )
                    )

            if not is_search and "/store/apps/details?id=" in html and dev_url not in seen_urls:
                seen_urls.add(dev_url)
                hits.append(
                    Hit(
                        target_entity_id=entity.id,
                        discovered_type=EntityType.PROFILE,
                        discovered_value=dev_url,
                        relation="developer_on",
                        source_tool="playstore_intel",
                        is_trusted=True,
                        raw_data={"site": "Google Play Store", "developer": query},
                        metadata={"site": "Google Play Store", "url": dev_url, "name": query},
                    )
                )

            app_ids = list(dict.fromkeys(re.findall(r"/store/apps/details\?id=([a-zA-Z0-9._]+)", html)))
            for app_id in app_ids[:10]:
                if is_search and query_norm not in re.sub(r"[^a-z0-9]", "", app_id.lower()):
                    continue
                app_url = f"https://play.google.com/store/apps/details?id={app_id}&hl=en"
                if app_url not in seen_urls:
                    seen_urls.add(app_url)
                    hits.append(
                        Hit(
                            target_entity_id=entity.id,
                            discovered_type=EntityType.PROFILE,
                            discovered_value=app_url,
                            relation="published_app",
                            source_tool="playstore_intel",
                            is_trusted=True,
                            raw_data={"site": "Google Play Store", "app_id": app_id, "developer": query},
                            metadata={"site": "Google Play Store", "url": app_url, "name": app_id},
                        )
                    )
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            logger.debug("Play Store lookup failed for %s (%s): %s", query, dev_url, exc)

    return hits


@register_tool(
    name="breach_intel",
    target_types=[EntityType.USERNAME, EntityType.PUBLIC_FIGURE, EntityType.CONSENTING_PERSON],
    description=(
        "Checks Hudson Rock Cavalier for infostealer compromise data tied to a username, "
        "reporting breach metadata (computer name, OS, stealer family, compromised date, "
        "service count, redacted login/password patterns). Never exposes plaintext passwords."
    ),
)
async def collect_breach_intel(entity: Entity, client: httpx.AsyncClient) -> list[Hit]:
    """Query Hudson Rock Cavalier OSINT API by username for infostealer compromise metadata."""
    if os.getenv("OSINT_ENABLE_BREACH_INTEL") != "1":
        return []
    from urllib.parse import quote

    username = entity.value.strip().lstrip("@")
    if not username or len(username) < 2:
        return []

    hits: list[Hit] = []

    try:
        resp = await client.get(
            f"https://cavalier.hudsonrock.com/api/json/v2/osint-tools/search-by-username?username={quote(username, safe='')}",
            headers={"User-Agent": DEFAULT_USER_AGENT},
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            stealers = data.get("stealers") or []
            for stealer in stealers[:5]:
                computer = stealer.get("computer_name", "")
                os_name = stealer.get("operating_system", "")
                stealer_family = stealer.get("stealer_family", "")
                date_comp = stealer.get("date_compromised", "")
                total_svcs = stealer.get("total_user_services", 0)
                ip_addr = stealer.get("ip", "")
                top_logins = stealer.get("top_logins") or []
                top_passwords = stealer.get("top_passwords") or []
                # NOTE: Hudson Rock already redacts passwords (e.g. "k********4").
                # We store and emit only the redacted patterns as metadata — never raw plaintext.
                summary_parts = []
                if computer:
                    summary_parts.append(f"Computer: {computer}")
                if os_name:
                    summary_parts.append(f"OS: {os_name}")
                if stealer_family:
                    summary_parts.append(f"Stealer: {stealer_family}")
                if date_comp:
                    summary_parts.append(f"Compromised: {date_comp}")
                if total_svcs:
                    summary_parts.append(f"Services in stealer log: {total_svcs}")
                if ip_addr:
                    summary_parts.append(f"IP (partial): {ip_addr}")
                if top_logins:
                    summary_parts.append(
                        f"Sample logins (redacted): {', '.join(str(l) for l in top_logins[:3])}"
                    )
                if top_passwords:
                    summary_parts.append(
                        f"Sample passwords (redacted by provider): {', '.join(str(p) for p in top_passwords[:3])}"
                    )

                hits.append(
                    Hit(
                        target_entity_id=entity.id,
                        discovered_type=EntityType.PROFILE,
                        discovered_value=(
                            f"infostealer:{username}:{date_comp or stealer_family or 'unknown'}"
                        ),
                        relation="infostealer_compromise",
                        source_tool="breach_intel",
                        is_trusted=True,
                        raw_data={
                            "source": "hudson_rock_cavalier",
                            "username": username,
                            "computer_name": computer,
                            "operating_system": os_name,
                            "stealer_type": stealer_family,
                            "date_compromised": date_comp,
                            "total_user_services": total_svcs,
                            "ip": ip_addr,
                            "top_logins_redacted": top_logins[:3],
                            "top_passwords_redacted": top_passwords[:3],
                            "description": " | ".join(summary_parts),
                        },
                        metadata={
                            "site": "Hudson Rock Cavalier",
                            "username": username,
                            "computer_name": computer,
                            "operating_system": os_name,
                            "stealer_type": stealer_family,
                            "date_compromised": date_comp,
                            "total_user_services": str(total_svcs),
                            "bio": " | ".join(summary_parts),
                            "note": (
                                "Infostealer compromise metadata only. "
                                "Passwords shown as redacted patterns as provided by Hudson Rock Cavalier."
                            ),
                        },
                    )
                )
    except (httpx.HTTPError, httpx.TimeoutException, ValueError) as exc:
        logger.debug("Hudson Rock username lookup failed for %s: %s", username, exc)

    return hits


# ---------------------------------------------------------------------------
# Exa AI Web Search Collector (via Hack Club proxy or direct Exa API)
# ---------------------------------------------------------------------------

_EXA_402_BLOCKED: bool = False
_EXA_COOLDOWN_UNTIL: float = 0.0


def reset_exa_state() -> None:
    """Reset Exa 402/429 circuit-breaker state (primarily for tests)."""
    global _EXA_402_BLOCKED, _EXA_COOLDOWN_UNTIL
    _EXA_402_BLOCKED = False
    _EXA_COOLDOWN_UNTIL = 0.0


def _build_search_queries(entity: Entity) -> list[str]:
    """Generate ranked query variants for an entity."""
    val = entity.value.strip()
    if entity.entity_type == EntityType.DOMAIN:
        return [f'site:{val} OR "{val}"', f'"{val}"']
    city = str((entity.attributes or {}).get("location") or "").strip()
    queries = [f'"{val}"']
    if entity.entity_type in (EntityType.USERNAME, EntityType.PUBLIC_FIGURE, EntityType.CONSENTING_PERSON):
        queries.append(f'"{val}" site:github.com')
    if city:
        queries.append(f'"{val}" "{city}"')
    return queries


async def _search_brave(query: str, api_key: str, client: httpx.AsyncClient) -> list[dict[str, str]]:
    resp = await client.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
        params={"q": query, "count": 5},
        timeout=10.0,
    )
    if resp.status_code != 200:
        return []
    results = (resp.json().get("web") or {}).get("results") or []
    return [
        {"url": str(r.get("url") or ""), "title": str(r.get("title") or ""), "snippet": str(r.get("description") or "")}
        for r in results
        if isinstance(r, dict)
    ]


async def _search_tavily(query: str, api_key: str, client: httpx.AsyncClient) -> list[dict[str, Any]]:
    resp = await client.post(
        "https://api.tavily.com/search",
        json={"api_key": api_key, "query": query, "max_results": 5, "include_images": True},
        timeout=10.0,
    )
    if resp.status_code != 200:
        return []
    body = resp.json()
    results = body.get("results") or []
    imgs = [str(u) for u in (body.get("images") or []) if isinstance(u, str) and u.startswith("http")][:8]
    out: list[dict[str, Any]] = []
    for idx, r in enumerate(results):
        if isinstance(r, dict):
            item: dict[str, Any] = {
                "url": str(r.get("url") or ""),
                "title": str(r.get("title") or ""),
                "snippet": str(r.get("content") or ""),
            }
            if idx == 0 and imgs:
                item["images"] = imgs
            out.append(item)
    return out


async def _search_google_pse(query: str, api_key: str, cx: str, client: httpx.AsyncClient) -> list[dict[str, Any]]:
    resp = await client.get(
        "https://www.googleapis.com/customsearch/v1",
        params={"key": api_key, "cx": cx, "q": query, "num": 5},
        timeout=10.0,
    )
    if resp.status_code != 200:
        return []
    results = resp.json().get("items") or []
    out: list[dict[str, Any]] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        pm = r.get("pagemap") or {}
        imgs = [
            str(im.get("src"))
            for im in (pm.get("cse_image") or pm.get("cse_thumbnail") or [])
            if isinstance(im, dict) and isinstance(im.get("src"), str) and im["src"].startswith("http")
        ][:4]
        item: dict[str, Any] = {
            "url": str(r.get("link") or ""),
            "title": str(r.get("title") or ""),
            "snippet": str(r.get("snippet") or ""),
        }
        if imgs:
            item["images"] = imgs
        out.append(item)
    return out


async def _fetch_pdf_text(url: str, client: httpx.AsyncClient, max_chars: int = 3000) -> str:
    """Download a public PDF (<= 5 MB) and extract plain text via pdftotext."""
    import asyncio
    import shutil

    try:
        resp = await client.get(url, timeout=10.0, follow_redirects=True)
        if resp.status_code != 200 or not resp.content or len(resp.content) > 5_000_000:
            return ""
        if not resp.content.lstrip().startswith(b"%PDF") and "pdf" not in resp.headers.get("content-type", "").lower():
            return ""
        pdftotext_bin = shutil.which("pdftotext")
        if pdftotext_bin:
            proc = await asyncio.create_subprocess_exec(
                pdftotext_bin,
                "-",
                "-",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(input=resp.content), timeout=8.0)
            text = re.sub(r"\s+", " ", stdout.decode("utf-8", errors="ignore")).strip()
            return text[:max_chars]
    except Exception:
        return ""
    return ""


@register_tool(
    name="exa_search",
    target_types=[
        EntityType.USERNAME,
        EntityType.EMAIL,
        EntityType.PUBLIC_FIGURE,
        EntityType.CONSENTING_PERSON,
        EntityType.ORG,
        EntityType.DOMAIN,
    ],
    description=(
        "Uses Exa AI neural/keyword web search to find public profiles, apps, mentions, "
        "and page text for an email, username, person, org, or domain. Never retries on 402/429."
    ),
)
async def collect_exa_search(entity: Entity, client: httpx.AsyncClient) -> list[Hit]:
    """Pluggable web-search collector (Brave, Tavily, Google PSE, or Exa). Skips with tool_unavailable if no key is set."""
    global _EXA_402_BLOCKED, _EXA_COOLDOWN_UNTIL
    import time
    from urllib.parse import urlparse
    from osint_agent.llm.client import _load_dotenv

    val = entity.value.strip()
    if not val or len(val) < 2:
        return []

    if not os.getenv("PYTEST_CURRENT_TEST"):
        _load_dotenv()
    brave_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    google_key = os.getenv("GOOGLE_PSE_API_KEY", "").strip()
    google_cx = os.getenv("GOOGLE_PSE_CX", "").strip()
    exa_key = os.getenv("EXA_API_KEY", "").strip()

    has_pluggable = bool(brave_key or tavily_key or (google_key and google_cx) or (exa_key and not _EXA_402_BLOCKED and time.monotonic() >= _EXA_COOLDOWN_UNTIL))
    if not has_pluggable:
        log_audit_event(
            "tool_unavailable",
            {
                "tool": "exa_search",
                "reason": "No web search API key configured (BRAVE_SEARCH_API_KEY, TAVILY_API_KEY, GOOGLE_PSE_API_KEY+GOOGLE_PSE_CX, or EXA_API_KEY).",
            },
        )
        return []

    queries = _build_search_queries(entity)
    raw_items: list[tuple[int, dict[str, str]]] = []

    for q_idx, query in enumerate(queries):
        batch: list[dict[str, str]] = []
        if brave_key:
            batch = await _search_brave(query, brave_key, client)
        elif tavily_key:
            batch = await _search_tavily(query, tavily_key, client)
        elif google_key and google_cx:
            batch = await _search_google_pse(query, google_key, google_cx, client)
        elif exa_key and not _EXA_402_BLOCKED and time.monotonic() >= _EXA_COOLDOWN_UNTIL:
            base_url = os.getenv("EXA_BASE_URL", "https://ai.hackclub.com/proxy/v1/exa").rstrip("/")
            try:
                resp = await client.post(
                    f"{base_url}/search",
                    headers={"Authorization": f"Bearer {exa_key}", "Content-Type": "application/json"},
                    json={"query": query, "numResults": 5, "contents": {"text": {"maxCharacters": 800}}},
                    timeout=12.0,
                )
                if resp.status_code == 402:
                    _EXA_402_BLOCKED = True
                    break
                elif resp.status_code == 429:
                    _EXA_COOLDOWN_UNTIL = time.monotonic() + 120.0
                    break
                elif resp.status_code == 200:
                    for r in resp.json().get("results") or []:
                        if isinstance(r, dict):
                            batch.append(
                                {
                                    "url": str(r.get("url") or ""),
                                    "title": str(r.get("title") or ""),
                                    "snippet": str(r.get("text") or "")[:800],
                                    "author": str(r.get("author") or "").strip(),
                                    "published_date": str(r.get("publishedDate") or "").strip(),
                                }
                            )
            except Exception:
                pass
        for item in batch:
            raw_items.append((q_idx, item))

    deduped: dict[str, tuple[float, dict[str, str]]] = {}
    val_lower = val.lower()
    for q_idx, item in raw_items:
        url = str(item.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        norm = url.rstrip("/").lower()
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        score = (3 - q_idx) + (2.0 if val_lower in url.lower() else 0.0) + (1.0 if val_lower in title.lower() else 0.0)
        if norm not in deduped or score > deduped[norm][0]:
            deduped[norm] = (
                score,
                {
                    "url": url,
                    "title": title,
                    "snippet": snippet,
                    "author": str(item.get("author") or ""),
                    "published_date": str(item.get("published_date") or ""),
                    "images": item.get("images") or [],
                },
            )

    ranked = sorted(deduped.values(), key=lambda pair: pair[0], reverse=True)
    hits: list[Hit] = []
    for _, item in ranked[:8]:
        url = item["url"]
        title = item["title"]
        snippet = item["snippet"]
        pdf_text = ""
        if urlparse(url).path.lower().endswith(".pdf"):
            pdf_text = await _fetch_pdf_text(url, client)
            if pdf_text:
                snippet = pdf_text
        author = item.get("author", "")
        pub_date = item.get("published_date", "")
        site = urlparse(url).netloc.lower()
        hits.append(
            Hit(
                target_entity_id=entity.id,
                discovered_type=EntityType.PROFILE,
                discovered_value=url,
                relation="web_mention",
                source_tool="exa_search",
                is_trusted=False,
                raw_data={"site": site, "url": url, "title": title, "description": snippet[:1500 if pdf_text else 300]},
                metadata={
                    "site": site,
                    "url": url,
                    "title": title,
                    "bio": (snippet[:1500] if pdf_text else snippet[:400]) if snippet else title,
                    **({"document_text": pdf_text} if pdf_text else {}),
                    **({"images": item["images"]} if item.get("images") else {}),
                    **({"name": author} if author else {}),
                    **({"published_date": pub_date} if pub_date else {}),
                },
            )
        )
    return hits
