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
    binary = _find_cli_binary("maigret")
    if not binary:
        logger.warning("maigret binary is not installed on PATH.")
        log_audit_event("tool_unavailable", {"tool": "maigret", "reason": "binary not found on PATH"})
        return []

    username = entity.value.strip().lstrip("@")

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
    description="Checks public email registrations via Holehe, Gravatar profile API, and custom domain linking.",
)
async def collect_email_intel(entity: Entity, client: httpx.AsyncClient) -> list[Hit]:
    """Gather public registration, Gravatar profile, and domain leads for an email address."""
    import hashlib

    email = entity.value.strip().lower()
    if "@" not in email:
        return []

    hits: list[Hit] = []
    seen_values: set[str] = set()

    # 1. Link non-free corporate/custom email domain
    domain = email.split("@", 1)[1].strip()
    if domain and domain not in FREE_EMAIL_PROVIDERS:
        seen_values.add(domain)
        hits.append(
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

    # 2. Query public Gravatar JSON profile
    email_hash = hashlib.md5(email.encode("utf-8")).hexdigest()
    gravatar_url = f"https://en.gravatar.com/{email_hash}.json"
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
                prof = entries[0]
                pref_user = prof.get("preferredUsername")
                if pref_user and pref_user not in seen_values:
                    seen_values.add(pref_user)
                    hits.append(
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
                    if acc_url and acc_url not in seen_values:
                        seen_values.add(acc_url)
                        hits.append(
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
        logger.debug("Gravatar lookup skipped/failed for %s: %s", email, exc)

    # 3. Run Holehe registration check (with -NP so no password reset emails are sent)
    binary = _find_cli_binary("holehe")
    if not binary:
        logger.warning("holehe binary is not installed; skipping email registration check.")
        log_audit_event("tool_unavailable", {"tool": "holehe", "reason": "binary not found on PATH"})
        return hits



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
            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            for h in parse_holehe_output(stdout, email, entity.id):
                if h.discovered_value not in seen_values:
                    seen_values.add(h.discovered_value)
                    hits.append(h)
                    log_audit_event(
                        "profile_discovered",
                        {
                            "site": h.metadata.get("site", "unknown"),
                            "url": h.discovered_value,
                            "username": email,
                        },
                    )
        except Exception as exc:
            logger.warning("Holehe execution error for %s: %s", email, exc)

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

    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/vnd.github+json"}
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
            for repo in r_resp.json()[:6]:
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
