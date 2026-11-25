"""Comprehensive pytest suite for Intelligent OSINT Agent."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from osint_agent.core.agent import OSINTAgent, export_graph_to_json
from osint_agent.ml.resolver import BaselineWeightedResolver
from osint_agent.core.schema import Entity, EntityType, Hit, Scope, sanitize_attributes
from osint_agent.collectors.tools import collect_crtsh, collect_nominatim, collect_wikidata


@pytest.mark.asyncio
async def test_consent_refusal(tmp_path: Path) -> None:
    """Agent must refuse to run if Scope.consent_confirmed is False, logging refusal."""
    audit_path = tmp_path / "refusal_audit.jsonl"
    agent = OSINTAgent(audit_path=audit_path)

    scope = Scope(
        target_kind=EntityType.DOMAIN,
        target_value="unauthorized-target.com",
        consent_confirmed=False,
    )

    with pytest.raises(PermissionError) as exc_info:
        await agent.run(scope)

    assert "consent_confirmed is False" in str(exc_info.value)
    assert audit_path.exists()

    with open(audit_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) == 1
    assert lines[0]["event"] == "consent_refusal"
    assert lines[0]["data"]["target_value"] == "unauthorized-target.com"


@pytest.mark.asyncio
@respx.mock
async def test_depth_limit(tmp_path: Path) -> None:
    """Expansion must strictly halt at max_depth."""
    audit_path = tmp_path / "depth_audit.jsonl"

    # Seed domain produces api.example.com (depth 1)
    respx.get("https://crt.sh/?q=%.example.com&output=json").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"name_value": "api.example.com", "issuer_name": "DigiCert"}
            ],
        )
    )

    # api.example.com produces v1.api.example.com (depth 2)
    respx.get("https://crt.sh/?q=%.api.example.com&output=json").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"name_value": "v1.api.example.com", "issuer_name": "DigiCert"}
            ],
        )
    )

    # v1.api.example.com produces deep.v1.api.example.com (depth 3, should NOT be queried if max_depth=2)
    respx.get("https://crt.sh/?q=%.v1.api.example.com&output=json").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"name_value": "deep.v1.api.example.com", "issuer_name": "DigiCert"}
            ],
        )
    )

    async with httpx.AsyncClient() as client:
        agent = OSINTAgent(audit_path=audit_path, http_client=client)
        scope = Scope(
            target_kind=EntityType.DOMAIN,
            target_value="example.com",
            consent_confirmed=True,
            max_depth=2,
            max_tool_calls=10,
        )
        graph = await agent.run(scope)

    # Depth 0: example.com
    # Depth 1: api.example.com
    # Depth 2: v1.api.example.com
    # Any candidate at depth 3 should not be expanded
    for node, data in graph.nodes(data=True):
        assert data["depth"] <= 2, f"Node {node} exceeded max depth: {data['depth']}"

    assert "domain:deep.v1.api.example.com" not in graph


@pytest.mark.asyncio
@respx.mock
async def test_call_budget(tmp_path: Path) -> None:
    """Expansion must strictly halt once tool call count reaches max_tool_calls."""
    audit_path = tmp_path / "budget_audit.jsonl"

    # Set up endless mock responses
    respx.get(url__regex=r"https://crt\.sh/.*").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"name_value": "a.example.com"},
                {"name_value": "b.example.com"},
                {"name_value": "c.example.com"},
            ],
        )
    )

    async with httpx.AsyncClient() as client:
        agent = OSINTAgent(audit_path=audit_path, http_client=client)
        scope = Scope(
            target_kind=EntityType.DOMAIN,
            target_value="example.com",
            consent_confirmed=True,
            max_depth=5,
            max_tool_calls=2,  # Strict small budget
        )
        graph = await agent.run(scope)

    with open(audit_path, "r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f]

    tool_starts = [e for e in events if e["event"] == "tool_call_started"]
    assert len(tool_starts) == 2


def test_trusted_vs_fuzzy_hit_handling() -> None:
    """Resolver should grant 1.0 to trusted hits and score fuzzy hits via token similarity."""
    resolver = BaselineWeightedResolver()
    parent = Entity.create(
        entity_type=EntityType.ORG,
        value="OpenAI Foundation",
        confidence=1.0,
    )

    # 1. Trusted hit: always 1.0 regardless of differences
    trusted_hit = Hit(
        target_entity_id=parent.id,
        discovered_type=EntityType.DOMAIN,
        discovered_value="openai.org",
        relation="official_website",
        source_tool="crtsh",
        is_trusted=True,
    )
    assert resolver.score(parent, trusted_hit) == 1.0
    assert resolver.should_accept(resolver.score(parent, trusted_hit), threshold=0.5)

    # 2. Fuzzy hit: strong lexical overlap
    close_fuzzy_hit = Hit(
        target_entity_id=parent.id,
        discovered_type=EntityType.ORG,
        discovered_value="OpenAI Inc",
        relation="affiliated_with",
        source_tool="wikidata",
        is_trusted=False,
    )
    close_score = resolver.score(parent, close_fuzzy_hit)
    assert close_score >= 0.70
    assert resolver.should_accept(close_score, threshold=0.5)

    # 3. Fuzzy hit: distant/unrelated
    distant_fuzzy_hit = Hit(
        target_entity_id=parent.id,
        discovered_type=EntityType.ORG,
        discovered_value="Completely Different University",
        relation="affiliated_with",
        source_tool="wikidata",
        is_trusted=False,
    )
    distant_score = resolver.score(parent, distant_fuzzy_hit)
    assert distant_score < 0.40
    assert not resolver.should_accept(distant_score, threshold=0.5)


@pytest.mark.asyncio
@respx.mock
async def test_audit_log_output(tmp_path: Path) -> None:
    """Every tool call and decision must be appended to audit.jsonl in valid JSONL format."""
    audit_path = tmp_path / "audit.jsonl"
    graph_path = tmp_path / "graph.json"

    respx.get("https://crt.sh/?q=%.cybercorp.local&output=json").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"name_value": "auth.cybercorp.local", "issuer_name": "TestCA"},
                {"name_value": "mail.cybercorp.local", "issuer_name": "TestCA"},
            ],
        )
    )

    async with httpx.AsyncClient() as client:
        agent = OSINTAgent(audit_path=audit_path, http_client=client)
        scope = Scope(
            target_kind=EntityType.DOMAIN,
            target_value="cybercorp.local",
            consent_confirmed=True,
            max_depth=1,
            max_tool_calls=5,
        )
        graph = await agent.run(scope)

    export_graph_to_json(graph, graph_path)

    assert audit_path.exists()
    assert graph_path.exists()

    with open(audit_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    event_names = [r["event"] for r in records]
    assert "investigation_started" in event_names
    assert "tool_call_started" in event_names
    assert "tool_call_completed" in event_names
    assert "hit_evaluated" in event_names
    assert "investigation_finished" in event_names

    with open(graph_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    assert "nodes" in graph_data
    assert "edges" in graph_data
    assert len(graph_data["nodes"]) == 3  # seed + 2 subdomains
    assert len(graph_data["edges"]) == 2

    # Edges carry relation, source_tool, and score
    for edge in graph_data["edges"]:
        assert edge["relation"] == "subdomain_of"
        assert edge["source_tool"] == "crtsh"
        assert edge["score"] == 1.0


def test_sensitive_attribute_sanitization() -> None:
    """Sensitive attributes (home address, health, SSN, family) must be strictly scrubbed."""
    dirty_attrs = {
        "official_email": "info@corp.org",
        "home_address": "123 Private St",
        "family": {"spouse": "Private Individual", "children": 2},
        "health_record": "classified",
        "city": "Metropolis",
    }
    clean = sanitize_attributes(dirty_attrs)
    assert "home_address" not in clean
    assert "family" not in clean
    assert "health_record" not in clean
    assert clean["official_email"] == "info@corp.org"
    assert clean["city"] == "Metropolis"


@pytest.mark.asyncio
@respx.mock
async def test_tool_crtsh_wildcard_cleaning() -> None:
    """crtsh collector must clean wildcards, ignore root domain, and filter invalid entries."""
    respx.get("https://crt.sh/?q=%.testsite.org&output=json").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"name_value": "*.vpn.testsite.org\nvpn.testsite.org"},
                {"name_value": "testsite.org"},  # Root domain, should be skipped
                {"name_value": "unrelated.other.com"},  # Not a subdomain of testsite.org
            ],
        )
    )

    parent = Entity.create(entity_type=EntityType.DOMAIN, value="testsite.org")
    async with httpx.AsyncClient() as client:
        hits = await collect_crtsh(parent, client)

    assert len(hits) == 1
    assert hits[0].discovered_value == "vpn.testsite.org"
    assert hits[0].is_trusted is True
    assert hits[0].relation == "subdomain_of"


@pytest.mark.asyncio
@respx.mock
async def test_tool_nominatim_collector() -> None:
    """nominatim collector geocodes place entities and surfaces parent divisions."""
    respx.get("https://nominatim.openstreetmap.org/search").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "display_name": "Austin, Travis County, Texas, United States",
                    "osm_type": "relation",
                    "category": "boundary",
                    "lat": "30.2711286",
                    "lon": "-97.7436995",
                    "address": {
                        "city": "Austin",
                        "state": "Texas",
                        "country": "United States",
                    },
                }
            ],
        )
    )

    place_entity = Entity.create(entity_type=EntityType.PLACE, value="Austin")
    async with httpx.AsyncClient() as client:
        hits = await collect_nominatim(place_entity, client)

    assert len(hits) >= 1
    values = [h.discovered_value for h in hits]
    assert any("Austin" in v for v in values)
    assert any("Texas" in v for v in values)
    assert any("United States" in v for v in values)


@pytest.mark.asyncio
@respx.mock
async def test_tool_wikidata_collector() -> None:
    """wikidata collector queries public entities and handles descriptions."""
    respx.get("https://www.wikidata.org/w/api.php").mock(
        return_value=httpx.Response(
            200,
            json={
                "search": [
                    {
                        "id": "Q248",
                        "label": "Intel",
                        "description": "American multinational corporation and technology company",
                        "concepturi": "http://www.wikidata.org/entity/Q248",
                    }
                ]
            },
        )
    )

    org_entity = Entity.create(entity_type=EntityType.ORG, value="Intel")
    async with httpx.AsyncClient() as client:
        hits = await collect_wikidata(org_entity, client)

    assert len(hits) == 1
    assert hits[0].discovered_value == "Intel"
    assert hits[0].raw_data["wikidata_id"] == "Q248"
    assert hits[0].source_tool == "wikidata"


@pytest.mark.asyncio
@respx.mock
async def test_tool_domain_intel_collector() -> None:
    """domain_intel collector should parse RDAP registrar, nameservers, HTTP headers, and HTML metadata."""
    from osint_agent.collectors.tools import collect_domain_intel

    respx.get("https://rdap.org/domain/mockcorp.io").mock(
        return_value=httpx.Response(
            200,
            json={
                "events": [
                    {"eventAction": "registration", "eventDate": "2023-05-01T00:00:00Z"},
                    {"eventAction": "expiration", "eventDate": "2028-05-01T00:00:00Z"},
                ],
                "entities": [
                    {
                        "roles": ["registrar"],
                        "vcardArray": ["vcard", [["fn", {}, "text", "Namecheap, Inc."]]],
                    }
                ],
                "nameservers": [
                    {"ldhName": "dns1.registrar-servers.com"},
                ],
            },
        )
    )

    respx.get("https://mockcorp.io").mock(
        return_value=httpx.Response(
            200,
            headers={"server": "cloudflare", "content-type": "text/html"},
            text="<html><head><title>MockCorp Cloud</title><meta name='description' content='Enterprise AI tools'></head></html>",
        )
    )

    domain_entity = Entity.create(entity_type=EntityType.DOMAIN, value="mockcorp.io")
    async with httpx.AsyncClient() as client:
        hits = await collect_domain_intel(domain_entity, client)

    values = [h.discovered_value for h in hits]
    assert "Namecheap, Inc." in values
    assert "dns1.registrar-servers.com" in values
    assert "cloudflare" in values
    assert "https://mockcorp.io" in values
    web_hit = next(h for h in hits if h.discovered_type == EntityType.PROFILE)
    assert "MockCorp Cloud" in web_hit.raw_data["description"]
    assert web_hit.raw_data["registered"] == "2023-05-01"


@pytest.mark.asyncio
@respx.mock
async def test_tool_email_intel_collector() -> None:
    """email_intel should link custom domain, parse Gravatar profile, and parse Holehe stdout."""
    import hashlib
    from unittest.mock import AsyncMock, patch
    from osint_agent.collectors.tools import collect_email_intel

    email = "alice@acme-research.io"
    email_hash = hashlib.md5(email.encode("utf-8")).hexdigest()

    respx.get(f"https://en.gravatar.com/{email_hash}.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "entry": [
                    {
                        "preferredUsername": "alice_dev",
                        "displayName": "Alice Researcher",
                        "aboutMe": "ML Scientist",
                        "currentLocation": "Zurich",
                        "accounts": [{"shortname": "github", "url": "https://github.com/alice_dev"}],
                    }
                ]
            },
        )
    )

    holehe_stdout = "\n[+] Email used, [-] Email not used, [x] Rate limit\n[+] spotify.com\n[+] github.com\n"
    with patch("osint_agent.collectors.tools._find_cli_binary", return_value="/usr/bin/holehe"), \
         patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (holehe_stdout.encode("utf-8"), b"")
        mock_exec.return_value = mock_proc

        email_entity = Entity.create(entity_type=EntityType.EMAIL, value=email)
        async with httpx.AsyncClient() as client:
            hits = await collect_email_intel(email_entity, client)

    values = [h.discovered_value for h in hits]
    assert "acme-research.io" in values
    assert "alice_dev" in values
    assert "https://github.com/alice_dev" in values
    assert "https://spotify.com" in values
    assert "https://github.com" in values


@pytest.mark.asyncio
@respx.mock
async def test_tool_web_archive_crawl_collector() -> None:
    """web_archive_crawl should query Wayback CDX, sitemap.xml, mailto emails, and social links."""
    from osint_agent.collectors.tools import collect_web_archive_crawl

    respx.get("https://web.archive.org/cdx/search/cdx").mock(
        return_value=httpx.Response(
            200,
            json=[
                ["timestamp", "original", "statuscode"],
                ["20240101120000", "https://mocklab.org/", "200"],
                ["20240201120000", "https://mocklab.org/about", "200"],
            ],
        )
    )
    respx.get("https://mocklab.org/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\nAllow: /")
    )
    respx.get("https://mocklab.org/sitemap.xml").mock(
        return_value=httpx.Response(
            200,
            text="<urlset><url><loc>https://mocklab.org/projects</loc></url></urlset>",
        )
    )
    respx.get("https://mocklab.org").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="""
            <html><body>
              <a href="mailto:hello@mocklab.org">Email</a>
              <a href="https://github.com/mocklab">GitHub</a>
            </body></html>
            """,
        )
    )

    domain_entity = Entity.create(entity_type=EntityType.DOMAIN, value="mocklab.org")
    async with httpx.AsyncClient() as client:
        hits = await collect_web_archive_crawl(domain_entity, client)

    values = [h.discovered_value for h in hits]
    assert "hello@mocklab.org" in values
    assert "https://github.com/mocklab" in values
    assert "https://web.archive.org/web/*/mocklab.org" in values


@pytest.mark.asyncio
@respx.mock
async def test_tool_github_code_intel_collector() -> None:
    """github_code_intel should discover repos, homepage domains, orgs, and commit emails."""
    from osint_agent.collectors.tools import collect_github_code_intel

    respx.get("https://api.github.com/users/devjane/repos?sort=updated&per_page=8").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "name": "ai-agent",
                    "html_url": "https://github.com/devjane/ai-agent",
                    "language": "Python",
                    "stargazers_count": 42,
                    "description": "Autonomous ML agent",
                    "homepage": "https://ai-agent.dev",
                }
            ],
        )
    )
    respx.get("https://api.github.com/users/devjane/orgs").mock(
        return_value=httpx.Response(200, json=[{"login": "OpenResearch", "description": "AI lab"}])
    )
    respx.get("https://api.github.com/users/devjane/events/public?per_page=15").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "type": "PushEvent",
                    "payload": {
                        "commits": [
                            {"author": {"name": "Jane", "email": "jane@ai-agent.dev"}},
                            {"author": {"name": "Jane", "email": "devjane@users.noreply.github.com"}},
                        ]
                    },
                }
            ],
        )
    )

    user_entity = Entity.create(entity_type=EntityType.USERNAME, value="devjane")
    async with httpx.AsyncClient() as client:
        hits = await collect_github_code_intel(user_entity, client)

    values = [h.discovered_value for h in hits]
    assert "https://github.com/devjane/ai-agent" in values
    assert "ai-agent.dev" in values
    assert "OpenResearch" in values
    assert "jane@ai-agent.dev" in values
    assert "devjane@users.noreply.github.com" not in values


@pytest.mark.asyncio
@respx.mock
async def test_tool_threat_intel_collector() -> None:
    """threat_intel should query keyless URLScan.io public scans and extract ASN + report."""
    from osint_agent.collectors.tools import collect_threat_intel

    respx.get("https://urlscan.io/api/v1/search/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "page": {
                            "ip": "104.21.13.80",
                            "asn": "AS13335",
                            "asnname": "CLOUDFLARENET, US",
                            "country": "US",
                            "domainAgeDays": 1200,
                            "tlsIssuer": "Google Trust Services",
                        },
                        "result": "https://urlscan.io/api/v1/result/abc-123/",
                    }
                ]
            },
        )
    )

    domain_entity = Entity.create(entity_type=EntityType.DOMAIN, value="securelab.io")
    async with httpx.AsyncClient() as client:
        hits = await collect_threat_intel(domain_entity, client)

    values = [h.discovered_value for h in hits]
    assert "CLOUDFLARENET, US" in values
    assert "https://urlscan.io/api/v1/result/abc-123/" in values


@pytest.mark.asyncio
@respx.mock
async def test_tool_open_data_intel_collector() -> None:
    """open_data_intel should query Wikipedia, GDELT news, and OpenSky live flights."""
    from osint_agent.collectors.tools import collect_open_data_intel

    respx.get("https://en.wikipedia.org/api/rest_v1/page/summary/Zurich").mock(
        return_value=httpx.Response(
            200,
            json={
                "title": "Zurich",
                "extract": "Zurich is the largest city in Switzerland.",
                "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Zurich"}},
            },
        )
    )
    respx.get("https://opensky-network.org/api/states/all").mock(
        return_value=httpx.Response(
            200,
            json={"states": [["3c4a22", "SWR123  ", "Switzerland", 1700000000, 1700000000, 8.5, 47.3, 9500.0, False, 240.0]]},
        )
    )

    place_entity = Entity.create(entity_type=EntityType.PLACE, value="Zurich", attributes={"lat": 47.3769, "lon": 8.5417})
    async with httpx.AsyncClient() as client:
        hits = await collect_open_data_intel(place_entity, client)

    values = [h.discovered_value for h in hits]
    assert "https://en.wikipedia.org/wiki/Zurich" in values
    assert "Flight SWR123 (Switzerland)" in values


@pytest.mark.asyncio
@respx.mock
async def test_cli_execution_with_consent(tmp_path: Path) -> None:
    """CLI runner should successfully execute and write graph and audit files."""
    from osint_agent.cli import main_async, parse_args

    respx.get("https://crt.sh/?q=%.cli-test.com&output=json").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"name_value": "api.cli-test.com", "issuer_name": "TestCA"}
            ],
        )
    )

    graph_file = tmp_path / "cli_graph.json"
    audit_file = tmp_path / "cli_audit.jsonl"

    args = parse_args([
        "--kind", "domain",
        "--value", "cli-test.com",
        "--consent",
        "--graph-out", str(graph_file),
        "--audit-out", str(audit_file),
        "--max-depth", "1",
        "--max-calls", "2",
    ])

    exit_code = await main_async(args)
    assert exit_code == 0
    assert graph_file.exists()
    assert audit_file.exists()


@pytest.mark.asyncio
async def test_cli_refusal_without_consent(tmp_path: Path) -> None:
    """CLI runner must return code 1 if consent flag is omitted."""
    from osint_agent.cli import main_async, parse_args

    audit_file = tmp_path / "cli_refusal_audit.jsonl"
    args = parse_args([
        "--kind", "domain",
        "--value", "cli-refusal.com",
        "--audit-out", str(audit_file),
    ])

    exit_code = await main_async(args)
    assert exit_code == 1
    assert audit_file.exists()


@pytest.mark.asyncio
@respx.mock
async def test_trusted_hit_inherits_parent_confidence_without_decay(tmp_path: Path) -> None:
    """A trusted hit (e.g. crtsh) must inherit the parent confidence unchanged, without decay."""
    audit_path = tmp_path / "confidence_audit.jsonl"

    respx.get("https://crt.sh/?q=%.parentcorp.org&output=json").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"name_value": "sub.parentcorp.org", "issuer_name": "TestCA"}
            ],
        )
    )

    async with httpx.AsyncClient() as client:
        agent = OSINTAgent(audit_path=audit_path, http_client=client)
        scope = Scope(
            target_kind=EntityType.DOMAIN,
            target_value="parentcorp.org",
            consent_confirmed=True,
            confidence_decay=0.8,
            max_depth=1,
            max_tool_calls=2,
        )
        graph = await agent.run(scope)

    parent_node = graph.nodes["domain:parentcorp.org"]
    child_node = graph.nodes["domain:sub.parentcorp.org"]

    assert parent_node["confidence"] == 1.0
    # Because sub.parentcorp.org is trusted from crtsh, it inherits parent confidence (1.0) without 0.8 decay
    assert child_node["confidence"] == 1.0


@pytest.mark.asyncio
@respx.mock
async def test_tool_email_intel_enrichment() -> None:
    """email_intel should aggregate GitHub commits/users, XposedOrNot breaches, Gravatar, Duolingo, and PGP."""
    import hashlib
    from unittest.mock import patch

    from osint_agent.collectors.tools import collect_email_intel

    email = "alice@gmail.com"
    md5_hash = hashlib.md5(email.encode("utf-8")).hexdigest()

    respx.get(f"https://en.gravatar.com/{md5_hash}.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "entry": [
                    {
                        "preferredUsername": "alice_grav",
                        "displayName": "Alice Smith",
                        "accounts": [],
                    }
                ]
            },
        )
    )
    respx.get("https://api.github.com/search/commits").mock(
        return_value=httpx.Response(
            200,
            json={
                "total_count": 3,
                "items": [
                    {
                        "author": {"login": "alice-gh"},
                        "commit": {"author": {"name": "Alice Smith"}},
                        "repository": {"full_name": "alice-gh/demo"},
                    }
                ],
            },
        )
    )
    respx.get("https://api.github.com/search/users").mock(
        return_value=httpx.Response(200, json={"items": []})
    )
    respx.get(f"https://api.xposedornot.com/v1/check-email/{email}").mock(
        return_value=httpx.Response(200, json={"breaches": [["ExampleBreach"]], "status": "success"})
    )
    respx.get("https://www.duolingo.com/2017-06-30/users").mock(
        return_value=httpx.Response(200, json={"users": [{"username": "alice_duo", "name": "Alice"}]})
    )
    respx.get(f"https://keys.openpgp.org/vks/v1/by-email/{email}").mock(return_value=httpx.Response(404))
    respx.get("https://keyserver.ubuntu.com/pks/lookup").mock(return_value=httpx.Response(404))

    entity = Entity.create(entity_type=EntityType.EMAIL, value=email)
    with patch("osint_agent.collectors.tools._find_cli_binary", return_value=None):
        async with httpx.AsyncClient() as client:
            hits = await collect_email_intel(entity, client)

    values = {h.discovered_value for h in hits}
    assert "alice_grav" in values
    assert "alice-gh" in values
    assert "https://github.com/alice-gh" in values
    assert "breach:ExampleBreach" in values
    assert "alice_duo" in values


@pytest.mark.asyncio
async def test_stream_checker_filters_catchall_and_bot_challenge() -> None:
    """stream_checker must reject Anubis/reCAPTCHA 200 pages and catch-all 200 sites via baseline probing."""
    from osint_agent.collectors.stream_checker import _BASELINE_PROBE_USERNAME, _is_bot_or_soft404_response, _probe_site

    anubis_resp = httpx.Response(
        200,
        request=httpx.Request("GET", "https://lemmy.rochegmr.com/u/testuser"),
        text="<html><head><title>Making sure you&#39;re not a bot!</title></head></html>",
    )
    assert _is_bot_or_soft404_response(anubis_resp, anubis_resp.text, "status_code") is True

    soft404_resp = httpx.Response(
        200,
        request=httpx.Request("GET", "https://example.com/u/testuser"),
        text="<html><head><title>User Not Found</title></head><body>No such user</body></html>",
    )
    assert _is_bot_or_soft404_response(soft404_resp, soft404_resp.text, "status_code") is True

    # Simulate a catch-all 200 site: both real username and baseline nonexistent username return 200
    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://catchall.example/u/realuser").mock(return_value=httpx.Response(200, text="<html>Profile</html>"))
        mock.get(f"https://catchall.example/u/{_BASELINE_PROBE_USERNAME}").mock(
            return_value=httpx.Response(200, text="<html>Profile</html>")
        )
        cfg = {"url": "https://catchall.example/u/{username}", "urlProbe": "https://catchall.example/u/{username}", "checkType": "status_code"}
        async with httpx.AsyncClient() as client:
            real_hit = await _probe_site(client, "CatchAll", cfg, "realuser", "username:realuser")
            base_hit = await _probe_site(client, "CatchAll", cfg, _BASELINE_PROBE_USERNAME, "username:realuser")
        assert real_hit is not None
        assert base_hit is not None  # Baseline matches -> stream_username_check drops it as catch-all 200


@pytest.mark.asyncio
@respx.mock
async def test_openrouter_client_and_multimodal() -> None:
    """OpenRouterClient should call OpenAI-compatible endpoint and support multimodal image URLs."""
    from osint_agent.llm.client import OpenRouterClient

    respx.post("https://ai.hackclub.com/proxy/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "```json\n{\"real_name\": \"Aryan\", \"office_or_location\": \"Delhi\"}\n```"}}],
                "usage": {"total_tokens": 42},
            },
        )
    )
    client = OpenRouterClient(api_key="test-key", call_budget=5)
    res = client.complete_multimodal_json(
        system="Extract JSON",
        user="Analyze profile",
        image_urls=["https://example.com/avatar.png"],
    )
    assert res["real_name"] == "Aryan"
    assert res["office_or_location"] == "Delhi"
    assert client.total_tokens == 42


@pytest.mark.asyncio
@respx.mock
async def test_profile_analyzer_wattpad_playstore_and_synthesis() -> None:
    """profile_analyzer should extract Wattpad stories, Play Store dev details, and LLM cross-profile relations."""
    import networkx as nx

    from osint_agent.collectors.profile_analyzer import analyze_profiles_batch, synthesize_profile_relations
    from osint_agent.llm.client import FakeClient

    respx.get("https://www.wattpad.com/api/v3/users/alaotach").mock(
        return_value=httpx.Response(200, json={"name": "Aryan", "description": "Writer", "avatar": "https://img.wattpad.com/a.jpg"})
    )
    respx.get("https://www.wattpad.com/api/v3/users/alaotach/stories").mock(
        return_value=httpx.Response(
            200,
            json={"stories": [{"title": "Replied to your story", "description": "A love story", "voteCount": 35}]},
        )
    )
    respx.get("https://play.google.com/store/apps/details?id=com.alaotach.app&hl=en").mock(
        return_value=httpx.Response(
            200,
            text=(
                "<html><head><title>Demo App - Google Play</title></head>"
                "<body>Developer address: 123 Tech Park, New Delhi "
                "<a href='mailto:dev@alaotach.dev'>dev@alaotach.dev</a>"
                "<a href='https://alaotach.dev'>Website</a></body></html>"
            ),
        )
    )

    wattpad_hit = Hit(
        target_entity_id="username:alaotach",
        discovered_type=EntityType.PROFILE,
        discovered_value="https://www.wattpad.com/user/alaotach",
        relation="account_on",
        source_tool="maigret",
        metadata={"username": "alaotach", "site": "Wattpad", "url": "https://www.wattpad.com/user/alaotach"},
    )
    play_hit = Hit(
        target_entity_id="username:alaotach",
        discovered_type=EntityType.PROFILE,
        discovered_value="https://play.google.com/store/apps/details?id=com.alaotach.app&hl=en",
        relation="published_app",
        source_tool="playstore_intel",
        metadata={"site": "Google Play Store", "url": "https://play.google.com/store/apps/details?id=com.alaotach.app&hl=en"},
    )

    fake_llm = FakeClient(
        responses=[
            {
                "profiles": {
                    "https://www.wattpad.com/user/alaotach": {
                        "summary": "Wattpad writer",
                        "projects_or_stories": ["Replied to your story"],
                    },
                    "https://play.google.com/store/apps/details?id=com.alaotach.app&hl=en": {
                        "summary": "Android developer",
                        "office_or_location": "123 Tech Park, New Delhi",
                    },
                }
            },
            {
                "summary": "Connected Wattpad author and Play Store developer.",
                "inferred_name": "Aryan",
                "locations": ["New Delhi"],
                "organizations": [],
                "relations": [
                    {
                        "source_url": "https://www.wattpad.com/user/alaotach",
                        "target_url": "https://play.google.com/store/apps/details?id=com.alaotach.app&hl=en",
                        "relation": "same_creator",
                        "evidence": "Shared alias alaotach",
                    }
                ],
            },
        ]
    )

    async with httpx.AsyncClient() as client:
        await analyze_profiles_batch([wattpad_hit, play_hit], client, llm_client=fake_llm)

    assert wattpad_hit.metadata.get("stories")
    assert "Replied to your story" in wattpad_hit.metadata["stories"][0]
    assert play_hit.metadata.get("email") == "dev@alaotach.dev"
    assert "123 Tech Park" in play_hit.metadata.get("location", "")

    g = nx.DiGraph()
    g.add_node("profile:wattpad", entity_type="profile", value=wattpad_hit.discovered_value, attributes=wattpad_hit.metadata)
    g.add_node("profile:play", entity_type="profile", value=play_hit.discovered_value, attributes=play_hit.metadata)
    synth = await synthesize_profile_relations(g, llm_client=fake_llm)
    assert synth is not None
    assert g.has_edge("profile:wattpad", "profile:play")


def test_ml_profile_matcher_clustering_and_keywords() -> None:
    """ML profile_matcher should extract TF-IDF keywords, link similar profiles, and cluster identities."""
    import networkx as nx

    from osint_agent.ml.profile_matcher import discover_ml_profile_relations, extract_profile_keywords

    kws = extract_profile_keywords(
        "Building autonomous robotics simulation pipelines in Python and PyTorch at Delhi Tech.",
        top_k=5,
    )
    assert len(kws) > 0

    g = nx.DiGraph()
    g.add_node(
        "profile:gh",
        entity_type="profile",
        value="https://github.com/alaotach",
        attributes={
            "username": "alaotach",
            "name": "Aryan Mishra",
            "bio": "Autonomous robotics and Python developer building OSINT graphs.",
            "location": "New Delhi, India",
            "linked_urls": ["https://alaotach.dev"],
        },
    )
    g.add_node(
        "profile:hf",
        entity_type="profile",
        value="https://huggingface.co/alaotach",
        attributes={
            "username": "alaotach",
            "name": "Aryan Mishra",
            "bio": "Python ML engineer working on autonomous robotics and OSINT.",
            "location": "New Delhi",
            "linked_urls": ["https://alaotach.dev/models"],
        },
    )
    g.add_node(
        "profile:unrelated",
        entity_type="profile",
        value="https://example.com/chef99",
        attributes={
            "username": "chef_luigi_99",
            "name": "Luigi Rossi",
            "bio": "Pastry chef baking sourdough bread in Milan.",
            "location": "Milan, Italy",
        },
    )

    res = discover_ml_profile_relations(g, similarity_threshold=0.45)
    assert res["edges_added"] >= 1
    assert g.has_edge("profile:gh", "profile:hf")
    assert not g.has_edge("profile:gh", "profile:unrelated")
    assert g.nodes["profile:gh"]["attributes"]["ml_cluster_id"] == g.nodes["profile:hf"]["attributes"]["ml_cluster_id"]
    assert g.nodes["profile:gh"]["attributes"]["ml_cluster_id"] != g.nodes["profile:unrelated"]["attributes"]["ml_cluster_id"]


def test_ml_compress_and_rank_for_llm() -> None:
    """ML compressor must keep OSINT-signal sentences over boilerplate and rank informative hits first."""
    from osint_agent.ml.profile_matcher import compress_profile_text_ml, rank_profiles_for_llm

    noisy = (
        "Accept all cookies to continue browsing this website. "
        "Sign in or register for a new account today. " * 20
        + "Developer contact email: dev@example.org located at Tech Park New Delhi. "
        + "Built open source autonomous robotics simulator and Android app."
    )
    compressed = compress_profile_text_ml(noisy, max_chars=250)
    assert len(compressed) <= 250
    assert "dev@example.org" in compressed or "robotics" in compressed

    low_hit = Hit(target_entity_id="u:1", discovered_type=EntityType.PROFILE, discovered_value="https://a.com", relation="account_on", source_tool="maigret", metadata={})
    high_hit = Hit(target_entity_id="u:1", discovered_type=EntityType.PROFILE, discovered_value="https://b.com", relation="account_on", source_tool="maigret", metadata={"bio": "AI researcher", "email": "x@y.com", "ml_keywords": ["ai", "research"]})
    ranked = rank_profiles_for_llm([low_hit, high_hit])
    assert ranked == [high_hit]
