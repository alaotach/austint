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


@pytest.mark.asyncio
@respx.mock
async def test_profile_analyzer_github_repo_deep_scrape() -> None:
    """GitHub repo PROFILE hits should be deeply inspected via repo API, README, commits, and manifests."""
    from osint_agent.collectors.profile_analyzer import analyze_profile_hit

    respx.get("https://api.github.com/repos/alaotach/eli-new").mock(
        return_value=httpx.Response(
            200,
            json={
                "name": "eli-new",
                "description": "AI assistant project",
                "language": "Python",
                "stargazers_count": 5,
                "topics": ["ai", "osint"],
                "homepage": "https://eli.example.dev",
            },
        )
    )
    respx.get("https://api.github.com/repos/alaotach/eli-new/readme").mock(
        return_value=httpx.Response(
            200,
            text="# Eli New\nDocs at https://docs.eli.example.dev\nContact: team@eli.example.dev",
        )
    )
    respx.get("https://api.github.com/repos/alaotach/eli-new/commits?per_page=15").mock(
        return_value=httpx.Response(
            200,
            json=[{"commit": {"author": {"email": "author@eli.example.dev"}}}],
        )
    )
    respx.get("https://api.github.com/repos/alaotach/eli-new/contents").mock(
        return_value=httpx.Response(200, json=[])
    )
    respx.get("https://api.github.com/repos/alaotach/eli-new/git/trees/HEAD?recursive=1").mock(
        return_value=httpx.Response(
            200,
            json={
                "tree": [
                    {"path": "src/config.py", "type": "blob"},
                    {"path": "app/build.gradle", "type": "blob"},
                ]
            },
        )
    )
    respx.get("https://raw.githubusercontent.com/alaotach/eli-new/HEAD/src/config.py").mock(
        return_value=httpx.Response(200, text='SUPPORT_EMAIL = "code@eli.example.dev"\nAPI_BASE = "https://api.eli.example.dev"\n')
    )
    respx.get("https://raw.githubusercontent.com/alaotach/eli-new/HEAD/app/build.gradle").mock(
        return_value=httpx.Response(200, text='defaultConfig { applicationId "com.alaotach.elinew" }\n')
    )

    repo_hit = Hit(
        target_entity_id="username:alaotach",
        discovered_type=EntityType.PROFILE,
        discovered_value="https://github.com/alaotach/eli-new",
        relation="authored_repo",
        source_tool="github_code_intel",
        metadata={"url": "https://github.com/alaotach/eli-new"},
    )
    async with httpx.AsyncClient() as client:
        info = await analyze_profile_hit(repo_hit, client)

    assert info["website"] == "https://eli.example.dev"
    assert repo_hit.metadata["email"] == "team@eli.example.dev"
    assert "code@eli.example.dev" in info.get("contact_emails", [])
    assert "https://docs.eli.example.dev" in repo_hit.metadata["linked_urls"]
    assert "https://api.eli.example.dev" in repo_hit.metadata["linked_urls"]
    assert "https://play.google.com/store/apps/details?id=com.alaotach.elinew&hl=en" in repo_hit.metadata["linked_urls"]
    assert repo_hit.metadata.get("ml_keywords")


# ---------------------------------------------------------------------------
# Breach intelligence tests (XposedOrNot analytics + Hudson Rock Cavalier)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_tool_email_intel_breach_analytics() -> None:
    """email_intel should parse XposedOrNot breach-analytics and emit a breach_analytics Hit."""
    import hashlib
    from unittest.mock import AsyncMock, patch
    from osint_agent.collectors.tools import collect_email_intel

    email = "breach_test@example.com"
    email_hash = hashlib.md5(email.encode("utf-8")).hexdigest()

    # Gravatar — 404 so no Gravatar hits
    respx.get(f"https://en.gravatar.com/{email_hash}.json").mock(
        return_value=httpx.Response(404)
    )
    # GitHub search — no results
    respx.get("https://api.github.com/search/commits").mock(
        return_value=httpx.Response(200, json={"items": [], "total_count": 0})
    )
    respx.get("https://api.github.com/search/users").mock(
        return_value=httpx.Response(200, json={"items": [], "total_count": 0})
    )
    # XposedOrNot check-email — returns two breaches
    respx.get(f"https://api.xposedornot.com/v1/check-email/{email}").mock(
        return_value=httpx.Response(
            200,
            json={"breaches": [["TestBreach2023", "AnotherLeak2022"]]},
        )
    )
    # XposedOrNot breach-analytics — returns detailed metrics
    respx.get("https://api.xposedornot.com/v1/breach-analytics").mock(
        return_value=httpx.Response(
            200,
            json={
                "BreachMetrics": {
                    "risk": [{"risk_label": "High", "risk_score": 7}],
                    "passwords_strength": [
                        {"PlainText": 1, "EasyToCrack": 2, "StrongHash": 3}
                    ],
                    "xposed_data": [
                        {
                            "children": [
                                {
                                    "children": [
                                        {"name": "data_emails", "value": 1},
                                        {"name": "data_passwords", "value": 1},
                                        {"name": "data_names", "value": 1},
                                    ]
                                }
                            ]
                        }
                    ],
                }
            },
        )
    )
    # Duolingo — no user
    respx.get("https://www.duolingo.com/2017-06-30/users").mock(
        return_value=httpx.Response(200, json={"users": []})
    )
    # PGP — no key
    respx.get("https://keys.openpgp.org/vks/v1/by-email/breach_test%40example.com").mock(
        return_value=httpx.Response(404)
    )
    respx.get("https://keyserver.ubuntu.com/pks/lookup").mock(
        return_value=httpx.Response(404)
    )
    # Hudson Rock by email — no stealers
    respx.get("https://cavalier.hudsonrock.com/api/json/v2/osint-tools/search-by-email").mock(
        return_value=httpx.Response(200, json={"stealers": []})
    )

    email_entity = Entity.create(entity_type=EntityType.EMAIL, value=email)
    with patch("osint_agent.collectors.tools._find_cli_binary", return_value=None):
        async with httpx.AsyncClient() as client:
            hits = await collect_email_intel(email_entity, client)

    relations = [h.relation for h in hits]
    # Should have individual breach names
    assert "breach_exposure" in relations
    # Should have detailed analytics hit
    assert "breach_analytics" in relations

    analytics_hit = next(h for h in hits if h.relation == "breach_analytics")
    assert analytics_hit.raw_data["risk_label"] == "High"
    assert analytics_hit.raw_data["risk_score"] == 7
    assert analytics_hit.raw_data["plaintext_passwords"] == 1
    assert analytics_hit.raw_data["easy_crack_hashes"] == 2
    assert analytics_hit.raw_data["strong_hashes"] == 3
    assert "emails" in analytics_hit.raw_data["exposed_data_types"]
    assert "passwords" in analytics_hit.raw_data["exposed_data_types"]


@pytest.mark.asyncio
@respx.mock
async def test_tool_breach_intel_hudson_rock_username() -> None:
    """breach_intel should emit infostealer_compromise Hits from Hudson Rock by username."""
    from osint_agent.collectors.tools import collect_breach_intel

    username = "testuser_osint"

    respx.get("https://cavalier.hudsonrock.com/api/json/v2/osint-tools/search-by-username").mock(
        return_value=httpx.Response(
            200,
            json={
                "stealers": [
                    {
                        "computer_name": "DESKTOP-XYZ123 (TestUser)",
                        "operating_system": "Windows 11 (Build 22631)",
                        "stealer_family": "RedLine",
                        "date_compromised": "2024-03-15T12:00:00.000Z",
                        "total_user_services": 42,
                        "ip": "1.2.3.*",
                        "top_logins": ["testuser@gmail.com", "testuser@outlook.com"],
                        "top_passwords": ["p*****d", "s*****t"],
                    }
                ]
            },
        )
    )

    username_entity = Entity.create(entity_type=EntityType.USERNAME, value=username)
    async with httpx.AsyncClient() as client:
        hits = await collect_breach_intel(username_entity, client)

    assert len(hits) == 1
    hit = hits[0]
    assert hit.relation == "infostealer_compromise"
    assert hit.source_tool == "breach_intel"
    assert hit.raw_data["computer_name"] == "DESKTOP-XYZ123 (TestUser)"
    assert hit.raw_data["operating_system"] == "Windows 11 (Build 22631)"
    assert hit.raw_data["stealer_type"] == "RedLine"
    assert hit.raw_data["total_user_services"] == 42
    assert hit.raw_data["ip"] == "1.2.3.*"
    # Verify passwords are stored in redacted form only
    assert all("*" in p for p in hit.raw_data["top_passwords_redacted"])
    assert "infostealer" in hit.discovered_value
    assert hit.metadata["site"] == "Hudson Rock Cavalier"


@pytest.mark.asyncio
@respx.mock
async def test_tool_breach_intel_hudson_rock_no_stealers() -> None:
    """breach_intel should return empty list when Hudson Rock finds no stealers."""
    from osint_agent.collectors.tools import collect_breach_intel

    respx.get("https://cavalier.hudsonrock.com/api/json/v2/osint-tools/search-by-username").mock(
        return_value=httpx.Response(200, json={"stealers": []})
    )

    entity = Entity.create(entity_type=EntityType.USERNAME, value="cleanuser_notcompromised")
    async with httpx.AsyncClient() as client:
        hits = await collect_breach_intel(entity, client)

    assert hits == []


@pytest.mark.asyncio
@respx.mock
async def test_tool_email_intel_hudson_rock_infostealer() -> None:
    """email_intel should emit infostealer_compromise Hit when Hudson Rock finds stealer data by email."""
    import hashlib
    from unittest.mock import patch
    from osint_agent.collectors.tools import collect_email_intel

    email = "stealer_victim@test.com"
    email_hash = hashlib.md5(email.encode("utf-8")).hexdigest()

    respx.get(f"https://en.gravatar.com/{email_hash}.json").mock(return_value=httpx.Response(404))
    respx.get("https://api.github.com/search/commits").mock(
        return_value=httpx.Response(200, json={"items": [], "total_count": 0})
    )
    respx.get("https://api.github.com/search/users").mock(
        return_value=httpx.Response(200, json={"items": [], "total_count": 0})
    )
    respx.get(f"https://api.xposedornot.com/v1/check-email/stealer_victim%40test.com").mock(
        return_value=httpx.Response(404)
    )
    respx.get("https://api.xposedornot.com/v1/breach-analytics").mock(
        return_value=httpx.Response(404)
    )
    respx.get("https://www.duolingo.com/2017-06-30/users").mock(
        return_value=httpx.Response(200, json={"users": []})
    )
    respx.get("https://keys.openpgp.org/vks/v1/by-email/stealer_victim%40test.com").mock(
        return_value=httpx.Response(404)
    )
    respx.get("https://keyserver.ubuntu.com/pks/lookup").mock(return_value=httpx.Response(404))

    # Hudson Rock by email — returns one stealer
    respx.get("https://cavalier.hudsonrock.com/api/json/v2/osint-tools/search-by-email").mock(
        return_value=httpx.Response(
            200,
            json={
                "stealers": [
                    {
                        "computer_name": "LAPTOP-VICTIM (VictimUser)",
                        "operating_system": "Windows 10 (Build 19045)",
                        "stealer_family": "Vidar",
                        "date_compromised": "2023-11-01T08:30:00.000Z",
                        "total_user_services": 15,
                        "ip": "5.6.7.*",
                        "top_logins": ["victim@test.com"],
                        "top_passwords": ["v*****m"],
                    }
                ]
            },
        )
    )

    email_entity = Entity.create(entity_type=EntityType.EMAIL, value=email)
    with patch("osint_agent.collectors.tools._find_cli_binary", return_value=None):
        async with httpx.AsyncClient() as client:
            hits = await collect_email_intel(email_entity, client)

    stealer_hits = [h for h in hits if h.relation == "infostealer_compromise"]
    assert len(stealer_hits) == 1
    sh = stealer_hits[0]
    assert sh.raw_data["stealer_type"] == "Vidar"
    assert sh.raw_data["computer_name"] == "LAPTOP-VICTIM (VictimUser)"
    assert sh.source_tool == "email_intel"
    assert "stealer_victim@test.com" in sh.discovered_value or "infostealer" in sh.discovered_value


# ---------------------------------------------------------------------------
# LLM client resilience tests: 429 backoff, empty body, truncated JSON repair
# ---------------------------------------------------------------------------


def test_repair_truncated_json_basic() -> None:
    """_repair_truncated_json should close open strings and braces."""
    from osint_agent.llm.client import _repair_truncated_json

    # Truncated mid-string
    truncated = '{"name": "Alice", "bio": "Developer at Acme'
    repaired = _repair_truncated_json(truncated)
    parsed = json.loads(repaired)
    assert parsed["name"] == "Alice"

    # Truncated before closing brace
    truncated2 = '{"key": "value"'
    repaired2 = _repair_truncated_json(truncated2)
    parsed2 = json.loads(repaired2)
    assert parsed2["key"] == "value"

    # Nested truncation
    truncated3 = '{"a": {"b": "hello'
    repaired3 = _repair_truncated_json(truncated3)
    parsed3 = json.loads(repaired3)
    assert parsed3["a"]["b"] == "hello"

    # Already valid — should parse fine
    valid = '{"x": 1}'
    assert json.loads(_repair_truncated_json(valid)) == {"x": 1}


def test_parse_payload_repairs_truncated_json() -> None:
    """_parse_payload should auto-repair truncated JSON instead of raising."""
    from osint_agent.llm.client import BaseClient

    class _ConcreteClient(BaseClient):
        pass

    client = _ConcreteClient(call_budget=5)
    # Simulate LLM truncating mid-string
    truncated_raw = '{"real_name": "Bob Smith", "location": "Mumbai'
    result = client._parse_payload(truncated_raw, token_count=50)
    assert result["real_name"] == "Bob Smith"
    assert client.calls_made == 1


def test_parse_payload_raises_on_truly_empty() -> None:
    """_parse_payload should raise ValueError on empty body, not silently return."""
    from osint_agent.llm.client import BaseClient

    class _ConcreteClient(BaseClient):
        pass

    client = _ConcreteClient(call_budget=5)
    with pytest.raises(ValueError, match="empty response body"):
        client._parse_payload("", token_count=0)

    # Whitespace-only also counts as empty
    with pytest.raises(ValueError, match="empty response body"):
        client._parse_payload("   \n  ", token_count=0)


@pytest.mark.asyncio
@respx.mock
async def test_openrouter_client_429_trips_circuit_breaker_without_spam() -> None:
    """OpenRouterClient must fail fast on 429 and trip cooldown instead of spamming retries."""
    from osint_agent.llm.client import LLMRateLimitError, OpenRouterClient, reset_rate_limit_cooldown

    reset_rate_limit_cooldown()
    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        return httpx.Response(429, headers={"Retry-After": "60"}, json={"error": "rate limited"})

    respx.post("https://ai.hackclub.com/proxy/v1/chat/completions").mock(side_effect=side_effect)

    client = OpenRouterClient(api_key="test-key", call_budget=5)
    try:
        with pytest.raises(LLMRateLimitError):
            client.complete_json(system="sys", user="usr", max_tokens=100)
        # Second call during cooldown must not even hit the HTTP endpoint
        with pytest.raises(LLMRateLimitError):
            client.complete_json(system="sys", user="usr", max_tokens=100)
        assert call_count == 1
    finally:
        reset_rate_limit_cooldown()


@pytest.mark.asyncio
@respx.mock
async def test_openrouter_client_empty_body_retries() -> None:
    """OpenRouterClient should retry when LLM returns empty content body."""
    from unittest.mock import patch
    from osint_agent.llm.client import OpenRouterClient

    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Empty content
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": ""}}], "usage": {"total_tokens": 5}},
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"answer": "yes"}'}}],
                "usage": {"total_tokens": 10},
            },
        )

    respx.post("https://ai.hackclub.com/proxy/v1/chat/completions").mock(side_effect=side_effect)

    client = OpenRouterClient(api_key="test-key", call_budget=5)
    with patch("osint_agent.llm.client.time.sleep"):
        res = client.complete_json(system="sys", user="usr", max_tokens=100)

    assert res["answer"] == "yes"
    assert call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_openrouter_client_truncated_json_repaired() -> None:
    """OpenRouterClient should auto-repair truncated JSON from LLM and return valid result."""
    from osint_agent.llm.client import OpenRouterClient

    respx.post("https://ai.hackclub.com/proxy/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"real_name": "Eve", "location": "London'}}],
                "usage": {"total_tokens": 20},
            },
        )
    )

    client = OpenRouterClient(api_key="test-key", call_budget=5)
    res = client.complete_json(system="sys", user="usr", max_tokens=100)
    assert res["real_name"] == "Eve"
    assert res["location"] == "London"


@pytest.mark.asyncio
@respx.mock
async def test_openrouter_client_400_multimodal_falls_back_to_text() -> None:
    """OpenRouterClient should strip image_urls and retry as text-only if 400 Bad Request occurs."""
    from unittest.mock import patch
    from osint_agent.llm.client import OpenRouterClient

    seen_contents: list = []

    def side_effect(request):
        body = json.loads(request.content.decode("utf-8"))
        seen_contents.append(body["messages"][1]["content"])
        if isinstance(body["messages"][1]["content"], list):
            return httpx.Response(400, json={"error": {"message": "Invalid image URL"}})
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": '{"ok": true}'}}],
                "usage": {"total_tokens": 15},
            },
        )

    respx.post("https://ai.hackclub.com/proxy/v1/chat/completions").mock(side_effect=side_effect)

    client = OpenRouterClient(api_key="test-key", call_budget=5)
    with patch("osint_agent.llm.client.time.sleep"):
        res = client.complete_multimodal_json(
            system="sys",
            user="usr",
            image_urls=["https://example.com/avatar.png", "https://example.com/icon.svg"],
            max_tokens=1200,
        )

    assert res["ok"] is True
    assert len(seen_contents) == 2
    # First attempt sent only .png (filtered out .svg), second fell back to plain string
    assert isinstance(seen_contents[0], list)
    assert len(seen_contents[0]) == 2  # text + 1 valid png
    assert seen_contents[1] == "usr"


@pytest.mark.asyncio
@respx.mock
async def test_openrouter_client_reasoning_exhaustion_bumps_max_tokens() -> None:
    """OpenRouterClient should bump max_tokens and retry when reasoning model hits length with empty content."""
    from unittest.mock import patch
    from osint_agent.llm.client import OpenRouterClient

    seen_max_tokens: list[int] = []

    def side_effect(request):
        body = json.loads(request.content.decode("utf-8"))
        seen_max_tokens.append(body["max_tokens"])
        if len(seen_max_tokens) == 1:
            return httpx.Response(
                200,
                json={
                    "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {
                                "content": "",
                                "reasoning": "Thinking step 1... cut off before answer",
                            },
                        }
                    ],
                    "usage": {"total_tokens": 4096},
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": '{"extracted": "yes"}'}}],
                "usage": {"total_tokens": 5000},
            },
        )

    respx.post("https://ai.hackclub.com/proxy/v1/chat/completions").mock(side_effect=side_effect)

    client = OpenRouterClient(api_key="test-key", call_budget=5)
    with patch("osint_agent.llm.client.time.sleep"):
        res = client.complete_json(system="sys", user="usr", max_tokens=1200)

    assert res["extracted"] == "yes"
    assert seen_max_tokens == [8192, 16384]


# ---------------------------------------------------------------------------
# Deep scraping, Play Store About-Developer, open-ended fields, and noise filtering tests
# ---------------------------------------------------------------------------


def test_extractor_ignores_android_packages_and_file_extensions() -> None:
    """extract_entities must not emit fake domain nodes for Android package IDs, filenames, or CDN hosts."""
    from osint_agent.collectors.extractor import extract_entities

    text = (
        "Built Android app com.alaotach.eli5 in build.gradle and config.py. "
        "Assets on s.pinimg.com, miro.medium.com, schemas.android.com. "
        "Official website: https://eryzalabs.cloud and contact eryzalabs@gmail.com"
    )
    entities = extract_entities(text)
    domains = {e.value for e in entities if e.entity_type == EntityType.DOMAIN}
    emails = {e.value for e in entities if e.entity_type == EntityType.EMAIL}

    assert "eryzalabs.cloud" in domains
    assert "com.alaotach.eli5" not in domains
    assert "build.gradle" not in domains
    assert "config.py" not in domains
    assert "s.pinimg.com" not in domains
    assert "schemas.android.com" not in domains
    assert "eryzalabs@gmail.com" in emails


@pytest.mark.asyncio
@respx.mock
async def test_profile_analyzer_playstore_about_developer_and_support() -> None:
    """_analyze_playstore should extract About the developer (name, email, office address, phone) and support links."""
    from osint_agent.collectors.profile_analyzer import analyze_profile_hit

    app_url = "https://play.google.com/store/apps/details?id=com.alaotach.eli5&hl=en"
    sample_html = """
    <html>
      <head>
        <meta property="og:image" content="https://play-lh.googleusercontent.com/icon123=w240-h480-rw" />
      </head>
      <body>
        <h1><span class="AfwdI" itemprop="name">ELI5: Explain Text Anywhere</span></h1>
        <div class="Vbfug auoIOc"><a href="/store/apps/developer?id=AlAoTach+Inc."><span>AlAoTach Inc.</span></a></div>
        <div data-g-id="description">Explain any text on your screen instantly.</div>
        <a href="https://eryzalabs.cloud">Website</a>
        <a href="https://www.termsfeed.com/live/privacy-123">Privacy policy</a>
        <div class="xFVDSb">Support email</div><div class="pSEeg">eryzalabs@gmail.com</div>
        <div class="xFVDSb">Support phone</div><div class="pSEeg">+91-9876543210</div>
        <section>
          <div class="Bne0R ">About the developer</div>
          <div class="HhKIQc">
            <div> Aryan Mishra </div>
            <div> aryanmi2001@gmail.com </div>
            <div class="mHsyY"> Vard No.7 Koraon
Koraon, Uttar Pradesh 212306
India </div>
          </div>
        </div></div></div></section>
      </body>
    </html>
    """
    respx.get(app_url).mock(return_value=httpx.Response(200, text=sample_html))

    hit = Hit(
        target_entity_id="username:alaotach",
        discovered_type=EntityType.PROFILE,
        discovered_value=app_url,
        relation="published_app",
        source_tool="playstore_intel",
        metadata={"site": "Google Play Store", "url": app_url},
    )
    async with httpx.AsyncClient() as client:
        info = await analyze_profile_hit(hit, client)

    assert hit.metadata["name"] == "Aryan Mishra"
    assert hit.metadata["company"] == "AlAoTach Inc."
    assert "Koraon, Uttar Pradesh 212306" in hit.metadata["location"]
    assert "eryzalabs@gmail.com" in hit.metadata["contact_emails"]
    assert "aryanmi2001@gmail.com" in hit.metadata["contact_emails"]
    assert "+91-9876543210" in hit.metadata["phone_numbers"]
    assert hit.metadata["website"] == "https://eryzalabs.cloud"
    assert hit.metadata["extra_fields"]["privacy_policy_url"] == "https://www.termsfeed.com/live/privacy-123"
    assert hit.metadata["extra_fields"]["developer_name"] == "AlAoTach Inc."
    assert info["app_name"] == "ELI5: Explain Text Anywhere"


@pytest.mark.asyncio
@respx.mock
async def test_profile_analyzer_generic_html_strips_boilerplate_and_extracts_extra_fields() -> None:
    """_analyze_generic_html should strip nav/footer/script junk links and extract <dt>/<dd>, tel:, and images."""
    from osint_agent.collectors.profile_analyzer import analyze_profile_hit

    url = "https://example-portfolio.dev/aryan"
    html = """
    <html>
      <head>
        <meta property="og:image" content="https://example-portfolio.dev/photo.jpg" />
        <script src="https://s.pinimg.com/webapp/polyfills.mjs"></script>
      </head>
      <body>
        <nav><a href="https://instagram.com/artstationhq">Platform IG</a></nav>
        <main>
          <h1>Aryan Mishra</h1>
          <p>AI Engineer building ELI5.</p>
          <dl>
            <dt>University</dt><dd>IIT Delhi</dd>
            <dt>Role</dt><dd>Founder at EryzaLabs</dd>
          </dl>
          <a href="mailto:hello@eryzalabs.cloud">Email</a>
          <a href="tel:+919988776655">Call</a>
          <a href="https://github.com/alaotach">GitHub</a>
        </main>
        <footer><a href="https://www.epicgames.com/site/de/impressum">Impressum</a></footer>
      </body>
    </html>
    """
    respx.get("https://example-portfolio.dev/robots.txt").mock(return_value=httpx.Response(404))
    respx.get(url).mock(
        return_value=httpx.Response(200, headers={"content-type": "text/html"}, text=html)
    )

    hit = Hit(
        target_entity_id="username:alaotach",
        discovered_type=EntityType.PROFILE,
        discovered_value=url,
        relation="account_on",
        source_tool="maigret",
        metadata={"site": "Portfolio", "url": url},
    )
    async with httpx.AsyncClient() as client:
        await analyze_profile_hit(hit, client)

    links = hit.metadata.get("linked_urls", [])
    assert "https://github.com/alaotach" in links
    assert not any("artstationhq" in u or "epicgames.com" in u or "polyfills.mjs" in u for u in links)
    assert "hello@eryzalabs.cloud" in hit.metadata.get("contact_emails", [])
    assert "+919988776655" in hit.metadata.get("phone_numbers", [])
    assert hit.metadata["extra_fields"]["university"] == "IIT Delhi"
    assert hit.metadata["extra_fields"]["role"] == "Founder at EryzaLabs"
    assert "https://example-portfolio.dev/photo.jpg" in hit.metadata.get("images", [])


@pytest.mark.asyncio
@respx.mock
async def test_agent_run_follows_high_value_linked_profile_urls() -> None:
    """OSINTAgent.run must follow high-value linked_urls (e.g. Play Store app) without NameError on Hit."""
    from unittest.mock import AsyncMock, patch
    from osint_agent.collectors.tools import TOOL_REGISTRY, ToolDefinition
    from osint_agent.core.agent import OSINTAgent

    app_url = "https://play.google.com/store/apps/details?id=com.alaotach.eli5&hl=en"
    respx.get(app_url).mock(
        return_value=httpx.Response(
            200,
            text=(
                '<html><body><h1><span itemprop="name">ELI5</span></h1>'
                '<div class="Bne0R ">About the developer</div><div class="HhKIQc">'
                '<div>Aryan Mishra</div><div>aryanmi2001@gmail.com</div>'
                '<div class="mHsyY">Koraon, Uttar Pradesh 212306 India</div>'
                '</div></div></div></section></body></html>'
            ),
        )
    )

    async def fake_repo_collector(entity: Entity, client: httpx.AsyncClient) -> list[Hit]:
        return [
            Hit(
                target_entity_id=entity.id,
                discovered_type=EntityType.PROFILE,
                discovered_value="https://github.com/alaotach/eli-new",
                relation="authored_repo",
                source_tool="github_code_intel",
                is_trusted=True,
                metadata={"url": "https://github.com/alaotach/eli-new", "linked_urls": [app_url]},
            )
        ]

    respx.get("https://api.github.com/repos/alaotach/eli-new").mock(return_value=httpx.Response(404))
    respx.get("https://api.github.com/repos/alaotach/eli-new/readme").mock(return_value=httpx.Response(404))
    respx.get("https://api.github.com/repos/alaotach/eli-new/commits?per_page=10").mock(return_value=httpx.Response(404))
    respx.get("https://api.github.com/repos/alaotach/eli-new/contents").mock(return_value=httpx.Response(404))
    respx.get("https://api.github.com/repos/alaotach/eli-new/git/trees/HEAD?recursive=1").mock(return_value=httpx.Response(404))

    orig_tool = TOOL_REGISTRY.get("github_code_intel")
    TOOL_REGISTRY["github_code_intel"] = ToolDefinition(
        name="github_code_intel",
        description="mock",
        target_types=[EntityType.USERNAME],
        func=fake_repo_collector,
    )

    from osint_agent.llm.planner import PlanStep

    class _SingleToolPlanner:
        def plan(self, entity, graph_summary, remaining_budget):
            return [PlanStep(tool_name="github_code_intel", reason="test")]

    try:
        agent = OSINTAgent(planner=_SingleToolPlanner())
        scope = Scope(
            target_kind=EntityType.USERNAME,
            target_value="alaotach",
            consent_confirmed=True,
            max_depth=1,
            max_tool_calls=1,
        )
        graph = await agent.run(scope)
    finally:
        if orig_tool is not None:
            TOOL_REGISTRY["github_code_intel"] = orig_tool

    assert f"profile:{app_url}" in graph.nodes
    attrs = graph.nodes[f"profile:{app_url}"]["attributes"]
    assert attrs["name"] == "Aryan Mishra"
    assert "Koraon, Uttar Pradesh 212306" in attrs["location"]


def test_ml_enrichment_handles_structured_profiles_without_llm() -> None:
    """enrich_profiles_ml should extract ml_summary and skills locally, and rank_profiles_for_llm should skip already-structured hits."""
    from osint_agent.ml.profile_matcher import enrich_profiles_ml, rank_profiles_for_llm

    structured_hit = Hit(
        target_entity_id="username:alaotach",
        discovered_type=EntityType.PROFILE,
        discovered_value="https://play.google.com/store/apps/details?id=com.alaotach.eli5",
        relation="linked_profile",
        source_tool="playstore_intel",
        metadata={
            "name": "Aryan Mishra",
            "location": "Koraon, India",
            "email": "eryzalabs@gmail.com",
            "bio": "Android Kotlin AI explainer app.",
        },
        raw_data={
            "_page_text_excerpt": "ELI5 is an Android Kotlin and Python AI app built for fast explanations on mobile." * 5,
        },
    )
    unstructured_hit = Hit(
        target_entity_id="username:alaotach",
        discovered_type=EntityType.PROFILE,
        discovered_value="https://example.org/u/alaotach",
        relation="account_on",
        source_tool="maigret",
        metadata={"site": "Example"},
        raw_data={
            "_page_text_excerpt": "AlAoTach is an engineer working with Python, FastAPI, Docker and PyTorch on open source OSINT tools. " * 6,
        },
    )

    enrich_profiles_ml([structured_hit, unstructured_hit])
    assert structured_hit.metadata.get("ml_summary")
    assert "kotlin" in structured_hit.metadata.get("skills", [])
    assert "python" in unstructured_hit.metadata.get("skills", [])

    for_llm = rank_profiles_for_llm([structured_hit, unstructured_hit], max_profiles=6)
    assert structured_hit not in for_llm
    assert unstructured_hit in for_llm


def test_parse_seeds_freetext_fallback_handles_usernames_urls_names() -> None:
    """parse_seeds without an LLM must extract usernames, profile URLs, emails, and person names."""
    from osint_agent.llm.seeds import parse_seeds

    # Bare username
    s1 = parse_seeds("alaotach", client=None)
    assert len(s1) == 1
    assert s1[0].entity_type == EntityType.USERNAME
    assert s1[0].value == "alaotach"

    # Natural language with handle cue
    s2 = parse_seeds("Investigate github alaotach", client=None)
    assert any(e.entity_type == EntityType.USERNAME and e.value == "alaotach" for e in s2)

    # Profile URL
    s3 = parse_seeds("Check https://github.com/alaotach", client=None)
    assert any(e.entity_type == EntityType.PROFILE and "github.com/alaotach" in e.value for e in s3)

    # Full name phrase
    s4 = parse_seeds("Investigate Aryan Mishra", client=None)
    assert len(s4) == 1
    assert s4[0].entity_type == EntityType.CONSENTING_PERSON
    assert s4[0].value == "Aryan Mishra"

    # Email + phrase
    s5 = parse_seeds("Find info for aryanmi2001@gmail.com", client=None)
    assert any(e.entity_type == EntityType.EMAIL and e.value == "aryanmi2001@gmail.com" for e in s5)


@pytest.mark.asyncio
async def test_api_force_stop_investigation() -> None:
    """POST /api/investigate/{job_id}/stop should cancel a running asyncio task and mark job stopped."""
    import asyncio
    from fastapi import HTTPException
    from osint_agent.api.server import JOBS, JobState, stop_investigation

    job_id = "test_stop_001"
    job = JobState(job_id=job_id, request_data={"kind": "username", "value": "alaotach"})
    job.status = "running"

    async def dummy_long_task() -> None:
        await asyncio.sleep(60)

    job.task = asyncio.create_task(dummy_long_task())
    JOBS[job_id] = job
    try:
        res = await stop_investigation(job_id)
        assert res["status"] == "stopped"
        assert job.status == "stopped"
        assert job.completed_at is not None
        # Allow cancellation to propagate
        await asyncio.sleep(0)
        assert job.task.cancelled()

        # Unknown job should return 404
        with pytest.raises(HTTPException) as exc_info:
            await stop_investigation("nonexistent_job")
        assert exc_info.value.status_code == 404
    finally:
        if job.task and not job.task.done():
            job.task.cancel()
        JOBS.pop(job_id, None)


@pytest.mark.asyncio
@respx.mock
async def test_tool_exa_search_returns_hits_and_never_retries_on_402() -> None:
    """exa_search should emit PROFILE hits from /search results and permanently short-circuit on 402 without retrying."""
    from unittest.mock import patch
    from osint_agent.collectors.tools import collect_exa_search, reset_exa_state

    reset_exa_state()
    try:
        respx.post("https://ai.hackclub.com/proxy/v1/exa/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "ELI5: Explain Text Anywhere - Apps on Google Play",
                            "url": "https://play.google.com/store/apps/details?id=com.alaotach.eli5",
                            "author": "AlAoTach Inc.",
                            "publishedDate": "2026-01-01T00:00:00.000Z",
                            "text": "ELI5 is an AI text explainer built by Aryan Mishra.",
                        }
                    ],
                    "costDollars": {"total": 0.008},
                },
            )
        )

        entity = Entity.create(entity_type=EntityType.USERNAME, value="alaotach")
        with patch.dict("os.environ", {"EXA_API_KEY": "test-exa-key"}):
            async with httpx.AsyncClient() as client:
                hits = await collect_exa_search(entity, client)

        assert len(hits) == 1
        assert hits[0].discovered_value == "https://play.google.com/store/apps/details?id=com.alaotach.eli5"
        assert hits[0].metadata["name"] == "AlAoTach Inc."
        assert "Aryan Mishra" in hits[0].metadata["bio"]

        # Now test 402: must NOT retry and must block subsequent calls
        call_count = 0

        def side_effect_402(request):
            nonlocal call_count
            call_count += 1
            return httpx.Response(402, json={"error": "Payment Required"})

        respx.post("https://ai.hackclub.com/proxy/v1/exa/search").mock(side_effect=side_effect_402)
        with patch.dict("os.environ", {"EXA_API_KEY": "test-exa-key"}):
            async with httpx.AsyncClient() as client:
                hits_402 = await collect_exa_search(entity, client)
                hits_after = await collect_exa_search(entity, client)

        assert hits_402 == []
        assert hits_after == []
        assert call_count == 1
    finally:
        reset_exa_state()


def test_stream_checker_filters_soft404_202_redirects_and_blocklist() -> None:
    """_is_bot_or_soft404_response and load_false_positive_hosts should block catch-all 200/202/soft-404 sites."""
    from osint_agent.collectors.stream_checker import _is_bot_or_soft404_response, load_false_positive_hosts

    fp = load_false_positive_hosts()
    assert "last.fm" in fp
    assert "www.7cups.com" in fp
    assert "houzz.com" in fp

    # 202 Accepted WAF challenge
    r_202 = httpx.Response(202, request=httpx.Request("GET", "https://www.7cups.com/@test"))
    assert _is_bot_or_soft404_response(r_202, "<html></html>", "status_code") is True

    # Redirect to root homepage
    r_root = httpx.Response(200, request=httpx.Request("GET", "https://www.nitrotype.com/"))
    assert _is_bot_or_soft404_response(
        r_root, "<html>Home</html>", "status_code", probe_url="https://www.nitrotype.com/racer/test"
    ) is True

    # Redirect to /notfound or /dashboard
    r_nf = httpx.Response(200, request=httpx.Request("GET", "https://likee.video/notfound"))
    assert _is_bot_or_soft404_response(r_nf, "<html>Not found</html>", "status_code") is True

    # Deep soft-404 marker past 8KB
    deep_body = ("x" * 12000) + "<div>User not found</div>"
    r_deep = httpx.Response(200, request=httpx.Request("GET", "https://example.org/u/test"))
    assert _is_bot_or_soft404_response(r_deep, deep_body, "message") is True

    # Empty JSON with spaces
    r_json = httpx.Response(
        200,
        headers={"content-type": "application/json"},
        request=httpx.Request("GET", "https://www.freelancer.com/api/users"),
    )
    assert _is_bot_or_soft404_response(r_json, '{"result": {"users": {} }}', "status_code") is True


@pytest.mark.asyncio
@respx.mock
async def test_image_intel_vision_and_web_search_with_avatar_match() -> None:
    """analyze_image_vision + search_image_web should identify clues, query Exa/DDG/Wiki, and score avatar similarity."""
    import io
    from unittest.mock import patch
    from PIL import Image
    from osint_agent.collectors.image_intel import analyze_image_vision, search_image_web
    from osint_agent.llm.client import OpenRouterClient

    # Create a tiny valid PNG/JPEG image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(30, 144, 255)).save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    # Mock Vision LLM
    respx.post("https://ai.hackclub.com/proxy/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "visual_description": "Developer avatar for alaotach",
                                    "visible_text": "alaotach",
                                    "detected_names": ["Aryan Mishra"],
                                    "detected_usernames": ["alaotach"],
                                    "detected_emails": [],
                                    "detected_domains": [],
                                    "detected_organizations": ["EryzaLabs"],
                                    "landmarks_or_locations": [],
                                    "search_queries": ["Aryan Mishra alaotach"],
                                }
                            )
                        }
                    }
                ],
                "usage": {"total_tokens": 50},
            },
        )
    )

    llm = OpenRouterClient(api_key="test-key", call_budget=5)
    vision_info = await analyze_image_vision(img_bytes, filename="avatar.jpg", meta={}, llm_client=llm)
    assert "alaotach" in vision_info["detected_usernames"]
    assert "Aryan Mishra" in vision_info["detected_names"]

    # Mock Exa, DuckDuckGo, Wikipedia, and target profile + matching avatar image
    respx.post("https://ai.hackclub.com/proxy/v1/exa/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Aryan Mishra (alaotach) - Portfolio",
                        "url": "https://example-dev-bio.org/alaotach",
                        "author": "Aryan Mishra",
                        "text": "Aryan Mishra builds AI and Android apps.",
                    }
                ]
            },
        )
    )
    respx.get("https://api.duckduckgo.com/").mock(return_value=httpx.Response(200, json={"AbstractURL": ""}))
    respx.get("https://en.wikipedia.org/w/api.php").mock(return_value=httpx.Response(200, json={"query": {"search": []}}))
    respx.get("https://example-dev-bio.org/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://example-dev-bio.org/alaotach").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text=(
                '<html><head><meta property="og:image" content="https://example-dev-bio.org/avatar.jpg" /></head>'
                '<body><main><h1>Aryan Mishra</h1><p>Developer at EryzaLabs.</p></main></body></html>'
            ),
        )
    )
    respx.get("https://example-dev-bio.org/avatar.jpg").mock(
        return_value=httpx.Response(200, content=img_bytes)
    )

    with patch.dict("os.environ", {"EXA_API_KEY": "test-exa-key"}):
        async with httpx.AsyncClient() as client:
            res = await search_image_web(
                image_bytes=img_bytes,
                filename="avatar.jpg",
                meta={},
                vision_info=vision_info,
                target_id="profile:image:avatar.jpg",
                client=client,
            )

    assert len(res["hits"]) >= 1
    profile_hit = next(h for h in res["hits"] if "example-dev-bio.org" in h.discovered_value)
    assert profile_hit.metadata.get("visual_similarity", 0) >= 0.9
    assert profile_hit.metadata.get("avatar_match") is True
    assert any(s.entity_type == EntityType.USERNAME and s.value == "alaotach" for s in res["discovered_seeds"])
