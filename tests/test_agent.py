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
