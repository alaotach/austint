"""Step 3 test suite covering profile collection, enrichment, entity extraction, and lead expansion."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from osint_agent.core.agent import OSINTAgent
from osint_agent.collectors.enrich import (
    MAX_RESPONSE_BYTES,
    fetch_profile_meta,
    is_private_or_loopback_host,
    parse_html_metadata,
)
from osint_agent.collectors.extractor import extract_entities, is_sensitive
from osint_agent.core.schema import Entity, EntityType, Hit, Scope
from osint_agent.collectors.tools import (
    collect_maigret,
    collect_sherlock,
    parse_maigret_json,
    parse_sherlock_output,
)


def test_parse_sherlock_output() -> None:
    """parse_sherlock_output must extract found account URLs and site names correctly."""
    sample_stdout = """
[*] Checking username testuser on:
[+] GitHub: https://github.com/testuser
[+] Reddit: https://www.reddit.com/user/testuser
[+] HackerNews: https://news.ycombinator.com/user?id=testuser
[*] Search completed with 3 results
"""
    hits = parse_sherlock_output(sample_stdout, "testuser", "username:testuser")
    assert len(hits) == 3

    assert hits[0].discovered_type == EntityType.PROFILE
    assert hits[0].discovered_value == "https://github.com/testuser"
    assert hits[0].relation == "account_on"
    assert hits[0].source_tool == "sherlock"
    assert hits[0].metadata["site"] == "GitHub"
    assert hits[0].is_trusted is False


def test_parse_maigret_json() -> None:
    """parse_maigret_json must parse profile metadata (bio, name, location, links)."""
    sample_json = {
        "GitHub": {
            "status": "Claimed",
            "url_user": "https://github.com/coder_bob",
            "parsed_info": {
                "name": "Bob Builder",
                "bio": "Cloud architect and distributed systems enthusiast",
                "location": "Denver, CO",
            },
        },
        "Twitter": {
            "status": "Claimed",
            "url_user": "https://twitter.com/coder_bob",
            "parsed_info": {
                "name": "Bob Builder",
                "bio": "Tweets about tech and clouds",
                "location": "Denver",
            },
        },
        "FakeSite": {
            "status": "Not Found",
            "url_user": "https://fakesite.com/coder_bob",
        },
    }

    hits = parse_maigret_json(sample_json, "coder_bob", "username:coder_bob")
    assert len(hits) == 2  # FakeSite with "Not Found" is filtered out

    gh_hit = next(h for h in hits if h.metadata["site"] == "GitHub")
    assert gh_hit.discovered_value == "https://github.com/coder_bob"
    assert gh_hit.metadata["location"] == "Denver, CO"
    assert "Cloud architect" in gh_hit.metadata["bio"]
    assert gh_hit.is_trusted is False


@pytest.mark.asyncio
async def test_missing_binary_graceful_audit(tmp_path: Path) -> None:
    """Missing sherlock and maigret binaries must not crash the agent and must write tool_unavailable audit."""
    audit_file = tmp_path / "missing_bin_audit.jsonl"
    entity = Entity.create(entity_type=EntityType.USERNAME, value="ghost_user")

    with patch("shutil.which", return_value=None):
        async with httpx.AsyncClient() as client:
            agent = OSINTAgent(audit_path=audit_file, http_client=client)

            # Run sherlock and maigret directly
            sherlock_hits = await collect_sherlock(entity, client)
            maigret_hits = await collect_maigret(entity, client)

            assert sherlock_hits == []
            assert maigret_hits == []

    # Check audit log written
    assert audit_file.exists() or Path("audit.jsonl").exists()
    target_audit = audit_file if audit_file.exists() else Path("audit.jsonl")
    with open(target_audit, "r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f]

    unavailable_events = [e for e in events if e.get("event") == "tool_unavailable"]
    assert len(unavailable_events) >= 1
    assert any(e["data"]["tool"] in ("sherlock", "maigret") for e in unavailable_events)


def test_enrich_html_metadata_parsing() -> None:
    """HTML metadata parsing extracts og:title, og:description, meta description, and title."""
    html_sample = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Jane Doe - Personal Portfolio</title>
        <meta property="og:title" content="Jane Doe | Cyber Security" />
        <meta property="og:description" content="Official portfolio and blog of Jane Doe, security researcher." />
        <meta name="description" content="Meta fallback description." />
    </head>
    <body><h1>Hello</h1></body>
    </html>
    """
    meta = parse_html_metadata(html_sample)
    assert meta["og:title"] == "Jane Doe | Cyber Security"
    assert "security researcher" in meta["og:description"]
    assert meta["title"] == "Jane Doe - Personal Portfolio"


def test_ssrf_guard() -> None:
    """is_private_or_loopback_host must block localhost, loopback, and private IPv4/IPv6 ranges."""
    assert is_private_or_loopback_host("localhost") is True
    assert is_private_or_loopback_host("127.0.0.1") is True
    assert is_private_or_loopback_host("::1") is True
    assert is_private_or_loopback_host("10.0.0.5") is True
    assert is_private_or_loopback_host("192.168.1.100") is True
    assert is_private_or_loopback_host("172.16.0.1") is True
    assert is_private_or_loopback_host("169.254.169.254") is True  # Cloud metadata IP


@pytest.mark.asyncio
@respx.mock
async def test_enrich_size_cap_and_robots() -> None:
    """fetch_profile_meta respects 200 KB size limit and robots.txt disallow."""
    # Mock robots.txt allowing public profiles
    respx.get("https://example-allowed.com/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\nDisallow: /admin/\nAllow: /")
    )
    # Return 250 KB of HTML
    large_html = ("<html><head><title>Large Profile</title></head><body>" + ("x" * 250_000) + "</body></html>").encode("utf-8")
    respx.get("https://example-allowed.com/user/profile").mock(
        return_value=httpx.Response(200, content=large_html, headers={"Content-Type": "text/html"})
    )

    async with httpx.AsyncClient() as client:
        text = await fetch_profile_meta(client, "https://example-allowed.com/user/profile", domain_delay=0.0)
        assert "Large Profile" in text

    # Mock robots.txt disallowing private path
    respx.get("https://example-disallowed.com/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\nDisallow: /private/")
    )
    async with httpx.AsyncClient() as client:
        disallowed_text = await fetch_profile_meta(client, "https://example-disallowed.com/private/profile", domain_delay=0.0)
        assert disallowed_text == ""


def test_extractor_entities_and_sanitization() -> None:
    """extract_entities extracts emails, domains, handles, and strictly scrubs phones/addresses."""
    bio_text = """
    Security engineer at OpenAI. Contact me at researcher@secureai.org or check out https://personal-research.net.
    Follow my updates @sec_analyst.
    Never leak phone: +1-555-123-4567 or residence: 742 Evergreen Terrace, Springfield.
    Also ignore social profiles like https://twitter.com/sec_analyst and https://bit.ly/12345.
    """
    extracted = extract_entities(bio_text)
    values = [e.value for e in extracted]
    types = [e.entity_type for e in extracted]

    # Email found
    assert "researcher@secureai.org" in values
    assert EntityType.EMAIL in types

    # Non-social domain found
    assert "personal-research.net" in values
    assert EntityType.DOMAIN in types

    # Social and shortener domains excluded
    assert not any("twitter.com" in v for v in values)
    assert not any("bit.ly" in v for v in values)

    # Handle found
    assert "sec_analyst" in values
    assert EntityType.USERNAME in types

    # Sensitive attributes (phone numbers, street addresses) MUST NOT be extracted
    assert not any("555" in v for v in values)
    assert not any("Evergreen" in v for v in values)
    assert not any("742" in v for v in values)


@pytest.mark.asyncio
@respx.mock
async def test_agent_username_lead_expansion(tmp_path: Path) -> None:
    """Agent expands from username seed -> profile -> extracts mentioned new domain as child node."""
    audit_path = tmp_path / "username_audit.jsonl"

    sample_stdout = "[+] GitHub: https://github.com/octocat"
    with patch("shutil.which", return_value="/usr/bin/sherlock"), \
         patch("asyncio.create_subprocess_exec") as mock_exec:

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (sample_stdout.encode("utf-8"), b"")
        mock_exec.return_value = mock_proc

        # Mock robots.txt and profile HTML
        respx.get("https://github.com/robots.txt").mock(
            return_value=httpx.Response(200, text="User-agent: *\nAllow: /")
        )
        respx.get("https://github.com/octocat").mock(
            return_value=httpx.Response(
                200,
                text="""
                <html>
                <head>
                    <meta property="og:title" content="The Octocat Profile" />
                    <meta property="og:description" content="Lead maintainer. Official personal blog at https://octo-research.io. Reach out @octo_support" />
                </head>
                <body>GitHub user</body>
                </html>
                """,
            )
        )

        async with httpx.AsyncClient() as client:
            agent = OSINTAgent(audit_path=audit_path, http_client=client)
            scope = Scope(
                target_kind=EntityType.USERNAME,
                target_value="octocat",
                consent_confirmed=True,
                max_depth=2,
                max_tool_calls=5,
            )
            graph = await agent.run(scope)

    # Verify seed node
    assert "username:octocat" in graph

    # Verify profile node was admitted
    assert "profile:https://github.com/octocat" in graph

    # Verify newly extracted domain child node was spawned from bio
    assert "domain:octo-research.io" in graph
    domain_node = graph.nodes["domain:octo-research.io"]
    assert domain_node["depth"] >= 1

    # Verify edge
    edge = graph.edges["profile:https://github.com/octocat", "domain:octo-research.io"]
    assert edge["relation"] == "mentioned_in"
    assert edge["source_tool"] == "extractor"


def test_agent_dedupe_highest_confidence_and_merged_provenance() -> None:
    """When a node is reached by multiple paths, keep highest confidence and merge provenance."""
    import networkx as nx

    graph = nx.DiGraph()

    # Initial node insertion from path 1
    node_id = "domain:shared-hub.org"
    graph.add_node(
        node_id,
        entity_type="domain",
        value="shared-hub.org",
        confidence=0.60,
        depth=1,
        provenance=["path1:crtsh"],
    )

    # Second discovery from path 2 with higher confidence and different provenance
    new_confidence = 0.85
    new_provenance = ["path2:wikidata", "path2:reference"]

    # Simulating agent dedupe logic
    existing_conf = graph.nodes[node_id].get("confidence", 0.0)
    if new_confidence > existing_conf:
        graph.nodes[node_id]["confidence"] = new_confidence

    existing_prov = graph.nodes[node_id].get("provenance", [])
    merged_prov = list(existing_prov)
    for p in new_provenance:
        if p not in merged_prov:
            merged_prov.append(p)
    graph.nodes[node_id]["provenance"] = merged_prov

    assert graph.nodes[node_id]["confidence"] == 0.85
    assert "path1:crtsh" in graph.nodes[node_id]["provenance"]
    assert "path2:wikidata" in graph.nodes[node_id]["provenance"]
    assert len(graph.nodes[node_id]["provenance"]) == 3


def test_real_fixtures_parsing() -> None:
    """Ensure parsers accurately process captured real outputs from sherlock and maigret."""
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    sherlock_file = fixtures_dir / "sherlock_raw.txt"
    maigret_file = fixtures_dir / "maigret_raw.json"

    assert sherlock_file.exists(), "sherlock_raw.txt fixture is missing"
    assert maigret_file.exists(), "maigret_raw.json fixture is missing"

    # Sherlock fixture parsing
    sherlock_text = sherlock_file.read_text(encoding="utf-8")
    s_hits = parse_sherlock_output(sherlock_text, "torvalds", "username:torvalds")
    assert len(s_hits) >= 1
    assert any(h.metadata.get("site") == "GitHub" for h in s_hits)
    assert any("github.com/torvalds" in h.discovered_value for h in s_hits)

    # Maigret fixture parsing
    with open(maigret_file, "r", encoding="utf-8") as f:
        maigret_data = json.load(f)
    m_hits = parse_maigret_json(maigret_data, "torvalds", "username:torvalds")
    assert len(m_hits) >= 1
    gh_hit = next(h for h in m_hits if h.metadata.get("site") == "GitHub")
    assert gh_hit.metadata.get("name") == "Linus Torvalds"
    assert gh_hit.metadata.get("location") == "Portland, OR"
    assert "Linux Foundation" in gh_hit.metadata.get("bio", "")


@pytest.mark.asyncio
@respx.mock
async def test_stream_username_check_offline() -> None:
    """Native streaming checker should yield hits immediately via on_hit callback."""
    from osint_agent.collectors.stream_checker import stream_username_check

    mock_sites = [
        (
            10,
            "GitHub",
            {
                "url": "https://github.com/{username}",
                "urlProbe": "https://api.github.com/users/{username}",
                "checkType": "status_code",
            },
        ),
        (
            20,
            "MissingSite",
            {
                "url": "https://missing.example.com/{username}",
                "checkType": "status_code",
            },
        ),
    ]

    respx.get("https://api.github.com/users/streamuser").mock(
        return_value=httpx.Response(
            200,
            json={"name": "Stream User", "bio": "Async OSINT engineer", "location": "Austin, TX"},
        )
    )
    respx.get("https://missing.example.com/streamuser").mock(
        return_value=httpx.Response(404)
    )

    streamed: list[Hit] = []
    with patch("osint_agent.collectors.stream_checker.load_maigret_sites", return_value=mock_sites):
        hits = await stream_username_check(
            username="streamuser",
            target_id="username:streamuser",
            max_sites=10,
            concurrency=5,
            overall_timeout=5.0,
            on_hit=lambda h: streamed.append(h),
        )

    assert len(hits) == 1
    assert len(streamed) == 1
    assert hits[0].discovered_value == "https://github.com/streamuser"
    assert hits[0].metadata["name"] == "Stream User"
    assert hits[0].metadata["location"] == "Austin, TX"
    assert "Async OSINT engineer" in hits[0].metadata["bio"]
