"""Offline parser, pluggable web-search, recall_check, and extraction_gap tests."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from osint_agent.api.server import app, compute_field_coverage
from osint_agent.collectors.profile_analyzer import _analyze_instagram, parse_public_profile_html
from osint_agent.collectors.tools import collect_exa_search, set_audit_callback
from osint_agent.core.schema import Entity, EntityType
from scripts.extraction_gap import diagnose_urls
from scripts.recall_check import evaluate_recall_against_graph


def test_parse_public_profile_html_fixture():
    fixture = Path(__file__).parent / "fixtures" / "profile_jsonld_nextdata.html"
    html = fixture.read_text(encoding="utf-8")
    parsed = parse_public_profile_html(html)

    assert parsed["name"] == "Ada Lovelace"
    assert parsed["job_title"] == "Principal Systems Architect"
    assert parsed["location"] == "London, UK"
    assert parsed["joined"] == "2018-04-12"
    assert parsed["activity_dates"] == "2026-09-01"
    assert parsed["followers"] == 1420
    assert parsed["avatar_url"] == "https://cdn.example.org/avatars/ada.png"
    assert "Python" in parsed["languages"]
    assert "Analytical Engine Labs" in parsed["orgs"]
    assert "https://github.com/ada_dev" in parsed["linked_urls"]


def test_compute_field_coverage_and_collectors_endpoint():
    cov_full = compute_field_coverage(
        "github_code_intel",
        "profile",
        {
            "username": "ada_dev",
            "name": "Ada Lovelace",
            "bio": "Researcher",
            "location": "London",
            "joined": "2018-04-12",
            "followers": 1420,
            "links": ["https://adalovelace.example"],
            "avatar_url": "https://cdn.example.org/ada.png",
            "pinned_repos": ["engine"],
            "languages": ["Python"],
            "orgs": ["OpenCompute"],
        },
    )
    assert cov_full["extracted_count"] == 11
    assert cov_full["expected_count"] == 11
    assert cov_full["is_partial"] is False

    cov_sparse = compute_field_coverage("sherlock", "profile", {"username": "ada_dev"})
    assert cov_sparse["expected_count"] == 11
    assert cov_sparse["is_partial"] is True
    assert cov_sparse["is_sparse"] is True

    client = TestClient(app)
    resp = client.get("/api/collectors")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 14


@pytest.mark.asyncio
@respx.mock
async def test_pluggable_search_skips_with_audit_when_no_keys(monkeypatch):
    for k in ("BRAVE_SEARCH_API_KEY", "TAVILY_API_KEY", "GOOGLE_PSE_API_KEY", "GOOGLE_PSE_CX", "EXA_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("osint_agent.llm.client._load_dotenv", lambda *a, **kw: None)

    events: list[tuple[str, dict]] = []
    set_audit_callback(lambda ev, dt: events.append((ev, dt)))
    try:
        ent = Entity.create(entity_type=EntityType.USERNAME, value="alaotach")
        async with httpx.AsyncClient() as client:
            hits = await collect_exa_search(ent, client)
    finally:
        set_audit_callback(None)

    assert hits == []
    assert any(ev == "tool_unavailable" and dt.get("tool") == "exa_search" for ev, dt in events)


@pytest.mark.asyncio
@respx.mock
async def test_pluggable_search_brave_and_instagram_og_only(monkeypatch):
    for k in ("TAVILY_API_KEY", "GOOGLE_PSE_API_KEY", "GOOGLE_PSE_CX", "EXA_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave_test_key")

    respx.get("https://api.search.brave.com/res/v1/web/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "url": "https://github.com/alaotach",
                            "title": "alaotach (Aryan) · GitHub",
                            "description": "Builder and ML researcher",
                        }
                    ]
                }
            },
        )
    )

    ent = Entity.create(entity_type=EntityType.USERNAME, value="alaotach")
    async with httpx.AsyncClient() as client:
        hits = await collect_exa_search(ent, client)

    assert len(hits) == 1
    assert hits[0].discovered_value == "https://github.com/alaotach"

    # Instagram public og tags
    ig_html = """
    <html><head>
      <meta property="og:title" content="Aryan (@alaotach) • Instagram photos and videos">
      <meta property="og:description" content="1,250 Followers, 300 Following, 12 Posts - Builder">
    </head></html>
    """
    respx.get("https://www.instagram.com/alaotach/").mock(return_value=httpx.Response(200, text=ig_html, headers={"content-type": "text/html"}))
    async with httpx.AsyncClient() as client:
        ig_ok = await _analyze_instagram("https://www.instagram.com/alaotach/", "alaotach", client)
    assert ig_ok["name"] == "Aryan"
    assert ig_ok["followers"] == "1,250"

    # Instagram blocked -> limited public data
    respx.get("https://www.instagram.com/blocked_user/").mock(return_value=httpx.Response(429))
    async with httpx.AsyncClient() as client:
        ig_blocked = await _analyze_instagram("https://www.instagram.com/blocked_user/", "blocked_user", client)
    assert ig_blocked["limited_public_data"] is True
    assert "limited public data" in ig_blocked["status_note"]


@pytest.mark.asyncio
@respx.mock
async def test_recall_check_and_extraction_gap_offline(tmp_path: Path):
    res = evaluate_recall_against_graph(
        ["https://github.com/alaotach", "https://www.instagram.com/alaotach/"],
        ["https://github.com/alaotach", "https://extra.example/alaotach"],
    )
    assert res["recall"] == 0.5
    assert res["precision"] == 0.5
    assert res["missed"][0]["reason"] == "login_walled_or_unindexed_without_search_api"

    respx.get("https://example.org/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://example.org/alaotach").mock(
        return_value=httpx.Response(
            200,
            text='<html><head><meta property="og:title" content="Aryan"><meta property="og:description" content="ML researcher"></head></html>',
            headers={"content-type": "text/html"},
        )
    )
    async with httpx.AsyncClient() as client:
        diag = await diagnose_urls(["https://example.org/alaotach"], tmp_path / "raw", client=client)
    assert len(diag) == 1
    assert "og_tags" in diag[0]["raw_signals_detected"]
    assert (tmp_path / "raw" / "example" / "response.json").exists()
