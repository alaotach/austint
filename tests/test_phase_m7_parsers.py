"""Offline parser & field-coverage tests for Milestone 7 (Data Depth)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from osint_agent.api.server import app, compute_field_coverage
from osint_agent.collectors.profile_analyzer import _analyze_instagram, parse_public_profile_html
from osint_agent.collectors.tools import collect_exa_search
from osint_agent.core.schema import Entity, EntityType


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
    assert cov_sparse["is_partial"] is True
    assert cov_sparse["is_sparse"] is True

    client = TestClient(app)
    resp = client.get("/api/collectors")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 14


@pytest.mark.asyncio
@respx.mock
async def test_exa_duckduckgo_fallback_and_instagram_extraction(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    ddg_html = """
    <html><body>
      <a class="result__a" href="https://www.instagram.com/alaotach/">Aryan (@alaotach) • Instagram photos and videos</a>
      <a class="result__snippet" href="#">1,250 Followers, 300 Following, 12 Posts - Builder &amp; ML Researcher</a>
    </body></html>
    """
    respx.post("https://ai.hackclub.com/proxy/v1/exa/search").mock(return_value=httpx.Response(429))
    respx.get("https://html.duckduckgo.com/html/").mock(return_value=httpx.Response(200, text=ddg_html))

    ent = Entity.create(entity_type=EntityType.USERNAME, value="alaotach")
    async with httpx.AsyncClient() as client:
        hits = await collect_exa_search(ent, client)

    assert len(hits) >= 1
    ig_hit = next(h for h in hits if "instagram.com/alaotach" in h.discovered_value)
    assert ig_hit.metadata["name"] == "Aryan"
    assert ig_hit.metadata["username"] == "alaotach"
    assert ig_hit.metadata["followers"] == "1,250"

    respx.get("https://www.instagram.com/alaotach/").mock(return_value=httpx.Response(429))
    async with httpx.AsyncClient() as client:
        ig_info = await _analyze_instagram("https://www.instagram.com/alaotach/", "alaotach", client)
    assert ig_info["name"] == "Aryan"
    assert ig_info["followers"] == "1,250"
