"""Test suite for Intelligent OSINT FastAPI service."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from osint_agent.core.schema import EntityType, Hit
from osint_agent.api.server import JOBS, app
from osint_agent.collectors.tools import TOOL_REGISTRY

client = TestClient(app)


def test_dashboard_ui_serving() -> None:
    """GET / must serve the frontend index.html dashboard and static assets."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Intelligent OSINT" in resp.text
    assert "g6.min.js" in resp.text

    static_resp = client.get("/static/css/style.css")
    assert static_resp.status_code == 200
    assert "--node-domain" in static_resp.text


def test_api_health() -> None:
    """GET /api/health must return online status and registered tools catalog."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert data["tools_count"] >= 3
    tool_names = [t["name"] for t in data["available_tools"]]
    assert "crtsh" in tool_names
    assert "nominatim" in tool_names
    assert "wikidata" in tool_names


def test_api_consent_refusal() -> None:
    """POST /api/investigate without consent_confirmed=True must return 403 Forbidden."""
    payload = {
        "kind": "domain",
        "value": "unauthorized.com",
        "consent_confirmed": False,
    }
    resp = client.post("/api/investigate", json=payload)
    assert resp.status_code == 403
    assert "Consent confirmation required" in resp.json()["detail"]


def test_api_missing_target_validation() -> None:
    """POST /api/investigate without structured target or free_text must return 422."""
    payload = {
        "consent_confirmed": True,
    }
    resp = client.post("/api/investigate", json=payload)
    assert resp.status_code == 422


def test_api_parse_seeds_endpoint() -> None:
    """POST /api/parse-seeds must parse candidates and filter sensitive data."""
    payload = {
        "text": "Check Acme Corp at acme.org or call +1 555-019-2834",
        "provider": "fake",
    }
    resp = client.post("/api/parse-seeds", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 1
    # Ensure phone number was never included
    values = [e["value"] for e in data["entities"]]
    assert not any("555" in v for v in values)


@pytest.mark.asyncio
async def test_api_investigation_lifecycle() -> None:
    """Submit investigation, await completion, inspect status, graph, and audit."""
    with respx.mock(assert_all_called=False, assert_all_mocked=False) as r:
        r.route(host="crt.sh").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {"name_value": "api.domain-corp.com", "issuer_name": "DigiCert"},
                ],
            )
        )

        payload = {
            "kind": "domain",
            "value": "domain-corp.com",
            "consent_confirmed": True,
            "max_depth": 1,
            "max_tool_calls": 5,
            "resolver": "baseline",
            "planner": "rule",
        }

        # Submit investigation
        resp = client.post("/api/investigate", json=payload)
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        assert resp.json()["status"] == "queued"

        # Wait for background runner to finish
        for _ in range(25):
            await asyncio.sleep(0.1)
            status_resp = client.get(f"/api/investigate/{job_id}")
            if status_resp.json()["status"] in {"completed", "failed", "refused"}:
                break

        final_status = client.get(f"/api/investigate/{job_id}").json()
        assert final_status["status"] == "completed"
        assert final_status["stats"]["nodes_count"] >= 2

        # Check Graph endpoint (Cytoscape format)
        graph_resp = client.get(f"/api/graph/{job_id}")
        assert graph_resp.status_code == 200
        graph_data = graph_resp.json()["graph"]
        nodes = graph_data["elements"]["nodes"]
        edges = graph_data["elements"]["edges"]

        node_ids = [n["data"]["id"] for n in nodes]
        assert "domain:domain-corp.com" in node_ids
        assert "domain:api.domain-corp.com" in node_ids
        assert len(edges) >= 1
        assert edges[0]["data"]["relation"] == "subdomain_of"

        # Check Audit endpoint
        audit_resp = client.get(f"/api/audit/{job_id}")
        assert audit_resp.status_code == 200
        audit_events = audit_resp.json()["events"]
        assert len(audit_events) >= 3
        event_names = [e["event"] for e in audit_events]
        assert "job_started" in event_names
        assert "job_completed" in event_names


def test_api_stream_events() -> None:
    """GET /api/investigate/{job_id}/stream must stream Server-Sent Events."""
    # Pre-populate a job
    from osint_agent.api.server import JobState
    dummy_job = JobState("test_stream_job", {})
    dummy_job.append_event("test_event_1", {"msg": "hello"})
    dummy_job.status = "completed"
    JOBS["test_stream_job"] = dummy_job

    with client.stream("GET", "/api/investigate/test_stream_job/stream") as stream:
        lines = []
        for line in stream.iter_lines():
            if line:
                lines.append(line)
            if len(lines) >= 1:
                break

    assert any("test_event_1" in line for line in lines)


def test_api_analyze_image_and_similarity() -> None:
    """POST /api/analyze-image must enforce consent and extract image metadata + perceptual hash."""
    import io
    from PIL import Image
    from osint_agent.collectors.image_intel import compare_image_bytes

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(30, 144, 255)).save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    # 1. Consent refusal
    resp_refused = client.post(
        "/api/analyze-image",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"consent_confirmed": "false"},
    )
    assert resp_refused.status_code == 403

    # 2. Valid analysis
    resp_ok = client.post(
        "/api/analyze-image",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"consent_confirmed": "true"},
    )
    assert resp_ok.status_code == 200
    data = resp_ok.json()
    assert data["metadata"]["width"] == 64
    assert data["metadata"]["height"] == 64
    assert data["metadata"]["format"] == "JPEG"
    assert data["stats"]["nodes_count"] >= 1

    # 3. Perceptual similarity identical image == 1.0
    assert compare_image_bytes(img_bytes, img_bytes) == 1.0



def test_saved_runs_crud(tmp_path, monkeypatch) -> None:
    """POST/GET/DELETE /api/saved-runs should persist and reload investigation snapshots."""
    import osint_agent.api.server as srv

    monkeypatch.setattr(srv, "SAVED_RUNS_DIR", tmp_path / "saved_runs")
    r_save = client.post(
        "/api/saved-runs",
        json={
            "name": "alaotach snapshot",
            "job_id": "job123",
            "nodes": [{"data": {"id": "p1", "label": "https://github.com/alaotach", "type": "profile"}}],
            "edges": [],
            "audit_events": [],
        },
    )
    assert r_save.status_code == 200
    rid = r_save.json()["id"]

    r_list = client.get("/api/saved-runs")
    assert r_list.status_code == 200
    assert len(r_list.json()["runs"]) == 1
    assert r_list.json()["runs"][0]["id"] == rid

    r_get = client.get(f"/api/saved-runs/{rid}")
    assert r_get.status_code == 200
    assert r_get.json()["nodes"][0]["data"]["label"] == "https://github.com/alaotach"

    r_del = client.delete(f"/api/saved-runs/{rid}")
    assert r_del.status_code == 200
    assert client.get("/api/saved-runs").json()["runs"] == []
