"""FastAPI REST API server for Intelligent OSINT with live event streaming and graph export."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from osint_agent.agent import OSINTAgent
from osint_agent.llm import get_llm_client
from osint_agent.planner import LLMPlanner, RuleBasedPlanner
from osint_agent.resolver import BaselineWeightedResolver, LearnedResolver
from osint_agent.schema import EntityType, Scope
from osint_agent.seeds import parse_seeds
from osint_agent.tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Intelligent OSINT API",
    description="Agentic OSINT reconnaissance engine with ML entity resolution and tool orchestration.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", tags=["Dashboard"])
async def serve_dashboard() -> FileResponse:
    """Serve the interactive Cytoscape.js knowledge graph dashboard."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard index.html not found.")
    return FileResponse(index_file)

# In-memory store for investigation jobs
JOBS: dict[str, JobState] = {}


class JobState:
    """State tracking for an autonomous investigation job."""

    def __init__(self, job_id: str, request_data: dict[str, Any]) -> None:
        self.job_id = job_id
        self.request_data = request_data
        self.status = "queued"
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.started_at: str | None = None
        self.completed_at: str | None = None
        self.error: str | None = None
        self.events: list[dict[str, Any]] = []
        self.subscribers: list[asyncio.Queue] = []
        self.graph_data: dict[str, Any] = {"elements": {"nodes": [], "edges": []}, "stats": {}}

    def append_event(self, event_type: str, details: dict[str, Any]) -> None:
        evt = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "data": details,
        }
        self.events.append(evt)
        # Notify SSE subscribers
        for q in list(self.subscribers):
            try:
                q.put_nowait(evt)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------

class ParseSeedsRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Free-text target description to parse")
    provider: str | None = Field("anthropic", description="LLM provider: anthropic, gemini, or fake")


class InvestigateRequest(BaseModel):
    kind: EntityType | None = Field(None, description="Target entity kind (org, domain, place, etc.)")
    value: str | None = Field(None, description="Target entity value")
    free_text: str | None = Field(None, description="Optional free-text target prompt to parse into seeds")
    consent_confirmed: bool = Field(False, description="MUST be True to authorize public OSINT search")
    max_depth: int = Field(2, ge=1, le=5, description="Maximum traversal depth along entity chain")
    max_tool_calls: int = Field(15, ge=1, le=100, description="Maximum allowed tool collector calls")
    confidence_decay: float = Field(0.8, gt=0.0, le=1.0, description="Multiplier decaying confidence per hop")
    accept_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Admission threshold for fuzzy hits")
    resolver: str = Field("learned", description="Entity resolver: 'baseline' or 'learned'")
    model_path: str = Field("model.joblib", description="Path to trained ML model checkpoint")
    planner: str = Field("rule", description="Tool execution planner: 'rule' or 'llm'")
    llm_provider: str = Field("anthropic", description="LLM provider for LLMPlanner: anthropic, gemini, or fake")
    max_llm_calls: int = Field(20, ge=1, le=100, description="Budget for LLM operations")


def format_graph_cytoscape(graph: Any) -> dict[str, Any]:
    """Convert NetworkX DiGraph into Cytoscape.js compatible graph structure."""
    nodes = []
    edges = []

    for node_id, attrs in graph.nodes(data=True):
        nodes.append({
            "data": {
                "id": str(node_id),
                "label": str(attrs.get("value", node_id)),
                "type": str(attrs.get("entity_type", "unknown")),
                "confidence": round(float(attrs.get("confidence", 1.0)), 4),
                "depth": int(attrs.get("depth", 0)),
                "provenance": list(attrs.get("provenance", [])),
            }
        })

    edge_index = 0
    for u, v, attrs in graph.edges(data=True):
        edges.append({
            "data": {
                "id": f"edge_{edge_index}",
                "source": str(u),
                "target": str(v),
                "relation": str(attrs.get("relation", "relates_to")),
                "score": round(float(attrs.get("score", 1.0)), 4),
                "source_tool": str(attrs.get("source_tool", attrs.get("source", "unknown"))),
            }
        })
        edge_index += 1

    return {
        "elements": {
            "nodes": nodes,
            "edges": edges,
        },
        "stats": {
            "nodes_count": len(nodes),
            "edges_count": len(edges),
        },
    }


async def _run_investigation_job(job: JobState, req: InvestigateRequest) -> None:
    """Asynchronous background runner for the OSINT agent investigation."""
    job.status = "running"
    job.started_at = datetime.now(timezone.utc).isoformat()
    job.append_event("job_started", {"job_id": job.job_id, "params": req.model_dump()})

    audit_file = Path(f"audit_job_{job.job_id}.jsonl")

    # 1. Resolve Target Entity
    target_kind = req.kind
    target_value = req.value

    if req.free_text and (not target_kind or not target_value):
        job.append_event("parsing_free_text_seeds", {"input": req.free_text[:64]})
        llm_client = None
        if req.planner == "llm":
            try:
                llm_client = get_llm_client(provider=req.llm_provider, call_budget=req.max_llm_calls)
            except Exception as exc:
                logger.warning("Could not instantiate LLM for seed parsing (%s); fallback to regex", exc)

        seeds = parse_seeds(
            req.free_text,
            client=llm_client,
            audit_callback=job.append_event,
        )
        if not seeds:
            job.status = "failed"
            job.error = "No valid public target entities found in provided free text."
            job.append_event("job_failed", {"reason": job.error})
            return

        target_kind = seeds[0].entity_type
        target_value = seeds[0].value

    if not target_kind or not target_value:
        job.status = "failed"
        job.error = "Could not determine valid target kind and value."
        job.append_event("job_failed", {"reason": job.error})
        return

    # 2. Configure Resolver
    if req.resolver == "learned":
        model_path = Path(req.model_path)
        if model_path.exists():
            resolver = LearnedResolver.load(model_path)
        else:
            job.append_event("model_not_found_fallback", {"path": str(model_path), "action": "fallback_baseline"})
            resolver = BaselineWeightedResolver()
    else:
        resolver = BaselineWeightedResolver()

    # 3. Configure Planner
    if req.planner == "llm":
        try:
            client = get_llm_client(provider=req.llm_provider, call_budget=req.max_llm_calls)
            planner = LLMPlanner(
                client=client,
                audit_path=audit_file,
                audit_callback=job.append_event,
            )
        except Exception as exc:
            job.append_event("llm_planner_init_failed", {"error": str(exc), "action": "fallback_rule_based"})
            planner = RuleBasedPlanner(audit_path=audit_file, audit_callback=job.append_event)
    else:
        planner = RuleBasedPlanner(audit_path=audit_file, audit_callback=job.append_event)

    # 4. Construct Agent & Execute Scope
    agent = OSINTAgent(
        resolver=resolver,
        planner=planner,
        audit_path=audit_file,
    )

    scope = Scope(
        target_kind=target_kind,
        target_value=target_value,
        consent_confirmed=req.consent_confirmed,
        max_depth=req.max_depth,
        max_tool_calls=req.max_tool_calls,
        confidence_decay=req.confidence_decay,
        accept_threshold=req.accept_threshold,
    )

    try:
        # Wrap agent audit logging to also notify job subscribers
        orig_log_event = agent.audit.log_event

        def hooked_log_event(ev_type: str, data: dict[str, Any]) -> None:
            orig_log_event(ev_type, data)
            job.append_event(ev_type, data)

        agent.audit.log_event = hooked_log_event

        graph = await agent.run(scope)
        job.graph_data = format_graph_cytoscape(graph)
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc).isoformat()
        job.append_event(
            "job_completed",
            {
                "nodes_count": job.graph_data["stats"]["nodes_count"],
                "edges_count": job.graph_data["stats"]["edges_count"],
            },
        )
    except PermissionError as perm_err:
        job.status = "refused"
        job.error = str(perm_err)
        job.append_event("job_refused", {"error": str(perm_err)})
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        job.append_event("job_failed", {"error": str(exc)})
    finally:
        # Clean up per-job audit file if created
        if audit_file.exists():
            try:
                audit_file.unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/api/health", tags=["System"])
async def health_check() -> dict[str, Any]:
    """Check API server health and catalog available tool collectors."""
    tools_catalog = [
        {
            "name": t.name,
            "description": t.description,
            "target_types": [tt.value for tt in t.target_types],
        }
        for t in TOOL_REGISTRY.values()
    ]
    return {
        "status": "online",
        "service": "Intelligent OSINT Agent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "available_tools": tools_catalog,
        "tools_count": len(tools_catalog),
    }


@app.post("/api/parse-seeds", tags=["Seeds"])
async def parse_seeds_endpoint(req: ParseSeedsRequest) -> dict[str, Any]:
    """Parse unstructured text into candidate OSINT seed entities with safety filters."""
    client = None
    try:
        client = get_llm_client(provider=req.provider or "anthropic")
    except Exception:
        pass

    entities = parse_seeds(req.text, client=client)
    return {
        "count": len(entities),
        "entities": [
            {
                "id": e.id,
                "type": e.entity_type.value,
                "value": e.value,
                "confidence": e.confidence,
            }
            for e in entities
        ],
    }


@app.post(
    "/api/investigate",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Investigation"],
)
async def start_investigation(
    req: InvestigateRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Submit a target to trigger autonomous OSINT reconnaissance and graph expansion.

    HARD RULE: Must confirm Scope.consent_confirmed=True. Refused otherwise.
    """
    # 1. HARD RULE: Consent Verification Gate
    if not req.consent_confirmed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Consent confirmation required: Scope.consent_confirmed must be True to run public OSINT expansion.",
        )

    # 2. Input validation: require either (kind + value) or free_text
    if not req.free_text and (not req.kind or not req.value):
        raise HTTPException(
            status_code=422,
            detail="Either structured (kind and value) or free_text must be provided.",
        )

    job_id = uuid.uuid4().hex[:12]
    job = JobState(job_id=job_id, request_data=req.model_dump())
    JOBS[job_id] = job

    # Launch background async runner
    background_tasks.add_task(_run_investigation_job, job, req)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Investigation job dispatched successfully.",
        "created_at": job.created_at,
    }


@app.get("/api/investigate/{job_id}", tags=["Investigation"])
async def get_investigation_status(job_id: str) -> dict[str, Any]:
    """Retrieve current status, statistics, and graph summary of an investigation."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Investigation job '{job_id}' not found.")

    return {
        "job_id": job.job_id,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "error": job.error,
        "stats": job.graph_data.get("stats", {}),
        "events_count": len(job.events),
    }


@app.get("/api/graph/{job_id}", tags=["Graph"])
async def get_investigation_graph(job_id: str) -> dict[str, Any]:
    """Retrieve Cytoscape.js formatted graph nodes and edges for frontend visualization."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Investigation job '{job_id}' not found.")

    return {
        "job_id": job.job_id,
        "status": job.status,
        "graph": job.graph_data,
    }


@app.get("/api/audit/{job_id}", tags=["Audit"])
async def get_investigation_audit(job_id: str) -> dict[str, Any]:
    """Retrieve full audit log trail for a given investigation."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Investigation job '{job_id}' not found.")

    return {
        "job_id": job.job_id,
        "events_count": len(job.events),
        "events": job.events,
    }


@app.get("/api/investigate/{job_id}/stream", tags=["Streaming"])
async def stream_investigation_events(job_id: str) -> StreamingResponse:
    """Server-Sent Events (SSE) streaming real-time audit decisions and node additions."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Investigation job '{job_id}' not found.")

    queue: asyncio.Queue = asyncio.Queue()
    # Replay historical events to new subscriber
    for ev in job.events:
        queue.put_nowait(ev)
    job.subscribers.append(queue)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                # Wait for next event with a periodic heartbeat
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=2.0)
                    yield f"data: {json.dumps(ev)}\n\n"
                    # Exit stream if job completed or failed
                    if ev.get("event") in {"job_completed", "job_failed", "job_refused"}:
                        break
                except asyncio.TimeoutError:
                    if job.status in {"completed", "failed", "refused"} and queue.empty():
                        break
                    # Send keep-alive heartbeat comment
                    yield ": keep-alive\n\n"
        finally:
            if queue in job.subscribers:
                job.subscribers.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
