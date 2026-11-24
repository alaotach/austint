"""Intelligent OSINT Agent expansion engine with priority queue and audit logging."""

from __future__ import annotations

import heapq
import itertools
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import networkx as nx

from osint_agent.core.audit import AuditLogger
from osint_agent.collectors.enrich import fetch_profile_meta
from osint_agent.collectors.extractor import extract_entities
from osint_agent.llm.planner import Planner, RuleBasedPlanner
from osint_agent.ml.resolver import BaseResolver, BaselineWeightedResolver
from osint_agent.core.schema import Entity, EntityType, Scope
from osint_agent.collectors.tools import TOOL_REGISTRY, set_audit_callback

logger = logging.getLogger(__name__)


class OSINTAgent:
    """Autonomous OSINT investigator performing depth-bounded, confidence-decayed expansion."""

    def __init__(
        self,
        resolver: BaseResolver | None = None,
        planner: Planner | None = None,
        audit_path: str | Path = "audit.jsonl",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.resolver = resolver or BaselineWeightedResolver()
        self.audit = AuditLogger(audit_path)
        self.planner = planner or RuleBasedPlanner(
            audit_path=audit_path, audit_callback=self.audit.log_event
        )
        self.http_client = http_client

    async def run(self, scope: Scope) -> nx.DiGraph:
        """Run the knowledge graph expansion loop for a given investigation scope."""
        # 1. Enforce strict consent refusal rule
        if not scope.consent_confirmed:
            self.audit.log_event(
                "consent_refusal",
                {
                    "target_kind": scope.target_kind.value,
                    "target_value": scope.target_value,
                    "reason": "Scope.consent_confirmed is False. Execution refused.",
                },
            )
            raise PermissionError(
                f"Agent refused to run: Scope.consent_confirmed is False for target '{scope.target_value}'."
            )

        self.audit.log_event(
            "investigation_started",
            {
                "target_kind": scope.target_kind.value,
                "target_value": scope.target_value,
                "max_depth": scope.max_depth,
                "max_tool_calls": scope.max_tool_calls,
                "confidence_decay": scope.confidence_decay,
            },
        )
        set_audit_callback(self.audit.log_event)

        graph = nx.DiGraph()
        seed_entity = Entity.create(
            entity_type=scope.target_kind,
            value=scope.target_value,
            confidence=1.0,
            depth=0,
            provenance=["seed:target"],
        )

        graph.add_node(
            seed_entity.id,
            entity_type=seed_entity.entity_type.value,
            value=seed_entity.value,
            confidence=seed_entity.confidence,
            depth=seed_entity.depth,
            provenance=seed_entity.provenance,
        )

        # Priority queue ordered by (-confidence, depth, sequence_id, entity)
        counter = itertools.count()
        pq: list[tuple[float, int, int, Entity]] = []
        heapq.heappush(pq, (-seed_entity.confidence, seed_entity.depth, next(counter), seed_entity))

        visited_entity_ids: set[str] = set()
        tool_call_count = 0

        owns_client = self.http_client is None
        client = self.http_client or httpx.AsyncClient(timeout=15.0)

        try:
            while pq and tool_call_count < scope.max_tool_calls:
                neg_conf, depth, _, current_entity = heapq.heappop(pq)
                entity_conf = -neg_conf

                if current_entity.id in visited_entity_ids:
                    continue
                visited_entity_ids.add(current_entity.id)

                # Check depth budget
                if depth >= scope.max_depth:
                    self.audit.log_event(
                        "depth_limit_reached",
                        {"entity_id": current_entity.id, "depth": depth, "max_depth": scope.max_depth},
                    )
                    continue

                graph_summary = {
                    "nodes_count": graph.number_of_nodes(),
                    "edges_count": graph.number_of_edges(),
                }
                remaining_budget = max(0, scope.max_tool_calls - tool_call_count)
                plan_steps = self.planner.plan(current_entity, graph_summary, remaining_budget)

                for step in plan_steps:
                    if tool_call_count >= scope.max_tool_calls:
                        self.audit.log_event(
                            "call_budget_exhausted",
                            {
                                "tool_call_count": tool_call_count,
                                "max_tool_calls": scope.max_tool_calls,
                                "entity_id": current_entity.id,
                            },
                        )
                        break

                    tool = TOOL_REGISTRY.get(step.tool_name)
                    if not tool:
                        continue

                    current_entity.attributes.setdefault("_tools_run", []).append(step.tool_name)
                    tool_call_count += 1
                    self.audit.log_event(
                        "tool_call_started",
                        {
                            "tool": tool.name,
                            "entity_id": current_entity.id,
                            "call_index": tool_call_count,
                            "reason": step.reason,
                        },
                    )

                    try:
                        hits = await tool.run(current_entity, client)
                    except Exception as err:
                        logger.exception("Error executing tool %s: %s", tool.name, err)
                        self.audit.log_event(
                            "tool_call_error",
                            {"tool": tool.name, "entity_id": current_entity.id, "error": str(err)},
                        )
                        continue

                    self.audit.log_event(
                        "tool_call_completed",
                        {
                            "tool": tool.name,
                            "entity_id": current_entity.id,
                            "hits_returned": len(hits),
                        },
                    )

                    # Enrich profile hit metadata if URL is present (up to 15 per username)
                    enriched_profiles_count = 0
                    for hit in hits:
                        if hit.discovered_type == EntityType.PROFILE and not hit.raw_data.get("description"):
                            if enriched_profiles_count < 15:
                                url = hit.metadata.get("url") or (
                                    hit.discovered_value if hit.discovered_value.startswith("http") else ""
                                )
                                if url:
                                    meta_text = await fetch_profile_meta(client, url)
                                    if meta_text:
                                        hit.raw_data["description"] = meta_text
                                        hit.metadata["description"] = meta_text
                                        hit.metadata["bio"] = meta_text
                                        enriched_profiles_count += 1

                        # 2. Score candidate hit using resolver
                        score = self.resolver.score(current_entity, hit)
                        accepted = self.resolver.should_accept(score, scope.accept_threshold)

                        # 3. Confidence calculation (trusted hits inherit parent confidence without decay)
                        if hit.is_trusted:
                            child_confidence = round(current_entity.confidence * score, 4)
                        else:
                            child_confidence = round(
                                current_entity.confidence * scope.confidence_decay * score, 4
                            )

                        self.audit.log_event(
                            "hit_evaluated",
                            {
                                "parent_id": current_entity.id,
                                "discovered_value": hit.discovered_value,
                                "discovered_type": hit.discovered_type.value,
                                "tool": hit.source_tool,
                                "is_trusted": hit.is_trusted,
                                "resolver_score": score,
                                "accepted": accepted,
                                "child_confidence": child_confidence,
                                "min_confidence": scope.min_confidence,
                            },
                        )

                        if not accepted or child_confidence < scope.min_confidence:
                            continue

                        child_entity = Entity.create(
                            entity_type=hit.discovered_type,
                            value=hit.discovered_value,
                            confidence=child_confidence,
                            depth=depth + 1,
                            attributes=hit.metadata,
                            provenance=current_entity.provenance + [f"{hit.source_tool}:{hit.relation}"],
                        )

                        # Insert or update node with provenance merge & max confidence
                        if child_entity.id not in graph:
                            graph.add_node(
                                child_entity.id,
                                entity_type=child_entity.entity_type.value,
                                value=child_entity.value,
                                confidence=child_entity.confidence,
                                depth=child_entity.depth,
                                provenance=list(child_entity.provenance),
                            )
                        else:
                            existing_conf = graph.nodes[child_entity.id].get("confidence", 0.0)
                            if child_entity.confidence > existing_conf:
                                graph.nodes[child_entity.id]["confidence"] = child_entity.confidence
                            existing_prov = graph.nodes[child_entity.id].get("provenance", [])
                            merged_prov = list(existing_prov)
                            for p in child_entity.provenance:
                                if p not in merged_prov:
                                    merged_prov.append(p)
                            graph.nodes[child_entity.id]["provenance"] = merged_prov

                        # Add relation edge carrying source, relation, and score
                        graph.add_edge(
                            current_entity.id,
                            child_entity.id,
                            relation=hit.relation,
                            source=hit.source_tool,
                            source_tool=hit.source_tool,
                            score=round(score, 4),
                        )

                        # Queue child entity if within depth budget
                        if child_entity.id not in visited_entity_ids and child_entity.depth < scope.max_depth:
                            heapq.heappush(
                                pq,
                                (
                                    -child_entity.confidence,
                                    child_entity.depth,
                                    next(counter),
                                    child_entity,
                                ),
                            )

                        # 4. Extract entities from confirmed hit's text
                        hit_text = (
                            hit.raw_data.get("description")
                            or hit.metadata.get("bio")
                            or hit.metadata.get("description")
                            or ""
                        )
                        if hit_text:
                            extracted_leads = extract_entities(hit_text)
                            for lead in extracted_leads:
                                lead_conf = round(child_entity.confidence * scope.confidence_decay, 4)
                                if lead_conf >= scope.min_confidence:
                                    lead_entity = Entity.create(
                                        entity_type=lead.entity_type,
                                        value=lead.value,
                                        confidence=lead_conf,
                                        depth=child_entity.depth + 1,
                                        provenance=child_entity.provenance + [f"extracted:{lead.entity_type.value}"],
                                    )
                                    if lead_entity.id not in graph:
                                        graph.add_node(
                                            lead_entity.id,
                                            entity_type=lead_entity.entity_type.value,
                                            value=lead_entity.value,
                                            confidence=lead_entity.confidence,
                                            depth=lead_entity.depth,
                                            provenance=list(lead_entity.provenance),
                                        )
                                    else:
                                        existing_conf = graph.nodes[lead_entity.id].get("confidence", 0.0)
                                        if lead_entity.confidence > existing_conf:
                                            graph.nodes[lead_entity.id]["confidence"] = lead_entity.confidence
                                        existing_prov = graph.nodes[lead_entity.id].get("provenance", [])
                                        merged_prov = list(existing_prov)
                                        for p in lead_entity.provenance:
                                            if p not in merged_prov:
                                                merged_prov.append(p)
                                        graph.nodes[lead_entity.id]["provenance"] = merged_prov

                                    graph.add_edge(
                                        child_entity.id,
                                        lead_entity.id,
                                        relation="mentioned_in",
                                        source="extractor",
                                        source_tool="extractor",
                                        score=1.0,
                                    )

                                    if lead_entity.id not in visited_entity_ids and lead_entity.depth < scope.max_depth:
                                        heapq.heappush(
                                            pq,
                                            (
                                                -lead_entity.confidence,
                                                lead_entity.depth,
                                                next(counter),
                                                lead_entity,
                                            ),
                                        )

            self.audit.log_event(
                "investigation_finished",
                {
                    "total_nodes": graph.number_of_nodes(),
                    "total_edges": graph.number_of_edges(),
                    "total_tool_calls": tool_call_count,
                },
            )
        finally:
            set_audit_callback(None)
            if owns_client:
                await client.aclose()

        return graph


def export_graph_to_json(graph: nx.DiGraph, output_path: str | Path) -> dict[str, Any]:
    """Serialize NetworkX DiGraph to standardized JSON with nodes and edges."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "directed": graph.is_directed(),
        "multigraph": graph.is_multigraph(),
        "graph_metadata": {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        },
        "nodes": [
            {"id": node_id, **attributes}
            for node_id, attributes in graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": u,
                "target": v,
                "relation": edge_data.get("relation"),
                "source_tool": edge_data.get("source"),
                "score": edge_data.get("score"),
            }
            for u, v, edge_data in graph.edges(data=True)
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data
