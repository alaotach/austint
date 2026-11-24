"""Planners for tool orchestration: RuleBasedPlanner (baseline) and LLMPlanner (dynamic)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from osint_agent.audit import write_audit_event as _log_planner_audit
from osint_agent.llm import LLMClient
from osint_agent.schema import Entity
from osint_agent.tools import TOOL_REGISTRY, get_tools_for_type

logger = logging.getLogger(__name__)


@dataclass
class PlanStep:
    """Single orchestrated action step to execute against a candidate entity."""
    tool_name: str
    reason: str


@runtime_checkable
class Planner(Protocol):
    """Protocol for entity expansion planners."""

    def plan(
        self,
        entity: Entity,
        graph_summary: dict[str, Any],
        remaining_budget: int,
    ) -> list[PlanStep]:
        """Produce ordered list of tools to execute against candidate entity."""
        ...


class RuleBasedPlanner:
    """Baseline heuristic planner executing all applicable registered tools."""

    def __init__(
        self,
        audit_path: str | Path | None = None,
        audit_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.audit_path = audit_path
        self.audit_callback = audit_callback

    def plan(
        self,
        entity: Entity,
        graph_summary: dict[str, Any],
        remaining_budget: int,
    ) -> list[PlanStep]:
        if remaining_budget <= 0:
            return []

        applicable = [t.name for t in get_tools_for_type(entity.entity_type)]
        already_run = set(entity.attributes.get("_tools_run", []))

        # Filter out already executed tools
        candidate_tools = [t for t in applicable if t not in already_run]

        # Apply call budget cap
        chosen = candidate_tools[:remaining_budget]
        steps = [
            PlanStep(
                tool_name=tool_name,
                reason=f"Rule-based match for {entity.entity_type.value}",
            )
            for tool_name in chosen
        ]

        _log_planner_audit(
            "planner_decision",
            {
                "planner": "RuleBasedPlanner",
                "entity_id": entity.id,
                "entity_type": entity.entity_type.value,
                "chosen_tools": [s.tool_name for s in steps],
                "reasons": {s.tool_name: s.reason for s in steps},
                "fallback_used": False,
                "token_usage": 0,
            },
            audit_path=self.audit_path,
            callback=self.audit_callback,
        )
        return steps


class LLMPlanner:
    """Adaptive LLM planner with strict safety guards, budget capping, and prompt injection defense."""

    def __init__(
        self,
        client: LLMClient,
        audit_path: str | Path | None = None,
        audit_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.client = client
        self.audit_path = audit_path
        self.audit_callback = audit_callback
        self.baseline_fallback = RuleBasedPlanner(
            audit_path=audit_path, audit_callback=audit_callback
        )
        self._plan_cache: dict[tuple[str, tuple[str, ...], int], list[PlanStep]] = {}

    def plan(
        self,
        entity: Entity,
        graph_summary: dict[str, Any],
        remaining_budget: int,
    ) -> list[PlanStep]:
        if remaining_budget <= 0:
            return []

        applicable = [t.name for t in get_tools_for_type(entity.entity_type)]
        already_run = set(entity.attributes.get("_tools_run", []))
        available_tools = [t for t in applicable if t not in already_run]

        if not available_tools:
            return []

        # Check plan cache: bucket budget in increments of 5 to minimize redundant calls
        budget_bucket = min(remaining_budget, 5)
        cache_key = (
            entity.entity_type.value,
            tuple(sorted(available_tools)),
            budget_bucket,
        )

        if cache_key in self._plan_cache:
            cached_steps = self._plan_cache[cache_key]
            # Cap cached steps to current remaining budget
            steps = cached_steps[:remaining_budget]
            logger.debug("Plan cache hit for %s", cache_key)
            _log_planner_audit(
                "planner_decision",
                {
                    "planner": "LLMPlanner",
                    "entity_id": entity.id,
                    "entity_type": entity.entity_type.value,
                    "chosen_tools": [s.tool_name for s in steps],
                    "reasons": {s.tool_name: s.reason for s in steps},
                    "fallback_used": False,
                    "cache_hit": True,
                    "token_usage": 0,
                },
                audit_path=self.audit_path,
                callback=self.audit_callback,
            )
            return steps

        # Prepare prompt with PROMPT INJECTION DEFENSE:
        # Never pass profile text, scraped bio, or user-controlled text into planner prompts!
        # Only pass entity_type, truncated entity.value[:64], tool registry descriptions, and numeric stats.
        system_prompt = (
            "You are an OSINT tool selection planner.\n"
            "Given a target entity type, sanitized value, numeric graph statistics, and a list of available tools, "
            "select the most relevant tools in prioritized execution order.\n"
            "Return JSON only with format:\n"
            '{"steps": [{"tool_name": "<name>", "reason": "<rationale>"}]}\n'
            "RULES:\n"
            "1. ONLY select tool names from the provided available tools list.\n"
            "2. Never select tools that are not in the provided list.\n"
            "3. Return at most the remaining budget count."
        )

        tools_desc = "\n".join(
            f"- {t}: {TOOL_REGISTRY[t].description}"
            for t in available_tools
            if t in TOOL_REGISTRY
        )

        # Truncate value and extract strictly numeric summary to prevent prompt injection
        safe_value = str(entity.value)[:64]
        nodes_count = int(graph_summary.get("nodes_count", 0))
        edges_count = int(graph_summary.get("edges_count", 0))
        depth = int(entity.depth)

        user_prompt = (
            f"Entity Type: {entity.entity_type.value}\n"
            f"Entity Value: {safe_value}\n"
            f"Entity Depth: {depth}\n"
            f"Remaining Call Budget: {remaining_budget}\n"
            f"Graph Stats: total_nodes={nodes_count}, total_edges={edges_count}\n\n"
            f"Available Registered Tools for this entity type:\n{tools_desc}\n\n"
            "Select the best tools and return JSON."
        )

        token_usage = 0
        try:
            res = self.client.complete_json(system=system_prompt, user=user_prompt, max_tokens=400)
            token_usage = getattr(res, "token_usage", 0) or res.get("_token_usage", 0)
            raw_steps = res.get("steps", [])
            if not isinstance(raw_steps, list):
                raise ValueError("Expected 'steps' list in LLM planner response.")

            valid_steps: list[PlanStep] = []
            seen_tools: set[str] = set()

            for s in raw_steps:
                if not isinstance(s, dict):
                    continue
                tool_name = str(s.get("tool_name", "")).strip().lower()
                reason = str(s.get("reason", "Selected by LLM planner")).strip()

                # HARD GUARD 1: Intersect with available registered tools for this entity
                if tool_name not in available_tools:
                    logger.debug("LLM planner proposed unregistered/inapplicable tool '%s'; dropping.", tool_name)
                    continue

                if tool_name not in seen_tools:
                    seen_tools.add(tool_name)
                    valid_steps.append(PlanStep(tool_name=tool_name, reason=reason))

            # HARD GUARD 2: If LLM proposed no valid tools, fallback to baseline rule-based planner
            if not valid_steps:
                logger.info("LLM planner proposed 0 valid tools; falling back to RuleBasedPlanner.")
                return self.baseline_fallback.plan(entity, graph_summary, remaining_budget)

            # HARD GUARD 3: Cap steps by remaining budget
            final_steps = valid_steps[:remaining_budget]

            # Cache the plan
            self._plan_cache[cache_key] = final_steps

            _log_planner_audit(
                "planner_decision",
                {
                    "planner": "LLMPlanner",
                    "entity_id": entity.id,
                    "entity_type": entity.entity_type.value,
                    "chosen_tools": [s.tool_name for s in final_steps],
                    "reasons": {s.tool_name: s.reason for s in final_steps},
                    "fallback_used": False,
                    "cache_hit": False,
                    "token_usage": token_usage,
                },
                audit_path=self.audit_path,
                callback=self.audit_callback,
            )
            return final_steps

        except Exception as exc:
            logger.warning("LLM planner execution failed (%s); falling back to RuleBasedPlanner.", exc)
            fallback_steps = self.baseline_fallback.plan(entity, graph_summary, remaining_budget)
            _log_planner_audit(
                "planner_decision",
                {
                    "planner": "LLMPlanner",
                    "entity_id": entity.id,
                    "entity_type": entity.entity_type.value,
                    "chosen_tools": [s.tool_name for s in fallback_steps],
                    "reasons": {s.tool_name: f"Fallback: {s.reason}" for s in fallback_steps},
                    "fallback_used": True,
                    "error": str(exc),
                    "token_usage": token_usage,
                },
                audit_path=self.audit_path,
                callback=self.audit_callback,
            )
            return fallback_steps
