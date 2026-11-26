"""Phase E: Contextual bandit expansion policy (LinUCB) and budget-vs-recall evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from osint_agent.collectors.tools import TOOL_REGISTRY, get_tools_for_type
from osint_agent.core.audit import write_audit_event as _log_planner_audit
from osint_agent.core.schema import Entity
from osint_agent.llm.planner import PlanStep


ENTITY_TYPE_BUCKETS = (
    "public_figure",
    "consenting_person",
    "org",
    "domain",
    "place",
    "username",
    "email",
    "profile",
)
STATE_DIM = len(ENTITY_TYPE_BUCKETS) + 8  # 16 dims


def extract_state_features(
    entity: Entity | dict[str, Any],
    graph_summary: dict[str, Any],
    remaining_budget: int,
    max_budget: int = 25,
) -> np.ndarray:
    """Build fixed-size state vector for contextual bandit arm scoring."""
    vec = np.zeros((STATE_DIM,), dtype=np.float64)

    if isinstance(entity, Entity):
        etype = entity.entity_type.value
        conf = float(entity.confidence)
        depth = float(entity.depth)
        attrs = entity.attributes or {}
    else:
        etype = str(entity.get("entity_type") or "")
        conf = float(entity.get("confidence", 0.5))
        depth = float(entity.get("depth", 0))
        attrs = entity.get("attributes") or {}

    if etype in ENTITY_TYPE_BUCKETS:
        vec[ENTITY_TYPE_BUCKETS.index(etype)] = 1.0

    offset = len(ENTITY_TYPE_BUCKETS)
    vec[offset + 0] = 1.0  # bias
    vec[offset + 1] = min(max(conf, 0.0), 1.0)
    vec[offset + 2] = min(depth / 4.0, 1.0)
    vec[offset + 3] = min(max(remaining_budget, 0) / max(float(max_budget), 1.0), 1.0)
    vec[offset + 4] = min(float(graph_summary.get("nodes_count", 1)) / 50.0, 1.0)
    vec[offset + 5] = min(float(graph_summary.get("edges_count", 0)) / 100.0, 1.0)
    vec[offset + 6] = min(len(attrs.get("_tools_run") or []) / 10.0, 1.0)
    vec[offset + 7] = 1.0 if (attrs.get("links") or attrs.get("website") or attrs.get("bio")) else 0.0
    return vec


class ContextualBanditPolicy:
    """Disjoint LinUCB contextual bandit over OSINT tool arms."""

    def __init__(self, alpha: float = 0.5, l2_reg: float = 1.0, state_dim: int = STATE_DIM) -> None:
        self.alpha = float(alpha)
        self.l2_reg = float(l2_reg)
        self.state_dim = int(state_dim)
        self.A: dict[str, np.ndarray] = {}
        self.b: dict[str, np.ndarray] = {}
        self.counts: dict[str, int] = {}

    def _ensure_arm(self, arm: str) -> None:
        if arm not in self.A:
            self.A[arm] = self.l2_reg * np.eye(self.state_dim, dtype=np.float64)
            self.b[arm] = np.zeros((self.state_dim,), dtype=np.float64)
            self.counts[arm] = 0

    def update(self, state_vec: np.ndarray, tool: str, reward: float, cost: float = 1.0) -> None:
        """Update arm statistics with observed reward-per-unit-cost."""
        self._ensure_arm(tool)
        x = np.asarray(state_vec, dtype=np.float64).reshape(-1)
        r_norm = float(reward) / max(float(cost), 1.0)
        self.A[tool] += np.outer(x, x)
        self.b[tool] += r_norm * x
        self.counts[tool] += 1

    def fit(self, transitions: list[dict[str, Any]]) -> "ContextualBanditPolicy":
        """Fit LinUCB from logged `(state, tool, cost, reward)` transition dicts."""
        for tr in transitions:
            state = tr.get("state_vec")
            if state is None:
                state = extract_state_features(
                    tr.get("state", {}),
                    tr.get("graph_summary", {}),
                    int(tr.get("remaining_budget", 10)),
                )
            self.update(
                np.asarray(state, dtype=np.float64),
                str(tr["tool"]),
                float(tr.get("reward", 0.0)),
                float(tr.get("cost", 1.0)),
            )
        return self

    def score_arm(self, state_vec: np.ndarray, tool: str, explore: bool = True) -> float:
        self._ensure_arm(tool)
        x = np.asarray(state_vec, dtype=np.float64).reshape(-1)
        A_inv = np.linalg.inv(self.A[tool])
        theta = A_inv @ self.b[tool]
        mean = float(theta @ x)
        if not explore:
            return mean
        ucb = float(self.alpha * np.sqrt(max(float(x @ A_inv @ x), 0.0)))
        return mean + ucb

    def rank_tools(
        self,
        state_vec: np.ndarray,
        candidate_tools: list[str],
        explore: bool = False,
    ) -> list[tuple[str, float]]:
        scored = [(tool, self.score_arm(state_vec, tool, explore=explore)) for tool in candidate_tools]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "alpha": self.alpha,
            "l2_reg": self.l2_reg,
            "state_dim": self.state_dim,
            "arms": {
                arm: {
                    "A": self.A[arm].tolist(),
                    "b": self.b[arm].tolist(),
                    "count": self.counts[arm],
                }
                for arm in self.A
            },
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> "ContextualBanditPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        policy = cls(
            alpha=float(payload.get("alpha", 0.5)),
            l2_reg=float(payload.get("l2_reg", 1.0)),
            state_dim=int(payload.get("state_dim", STATE_DIM)),
        )
        for arm, data in (payload.get("arms") or {}).items():
            policy.A[arm] = np.asarray(data["A"], dtype=np.float64)
            policy.b[arm] = np.asarray(data["b"], dtype=np.float64)
            policy.counts[arm] = int(data.get("count", 0))
        return policy


class BanditPlanner:
    """Learned contextual-bandit planner implementing the Planner protocol."""

    def __init__(
        self,
        policy: ContextualBanditPolicy | None = None,
        *,
        explore: bool = False,
        max_tools_per_entity: int = 3,
        audit_path: str | Path | None = None,
        audit_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.policy = policy or ContextualBanditPolicy()
        self.explore = explore
        self.max_tools_per_entity = max_tools_per_entity
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
        candidate_tools = [t for t in applicable if t not in already_run]
        if not candidate_tools:
            return []

        state_vec = extract_state_features(entity, graph_summary, remaining_budget)
        ranked = self.policy.rank_tools(state_vec, candidate_tools, explore=self.explore)
        limit = min(remaining_budget, self.max_tools_per_entity)
        chosen = ranked[:limit]

        steps = [
            PlanStep(
                tool_name=tool_name,
                reason=f"LinUCB bandit score={score:.3f} for {entity.entity_type.value}",
            )
            for tool_name, score in chosen
        ]

        _log_planner_audit(
            "planner_decision",
            {
                "planner": "BanditPlanner",
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


class BFSPlanner:
    """Unprioritized BFS baseline planner iterating registry tools in fixed order."""

    def plan(
        self,
        entity: Entity,
        graph_summary: dict[str, Any],
        remaining_budget: int,
    ) -> list[PlanStep]:
        if remaining_budget <= 0:
            return []
        already_run = set(entity.attributes.get("_tools_run", []))
        # Registry order across all tools applicable to the entity
        applicable = [t.name for t in get_tools_for_type(entity.entity_type)]
        candidates = [t for t in applicable if t not in already_run][:remaining_budget]
        return [PlanStep(tool_name=t, reason="BFS registry order") for t in candidates]


def evaluate_budget_vs_recall(
    episodes: list[dict[str, Any]],
    planners: dict[str, Any],
    budgets: list[int] | None = None,
) -> dict[str, Any]:
    """Simulate budget-constrained expansion episodes and compute budget-vs-recall curves.

    Each episode has:
      - `seed`: dict representing initial entity (`entity_type`, `confidence`, `depth`, `attributes`)
      - `tool_yields`: dict[tool_name -> list[str]] of true confirmed entity IDs discoverable by that tool
      - `ground_truth_entities`: list[str] of all true discoverable entities for the target
    """
    budgets = budgets or [1, 2, 3, 4, 6, 8]
    curves: dict[str, list[dict[str, float]]] = {}

    for planner_name, planner in planners.items():
        pts: list[dict[str, float]] = []
        for budget in budgets:
            recalls: list[float] = []
            avg_discovered: list[float] = []
            for ep in episodes:
                gt = set(ep.get("ground_truth_entities") or [])
                if not gt:
                    continue
                seed_dict = ep["seed"]
                from osint_agent.core.schema import EntityType

                entity = Entity.create(
                    entity_type=EntityType(seed_dict["entity_type"]),
                    value=str(seed_dict.get("value") or "target"),
                    confidence=float(seed_dict.get("confidence", 1.0)),
                    depth=int(seed_dict.get("depth", 0)),
                    attributes=dict(seed_dict.get("attributes") or {}),
                )
                found: set[str] = set()
                calls_used = 0
                graph_summary = {"nodes_count": 1, "edges_count": 0}
                tool_yields: dict[str, list[str]] = ep.get("tool_yields") or {}

                # Run up to `budget` calls via planner steps
                while calls_used < budget:
                    steps = planner.plan(entity, graph_summary, budget - calls_used)
                    if not steps:
                        break
                    progressed = False
                    for step in steps:
                        if calls_used >= budget:
                            break
                        already = entity.attributes.setdefault("_tools_run", [])
                        if step.tool_name in already:
                            continue
                        already.append(step.tool_name)
                        calls_used += 1
                        progressed = True
                        discovered = tool_yields.get(step.tool_name, [])
                        found.update(discovered)
                        graph_summary["nodes_count"] = 1 + len(found)
                        graph_summary["edges_count"] = len(found)
                    if not progressed:
                        break

                recalls.append(len(found & gt) / float(len(gt)))
                avg_discovered.append(float(len(found & gt)))

            mean_recall = float(np.mean(recalls)) if recalls else 0.0
            pts.append(
                {
                    "budget": int(budget),
                    "recall": round(mean_recall, 4),
                    "avg_confirmed_entities": round(float(np.mean(avg_discovered)) if avg_discovered else 0.0, 4),
                }
            )
        curves[planner_name] = pts

    # Area under normalized budget-vs-recall curve (trapezoidal)
    auc_summary: dict[str, float] = {}
    for pname, pts in curves.items():
        if len(pts) < 2:
            auc_summary[pname] = pts[0]["recall"] if pts else 0.0
        else:
            xs = np.array([p["budget"] for p in pts], dtype=np.float64)
            ys = np.array([p["recall"] for p in pts], dtype=np.float64)
            span = max(float(xs[-1] - xs[0]), 1.0)
            auc_summary[pname] = round(float(np.trapezoid(ys, xs) / span), 4)

    return {
        "budgets": budgets,
        "curves": curves,
        "auc_recall": auc_summary,
    }
