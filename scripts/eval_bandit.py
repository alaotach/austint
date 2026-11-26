#!/usr/bin/env python3
"""Phase E: Train ContextualBanditPolicy and evaluate budget-vs-recall curves
for BFS, RuleBasedPlanner, LLMPlanner, and BanditPlanner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osint_agent.llm.client import get_llm_client
from osint_agent.llm.planner import LLMPlanner, RuleBasedPlanner
from osint_agent.ml.bandit import (
    BFSPlanner,
    BanditPlanner,
    ContextualBanditPolicy,
    evaluate_budget_vs_recall,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_bandit_training_and_eval(
    *,
    transitions: list[dict[str, Any]],
    eval_episodes: list[dict[str, Any]],
    model_out_path: Path,
    budgets: list[int] | None = None,
    alpha: float = 0.5,
    llm_provider: str = "fake",
) -> dict[str, Any]:
    if not transitions or not eval_episodes:
        return {
            "status": "skipped_no_agent_transitions",
            "transition_count": len(transitions),
            "episode_count": len(eval_episodes),
            "message": (
                "No logged agent-run policy_transitions.jsonl or eval episodes found. "
                "Run OSINTAgent with policy_log_path enabled first."
            ),
        }

    policy = ContextualBanditPolicy(alpha=alpha)
    policy.fit(transitions)
    policy.save(model_out_path)

    llm_client = get_llm_client(provider=llm_provider, call_budget=100)
    planners = {
        "BFS": BFSPlanner(),
        "RuleBasedPlanner": RuleBasedPlanner(),
        "LLMPlanner": LLMPlanner(client=llm_client),
        "BanditPlanner": BanditPlanner(policy=policy, explore=False),
    }
    eval_res = evaluate_budget_vs_recall(
        eval_episodes,
        planners,
        budgets=budgets or [1, 2, 3, 4, 6, 8],
    )
    return {
        "status": "ok",
        "checkpoint": str(model_out_path),
        "transition_count": len(transitions),
        "episode_count": len(eval_episodes),
        "arm_counts": policy.counts,
        **eval_res,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train & evaluate Phase E contextual bandit policy")
    parser.add_argument("--config", type=str, default="configs/osint_link.yaml")
    parser.add_argument("--transitions", type=str, default="")
    parser.add_argument("--episodes", type=str, default="")
    parser.add_argument("--model-out", type=str, default="models/bandit_policy.json")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_dir = Path(cfg["dataset"]["output_dir"])
    tr_path = Path(args.transitions) if args.transitions else (data_dir / "policy_transitions.jsonl")
    ep_path = Path(args.episodes) if args.episodes else (data_dir / "policy_episodes.jsonl")
    out_path = Path(args.out) if args.out else (data_dir / "bandit_eval.json")

    report = run_bandit_training_and_eval(
        transitions=_load_jsonl(tr_path),
        eval_episodes=_load_jsonl(ep_path),
        model_out_path=Path(args.model_out),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
