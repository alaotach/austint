"""Offline tests for Phase E: transition logging, LinUCB bandit planner, and budget-vs-recall curves."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from osint_agent.collectors.tools import TOOL_REGISTRY, get_tools_for_type
from osint_agent.core.agent import OSINTAgent
from osint_agent.core.schema import Entity, EntityType, Hit, Scope
from osint_agent.ml.bandit import (
    BanditPlanner,
    ContextualBanditPolicy,
    extract_state_features,
)
from scripts.eval_bandit import run_bandit_training_and_eval


@pytest.mark.asyncio
async def test_agent_logs_policy_transitions_offline(tmp_path: Path):
    policy_log = tmp_path / "policy_transitions.jsonl"
    audit_log = tmp_path / "audit.jsonl"

    scope = Scope(
        target_kind=EntityType.PUBLIC_FIGURE,
        target_value="ada_lovelace",
        consent_confirmed=True,
        max_depth=1,
        max_tool_calls=1,
    )

    fake_hits = [
        Hit(
            target_entity_id="public_figure:ada_lovelace",
            discovered_type=EntityType.PROFILE,
            discovered_value="https://github.com/ada_lovelace",
            relation="profile_match",
            source_tool="github",
            is_trusted=True,
            metadata={"username": "ada_lovelace", "name": "Ada Lovelace"},
        ),
        Hit(
            target_entity_id="public_figure:ada_lovelace",
            discovered_type=EntityType.DOMAIN,
            discovered_value="adalovelace.example",
            relation="website",
            source_tool="github",
            is_trusted=True,
            metadata={},
        ),
    ]

    agent = OSINTAgent(audit_path=audit_log, policy_log_path=policy_log)
    first_tool_name = get_tools_for_type(EntityType.PUBLIC_FIGURE)[0].name
    with (
        patch.object(
            TOOL_REGISTRY[first_tool_name],
            "run",
            new=AsyncMock(return_value=fake_hits),
        ),
        patch("osint_agent.core.agent.analyze_profiles_batch", new=AsyncMock(return_value=None)),
        patch("osint_agent.core.agent.synthesize_profile_relations", new=AsyncMock(return_value=None)),
    ):
        graph = await agent.run(scope)

    assert graph.number_of_nodes() >= 3
    assert len(agent.policy_transitions) == 1
    tr = agent.policy_transitions[0]
    assert tr["tool"] == first_tool_name
    assert tr["cost"] == 1.0
    assert tr["reward"] >= 2.0
    assert policy_log.exists()
    lines = [json.loads(line) for line in policy_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0]["reward"] == tr["reward"]


def test_contextual_bandit_learns_tool_ranking_and_roundtrip(tmp_path: Path):
    applicable = [t.name for t in get_tools_for_type(EntityType.PUBLIC_FIGURE)]
    assert len(applicable) >= 2
    high_tool = applicable[-1]  # intentionally last in default order
    low_tool = applicable[0]

    entity = Entity.create(
        entity_type=EntityType.PUBLIC_FIGURE,
        value="ada_lovelace",
        confidence=1.0,
        depth=0,
    )
    sv = extract_state_features(entity, {"nodes_count": 1, "edges_count": 0}, remaining_budget=5).tolist()

    transitions = []
    for _ in range(10):
        transitions.append({"state_vec": sv, "tool": high_tool, "cost": 1.0, "reward": 3.0})
        transitions.append({"state_vec": sv, "tool": low_tool, "cost": 1.0, "reward": 0.0})

    policy = ContextualBanditPolicy(alpha=0.25).fit(transitions)
    ckpt = tmp_path / "bandit_policy.json"
    policy.save(ckpt)
    loaded = ContextualBanditPolicy.load(ckpt)

    planner = BanditPlanner(policy=loaded, explore=False, max_tools_per_entity=2)
    steps = planner.plan(entity, {"nodes_count": 1, "edges_count": 0}, remaining_budget=2)
    assert steps[0].tool_name == high_tool


def test_budget_vs_recall_curves_four_planners(tmp_path: Path):
    applicable = [t.name for t in get_tools_for_type(EntityType.PUBLIC_FIGURE)]
    best_tool = applicable[-1]
    weak_tool = applicable[0]

    entity = Entity.create(
        entity_type=EntityType.PUBLIC_FIGURE,
        value="linus",
        confidence=1.0,
        depth=0,
    )
    sv = extract_state_features(entity, {"nodes_count": 1, "edges_count": 0}, remaining_budget=4).tolist()

    transitions = []
    for _ in range(8):
        transitions.append({"state_vec": sv, "tool": best_tool, "cost": 1.0, "reward": 4.0})
        transitions.append({"state_vec": sv, "tool": weak_tool, "cost": 1.0, "reward": 0.0})

    episodes = [
        {
            "seed": {"entity_type": "public_figure", "value": f"target_{i}", "confidence": 1.0, "depth": 0},
            "tool_yields": {
                best_tool: [f"e_{i}_1", f"e_{i}_2", f"e_{i}_3"],
                weak_tool: [],
            },
            "ground_truth_entities": [f"e_{i}_1", f"e_{i}_2", f"e_{i}_3", f"e_{i}_4"],
        }
        for i in range(4)
    ]

    report = run_bandit_training_and_eval(
        transitions=transitions,
        eval_episodes=episodes,
        model_out_path=tmp_path / "bandit_policy.json",
        budgets=[1, 2, 4],
        llm_provider="fake",
    )
    assert report["status"] == "ok"
    for name in ("BFS", "RuleBasedPlanner", "LLMPlanner", "BanditPlanner"):
        assert name in report["curves"]
        assert name in report["auc_recall"]
    # At budget=1, BanditPlanner chooses best_tool (0.75 recall) over weak_tool (0.0 recall)
    assert report["curves"]["BanditPlanner"][0]["recall"] > report["curves"]["RuleBasedPlanner"][0]["recall"]


def test_cli_parser_supports_bandit_planner():
    from osint_agent.cli import parse_args

    args = parse_args(
        [
            "--kind",
            "public_figure",
            "--value",
            "ada",
            "--planner",
            "bandit",
            "--bandit-path",
            "models/bandit_policy.json",
        ]
    )
    assert args.planner == "bandit"
    assert args.bandit_path == "models/bandit_policy.json"
