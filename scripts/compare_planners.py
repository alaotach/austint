#!/usr/bin/env python3
"""Benchmark comparison between RuleBasedPlanner and LLMPlanner across representative seeds."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
import respx

from osint_agent.core.agent import OSINTAgent
from osint_agent.llm.client import FakeClient, get_llm_client
from osint_agent.llm.planner import LLMPlanner, RuleBasedPlanner
from osint_agent.ml.resolver import BaselineWeightedResolver
from osint_agent.core.schema import EntityType, Scope

RECORDED_RESPONSES = [
    # Plan for domain: example.com
    {"steps": [{"tool_name": "crtsh", "reason": "Query Certificate Transparency logs for public subdomains"}]},
    # Plan for username: octocat
    {"steps": [
        {"tool_name": "sherlock", "reason": "Fast public social profile discovery"},
        {"tool_name": "maigret", "reason": "Detailed dossier metadata extraction"}
    ]},
    # Plan for place: Denver
    {"steps": [{"tool_name": "nominatim", "reason": "Geocoding and parent administrative divisions"}]},
]


async def run_scenario(
    planner_kind: str,
    seeds: list[tuple[EntityType, str]],
    offline: bool = True,
    max_tool_calls: int = 15,
) -> dict[str, float | int]:
    """Execute evaluation scenario for a given planner across seed entities."""
    audit_file = PROJECT_ROOT / f"audit_eval_{planner_kind}.jsonl"
    if audit_file.exists():
        audit_file.unlink()

    # Configure client
    fake_client: FakeClient | None = None
    if planner_kind == "llm":
        if offline or not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("GEMINI_API_KEY")):
            fake_client = FakeClient(responses=list(RECORDED_RESPONSES), tokens_per_call=135)
            planner = LLMPlanner(client=fake_client, audit_path=audit_file)
        else:
            client = get_llm_client()
            planner = LLMPlanner(client=client, audit_path=audit_file)
    else:
        planner = RuleBasedPlanner(audit_path=audit_file)

    resolver = BaselineWeightedResolver()
    total_nodes = 0
    total_edges = 0

    # Mock external HTTP endpoints and subprocesses so benchmark is 100% offline and deterministic
    from unittest.mock import AsyncMock
    from osint_agent.core.schema import Hit
    from osint_agent.collectors.tools import TOOL_REGISTRY

    mock_sherlock_hits = [
        Hit(
            target_entity_id="username:octocat",
            discovered_type=EntityType.PROFILE,
            discovered_value="https://github.com/octocat",
            relation="account_on",
            source_tool="sherlock",
            is_trusted=False,
            metadata={"site": "GitHub", "name": "octocat"},
        )
    ]
    mock_maigret_hits = [
        Hit(
            target_entity_id="username:octocat",
            discovered_type=EntityType.PROFILE,
            discovered_value="https://github.com/octocat",
            relation="account_on",
            source_tool="maigret",
            is_trusted=False,
            metadata={"site": "GitHub", "name": "Mona Lisa Octocat", "bio": "GitHub mascot"},
        )
    ]

    orig_sherlock = TOOL_REGISTRY["sherlock"].func
    orig_maigret = TOOL_REGISTRY["maigret"].func

    TOOL_REGISTRY["sherlock"].func = AsyncMock(return_value=mock_sherlock_hits)
    TOOL_REGISTRY["maigret"].func = AsyncMock(return_value=mock_maigret_hits)

    with respx.mock(assert_all_called=False, assert_all_mocked=False) as respx_mock:
        # crt.sh mock
        respx_mock.route(host="crt.sh").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {"name_value": "api.example.com", "issuer_name": "Let's Encrypt"},
                    {"name_value": "admin.example.com", "issuer_name": "DigiCert"},
                ],
            )
        )
        # Nominatim mock
        respx_mock.route(host="nominatim.openstreetmap.org").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {"display_name": "Denver, Colorado, USA", "address": {"state": "Colorado", "country": "USA"}},
                ],
            )
        )
        # Wikidata mock
        respx_mock.route(host="www.wikidata.org").mock(
            return_value=httpx.Response(200, json={"search": []})
        )

        try:
            start_time = time.perf_counter()

            async with httpx.AsyncClient() as http_client:
                agent = OSINTAgent(
                    resolver=resolver,
                    planner=planner,
                    audit_path=audit_file,
                    http_client=http_client,
                )

                for kind, val in seeds:
                    scope = Scope(
                        target_kind=kind,
                        target_value=val,
                        consent_confirmed=True,
                        max_depth=1,
                        max_tool_calls=max_tool_calls,
                    )
                    graph = await agent.run(scope)
                    total_nodes += graph.number_of_nodes()
                    total_edges += graph.number_of_edges()

            elapsed = time.perf_counter() - start_time
        finally:
            TOOL_REGISTRY["sherlock"].func = orig_sherlock
            TOOL_REGISTRY["maigret"].func = orig_maigret

    # Calculate metrics from audit log
    tool_calls_count = 0
    rejected_count = 0
    llm_calls_count = 0
    token_usage = 0

    if audit_file.exists():
        import json
        with open(audit_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                evt = json.loads(line)
                name = evt.get("event")
                data = evt.get("data") or evt.get("details", {})
                if name == "tool_call_started":
                    tool_calls_count += 1
                elif name == "resolver_decision" and not data.get("accepted"):
                    rejected_count += 1
                elif name == "planner_decision":
                    if data.get("planner") == "LLMPlanner" and not data.get("cache_hit"):
                        llm_calls_count += 1
                        token_usage += data.get("token_usage", 0)

    # Clean up temp audit file
    if audit_file.exists():
        audit_file.unlink()

    return {
        "tool_calls": tool_calls_count,
        "llm_calls": llm_calls_count,
        "tokens": token_usage,
        "confirmed_nodes": total_nodes,
        "rejected_nodes": rejected_count,
        "wall_time": elapsed,
    }


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Compare RuleBasedPlanner vs LLMPlanner")
    parser.add_argument("--live", action="store_true", default=False, help="Use live LLM provider rather than recorded responses")
    args = parser.parse_args()

    seeds = [
        (EntityType.DOMAIN, "example.com"),
        (EntityType.USERNAME, "octocat"),
        (EntityType.PLACE, "Denver"),
    ]

    print("\n==============================================================================")
    print("  Intelligent OSINT: Planner Comparison (Rule-Based vs LLM-Orchestrated)")
    print("  [Mode: " + ("Live API" if args.live else "Recorded / Offline Offline-Safe") + "]")
    print("==============================================================================")
    print(f"[*] Evaluating 3 representative seeds across depth=1 scopes...")

    rule_metrics = await run_scenario("rule", seeds, offline=not args.live)
    llm_metrics = await run_scenario("llm", seeds, offline=not args.live)

    # Render Comparative Table
    print("\n" + "=" * 78)
    print(f"{'Performance Metric':<32} | {'RuleBasedPlanner':<20} | {'LLMPlanner':<18}")
    print("-" * 78)
    print(f"{'Tool Calls Dispatched':<32} | {rule_metrics['tool_calls']:<20} | {llm_metrics['tool_calls']:<18}")
    print(f"{'LLM Calls Made':<32} | {rule_metrics['llm_calls']:<20} | {llm_metrics['llm_calls']:<18}")
    print(f"{'Total LLM Tokens':<32} | {rule_metrics['tokens']:<20} | {llm_metrics['tokens']:<18}")
    print(f"{'Confirmed Graph Nodes':<32} | {rule_metrics['confirmed_nodes']:<20} | {llm_metrics['confirmed_nodes']:<18}")
    print(f"{'Rejected Decoy Hits':<32} | {rule_metrics['rejected_nodes']:<20} | {llm_metrics['rejected_nodes']:<18}")
    print(f"{'Wall-Clock Execution Time':<32} | {rule_metrics['wall_time']:<18.3f}s | {llm_metrics['wall_time']:<16.3f}s")
    print("=" * 78 + "\n")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
