"""CLI entry point for the Intelligent OSINT Agent."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from osint_agent.core.agent import OSINTAgent, export_graph_to_json
from osint_agent.core.schema import EntityType, Scope


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="osint_agent",
        description="Intelligent OSINT: Agentic Knowledge Graph Constructor across Public Sources",
    )
    parser.add_argument(
        "--kind",
        type=str,
        required=True,
        choices=[e.value for e in EntityType],
        help="Target entity kind (org, domain, place, public_figure, consenting_person)",
    )
    parser.add_argument(
        "--value",
        type=str,
        required=True,
        help="Target seed value (e.g. domain name, organization title, location)",
    )
    parser.add_argument(
        "--consent",
        action="store_true",
        default=False,
        help="Confirm user/investigator authorization to run public OSINT expansion",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum expansion depth along entity chains (default: 2)",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=15,
        help="Maximum allowed collector tool calls (default: 15)",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.8,
        help="Confidence decay multiplier per hops (default: 0.8)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Resolver acceptance confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--resolver",
        type=str,
        default="baseline",
        choices=["baseline", "learned", "neural", "neural_gnn"],
        help="Entity resolver backend (baseline heuristics, learned logreg, neural multimodal, or neural_gnn)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/model.joblib",
        help="Path to trained resolver model checkpoint for --resolver learned",
    )
    parser.add_argument(
        "--planner",
        type=str,
        default="rule",
        choices=["rule", "llm", "bandit"],
        help="Tool execution planner: rule (RuleBasedPlanner), llm (LLMPlanner), or bandit (BanditPlanner)",
    )
    parser.add_argument(
        "--bandit-path",
        type=str,
        default="models/bandit_policy.json",
        help="Path to trained contextual bandit policy JSON for --planner bandit",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="anthropic",
        choices=["anthropic", "gemini", "fake"],
        help="LLM provider for LLMPlanner (default: anthropic)",
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=20,
        help="Call budget for LLM operations (default: 20)",
    )
    parser.add_argument(
        "--free-text",
        type=str,
        default=None,
        help="Optional free-text target prompt to parse into seeds with LLM",
    )
    parser.add_argument(
        "--graph-out",
        type=str,
        default="graph.json",
        help="Output path for the generated knowledge graph JSON (default: graph.json)",
    )
    parser.add_argument(
        "--audit-out",
        type=str,
        default="audit.jsonl",
        help="Output path for append-only audit log JSONL (default: audit.jsonl)",
    )
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    scope = Scope(
        target_kind=EntityType(args.kind),
        target_value=args.value,
        consent_confirmed=args.consent,
        max_depth=args.max_depth,
        max_tool_calls=args.max_calls,
        confidence_decay=args.decay,
        accept_threshold=args.threshold,
    )

    from osint_agent.ml.resolver import BaselineWeightedResolver, LearnedResolver

    if args.resolver in ("learned", "neural", "neural_gnn"):
        model_p = Path(args.model_path)
        if not model_p.exists():
            print(f"\n[ERROR] Model checkpoint not found: {model_p}", file=sys.stderr)
            print("Train a model or specify a valid checkpoint with --model-path.\n", file=sys.stderr)
            return 2
        if args.resolver == "neural_gnn":
            from osint_agent.ml.gnn import NeuralGNNResolver
            resolver = NeuralGNNResolver.load(model_p)
        elif args.resolver == "neural":
            from osint_agent.ml.neural import NeuralResolver
            resolver = NeuralResolver.load(model_p)
        else:
            resolver = LearnedResolver.load(model_p)
    else:
        resolver = BaselineWeightedResolver()

    # Configure Planner
    from osint_agent.llm.planner import LLMPlanner, RuleBasedPlanner

    if args.planner == "llm":
        from osint_agent.llm.client import get_llm_client
        client = get_llm_client(provider=args.llm_provider, call_budget=args.max_llm_calls)
        planner = LLMPlanner(client=client, audit_path=args.audit_out)
    elif args.planner == "bandit":
        from osint_agent.ml.bandit import BanditPlanner, ContextualBanditPolicy

        b_path = Path(args.bandit_path)
        policy = ContextualBanditPolicy.load(b_path) if b_path.exists() else ContextualBanditPolicy()
        planner = BanditPlanner(policy=policy, audit_path=args.audit_out)
    else:
        planner = RuleBasedPlanner(audit_path=args.audit_out)

    agent = OSINTAgent(resolver=resolver, planner=planner, audit_path=args.audit_out)

    try:
        graph = await agent.run(scope)
    except PermissionError as perm_err:
        print(f"\n[REFUSAL] {perm_err}", file=sys.stderr)
        print("Set --consent flag to confirm authorization on target scope.\n", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[ERROR] Investigation failed: {exc}", file=sys.stderr)
        return 2

    export_graph_to_json(graph, args.graph_out)
    print(f"\n[SUCCESS] Intelligent OSINT Investigation Complete")
    print(f"  Target:     {scope.target_kind.value} -> '{scope.target_value}'")
    print(f"  Nodes:      {graph.number_of_nodes()}")
    print(f"  Edges:      {graph.number_of_edges()}")
    print(f"  Graph File: {Path(args.graph_out).resolve()}")
    print(f"  Audit File: {Path(args.audit_out).resolve()}\n")
    return 0


def main() -> None:
    args = parse_args()
    code = asyncio.run(main_async(args))
    sys.exit(code)


if __name__ == "__main__":
    main()
