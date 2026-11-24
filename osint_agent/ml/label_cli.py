"""Interactive terminal CLI tool for human-in-the-loop OSINT dataset annotation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from osint_agent.ml.dataset import LabeledPair
from osint_agent.ml.features import TargetContext
from osint_agent.core.schema import EntityType, Hit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="osint_agent.ml.label_cli",
        description="Annotate candidate OSINT entity pairs for resolver model training",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="graph.json",
        help="Path to input graph.json or JSONL hits file (default: graph.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pairs.jsonl",
        help="Path to append labeled pairs JSONL (default: pairs.jsonl)",
    )
    return parser.parse_args(argv)


def load_candidate_pairs_from_graph(graph_path: Path) -> list[tuple[TargetContext, Hit]]:
    """Parse graph.json nodes and edges into candidate target-context and hit pairs."""
    with open(graph_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    nodes_by_id = {n["id"]: n for n in graph_data.get("nodes", [])}
    candidates: list[tuple[TargetContext, Hit]] = []

    for edge in graph_data.get("edges", []):
        src_id = edge.get("source")
        tgt_id = edge.get("target")
        src_node = nodes_by_id.get(src_id)
        tgt_node = nodes_by_id.get(tgt_id)

        if not src_node or not tgt_node:
            continue

        ctx = TargetContext()
        src_type = src_node.get("entity_type")
        src_val = src_node.get("value", "")
        if src_type == "domain":
            ctx.absorb(domains=src_val)
        elif src_type == "place":
            ctx.absorb(locations=src_val)
        else:
            ctx.absorb(names=src_val)

        cand_type_str = tgt_node.get("entity_type", "domain")
        cand_type = EntityType(cand_type_str) if cand_type_str in [e.value for e in EntityType] else EntityType.DOMAIN

        hit = Hit(
            target_entity_id=src_id,
            discovered_type=cand_type,
            discovered_value=tgt_node.get("value", ""),
            relation=edge.get("relation", "related"),
            source_tool=edge.get("source_tool", "unknown"),
            is_trusted=False,
            raw_data={"score": edge.get("score")},
            metadata={"source_node": src_id},
        )
        candidates.append((ctx, hit))

    return candidates


def run_labeler(input_path: str, output_path: str) -> None:
    in_file = Path(input_path)
    out_file = Path(output_path)

    if not in_file.exists():
        print(f"[ERROR] Input file {in_file} does not exist.", file=sys.stderr)
        sys.exit(1)

    if in_file.suffix.lower() == ".json":
        candidate_pairs = load_candidate_pairs_from_graph(in_file)
    else:
        # Load JSONL pairs
        candidate_pairs = []
        with open(in_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                ctx = TargetContext.from_dict(d.get("target_context", {}))
                hit = Hit.model_validate(d.get("hit", {}))
                candidate_pairs.append((ctx, hit))

    print(f"\nLoaded {len(candidate_pairs)} candidate pairs from {in_file}.")
    print("Commands: [y] Yes (Match) | [n] No (Mismatch) | [s] Skip | [q] Save & Quit\n" + "-" * 60)

    labeled_pairs: list[LabeledPair] = []

    for idx, (ctx, hit) in enumerate(candidate_pairs, 1):
        print(f"\n[Pair {idx}/{len(candidate_pairs)}]")
        print(f"  Target Context: Names={ctx.names} | Domains={ctx.domains} | Locations={ctx.locations}")
        print(f"  Candidate Hit:  [{hit.discovered_type.value}] '{hit.discovered_value}'")
        print(f"  Relation:       {hit.relation} (via tool: {hit.source_tool})")

        while True:
            try:
                choice = input("  Match? (y/n/s/q): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = "q"

            if choice in ("y", "yes"):
                labeled_pairs.append(
                    LabeledPair(
                        target_context=ctx.to_dict(),
                        hit=hit.model_dump(),
                        label=1,
                        notes="human_annotated:positive",
                    )
                )
                break
            elif choice in ("n", "no"):
                labeled_pairs.append(
                    LabeledPair(
                        target_context=ctx.to_dict(),
                        hit=hit.model_dump(),
                        label=0,
                        notes="human_annotated:negative",
                    )
                )
                break
            elif choice in ("s", "skip"):
                print("  Skipped.")
                break
            elif choice in ("q", "quit"):
                print("\nQuitting annotation session...")
                if labeled_pairs:
                    with open(out_file, "a", encoding="utf-8") as f:
                        for p in labeled_pairs:
                            f.write(json.dumps(p.to_dict(), default=str) + "\n")
                    print(f"Appended {len(labeled_pairs)} annotations to {out_file}.")
                sys.exit(0)
            else:
                print("  Invalid choice. Enter 'y', 'n', 's', or 'q'.")

    if labeled_pairs:
        with open(out_file, "a", encoding="utf-8") as f:
            for p in labeled_pairs:
                f.write(json.dumps(p.to_dict(), default=str) + "\n")
        print(f"\n[DONE] Successfully appended {len(labeled_pairs)} labeled pairs to {out_file}.\n")


def main() -> None:
    args = parse_args()
    run_labeler(args.input, args.output)


if __name__ == "__main__":
    main()
