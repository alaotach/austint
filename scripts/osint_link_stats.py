#!/usr/bin/env python3
"""Compute OSINT-Link dataset statistics, verify split leakage, and run manual label-noise audits."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from osint_agent.ml.datasets.builder import load_config, verify_no_person_leakage


def load_splits(data_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load exported JSONL splits from `data_dir`."""
    base = Path(data_dir)
    splits: dict[str, list[dict[str, Any]]] = {}
    for sname in ("train", "val", "test", "ood_platform_test"):
        fpath = base / f"{sname}.jsonl"
        rows: list[dict[str, Any]] = []
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        splits[sname] = rows
    return splits


def compute_stats(splits: dict[str, list[dict[str, Any]]], ood_platforms: set[str] | None = None) -> dict[str, Any]:
    """Compute counts, class balance, platform coverage, neg_type breakdown, and verify zero leakage."""
    verify_no_person_leakage(splits, ood_platforms=ood_platforms)

    per_split: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    platform_counts: Counter[str] = Counter()
    neg_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for sname, rows in splits.items():
        pos = sum(1 for r in rows if r["label"] == 1)
        neg = sum(1 for r in rows if r["label"] == 0)
        persons = {r["person_id_a"] for r in rows} | {r["person_id_b"] for r in rows}
        per_split[sname] = {
            "total": len(rows),
            "positives": pos,
            "negatives": neg,
            "pos_ratio": round(pos / len(rows), 4) if rows else 0.0,
            "unique_persons": len(persons),
        }
        all_rows.extend(rows)
        for r in rows:
            platform_counts[r["platform_a"]] += 1
            platform_counts[r["platform_b"]] += 1
            source_counts[r.get("label_source", "unknown")] += 1
            if r["label"] == 0 and r.get("neg_type"):
                neg_counts[r["neg_type"]] += 1

    total_pos = sum(v["positives"] for v in per_split.values())
    total_neg = sum(v["negatives"] for v in per_split.values())
    return {
        "total_pairs": len(all_rows),
        "total_positives": total_pos,
        "total_negatives": total_neg,
        "splits": per_split,
        "platform_coverage": dict(platform_counts.most_common()),
        "negative_types": dict(neg_counts),
        "label_sources": dict(source_counts),
        "person_leakage_verified": True,
    }


def run_sample_check(splits: dict[str, list[dict[str, Any]]], sample_size: int = 100, seed: int = 42, out_path: Path | None = None) -> dict[str, Any]:
    """Interactive CLI to manually audit `sample_size` pairs and estimate weak-supervision label noise."""
    rng = random.Random(seed)
    pool = [r for rows in splits.values() for r in rows]
    if not pool:
        print("No pairs available to check.")
        return {"checked": 0, "noisy": 0, "noise_rate": 0.0}

    sample = rng.sample(pool, min(sample_size, len(pool)))
    noisy = 0
    checked = 0
    decisions: list[dict[str, Any]] = []

    print(f"\n=== Manual Label-Noise Audit ({len(sample)} pairs) ===")
    print("Enter [y] label is correct, [n] noisy/incorrect, [q] quit early\n")
    for idx, row in enumerate(sample, 1):
        print(
            f"[{idx}/{len(sample)}] label={row['label']} ({row.get('label_source')}) | "
            f"{row['platform_a']}:{row['username_a']} ({row['display_name_a']}) <-> "
            f"{row['platform_b']}:{row['username_b']} ({row['display_name_b']}) | "
            f"bio_sim={row.get('bio_emb_sim')} phash_sim={row.get('phash_sim')} links_jaccard={row.get('links_jaccard')}"
        )
        ans = input("Correct? [y/n/q]: ").strip().lower()
        if ans == "q":
            break
        is_noisy = ans == "n"
        if is_noisy:
            noisy += 1
        checked += 1
        decisions.append({"pair_id": row["pair_id"], "label": row["label"], "noisy": is_noisy})

    noise_rate = round(noisy / checked, 4) if checked else 0.0
    report = {"checked": checked, "noisy": noisy, "estimated_label_noise_rate": noise_rate, "decisions": decisions}
    if out_path is not None:
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nAudit complete: {noisy}/{checked} noisy (estimated noise rate = {noise_rate:.2%})")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect OSINT-Link dataset statistics and label noise.")
    parser.add_argument("--config", default="configs/osint_link.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--sample-check", type=int, default=0, help="Interactively check N random pairs (e.g. 100).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data_dir or cfg["dataset"]["output_dir"])
    ood_plats = set(cfg["splits"].get("ood_platforms") or [])
    splits = load_splits(data_dir)
    stats = compute_stats(splits, ood_platforms=ood_plats)
    print(json.dumps(stats, indent=2))

    if args.sample_check > 0:
        run_sample_check(splits, sample_size=args.sample_check, seed=int(cfg.get("seed", 42)), out_path=data_dir / "label_noise_audit.json")


if __name__ == "__main__":
    main()
