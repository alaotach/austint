#!/usr/bin/env python3
"""Evaluate Phase B Baselines (Rule, Monotonic LogReg, Monotonic HistGBM, Zero-Shot LLM Judge) with 95% Bootstrap CIs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from osint_agent.llm.client import FakeClient, get_llm_client
from osint_agent.ml.baselines import (
    HistGBMResolver,
    ZeroShotLLMJudge,
    bootstrap_metrics_ci,
    pair_dict_to_feature_vector,
    score_rule_baseline,
)
from osint_agent.ml.datasets.builder import load_config, verify_no_person_leakage
from osint_agent.ml.resolver import BaselineWeightedResolver, LearnedResolver
from scripts.osint_link_stats import load_splits


def evaluate_baselines(
    data_dir: str | Path,
    seed: int = 42,
    llm_mode: str = "recorded",
    llm_cache_path: str | Path | None = None,
    llm_recorded_path: str | Path | None = None,
    llm_max_calls: int = 100,
    n_bootstrap: int = 500,
    ood_platforms: set[str] | None = None,
) -> dict[str, Any]:
    """Run all 4 baselines on person-held-out `test` and `ood_platform_test` splits."""
    splits = load_splits(data_dir)
    verify_no_person_leakage(splits, ood_platforms=ood_platforms)

    train_rows = splits.get("train", [])
    X_train = np.vstack([pair_dict_to_feature_vector(r) for r in train_rows]) if train_rows else np.empty((0, 9))
    y_train = np.array([int(r["label"]) for r in train_rows], dtype=int) if train_rows else np.empty((0,), dtype=int)
    can_fit_supervised = len(train_rows) >= 4 and len(np.unique(y_train)) >= 2

    rule_resolver = BaselineWeightedResolver()
    logreg_resolver: LearnedResolver | None = None
    hgb_resolver: HistGBMResolver | None = None

    if can_fit_supervised:
        logreg_resolver = LearnedResolver().fit(X_train, y_train)
        hgb_resolver = HistGBMResolver(seed=seed).fit(X_train, y_train)

    llm_client = None
    if llm_mode == "live":
        llm_client = get_llm_client(provider="openrouter", call_budget=llm_max_calls)
    elif llm_mode == "fake":
        llm_client = FakeClient(responses=[{"match_probability": 0.5, "rationale": "offline"} for _ in range(max(1, llm_max_calls))], call_budget=llm_max_calls)

    judge = ZeroShotLLMJudge(
        client=llm_client,
        cache_path=llm_cache_path or (Path(data_dir) / "llm_judge_cache.jsonl"),
        recorded_path=llm_recorded_path or (Path(data_dir) / "llm_judge_recorded.jsonl"),
        max_calls=llm_max_calls,
    )

    results: dict[str, Any] = {
        "seed": seed,
        "can_fit_supervised": can_fit_supervised,
        "train_pairs": len(train_rows),
        "splits": {},
    }

    for split_name in ("test", "ood_platform_test"):
        rows = splits.get(split_name, [])
        if not rows:
            results["splits"][split_name] = {"count": 0, "models": {}}
            continue

        y_true = np.array([int(r["label"]) for r in rows], dtype=int)
        X_split = np.vstack([pair_dict_to_feature_vector(r) for r in rows])
        split_res: dict[str, Any] = {}

        # 1. Rule Baseline
        t0 = time.perf_counter()
        probs_rule = np.array([score_rule_baseline(r, rule_resolver) for r in rows], dtype=np.float64)
        lat_rule = round(((time.perf_counter() - t0) * 1000.0) / len(rows), 4)
        split_res["rule_baseline"] = {
            "metrics": bootstrap_metrics_ci(y_true, probs_rule, n_bootstrap=n_bootstrap, seed=seed),
            "latency_ms_per_pair": lat_rule,
        }

        # 2. Monotonic LogisticRegression
        if logreg_resolver is not None:
            t0 = time.perf_counter()
            probs_lr = logreg_resolver.model.predict_proba(X_split)[:, 1]
            lat_lr = round(((time.perf_counter() - t0) * 1000.0) / len(rows), 4)
            split_res["logistic_regression"] = {
                "metrics": bootstrap_metrics_ci(y_true, probs_lr, n_bootstrap=n_bootstrap, seed=seed),
                "latency_ms_per_pair": lat_lr,
            }

        # 3. Monotonic HistGradientBoosting
        if hgb_resolver is not None:
            t0 = time.perf_counter()
            probs_hgb = hgb_resolver.predict_proba_matrix(X_split)
            lat_hgb = round(((time.perf_counter() - t0) * 1000.0) / len(rows), 4)
            split_res["hist_gradient_boosting"] = {
                "metrics": bootstrap_metrics_ci(y_true, probs_hgb, n_bootstrap=n_bootstrap, seed=seed),
                "latency_ms_per_pair": lat_hgb,
            }

        # 4. Zero-Shot LLM Judge (only if recorded/cached/live/fake produces non-stub or explicitly requested)
        t0 = time.perf_counter()
        j_results = [judge.score_pair(r) for r in rows]
        lat_llm = round(((time.perf_counter() - t0) * 1000.0) / len(rows), 4)
        probs_llm = np.array([jr.probability for jr in j_results], dtype=np.float64)
        split_res["zero_shot_llm_judge"] = {
            "metrics": bootstrap_metrics_ci(y_true, probs_llm, n_bootstrap=n_bootstrap, seed=seed),
            "latency_ms_per_pair": lat_llm,
            "total_tokens": sum(jr.tokens for jr in j_results),
            "total_cost_usd": round(sum(jr.cost_usd for jr in j_results), 6),
        }

        results["splits"][split_name] = {"count": len(rows), "models": split_res}

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase B baseline models on OSINT-Link splits.")
    parser.add_argument("--config", default="configs/osint_link.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--llm-mode", choices=["recorded", "fake", "live"], default="recorded")
    parser.add_argument("--llm-max-calls", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data_dir or cfg["dataset"]["output_dir"])
    ood_plats = set(cfg["splits"].get("ood_platforms") or [])
    res = evaluate_baselines(
        data_dir=data_dir,
        seed=int(cfg.get("seed", 42)),
        llm_mode=args.llm_mode,
        llm_max_calls=args.llm_max_calls,
        n_bootstrap=args.n_bootstrap,
        ood_platforms=ood_plats,
    )
    out_path = Path(args.output or (data_dir / "baseline_eval.json"))
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
