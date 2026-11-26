#!/usr/bin/env python3
"""Phase G: Comprehensive validation & stress-testing script."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osint_agent.ml.baselines import (
    HistGBMResolver,
    compute_metrics,
    pair_dict_to_feature_vector,
    score_rule_baseline,
)
from osint_agent.ml.gnn import (
    NeuralGNNResolver,
    _predict_probs,
    build_dataset_graph,
    evaluate_three_way,
    train_gnn_model,
    verify_split_graph_isolation,
)
from osint_agent.ml.neural import NeuralResolver
from osint_agent.ml.resolver import BaselineWeightedResolver, LearnedResolver
from osint_agent.ml.validation import (
    evaluate_by_negative_type,
    evaluate_shortcut_baselines,
    paired_person_bootstrap,
    run_llm_judge_two_variants,
    tune_threshold_on_val,
)
from scripts.train_gnn import _load_jsonl


def run_validation(
    *,
    data_dir: Path,
    neural_model_path: Path,
    gnn_model_out_path: Path,
    seed: int = 42,
    n_bootstrap: int = 1000,
    run_llm: bool = True,
) -> dict[str, Any]:
    train_rows = _load_jsonl(data_dir / "train.jsonl")
    val_rows = _load_jsonl(data_dir / "val.jsonl")
    test_rows = _load_jsonl(data_dir / "test.jsonl")
    ood_rows = _load_jsonl(data_dir / "ood_platform_test.jsonl")

    y_train = np.array([int(r["label"]) for r in train_rows], dtype=int)
    y_val = np.array([int(r["label"]) for r in val_rows], dtype=int)
    y_test = np.array([int(r["label"]) for r in test_rows], dtype=int)
    y_ood = np.array([int(r["label"]) for r in ood_rows], dtype=int)

    # Fit Phase B models on train
    X_train = np.vstack([pair_dict_to_feature_vector(r) for r in train_rows])
    X_test = np.vstack([pair_dict_to_feature_vector(r) for r in test_rows])
    X_ood = np.vstack([pair_dict_to_feature_vector(r) for r in ood_rows])

    rule_res = BaselineWeightedResolver()
    rule_val_probs = np.array([score_rule_baseline(r, rule_res) for r in val_rows], dtype=np.float64)
    rule_thresh = tune_threshold_on_val(y_val, rule_val_probs)
    rule_test_probs = np.array([score_rule_baseline(r, rule_res) for r in test_rows], dtype=np.float64)
    rule_ood_probs = np.array([score_rule_baseline(r, rule_res) for r in ood_rows], dtype=np.float64)

    logreg = LearnedResolver().fit(X_train, y_train)
    lr_test_probs = logreg.model.predict_proba(X_test)[:, 1]
    lr_ood_probs = logreg.model.predict_proba(X_ood)[:, 1]

    hgb = HistGBMResolver(seed=seed).fit(X_train, y_train)
    hgb_test_probs = hgb.predict_proba_matrix(X_test)
    hgb_ood_probs = hgb.predict_proba_matrix(X_ood)

    # Phase C Neural scores
    neural_res = NeuralResolver.load(neural_model_path)
    neural_train_probs = neural_res.predict_proba_rows(train_rows)
    neural_val_probs = neural_res.predict_proba_rows(val_rows)
    neural_test_probs = neural_res.predict_proba_rows(test_rows)
    neural_ood_probs = neural_res.predict_proba_rows(ood_rows)

    # 1. GNN Leakage Audit & Retraining
    old_gnn_eval_path = data_dir / "gnn_eval.json"
    old_gnn_eval = json.loads(old_gnn_eval_path.read_text(encoding="utf-8")) if old_gnn_eval_path.exists() else {}

    def _run_gnn_variant(*, candidate_only_edges: bool) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tr_g = build_dataset_graph(
            train_rows, neural_train_probs, exclude_pair_forming_edges=True, candidate_only_edges=candidate_only_edges
        )
        va_g = build_dataset_graph(
            val_rows, neural_val_probs, exclude_pair_forming_edges=True, candidate_only_edges=candidate_only_edges
        )
        te_g = build_dataset_graph(
            test_rows, neural_test_probs, exclude_pair_forming_edges=True, candidate_only_edges=candidate_only_edges
        )
        oo_g = build_dataset_graph(
            ood_rows, neural_ood_probs, exclude_pair_forming_edges=True, candidate_only_edges=candidate_only_edges
        )
        verify_split_graph_isolation({"train": tr_g, "val": va_g, "test": te_g, "ood_platform_test": oo_g})
        trained = train_gnn_model(tr_g, va_g, seed=seed, epochs=80)
        g_model = trained["model"]
        g_temp = trained["temperature"]
        te_res = evaluate_three_way(te_g, g_model, g_temp, n_bootstrap=200, seed=seed)
        oo_res = evaluate_three_way(oo_g, g_model, g_temp, n_bootstrap=200, seed=seed)
        return (
            {
                "test": te_res["phase_c_plus_gnn"],
                "ood_platform_test": oo_res["phase_c_plus_gnn"],
                "test_graph_stats": te_res["graph_stats"],
                "ood_graph_stats": oo_res["graph_stats"],
            },
            _predict_probs(g_model, te_g, g_temp),
            _predict_probs(g_model, oo_g, g_temp),
            va_g.chain_scores.numpy(),
            te_g.chain_scores.numpy(),
            oo_g.chain_scores.numpy(),
            g_model,
            g_temp,
        )

    audited_res, gnn_test_probs, gnn_ood_probs, chain_val_probs, chain_test_probs, chain_ood_probs, g_model, g_temp = _run_gnn_variant(
        candidate_only_edges=False
    )
    cand_only_res, gnn_cand_test_probs, gnn_cand_ood_probs, *_ = _run_gnn_variant(
        candidate_only_edges=True
    )

    # Save audited GNN checkpoint
    NeuralGNNResolver(neural_res, g_model, temperature=g_temp).save(gnn_model_out_path)

    gnn_leakage_audit = {
        "split_isolation_verified": True,
        "pair_forming_edges_masked": True,
        "old_gnn": {
            "test": (old_gnn_eval.get("splits") or {}).get("test", {}).get("phase_c_plus_gnn", {}),
            "ood_platform_test": (old_gnn_eval.get("splits") or {}).get("ood_platform_test", {}).get("phase_c_plus_gnn", {}),
        },
        "new_gnn_audited": {
            "test": audited_res["test"],
            "ood_platform_test": audited_res["ood_platform_test"],
            "test_graph_stats": audited_res["test_graph_stats"],
            "ood_graph_stats": audited_res["ood_graph_stats"],
        },
        "new_gnn_candidate_only_edges": {
            "test": cand_only_res["test"],
            "ood_platform_test": cand_only_res["ood_platform_test"],
            "test_graph_stats": cand_only_res["test_graph_stats"],
            "ood_graph_stats": cand_only_res["ood_graph_stats"],
        },
    }

    # 2. Paired Bootstrap (1000 iterations, clustered by person_id)
    paired_bootstrap = {
        "test": {
            "GNN vs HistGBM": paired_person_bootstrap(
                test_rows, gnn_test_probs, hgb_test_probs, n_bootstrap=n_bootstrap, seed=seed
            ),
            "Neural vs HistGBM": paired_person_bootstrap(
                test_rows, neural_test_probs, hgb_test_probs, n_bootstrap=n_bootstrap, seed=seed
            ),
            "GNN vs Neural": paired_person_bootstrap(
                test_rows, gnn_test_probs, neural_test_probs, n_bootstrap=n_bootstrap, seed=seed
            ),
        },
        "ood_platform_test": {
            "GNN vs HistGBM": paired_person_bootstrap(
                ood_rows, gnn_ood_probs, hgb_ood_probs, n_bootstrap=n_bootstrap, seed=seed
            ),
            "Neural vs HistGBM": paired_person_bootstrap(
                ood_rows, neural_ood_probs, hgb_ood_probs, n_bootstrap=n_bootstrap, seed=seed
            ),
            "GNN vs Neural": paired_person_bootstrap(
                ood_rows, gnn_ood_probs, neural_ood_probs, n_bootstrap=n_bootstrap, seed=seed
            ),
        },
    }

    # 3. Shortcut Baselines
    shortcut_baselines = evaluate_shortcut_baselines(
        train_rows,
        {"test": test_rows, "ood_platform_test": ood_rows},
        seed=seed,
    )

    # 4. Breakdown by Negative Type
    neg_breakdown = {
        "test": evaluate_by_negative_type(
            test_rows,
            {
                "rule_baseline": rule_test_probs,
                "logistic_regression": lr_test_probs,
                "hist_gradient_boosting": hgb_test_probs,
                "phase_c_neural": neural_test_probs,
                "phase_c_plus_gnn": gnn_test_probs,
            },
            thresholds={"rule_baseline": rule_thresh},
        ),
        "ood_platform_test": evaluate_by_negative_type(
            ood_rows,
            {
                "rule_baseline": rule_ood_probs,
                "logistic_regression": lr_ood_probs,
                "hist_gradient_boosting": hgb_ood_probs,
                "phase_c_neural": neural_ood_probs,
                "phase_c_plus_gnn": gnn_ood_probs,
            },
            thresholds={"rule_baseline": rule_thresh},
        ),
    }

    # 5. Val-tuned Chain-Confidence Baseline
    chain_thresh = tune_threshold_on_val(y_val, chain_val_probs)
    chain_eval = {
        "val_tuned_threshold": chain_thresh,
        "test": {
            **compute_metrics(y_test, chain_test_probs, threshold=chain_thresh),
            "mean_pos_score": round(float(np.mean(chain_test_probs[y_test == 1])), 4),
            "mean_neg_score": round(float(np.mean(chain_test_probs[y_test == 0])), 4),
            "pct_below_0_1": round(float(np.mean(chain_test_probs < 0.1)), 4),
        },
        "ood_platform_test": {
            **compute_metrics(y_ood, chain_ood_probs, threshold=chain_thresh),
            "mean_pos_score": round(float(np.mean(chain_ood_probs[y_ood == 1])), 4),
            "mean_neg_score": round(float(np.mean(chain_ood_probs[y_ood == 0])), 4),
            "pct_below_0_1": round(float(np.mean(chain_ood_probs < 0.1)), 4),
        },
    }

    # 6. Zero-Shot LLM Judge (2 variants on test + ood_platform_test)
    if run_llm:
        llm_judge = run_llm_judge_two_variants(
            test_rows + ood_rows,
            cache_dir=data_dir / "llm_cache_phase_g",
        )
    else:
        llm_judge = {"status": "skipped", "variants": {}}

    return {
        "seed": seed,
        "gnn_leakage_audit": gnn_leakage_audit,
        "paired_bootstrap": paired_bootstrap,
        "shortcut_baselines": shortcut_baselines,
        "negative_type_breakdown": neg_breakdown,
        "chain_confidence_tuned": chain_eval,
        "zero_shot_llm_judge": llm_judge,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase G validation & stress tests")
    parser.add_argument("--config", type=str, default="configs/osint_link.yaml")
    parser.add_argument("--neural-model", type=str, default="models/neural_resolver.pt")
    parser.add_argument("--gnn-model-out", type=str, default="models/neural_gnn_resolver.pt")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_dir = Path(cfg["dataset"]["output_dir"])
    seed = int(cfg["dataset"].get("seed", 42))
    out_path = Path(args.out) if args.out else (data_dir / "validation_eval.json")

    report = run_validation(
        data_dir=data_dir,
        neural_model_path=Path(args.neural_model),
        gnn_model_out_path=Path(args.gnn_model_out),
        seed=seed,
        n_bootstrap=args.n_bootstrap,
        run_llm=not args.skip_llm,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
