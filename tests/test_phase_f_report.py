"""Offline tests for Phase F: report asset generation and traceable RESULTS.md."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.make_report_assets import generate_report_assets


def test_make_report_assets_full_fixture(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    fig_dir = tmp_path / "report" / "figures"
    results_md = tmp_path / "report" / "RESULTS.md"

    (data_dir / "manifest.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "ood_platforms": ["mastodon"],
                "total_profiles_crawled": 120,
                "store_raw_bios": False,
                "store_raw_photos": False,
                "counts": {
                    "train": {"total": 100, "positives": 40, "negatives": 60},
                    "val": {"total": 20, "positives": 8, "negatives": 12},
                    "test": {"total": 30, "positives": 12, "negatives": 18},
                    "ood_platform_test": {"total": 30, "positives": 10, "negatives": 20},
                },
            }
        ),
        encoding="utf-8",
    )
    (data_dir / "baseline_eval.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "splits": {
                    "ood_platform_test": {
                        "models": {
                            "rule_baseline": {
                                "metrics": {
                                    "precision": {"point": 0.9, "ci_low": 0.8, "ci_high": 1.0},
                                    "recall": {"point": 0.3, "ci_low": 0.2, "ci_high": 0.4},
                                    "f1": {"point": 0.45, "ci_low": 0.3, "ci_high": 0.6},
                                    "pr_auc": {"point": 0.72, "ci_low": 0.6, "ci_high": 0.84},
                                    "ece": {"point": 0.12, "ci_low": 0.05, "ci_high": 0.2},
                                },
                                "latency_ms_per_pair": 0.01,
                            }
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (data_dir / "neural_eval.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "calibration": {
                    "val_reliability_bins": [
                        {"avg_confidence": 0.2, "empirical_accuracy": 0.22, "count": 5},
                        {"avg_confidence": 0.8, "empirical_accuracy": 0.78, "count": 5},
                    ]
                },
                "splits": {
                    "ood_platform_test": {"precision": 0.85, "recall": 0.8, "f1": 0.82, "pr_auc": 0.88, "ece": 0.04}
                },
                "ablations": {
                    "full_model": {"test": {"pr_auc": 0.9, "f1": 0.85, "ece": 0.03}},
                    "no_photo_clip_phash": {"test": {"pr_auc": 0.84, "f1": 0.79, "ece": 0.06}},
                },
            }
        ),
        encoding="utf-8",
    )
    (data_dir / "gnn_eval.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "splits": {
                    "test": {
                        "phase_c_alone": {"pr_auc": 0.88},
                        "phase_c_plus_gnn": {"pr_auc": 0.91},
                        "chain_confidence_heuristic": {"pr_auc": 0.74},
                        "helps_vs_hurts": {"helped_count": 3, "hurt_count": 1},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (data_dir / "bandit_eval.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "budgets": [1, 2, 4],
                "curves": {
                    "BFS": [{"budget": 1, "recall": 0.2}, {"budget": 4, "recall": 0.8}],
                    "BanditPlanner": [{"budget": 1, "recall": 0.6}, {"budget": 4, "recall": 0.95}],
                },
                "auc_recall": {"BFS": 0.5, "BanditPlanner": 0.8},
            }
        ),
        encoding="utf-8",
    )

    out = generate_report_assets(
        data_dir=data_dir,
        figures_dir=fig_dir,
        results_md_path=results_md,
    )
    assert out["status"] == "ok"
    for fname in ("pr_curves.png", "reliability_diagram.png", "budget_vs_recall.png", "ablation_table.md"):
        p = fig_dir / fname
        assert p.exists() and p.stat().st_size > 0
    txt = results_md.read_text(encoding="utf-8")
    assert "0.7200" in txt
    assert "BanditPlanner" in txt


def test_make_report_assets_on_real_data_dir(tmp_path: Path):
    fig_dir = tmp_path / "figures"
    results_md = tmp_path / "RESULTS.md"
    out = generate_report_assets(
        data_dir=Path("data/osint_link"),
        figures_dir=fig_dir,
        results_md_path=results_md,
    )
    assert out["status"] == "ok"
    assert results_md.exists()
    assert "0.7140" in results_md.read_text(encoding="utf-8")
