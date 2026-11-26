#!/usr/bin/env python3
"""Train Phase C Multimodal Identity-Linking Model, run ablations, calibrate, and compare against Phase B."""

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

from osint_agent.ml.baselines import bootstrap_metrics_ci
from osint_agent.ml.datasets.builder import load_config, verify_no_person_leakage
from osint_agent.ml.neural import NeuralResolver, train_multimodal_model
from scripts.osint_link_stats import load_splits

ABLATION_CONFIGS: list[tuple[str, dict[str, bool]]] = [
    ("full_model", {"use_username": True, "use_bio_loc": True, "use_photo": True, "use_links": True, "use_contrastive": True, "use_hard_negatives": True}),
    ("no_username_char_cnn", {"use_username": False, "use_bio_loc": True, "use_photo": True, "use_links": True, "use_contrastive": True, "use_hard_negatives": True}),
    ("no_bio_location", {"use_username": True, "use_bio_loc": False, "use_photo": True, "use_links": True, "use_contrastive": True, "use_hard_negatives": True}),
    ("no_photo_clip_phash", {"use_username": True, "use_bio_loc": True, "use_photo": False, "use_links": True, "use_contrastive": True, "use_hard_negatives": True}),
    ("no_links_overlap", {"use_username": True, "use_bio_loc": True, "use_photo": True, "use_links": False, "use_contrastive": True, "use_hard_negatives": True}),
    ("no_contrastive_pretrain", {"use_username": True, "use_bio_loc": True, "use_photo": True, "use_links": True, "use_contrastive": False, "use_hard_negatives": True}),
    ("no_hard_negatives", {"use_username": True, "use_bio_loc": True, "use_photo": True, "use_links": True, "use_contrastive": True, "use_hard_negatives": False}),
]


def run_neural_training_and_ablations(
    data_dir: str | Path,
    model_out_path: str | Path = "models/neural_resolver.pt",
    seed: int = 42,
    epochs_contrastive: int = 8,
    epochs_cls: int = 15,
    n_bootstrap: int = 500,
    run_ablations: bool = True,
    ood_platforms: set[str] | None = None,
) -> dict[str, Any]:
    """Train full multimodal model, calibrate on val, run ablations, and evaluate on test + OOD splits."""
    splits = load_splits(data_dir)
    verify_no_person_leakage(splits, ood_platforms=ood_platforms)

    train_rows = splits.get("train", [])
    val_rows = splits.get("val", [])
    if len(train_rows) < 4 or len({r["label"] for r in train_rows}) < 2:
        return {
            "status": "skipped_insufficient_train_data",
            "train_pairs": len(train_rows),
            "message": "Cannot train neural model on empty/single-class train split. Run build_osint_link.py with GITHUB_TOKEN first.",
        }

    # 1. Train Full Model
    model, temp, cal_report = train_multimodal_model(
        train_rows=train_rows,
        val_rows=val_rows,
        seed=seed,
        epochs_contrastive=epochs_contrastive,
        epochs_cls=epochs_cls,
    )
    resolver = NeuralResolver(model=model, temperature=temp, bio_dim=model.bio_dim, clip_dim=model.clip_dim)
    resolver.save(model_out_path)

    split_eval: dict[str, Any] = {}
    for sname in ("test", "ood_platform_test"):
        rows = splits.get(sname, [])
        if not rows:
            split_eval[sname] = {"count": 0, "metrics": {}}
            continue
        y_true = np.array([int(r["label"]) for r in rows], dtype=int)
        t0 = time.perf_counter()
        probs = resolver.predict_proba_rows(rows)
        lat_ms = round(((time.perf_counter() - t0) * 1000.0) / len(rows), 4)
        split_eval[sname] = {
            "count": len(rows),
            "latency_ms_per_pair": lat_ms,
            "metrics": bootstrap_metrics_ci(y_true, probs, n_bootstrap=n_bootstrap, seed=seed),
        }

    ablations_table: dict[str, Any] = {}
    if run_ablations:
        for ab_name, flags in ABLATION_CONFIGS:
            ab_model, ab_temp, ab_cal = train_multimodal_model(
                train_rows=train_rows,
                val_rows=val_rows,
                seed=seed,
                epochs_contrastive=epochs_contrastive,
                epochs_cls=epochs_cls,
                **flags,
            )
            ab_res = NeuralResolver(model=ab_model, temperature=ab_temp, bio_dim=ab_model.bio_dim, clip_dim=ab_model.clip_dim)
            ab_entry: dict[str, Any] = {"val_ece_before": ab_cal["val_ece_before"], "val_ece_after": ab_cal["val_ece_after"]}
            for sname in ("test", "ood_platform_test"):
                rows = splits.get(sname, [])
                if rows:
                    yt = np.array([int(r["label"]) for r in rows], dtype=int)
                    pr = ab_res.predict_proba_rows(rows)
                    ab_entry[sname] = bootstrap_metrics_ci(yt, pr, n_bootstrap=min(200, n_bootstrap), seed=seed)
            ablations_table[ab_name] = ab_entry

    return {
        "status": "completed",
        "seed": seed,
        "model_path": str(model_out_path),
        "calibration": cal_report,
        "splits": split_eval,
        "ablations": ablations_table,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase C multimodal identity-linking model & ablations.")
    parser.add_argument("--config", default="configs/osint_link.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--model-out", default="models/neural_resolver.pt")
    parser.add_argument("--epochs-contrastive", type=int, default=8)
    parser.add_argument("--epochs-cls", type=int, default=15)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data_dir or cfg["dataset"]["output_dir"])
    ood_plats = set(cfg["splits"].get("ood_platforms") or [])

    res = run_neural_training_and_ablations(
        data_dir=data_dir,
        model_out_path=args.model_out,
        seed=int(cfg.get("seed", 42)),
        epochs_contrastive=args.epochs_contrastive,
        epochs_cls=args.epochs_cls,
        n_bootstrap=args.n_bootstrap,
        run_ablations=not args.no_ablations,
        ood_platforms=ood_plats,
    )
    out_path = Path(args.output or (data_dir / "neural_eval.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
