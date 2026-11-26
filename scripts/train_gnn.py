#!/usr/bin/env python3
"""Phase D training and 3-way evaluation script for CollectiveGNN.

Compares:
  1. Phase C alone (MultimodalIdentityNet)
  2. Phase C + GNN (CollectiveGNN)
  3. Old "confidence multiplied along the chain" heuristic
on both the person-held-out test split and the new-platform OOD split,
including a helps-vs-hurts diagnostic breakdown.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml

from osint_agent.ml.gnn import (
    NeuralGNNResolver,
    build_dataset_graph,
    evaluate_three_way,
    train_gnn_model,
)
from osint_agent.ml.neural import (
    NeuralResolver,
    train_multimodal_model,
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


def run_gnn_training_and_eval(
    *,
    data_dir: Path,
    neural_model_path: Path,
    gnn_model_out_path: Path,
    seed: int = 42,
    epochs_gnn: int = 80,
    n_bootstrap: int = 200,
    confidence_decay: float = 0.85,
) -> dict[str, Any]:
    train_rows = _load_jsonl(data_dir / "train.jsonl")
    val_rows = _load_jsonl(data_dir / "val.jsonl")
    test_rows = _load_jsonl(data_dir / "test.jsonl")
    ood_rows = _load_jsonl(data_dir / "ood_platform_test.jsonl")

    y_train = np.array([int(r.get("label", 0)) for r in train_rows], dtype=int)
    if len(train_rows) == 0 or len(np.unique(y_train)) < 2:
        return {
            "status": "skipped_insufficient_train_data",
            "train_pairs": len(train_rows),
            "message": (
                "Cannot train CollectiveGNN on empty/single-class train split. "
                "Run build_osint_link.py with GITHUB_TOKEN first."
            ),
        }

    # Load or train Phase C resolver
    if neural_model_path.exists():
        neural_res = NeuralResolver.load(neural_model_path)
    else:
        phase_c_model, phase_c_temp, _ = train_multimodal_model(
            train_rows,
            val_rows,
            seed=seed,
            epochs_contrastive=8,
            epochs_cls=12,
        )
        neural_res = NeuralResolver(
            model=phase_c_model,
            temperature=phase_c_temp,
            bio_dim=phase_c_model.bio_dim,
            clip_dim=phase_c_model.clip_dim,
        )
        neural_res.save(neural_model_path)

    def _score_rows(rows: list[dict[str, Any]]) -> np.ndarray:
        if not rows:
            return np.zeros((0,), dtype=np.float32)
        return neural_res.predict_proba_rows(rows)

    train_scores = _score_rows(train_rows)
    val_scores = _score_rows(val_rows)
    test_scores = _score_rows(test_rows)
    ood_scores = _score_rows(ood_rows)

    train_graph = build_dataset_graph(train_rows, train_scores, confidence_decay=confidence_decay)
    val_graph = build_dataset_graph(val_rows, val_scores, confidence_decay=confidence_decay)
    test_graph = build_dataset_graph(test_rows, test_scores, confidence_decay=confidence_decay)
    ood_graph = build_dataset_graph(ood_rows, ood_scores, confidence_decay=confidence_decay)

    gnn_res = train_gnn_model(
        train_graph,
        val_graph,
        seed=seed,
        epochs=epochs_gnn,
    )
    gnn_model = gnn_res["model"]
    gnn_temp = gnn_res["temperature"]

    resolver = NeuralGNNResolver(
        neural_resolver=neural_res,
        gnn_model=gnn_model,
        temperature=gnn_temp,
        confidence_decay=confidence_decay,
    )
    resolver.save(gnn_model_out_path)

    splits: dict[str, Any] = {}
    if test_rows:
        splits["test"] = evaluate_three_way(
            test_graph, gnn_model, gnn_temp, n_bootstrap=n_bootstrap, seed=seed
        )
    if ood_rows:
        splits["ood_platform_test"] = evaluate_three_way(
            ood_graph, gnn_model, gnn_temp, n_bootstrap=n_bootstrap, seed=seed
        )

    return {
        "status": "ok",
        "seed": seed,
        "checkpoint": str(gnn_model_out_path),
        "calibration": {
            "temperature": round(gnn_temp, 4),
            "val_ece_before": gnn_res["val_ece_before"],
            "val_ece_after": gnn_res["val_ece_after"],
            "val_reliability_bins": gnn_res["val_reliability_bins"],
        },
        "splits": splits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train & evaluate Phase D CollectiveGNN")
    parser.add_argument("--config", type=str, default="configs/osint_link.yaml")
    parser.add_argument("--neural-model", type=str, default="models/neural_resolver.pt")
    parser.add_argument("--model-out", type=str, default="models/neural_gnn_resolver.pt")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--epochs-gnn", type=int, default=80)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_dir = Path(cfg["dataset"]["output_dir"])
    seed = int(cfg["dataset"].get("seed", 42))
    out_path = Path(args.out) if args.out else (data_dir / "gnn_eval.json")

    report = run_gnn_training_and_eval(
        data_dir=data_dir,
        neural_model_path=Path(args.neural_model),
        gnn_model_out_path=Path(args.model_out),
        seed=seed,
        epochs_gnn=args.epochs_gnn,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
