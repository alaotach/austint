"""Training and evaluation script comparing Baseline vs LearnedResolver on synthetic pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from osint_agent.ml.dataset import build_xy, generate_synthetic_pairs
from osint_agent.ml.features import FEATURE_NAMES, TargetContext
from osint_agent.ml.resolver import BaselineWeightedResolver, LearnedResolver
from osint_agent.core.schema import Hit


def run_experiment(
    n_pairs: int = 500,
    test_size: float = 0.25,
    seed: int = 42,
    ood_seed: int = 999,
    model_out: str | Path | None = "model.joblib",
) -> None:
    print(f"\n========================================================")
    print(f"  Intelligent OSINT: Resolver Benchmark & Leakage Check")
    print(f"  [NOTE: All data generated below is synthetic, not for the report]")
    print(f"========================================================")
    print(f"[*] Generating {n_pairs} in-distribution synthetic pairs (seed={seed})...")
    pairs_id = generate_synthetic_pairs(n=n_pairs, seed=seed, ood=False, noise_rate=0.05)

    print("[*] Extracting 9-feature vectors for in-distribution data...")
    X_id, y_id, _ = build_xy(pairs_id)

    X_train, X_test_id, y_train, y_test_id, pairs_train, pairs_test_id = train_test_split(
        X_id, y_id, pairs_id, test_size=test_size, random_state=seed, stratify=y_id
    )

    print(f"[*] In-Distribution Split: {len(X_train)} training pairs, {len(X_test_id)} held-out test pairs.")

    # Generate Out-of-Distribution (OOD) test set
    n_ood = len(X_test_id)
    print(f"[*] Generating {n_ood} out-of-distribution synthetic pairs (seed={ood_seed}, distinct templates)...")
    pairs_ood = generate_synthetic_pairs(n=n_ood, seed=ood_seed, ood=True, noise_rate=0.05)
    X_test_ood, y_test_ood, _ = build_xy(pairs_ood)

    # 1. Fit Learned Resolver
    print("[*] Fitting LearnedResolver (LogisticRegression with balanced class weights)...")
    learned = LearnedResolver(random_state=seed)
    learned.fit(X_train, y_train)

    # 2. Evaluate on In-Distribution Held-Out Split
    baseline = BaselineWeightedResolver()
    b_preds_id = [
        1 if baseline.score(TargetContext.from_dict(p.target_context), Hit.model_validate(p.hit)) >= 0.5 else 0
        for p in pairs_test_id
    ]
    l_preds_id = (learned.model.predict_proba(X_test_id)[:, 1] >= 0.5).astype(int)

    b_prec_id = precision_score(y_test_id, b_preds_id, zero_division=0)
    b_rec_id = recall_score(y_test_id, b_preds_id, zero_division=0)
    b_f1_id = f1_score(y_test_id, b_preds_id, zero_division=0)

    l_prec_id = precision_score(y_test_id, l_preds_id, zero_division=0)
    l_rec_id = recall_score(y_test_id, l_preds_id, zero_division=0)
    l_f1_id = f1_score(y_test_id, l_preds_id, zero_division=0)

    # 3. Evaluate on Out-of-Distribution (OOD) Leakage Check Split
    b_preds_ood = [
        1 if baseline.score(TargetContext.from_dict(p.target_context), Hit.model_validate(p.hit)) >= 0.5 else 0
        for p in pairs_ood
    ]
    l_preds_ood = (learned.model.predict_proba(X_test_ood)[:, 1] >= 0.5).astype(int)

    b_prec_ood = precision_score(y_test_ood, b_preds_ood, zero_division=0)
    b_rec_ood = recall_score(y_test_ood, b_preds_ood, zero_division=0)
    b_f1_ood = f1_score(y_test_ood, b_preds_ood, zero_division=0)

    l_prec_ood = precision_score(y_test_ood, l_preds_ood, zero_division=0)
    l_rec_ood = recall_score(y_test_ood, l_preds_ood, zero_division=0)
    l_f1_ood = f1_score(y_test_ood, l_preds_ood, zero_division=0)

    # 4. Print Comparison Table
    print("\n" + "=" * 78)
    print(f"BENCHMARK RESULTS [Synthetic, not for the report]")
    print("=" * 78)
    print(f"{'Evaluation Set & Metric':<32} | {'Baseline (Hand-Weighted)':<20} | {'Learned (ML)':<16}")
    print("-" * 78)
    print(f"{'In-Distribution (ID) Precision':<32} | {b_prec_id:<20.4f} | {l_prec_id:<16.4f}")
    print(f"{'In-Distribution (ID) Recall':<32} | {b_rec_id:<20.4f} | {l_rec_id:<16.4f}")
    print(f"{'In-Distribution (ID) F1-Score':<32} | {b_f1_id:<20.4f} | {l_f1_id:<16.4f}")
    print("-" * 78)
    print(f"{'OOD Leakage Check Precision':<32} | {b_prec_ood:<20.4f} | {l_prec_ood:<16.4f}")
    print(f"{'OOD Leakage Check Recall':<32} | {b_rec_ood:<20.4f} | {l_rec_ood:<16.4f}")
    print(f"{'OOD Leakage Check F1-Score':<32} | {b_f1_ood:<20.4f} | {l_f1_ood:<16.4f}")
    print("=" * 78)

    # 5. Feature Importance (Coefficients)
    print("\n[*] Learned Feature Coefficients (Importance):")
    coef_dict = learned.feature_importance()
    sorted_coefs = sorted(coef_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)
    for feat_name, weight in sorted_coefs:
        direction = "+" if weight >= 0 else "-"
        print(f"  {direction} {feat_name:<20}: {weight:+.4f}")

    if model_out:
        out_path = Path(model_out)
        learned.save(out_path)
        print(f"\n[+] Saved trained model checkpoint to: {out_path.resolve()}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate OSINT resolvers")
    parser.add_argument("--pairs", type=int, default=500, help="Number of synthetic pairs (default: 500)")
    parser.add_argument("--test-size", type=float, default=0.25, help="Held-out fraction (default: 0.25)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--model-out", type=str, default="model.joblib", help="Output checkpoint path")
    args = parser.parse_args()

    run_experiment(
        n_pairs=args.pairs,
        test_size=args.test_size,
        seed=args.seed,
        model_out=args.model_out,
    )


if __name__ == "__main__":
    main()
