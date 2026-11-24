"""Tests for ML LearnedResolver training, evaluation against baseline, and serialization."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from osint_agent.ml.dataset import build_xy, generate_synthetic_pairs
from osint_agent.ml.features import FEATURE_NAMES, TargetContext
from osint_agent.ml.resolver import BaselineWeightedResolver, LearnedResolver
from osint_agent.core.schema import Hit


def test_learned_resolver_train_and_beats_baseline() -> None:
    """LearnedResolver should train on synthetic pairs and outperform baseline F1 on held-out split."""
    with patch("osint_agent.ml.embeddings.get_model", return_value=None):
        pairs = generate_synthetic_pairs(n=300, seed=42)
        X, y, feature_names = build_xy(pairs)

        X_train, X_test, y_train, y_test, pairs_train, pairs_test = train_test_split(
            X, y, pairs, test_size=0.25, random_state=42, stratify=y
        )

        # Baseline evaluation
        baseline = BaselineWeightedResolver()
        baseline_preds = []
        for pair in pairs_test:
            ctx = TargetContext.from_dict(pair.target_context)
            hit = Hit.model_validate(pair.hit)
            score = baseline.score(ctx, hit)
            baseline_preds.append(1 if score >= 0.5 else 0)

        baseline_f1 = f1_score(y_test, baseline_preds)

        # Learned resolver training & evaluation
        learned = LearnedResolver(random_state=42)
        learned.fit(X_train, y_train)

        learned_probs = learned.model.predict_proba(X_test)[:, 1]
        learned_preds = (learned_probs >= 0.5).astype(int)
        learned_f1 = f1_score(y_test, learned_preds)

        # The supervised classifier should easily outperform or match baseline on hard negatives
        assert learned_f1 >= baseline_f1
        assert learned_f1 > 0.85

        # Feature importance check
        importance = learned.feature_importance()
        assert set(importance.keys()) == set(FEATURE_NAMES)


def test_learned_resolver_save_and_load_roundtrip(tmp_path: Path) -> None:
    """Saving and loading LearnedResolver with joblib must preserve exact predictions."""
    with patch("osint_agent.ml.embeddings.get_model", return_value=None):
        pairs = generate_synthetic_pairs(n=50, seed=123)
        X, y, _ = build_xy(pairs)

        learned = LearnedResolver(random_state=123)
        learned.fit(X, y)

        sample_pair = pairs[0]
        ctx = TargetContext.from_dict(sample_pair.target_context)
        hit = Hit.model_validate(sample_pair.hit)

        score_before = learned.score(ctx, hit)

        model_file = tmp_path / "model.joblib"
        learned.save(model_file)
        assert model_file.exists()

        loaded_resolver = LearnedResolver.load(model_file)
        score_after = loaded_resolver.score(ctx, hit)

        assert score_before == score_after
        assert loaded_resolver.is_fitted is True


def test_learned_resolver_coefficients_are_non_negative() -> None:
    """Enforce monotonic behavior: every feature coefficient must be >= 0.0 (no negative effects)."""
    with patch("osint_agent.ml.embeddings.get_model", return_value=None):
        pairs = generate_synthetic_pairs(n=100, seed=777)
        X, y, _ = build_xy(pairs)

        learned = LearnedResolver(random_state=777)
        learned.fit(X, y)

        importances = learned.feature_importance()
        for name, coef in importances.items():
            assert coef >= 0.0, f"Feature '{name}' has negative coefficient {coef}; expected >= 0.0"
