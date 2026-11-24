"""Swappable entity resolvers: baseline weighted heuristics and learned ML classifier."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from osint_agent.features import FEATURE_NAMES, TargetContext, extract_features
from osint_agent.schema import Entity, Hit

logger = logging.getLogger(__name__)

DEFAULT_BASELINE_WEIGHTS: dict[str, float] = {
    "name_sim": 0.40,
    "exact_name_match": 0.20,
    "token_overlap": 0.15,
    "text_sim": 0.20,
    "domain_link": 0.20,
    "loc_match": 0.15,
    "username_sim": 0.20,
    "has_text": 0.05,
    "uname_len": 0.0,
}


def _resolve_context(target: TargetContext | Entity) -> TargetContext:
    """Ensure target representation is converted to TargetContext."""
    if isinstance(target, TargetContext):
        return target
    if isinstance(target, Entity):
        return TargetContext.from_entity(target)
    raise TypeError(f"Expected TargetContext or Entity, got {type(target).__name__}")


class BaseResolver(ABC):
    """Abstract interface for swappable entity resolvers."""

    @abstractmethod
    def score(self, target: TargetContext | Entity, hit: Hit) -> float:
        """Calculate confidence score in [0.0, 1.0] indicating whether hit belongs to target."""
        pass

    def should_accept(self, score: float, threshold: float = 0.5) -> bool:
        """Determine whether candidate score meets or exceeds admission threshold."""
        return score >= threshold


class BaselineWeightedResolver(BaseResolver):
    """Hand-weighted multi-feature heuristic resolver for baseline scoring."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(weights or DEFAULT_BASELINE_WEIGHTS)

    def score(self, target: TargetContext | Entity, hit: Hit) -> float:
        """Score candidate hit using trusted indicators or weighted feature composition."""
        # Authoritative/cryptographic/exact-relation sources carry direct trust
        if hit.is_trusted:
            return 1.0

        ctx = _resolve_context(target)
        features = extract_features(ctx, hit)

        if features.get("exact_name_match", 0.0) == 1.0:
            return 1.0

        # Dynamically determine active attribute weights based on context and hit signals
        active_weights: dict[str, float] = {}

        if ctx.names:
            active_weights["name_sim"] = self.weights.get("name_sim", 0.40)
            active_weights["token_overlap"] = self.weights.get("token_overlap", 0.15)
        if ctx.usernames and features.get("username_sim", 0.0) > 0:
            active_weights["username_sim"] = self.weights.get("username_sim", 0.20)
        if ctx.domains and features.get("domain_link", 0.0) > 0:
            active_weights["domain_link"] = self.weights.get("domain_link", 0.20)
        if ctx.locations and features.get("loc_match", 0.0) > 0:
            active_weights["loc_match"] = self.weights.get("loc_match", 0.15)
        if ctx.texts and features.get("text_sim", 0.0) > 0:
            active_weights["text_sim"] = self.weights.get("text_sim", 0.20)
        if features.get("has_text", 0.0) > 0 and "text_sim" in active_weights:
            active_weights["has_text"] = self.weights.get("has_text", 0.05)

        if not active_weights:
            # Fallback to pure lexical similarity
            return max(0.0, min(1.0, float(features.get("name_sim", 0.0))))

        total_weight = sum(active_weights.values())
        weighted_sum = sum(w * float(features.get(k, 0.0)) for k, w in active_weights.items())
        normalized_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        final_score = max(0.0, min(1.0, round(normalized_score, 4)))
        logger.debug(
            "Baseline scored hit '%s' against context: %.4f",
            hit.discovered_value,
            final_score,
        )
        return final_score


class LearnedResolver(BaseResolver):
    """Supervised logistic regression resolver with balanced class weights and non-negative monotonic constraints."""

    def __init__(
        self,
        model: LogisticRegression | None = None,
        random_state: int = 42,
    ) -> None:
        if model is not None:
            self.model = model
            self.is_fitted = True
        else:
            self.model = LogisticRegression(
                class_weight="balanced",
                random_state=random_state,
                max_iter=1000,
            )
            self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> LearnedResolver:
        """Train the classifier enforcing strictly non-negative monotonic coefficients.

        Every feature may only increase match probability or remain neutral (zero coefficient).
        """
        n_samples, n_features = X.shape
        pos_count = np.sum(y == 1)
        neg_count = np.sum(y == 0)
        pos_weight = float(n_samples / (2.0 * pos_count)) if pos_count > 0 else 1.0
        neg_weight = float(n_samples / (2.0 * neg_count)) if neg_count > 0 else 1.0
        sample_weights = np.where(y == 1, pos_weight, neg_weight)

        reg_scale = 1.0 / (2.0 * max(1, n_samples))

        def loss_func(params: np.ndarray) -> float:
            w = params[:-1]
            b = params[-1]
            z = X @ w + b
            p = np.clip(expit(z), 1e-15, 1.0 - 1e-15)
            log_losses = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
            # Standard L2 regularization on weights
            return float(np.average(log_losses, weights=sample_weights) + reg_scale * np.sum(w**2))

        x0 = np.zeros(n_features + 1, dtype=np.float64)
        bounds = [(0.0, None)] * n_features + [(None, None)]

        res = minimize(loss_func, x0=x0, bounds=bounds, method="L-BFGS-B")
        w = np.maximum(0.0, res.x[:-1])
        b = float(res.x[-1])

        self.model.classes_ = np.array([0, 1])
        self.model.coef_ = np.array([w], dtype=np.float64)
        self.model.intercept_ = np.array([b], dtype=np.float64)
        self.is_fitted = True
        return self

    def score(self, target: TargetContext | Entity, hit: Hit) -> float:
        """Predict calibrated match probability for candidate hit."""
        if hit.is_trusted:
            return 1.0

        if not self.is_fitted:
            raise RuntimeError(
                "LearnedResolver must be fitted or loaded before calling score()."
            )

        ctx = _resolve_context(target)
        features = extract_features(ctx, hit)
        x_row = np.array([[float(features[col]) for col in FEATURE_NAMES]], dtype=np.float32)

        proba = float(self.model.predict_proba(x_row)[0][1])

        # Cold-start guard for bare username seeds (before any bio/name context is absorbed):
        # An exact handle match on a distinctive username (>= 5 chars) is a valid research lead.
        if (
            not ctx.names
            and not ctx.texts
            and not ctx.locations
            and not ctx.domains
            and features.get("username_sim", 0.0) >= 0.95
            and features.get("uname_len", 0.0) >= 5.0
        ):
            proba = max(proba, 0.65)

        return max(0.0, min(1.0, round(proba, 4)))

    def save(self, path: str | Path) -> None:
        """Persist model and feature schema using joblib."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "feature_names": FEATURE_NAMES, "is_fitted": self.is_fitted},
            target_path,
        )

    @classmethod
    def load(cls, path: str | Path) -> LearnedResolver:
        """Load trained model checkpoint from joblib file."""
        data = joblib.load(path)
        instance = cls(model=data["model"])
        instance.is_fitted = data.get("is_fitted", True)
        return instance

    def feature_importance(self) -> dict[str, float]:
        """Return learned linear model coefficients indexed by feature name."""
        if not self.is_fitted or not hasattr(self.model, "coef_"):
            return {}
        coefs = self.model.coef_[0]
        return {name: round(float(c), 4) for name, c in zip(FEATURE_NAMES, coefs)}
