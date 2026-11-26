"""Phase B Baselines: Monotonic HistGradientBoosting, Zero-Shot LLM Judge (cached/recorded), and Bootstrap Metrics."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from osint_agent.core.schema import EntityType, Hit
from osint_agent.llm.client import FakeClient, LLMClient
from osint_agent.ml.features import FEATURE_NAMES, TargetContext, extract_features
from osint_agent.ml.resolver import BaseResolver, BaselineWeightedResolver, LearnedResolver

logger = logging.getLogger(__name__)


def pair_dict_to_feature_vector(row: dict[str, Any]) -> np.ndarray:
    """Extract ordered 9-feature numpy vector from an exported OSINT-Link pair dict."""
    hf = row.get("hand_features") or {}
    return np.array([float(hf.get(name, 0.0)) for name in FEATURE_NAMES], dtype=np.float64)


def score_rule_baseline(row: dict[str, Any], resolver: BaselineWeightedResolver | None = None) -> float:
    """Score an exported pair dict using BaselineWeightedResolver's exact hand-feature weights."""
    res = resolver or BaselineWeightedResolver()
    hf = row.get("hand_features") or {}
    raw = sum(res.weights.get(k, 0.0) * float(hf.get(k, 0.0)) for k in res.weights)
    return round(float(max(0.0, min(1.0, raw))), 4)


class HistGBMResolver(BaseResolver):
    """Monotonic HistGradientBoostingClassifier over the 9 hand-engineered OSINT features."""

    def __init__(self, seed: int = 42, max_iter: int = 100, learning_rate: float = 0.08) -> None:
        self.seed = seed
        self.model = HistGradientBoostingClassifier(
            monotonic_cst=[1] * len(FEATURE_NAMES),
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=4,
            min_samples_leaf=10,
            random_state=seed,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> HistGBMResolver:
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("HistGBMResolver must be fitted before scoring.")
        return self.model.predict_proba(X)[:, 1]

    def score(self, target: TargetContext | Entity, hit: Hit) -> float:
        ctx = TargetContext.from_entity(target) if isinstance(target, Entity) else target
        feats = extract_features(ctx, hit)
        vec = np.array([[feats[k] for k in FEATURE_NAMES]], dtype=np.float64)
        return round(float(self.predict_proba_matrix(vec)[0]), 4)

    def save(self, path: str | Path) -> None:
        joblib.dump({"model": self.model, "seed": self.seed, "is_fitted": self.is_fitted}, path)

    @classmethod
    def load(cls, path: str | Path) -> HistGBMResolver:
        data = joblib.load(path)
        obj = cls(seed=int(data.get("seed", 42)))
        obj.model = data["model"]
        obj.is_fitted = bool(data.get("is_fitted", True))
        return obj


@dataclass
class JudgeResult:
    probability: float
    tokens: int
    cost_usd: float
    latency_ms: float
    from_cache: bool = False


class ZeroShotLLMJudge:
    """Zero-shot LLM pair-matching judge with strict JSON output, budget cap, disk cache, and recorded mode."""

    SYSTEM_PROMPT = (
        "You are an OSINT identity-resolution judge. Given structured similarity evidence between two public profiles "
        "(without raw private bios), estimate the probability that both profiles belong to the SAME person or organization. "
        "Return ONLY valid JSON: {\"match_probability\": <float between 0.0 and 1.0>, \"rationale\": \"<short reason>\"}."
    )

    def __init__(
        self,
        client: LLMClient | None = None,
        cache_path: str | Path | None = None,
        recorded_path: str | Path | None = None,
        max_calls: int = 100,
        usd_per_1k_tokens: float = 0.0,
    ) -> None:
        self.client = client
        self.cache_path = Path(cache_path) if cache_path else None
        self.recorded_path = Path(recorded_path) if recorded_path else None
        self.max_calls = max_calls
        self.usd_per_1k_tokens = usd_per_1k_tokens
        self.calls_made = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0

        self._cache: dict[str, dict[str, Any]] = {}
        self._recorded: dict[str, dict[str, Any]] = {}
        self._load_store(self.cache_path, self._cache)
        self._load_store(self.recorded_path, self._recorded)

    @staticmethod
    def _load_store(path: Path | None, target: dict[str, dict[str, Any]]) -> None:
        if path and path.exists():
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict) and obj.get("pair_id"):
                        target[str(obj["pair_id"])] = obj
            except Exception as exc:
                logger.debug("Could not load judge store %s: %s", path, exc)

    def _append_cache(self, entry: dict[str, Any]) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    @staticmethod
    def _pair_prompt(row: dict[str, Any]) -> str:
        payload = {
            "platform_a": row.get("platform_a"),
            "platform_b": row.get("platform_b"),
            "username_a": row.get("username_a"),
            "username_b": row.get("username_b"),
            "display_name_a": row.get("display_name_a"),
            "display_name_b": row.get("display_name_b"),
            "location_a": row.get("location_a"),
            "location_b": row.get("location_b"),
            "bio_emb_sim": row.get("bio_emb_sim"),
            "phash_sim": row.get("phash_sim"),
            "clip_sim": row.get("clip_sim"),
            "links_jaccard": row.get("links_jaccard"),
            "shared_domain_count": row.get("shared_domain_count"),
            "hand_features": row.get("hand_features"),
        }
        return json.dumps(payload, ensure_ascii=False)

    def score_pair(self, row: dict[str, Any]) -> JudgeResult:
        pid = str(row.get("pair_id") or hashlib.sha256(self._pair_prompt(row).encode("utf-8")).hexdigest()[:16])

        # 1. Recorded-response mode or disk cache hit
        cached = self._recorded.get(pid) or self._cache.get(pid)
        if cached is not None:
            return JudgeResult(
                probability=float(max(0.0, min(1.0, cached.get("match_probability", 0.5)))),
                tokens=int(cached.get("tokens", 0)),
                cost_usd=float(cached.get("cost_usd", 0.0)),
                latency_ms=float(cached.get("latency_ms", 0.0)),
                from_cache=True,
            )

        # 2. Budget guard
        if self.client is None or self.calls_made >= self.max_calls:
            return JudgeResult(probability=0.5, tokens=0, cost_usd=0.0, latency_ms=0.0, from_cache=True)

        t0 = time.perf_counter()
        try:
            res = self.client.complete_json(
                system=self.SYSTEM_PROMPT,
                user=self._pair_prompt(row),
                max_tokens=256,
            )
            prob = float(res.get("match_probability", 0.5))
            prob = max(0.0, min(1.0, prob))
            tokens = int(res.get("_token_usage", 0))
        except Exception as exc:
            logger.debug("ZeroShotLLMJudge fallback for %s: %s", pid, exc)
            prob = 0.5
            tokens = 0
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        cost_usd = round((tokens / 1000.0) * self.usd_per_1k_tokens, 6)

        self.calls_made += 1
        self.total_tokens += tokens
        self.total_cost_usd += cost_usd

        entry = {
            "pair_id": pid,
            "match_probability": round(prob, 4),
            "tokens": tokens,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
        }
        self._cache[pid] = entry
        self._append_cache(entry)
        return JudgeResult(probability=round(prob, 4), tokens=tokens, cost_usd=cost_usd, latency_ms=latency_ms)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across `n_bins` uniform probability bins."""
    if len(y_true) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += (cnt / n) * abs(acc - conf)
    return round(float(ece), 4)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute Precision, Recall, F1, PR-AUC, and ECE."""
    if len(y_true) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "pr_auc": 0.0, "ece": 0.0}
    y_pred = (y_prob >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    if len(np.unique(y_true)) > 1:
        pr_auc = float(average_precision_score(y_true, y_prob))
    else:
        pr_auc = float(y_true.mean())
    ece = compute_ece(y_true, y_prob)
    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "pr_auc": round(pr_auc, 4),
        "ece": ece,
    }


def bootstrap_metrics_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 500,
    seed: int = 42,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute point estimates and 95% bootstrap confidence intervals for all metrics."""
    point = compute_metrics(y_true, y_prob, threshold=threshold)
    n = len(y_true)
    if n < 2:
        return {k: {"mean": v, "ci_low": v, "ci_high": v} for k, v in point.items()}

    rng = np.random.default_rng(seed)
    boot_vals: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m = compute_metrics(y_true[idx], y_prob[idx], threshold=threshold)
        for k, v in m.items():
            boot_vals[k].append(v)

    out: dict[str, Any] = {}
    for k, v in point.items():
        arr = np.asarray(boot_vals[k], dtype=np.float64)
        out[k] = {
            "point": v,
            "ci_low": round(float(np.percentile(arr, 2.5)), 4),
            "ci_high": round(float(np.percentile(arr, 97.5)), 4),
        }
    return out
