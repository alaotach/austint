"""Offline unit tests for Phase B: Rule, Monotonic LogReg, Monotonic HistGBM, Zero-Shot LLM Judge, and Bootstrap CIs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from osint_agent.llm.client import FakeClient
from osint_agent.ml.baselines import (
    HistGBMResolver,
    ZeroShotLLMJudge,
    bootstrap_metrics_ci,
    compute_ece,
    compute_metrics,
)
from osint_agent.ml.datasets.builder import ProfileRecord, compute_bio_embedding, hash_person_id, split_and_export_dataset
from scripts.eval_baselines import evaluate_baselines


def test_histgbm_resolver_monotonic_and_save_load(tmp_path: Path) -> None:
    """HistGBMResolver must respect monotonic_cst=1 and round-trip save/load."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 1.0, size=(80, 9))
    y = (X[:, 0] + X[:, 3] + 0.5 * X[:, 5] > 1.25).astype(int)

    clf = HistGBMResolver(seed=42, max_iter=30).fit(X, y)
    p_low = clf.predict_proba_matrix(np.array([[0.0, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0, 0.0, 0.8]]))[0]
    p_high = clf.predict_proba_matrix(np.array([[1.0, 1.0, 0.9, 0.9, 1.0, 0.9, 1.0, 1.0, 0.8]]))[0]
    assert p_high >= p_low

    model_path = tmp_path / "hgb.joblib"
    clf.save(model_path)
    loaded = HistGBMResolver.load(model_path)
    assert loaded.predict_proba_matrix(X[:5]).tolist() == pytest.approx(clf.predict_proba_matrix(X[:5]).tolist())


def test_zero_shot_llm_judge_cache_budget_and_recorded_mode(tmp_path: Path) -> None:
    """ZeroShotLLMJudge must track tokens/latency, cache hits, enforce budget, and support recorded mode."""
    fake = FakeClient(
        responses=[
            {"match_probability": 0.92, "rationale": "Exact handle and bio match"},
            {"match_probability": 0.12, "rationale": "Different city and org"},
        ],
        tokens_per_call=150,
    )
    cache_file = tmp_path / "cache.jsonl"
    judge = ZeroShotLLMJudge(client=fake, cache_path=cache_file, max_calls=1, usd_per_1k_tokens=0.002)

    row1 = {"pair_id": "pair_001", "username_a": "alice", "username_b": "alice"}
    row2 = {"pair_id": "pair_002", "username_a": "alice", "username_b": "bob"}

    r1 = judge.score_pair(row1)
    assert r1.probability == 0.92
    assert r1.tokens == 150
    assert r1.cost_usd == pytest.approx(0.0003)
    assert r1.from_cache is False

    # Repeat row1 -> must hit in-memory/disk cache without consuming budget
    r1_cached = judge.score_pair(row1)
    assert r1_cached.probability == 0.92
    assert r1_cached.from_cache is True
    assert judge.calls_made == 1

    # row2 exceeds max_calls=1 -> returns neutral 0.5 without calling FakeClient
    r2 = judge.score_pair(row2)
    assert r2.probability == 0.5
    assert r2.tokens == 0

    # New judge in recorded-only mode loads cache_file deterministically
    rec_judge = ZeroShotLLMJudge(client=None, recorded_path=cache_file)
    r1_rec = rec_judge.score_pair(row1)
    assert r1_rec.probability == 0.92
    assert r1_rec.from_cache is True


def test_bootstrap_metrics_ci_and_ece() -> None:
    """compute_ece, compute_metrics, and bootstrap_metrics_ci should return valid bounded metrics."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.6, 0.4])

    m = compute_metrics(y_true, y_prob)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["pr_auc"] == 1.0
    assert 0.0 <= compute_ece(y_true, y_prob) <= 1.0

    ci = bootstrap_metrics_ci(y_true, y_prob, n_bootstrap=100, seed=42)
    for key in ("precision", "recall", "f1", "pr_auc", "ece"):
        assert ci[key]["ci_low"] <= ci[key]["point"] <= ci[key]["ci_high"]


def test_eval_baselines_end_to_end_offline(tmp_path: Path) -> None:
    """evaluate_baselines must evaluate all 4 baselines on test and ood_platform_test splits offline."""
    profiles: list[ProfileRecord] = []
    pos_pairs: list[tuple[ProfileRecord, ProfileRecord, str]] = []

    for i in range(20):
        pid = hash_person_id(f"person_{i}", salt="s")
        p1 = ProfileRecord(
            person_id=pid,
            platform="github",
            url=f"https://github.com/dev{i}",
            username=f"dev{i}",
            display_name=f"Dev {i}",
            bio=f"Python ML engineer {i}",
            location="Delhi" if i % 2 == 0 else "Zurich",
            links=[f"https://dev{i}.io"],
            label_source="github_social_accounts",
            bio_emb=compute_bio_embedding(f"Python ML engineer {i}"),
        )
        plat2 = "mastodon" if i >= 16 else "personal_site"
        p2 = ProfileRecord(
            person_id=pid,
            platform=plat2,
            url=f"https://dev{i}.io",
            username=f"dev{i}",
            display_name=f"Dev {i}",
            bio=f"Python ML engineer {i}",
            location="Delhi" if i % 2 == 0 else "Zurich",
            links=[f"https://github.com/dev{i}"],
            label_source="github_plus_bidirectional_rel_me",
            bio_emb=compute_bio_embedding(f"Python ML engineer {i}"),
        )
        profiles.extend([p1, p2])
        pos_pairs.append((p1, p2, p2.label_source))

    cfg = {
        "seed": 42,
        "dataset": {"name": "osint-link", "output_dir": str(tmp_path), "negatives_per_positive": 2},
        "splits": {"train_ratio": 0.60, "val_ratio": 0.20, "test_ratio": 0.20, "ood_platforms": ["mastodon", "devto"]},
    }
    split_and_export_dataset(profiles, pos_pairs, cfg)

    res = evaluate_baselines(
        data_dir=tmp_path,
        seed=42,
        llm_mode="fake",
        llm_max_calls=30,
        n_bootstrap=50,
        ood_platforms={"mastodon", "devto"},
    )
    assert res["can_fit_supervised"] is True
    for sp in ("test", "ood_platform_test"):
        models = res["splits"][sp]["models"]
        assert set(models.keys()) == {"rule_baseline", "logistic_regression", "hist_gradient_boosting", "zero_shot_llm_judge"}
        for mname, mdata in models.items():
            assert "pr_auc" in mdata["metrics"]
            assert "ece" in mdata["metrics"]
            assert mdata["latency_ms_per_pair"] >= 0.0
