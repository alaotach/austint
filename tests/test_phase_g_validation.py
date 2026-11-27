"""Offline tests for Phase G validation: GNN leakage check, paired bootstrap, shortcuts, neg-type breakdown, and 2-variant LLM judge."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from osint_agent.llm.client import FakeClient
from osint_agent.ml.gnn import build_dataset_graph, verify_split_graph_isolation
from osint_agent.ml.validation import (
    evaluate_by_negative_type,
    evaluate_shortcut_baselines,
    paired_person_bootstrap,
    run_llm_judge_two_variants,
    tune_threshold_on_val,
)


def _sample_rows() -> list[dict]:
    return [
        {
            "pair_id": "p1",
            "person_id_a": "per_1",
            "label": 1,
            "neg_type": "",
            "platform_a": "github",
            "platform_b": "personal_site",
            "bio_len_a": 40,
            "bio_len_b": 50,
            "hand_features": {"username_sim": 0.9},
        },
        {
            "pair_id": "p2",
            "person_id_a": "per_1",
            "label": 0,
            "neg_type": "similar_username_diff_person",
            "platform_a": "github",
            "platform_b": "github",
            "bio_len_a": 40,
            "bio_len_b": 10,
            "hand_features": {"username_sim": 0.7},
        },
        {
            "pair_id": "p3",
            "person_id_a": "per_2",
            "label": 1,
            "neg_type": "",
            "platform_a": "github",
            "platform_b": "personal_site",
            "bio_len_a": 30,
            "bio_len_b": 35,
            "hand_features": {"username_sim": 0.95},
        },
        {
            "pair_id": "p4",
            "person_id_a": "per_2",
            "label": 0,
            "neg_type": "diff_person_fallback",
            "platform_a": "github",
            "platform_b": "mastodon",
            "bio_len_a": 30,
            "bio_len_b": 0,
            "hand_features": {"username_sim": 0.1},
        },
    ]


def test_verify_split_graph_isolation_catches_train_test_overlap():
    rows = _sample_rows()
    g_tr = build_dataset_graph(rows[:2], [0.8, 0.2], exclude_pair_forming_edges=True)
    g_te = build_dataset_graph(rows[2:], [0.9, 0.1], exclude_pair_forming_edges=True)
    assert verify_split_graph_isolation({"train": g_tr, "test": g_te}) is True

    g_bad = build_dataset_graph(rows[:3], [0.8, 0.2, 0.9], exclude_pair_forming_edges=True)
    with pytest.raises(AssertionError, match="Cross-split leakage"):
        verify_split_graph_isolation({"train": g_bad, "test": g_te})


def test_paired_person_bootstrap_clustered_and_deterministic():
    rows = _sample_rows()
    probs_a = np.array([0.9, 0.1, 0.95, 0.05])
    probs_b = np.array([0.55, 0.45, 0.55, 0.45])
    res1 = paired_person_bootstrap(rows, probs_a, probs_b, n_bootstrap=50, seed=42)
    res2 = paired_person_bootstrap(rows, probs_a, probs_b, n_bootstrap=50, seed=42)
    assert res1 == res2
    assert res1["n_clusters"] == 2
    assert res1["pr_auc_diff"]["status"] in ("significant", "not significant")


def test_shortcut_baselines_and_negative_type_breakdown():
    rows = _sample_rows()
    sb = evaluate_shortcut_baselines(rows, {"test": rows}, seed=42)
    assert sb["status"] == "ok"
    assert set(sb["splits"]["test"]["models"].keys()) == {
        "platform_pair_only",
        "username_sim_only",
        "bio_len_has_bio_only",
    }

    nb = evaluate_by_negative_type(rows, {"m1": np.array([0.9, 0.2, 0.95, 0.1])})
    assert "similar_username_diff_person" in nb
    assert nb["similar_username_diff_person"]["is_hard_negative"] is True
    assert nb["diff_person_fallback"]["is_hard_negative"] is False


def test_tune_threshold_on_val_and_llm_judge_two_variants_offline(tmp_path: Path):
    y_val = np.array([1, 0, 1, 0])
    p_val = np.array([0.08, 0.02, 0.09, 0.01])
    t = tune_threshold_on_val(y_val, p_val)
    assert 0.01 <= t <= 0.08

    rows = _sample_rows()
    client = FakeClient(
        responses=[{"match_probability": 0.85 if r["label"] == 1 else 0.15, "rationale": "ok"} for r in rows] * 2,
        call_budget=20,
    )
    res = run_llm_judge_two_variants(rows, cache_dir=tmp_path / "llm_cache", client=client)
    assert res["status"] == "ok"
    assert res["variants"]["minimal"]["status"] == "ok"
    assert res["variants"]["field_by_field"]["status"] == "ok"
    assert (tmp_path / "llm_cache" / "llm_judge_minimal.jsonl").exists()



def test_select_live_llm_provider_g4f(monkeypatch):
    from osint_agent.llm.client import G4FClient, get_llm_client
    from osint_agent.ml.validation import select_live_llm_provider

    monkeypatch.setenv("LLM_PROVIDER", "g4f")
    assert select_live_llm_provider() == "g4f"
    cli = get_llm_client("g4f", call_budget=5)
    assert isinstance(cli, G4FClient)
