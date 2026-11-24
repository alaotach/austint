"""Tests for OSINT dataset storage, synthetic pair generation, and matrix construction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from osint_agent.ml.dataset import (
    LabeledPair,
    build_xy,
    generate_synthetic_pairs,
    load_pairs,
    save_pairs,
)
from osint_agent.ml.features import FEATURE_NAMES, TargetContext
from osint_agent.core.schema import EntityType, Hit


def test_labeled_pair_jsonl_roundtrip(tmp_path: Path) -> None:
    """LabeledPairs must serialize to and deserialize from JSONL without data loss."""
    jsonl_path = tmp_path / "test_pairs.jsonl"

    ctx = TargetContext(names=["Dr. Evelyn Reed"], locations=["Boston"])
    hit = Hit(
        target_entity_id="test:1",
        discovered_type=EntityType.PUBLIC_FIGURE,
        discovered_value="Evelyn Reed",
        relation="alias",
        source_tool="wikidata",
    )

    original_pairs = [
        LabeledPair(
            target_context=ctx.to_dict(),
            hit=hit.model_dump(),
            label=1,
            notes="verified match",
        ),
        LabeledPair(
            target_context=ctx.to_dict(),
            hit=hit.model_dump(),
            label=0,
            notes="false positive",
        ),
    ]

    save_pairs(original_pairs, jsonl_path)
    assert jsonl_path.exists()

    loaded_pairs = load_pairs(jsonl_path)
    assert len(loaded_pairs) == 2
    assert loaded_pairs[0].target_context["names"] == ["Dr. Evelyn Reed"]
    assert loaded_pairs[0].label == 1
    assert loaded_pairs[1].label == 0
    assert loaded_pairs[0].notes == "verified match"


def test_build_xy_feature_names_order() -> None:
    """build_xy output matrix X columns must match FEATURE_NAMES ordering."""
    with patch("osint_agent.ml.embeddings.get_model", return_value=None):
        pairs = generate_synthetic_pairs(n=10, seed=42)
        X, y, feature_names = build_xy(pairs)

        assert feature_names == FEATURE_NAMES
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape == (10, len(FEATURE_NAMES))
        assert y.shape == (10,)
        assert set(np.unique(y)).issubset({0, 1})


def test_synthetic_generation_balance_and_notes() -> None:
    """Synthetic generator must generate balanced classes with synthetic markers."""
    n = 20
    pairs = generate_synthetic_pairs(n=n, seed=99)
    assert len(pairs) == n

    positives = [p for p in pairs if p.label == 1]
    negatives = [p for p in pairs if p.label == 0]

    assert len(positives) == n // 2
    assert len(negatives) == n // 2

    for p in pairs:
        assert "[synthetic]" in p.notes


def test_label_cli_graph_loading(tmp_path: Path) -> None:
    """label_cli should load candidate pairs from graph.json without errors."""
    import json
    from osint_agent.ml.label_cli import load_candidate_pairs_from_graph

    graph_file = tmp_path / "test_graph.json"
    dummy_graph = {
        "nodes": [
            {"id": "org:acme", "entity_type": "org", "value": "Acme Corp"},
            {"id": "domain:acme.com", "entity_type": "domain", "value": "acme.com"},
        ],
        "edges": [
            {"source": "org:acme", "target": "domain:acme.com", "relation": "website", "source_tool": "wikidata", "score": 0.9}
        ],
    }
    with open(graph_file, "w", encoding="utf-8") as f:
        json.dump(dummy_graph, f)

    pairs = load_candidate_pairs_from_graph(graph_file)
    assert len(pairs) == 1
    ctx, hit = pairs[0]
    assert "Acme Corp" in ctx.names
    assert hit.discovered_value == "acme.com"
