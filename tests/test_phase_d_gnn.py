"""Offline tests for Phase D: CollectiveGNN, graph construction, 3-way comparison, and --resolver neural_gnn."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from osint_agent.core.schema import Hit
from osint_agent.ml.features import TargetContext
from osint_agent.ml.gnn import (
    NODE_FEATURE_DIM,
    CollectiveGNN,
    NeuralGNNResolver,
    build_agent_run_graph,
    build_dataset_graph,
)
from osint_agent.ml.neural import MultimodalIdentityNet, NeuralResolver
from scripts.train_gnn import run_gnn_training_and_eval


def _make_pair(
    pair_id: str,
    person_id: str,
    uname: str,
    name: str,
    bio: str,
    links: list[str],
    platform: str,
    label: int,
    *,
    right_uname: str | None = None,
    right_name: str | None = None,
    right_bio: str | None = None,
    right_links: list[str] | None = None,
    parent_idx: int | None = None,
    hop_depth: int = 1,
) -> dict:
    r_u = right_uname or uname
    r_n = right_name or name
    r_b = right_bio or bio
    r_l = right_links if right_links is not None else links
    row = {
        "pair_id": pair_id,
        "person_id": person_id,
        "label": label,
        "hop_depth": hop_depth,
        "username_a": uname,
        "username_b": r_u,
        "display_name_a": name,
        "display_name_b": r_n,
        "bio_a": bio,
        "bio_b": r_b,
        "location_a": "Bengaluru",
        "location_b": "Bengaluru" if label == 1 else "Berlin",
        "links_a": links,
        "links_b": r_l,
        "bio_emb_a": [0.4] * 64,
        "bio_emb_b": [0.4] * 64 if label == 1 else [-0.4] * 64,
        "clip_emb_a": [0.5] * 64,
        "clip_emb_b": [0.5] * 64 if label == 1 else [-0.5] * 64,
        "links_jaccard": 1.0 if label == 1 else 0.0,
        "shared_domain_count": 1 if label == 1 else 0,
        "link_count_a": len(links),
        "link_count_b": len(r_l),
        "phash_sim": 1.0 if label == 1 else 0.0,
        "hand_features": {
            "name_sim": 1.0 if label == 1 else 0.1,
            "username_sim": 1.0 if label == 1 else 0.1,
            "location_match": 1.0 if label == 1 else 0.0,
            "domain_overlap": 1.0 if label == 1 else 0.0,
            "bio_sim": 1.0 if label == 1 else 0.0,
            "avatar_sim": 1.0 if label == 1 else 0.0,
            "trusted_source": 1.0 if label == 1 else 0.0,
            "cross_source_count": 1.0 if label == 1 else 0.0,
        },
        "left": {
            "username": uname,
            "display_name": name,
            "bio": bio,
            "location": "Bengaluru",
            "links": links,
            "platform": "github",
            "photo_phash": "a1b2c3d4" if label == 1 else "11223344",
            "clip_embedding": [0.5, 0.5, 0.5, 0.5],
        },
        "right": {
            "username": r_u,
            "display_name": r_n,
            "bio": r_b,
            "location": "Bengaluru" if label == 1 else "Berlin",
            "links": r_l,
            "platform": platform,
            "photo_phash": "a1b2c3d4" if label == 1 else "ffeeddcc",
            "clip_embedding": [0.5, 0.5, 0.5, 0.5] if label == 1 else [-0.5, 0.5, -0.5, 0.5],
        },
    }
    if parent_idx is not None:
        row["parent_idx"] = parent_idx
    return row


def test_dataset_graph_construction_edges_and_features():
    rows = [
        _make_pair(
            "p0",
            "per0",
            "ada_dev",
            "Ada Lovelace",
            "Compiler engineer @ada_dev",
            ["https://adalovelace.example/about"],
            "mastodon",
            1,
        ),
        _make_pair(
            "p1",
            "per0",
            "ada_dev",
            "Ada Lovelace",
            "Notes by @ada_dev on compilers",
            ["https://adalovelace.example/blog"],
            "personal_site",
            1,
            parent_idx=0,
            hop_depth=2,
        ),
        _make_pair(
            "p2",
            "per1",
            "ada_clone",
            "Ada Clone",
            "Random chef account",
            ["https://otherchef.example/"],
            "mastodon",
            0,
            right_uname="ada_chef",
            right_name="Ada Chef",
            right_bio="Pastry chef",
            right_links=["https://otherchef.example/"],
        ),
    ]
    rows[1]["left"]["links"] = ["https://adalovelace.example/about"]

    gdata = build_dataset_graph(rows, [0.9, 0.7, 0.2], confidence_decay=0.85)
    assert gdata.node_features.shape == (3, NODE_FEATURE_DIM)
    assert abs(float(gdata.node_features[0, 0]) - 0.9) < 1e-5
    assert abs(float(gdata.chain_scores[1]) - (0.9 * 0.85 * 0.7)) < 1e-4
    # Expect links, mentions, shared_domain, and/or similarity edges
    assert gdata.edge_index.size(1) > 0
    assert gdata.edge_counts_by_type["links"] >= 2
    assert gdata.edge_counts_by_type["shared_domain"] >= 2
    assert gdata.edge_counts_by_type["mentions"] >= 2


def test_agent_run_graph_and_neural_gnn_resolver_roundtrip(tmp_path: Path):
    graph = nx.DiGraph()
    graph.add_node(
        "seed:ada",
        entity_type="username",
        value="ada_dev",
        confidence=1.0,
        depth=0,
        attributes={"links": ["https://adalovelace.example/"]},
    )
    graph.add_node(
        "profile:mstdn_ada",
        entity_type="profile",
        value="ada_dev",
        confidence=0.72,
        depth=1,
        attributes={
            "username": "ada_dev",
            "bio": "Systems & OSINT @ada_dev",
            "links": ["https://adalovelace.example/"],
        },
    )
    graph.add_node(
        "profile:site_ada",
        entity_type="profile",
        value="adalovelace.example",
        confidence=0.51,
        depth=2,
        attributes={
            "username": "ada_dev",
            "bio": "Personal page @ada_dev",
            "website": "https://adalovelace.example/",
        },
    )
    graph.add_edge("seed:ada", "profile:mstdn_ada", relation="discovered", source="sherlock", score=0.72)
    graph.add_edge("profile:mstdn_ada", "profile:site_ada", relation="links", source="web", score=0.71)

    ag = build_agent_run_graph(graph)
    assert ag.node_features.shape == (3, NODE_FEATURE_DIM)
    assert ag.edge_index.size(1) >= 4

    n_res = NeuralResolver(MultimodalIdentityNet(), temperature=1.0)
    gnn_res = NeuralGNNResolver(n_res, CollectiveGNN(), temperature=1.0)

    ckpt = tmp_path / "neural_gnn_resolver.pt"
    gnn_res.save(ckpt)
    loaded = NeuralGNNResolver.load(ckpt)

    refined = loaded.refine_graph(graph)
    assert refined.nodes["seed:ada"]["confidence"] == 1.0
    assert "gnn_confidence" in refined.nodes["profile:mstdn_ada"]
    assert "chain_confidence" in refined.nodes["profile:mstdn_ada"]
    assert 0.0 <= refined.nodes["profile:site_ada"]["confidence"] <= 1.0

    score = loaded.score(
        TargetContext(usernames=["ada_dev"]),
        Hit(
            target_entity_id="seed:ada",
            source_tool="mastodon",
            relation="profile_match",
            discovered_type="profile",
            discovered_value="ada_dev",
            metadata={"username": "ada_dev", "bio": "Systems"},
        ),
    )
    assert 0.0 <= score <= 1.0


def test_phase_d_end_to_end_three_way_comparison_and_helps_vs_hurts(tmp_path: Path):
    data_dir = tmp_path / "osint_link"
    data_dir.mkdir(parents=True, exist_ok=True)

    def _build_split(n_people: int, platform: str) -> list[dict]:
        rows: list[dict] = []
        for idx in range(n_people):
            pid = f"{platform}_p{idx}"
            uname = f"user_{platform}_{idx}"
            dom = f"https://{uname}.example"
            pos1 = _make_pair(
                f"{pid}_pos1",
                pid,
                uname,
                f"User {idx}",
                f"Open source dev @{uname}",
                [dom],
                platform,
                1,
                hop_depth=1,
            )
            pos2 = _make_pair(
                f"{pid}_pos2",
                pid,
                uname,
                f"User {idx}",
                f"Personal notes @{uname}",
                [dom],
                platform,
                1,
                parent_idx=len(rows),
                hop_depth=2,
            )
            neg1 = _make_pair(
                f"{pid}_neg1",
                pid,
                uname,
                f"User {idx}",
                f"Open source dev @{uname}",
                [dom],
                platform,
                0,
                right_uname=f"other_{idx}",
                right_name=f"Other Person {idx}",
                right_bio="Totally unrelated chef bio",
                right_links=[f"https://unrelated-{idx}.example"],
                hop_depth=2,
            )
            rows.extend([pos1, pos2, neg1])
        return rows

    for split_name, count, plat in [
        ("train", 8, "mastodon"),
        ("val", 4, "mastodon"),
        ("test", 4, "mastodon"),
        ("ood_platform_test", 4, "personal_site"),
    ]:
        rows = _build_split(count, plat)
        (data_dir / f"{split_name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n",
            encoding="utf-8",
        )

    report = run_gnn_training_and_eval(
        data_dir=data_dir,
        neural_model_path=tmp_path / "neural_resolver.pt",
        gnn_model_out_path=tmp_path / "neural_gnn_resolver.pt",
        seed=42,
        epochs_gnn=15,
        n_bootstrap=30,
    )
    assert report["status"] == "ok"
    assert (tmp_path / "neural_gnn_resolver.pt").exists()

    for split_key in ("test", "ood_platform_test"):
        sp = report["splits"][split_key]
        for method in ("phase_c_alone", "phase_c_plus_gnn", "chain_confidence_heuristic"):
            assert method in sp
            assert np.isfinite(sp[method]["pr_auc"])
            assert np.isfinite(sp[method]["ece"])
            assert "ci95" in sp[method]
        assert "helps_vs_hurts" in sp
        assert "helped_count" in sp["helps_vs_hurts"]
        assert "hurt_count" in sp["helps_vs_hurts"]
        assert "graph_stats" in sp


def test_cli_parser_supports_neural_gnn_resolver():
    from osint_agent.cli import parse_args

    args = parse_args(
        [
            "--kind",
            "public_figure",
            "--value",
            "ada",
            "--resolver",
            "neural_gnn",
            "--model-path",
            "models/neural_gnn_resolver.pt",
        ]
    )
    assert args.resolver == "neural_gnn"
    assert args.model_path == "models/neural_gnn_resolver.pt"
