"""Offline unit tests for Phase C: Multimodal Neural Identity-Linking, Calibration, Ablations, and CLI integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from osint_agent.core.schema import Entity, EntityType, Hit
from osint_agent.ml.datasets.builder import ProfileRecord, compute_bio_embedding, compute_clip_embedding, compute_phash, hash_person_id, split_and_export_dataset
from osint_agent.ml.neural import CharCNNEncoder, NeuralResolver, encode_username_chars
from scripts.train_neural import run_neural_training_and_ablations


def _build_temp_osint_link(tmp_path: Path) -> Path:
    profiles: list[ProfileRecord] = []
    pos_pairs: list[tuple[ProfileRecord, ProfileRecord, str]] = []

    for i in range(24):
        pid = hash_person_id(f"person_{i}", salt="s")
        p1 = ProfileRecord(
            person_id=pid,
            platform="github",
            url=f"https://github.com/dev_{i}",
            username=f"dev_{i}",
            display_name=f"Developer {i}",
            bio=f"Building Python & PyTorch ML systems #{i}",
            location="Delhi" if i % 2 == 0 else "Zurich",
            links=[f"https://dev{i}.io"],
            label_source="github_social_accounts",
            bio_emb=compute_bio_embedding(f"Building Python & PyTorch ML systems #{i}"),
        )
        plat2 = "mastodon" if i >= 19 else "personal_site"
        p2 = ProfileRecord(
            person_id=pid,
            platform=plat2,
            url=f"https://dev{i}.io",
            username=f"dev_{i}",
            display_name=f"Developer {i}",
            bio=f"Building Python & PyTorch ML systems #{i}",
            location="Delhi" if i % 2 == 0 else "Zurich",
            links=[f"https://github.com/dev_{i}"],
            label_source="github_plus_bidirectional_rel_me",
            bio_emb=compute_bio_embedding(f"Building Python & PyTorch ML systems #{i}"),
        )
        profiles.extend([p1, p2])
        pos_pairs.append((p1, p2, p2.label_source))

    cfg = {
        "seed": 42,
        "dataset": {"name": "osint-link", "output_dir": str(tmp_path), "negatives_per_positive": 2},
        "splits": {"train_ratio": 0.60, "val_ratio": 0.20, "test_ratio": 0.20, "ood_platforms": ["mastodon", "devto"]},
    }
    split_and_export_dataset(profiles, pos_pairs, cfg)
    return tmp_path


def test_char_cnn_encoder_output_shape_and_unit_norm() -> None:
    """CharCNNEncoder must encode usernames into L2-normalized embeddings."""
    enc = CharCNNEncoder(out_dim=64)
    ids = torch.stack([
        torch.from_numpy(encode_username_chars("alaotach")),
        torch.from_numpy(encode_username_chars("aryanmi2001")),
    ])
    emb = enc(ids)
    assert emb.shape == (2, 64)
    norms = torch.linalg.norm(emb, dim=-1)
    assert norms.tolist() == pytest.approx([1.0, 1.0], abs=1e-4)


def test_phase_c_training_calibration_ablations_and_resolver_compat(tmp_path: Path) -> None:
    """Full Phase C training, temperature calibration, 7 ablations, and NeuralResolver.score() must run offline."""
    data_dir = _build_temp_osint_link(tmp_path / "data")
    ckpt_path = tmp_path / "neural_resolver.pt"

    res = run_neural_training_and_ablations(
        data_dir=data_dir,
        model_out_path=ckpt_path,
        seed=42,
        epochs_contrastive=2,
        epochs_cls=3,
        n_bootstrap=30,
        run_ablations=True,
        ood_platforms={"mastodon", "devto"},
    )
    assert res["status"] == "completed"
    assert ckpt_path.exists()
    assert res["calibration"]["temperature"] > 0.0
    assert len(res["calibration"]["reliability_after"]) == 10
    assert set(res["ablations"].keys()) == {
        "full_model",
        "no_username_char_cnn",
        "no_bio_location",
        "no_photo_clip_phash",
        "no_links_overlap",
        "no_contrastive_pretrain",
        "no_hard_negatives",
    }
    for sp in ("test", "ood_platform_test"):
        assert "pr_auc" in res["splits"][sp]["metrics"]
        assert "ece" in res["splits"][sp]["metrics"]

    # Verify NeuralResolver.load() and score(target, hit) compatibility with OSINTAgent
    loaded = NeuralResolver.load(ckpt_path)
    target = Entity.create(entity_type=EntityType.USERNAME, value="dev_1")
    hit = Hit(
        target_entity_id=target.id,
        discovered_type=EntityType.PROFILE,
        discovered_value="https://dev1.io",
        relation="account_on",
        source_tool="maigret",
        metadata={"username": "dev_1", "name": "Developer 1", "bio": "Python ML"},
    )
    score = loaded.score(target, hit)
    assert 0.0 <= score <= 1.0
