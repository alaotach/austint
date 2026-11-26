"""Offline unit tests for Phase A: OSINT-Link weak-supervision dataset builder, splits, and stats."""

from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
import pytest
import respx
from PIL import Image

from osint_agent.ml.datasets.builder import (
    ProfileRecord,
    compute_bio_embedding,
    compute_clip_embedding,
    compute_phash,
    crawl_self_declared_profiles,
    hash_person_id,
    mine_hard_negatives,
    phash_similarity,
    split_and_export_dataset,
    verify_no_person_leakage,
)
from scripts.osint_link_stats import compute_stats, load_splits


def _make_test_jpeg(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=color).save(buf, format="JPEG")
    return buf.getvalue()


def test_phase_a_phash_and_offline_clip_and_privacy() -> None:
    """pHash and CLIP fallback embeddings must compute cleanly without storing raw photos or raw bios."""
    img1 = _make_test_jpeg((20, 120, 220))
    img2 = _make_test_jpeg((22, 122, 222))

    h1 = compute_phash(img1)
    h2 = compute_phash(img2)
    assert len(h1) == 16
    assert phash_similarity(h1, h1) == 1.0
    assert phash_similarity(h1, h2) >= 0.5

    clip1 = compute_clip_embedding(img1, allow_download=False)
    assert len(clip1) == 64
    assert any(v != 0.0 for v in clip1)

    pid = hash_person_id("github:alice", salt="test-salt")
    assert pid.startswith("person_") and "alice" not in pid


@pytest.mark.asyncio
@respx.mock
async def test_phase_a_crawler_self_declared_links_and_ssrf_guard() -> None:
    """Crawler should extract GitHub + Mastodon self-declared links while blocking SSRF private URLs."""
    img_bytes = _make_test_jpeg((10, 80, 180))

    respx.get("https://api.github.com/orgs/testorg/public_members").mock(
        return_value=httpx.Response(200, json=[{"login": "alice_oss"}])
    )
    respx.get("https://api.github.com/users/alice_oss").mock(
        return_value=httpx.Response(
            200,
            json={
                "type": "User",
                "login": "alice_oss",
                "name": "Alice Maintainer",
                "bio": "Python & ML maintainer",
                "location": "Berlin",
                "company": "OpenLab",
                "blog": "https://alice-oss.dev",
                "avatar_url": "https://avatars.githubusercontent.com/u/1?v=4",
            },
        )
    )
    respx.get("https://api.github.com/users/alice_oss/social_accounts").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"provider": "generic", "url": "http://127.0.0.1/secret"},  # SSRF must be blocked
                {"provider": "mastodon", "url": "https://fosstodon.org/@alice_oss"},
            ],
        )
    )
    respx.get("https://avatars.githubusercontent.com/u/1?v=4").mock(
        return_value=httpx.Response(200, content=img_bytes)
    )
    respx.get("https://alice-oss.dev/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://alice-oss.dev").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text='<html><head><title>Alice Dev</title><meta name="description" content="ML Researcher"></head>'
                 '<body><a rel="me" href="https://github.com/alice_oss">GitHub</a></body></html>',
        )
    )
    respx.get("https://fosstodon.org/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://fosstodon.org/@alice_oss").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text='<html><head><title>Alice (@alice_oss)</title></head><body>Mastodon profile</body></html>',
        )
    )
    respx.get("https://mastodon.social/api/v1/directory").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "url": "https://mastodon.social/@bob_m",
                    "username": "bob_m",
                    "display_name": "Bob M",
                    "note": "<p>Rust dev</p>",
                    "fields": [{"name": "Web", "value": '<a href="https://bob.example.org">bob.example.org</a>', "verified_at": "2025-01-01T00:00:00Z"}],
                }
            ],
        )
    )

    cfg = {
        "seed": 42,
        "dataset": {"default_salt": "test-salt"},
        "crawl": {
            "max_seed_users": 10,
            "max_profiles_per_person": 5,
            "per_domain_delay_sec": 0.0,
            "concurrency": 4,
            "request_timeout_sec": 5.0,
            "max_response_bytes": 204800,
            "seed_orgs": ["testorg"],
            "mastodon_instances": ["mastodon.social"],
        },
        "features": {"clip_model": "openai/clip-vit-base-patch32"},
    }

    profiles, pos_pairs = await crawl_self_declared_profiles(cfg)
    assert len(profiles) >= 4
    assert len(pos_pairs) >= 3
    # SSRF localhost must never appear
    assert not any("127.0.0.1" in p.url for p in profiles)
    # Bidirectional rel="me" detected
    assert any(src == "github_plus_bidirectional_rel_me" for _, _, src in pos_pairs)


def test_phase_a_group_split_zero_person_leakage_and_ood_isolation(tmp_path: Path) -> None:
    """Person-group split must have zero person overlap across train/val/test and zero OOD leakage."""
    img_a = _make_test_jpeg((30, 60, 90))
    img_b = _make_test_jpeg((200, 50, 50))
    profiles: list[ProfileRecord] = []
    pos_pairs: list[tuple[ProfileRecord, ProfileRecord, str]] = []

    for i in range(15):
        pid = hash_person_id(f"user_{i}", salt="s")
        p_gh = ProfileRecord(
            person_id=pid,
            platform="github",
            url=f"https://github.com/dev_{i}",
            username=f"dev_{i}",
            display_name=f"Developer {i}",
            bio=f"Building open source Python library #{i}",
            location="Delhi" if i % 2 == 0 else "Berlin",
            company="AcmeOrg" if i % 2 == 0 else "BetaLabs",
            links=[f"https://dev{i}.example.io"],
            label_source="github_social_accounts",
            phash=compute_phash(img_a if i % 2 == 0 else img_b),
            clip_emb=compute_clip_embedding(img_a if i % 2 == 0 else img_b, allow_download=False),
            bio_emb=compute_bio_embedding(f"Building open source Python library #{i}"),
        )
        # OOD platform for i >= 12, in-distribution personal_site for i < 12
        plat2 = "mastodon" if i >= 12 else "personal_site"
        p_other = ProfileRecord(
            person_id=pid,
            platform=plat2,
            url=f"https://{'mastodon.social/@' if plat2 == 'mastodon' else ''}dev{i}.example.io",
            username=f"dev_{i}",
            display_name=f"Developer {i}",
            bio=f"Personal homepage for Python developer #{i}",
            location="Delhi" if i % 2 == 0 else "Berlin",
            links=[f"https://github.com/dev_{i}"],
            label_source="github_plus_bidirectional_rel_me",
            phash=p_gh.phash,
            clip_emb=p_gh.clip_emb,
            bio_emb=compute_bio_embedding(f"Personal homepage for Python developer #{i}"),
        )
        profiles.extend([p_gh, p_other])
        pos_pairs.append((p_gh, p_other, p_other.label_source))

    cfg = {
        "seed": 42,
        "dataset": {"name": "osint-link", "output_dir": str(tmp_path), "negatives_per_positive": 2},
        "splits": {"train_ratio": 0.70, "val_ratio": 0.15, "test_ratio": 0.15, "ood_platforms": ["mastodon", "devto"]},
    }
    manifest = split_and_export_dataset(profiles, pos_pairs, cfg)
    assert manifest["store_raw_bios"] is False
    assert manifest["store_raw_photos"] is False

    splits = load_splits(tmp_path)
    stats = compute_stats(splits, ood_platforms={"mastodon", "devto"})
    assert stats["person_leakage_verified"] is True
    assert stats["total_positives"] == 15
    assert stats["splits"]["ood_platform_test"]["positives"] == 3

    # Verify no raw bio text is stored in exported rows
    for rows in splits.values():
        for row in rows:
            assert "bio" not in row
            assert "raw_image" not in row
            assert "hand_features" in row
            assert "phash_a" in row and "clip_emb_a" in row

    # Prove verify_no_person_leakage catches injected leakage
    bad_splits = {
        "train": [splits["train"][0]],
        "val": [splits["train"][0]],  # intentional person leakage
        "test": splits["test"],
    }
    with pytest.raises(AssertionError, match="Person leakage"):
        verify_no_person_leakage(bad_splits)


def test_phase_a_hard_negatives_all_distinct_persons() -> None:
    """mine_hard_negatives must only pair distinct person_ids and tag neg_type."""
    p1 = ProfileRecord(person_id="p1", platform="github", url="https://github.com/alex_dev", username="alex_dev", display_name="Alex Kumar", location="Delhi", company="OrgA", links=["https://a.io"])
    p1_b = ProfileRecord(person_id="p1", platform="personal_site", url="https://a.io", username="alex_dev", display_name="Alex Kumar")
    p2 = ProfileRecord(person_id="p2", platform="github", url="https://github.com/alex_dev2", username="alex_dev2", display_name="Alex Kumar", location="London", company="OrgB", links=["https://b.io"])
    p3 = ProfileRecord(person_id="p3", platform="github", url="https://github.com/alex_d", username="alex_d", display_name="Alex K", location="NYC", company="OrgC", links=["https://c.io"])

    negs = mine_hard_negatives([(p1, p1_b, "rel_me")], [p1, p1_b, p2, p3], negatives_per_positive=2, seed=42)
    assert len(negs) == 2
    for a, b, src, ntype in negs:
        assert a.person_id != b.person_id
        assert src == "hard_negative_mining"
        assert ntype in {"similar_username_diff_person", "same_name_diff_org_city", "same_display_name_diff_links", "diff_person_fallback"}
