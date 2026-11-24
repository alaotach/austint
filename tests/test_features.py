"""Tests for OSINT feature extraction and TargetContext handling."""

from __future__ import annotations

import pytest

from osint_agent.ml.features import FEATURE_NAMES, TargetContext, extract_features
from osint_agent.core.schema import Entity, EntityType, Hit


def test_feature_names_order() -> None:
    """FEATURE_NAMES must be immutable single source of truth for column order."""
    expected = [
        "name_sim",
        "username_sim",
        "text_sim",
        "loc_match",
        "domain_link",
        "uname_len",
        "token_overlap",
        "has_text",
        "exact_name_match",
    ]
    assert FEATURE_NAMES == expected


def test_target_context_absorb() -> None:
    """TargetContext should absorb entities, hits, and kwargs without duplicates."""
    ctx = TargetContext()

    # Absorb Entity
    entity = Entity.create(
        entity_type=EntityType.ORG,
        value="Acme Corp",
        attributes={"location": "San Francisco", "domain": "acme.com", "bio": "Security org"},
    )
    ctx.absorb(entity)
    assert "Acme Corp" in ctx.names
    assert "San Francisco" in ctx.locations
    assert "acme.com" in ctx.domains
    assert "Security org" in ctx.texts

    # Absorb Hit
    hit = Hit(
        target_entity_id=entity.id,
        discovered_type=EntityType.DOMAIN,
        discovered_value="api.acme.com",
        relation="subdomain_of",
        source_tool="crtsh",
        metadata={"username": "acme_admin"},
    )
    ctx.absorb(hit)
    assert "api.acme.com" in ctx.domains
    assert "acme_admin" in ctx.usernames

    # Absorb kwargs & deduplication
    ctx.absorb(names=["Acme Corp", "Acme International"], locations="San Francisco")
    assert ctx.names.count("Acme Corp") == 1
    assert "Acme International" in ctx.names
    assert ctx.locations.count("San Francisco") == 1


def test_extract_features_exact_name_match() -> None:
    """exact_name_match should be 1.0 on match (case-insensitive) and 0.0 otherwise."""
    ctx = TargetContext(names=["Alice Smith"])

    hit_match = Hit(
        target_entity_id="test",
        discovered_type=EntityType.PUBLIC_FIGURE,
        discovered_value="alice smith",
        relation="alias",
        source_tool="wikidata",
    )
    feats_match = extract_features(ctx, hit_match)
    assert feats_match["exact_name_match"] == 1.0
    assert feats_match["name_sim"] == 1.0

    hit_diff = Hit(
        target_entity_id="test",
        discovered_type=EntityType.PUBLIC_FIGURE,
        discovered_value="Bob Jones",
        relation="alias",
        source_tool="wikidata",
    )
    feats_diff = extract_features(ctx, hit_diff)
    assert feats_diff["exact_name_match"] == 0.0
    assert feats_diff["name_sim"] < 0.40


def test_extract_features_username_and_uname_len() -> None:
    """username_sim and uname_len should correctly compute similarity and character length."""
    ctx = TargetContext(usernames=["cyber_warrior"])

    hit = Hit(
        target_entity_id="test",
        discovered_type=EntityType.CONSENTING_PERSON,
        discovered_value="cyber_warrior99",
        relation="profile",
        source_tool="social",
        metadata={"username": "cyber_warrior99"},
    )
    feats = extract_features(ctx, hit)
    assert feats["username_sim"] > 0.85
    assert feats["uname_len"] == float(len("cyber_warrior99"))


def test_extract_features_domain_link() -> None:
    """domain_link must identify root and subdomain connections."""
    ctx = TargetContext(domains=["targetcompany.com"])

    # Valid subdomain
    hit_sub = Hit(
        target_entity_id="test",
        discovered_type=EntityType.DOMAIN,
        discovered_value="vpn.targetcompany.com",
        relation="subdomain_of",
        source_tool="crtsh",
    )
    assert extract_features(ctx, hit_sub)["domain_link"] == 1.0

    # Unrelated domain
    hit_unrelated = Hit(
        target_entity_id="test",
        discovered_type=EntityType.DOMAIN,
        discovered_value="malicious-phishing.org",
        relation="external",
        source_tool="crawler",
    )
    assert extract_features(ctx, hit_unrelated)["domain_link"] == 0.0


def test_extract_features_empty_text_and_defaults() -> None:
    """Feature extraction must handle empty contexts and empty candidate texts gracefully."""
    empty_ctx = TargetContext()
    hit_empty = Hit(
        target_entity_id="test",
        discovered_type=EntityType.ORG,
        discovered_value="Minimal Corp",
        relation="related",
        source_tool="test",
    )

    feats = extract_features(empty_ctx, hit_empty)

    assert feats["has_text"] == 0.0
    assert feats["text_sim"] == 0.0
    assert feats["name_sim"] == 0.0
    assert feats["username_sim"] == 0.0
    assert feats["loc_match"] == 0.0
    assert feats["domain_link"] == 0.0
    assert feats["token_overlap"] == 0.0
    assert feats["exact_name_match"] == 0.0
    assert feats["uname_len"] == float(len("Minimal Corp"))
