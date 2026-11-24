"""Tests for embedding generation, LRU cache hits, and TF-IDF fallback."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from osint_agent import embeddings


@pytest.fixture(autouse=True)
def reset_embeddings():
    """Reset global embedding state before and after each test."""
    embeddings.reset_model_state()
    yield
    embeddings.reset_model_state()


def test_embedding_cache_hit() -> None:
    """Repeated calls with identical text should hit in-memory LRU cache without re-encoding."""
    mock_model = MagicMock()
    # Return mock unit vector of dimension 8
    dummy_vec = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    dummy_vec /= np.linalg.norm(dummy_vec)
    mock_model.encode.return_value = dummy_vec

    with patch("osint_agent.embeddings.get_model", return_value=mock_model):
        text = "Cybersecurity intelligence and OSINT analysis"

        # First call: cache miss, calls encode
        vec1 = embeddings.get_embedding(text)
        assert vec1 is not None
        assert mock_model.encode.call_count == 1

        # Second call: cache hit, encode must NOT be called again
        vec2 = embeddings.get_embedding(text)
        assert vec2 is not None
        assert mock_model.encode.call_count == 1
        assert np.array_equal(vec1, vec2)


def test_embedding_fallback_to_tfidf() -> None:
    """When SentenceTransformer fails or is unavailable, fallback to TF-IDF cosine."""
    # Force get_model to return None (model failure)
    with patch("osint_agent.embeddings.get_model", return_value=None):
        text1 = "artificial intelligence machine learning algorithm"
        text2 = "machine learning and deep neural network algorithms"
        text3 = "completely unrelated acoustic music guitar strings"

        sim_related = embeddings.compute_text_similarity(text1, text2)
        sim_unrelated = embeddings.compute_text_similarity(text1, text3)

        assert sim_related > 0.20
        assert sim_unrelated == 0.0


def test_embedding_empty_text_handling() -> None:
    """Empty or whitespace-only strings must return None/0.0 without errors."""
    with patch("osint_agent.embeddings.get_model", return_value=None):
        assert embeddings.get_embedding("") is None
        assert embeddings.get_embedding("   ") is None
        assert embeddings.compute_text_similarity("", "valid text") == 0.0
        assert embeddings.compute_text_similarity("valid text", "") == 0.0
