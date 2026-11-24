"""Text embedding utilities with lazy-loaded SentenceTransformer, LRU cache, and TF-IDF fallback."""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# Global state for lazy model loading and caching
_MODEL: Any | None = None
_MODEL_LOAD_FAILED: bool = False
_EMBEDDING_CACHE: OrderedDict[str, np.ndarray] = OrderedDict()
MAX_CACHE_SIZE = 2048


def reset_model_state() -> None:
    """Reset model loader state (used primarily in test mocking)."""
    global _MODEL, _MODEL_LOAD_FAILED, _EMBEDDING_CACHE
    _MODEL = None
    _MODEL_LOAD_FAILED = False
    _EMBEDDING_CACHE.clear()


def get_model() -> Any | None:
    """Lazy load SentenceTransformer model or mark fallback on failure."""
    global _MODEL, _MODEL_LOAD_FAILED
    if _MODEL is not None:
        return _MODEL
    if _MODEL_LOAD_FAILED:
        return None

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        logger.info("Loading SentenceTransformer('all-MiniLM-L6-v2')...")
        try:
            _MODEL = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        except Exception:
            _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return _MODEL
    except Exception as exc:
        logger.warning(
            "SentenceTransformer failed to load: %s. Falling back to TF-IDF cosine similarity.",
            exc,
        )
        _MODEL_LOAD_FAILED = True
        return None


def _hash_text(text: str) -> str:
    """Deterministic hash for cache lookup."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def compute_tfidf_cosine(text1: str, text2: str) -> float:
    """Calculate TF-IDF cosine similarity between two strings as a reliable fallback."""
    t1 = text1.strip()
    t2 = text2.strip()
    if not t1 or not t2:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform([t1, t2])
        sim = float((tfidf[0] @ tfidf[1].T).toarray()[0][0])
        return max(0.0, min(1.0, round(sim, 4)))
    except ValueError:
        # Occurs if texts contain only stop words or punctuation
        return 0.0


def get_embedding(text: str) -> np.ndarray | None:
    """Obtain a normalized vector representation of text with LRU caching."""
    clean_text = text.strip()
    if not clean_text:
        return None

    key = _hash_text(clean_text)
    if key in _EMBEDDING_CACHE:
        # Move to end for LRU order
        _EMBEDDING_CACHE.move_to_end(key)
        return _EMBEDDING_CACHE[key]

    model = get_model()
    if model is None:
        return None

    try:
        vec = model.encode(clean_text, normalize_embeddings=True, show_progress_bar=False)
        vec = np.asarray(vec, dtype=np.float32)

        # Enforce LRU cache bound
        if len(_EMBEDDING_CACHE) >= MAX_CACHE_SIZE:
            _EMBEDDING_CACHE.popitem(last=False)
        _EMBEDDING_CACHE[key] = vec
        return vec
    except Exception as err:
        logger.warning("Error generating embedding for text: %s", err)
        return None


def warm_embedding_cache(texts: list[str]) -> None:
    """Pre-compute embeddings in a single batched pass for high-throughput feature extraction."""
    clean_texts = [t.strip() for t in texts if t and t.strip()]
    if not clean_texts:
        return

    uncached: list[str] = []
    uncached_keys: list[str] = []
    for t in clean_texts:
        k = _hash_text(t)
        if k not in _EMBEDDING_CACHE:
            uncached.append(t)
            uncached_keys.append(k)

    if not uncached:
        return

    model = get_model()
    if model is None:
        return

    try:
        vecs = model.encode(
            uncached,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for key, vec in zip(uncached_keys, vecs):
            if len(_EMBEDDING_CACHE) >= MAX_CACHE_SIZE:
                _EMBEDDING_CACHE.popitem(last=False)
            _EMBEDDING_CACHE[key] = np.asarray(vec, dtype=np.float32)
    except Exception as err:
        logger.debug("warm_embedding_cache error: %s", err)


def compute_text_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity using dense embeddings or TF-IDF fallback."""
    t1 = text1.strip()
    t2 = text2.strip()
    if not t1 or not t2:
        return 0.0
    if t1.lower() == t2.lower():
        return 1.0

    emb1 = get_embedding(t1)
    emb2 = get_embedding(t2)

    if emb1 is not None and emb2 is not None:
        dot = float(np.dot(emb1, emb2))
        return max(0.0, min(1.0, round(dot, 4)))

    return compute_tfidf_cosine(t1, t2)
