"""ML-based profile keyword extraction, pairwise similarity scoring, and agglomerative clustering."""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlparse

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from osint_agent.ml.embeddings import get_embedding

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?|\n])\s+|\s*\|\s*")
_SIGNAL_RE = re.compile(r"(@|https?://|\b(?:email|contact|address|location|office|story|stories|project|repo|app|developer|built|founder|student)\b)", re.IGNORECASE)


def compress_profile_text_ml(text: str, max_chars: int = 1200) -> str:
    """Use sentence-level TF-IDF + signal density to strip boilerplate and keep high-value sentences for LLM."""
    raw = (text or "").strip()
    if len(raw) <= max_chars:
        return raw

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(raw) if len(s.strip()) >= 20]
    if len(sentences) <= 2:
        return raw[:max_chars]

    try:
        vec = TfidfVectorizer(stop_words="english", max_features=128)
        mat = vec.fit_transform(sentences)
        # Score each sentence by mean TF-IDF weight + bonus for OSINT contact/project/location signals
        tfidf_scores = np.asarray(mat.sum(axis=1)).ravel() / np.maximum(1, np.asarray((mat > 0).sum(axis=1)).ravel())
        ranked: list[tuple[float, int, str]] = []
        for idx, sent in enumerate(sentences):
            bonus = 0.35 * len(_SIGNAL_RE.findall(sent))
            ranked.append((float(tfidf_scores[idx]) + bonus, idx, sent))
        ranked.sort(key=lambda x: x[0], reverse=True)

        chosen: list[tuple[int, str]] = []
        total = 0
        for _, idx, sent in ranked:
            if total + len(sent) + 2 > max_chars and chosen:
                continue
            chosen.append((idx, sent))
            total += len(sent) + 2
            if total >= max_chars:
                break
        # Restore original reading order
        chosen.sort(key=lambda x: x[0])
        return " | ".join(s for _, s in chosen)[:max_chars]
    except ValueError:
        return raw[:max_chars]


_SKILL_VOCAB = (
    "python", "kotlin", "java", "javascript", "typescript", "react", "nextjs", "node",
    "fastapi", "django", "flask", "android", "flutter", "swift", "rust", "go", "c++",
    "pytorch", "tensorflow", "sklearn", "scikit-learn", "machine learning", "deep learning",
    "nlp", "ai", "llm", "docker", "kubernetes", "aws", "gcp", "sql", "postgresql", "mongodb",
)


def enrich_profiles_ml(hits: list[Any]) -> None:
    """Use local ML/NLP (TF-IDF extractive summary, skill matching, spaCy NER) to enrich profiles without any LLM API call."""
    nlp = None
    try:
        from osint_agent.collectors.extractor import get_spacy_nlp
        nlp = get_spacy_nlp()
    except Exception:
        nlp = None

    for h in hits:
        meta = getattr(h, "metadata", None)
        raw = getattr(h, "raw_data", None)
        if not isinstance(meta, dict) or not isinstance(raw, dict):
            continue

        excerpt = str(raw.get("_page_text_excerpt") or "")
        bio = str(meta.get("bio") or "")
        combined = " ".join(p for p in (bio, excerpt) if p).strip()
        if not combined:
            continue

        # 1. Extractive TF-IDF summary when bio is short/missing or no summary exists
        if excerpt and not meta.get("ml_summary"):
            summary = compress_profile_text_ml(excerpt, max_chars=280)
            if summary and len(summary) >= 30:
                meta["ml_summary"] = summary

        # 2. Tech/domain skill extraction
        low = combined.lower()
        found_skills = [
            sk for sk in _SKILL_VOCAB
            if re.search(rf"(?<![a-z0-9]){re.escape(sk)}(?![a-z0-9])", low)
        ]
        if found_skills:
            existing_sk = list(meta.get("skills") or [])
            meta["skills"] = list(dict.fromkeys(existing_sk + found_skills))[:12]

        # 3. Local spaCy NER for missing name / location / organizations
        if nlp is not None and (not meta.get("name") or not meta.get("location") or not meta.get("organizations")):
            try:
                doc = nlp(combined[:1500])
                orgs: list[str] = list(meta.get("organizations") or [])
                for ent in doc.ents:
                    val = ent.text.strip()
                    if len(val) < 2 or len(val) > 80:
                        continue
                    if ent.label_ == "PERSON" and not meta.get("name"):
                        meta["name"] = val
                    elif ent.label_ == "GPE" and not meta.get("location"):
                        meta["location"] = val
                    elif ent.label_ == "ORG" and val not in orgs:
                        orgs.append(val)
                if orgs:
                    meta["organizations"] = orgs[:5]
            except Exception:
                pass


def rank_profiles_for_llm(hits: list[Any], max_profiles: int = 6) -> list[Any]:
    """Select ONLY unstructured, high-value profiles that still need LLM extraction (after ML enrichment)."""
    scored: list[tuple[float, Any]] = []
    for h in hits:
        meta = getattr(h, "metadata", {}) or {}
        raw = getattr(h, "raw_data", {}) or {}
        excerpt = str(raw.get("_page_text_excerpt") or "")
        bio = str(meta.get("bio") or "")
        stories = meta.get("stories") or []
        kws = meta.get("ml_keywords") or []

        # Skip profiles with no text at all
        if not excerpt and not bio and not stories:
            continue

        # Skip profiles already richly structured by deterministic APIs/Play Store/GitHub/ML
        already_structured = bool(
            meta.get("name")
            and (meta.get("location") or meta.get("office_address") or meta.get("email") or meta.get("extra_fields"))
            and (bio or stories)
        )
        if already_structured:
            continue

        info_score = (
            min(1.0, len(bio) / 200.0) * 0.30
            + min(1.0, len(excerpt) / 1500.0) * 0.35
            + min(1.0, len(kws) / 6.0) * 0.20
            + (0.15 if stories or meta.get("email") or meta.get("location") else 0.0)
        )
        scored.append((info_score, h))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in scored[:max_profiles]]


def extract_profile_keywords(text: str, top_k: int = 8) -> list[str]:
    """Extract top distinguishing keywords/phrases from scraped profile text using TF-IDF."""
    clean = " ".join(_TOKEN_RE.findall((text or "").lower()))
    if len(clean) < 12:
        return []
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=64)
        mat = vec.fit_transform([clean])
        names = vec.get_feature_names_out()
        scores = mat.toarray()[0]
        ranked = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
        return [term for term, score in ranked[:top_k] if score > 0]
    except ValueError:
        return []


def _profile_text_blob(attrs: dict[str, Any]) -> str:
    parts = [
        str(attrs.get("name") or ""),
        str(attrs.get("bio") or ""),
        str(attrs.get("location") or ""),
        str(attrs.get("company") or ""),
        str(attrs.get("llm_summary") or ""),
        " ".join(str(s) for s in (attrs.get("stories") or []) if s),
        " ".join(str(p) for p in (attrs.get("projects_or_stories") or []) if p),
        " ".join(str(k) for k in (attrs.get("ml_keywords") or []) if k),
    ]
    return " ".join(p.strip() for p in parts if p and p.strip())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _compute_text_sim_matrix(texts: list[str]) -> np.ndarray:
    """Compute pairwise text cosine similarity using SentenceTransformer when cached, else TF-IDF."""
    n = len(texts)
    sim = np.eye(n, dtype=np.float64)
    if n < 2:
        return sim

    non_empty_idx = [i for i, t in enumerate(texts) if t.strip()]
    if len(non_empty_idx) < 2:
        return sim

    # Try dense SentenceTransformer embeddings first (no network download; uses local cache if present)
    dense_vecs: list[np.ndarray | None] = [get_embedding(texts[i]) if i in non_empty_idx else None for i in range(n)]
    if all(dense_vecs[i] is not None for i in non_empty_idx):
        for i in non_empty_idx:
            for j in non_empty_idx:
                if i < j:
                    val = float(np.dot(dense_vecs[i], dense_vecs[j]))  # type: ignore[arg-type]
                    val = max(0.0, min(1.0, val))
                    sim[i, j] = sim[j, i] = val
        return sim

    # Reliable offline TF-IDF cosine similarity matrix
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        mat = vec.fit_transform([texts[i] for i in non_empty_idx])
        sub = cosine_similarity(mat)
        for a_idx, i in enumerate(non_empty_idx):
            for b_idx, j in enumerate(non_empty_idx):
                if i != j:
                    sim[i, j] = max(0.0, min(1.0, float(sub[a_idx, b_idx])))
    except ValueError:
        pass
    return sim


def _compute_handle_sim_matrix(handles: list[str]) -> np.ndarray:
    """Compute character n-gram TF-IDF cosine similarity across usernames/display names."""
    n = len(handles)
    sim = np.eye(n, dtype=np.float64)
    if n < 2:
        return sim
    non_empty = [i for i, h in enumerate(handles) if h.strip()]
    if len(non_empty) < 2:
        return sim
    try:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        mat = vec.fit_transform([handles[i].lower() for i in non_empty])
        sub = cosine_similarity(mat)
        for a_idx, i in enumerate(non_empty):
            for b_idx, j in enumerate(non_empty):
                if i != j:
                    sim[i, j] = max(0.0, min(1.0, float(sub[a_idx, b_idx])))
    except ValueError:
        pass
    return sim


def discover_ml_profile_relations(
    graph: Any,
    similarity_threshold: float = 0.45,
    cluster_distance_threshold: float = 0.65,
) -> dict[str, Any]:
    """Run ML feature extraction, pairwise similarity, and AgglomerativeClustering across profile nodes."""
    profile_nodes: list[tuple[str, dict[str, Any]]] = [
        (nid, data)
        for nid, data in graph.nodes(data=True)
        if data.get("entity_type") == "profile"
    ]
    if len(profile_nodes) < 2:
        return {"clusters": [], "edges_added": 0}

    n = len(profile_nodes)
    texts: list[str] = []
    handles: list[str] = []
    link_sets: list[set[str]] = []
    loc_sets: list[set[str]] = []
    emails: list[str] = []

    for _, data in profile_nodes:
        attrs = data.get("attributes") or {}
        if not attrs.get("ml_keywords"):
            blob = _profile_text_blob(attrs)
            kws = extract_profile_keywords(blob, top_k=6)
            if kws:
                attrs["ml_keywords"] = kws
                data["attributes"] = attrs

        texts.append(_profile_text_blob(attrs))
        handles.append(f"{attrs.get('username', '')} {attrs.get('name', '')}".strip())
        links = {
            urlparse(str(u)).netloc.lower()
            for u in (attrs.get("linked_urls") or [])
            if str(u).startswith("http")
        }
        link_sets.append({h for h in links if h})
        loc_tokens = set(_TOKEN_RE.findall(str(attrs.get("location") or "").lower()))
        loc_sets.append(loc_tokens)
        emails.append(str(attrs.get("email") or "").strip().lower())

    text_sim = _compute_text_sim_matrix(texts)
    handle_sim = _compute_handle_sim_matrix(handles)

    combined_sim = np.eye(n, dtype=np.float64)
    edges_added = 0

    for i in range(n):
        for j in range(i + 1, n):
            signals: list[str] = []
            t_s = float(text_sim[i, j])
            h_s = float(handle_sim[i, j])
            l_s = _jaccard(link_sets[i], link_sets[j])
            loc_s = _jaccard(loc_sets[i], loc_sets[j])
            email_match = 1.0 if (emails[i] and emails[i] == emails[j]) else 0.0

            if t_s >= 0.25:
                signals.append(f"semantic_text:{t_s:.2f}")
            if h_s >= 0.5:
                signals.append(f"handle_ngram:{h_s:.2f}")
            if l_s > 0.0:
                signals.append(f"shared_links:{l_s:.2f}")
            if loc_s > 0.0:
                signals.append(f"shared_location:{loc_s:.2f}")
            if email_match > 0.0:
                signals.append("shared_email")

            score = min(
                1.0,
                0.45 * t_s + 0.25 * h_s + 0.15 * l_s + 0.10 * loc_s + 0.25 * email_match,
            )
            combined_sim[i, j] = combined_sim[j, i] = score

            if score >= similarity_threshold and signals:
                src_id = profile_nodes[i][0]
                dst_id = profile_nodes[j][0]
                if not graph.has_edge(src_id, dst_id):
                    graph.add_edge(
                        src_id,
                        dst_id,
                        relation="ml_similar_profile",
                        source="ml_profile_matcher",
                        source_tool="ml_profile_matcher",
                        score=round(score, 4),
                        evidence=", ".join(signals),
                    )
                    edges_added += 1

    # Agglomerative clustering on precomputed distance matrix (1 - similarity)
    dist_matrix = np.clip(1.0 - combined_sim, 0.0, 1.0)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=cluster_distance_threshold,
    )
    labels = clustering.fit_predict(dist_matrix)

    clusters_map: dict[int, list[str]] = {}
    for idx, label in enumerate(labels):
        cid = int(label)
        nid, data = profile_nodes[idx]
        attrs = data.get("attributes") or {}
        attrs["ml_cluster_id"] = cid
        data["attributes"] = attrs
        clusters_map.setdefault(cid, []).append(str(data.get("value") or nid))

    multi_clusters = [
        {"cluster_id": cid, "profiles": urls}
        for cid, urls in sorted(clusters_map.items())
        if len(urls) >= 2
    ]
    result = {"clusters": multi_clusters, "edges_added": edges_added}
    graph.graph["ml_profile_clusters"] = result
    return result
