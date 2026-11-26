"""Phase D: Collective identity resolution with a GraphSAGE/GAT neural network.

Builds candidate-entity graphs from both dataset splits and OSINTAgent runs
(nodes = candidate profiles/entities, edges = links, mentions, shared_domain,
similarity), refines Phase C pair scores via neighborhood message passing, and
exposes `NeuralGNNResolver` for `--resolver neural_gnn`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from osint_agent.core.schema import Entity, Hit
from osint_agent.ml.baselines import bootstrap_metrics_ci, compute_metrics, pair_dict_to_feature_vector
from osint_agent.ml.features import TargetContext
from osint_agent.ml.neural import (
    NeuralResolver,
    fit_temperature,
    reliability_bins,
)
from osint_agent.ml.resolver import BaseResolver


EDGE_TYPES = ("links", "mentions", "shared_domain", "similarity")
EDGE_TYPE_TO_ID = {name: idx for idx, name in enumerate(EDGE_TYPES)}

# Generic hosting domains that shouldn't by themselves create strong shared-domain edges
GENERIC_DOMAINS = {
    "github.com",
    "gitlab.com",
    "bitbucket.org",
    "mastodon.social",
    "twitter.com",
    "x.com",
    "instagram.com",
    "facebook.com",
    "linkedin.com",
    "youtube.com",
    "reddit.com",
    "medium.com",
    "t.me",
    "bsky.app",
    "gmail.com",
    "outlook.com",
    "yahoo.com",
    "proton.me",
    "protonmail.com",
}

NODE_FEATURE_DIM = 12


def _extract_domain(url_or_email: str) -> str:
    s = (url_or_email or "").strip().lower()
    if not s:
        return ""
    if "@" in s and not s.startswith("http"):
        dom = s.split("@", 1)[-1].strip()
        return dom if "." in dom and dom not in GENERIC_DOMAINS else ""
    try:
        parsed = urlparse(s if "://" in s else f"https://{s}")
        host = (parsed.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if not host or "." not in host or host in GENERIC_DOMAINS:
            return ""
        return host
    except Exception:
        return ""


def _extract_mentions(text: str) -> set[str]:
    if not text:
        return set()
    return {m.lower() for m in re.findall(r"@([a-zA-Z0-9_.-]{2,32})", text)}


def _norm_url(url: str) -> str:
    s = (url or "").strip().lower().rstrip("/")
    if s.startswith("https://"):
        s = s[8:]
    elif s.startswith("http://"):
        s = s[7:]
    if s.startswith("www."):
        s = s[4:]
    return s


def _safe_logit(p: float, eps: float = 1e-4) -> float:
    clipped = min(max(float(p), eps), 1.0 - eps)
    return math.log(clipped / (1.0 - clipped))


@dataclass
class CandidateGraphData:
    """Tensorized candidate-entity graph for CollectiveGNN."""

    node_features: torch.Tensor  # [N, NODE_FEATURE_DIM]
    edge_index: torch.Tensor  # [2, E] (src, dst)
    edge_type: torch.Tensor  # [E] in {0, 1, 2, 3}
    phase_c_scores: torch.Tensor  # [N]
    chain_scores: torch.Tensor  # [N]
    labels: torch.Tensor | None  # [N] float or None
    node_ids: list[str]
    edge_counts_by_type: dict[str, int]
    depths: list[int]


def _profile_signals(profile: dict[str, Any]) -> tuple[set[str], set[str], set[str], str, str, str]:
    links = {_norm_url(u) for u in (profile.get("links") or []) if _norm_url(u)}
    domains = {_extract_domain(u) for u in (profile.get("links") or [])}
    domains.discard("")
    bio = str(profile.get("bio") or "")
    uname = str(profile.get("username") or "").strip().lower()
    dname = str(profile.get("display_name") or "").strip().lower()
    mentions = _extract_mentions(bio)
    if uname:
        mentions.add(uname)
    return links, domains, mentions, uname, dname, bio.lower()


def _char_jaccard(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    sa = {a[i : i + 2] for i in range(max(len(a) - 1, 1))}
    sb = {b[i : i + 2] for i in range(max(len(b) - 1, 1))}
    union = sa | sb
    return float(len(sa & sb) / len(union)) if union else 0.0


def build_dataset_graph(
    rows: list[dict[str, Any]],
    phase_c_scores: list[float] | np.ndarray,
    *,
    confidence_decay: float = 0.85,
    similarity_threshold: float = 0.65,
) -> CandidateGraphData:
    """Build a candidate graph from dataset pair rows and Phase C pair scores.

    Each row is a candidate pair `(left, right)` with its Phase C score.
    Edges connect pairs that share declared links (`links`), `@mentions` or
    usernames (`mentions`), personal/org domains (`shared_domain`), or high
    profile/pair similarity (`similarity`). Also computes `chain_scores` using
    the legacy "confidence multiplied along the chain" heuristic.
    """
    n = len(rows)
    if n == 0:
        return CandidateGraphData(
            node_features=torch.zeros((0, NODE_FEATURE_DIM), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros((0,), dtype=torch.long),
            phase_c_scores=torch.zeros((0,), dtype=torch.float32),
            chain_scores=torch.zeros((0,), dtype=torch.float32),
            labels=torch.zeros((0,), dtype=torch.float32),
            node_ids=[],
            edge_counts_by_type={t: 0 for t in EDGE_TYPES},
            depths=[],
        )

    scores = [float(s) for s in phase_c_scores]
    labels = [float(r.get("label", 0)) for r in rows]
    node_ids = [str(r.get("pair_id") or f"pair_{i}") for i, r in enumerate(rows)]

    # Precompute pair signals and hand features
    node_links: list[set[str]] = []
    node_domains: list[set[str]] = []
    node_mentions: list[set[str]] = []
    node_unames: list[str] = []
    node_dnames: list[str] = []
    hand_feats: list[list[float]] = []

    for r in rows:
        if "left" in r or "right" in r:
            left = r.get("left") or {}
            right = r.get("right") or {}
        else:
            left = {
                "username": r.get("username_a", ""),
                "display_name": r.get("display_name_a", ""),
                "bio": r.get("bio_a", ""),
                "links": r.get("links_a") or r.get("linked_urls_a") or [],
            }
            right = {
                "username": r.get("username_b", ""),
                "display_name": r.get("display_name_b", ""),
                "bio": r.get("bio_b", ""),
                "links": r.get("links_b") or r.get("linked_urls_b") or [],
            }
        l_links, l_doms, l_ments, l_u, l_d, _ = _profile_signals(left)
        r_links, r_doms, r_ments, r_u, r_d, _ = _profile_signals(right)

        node_links.append(l_links | r_links)
        node_domains.append(l_doms | r_doms)
        node_mentions.append(l_ments | r_ments)
        node_unames.append(r_u or l_u)
        node_dnames.append(r_d or l_d)
        hand_feats.append(pair_dict_to_feature_vector(r).tolist())

    # Build typed edges
    src_list: list[int] = []
    dst_list: list[int] = []
    etype_list: list[int] = []
    edge_counts = {t: 0 for t in EDGE_TYPES}

    # Index inverted maps for O(N) discovery of links/domains/mentions
    link_to_nodes: dict[str, list[int]] = {}
    domain_to_nodes: dict[str, list[int]] = {}
    mention_to_nodes: dict[str, list[int]] = {}

    for idx in range(n):
        for lk in node_links[idx]:
            link_to_nodes.setdefault(lk, []).append(idx)
        for dm in node_domains[idx]:
            domain_to_nodes.setdefault(dm, []).append(idx)
        for mn in node_mentions[idx]:
            mention_to_nodes.setdefault(mn, []).append(idx)

    added_typed_edges: set[tuple[int, int, int]] = set()

    def _add_undirected(u: int, v: int, etype_name: str) -> None:
        if u == v:
            return
        et_id = EDGE_TYPE_TO_ID[etype_name]
        if (u, v, et_id) not in added_typed_edges:
            added_typed_edges.add((u, v, et_id))
            src_list.extend([u, v])
            dst_list.extend([v, u])
            etype_list.extend([et_id, et_id])
            added_typed_edges.add((v, u, et_id))
            edge_counts[etype_name] += 2

    for bucket in link_to_nodes.values():
        if 1 < len(bucket) <= 32:
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    _add_undirected(bucket[i], bucket[j], "links")

    for bucket in domain_to_nodes.values():
        if 1 < len(bucket) <= 32:
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    _add_undirected(bucket[i], bucket[j], "shared_domain")

    for bucket in mention_to_nodes.values():
        if 1 < len(bucket) <= 32:
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    _add_undirected(bucket[i], bucket[j], "mentions")

    # Similarity edges (bounded per node to keep graph sparse on large splits)
    for i in range(n):
        sims: list[tuple[float, int]] = []
        for j in range(i + 1, min(n, i + 128)):
            u_sim = _char_jaccard(node_unames[i], node_unames[j])
            d_sim = _char_jaccard(node_dnames[i], node_dnames[j])
            sim = 0.6 * u_sim + 0.4 * d_sim
            if sim >= similarity_threshold:
                sims.append((sim, j))
        sims.sort(reverse=True)
        for _, j in sims[:4]:
            _add_undirected(i, j, "similarity")

    # Compute node degrees and chain-confidence heuristic along shortest path from highest-score anchor
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in zip(src_list, dst_list):
        adj[u].append(v)

    # Hop depth & chain confidence: if a row specifies `hop_depth` / `parent_idx`, use it;
    # otherwise compute BFS hop depth from local component's highest Phase C anchor.
    depths = [int(r.get("hop_depth", 1)) if "hop_depth" in r else 1 for r in rows]
    chain_scores = [0.0] * n

    if any("parent_idx" in r for r in rows):
        for i, r in enumerate(rows):
            p_idx = r.get("parent_idx")
            if p_idx is None or not (0 <= int(p_idx) < n) or int(p_idx) == i:
                chain_scores[i] = scores[i]
            else:
                parent_conf = chain_scores[int(p_idx)] or scores[int(p_idx)]
                chain_scores[i] = float(parent_conf * confidence_decay * scores[i])
    else:
        visited = [False] * n
        for start in sorted(range(n), key=lambda k: scores[k], reverse=True):
            if visited[start]:
                continue
            visited[start] = True
            depths[start] = 1
            chain_scores[start] = scores[start]
            queue = [start]
            while queue:
                curr = queue.pop(0)
                for nb in adj[curr]:
                    if not visited[nb]:
                        visited[nb] = True
                        depths[nb] = depths[curr] + 1
                        chain_scores[nb] = float(chain_scores[curr] * confidence_decay * scores[nb])
                        queue.append(nb)

    # Assemble node feature matrix [N, 12]
    max_deg = max((len(adj[i]) for i in range(n)), default=1) or 1
    feats = np.zeros((n, NODE_FEATURE_DIM), dtype=np.float32)
    for i in range(n):
        hf = hand_feats[i]  # 8 dims from extract_pair_features
        p_c = scores[i]
        feats[i, 0] = p_c
        feats[i, 1] = _safe_logit(p_c) / 6.0
        feats[i, 2] = chain_scores[i]
        feats[i, 3] = min(depths[i] / 4.0, 1.0)
        feats[i, 4] = len(adj[i]) / float(max_deg)
        feats[i, 5] = float(hf[0]) if len(hf) > 0 else 0.0  # name/username sim
        feats[i, 6] = float(hf[1]) if len(hf) > 1 else 0.0
        feats[i, 7] = float(hf[2]) if len(hf) > 2 else 0.0
        feats[i, 8] = float(hf[3]) if len(hf) > 3 else 0.0
        feats[i, 9] = float(hf[4]) if len(hf) > 4 else 0.0
        feats[i, 10] = 1.0 if node_links[i] else 0.0
        feats[i, 11] = 1.0 if node_domains[i] else 0.0

    edge_index = (
        torch.tensor([src_list, dst_list], dtype=torch.long)
        if src_list
        else torch.zeros((2, 0), dtype=torch.long)
    )
    edge_type = (
        torch.tensor(etype_list, dtype=torch.long)
        if etype_list
        else torch.zeros((0,), dtype=torch.long)
    )

    return CandidateGraphData(
        node_features=torch.from_numpy(feats),
        edge_index=edge_index,
        edge_type=edge_type,
        phase_c_scores=torch.tensor(scores, dtype=torch.float32),
        chain_scores=torch.tensor(chain_scores, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.float32),
        node_ids=node_ids,
        edge_counts_by_type=edge_counts,
        depths=depths,
    )


def build_agent_run_graph(
    graph: nx.DiGraph,
    *,
    confidence_decay: float = 0.85,
    similarity_threshold: float = 0.65,
) -> CandidateGraphData:
    """Convert an OSINTAgent NetworkX DiGraph into a CandidateGraphData object."""
    node_ids = list(graph.nodes())
    n = len(node_ids)
    if n == 0:
        return CandidateGraphData(
            node_features=torch.zeros((0, NODE_FEATURE_DIM), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros((0,), dtype=torch.long),
            phase_c_scores=torch.zeros((0,), dtype=torch.float32),
            chain_scores=torch.zeros((0,), dtype=torch.float32),
            labels=None,
            node_ids=[],
            edge_counts_by_type={t: 0 for t in EDGE_TYPES},
            depths=[],
        )

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    src_list: list[int] = []
    dst_list: list[int] = []
    etype_list: list[int] = []
    edge_counts = {t: 0 for t in EDGE_TYPES}
    added_edges: set[tuple[int, int, int]] = set()

    def _add_edge(u: int, v: int, etype_name: str) -> None:
        if u == v:
            return
        et_id = EDGE_TYPE_TO_ID[etype_name]
        if (u, v, et_id) not in added_edges:
            added_edges.add((u, v, et_id))
            src_list.append(u)
            dst_list.append(v)
            etype_list.append(et_id)
            edge_counts[etype_name] += 1

    # 1. Existing agent-run edges
    incoming_scores: dict[int, float] = {}
    for u_id, v_id, edata in graph.edges(data=True):
        u = id_to_idx[u_id]
        v = id_to_idx[v_id]
        rel = str(edata.get("relation") or "").lower()
        if "domain" in rel:
            et = "shared_domain"
        elif "mention" in rel:
            et = "mentions"
        elif "sim" in rel or "match" in rel:
            et = "similarity"
        else:
            et = "links"
        _add_edge(u, v, et)
        _add_edge(v, u, et)
        if edata.get("score") is not None:
            sc = float(edata["score"])
            incoming_scores[v] = max(incoming_scores.get(v, 0.0), sc)

    # 2. Additional shared-domain / mention / similarity edges between candidate nodes
    node_domains: list[set[str]] = []
    node_mentions: list[set[str]] = []
    node_values: list[str] = []

    for nid in node_ids:
        ndata = graph.nodes[nid]
        attrs = ndata.get("attributes") or {}
        val = str(ndata.get("value") or "")
        bio = str(attrs.get("bio") or attrs.get("description") or "")
        links = list(attrs.get("links") or [])
        if attrs.get("website"):
            links.append(str(attrs["website"]))
        doms = {_extract_domain(lk) for lk in links}
        if ndata.get("entity_type") == "domain":
            doms.add(val.lower())
        doms.discard("")
        ments = _extract_mentions(bio)
        if attrs.get("username"):
            ments.add(str(attrs["username"]).lower())
        node_domains.append(doms)
        node_mentions.append(ments)
        node_values.append(val.lower())

    for i in range(n):
        for j in range(i + 1, n):
            if node_domains[i] & node_domains[j]:
                _add_edge(i, j, "shared_domain")
                _add_edge(j, i, "shared_domain")
            if node_mentions[i] & node_mentions[j]:
                _add_edge(i, j, "mentions")
                _add_edge(j, i, "mentions")
            if _char_jaccard(node_values[i], node_values[j]) >= similarity_threshold:
                _add_edge(i, j, "similarity")
                _add_edge(j, i, "similarity")

    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in zip(src_list, dst_list):
        adj[u].append(v)
    max_deg = max((len(adj[i]) for i in range(n)), default=1) or 1

    feats = np.zeros((n, NODE_FEATURE_DIM), dtype=np.float32)
    phase_c_scores = np.zeros((n,), dtype=np.float32)
    chain_scores = np.zeros((n,), dtype=np.float32)
    depths: list[int] = []

    for i, nid in enumerate(node_ids):
        ndata = graph.nodes[nid]
        depth = int(ndata.get("depth", 0))
        depths.append(depth)
        chain_conf = float(ndata.get("confidence", 0.5))
        p_c = incoming_scores.get(i, chain_conf)
        if depth == 0:
            p_c = 1.0
            chain_conf = 1.0
        phase_c_scores[i] = p_c
        chain_scores[i] = chain_conf

        feats[i, 0] = p_c
        feats[i, 1] = _safe_logit(p_c) / 6.0
        feats[i, 2] = chain_conf
        feats[i, 3] = min(depth / 4.0, 1.0)
        feats[i, 4] = len(adj[i]) / float(max_deg)
        feats[i, 5] = 1.0 if depth == 0 else 0.0
        feats[i, 10] = 1.0 if node_mentions[i] else 0.0
        feats[i, 11] = 1.0 if node_domains[i] else 0.0

    return CandidateGraphData(
        node_features=torch.from_numpy(feats),
        edge_index=(
            torch.tensor([src_list, dst_list], dtype=torch.long)
            if src_list
            else torch.zeros((2, 0), dtype=torch.long)
        ),
        edge_type=(
            torch.tensor(etype_list, dtype=torch.long)
            if etype_list
            else torch.zeros((0,), dtype=torch.long)
        ),
        phase_c_scores=torch.from_numpy(phase_c_scores),
        chain_scores=torch.from_numpy(chain_scores),
        labels=None,
        node_ids=node_ids,
        edge_counts_by_type=edge_counts,
        depths=depths,
    )


class GraphSAGEGATLayer(nn.Module):
    """Plain-PyTorch GraphSAGE + GAT attention layer with typed edge embeddings."""

    def __init__(self, in_dim: int, out_dim: int, num_edge_types: int = len(EDGE_TYPES), dropout: float = 0.1) -> None:
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        self.edge_emb = nn.Embedding(num_edge_types, out_dim)
        self.attn_vec = nn.Linear(out_dim * 3, 1)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        h_self = self.lin_self(h)
        n = h.size(0)
        if edge_index.numel() == 0:
            return F.gelu(self.norm(h_self))

        src = edge_index[0]
        dst = edge_index[1]

        h_src = self.lin_neigh(h[src])
        h_dst = h_self[dst]
        e_emb = self.edge_emb(edge_type)

        # GAT attention logit per directed edge src -> dst
        attn_in = torch.cat([h_dst, h_src, e_emb], dim=-1)
        raw_alpha = F.leaky_relu(self.attn_vec(attn_in).squeeze(-1), negative_slope=0.2)

        # Numerically stable segment softmax grouped by dst node
        max_per_dst = torch.full((n,), -1e9, device=h.device, dtype=h.dtype)
        max_per_dst.scatter_reduce_(0, dst, raw_alpha, reduce="amax", include_self=True)
        exp_alpha = torch.exp(raw_alpha - max_per_dst[dst])

        denom = torch.zeros((n,), device=h.device, dtype=h.dtype)
        denom.index_add_(0, dst, exp_alpha)
        alpha = exp_alpha / (denom[dst] + 1e-8)
        alpha = self.dropout(alpha)

        msg = (h_src + e_emb) * alpha.unsqueeze(-1)
        agg = torch.zeros((n, h_self.size(-1)), device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, msg)

        out = self.norm(h_self + agg)
        return F.gelu(out)


class CollectiveGNN(nn.Module):
    """2-layer GraphSAGE+GAT model refining Phase C logits using neighbor context."""

    def __init__(self, in_dim: int = NODE_FEATURE_DIM, hidden_dim: int = 32, dropout: float = 0.15) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layer1 = GraphSAGEGATLayer(hidden_dim, hidden_dim, dropout=dropout)
        self.layer2 = GraphSAGEGATLayer(hidden_dim, hidden_dim, dropout=dropout)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_data: CandidateGraphData) -> torch.Tensor:
        x = graph_data.node_features
        h0 = F.gelu(self.in_proj(x))
        h1 = self.layer1(h0, graph_data.edge_index, graph_data.edge_type)
        h2 = self.layer2(h1, graph_data.edge_index, graph_data.edge_type)
        delta = self.delta_head(torch.cat([h0, h2], dim=-1)).squeeze(-1)

        base_logits = torch.logit(graph_data.phase_c_scores.clamp(1e-4, 1.0 - 1e-4))
        return base_logits + delta


def _predict_probs(model: CollectiveGNN, graph_data: CandidateGraphData, temperature: float) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(graph_data)
        probs = torch.sigmoid(logits / max(float(temperature), 1e-3))
    return probs.cpu().numpy()


def analyze_helps_vs_hurts(
    graph_data: CandidateGraphData,
    gnn_probs: np.ndarray,
    *,
    threshold: float = 0.5,
    max_examples: int = 10,
) -> dict[str, Any]:
    """Compare Phase C alone vs Phase C + GNN per node to show where GNN helps or hurts."""
    assert graph_data.labels is not None
    y = graph_data.labels.cpu().numpy().astype(int)
    p_c = graph_data.phase_c_scores.cpu().numpy()
    p_gnn = np.asarray(gnn_probs, dtype=np.float32)

    n = len(y)
    degrees = [0] * n
    if graph_data.edge_index.numel() > 0:
        for dst in graph_data.edge_index[1].tolist():
            degrees[dst] += 1

    helped: list[dict[str, Any]] = []
    hurt: list[dict[str, Any]] = []
    unchanged_correct = 0
    unchanged_wrong = 0

    for i in range(n):
        c_pred = int(p_c[i] >= threshold)
        g_pred = int(p_gnn[i] >= threshold)
        c_ok = c_pred == y[i]
        g_ok = g_pred == y[i]
        if not c_ok and g_ok:
            helped.append(
                {
                    "node_id": graph_data.node_ids[i],
                    "label": int(y[i]),
                    "phase_c_prob": round(float(p_c[i]), 4),
                    "gnn_prob": round(float(p_gnn[i]), 4),
                    "degree": degrees[i],
                    "depth": graph_data.depths[i],
                    "flip": "FN->TP" if y[i] == 1 else "FP->TN",
                }
            )
        elif c_ok and not g_ok:
            hurt.append(
                {
                    "node_id": graph_data.node_ids[i],
                    "label": int(y[i]),
                    "phase_c_prob": round(float(p_c[i]), 4),
                    "gnn_prob": round(float(p_gnn[i]), 4),
                    "degree": degrees[i],
                    "depth": graph_data.depths[i],
                    "flip": "TP->FN" if y[i] == 1 else "TN->FP",
                }
            )
        elif g_ok:
            unchanged_correct += 1
        else:
            unchanged_wrong += 1

    return {
        "total_nodes": n,
        "helped_count": len(helped),
        "hurt_count": len(hurt),
        "net_gain": len(helped) - len(hurt),
        "unchanged_correct": unchanged_correct,
        "unchanged_wrong": unchanged_wrong,
        "avg_degree_helped": round(float(np.mean([h["degree"] for h in helped])), 2) if helped else 0.0,
        "avg_degree_hurt": round(float(np.mean([h["degree"] for h in hurt])), 2) if hurt else 0.0,
        "helped_examples": helped[:max_examples],
        "hurt_examples": hurt[:max_examples],
    }


def train_gnn_model(
    train_graph: CandidateGraphData,
    val_graph: CandidateGraphData,
    *,
    seed: int = 42,
    epochs: int = 80,
    lr: float = 5e-3,
    weight_decay: float = 1e-3,
    hidden_dim: int = 32,
) -> dict[str, Any]:
    """Train CollectiveGNN on `train_graph` and temperature-calibrate on `val_graph`."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CollectiveGNN(in_dim=NODE_FEATURE_DIM, hidden_dim=hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    assert train_graph.labels is not None
    y_train = train_graph.labels

    best_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(train_graph)
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        loss.backward()
        optimizer.step()

        if val_graph.node_features.size(0) > 0 and val_graph.labels is not None:
            model.eval()
            with torch.no_grad():
                v_loss = float(F.binary_cross_entropy_with_logits(model(val_graph), val_graph.labels).item())
            if v_loss <= best_loss:
                best_loss = v_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    # Temperature scaling on validation graph
    temperature = 1.0
    val_ece_before = 0.0
    val_ece_after = 0.0
    val_bins: list[dict[str, Any]] = []
    if val_graph.node_features.size(0) > 0 and val_graph.labels is not None:
        with torch.no_grad():
            val_logits = model(val_graph).cpu().numpy()
        val_y = val_graph.labels.cpu().numpy().astype(int)
        val_probs_before = 1.0 / (1.0 + np.exp(-val_logits))
        val_ece_before = compute_metrics(val_y, val_probs_before)["ece"]
        temperature = fit_temperature(val_logits, val_y)
        val_probs_after = 1.0 / (1.0 + np.exp(-val_logits / temperature))
        val_ece_after = compute_metrics(val_y, val_probs_after)["ece"]
        val_bins = reliability_bins(val_y, val_probs_after)

    return {
        "model": model,
        "temperature": float(temperature),
        "val_ece_before": round(float(val_ece_before), 4),
        "val_ece_after": round(float(val_ece_after), 4),
        "val_reliability_bins": val_bins,
    }


def evaluate_three_way(
    graph_data: CandidateGraphData,
    gnn_model: CollectiveGNN,
    temperature: float,
    *,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare Phase C alone vs Phase C + GNN vs old chain heuristic on a labeled graph."""
    assert graph_data.labels is not None
    y = graph_data.labels.cpu().numpy().astype(int)
    p_c = graph_data.phase_c_scores.cpu().numpy()
    p_chain = graph_data.chain_scores.cpu().numpy()
    p_gnn = _predict_probs(gnn_model, graph_data, temperature)

    methods = {
        "phase_c_alone": p_c,
        "phase_c_plus_gnn": p_gnn,
        "chain_confidence_heuristic": p_chain,
    }
    results: dict[str, Any] = {}
    for name, probs in methods.items():
        metrics = compute_metrics(y, probs)
        ci = bootstrap_metrics_ci(y, probs, n_bootstrap=n_bootstrap, seed=seed)
        results[name] = {
            **metrics,
            "ci95": ci,
        }

    results["helps_vs_hurts"] = analyze_helps_vs_hurts(graph_data, p_gnn)
    results["graph_stats"] = {
        "num_nodes": int(graph_data.node_features.size(0)),
        "num_edges": int(graph_data.edge_index.size(1)),
        "edge_counts_by_type": graph_data.edge_counts_by_type,
    }
    return results


class NeuralGNNResolver(BaseResolver):
    """Drop-in resolver combining Phase C pair scoring + Phase D CollectiveGNN refinement."""

    def __init__(
        self,
        neural_resolver: NeuralResolver,
        gnn_model: CollectiveGNN,
        *,
        temperature: float = 1.0,
        confidence_decay: float = 0.85,
    ) -> None:
        self.neural_resolver = neural_resolver
        self.gnn_model = gnn_model
        self.gnn_model.eval()
        self.temperature = float(temperature)
        self.confidence_decay = float(confidence_decay)

    def score(self, target: TargetContext | Entity, hit: Hit) -> float:
        """Per-hit Phase C score used during incremental frontier expansion."""
        return self.neural_resolver.score(target, hit)

    def refine_graph(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Run collective GNN message passing over the completed agent graph."""
        if graph.number_of_nodes() == 0:
            return graph
        gdata = build_agent_run_graph(graph, confidence_decay=self.confidence_decay)
        gnn_probs = _predict_probs(self.gnn_model, gdata, self.temperature)
        for i, nid in enumerate(gdata.node_ids):
            ndata = graph.nodes[nid]
            depth = int(ndata.get("depth", 0))
            if depth == 0:
                continue
            ndata["chain_confidence"] = float(ndata.get("confidence", 0.0))
            ndata["phase_c_score"] = float(gdata.phase_c_scores[i].item())
            refined = round(float(gnn_probs[i]), 4)
            ndata["gnn_confidence"] = refined
            ndata["confidence"] = refined
        return graph

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        n_model = self.neural_resolver.model
        assert n_model is not None
        payload = {
            "neural_state_dict": {k: v.cpu() for k, v in n_model.state_dict().items()},
            "neural_temperature": self.neural_resolver.temperature,
            "bio_dim": n_model.bio_dim,
            "clip_dim": n_model.clip_dim,
            "use_username": n_model.use_username,
            "use_bio_loc": n_model.use_bio_loc,
            "use_photo": n_model.use_photo,
            "use_links": n_model.use_links,
            "gnn_state_dict": {k: v.cpu() for k, v in self.gnn_model.state_dict().items()},
            "gnn_temperature": self.temperature,
            "confidence_decay": self.confidence_decay,
            "gnn_hidden_dim": self.gnn_model.in_proj.out_features,
        }
        torch.save(payload, out)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "NeuralGNNResolver":
        from osint_agent.ml.neural import MultimodalIdentityNet

        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        n_model = MultimodalIdentityNet(
            bio_dim=int(payload["bio_dim"]),
            clip_dim=int(payload["clip_dim"]),
            use_username=bool(payload.get("use_username", True)),
            use_bio_loc=bool(payload.get("use_bio_loc", True)),
            use_photo=bool(payload.get("use_photo", True)),
            use_links=bool(payload.get("use_links", True)),
        )
        n_model.load_state_dict(payload["neural_state_dict"])
        n_res = NeuralResolver(
            n_model,
            temperature=float(payload.get("neural_temperature", 1.0)),
            bio_dim=int(payload["bio_dim"]),
            clip_dim=int(payload["clip_dim"]),
        )

        g_model = CollectiveGNN(
            in_dim=NODE_FEATURE_DIM,
            hidden_dim=int(payload.get("gnn_hidden_dim", 32)),
        )
        g_model.load_state_dict(payload["gnn_state_dict"])
        return cls(
            neural_resolver=n_res,
            gnn_model=g_model,
            temperature=float(payload.get("gnn_temperature", 1.0)),
            confidence_decay=float(payload.get("confidence_decay", 0.85)),
        )
