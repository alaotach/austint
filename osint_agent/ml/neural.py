"""Phase C: Multimodal Identity-Linking Model (CharCNN + Bio/Location + CLIP/pHash + Links) with Contrastive Pretraining & Temperature Calibration."""

from __future__ import annotations

import hashlib
import logging
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from osint_agent.core.schema import Entity, Hit
from osint_agent.ml.baselines import bootstrap_metrics_ci, compute_ece, pair_dict_to_feature_vector
from osint_agent.ml.datasets.builder import compute_bio_embedding, phash_similarity
from osint_agent.ml.features import FEATURE_NAMES, TargetContext, extract_features
from osint_agent.ml.resolver import BaseResolver

logger = logging.getLogger(__name__)

_CHAR_VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789_.-@"
_CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(_CHAR_VOCAB)}  # 0 = pad/unk
_MAX_UNAME_LEN = 32


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_username_chars(text: str, max_len: int = _MAX_UNAME_LEN) -> np.ndarray:
    """Convert username/display-name string to fixed-length integer character IDs."""
    arr = np.zeros(max_len, dtype=np.int64)
    clean = (text or "").lower().strip()[:max_len]
    for i, ch in enumerate(clean):
        arr[i] = _CHAR_TO_IDX.get(ch, 0)
    return arr


def _pad_or_trim(vec: list[float], target_dim: int) -> np.ndarray:
    out = np.zeros(target_dim, dtype=np.float32)
    if vec:
        n = min(len(vec), target_dim)
        out[:n] = np.asarray(vec[:n], dtype=np.float32)
    return out


def _location_hash_vec(loc: str, dim: int = 16) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9]+", (loc or "").lower())
    for tok in tokens:
        idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16) % dim
        out[idx] += 1.0
    norm = float(np.linalg.norm(out))
    if norm > 0:
        out /= norm
    return out


class CharCNNEncoder(nn.Module):
    """Multi-scale 1D Character CNN for usernames and display handles."""

    def __init__(self, vocab_size: int = len(_CHAR_VOCAB) + 1, emb_dim: int = 24, out_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, 32, kernel_size=k, padding=k // 2) for k in (2, 3, 4)])
        self.proj = nn.Linear(32 * 3, out_dim)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        # char_ids: (B, L)
        x = self.embedding(char_ids).transpose(1, 2)  # (B, E, L)
        pooled = [F.relu(conv(x)).max(dim=-1).values for conv in self.convs]
        return F.normalize(self.proj(torch.cat(pooled, dim=-1)), dim=-1)


class MultimodalIdentityNet(nn.Module):
    """Dual-encoder + pair-classification network supporting modality ablations."""

    def __init__(
        self,
        bio_dim: int = 64,
        clip_dim: int = 64,
        loc_dim: int = 16,
        link_dim: int = 4,
        hand_dim: int = len(FEATURE_NAMES),
        proj_dim: int = 64,
        use_username: bool = True,
        use_bio_loc: bool = True,
        use_photo: bool = True,
        use_links: bool = True,
    ) -> None:
        super().__init__()
        self.bio_dim = bio_dim
        self.clip_dim = clip_dim
        self.loc_dim = loc_dim
        self.link_dim = link_dim
        self.hand_dim = hand_dim
        self.proj_dim = proj_dim
        self.use_username = use_username
        self.use_bio_loc = use_bio_loc
        self.use_photo = use_photo
        self.use_links = use_links

        self.uname_enc = CharCNNEncoder(out_dim=proj_dim)
        self.bio_loc_enc = nn.Sequential(
            nn.Linear(bio_dim + loc_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.photo_enc = nn.Sequential(
            nn.Linear(clip_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.fuse_profile = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

        # Pair classifier takes [|ea - eb|, ea * eb, link_feats, hand_feats, phash_sim]
        pair_in = proj_dim * 2 + link_dim + hand_dim + 1
        self.classifier = nn.Sequential(
            nn.Linear(pair_in, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def encode_profile(
        self,
        char_ids: torch.Tensor,
        bio_vec: torch.Tensor,
        loc_vec: torch.Tensor,
        clip_vec: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_username:
            u_emb = self.uname_enc(char_ids)
        else:
            u_emb = torch.zeros(char_ids.size(0), self.proj_dim, device=char_ids.device)

        if self.use_bio_loc:
            b_emb = F.normalize(self.bio_loc_enc(torch.cat([bio_vec, loc_vec], dim=-1)), dim=-1)
        else:
            b_emb = torch.zeros(char_ids.size(0), self.proj_dim, device=char_ids.device)

        if self.use_photo:
            p_emb = F.normalize(self.photo_enc(clip_vec), dim=-1)
        else:
            p_emb = torch.zeros(char_ids.size(0), self.proj_dim, device=char_ids.device)

        fused = self.fuse_profile(torch.cat([u_emb, b_emb, p_emb], dim=-1))
        return F.normalize(fused, dim=-1)

    def forward(
        self,
        char_a: torch.Tensor,
        char_b: torch.Tensor,
        bio_a: torch.Tensor,
        bio_b: torch.Tensor,
        loc_a: torch.Tensor,
        loc_b: torch.Tensor,
        clip_a: torch.Tensor,
        clip_b: torch.Tensor,
        link_feats: torch.Tensor,
        hand_feats: torch.Tensor,
        phash_sim: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ea = self.encode_profile(char_a, bio_a, loc_a, clip_a)
        eb = self.encode_profile(char_b, bio_b, loc_b, clip_b)
        lf = link_feats if self.use_links else torch.zeros_like(link_feats)
        ps = phash_sim if self.use_photo else torch.zeros_like(phash_sim)
        pair_vec = torch.cat([torch.abs(ea - eb), ea * eb, lf, hand_feats, ps], dim=-1)
        logits = self.classifier(pair_vec).squeeze(-1)
        return logits, ea, eb


def rows_to_tensors(
    rows: list[dict[str, Any]],
    bio_dim: int = 64,
    clip_dim: int = 64,
) -> dict[str, torch.Tensor]:
    """Convert exported OSINT-Link pair dicts into aligned PyTorch tensors."""
    n = len(rows)
    char_a = np.zeros((n, _MAX_UNAME_LEN), dtype=np.int64)
    char_b = np.zeros((n, _MAX_UNAME_LEN), dtype=np.int64)
    bio_a = np.zeros((n, bio_dim), dtype=np.float32)
    bio_b = np.zeros((n, bio_dim), dtype=np.float32)
    loc_a = np.zeros((n, 16), dtype=np.float32)
    loc_b = np.zeros((n, 16), dtype=np.float32)
    clip_a = np.zeros((n, clip_dim), dtype=np.float32)
    clip_b = np.zeros((n, clip_dim), dtype=np.float32)
    link_f = np.zeros((n, 4), dtype=np.float32)
    hand_f = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    phash_s = np.zeros((n, 1), dtype=np.float32)
    labels = np.zeros(n, dtype=np.float32)

    for i, r in enumerate(rows):
        char_a[i] = encode_username_chars(str(r.get("username_a") or r.get("display_name_a") or ""))
        char_b[i] = encode_username_chars(str(r.get("username_b") or r.get("display_name_b") or ""))
        bio_a[i] = _pad_or_trim(r.get("bio_emb_a") or [], bio_dim)
        bio_b[i] = _pad_or_trim(r.get("bio_emb_b") or [], bio_dim)
        loc_a[i] = _location_hash_vec(str(r.get("location_a") or ""))
        loc_b[i] = _location_hash_vec(str(r.get("location_b") or ""))
        clip_a[i] = _pad_or_trim(r.get("clip_emb_a") or [], clip_dim)
        clip_b[i] = _pad_or_trim(r.get("clip_emb_b") or [], clip_dim)
        link_f[i] = np.array(
            [
                float(r.get("links_jaccard", 0.0)),
                min(1.0, float(r.get("shared_domain_count", 0)) / 5.0),
                min(1.0, float(r.get("link_count_a", 0)) / 10.0),
                min(1.0, float(r.get("link_count_b", 0)) / 10.0),
            ],
            dtype=np.float32,
        )
        hand_f[i] = pair_dict_to_feature_vector(r).astype(np.float32)
        phash_s[i, 0] = float(r.get("phash_sim", 0.0))
        labels[i] = float(r.get("label", 0))

    return {
        "char_a": torch.from_numpy(char_a),
        "char_b": torch.from_numpy(char_b),
        "bio_a": torch.from_numpy(bio_a),
        "bio_b": torch.from_numpy(bio_b),
        "loc_a": torch.from_numpy(loc_a),
        "loc_b": torch.from_numpy(loc_b),
        "clip_a": torch.from_numpy(clip_a),
        "clip_b": torch.from_numpy(clip_b),
        "link_feats": torch.from_numpy(link_f),
        "hand_feats": torch.from_numpy(hand_f),
        "phash_sim": torch.from_numpy(phash_s),
        "labels": torch.from_numpy(labels),
    }


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Optimize scalar temperature T > 0 on validation logits by minimizing NLL."""
    if len(logits) == 0:
        return 1.0
    t_param = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
    opt = torch.optim.LBFGS([t_param], lr=0.1, max_iter=50)
    log_t = torch.tensor(logits, dtype=torch.float32)
    y_t = torch.tensor(y_true, dtype=torch.float32)

    def _closure() -> torch.Tensor:
        opt.zero_grad()
        temp = torch.clamp(t_param, min=0.05, max=10.0)
        loss = F.binary_cross_entropy_with_logits(log_t / temp, y_t)
        loss.backward()
        return loss

    opt.step(_closure)
    return round(float(torch.clamp(t_param, min=0.05, max=10.0).item()), 4)


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    """Compute per-bin confidence vs empirical accuracy for reliability diagrams."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = float(bins[i]), float(bins[i + 1])
        mask = (y_prob >= lo) & (y_prob <= hi if i == n_bins - 1 else y_prob < hi)
        cnt = int(mask.sum())
        out.append(
            {
                "bin_low": round(lo, 2),
                "bin_high": round(hi, 2),
                "count": cnt,
                "mean_conf": round(float(y_prob[mask].mean()), 4) if cnt > 0 else None,
                "empirical_acc": round(float(y_true[mask].mean()), 4) if cnt > 0 else None,
            }
        )
    return out


def train_multimodal_model(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    seed: int = 42,
    epochs_contrastive: int = 8,
    epochs_cls: int = 15,
    batch_size: int = 32,
    lr: float = 2e-3,
    use_username: bool = True,
    use_bio_loc: bool = True,
    use_photo: bool = True,
    use_links: bool = True,
    use_contrastive: bool = True,
    use_hard_negatives: bool = True,
) -> tuple[MultimodalIdentityNet, float, dict[str, Any]]:
    """Train MultimodalIdentityNet (contrastive pretraining + pair classification + temperature scaling)."""
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Infer embedding dimensions dynamically from dataset
    bio_dim = max([len(r.get("bio_emb_a") or []) for r in train_rows] + [64])
    clip_dim = max([len(r.get("clip_emb_a") or []) for r in train_rows] + [64])

    filtered_train = train_rows
    if not use_hard_negatives:
        # Keep only fallback negatives + positives when ablating hard negative mining
        filtered_train = [r for r in train_rows if r["label"] == 1 or r.get("neg_type") == "diff_person_fallback"]
        if len(filtered_train) < 4 or len({r["label"] for r in filtered_train}) < 2:
            filtered_train = train_rows

    tr_t = rows_to_tensors(filtered_train, bio_dim=bio_dim, clip_dim=clip_dim)
    va_t = rows_to_tensors(val_rows or filtered_train, bio_dim=bio_dim, clip_dim=clip_dim)

    model = MultimodalIdentityNet(
        bio_dim=bio_dim,
        clip_dim=clip_dim,
        use_username=use_username,
        use_bio_loc=use_bio_loc,
        use_photo=use_photo,
        use_links=use_links,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Stage 1: Contrastive pretraining on positive pairs (in-batch negatives)
    pos_idx = (tr_t["labels"] == 1).nonzero(as_tuple=True)[0]
    if use_contrastive and len(pos_idx) >= 2:
        pos_ds = TensorDataset(
            tr_t["char_a"][pos_idx], tr_t["char_b"][pos_idx],
            tr_t["bio_a"][pos_idx], tr_t["bio_b"][pos_idx],
            tr_t["loc_a"][pos_idx], tr_t["loc_b"][pos_idx],
            tr_t["clip_a"][pos_idx], tr_t["clip_b"][pos_idx],
        )
        pos_loader = DataLoader(pos_ds, batch_size=min(batch_size, len(pos_idx)), shuffle=True)
        model.train()
        tau = 0.1
        for _ in range(epochs_contrastive):
            for ca, cb, ba, bb, la, lb, xa, xb in pos_loader:
                if ca.size(0) < 2:
                    continue
                ca, cb = ca.to(device), cb.to(device)
                ba, bb = ba.to(device), bb.to(device)
                la, lb = la.to(device), lb.to(device)
                xa, xb = xa.to(device), xb.to(device)
                ea = model.encode_profile(ca, ba, la, xa)
                eb = model.encode_profile(cb, bb, lb, xb)
                sim_mat = (ea @ eb.T) / tau
                targets = torch.arange(ea.size(0), device=device)
                loss = 0.5 * (F.cross_entropy(sim_mat, targets) + F.cross_entropy(sim_mat.T, targets))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # Stage 2: Pair-classification head
    cls_ds = TensorDataset(
        tr_t["char_a"], tr_t["char_b"], tr_t["bio_a"], tr_t["bio_b"],
        tr_t["loc_a"], tr_t["loc_b"], tr_t["clip_a"], tr_t["clip_b"],
        tr_t["link_feats"], tr_t["hand_feats"], tr_t["phash_sim"], tr_t["labels"],
    )
    cls_loader = DataLoader(cls_ds, batch_size=min(batch_size, len(filtered_train)), shuffle=True)
    pos_c = float((tr_t["labels"] == 1).sum().item())
    neg_c = float((tr_t["labels"] == 0).sum().item())
    pos_weight = torch.tensor([neg_c / max(1.0, pos_c)], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float("inf")
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for _ in range(epochs_cls):
        model.train()
        for ca, cb, ba, bb, la, lb, xa, xb, lf, hf, ps, y in cls_loader:
            ca, cb = ca.to(device), cb.to(device)
            ba, bb = ba.to(device), bb.to(device)
            la, lb = la.to(device), lb.to(device)
            xa, xb = xa.to(device), xb.to(device)
            lf, hf, ps, y = lf.to(device), hf.to(device), ps.to(device), y.to(device)
            logits, _, _ = model(ca, cb, ba, bb, la, lb, xa, xb, lf, hf, ps)
            loss = bce(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            v_logits, _, _ = model(
                va_t["char_a"].to(device), va_t["char_b"].to(device),
                va_t["bio_a"].to(device), va_t["bio_b"].to(device),
                va_t["loc_a"].to(device), va_t["loc_b"].to(device),
                va_t["clip_a"].to(device), va_t["clip_b"].to(device),
                va_t["link_feats"].to(device), va_t["hand_feats"].to(device),
                va_t["phash_sim"].to(device),
            )
            v_loss = float(F.binary_cross_entropy_with_logits(v_logits, va_t["labels"].to(device)).item())
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # Calibration on val split
    with torch.no_grad():
        val_logits_t, _, _ = model(
            va_t["char_a"].to(device), va_t["char_b"].to(device),
            va_t["bio_a"].to(device), va_t["bio_b"].to(device),
            va_t["loc_a"].to(device), va_t["loc_b"].to(device),
            va_t["clip_a"].to(device), va_t["clip_b"].to(device),
            va_t["link_feats"].to(device), va_t["hand_feats"].to(device),
            va_t["phash_sim"].to(device),
        )
        val_logits = val_logits_t.cpu().numpy()
    val_y = va_t["labels"].numpy().astype(int)
    raw_val_probs = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -30.0, 30.0)))
    ece_before = compute_ece(val_y, raw_val_probs)

    temperature = fit_temperature(val_logits, val_y)
    cal_val_probs = 1.0 / (1.0 + np.exp(-np.clip(val_logits / temperature, -30.0, 30.0)))
    ece_after = compute_ece(val_y, cal_val_probs)

    cal_report = {
        "temperature": temperature,
        "val_ece_before": ece_before,
        "val_ece_after": ece_after,
        "reliability_before": reliability_bins(val_y, raw_val_probs),
        "reliability_after": reliability_bins(val_y, cal_val_probs),
    }
    return model, temperature, cal_report


class NeuralResolver(BaseResolver):
    """LearnedResolver-compatible wrapper around MultimodalIdentityNet (--resolver neural)."""

    def __init__(
        self,
        model: MultimodalIdentityNet | None = None,
        temperature: float = 1.0,
        bio_dim: int = 64,
        clip_dim: int = 64,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.bio_dim = bio_dim
        self.clip_dim = clip_dim
        self.is_fitted = model is not None

    @torch.no_grad()
    def predict_proba_rows(self, rows: list[dict[str, Any]]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("NeuralResolver must be fitted or loaded before scoring.")
        device = next(self.model.parameters()).device
        t = rows_to_tensors(rows, bio_dim=self.bio_dim, clip_dim=self.clip_dim)
        self.model.eval()
        logits, _, _ = self.model(
            t["char_a"].to(device), t["char_b"].to(device),
            t["bio_a"].to(device), t["bio_b"].to(device),
            t["loc_a"].to(device), t["loc_b"].to(device),
            t["clip_a"].to(device), t["clip_b"].to(device),
            t["link_feats"].to(device), t["hand_feats"].to(device),
            t["phash_sim"].to(device),
        )
        scaled = logits.cpu().numpy() / max(0.05, self.temperature)
        return 1.0 / (1.0 + np.exp(-np.clip(scaled, -30.0, 30.0)))

    def score(self, target: TargetContext | Entity, hit: Hit) -> float:
        if hit.is_trusted:
            return 1.0
        if self.model is None:
            raise RuntimeError("NeuralResolver must be fitted or loaded before calling score().")

        ctx = TargetContext.from_entity(target) if isinstance(target, Entity) else target
        hf = extract_features(ctx, hit)
        bio_a_txt = " ".join(ctx.texts)
        bio_b_txt = str(hit.metadata.get("bio") or hit.raw_data.get("description") or "")
        row = {
            "username_a": ctx.usernames[0] if ctx.usernames else (ctx.names[0] if ctx.names else ""),
            "username_b": str(hit.metadata.get("username") or hit.discovered_value),
            "display_name_a": ctx.names[0] if ctx.names else "",
            "display_name_b": str(hit.metadata.get("name") or ""),
            "location_a": ctx.locations[0] if ctx.locations else "",
            "location_b": str(hit.metadata.get("location") or ""),
            "bio_emb_a": compute_bio_embedding(bio_a_txt, dim=self.bio_dim),
            "bio_emb_b": compute_bio_embedding(bio_b_txt, dim=self.bio_dim),
            "links_jaccard": float(hf.get("domain_overlap", 0.0)),
            "shared_domain_count": int(bool(hf.get("domain_overlap", 0.0))),
            "link_count_a": len(ctx.domains),
            "link_count_b": len(hit.metadata.get("linked_urls") or []),
            "phash_sim": float(hf.get("avatar_sim", 0.0)),
            "hand_features": hf,
        }
        return round(float(self.predict_proba_rows([row])[0]), 4)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save unfitted NeuralResolver.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
                "temperature": self.temperature,
                "bio_dim": self.model.bio_dim,
                "clip_dim": self.model.clip_dim,
                "use_username": self.model.use_username,
                "use_bio_loc": self.model.use_bio_loc,
                "use_photo": self.model.use_photo,
                "use_links": self.model.use_links,
            },
            target,
        )

    @classmethod
    def load(cls, path: str | Path) -> NeuralResolver:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = MultimodalIdentityNet(
            bio_dim=int(ckpt["bio_dim"]),
            clip_dim=int(ckpt["clip_dim"]),
            use_username=bool(ckpt.get("use_username", True)),
            use_bio_loc=bool(ckpt.get("use_bio_loc", True)),
            use_photo=bool(ckpt.get("use_photo", True)),
            use_links=bool(ckpt.get("use_links", True)),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return cls(
            model=model,
            temperature=float(ckpt.get("temperature", 1.0)),
            bio_dim=int(ckpt["bio_dim"]),
            clip_dim=int(ckpt["clip_dim"]),
        )
