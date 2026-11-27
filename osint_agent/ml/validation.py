"""Phase G: Validation & stress-testing utilities (no new model features).

Implements:
  1. GNN leakage audit (pair-forming edge removal + split isolation verification)
  2. Person-clustered paired bootstrap (1000 resamples, 95% CIs, significance)
  3. Shortcut baselines (platform-pair-only, username-sim-only, bio-length/has-bio-only)
  4. Per-negative-type breakdown (highlighting hard negatives)
  5. Val-tuned chain-confidence baseline evaluation & decay diagnostics
  6. Two-variant zero-shot LLM judge with disk cache, backoff, and resume
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from osint_agent.ml.baselines import ZeroShotLLMJudge, compute_metrics


HARD_NEGATIVE_TYPES = {
    "similar_username_diff_person",
    "same_name_diff_org_city",
    "same_display_name_diff_links",
}
ALL_NEGATIVE_TYPES = (
    "diff_person_fallback",
    "similar_username_diff_person",
    "same_name_diff_org_city",
    "same_display_name_diff_links",
)


def tune_threshold_on_val(
    y_val: np.ndarray,
    probs_val: np.ndarray,
    *,
    grid_points: int = 90,
) -> float:
    """Tune classification threshold on val split to maximize F1."""
    if len(y_val) == 0 or len(np.unique(y_val)) < 2:
        return 0.5
    best_t = 0.5
    best_f1 = -1.0
    for cand in np.linspace(0.01, 0.90, grid_points):
        f1 = compute_metrics(y_val, probs_val, threshold=float(cand))["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(cand)
    return round(best_t, 4)


def paired_person_bootstrap(
    rows: list[dict[str, Any]],
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    *,
    thresh_a: float = 0.5,
    thresh_b: float = 0.5,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Person-clustered paired bootstrap comparing model A minus model B.

    Resamples `person_id_a` clusters with replacement for `n_bootstrap`
    iterations using the exact same resampled pairs for both models.
    """
    y_true = np.array([int(r["label"]) for r in rows], dtype=int)
    p_a = np.asarray(probs_a, dtype=np.float64)
    p_b = np.asarray(probs_b, dtype=np.float64)

    cluster_to_indices: dict[str, list[int]] = {}
    for idx, r in enumerate(rows):
        pid = str(r.get("person_id_a") or r.get("person_id") or f"row_{idx}")
        cluster_to_indices.setdefault(pid, []).append(idx)

    clusters = list(cluster_to_indices.values())
    n_clusters = len(clusters)
    if n_clusters == 0:
        return {}

    rng = np.random.default_rng(seed)
    diffs_pr: list[float] = []
    diffs_f1: list[float] = []

    for _ in range(n_bootstrap):
        sampled_cluster_ids = rng.integers(0, n_clusters, size=n_clusters)
        idx_list: list[int] = []
        for cid in sampled_cluster_ids:
            idx_list.extend(clusters[int(cid)])
        idx_arr = np.asarray(idx_list, dtype=int)
        y_sub = y_true[idx_arr]
        if len(np.unique(y_sub)) < 2:
            continue
        m_a = compute_metrics(y_sub, p_a[idx_arr], threshold=thresh_a)
        m_b = compute_metrics(y_sub, p_b[idx_arr], threshold=thresh_b)
        diffs_pr.append(m_a["pr_auc"] - m_b["pr_auc"])
        diffs_f1.append(m_a["f1"] - m_b["f1"])

    def _summarize(vals: list[float]) -> dict[str, Any]:
        if not vals:
            return {
                "mean_diff": 0.0,
                "ci_low": 0.0,
                "ci_high": 0.0,
                "significant": False,
                "status": "not significant",
            }
        arr = np.asarray(vals, dtype=np.float64)
        lo = float(np.percentile(arr, 2.5))
        hi = float(np.percentile(arr, 97.5))
        sig = bool(lo > 0.0 or hi < 0.0)
        return {
            "mean_diff": round(float(np.mean(arr)), 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "significant": sig,
            "status": "significant" if sig else "not significant",
        }

    return {
        "n_clusters": n_clusters,
        "n_bootstrap_valid": len(diffs_pr),
        "pr_auc_diff": _summarize(diffs_pr),
        "f1_diff": _summarize(diffs_f1),
    }


def evaluate_shortcut_baselines(
    train_rows: list[dict[str, Any]],
    eval_splits: dict[str, list[dict[str, Any]]],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Train and evaluate 3 single-shortcut baselines to probe dataset bias."""
    y_train = np.array([int(r["label"]) for r in train_rows], dtype=int)
    if len(train_rows) < 4 or len(np.unique(y_train)) < 2:
        return {"status": "skipped_insufficient_train_data"}

    # 1. Platform-pair-only
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    plat_train = [[str(r.get("platform_a") or ""), str(r.get("platform_b") or "")] for r in train_rows]
    X_plat_tr = enc.fit_transform(plat_train)
    clf_plat = LogisticRegression(random_state=seed, max_iter=500).fit(X_plat_tr, y_train)

    # 2. Username-similarity-only
    def _uname_feat(rows: list[dict[str, Any]]) -> np.ndarray:
        return np.array(
            [[float((r.get("hand_features") or {}).get("username_sim", 0.0))] for r in rows],
            dtype=np.float64,
        )

    clf_uname = LogisticRegression(random_state=seed, max_iter=500).fit(_uname_feat(train_rows), y_train)

    # 3. Bio-length / has-bio-only
    def _bio_feat(rows: list[dict[str, Any]]) -> np.ndarray:
        out = []
        for r in rows:
            la = float(r.get("bio_len_a", 0))
            lb = float(r.get("bio_len_b", 0))
            out.append([la / 200.0, lb / 200.0, 1.0 if la > 0 else 0.0, 1.0 if lb > 0 else 0.0])
        return np.asarray(out, dtype=np.float64)

    clf_bio = LogisticRegression(random_state=seed, max_iter=500).fit(_bio_feat(train_rows), y_train)

    # 4. Exact display-name match only
    def _exact_dn_feat(rows: list[dict[str, Any]]) -> np.ndarray:
        out = []
        for r in rows:
            da = str(r.get("display_name_a") or "").strip().lower()
            db = str(r.get("display_name_b") or "").strip().lower()
            out.append([1.0 if (da and db and da == db) else 0.0])
        return np.asarray(out, dtype=np.float64)

    clf_exact_dn = LogisticRegression(random_state=seed, max_iter=500).fit(_exact_dn_feat(train_rows), y_train)

    # 5. Exact username match only
    def _exact_un_feat(rows: list[dict[str, Any]]) -> np.ndarray:
        out = []
        for r in rows:
            ua = str(r.get("username_a") or "").strip().lower()
            ub = str(r.get("username_b") or "").strip().lower()
            out.append([1.0 if (ua and ub and ua == ub) else 0.0])
        return np.asarray(out, dtype=np.float64)

    clf_exact_un = LogisticRegression(random_state=seed, max_iter=500).fit(_exact_un_feat(train_rows), y_train)

    # 6. Has-bio only
    def _has_bio_feat(rows: list[dict[str, Any]]) -> np.ndarray:
        out = []
        for r in rows:
            ha = 1.0 if float(r.get("bio_len_a", 0)) > 0 else 0.0
            hb = 1.0 if float(r.get("bio_len_b", 0)) > 0 else 0.0
            out.append([ha, hb, 1.0 if (ha and hb) else 0.0])
        return np.asarray(out, dtype=np.float64)

    clf_has_bio = LogisticRegression(random_state=seed, max_iter=500).fit(_has_bio_feat(train_rows), y_train)

    results: dict[str, Any] = {"splits": {}, "dataset_bias_flags": []}
    for sp_name, rows in eval_splits.items():
        if not rows:
            continue
        y_sp = np.array([int(r["label"]) for r in rows], dtype=int)
        pos_rate = float(np.mean(y_sp))
        X_plat_sp = enc.transform([[str(r.get("platform_a") or ""), str(r.get("platform_b") or "")] for r in rows])

        preds = {
            "platform_pair_only": clf_plat.predict_proba(X_plat_sp)[:, 1],
            "username_sim_only": clf_uname.predict_proba(_uname_feat(rows))[:, 1],
            "bio_len_has_bio_only": clf_bio.predict_proba(_bio_feat(rows))[:, 1],
            "exact_display_name_only": clf_exact_dn.predict_proba(_exact_dn_feat(rows))[:, 1],
            "exact_username_only": clf_exact_un.predict_proba(_exact_un_feat(rows))[:, 1],
            "has_bio_only": clf_has_bio.predict_proba(_has_bio_feat(rows))[:, 1],
        }
        sp_out: dict[str, Any] = {"pos_rate": round(pos_rate, 4), "models": {}}
        for mname, probs in preds.items():
            m = compute_metrics(y_sp, probs)
            sp_out["models"][mname] = m
            if mname == "platform_pair_only" and m["pr_auc"] >= pos_rate + 0.15:
                results["dataset_bias_flags"].append(
                    f"platform_pair_only achieves PR-AUC={m['pr_auc']:.4f} vs random baseline={pos_rate:.4f} on {sp_name}"
                )
        results["splits"][sp_name] = sp_out

    results["status"] = "ok"
    return results


def evaluate_by_negative_type(
    rows: list[dict[str, Any]],
    model_probs: dict[str, np.ndarray],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute per-model Precision, Recall, F1, PR-AUC split by negative strategy."""
    thresholds = thresholds or {}
    pos_idx = [i for i, r in enumerate(rows) if int(r["label"]) == 1]
    out: dict[str, Any] = {}

    for neg_t in ALL_NEGATIVE_TYPES:
        neg_idx = [
            i for i, r in enumerate(rows) if int(r["label"]) == 0 and str(r.get("neg_type") or "") == neg_t
        ]
        if not pos_idx or not neg_idx:
            continue
        sub_idx = np.asarray(pos_idx + neg_idx, dtype=int)
        y_sub = np.array([int(rows[i]["label"]) for i in sub_idx], dtype=int)
        models_res: dict[str, Any] = {}
        for mname, probs in model_probs.items():
            p_sub = np.asarray(probs, dtype=np.float64)[sub_idx]
            t = float(thresholds.get(mname, 0.5))
            m = compute_metrics(y_sub, p_sub, threshold=t)
            models_res[mname] = {
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "pr_auc": m["pr_auc"],
            }
        out[neg_t] = {
            "is_hard_negative": neg_t in HARD_NEGATIVE_TYPES,
            "pos_count": len(pos_idx),
            "neg_count": len(neg_idx),
            "models": models_res,
        }
    return out


def select_live_llm_provider() -> str | None:
    """Select live LLM provider (explicit g4f, or gemini -> anthropic -> openrouter)."""
    if os.getenv("LLM_PROVIDER", "").lower() == "g4f":
        return "g4f"
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    return None


def _build_judge_prompt(pair: dict[str, Any], variant: str) -> tuple[str, str]:
    if variant == "raw_no_sim":
        system = (
            "Decide if two public social profiles belong to the same person using ONLY raw profile fields "
            "(no precomputed similarity scores). Return ONLY strict JSON: "
            '{"match_probability": <float 0..1>, "rationale": "<short>"}.'
        )
        user = json.dumps(
            {
                "platform_a": pair.get("platform_a", ""),
                "platform_b": pair.get("platform_b", ""),
                "username_a": pair.get("username_a", ""),
                "username_b": pair.get("username_b", ""),
                "display_name_a": pair.get("display_name_a", ""),
                "display_name_b": pair.get("display_name_b", ""),
                "location_a": pair.get("location_a", ""),
                "location_b": pair.get("location_b", ""),
                "bio_a": pair.get("bio_a", ""),
                "bio_b": pair.get("bio_b", ""),
                "bio_len_a": pair.get("bio_len_a", 0),
                "bio_len_b": pair.get("bio_len_b", 0),
            },
            ensure_ascii=True,
        )
        return system, user
    if variant == "field_by_field":
        system = (
            "You are an OSINT identity resolution judge. Compare each field carefully: "
            "(1) username similarity and handle conventions, (2) display name match, "
            "(3) location consistency, (4) bio topic & semantic alignment, "
            "(5) avatar similarity (pHash/CLIP). Return ONLY strict JSON: "
            '{"match_probability": <float 0..1>, "rationale": "<short string>"}.'
        )
    else:
        system = (
            "Decide if two public social profiles belong to the same person. "
            'Return ONLY strict JSON: {"match_probability": <float 0..1>, "rationale": "<short>"}.'
        )
    user = json.dumps(
        {
            "platform_a": pair.get("platform_a", ""),
            "platform_b": pair.get("platform_b", ""),
            "username_a": pair.get("username_a", ""),
            "username_b": pair.get("username_b", ""),
            "display_name_a": pair.get("display_name_a", ""),
            "display_name_b": pair.get("display_name_b", ""),
            "location_a": pair.get("location_a", ""),
            "location_b": pair.get("location_b", ""),
            "bio_emb_sim": pair.get("bio_emb_sim", 0.0),
            "avatar_sim": (pair.get("hand_features") or {}).get("avatar_sim", 0.0),
        },
        ensure_ascii=True,
    )
    return system, user


def recalibrate_isotonic_on_val(
    y_val: np.ndarray,
    probs_val: np.ndarray,
    probs_list: list[np.ndarray],
) -> list[np.ndarray]:
    """Fit IsotonicRegression on val probabilities and transform target probability arrays."""
    from sklearn.isotonic import IsotonicRegression

    if len(y_val) < 4 or len(np.unique(y_val)) < 2:
        return [np.asarray(p, dtype=np.float64) for p in probs_list]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(np.asarray(probs_val, dtype=np.float64), np.asarray(y_val, dtype=int))
    return [np.asarray(iso.transform(np.asarray(p, dtype=np.float64)), dtype=np.float64) for p in probs_list]


def run_llm_judge_two_variants(
    rows: list[dict[str, Any]],
    *,
    cache_dir: Path,
    client: Any | None = None,
    provider_name: str | None = None,
    max_calls_per_variant: int = 400,
    max_consecutive_rate_limits: int = 3,
    base_backoff_sec: float = 1.0,
    concurrency: int = 1,
) -> dict[str, Any]:
    """Evaluate 2 prompt variants (`minimal`, `field_by_field`) with disk cache + backoff."""
    import concurrent.futures
    import threading

    if client is None:
        provider_name = provider_name or select_live_llm_provider()
        if not provider_name:
            return {
                "status": "excluded_no_api_key",
                "reason": "No GEMINI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY set.",
                "variants": {},
            }
        from osint_agent.llm.client import get_llm_client

        client = get_llm_client(provider=provider_name, call_budget=max_calls_per_variant * 2)

    cache_dir.mkdir(parents=True, exist_ok=True)
    variants_out: dict[str, Any] = {}
    y_true = np.array([int(r["label"]) for r in rows], dtype=int)
    lock = threading.Lock()

    def _eval_one(r: dict[str, Any], variant: str) -> dict[str, Any] | None:
        sys_p, usr_p = _build_judge_prompt(r, variant)
        t0 = time.perf_counter()
        ok = False
        prob = 0.5
        tokens = 0
        cost = 0.0
        for attempt in range(3):
            try:
                raw = client.complete_json(sys_p, usr_p)
                if isinstance(raw, dict) and "match_probability" in raw and not raw.get("_rate_limited"):
                    prob = float(np.clip(float(raw["match_probability"]), 0.0, 1.0))
                    tokens = int(raw.get("_token_usage") or getattr(client, "last_tokens", 0) or (len(usr_p) // 4 + 32))
                    cost = 0.0 if (provider_name == "g4f") else round(tokens * 0.0000005, 6)
                    ok = True
                    break
            except Exception:
                pass
            time.sleep(base_backoff_sec * (2**attempt))
        if not ok:
            return None
        lat_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "pair_id": str(r.get("pair_id") or ""),
            "variant": variant,
            "match_probability": prob,
            "tokens": tokens,
            "cost_usd": cost,
            "latency_ms": round(lat_ms, 3),
            "live_ok": True,
        }

    for variant in ("minimal", "field_by_field"):
        cache_path = cache_dir / f"llm_judge_{variant}.jsonl"
        cached: dict[str, dict[str, Any]] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("live_ok"):
                    cached[str(rec["pair_id"])] = rec

        consec_rl = 0
        rate_limited = False
        pending = [r for r in rows if str(r.get("pair_id") or "") not in cached]

        if concurrency > 1 and pending:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                futs = {pool.submit(_eval_one, r, variant): r for r in pending}
                for fut in concurrent.futures.as_completed(futs):
                    rec = fut.result()
                    if rec is not None:
                        consec_rl = 0
                        with lock:
                            cached[rec["pair_id"]] = rec
                            with open(cache_path, "a", encoding="utf-8") as cf:
                                cf.write(json.dumps(rec) + "\n")
                    else:
                        consec_rl += 1
        else:
            for r in pending:
                rec = _eval_one(r, variant)
                if rec is not None:
                    consec_rl = 0
                    cached[rec["pair_id"]] = rec
                    with open(cache_path, "a", encoding="utf-8") as cf:
                        cf.write(json.dumps(rec) + "\n")
                else:
                    consec_rl += 1
                    if consec_rl >= max_consecutive_rate_limits:
                        rate_limited = True
                        break

        probs = [float(cached[str(r.get("pair_id") or "")]["match_probability"]) for r in rows if str(r.get("pair_id") or "") in cached]
        latencies = [float(cached[str(r.get("pair_id") or "")].get("latency_ms", 0.0)) for r in rows if str(r.get("pair_id") or "") in cached]
        total_tokens = sum(int(cached[str(r.get("pair_id") or "")].get("tokens", 0)) for r in rows if str(r.get("pair_id") or "") in cached)
        total_cost = sum(float(cached[str(r.get("pair_id") or "")].get("cost_usd", 0.0)) for r in rows if str(r.get("pair_id") or "") in cached)

        if rate_limited or len(probs) < len(rows):
            variants_out[variant] = {
                "status": "excluded_rate_limited",
                "provider": provider_name or type(client).__name__,
                "scored_pairs": len(probs),
                "total_pairs": len(rows),
                "reason": "Provider rate-limited or failed before completing split; excluded from comparisons.",
            }
        else:
            p_arr = np.asarray(probs, dtype=np.float64)
            m = compute_metrics(y_true, p_arr)
            variants_out[variant] = {
                "status": "ok",
                "provider": provider_name or type(client).__name__,
                **m,
                "total_tokens": total_tokens,
                "cost_estimate_usd": round(total_cost, 6),
                "latency_ms_per_pair": round(float(np.mean(latencies)) if latencies else 0.0, 4),
            }

    overall_ok = any(v.get("status") == "ok" for v in variants_out.values())
    return {
        "status": "ok" if overall_ok else "excluded_rate_limited",
        "provider": provider_name or type(client).__name__,
        "variants": variants_out,
    }
