#!/usr/bin/env python3
"""Ground-truth recall & precision check against data/ground_truth/self.json (gitignored)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osint_agent.core.agent import OSINTAgent
from osint_agent.core.schema import EntityType, Scope


def _norm_url(url: str) -> str:
    s = (url or "").strip().lower().rstrip("/")
    if s.startswith("https://"):
        s = s[8:]
    elif s.startswith("http://"):
        s = s[7:]
    if s.startswith("www."):
        s = s[4:]
    return s


def load_ground_truth(path: Path) -> tuple[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        urls = [str(u) for u in data if str(u).strip()]
        handle = ""
        for u in urls:
            parts = [p for p in urlparse(u).path.split("/") if p]
            if parts:
                handle = parts[-1].lstrip("@")
                break
        return handle, urls
    return str(data.get("handle") or ""), [str(u) for u in (data.get("urls") or []) if str(u).strip()]


def evaluate_recall_against_graph(
    gt_urls: list[str],
    discovered_profile_urls: list[str],
    audit_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    gt_norm = {_norm_url(u): u for u in gt_urls}
    disc_norm = {_norm_url(u): u for u in discovered_profile_urls}

    matched = [gt_norm[k] for k in gt_norm if k in disc_norm]
    missed = [gt_norm[k] for k in gt_norm if k not in disc_norm]
    extra = [disc_norm[k] for k in disc_norm if k not in gt_norm]

    recall = round(len(matched) / len(gt_norm), 4) if gt_norm else 0.0
    precision = round(len(matched) / len(disc_norm), 4) if disc_norm else 0.0

    rejected_vals = {
        _norm_url(str((e.get("details") or {}).get("discovered_value") or "")): e.get("details")
        for e in (audit_events or [])
        if e.get("event") == "hit_evaluated" and not (e.get("details") or {}).get("accepted")
    }

    missed_details = []
    for u in missed:
        nu = _norm_url(u)
        if nu in rejected_vals:
            reason = f"discovered_but_rejected (score={rejected_vals[nu].get('resolver_score')})"
        elif any(d in nu for d in ("instagram.com", "x.com", "twitter.com", "facebook.com", "linkedin.com")):
            reason = "login_walled_or_unindexed_without_search_api"
        else:
            reason = "not_discovered_by_active_collectors"
        missed_details.append({"url": u, "reason": reason})

    return {
        "ground_truth_count": len(gt_norm),
        "discovered_profile_count": len(disc_norm),
        "matched_count": len(matched),
        "missed_count": len(missed),
        "extra_count": len(extra),
        "recall": recall,
        "precision": precision,
        "matched_urls": matched,
        "missed": missed_details,
    }


async def _run(gt_path: Path, handle_override: str, max_depth: int, max_calls: int) -> dict[str, Any]:
    handle, urls = load_ground_truth(gt_path)
    target_handle = handle_override or handle
    if not target_handle:
        raise ValueError("Could not determine handle from ground truth file; pass --handle.")

    audit_path = REPO_ROOT / "data" / "ground_truth" / "recall_audit.jsonl"
    agent = OSINTAgent(audit_path=audit_path)
    scope = Scope(
        target_kind=EntityType.USERNAME,
        target_value=target_handle,
        consent_confirmed=True,
        max_depth=max_depth,
        max_tool_calls=max_calls,
    )
    graph = await agent.run(scope)
    disc_urls = [
        str(attrs.get("value") or nid)
        for nid, attrs in graph.nodes(data=True)
        if attrs.get("entity_type") == "profile" and str(attrs.get("value") or "").startswith("http")
    ]
    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if audit_path.exists() else []
    res = evaluate_recall_against_graph(urls, disc_urls, events)
    res["handle"] = target_handle
    return res


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure recall & precision against data/ground_truth/self.json")
    parser.add_argument("--ground-truth", type=str, default="data/ground_truth/self.json")
    parser.add_argument("--handle", type=str, default="")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-calls", type=int, default=20)
    args = parser.parse_args()

    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print(
            json.dumps(
                {
                    "status": "missing_ground_truth",
                    "expected_path": str(gt_path),
                    "example_format": {"handle": "alaotach", "urls": ["https://github.com/alaotach", "https://hub.docker.com/u/alaotach"]},
                },
                indent=2,
            )
        )
        return 0

    res = asyncio.run(_run(gt_path, args.handle, args.max_depth, args.max_calls))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
