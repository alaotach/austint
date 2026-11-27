#!/usr/bin/env python3
"""Diagnostic script: saves raw HTTP responses for data/ground_truth/self.json
and reports extracted fields vs unextracted signals present in raw responses.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osint_agent.collectors.enrich import can_fetch_robots, is_private_or_loopback_host
from osint_agent.collectors.profile_analyzer import analyze_profile_hit
from osint_agent.collectors.tools import DEFAULT_USER_AGENT
from osint_agent.core.schema import EntityType, Hit
from scripts.recall_check import load_ground_truth


SIGNAL_PATTERNS = {
    "og_tags": re.compile(r'property=["\']og:(?:title|description|image)["\']', re.I),
    "json_ld": re.compile(r'type=["\']application/ld\+json["\']', re.I),
    "next_data": re.compile(r'id=["\']__NEXT_DATA__["\']', re.I),
    "nuxt_data": re.compile(r'window\.__NUXT__|id=["\']__NUXT_DATA__["\']', re.I),
    "initial_state": re.compile(r'window\.__INITIAL_STATE__|__APOLLO_STATE__', re.I),
    "followers_token": re.compile(r'"followers(?:_count|Count)?"|Followers', re.I),
    "created_token": re.compile(r'"created_at"|"dateCreated"|"joined"', re.I),
    "location_token": re.compile(r'"location"|"addressLocality"', re.I),
}


REAL_PROFILE_FIELDS = {
    "name",
    "bio",
    "avatar_url",
    "banner_url",
    "location",
    "country",
    "joined",
    "activity_dates",
    "followers",
    "following",
    "post_count",
    "total_views",
    "stories",
    "repos",
    "packages",
    "linked_urls",
    "verified",
    "orgs",
    "languages",
    "ratings",
    "company",
}


def detect_raw_signals(body: str) -> list[str]:
    return [name for name, pat in SIGNAL_PATTERNS.items() if pat.search(body or "")]


async def diagnose_urls(urls: list[str], raw_dir: Path, client: httpx.AsyncClient | None = None) -> list[dict[str, Any]]:
    owns = client is None
    if owns:
        client = httpx.AsyncClient(timeout=10.0)
    rows: list[dict[str, Any]] = []
    try:
        for url in urls:
            parsed = urlparse(url)
            host = (parsed.hostname or "").lower()
            platform = host.replace("www.", "").split(".")[0] or "unknown"
            if not host or is_private_or_loopback_host(host):
                rows.append({"platform": platform, "url": url, "status": "blocked_ssrf", "extracted": [], "raw_signals": []})
                continue

            allowed = await can_fetch_robots(client, parsed)
            if not allowed:
                is_lw = any(d in host for d in ("pinterest.", "instagram.", "x.com", "twitter.", "facebook.", "linkedin."))
                rows.append(
                    {
                        "platform": platform,
                        "url": url,
                        "status": "blocked_robots_txt",
                        "extracted_fields": [],
                        "raw_signals": [],
                        "limited_public_data": is_lw,
                        "status_note": "exists, limited public data" if is_lw else "blocked_robots_txt",
                    }
                )
                continue

            status_code = 0
            body = ""
            headers_dict: dict[str, str] = {}
            try:
                resp = await client.get(url, headers={"User-Agent": DEFAULT_USER_AGENT}, follow_redirects=True, timeout=10.0)
                status_code = resp.status_code
                body = resp.text[:300_000]
                headers_dict = dict(resp.headers)
            except Exception as exc:
                rows.append({"platform": platform, "url": url, "status": f"fetch_error:{exc}", "extracted": [], "raw_signals": []})
                continue

            pdir = raw_dir / platform
            pdir.mkdir(parents=True, exist_ok=True)
            (pdir / "response.json").write_text(
                json.dumps({"url": url, "status": status_code, "headers": headers_dict, "body_preview": body[:50_000]}, indent=2),
                encoding="utf-8",
            )

            hit = Hit(
                target_entity_id="seed:self",
                discovered_type=EntityType.PROFILE,
                discovered_value=url,
                relation="ground_truth",
                source_tool="extraction_gap",
                metadata={"url": url},
            )
            await analyze_profile_hit(hit, client)
            extracted = sorted(
                k
                for k, v in hit.metadata.items()
                if k in REAL_PROFILE_FIELDS and v not in (None, "", [], {})
            )
            raw_signals = detect_raw_signals(body)
            rows.append(
                {
                    "platform": platform,
                    "url": url,
                    "http_status": status_code,
                    "extracted_fields": extracted,
                    "raw_signals_detected": raw_signals,
                    "limited_public_data": bool(hit.metadata.get("limited_public_data")),
                }
            )
    finally:
        if owns and client is not None:
            await client.aclose()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose extraction gaps on data/ground_truth/self.json")
    parser.add_argument("--ground-truth", type=str, default="data/ground_truth/self.json")
    parser.add_argument("--raw-dir", type=str, default="tests/fixtures/raw")
    args = parser.parse_args()

    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print(json.dumps({"status": "missing_ground_truth", "expected_path": str(gt_path)}, indent=2))
        return 0

    _, urls, login_walled = load_ground_truth(gt_path)
    rows = asyncio.run(diagnose_urls(urls + login_walled, Path(args.raw_dir)))
    print(json.dumps({"count": len(rows), "diagnostics": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
