#!/usr/bin/env python3
"""Reproducible CLI to crawl and export the OSINT-Link weak-supervision dataset."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from osint_agent.ml.datasets.builder import crawl_self_declared_profiles, load_config, split_and_export_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OSINT-Link dataset from self-declared public links.")
    parser.add_argument("--config", default="configs/osint_link.yaml", help="Path to YAML config file.")
    parser.add_argument("--max-seed-users", type=int, default=None, help="Optional override for max GitHub seed users.")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)
    if args.max_seed_users is not None:
        cfg.setdefault("crawl", {})["max_seed_users"] = args.max_seed_users

    async def _run():
        profiles, pos_pairs = await crawl_self_declared_profiles(cfg)
        return split_and_export_dataset(profiles, pos_pairs, cfg)

    manifest = asyncio.run(_run())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
