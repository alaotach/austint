#!/usr/bin/env python3
"""Manual smoke test script executing real sherlock and maigret CLI binaries.

Saves captured raw stdout and raw JSON to tests/fixtures/ and prints parsed hits.
This script is manual and never part of automated pytest runs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from osint_agent.tools import parse_maigret_json, parse_sherlock_output


def run_smoke(username: str, fixture_dir: Path) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    sherlock_fixture = fixture_dir / "sherlock_raw.txt"
    maigret_fixture = fixture_dir / "maigret_raw.json"

    sherlock_bin = shutil.which("sherlock") or shutil.which("sherlock", path=os.path.join(sys.prefix, "bin"))
    maigret_bin = shutil.which("maigret") or shutil.which("maigret", path=os.path.join(sys.prefix, "bin"))

    print(f"\n========================================================")
    print(f"  Intelligent OSINT: Real Tools Smoke Test")
    print(f"  Target Username: {username}")
    print(f"========================================================")
    print(f"[*] Sherlock binary: {sherlock_bin}")
    print(f"[*] Maigret binary:  {maigret_bin}")

    if not sherlock_bin or not maigret_bin:
        print("[!] Error: Both sherlock and maigret must be installed in the venv.")
        sys.exit(1)

    # 1. Run Sherlock
    print(f"\n[*] Executing Sherlock on @{username} (checking top sites)...")
    with tempfile.TemporaryDirectory() as tmp_s:
        # Check GitHub and Reddit for fast smoke testing
        cmd_s = [
            sherlock_bin,
            username,
            "--site", "GitHub",
            "--site", "Reddit",
            "--print-found",
            "--no-color",
            "--timeout", "10",
        ]
        res_s = subprocess.run(cmd_s, capture_output=True, text=True, cwd=tmp_s)
        sherlock_stdout = res_s.stdout
        sherlock_fixture.write_text(sherlock_stdout, encoding="utf-8")
        print(f"[+] Saved raw Sherlock stdout to: {sherlock_fixture}")

    # Parse Sherlock
    s_hits = parse_sherlock_output(sherlock_stdout, username, f"username:{username}")
    print(f"[*] Sherlock Parser extracted {len(s_hits)} hit(s):")
    for h in s_hits:
        print(f"    - [{h.metadata.get('site')}] {h.discovered_value} (trusted={h.is_trusted})")

    # 2. Run Maigret
    print(f"\n[*] Executing Maigret on @{username} (checking top sites)...")
    with tempfile.TemporaryDirectory() as tmp_m:
        cmd_m = [
            maigret_bin,
            username,
            "--site", "GitHub",
            "--site", "Reddit",
            "--json", "simple",
            "--timeout", "10",
            "--no-progressbar",
        ]
        res_m = subprocess.run(cmd_m, capture_output=True, text=True, cwd=tmp_m)

        found_json_data = {}
        found_json_path = None
        for root, _, files in os.walk(tmp_m):
            for f in files:
                if f.endswith(".json"):
                    found_json_path = os.path.join(root, f)
                    break
            if found_json_path:
                break

        if found_json_path and os.path.exists(found_json_path):
            with open(found_json_path, "r", encoding="utf-8") as f:
                found_json_data = json.load(f)
        else:
            try:
                found_json_data = json.loads(res_m.stdout)
            except Exception:
                found_json_data = {"error": "No JSON produced", "stdout": res_m.stdout}

        maigret_fixture.write_text(json.dumps(found_json_data, indent=2), encoding="utf-8")
        print(f"[+] Saved raw Maigret JSON to: {maigret_fixture}")

    # Parse Maigret
    m_hits = parse_maigret_json(found_json_data, username, f"username:{username}")
    print(f"[*] Maigret Parser extracted {len(m_hits)} hit(s):")
    for h in m_hits:
        print(f"    - [{h.metadata.get('site')}] {h.discovered_value}")
        print(f"        Name:     {h.metadata.get('name')}")
        print(f"        Location: {h.metadata.get('location')}")
        print(f"        Bio:      {h.metadata.get('bio')}")

    print("\n[+] Smoke test completed successfully.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test real sherlock and maigret tools")
    parser.add_argument("username", nargs="?", default="torvalds", help="Target username to inspect (default: torvalds)")
    args = parser.parse_args()

    fixture_dir = PROJECT_ROOT / "tests" / "fixtures"
    run_smoke(args.username, fixture_dir)


if __name__ == "__main__":
    main()
