#!/usr/bin/env python3
"""Playwright UI verification harness for Intelligent OSINT dashboard."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from playwright.sync_api import sync_playwright


def _wait_for_server(url: str, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=1.5)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"Server did not become ready at {url}")


def run_audit(
    *,
    out_dir: Path,
    views: list[str],
    widths: list[int],
    seed_demo: bool = False,
    open_inspector: bool = False,
    port: int = 8765,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "osint_agent.api.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    base_url = f"http://127.0.0.1:{port}"
    report: dict[str, Any] = {"checks": [], "console_errors": [], "page_errors": [], "failed_requests": []}

    try:
        _wait_for_server(f"{base_url}/api/health")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            for w in widths:
                h = 812 if w == 375 else (1024 if w == 768 else 900)
                context = browser.new_context(viewport={"width": w, "height": h})
                page = context.new_page()

                def _on_console(msg: Any) -> None:
                    if msg.type == "error":
                        report["console_errors"].append({"width": w, "text": msg.text})

                def _on_pageerror(err: Any) -> None:
                    report["page_errors"].append({"width": w, "error": str(err)})

                def _on_reqfail(req: Any) -> None:
                    report["failed_requests"].append({"width": w, "url": req.url, "failure": req.failure})

                page.on("console", _on_console)
                page.on("pageerror", _on_pageerror)
                page.on("requestfailed", _on_reqfail)

                page.goto(base_url, wait_until="domcontentloaded", timeout=20000)
                page.wait_for_timeout(400)

                if seed_demo:
                    page.evaluate(
                        """() => {
                            if (typeof window.loadRecordedDemoRun === 'function') {
                                window.loadRecordedDemoRun();
                            }
                        }"""
                    )
                    page.wait_for_timeout(300)

                for v in views:
                    btn = page.locator(f'.nav-item[data-view="{v}"]')
                    if btn.count() > 0:
                        btn.first.click()
                        page.wait_for_timeout(250)

                    overflow = page.evaluate(
                        "() => ({scrollW: document.documentElement.scrollWidth, innerW: window.innerWidth})"
                    )
                    shot_path = out_dir / f"{v}_{w}.png"
                    page.screenshot(path=str(shot_path), full_page=False)
                    report["checks"].append(
                        {
                            "view": v,
                            "width": w,
                            "horizontal_overflow": overflow["scrollW"] > overflow["innerW"],
                            "scrollWidth": overflow["scrollW"],
                            "innerWidth": overflow["innerW"],
                            "screenshot": str(shot_path),
                        }
                    )

                if open_inspector:
                    rbtn = page.locator('.nav-item[data-view="results"]')
                    if rbtn.count() > 0:
                        rbtn.first.click()
                        page.wait_for_timeout(200)
                    card = page.locator(".result-card").first
                    if card.count() > 0:
                        card.click()
                        page.wait_for_timeout(250)
                        for itab in ("overview", "evidence", "ml", "related", "raw"):
                            tbtn = page.locator(f'[data-itab="{itab}"]')
                            if tbtn.count() > 0:
                                tbtn.first.click()
                                page.wait_for_timeout(150)
                        # Return to ML tab for screenshot
                        ml_btn = page.locator('[data-itab="ml"]')
                        if ml_btn.count() > 0:
                            ml_btn.first.click()
                            page.wait_for_timeout(150)
                        ishot = out_dir / f"inspector_{w}.png"
                        page.screenshot(path=str(ishot), full_page=False)
                        report["checks"].append({"view": "inspector", "width": w, "screenshot": str(ishot)})

                context.close()
            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Playwright UI audit")
    parser.add_argument("--out-dir", type=str, default="docs/screenshots")
    parser.add_argument("--views", type=str, default="launch,scan,results,graph,timeline,audit,report")
    parser.add_argument("--widths", type=str, default="375,768,1440")
    parser.add_argument("--seed-demo", action="store_true")
    parser.add_argument("--open-inspector", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    res = run_audit(
        out_dir=Path(args.out_dir),
        views=[v.strip() for v in args.views.split(",") if v.strip()],
        widths=[int(w.strip()) for w in args.widths.split(",") if w.strip()],
        seed_demo=args.seed_demo,
        open_inspector=args.open_inspector,
        port=args.port,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
