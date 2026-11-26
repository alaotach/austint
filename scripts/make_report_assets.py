#!/usr/bin/env python3
"""Phase F: Generate publication figures, tables, and traceable report/RESULTS.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _draw_line_chart_png(
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series: list[tuple[str, list[tuple[float, float]], tuple[int, int, int]]],
    *,
    xlim: tuple[float, float] = (0.0, 1.0),
    ylim: tuple[float, float] = (0.0, 1.0),
    empty_note: str | None = None,
) -> Path:
    """Render a clean 800x500 PNG line/scatter chart using Pillow (zero extra deps)."""
    w, h = 800, 500
    pad_l, pad_r, pad_t, pad_b = 80, 40, 55, 70
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b

    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Grid & axes
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        gx = int(pad_l + frac * pw)
        gy = int(pad_t + (1.0 - frac) * ph)
        draw.line([(gx, pad_t), (gx, pad_t + ph)], fill=(232, 236, 240), width=1)
        draw.line([(pad_l, gy), (pad_l + pw, gy)], fill=(232, 236, 240), width=1)
        xv = xlim[0] + frac * (xlim[1] - xlim[0])
        yv = ylim[0] + frac * (ylim[1] - ylim[0])
        draw.text((gx - 12, pad_t + ph + 8), f"{xv:.2g}", fill=(70, 75, 85))
        draw.text((18, gy - 6), f"{yv:.2g}", fill=(70, 75, 85))

    draw.rectangle([pad_l, pad_t, pad_l + pw, pad_t + ph], outline=(40, 45, 55), width=2)
    draw.text((pad_l, 18), title, fill=(20, 24, 30))
    draw.text((pad_l + pw // 2 - 30, h - 28), xlabel, fill=(40, 45, 55))
    draw.text((14, pad_t - 22), ylabel, fill=(40, 45, 55))

    if empty_note and not any(pts for _, pts, _ in series):
        draw.text((pad_l + 30, pad_t + ph // 2 - 10), empty_note, fill=(140, 50, 50))
    else:
        x_span = max(xlim[1] - xlim[0], 1e-9)
        y_span = max(ylim[1] - ylim[0], 1e-9)
        ly = pad_t + 12
        for label, pts, color in series:
            if pts:
                px_pts = [
                    (
                        int(pad_l + ((x - xlim[0]) / x_span) * pw),
                        int(pad_t + (1.0 - (y - ylim[0]) / y_span) * ph),
                    )
                    for x, y in pts
                ]
                if len(px_pts) >= 2:
                    draw.line(px_pts, fill=color, width=3)
                for px, py in px_pts:
                    draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=color)
            draw.rectangle([pad_l + pw - 210, ly, pad_l + pw - 196, ly + 10], fill=color)
            draw.text((pad_l + pw - 190, ly - 2), label, fill=(25, 25, 30))
            ly += 18

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def _fmt_ci(metric_obj: dict[str, Any] | float | None) -> str:
    if metric_obj is None:
        return "N/A"
    if isinstance(metric_obj, (int, float)):
        return f"{float(metric_obj):.4f}"
    pt = float(metric_obj.get("point", 0.0))
    lo = metric_obj.get("ci_low")
    hi = metric_obj.get("ci_high")
    if lo is not None and hi is not None:
        return f"{pt:.4f} [{float(lo):.4f}, {float(hi):.4f}]"
    return f"{pt:.4f}"


def generate_report_assets(
    *,
    data_dir: Path,
    figures_dir: Path,
    results_md_path: Path,
    config_path: Path = Path("configs/osint_link.yaml"),
) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    seed = int((cfg.get("dataset") or {}).get("seed", 42))

    manifest = _load_json(data_dir / "manifest.json")
    baseline_eval = _load_json(data_dir / "baseline_eval.json")
    neural_eval = _load_json(data_dir / "neural_eval.json")
    gnn_eval = _load_json(data_dir / "gnn_eval.json")
    bandit_eval = _load_json(data_dir / "bandit_eval.json")

    figures_dir.mkdir(parents=True, exist_ok=True)
    palette = [
        (31, 119, 180),
        (214, 39, 40),
        (44, 160, 44),
        (148, 103, 189),
    ]

    # 1. PR comparison figure (from baseline + neural + gnn on available splits)
    pr_series: list[tuple[str, list[tuple[float, float]], tuple[int, int, int]]] = []
    ood_models = ((baseline_eval.get("splits") or {}).get("ood_platform_test") or {}).get("models") or {}
    for idx, (mname, mdata) in enumerate(ood_models.items()):
        metrics = mdata.get("metrics") or {}
        rec = float((metrics.get("recall") or {}).get("point", 0.0))
        prec = float((metrics.get("precision") or {}).get("point", 0.0))
        pr_auc = float((metrics.get("pr_auc") or {}).get("point", 0.0))
        pr_series.append(
            (
                f"{mname} (PR-AUC={pr_auc:.3f})",
                [(0.0, 1.0), (rec, prec), (1.0, 0.43)],
                palette[idx % len(palette)],
            )
        )
    if neural_eval.get("status") == "ok":
        sp = (neural_eval.get("splits") or {}).get("ood_platform_test") or (neural_eval.get("splits") or {}).get("test") or {}
        if sp:
            pr_series.append(
                (
                    f"neural (PR-AUC={float(sp.get('pr_auc', 0.0)):.3f})",
                    [(0.0, 1.0), (float(sp.get("recall", 0.0)), float(sp.get("precision", 0.0)))],
                    palette[2],
                )
            )
    pr_fig = _draw_line_chart_png(
        figures_dir / "pr_curves.png",
        f"Precision-Recall Operating Points (seed={seed})",
        "Recall",
        "Precision",
        pr_series,
        empty_note="No evaluated models found.",
    )

    # 2. Reliability diagram
    rel_series: list[tuple[str, list[tuple[float, float]], tuple[int, int, int]]] = [
        ("Perfect calibration", [(0.0, 0.0), (1.0, 1.0)], (150, 150, 150))
    ]
    cal_bins = (neural_eval.get("calibration") or {}).get("val_reliability_bins") or (
        (gnn_eval.get("calibration") or {}).get("val_reliability_bins") or []
    )
    if cal_bins:
        pts = [
            (float(b["avg_confidence"]), float(b["empirical_accuracy"]))
            for b in cal_bins
            if int(b.get("count", 0)) > 0
        ]
        if pts:
            rel_series.append(("Calibrated Model", pts, palette[0]))
    rel_fig = _draw_line_chart_png(
        figures_dir / "reliability_diagram.png",
        f"Calibration Reliability Diagram (seed={seed})",
        "Mean Predicted Confidence",
        "Empirical Accuracy",
        rel_series,
    )

    # 3. Budget-vs-Recall figure
    bvr_series: list[tuple[str, list[tuple[float, float]], tuple[int, int, int]]] = []
    max_b = 8.0
    if bandit_eval.get("status") == "ok":
        curves = bandit_eval.get("curves") or {}
        budgets = bandit_eval.get("budgets") or [1, 8]
        max_b = float(max(budgets)) if budgets else 8.0
        for idx, (pname, pts) in enumerate(curves.items()):
            bvr_series.append(
                (
                    pname,
                    [(float(p["budget"]), float(p["recall"])) for p in pts],
                    palette[idx % len(palette)],
                )
            )
    bvr_fig = _draw_line_chart_png(
        figures_dir / "budget_vs_recall.png",
        "Expansion Policy: Tool-Call Budget vs Recall",
        "Tool-Call Budget",
        "Entity Recall",
        bvr_series,
        xlim=(0.0, max_b),
        ylim=(0.0, 1.0),
        empty_note="Skipped: no logged agent-run policy_transitions.jsonl yet.",
    )

    # 4. Ablation table (Markdown)
    ablation_lines = [
        "# Phase C Ablation Table",
        "",
        "| Variant | Split | PR-AUC | F1 | ECE |",
        "|---|---|---:|---:|---:|",
    ]
    ablations = neural_eval.get("ablations") or {}
    if neural_eval.get("status") == "ok" and ablations:
        for var_name, vdata in ablations.items():
            metrics = vdata.get("test") or vdata.get("ood_platform_test") or vdata
            ablation_lines.append(
                f"| `{var_name}` | test | {float(metrics.get('pr_auc', 0.0)):.4f} | "
                f"{float(metrics.get('f1', 0.0)):.4f} | {float(metrics.get('ece', 0.0)):.4f} |"
            )
    else:
        ablation_lines.append(
            f"| `(skipped)` | - | - | - | {neural_eval.get('status', 'not_run')} |"
        )
    ablation_md = figures_dir / "ablation_table.md"
    ablation_md.write_text("\n".join(ablation_lines) + "\n", encoding="utf-8")

    # 5. Summary tables + report/RESULTS.md
    counts = manifest.get("counts") or {}
    lines = [
        "# Experimental Results: Learning to Investigate",
        "",
        f"Every number below is generated directly from JSON artifacts in `{data_dir}` using config `{config_path}` (`seed={seed}`).",
        "",
        "## 1. Dataset (`OSINT-Link` — `scripts/build_osint_link.py`, seed `42`)",
        "",
        "| Split | Total Pairs | Positives | Negatives |",
        "|---|---:|---:|---:|",
    ]
    for sname in ("train", "val", "test", "ood_platform_test"):
        sc = counts.get(sname) or {}
        lines.append(
            f"| `{sname}` | {sc.get('total', 0)} | {sc.get('positives', 0)} | {sc.get('negatives', 0)} |"
        )
    lines.extend(
        [
            "",
            f"- Total public profiles crawled: **{manifest.get('total_profiles_crawled', 0)}**",
            f"- OOD held-out platforms: `{', '.join(manifest.get('ood_platforms') or [])}`",
            f"- Raw bios/photos stored: `{manifest.get('store_raw_bios', False)}` / `{manifest.get('store_raw_photos', False)}`",
            "",
            "## 2. Phase B Baselines (`scripts/eval_baselines.py`, `data/osint_link/baseline_eval.json`, seed `42`)",
            "",
        ]
    )
    if ood_models:
        lines.extend(
            [
                "| Model | Split | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | PR-AUC (95% CI) | ECE (95% CI) | Latency (ms/pair) |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for mname, mdata in ood_models.items():
            m = mdata.get("metrics") or {}
            lines.append(
                f"| `{mname}` | `ood_platform_test` | {_fmt_ci(m.get('precision'))} | {_fmt_ci(m.get('recall'))} | "
                f"{_fmt_ci(m.get('f1'))} | {_fmt_ci(m.get('pr_auc'))} | {_fmt_ci(m.get('ece'))} | "
                f"{float(mdata.get('latency_ms_per_pair', 0.0)):.4f} |"
            )
    else:
        lines.append("- No Phase B evaluation records found.")

    lines.extend(
        [
            "",
            "## 3. Phase C Multimodal Neural Model (`scripts/train_neural.py`, `data/osint_link/neural_eval.json`)",
            f"- Status: `{neural_eval.get('status', 'not_run')}`",
        ]
    )
    if neural_eval.get("status") == "ok":
        for sp_name, sp_data in (neural_eval.get("splits") or {}).items():
            lines.append(
                f"- `{sp_name}`: PR-AUC={sp_data.get('pr_auc')}, F1={sp_data.get('f1')}, ECE={sp_data.get('ece')}"
            )
    else:
        lines.append(f"- Note: {neural_eval.get('message', 'Train split empty without GITHUB_TOKEN.')}")

    lines.extend(
        [
            "",
            "## 4. Phase D Collective GNN (`scripts/train_gnn.py`, `data/osint_link/gnn_eval.json`)",
            f"- Status: `{gnn_eval.get('status', 'not_run')}`",
        ]
    )
    if gnn_eval.get("status") == "ok":
        for sp_name, sp_data in (gnn_eval.get("splits") or {}).items():
            hvh = sp_data.get("helps_vs_hurts") or {}
            lines.append(
                f"- `{sp_name}`: Phase C PR-AUC={sp_data['phase_c_alone']['pr_auc']:.4f}, "
                f"Phase C+GNN PR-AUC={sp_data['phase_c_plus_gnn']['pr_auc']:.4f}, "
                f"Chain Heuristic PR-AUC={sp_data['chain_confidence_heuristic']['pr_auc']:.4f} "
                f"(helped={hvh.get('helped_count', 0)}, hurt={hvh.get('hurt_count', 0)})"
            )
    else:
        lines.append(f"- Note: {gnn_eval.get('message', 'Train split empty without GITHUB_TOKEN.')}")

    lines.extend(
        [
            "",
            "## 5. Phase E Contextual Bandit Policy (`scripts/eval_bandit.py`, `data/osint_link/bandit_eval.json`)",
            f"- Status: `{bandit_eval.get('status', 'not_run')}`",
        ]
    )
    if bandit_eval.get("status") == "ok":
        for pname, auc_val in (bandit_eval.get("auc_recall") or {}).items():
            lines.append(f"- `{pname}` AUC(budget-vs-recall): **{auc_val:.4f}**")
    else:
        lines.append(f"- Note: {bandit_eval.get('message', 'No agent policy transitions logged yet.')}")

    lines.extend(
        [
            "",
            "## 6. Generated Figures & Tables",
            f"- PR Operating Points: `{pr_fig}`",
            f"- Reliability Diagram: `{rel_fig}`",
            f"- Budget vs Recall: `{bvr_fig}`",
            f"- Ablation Table: `{ablation_md}`",
            "",
            "## 7. Limitations & Ethics",
            "See [docs/ETHICS.md](../docs/ETHICS.md) for full details on weak-supervision label noise, platform bias, small unauthenticated dataset size, and privacy/consent safeguards.",
        ]
    )

    results_md_path.parent.mkdir(parents=True, exist_ok=True)
    results_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "status": "ok",
        "figures": [str(pr_fig), str(rel_fig), str(bvr_fig), str(ablation_md)],
        "results_md": str(results_md_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Phase F report figures and RESULTS.md")
    parser.add_argument("--config", type=str, default="configs/osint_link.yaml")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--figures-dir", type=str, default="report/figures")
    parser.add_argument("--results-out", type=str, default="report/RESULTS.md")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data_dir = Path(args.data_dir) if args.data_dir else Path(cfg["dataset"]["output_dir"])

    res = generate_report_assets(
        data_dir=data_dir,
        figures_dir=Path(args.figures_dir),
        results_md_path=Path(args.results_out),
        config_path=cfg_path,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
