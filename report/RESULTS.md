# Experimental Results: Learning to Investigate

Every number below is generated directly from JSON artifacts in `data/osint_link` using config `configs/osint_link.yaml` (`seed=42`).

## 1. Dataset (`OSINT-Link` — `scripts/build_osint_link.py`, seed `42`)

| Split | Total Pairs | Positives | Negatives |
|---|---:|---:|---:|
| `train` | 1017 | 363 | 654 |
| `val` | 196 | 70 | 126 |
| `test` | 226 | 83 | 143 |
| `ood_platform_test` | 140 | 58 | 82 |

- Total public profiles crawled: **1003**
- OOD held-out platforms: `devto, mastodon`
- Raw bios/photos stored: `False` / `False`

## 2. Phase B Baselines (`scripts/eval_baselines.py`, `data/osint_link/baseline_eval.json`, seed `42`)

| Model | Split | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | PR-AUC (95% CI) | ECE (95% CI) | Latency (ms/pair) |
|---|---|---:|---:|---:|---:|---:|---:|
| `rule_baseline` | `test` | 0.8431 [0.7379, 0.9238] | 0.5181 [0.4144, 0.6095] | 0.6418 [0.5433, 0.7211] | 0.7414 [0.6530, 0.8151] | 0.1369 [0.0926, 0.1879] | 0.0029 |
| `logistic_regression` | `test` | 0.8955 [0.8068, 0.9600] | 0.7229 [0.6311, 0.8158] | 0.8000 [0.7211, 0.8722] | 0.8685 [0.8080, 0.9235] | 0.1323 [0.1001, 0.1800] | 0.0020 |
| `hist_gradient_boosting` | `test` | 0.9531 [0.8991, 1.0000] | 0.7349 [0.6410, 0.8256] | 0.8299 [0.7602, 0.8919] | 0.9174 [0.8705, 0.9558] | 0.0549 [0.0369, 0.1051] | 0.0078 |
| `rule_baseline` | `ood_platform_test` | 0.8824 [0.7000, 1.0000] | 0.2586 [0.1557, 0.3661] | 0.4000 [0.2623, 0.5174] | 0.6856 [0.5730, 0.7743] | 0.1745 [0.1190, 0.2510] | 0.0018 |
| `logistic_regression` | `ood_platform_test` | 0.9375 [0.8308, 1.0000] | 0.5172 [0.3783, 0.6374] | 0.6667 [0.5370, 0.7624] | 0.7572 [0.6524, 0.8362] | 0.1752 [0.1240, 0.2429] | 0.0012 |
| `hist_gradient_boosting` | `ood_platform_test` | 0.9677 [0.9032, 1.0000] | 0.5172 [0.3783, 0.6374] | 0.6742 [0.5455, 0.7702] | 0.7605 [0.6517, 0.8392] | 0.1514 [0.0949, 0.2251] | 0.0112 |

- Note: `zero_shot_llm_judge` is omitted because OpenRouter returned HTTP 429 rate-limited (`total_tokens=0`); 0.5 fallback values are not reported.

## 3. Phase C Multimodal Neural Model (`scripts/train_neural.py`, `data/osint_link/neural_eval.json`)
- Status: `completed`
- Calibration: temperature=`1.449`, val ECE before=`0.1121`, val ECE after=`0.1304`
- `test`: PR-AUC=0.8216 [0.7574, 0.8801], F1=0.6909 [0.6093, 0.7645], ECE=0.0875 [0.0702, 0.1502], latency=0.1609 ms/pair
- `ood_platform_test`: PR-AUC=0.8012 [0.6990, 0.8787], F1=0.7257 [0.6104, 0.8108], ECE=0.1426 [0.1150, 0.2187], latency=0.0826 ms/pair

## 4. Phase D Collective GNN (`scripts/train_gnn.py`, `data/osint_link/gnn_eval.json`)
- Status: `ok`
- Calibration: temperature=`2.216`, val ECE before=`0.0351`, val ECE after=`0.028`
- `test`: Phase C PR-AUC=0.8216 (F1=0.6909), Phase C+GNN PR-AUC=0.9250 (F1=0.8391), Chain Heuristic PR-AUC=0.4569 (F1=0.0465) | helps_vs_hurts: helped=28, hurt=5, net=23
- `ood_platform_test`: Phase C PR-AUC=0.8012 (F1=0.7257), Phase C+GNN PR-AUC=0.8481 (F1=0.8033), Chain Heuristic PR-AUC=0.4939 (F1=0.1212) | helps_vs_hurts: helped=12, hurt=5, net=7

## 5. Phase E Contextual Bandit Policy (`scripts/eval_bandit.py`, `data/osint_link/bandit_eval.json`)
- Status: `skipped_no_agent_transitions` (future work for live agent-run logging)
- Note: No logged agent-run policy_transitions.jsonl or eval episodes found. Run OSINTAgent with policy_log_path enabled first.

## 6. Generated Figures & Tables
- PR Operating Points: `report/figures/pr_curves.png`
- Reliability Diagram: `report/figures/reliability_diagram.png`
- Budget vs Recall: `report/figures/budget_vs_recall.png`
- Ablation Table: `report/figures/ablation_table.md`

## 7. Limitations & Ethics
See [docs/ETHICS.md](../docs/ETHICS.md) for full details on weak-supervision label noise, platform bias, small unauthenticated dataset size, and privacy/consent safeguards.
