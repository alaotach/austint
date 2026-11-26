# Experimental Results: Learning to Investigate

Every number below is generated directly from JSON artifacts in `data/osint_link` using config `configs/osint_link.yaml` (`seed=42`).

## 1. Dataset (`OSINT-Link` — `scripts/build_osint_link.py`, seed `42`)

| Split | Total Pairs | Positives | Negatives |
|---|---:|---:|---:|
| `train` | 0 | 0 | 0 |
| `val` | 0 | 0 | 0 |
| `test` | 0 | 0 | 0 |
| `ood_platform_test` | 114 | 49 | 65 |

- Total public profiles crawled: **84**
- OOD held-out platforms: `devto, mastodon`
- Raw bios/photos stored: `False` / `False`

## 2. Phase B Baselines (`scripts/eval_baselines.py`, `data/osint_link/baseline_eval.json`, seed `42`)

| Model | Split | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | PR-AUC (95% CI) | ECE (95% CI) | Latency (ms/pair) |
|---|---|---:|---:|---:|---:|---:|---:|
| `rule_baseline` | `ood_platform_test` | 1.0000 [0.0000, 1.0000] | 0.0204 [0.0000, 0.0721] | 0.0400 [0.0000, 0.1345] | 0.7140 [0.5973, 0.8175] | 0.1628 [0.0927, 0.2472] | 0.0016 |
| `zero_shot_llm_judge` | `ood_platform_test` | 0.4298 [0.3246, 0.5175] | 1.0000 [1.0000, 1.0000] | 0.6012 [0.4901, 0.6821] | 0.4298 [0.3246, 0.5175] | 0.0702 [0.0000, 0.1754] | 0.0007 |

## 3. Phase C Multimodal Neural Model (`scripts/train_neural.py`, `data/osint_link/neural_eval.json`)
- Status: `skipped_insufficient_train_data`
- Note: Cannot train neural model on empty/single-class train split. Run build_osint_link.py with GITHUB_TOKEN first.

## 4. Phase D Collective GNN (`scripts/train_gnn.py`, `data/osint_link/gnn_eval.json`)
- Status: `skipped_insufficient_train_data`
- Note: Cannot train CollectiveGNN on empty/single-class train split. Run build_osint_link.py with GITHUB_TOKEN first.

## 5. Phase E Contextual Bandit Policy (`scripts/eval_bandit.py`, `data/osint_link/bandit_eval.json`)
- Status: `skipped_no_agent_transitions`
- Note: No logged agent-run policy_transitions.jsonl or eval episodes found. Run OSINTAgent with policy_log_path enabled first.

## 6. Generated Figures & Tables
- PR Operating Points: `report/figures/pr_curves.png`
- Reliability Diagram: `report/figures/reliability_diagram.png`
- Budget vs Recall: `report/figures/budget_vs_recall.png`
- Ablation Table: `report/figures/ablation_table.md`

## 7. Limitations & Ethics
See [docs/ETHICS.md](../docs/ETHICS.md) for full details on weak-supervision label noise, platform bias, small unauthenticated dataset size, and privacy/consent safeguards.
