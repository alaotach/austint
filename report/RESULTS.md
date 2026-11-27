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

## 6. Phase G Validation & Stress Tests (`scripts/run_validation.py`, `data/osint_link/validation_eval.json`)

### 6.1 GNN Leakage Audit (Before vs After)

| GNN Variant | Split | PR-AUC | F1 | Precision | Recall | ECE |
|---|---|---:|---:|---:|---:|---:|
| Old GNN (pre-audit) | `test` | 0.9250 | 0.8391 | 0.8022 | 0.8795 | 0.0742 |
| Old GNN (pre-audit) | `ood_platform_test` | 0.8481 | 0.8033 | 0.7656 | 0.8448 | 0.1462 |
| New GNN (pair-links masked, split-isolated) | `test` | 0.9348 | 0.8362 | 0.7872 | 0.8916 | 0.0797 |
| New GNN (pair-links masked, split-isolated) | `ood_platform_test` | 0.8585 | 0.7656 | 0.7000 | 0.8448 | 0.1615 |
| New GNN (candidate-only edges, no anchor broadcast) | `test` | 0.8532 | 0.7123 | 0.8254 | 0.6265 | 0.0665 |
| New GNN (candidate-only edges, no anchor broadcast) | `ood_platform_test` | 0.6967 | 0.5625 | 0.7105 | 0.4655 | 0.1044 |

### 6.2 Person-Clustered Paired Bootstrap (1,000 iterations, seed `42`)

| Split | Comparison | ΔPR-AUC (95% CI) | PR-AUC Sig. | ΔF1 (95% CI) | F1 Sig. |
|---|---|---:|---|---:|---|
| `test` | GNN vs HistGBM | +0.0196 [-0.0216, +0.0644] | not significant | +0.0100 [-0.0691, +0.0886] | not significant |
| `test` | Neural vs HistGBM | -0.0948 [-0.1541, -0.0429] | significant | -0.1373 [-0.2089, -0.0699] | significant |
| `test` | GNN vs Neural | +0.1145 [+0.0744, +0.1585] | significant | +0.1473 [+0.0849, +0.2171] | significant |
| `ood_platform_test` | GNN vs HistGBM | +0.0977 [+0.0059, +0.2091] | significant | +0.0939 [-0.0552, +0.2571] | not significant |
| `ood_platform_test` | Neural vs HistGBM | +0.0421 [-0.0363, +0.1278] | not significant | +0.0557 [-0.0890, +0.2144] | not significant |
| `ood_platform_test` | GNN vs Neural | +0.0556 [+0.0089, +0.1077] | significant | +0.0382 [-0.0381, +0.1368] | not significant |

### 6.3 Shortcut Baselines (Dataset Bias Probe)

| Shortcut Model | Split | Pos Rate | PR-AUC | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| `platform_pair_only` | `test` | 0.3673 | 0.6156 | 0.7315 | 0.5940 | 0.9518 |
| `username_sim_only` | `test` | 0.3673 | 0.7621 | 0.7519 | 1.0000 | 0.6024 |
| `bio_len_has_bio_only` | `test` | 0.3673 | 0.4420 | 0.0465 | 0.6667 | 0.0241 |
| `platform_pair_only` | `ood_platform_test` | 0.4143 | 0.5275 | 0.5827 | 0.5362 | 0.6379 |
| `username_sim_only` | `ood_platform_test` | 0.4143 | 0.6842 | 0.6047 | 0.9286 | 0.4483 |
| `bio_len_has_bio_only` | `ood_platform_test` | 0.4143 | 0.4849 | 0.0000 | 0.0000 | 0.0000 |
- **Dataset Bias Flag**: platform_pair_only achieves PR-AUC=0.6156 vs random baseline=0.3673 on test

### 6.4 Per-Negative-Type Breakdown (`test` & `ood_platform_test`)

| Split | Negative Type | Hard Neg? | Model | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---:|---:|---:|---:|
| `test` | `diff_person_fallback` | no | `rule_baseline` | 0.9327 | 0.6825 | 1.0000 | 0.5181 |
| `test` | `diff_person_fallback` | no | `logistic_regression` | 0.9573 | 0.8333 | 0.9836 | 0.7229 |
| `test` | `diff_person_fallback` | no | `hist_gradient_boosting` | 0.9649 | 0.8356 | 0.9683 | 0.7349 |
| `test` | `diff_person_fallback` | no | `phase_c_neural` | 0.9366 | 0.7808 | 0.9048 | 0.6867 |
| `test` | `diff_person_fallback` | no | `phase_c_plus_gnn` | 0.9854 | 0.9136 | 0.9367 | 0.8916 |
| `test` | `similar_username_diff_person` | **YES (hard)** | `rule_baseline` | 0.8401 | 0.6466 | 0.8600 | 0.5181 |
| `test` | `similar_username_diff_person` | **YES (hard)** | `logistic_regression` | 0.9163 | 0.8219 | 0.9524 | 0.7229 |
| `test` | `similar_username_diff_person` | **YES (hard)** | `hist_gradient_boosting` | 0.9675 | 0.8472 | 1.0000 | 0.7349 |
| `test` | `similar_username_diff_person` | **YES (hard)** | `phase_c_neural` | 0.9017 | 0.7600 | 0.8507 | 0.6867 |
| `test` | `similar_username_diff_person` | **YES (hard)** | `phase_c_plus_gnn` | 0.9671 | 0.8970 | 0.9024 | 0.8916 |
| `test` | `same_name_diff_org_city` | **YES (hard)** | `rule_baseline` | 0.8735 | 0.6615 | 0.9149 | 0.5181 |
| `test` | `same_name_diff_org_city` | **YES (hard)** | `logistic_regression` | 0.9380 | 0.8219 | 0.9524 | 0.7229 |
| `test` | `same_name_diff_org_city` | **YES (hard)** | `hist_gradient_boosting` | 0.9575 | 0.8414 | 0.9839 | 0.7349 |
| `test` | `same_name_diff_org_city` | **YES (hard)** | `phase_c_neural` | 0.9412 | 0.7917 | 0.9344 | 0.6867 |
| `test` | `same_name_diff_org_city` | **YES (hard)** | `phase_c_plus_gnn` | 0.9843 | 0.9193 | 0.9487 | 0.8916 |
| `test` | `same_display_name_diff_links` | **YES (hard)** | `rule_baseline` | 0.9639 | 0.6772 | 0.9773 | 0.5181 |
| `test` | `same_display_name_diff_links` | **YES (hard)** | `logistic_regression` | 0.9819 | 0.8392 | 1.0000 | 0.7229 |
| `test` | `same_display_name_diff_links` | **YES (hard)** | `hist_gradient_boosting` | 0.9884 | 0.8472 | 1.0000 | 0.7349 |
| `test` | `same_display_name_diff_links` | **YES (hard)** | `phase_c_neural` | 0.9656 | 0.7862 | 0.9194 | 0.6867 |
| `test` | `same_display_name_diff_links` | **YES (hard)** | `phase_c_plus_gnn` | 0.9858 | 0.9250 | 0.9610 | 0.8916 |
| `ood_platform_test` | `diff_person_fallback` | no | `rule_baseline` | 0.8517 | 0.4935 | 1.0000 | 0.3276 |
| `ood_platform_test` | `diff_person_fallback` | no | `logistic_regression` | 0.8915 | 0.6818 | 1.0000 | 0.5172 |
| `ood_platform_test` | `diff_person_fallback` | no | `hist_gradient_boosting` | 0.8499 | 0.6818 | 1.0000 | 0.5172 |
| `ood_platform_test` | `diff_person_fallback` | no | `phase_c_neural` | 0.9040 | 0.7961 | 0.9111 | 0.7069 |
| `ood_platform_test` | `diff_person_fallback` | no | `phase_c_plus_gnn` | 0.9349 | 0.8448 | 0.8448 | 0.8448 |
| `ood_platform_test` | `similar_username_diff_person` | **YES (hard)** | `rule_baseline` | 0.8231 | 0.4872 | 0.9500 | 0.3276 |
| `ood_platform_test` | `similar_username_diff_person` | **YES (hard)** | `logistic_regression` | 0.8591 | 0.6742 | 0.9677 | 0.5172 |
| `ood_platform_test` | `similar_username_diff_person` | **YES (hard)** | `hist_gradient_boosting` | 0.8946 | 0.6818 | 1.0000 | 0.5172 |
| `ood_platform_test` | `similar_username_diff_person` | **YES (hard)** | `phase_c_neural` | 0.8968 | 0.7810 | 0.8723 | 0.7069 |
| `ood_platform_test` | `similar_username_diff_person` | **YES (hard)** | `phase_c_plus_gnn` | 0.9312 | 0.8522 | 0.8596 | 0.8448 |
| `ood_platform_test` | `same_name_diff_org_city` | **YES (hard)** | `rule_baseline` | 0.8426 | 0.4872 | 0.9500 | 0.3276 |
| `ood_platform_test` | `same_name_diff_org_city` | **YES (hard)** | `logistic_regression` | 0.8843 | 0.6742 | 0.9677 | 0.5172 |
| `ood_platform_test` | `same_name_diff_org_city` | **YES (hard)** | `hist_gradient_boosting` | 0.8827 | 0.6742 | 0.9677 | 0.5172 |
| `ood_platform_test` | `same_name_diff_org_city` | **YES (hard)** | `phase_c_neural` | 0.9300 | 0.7961 | 0.9111 | 0.7069 |
| `ood_platform_test` | `same_name_diff_org_city` | **YES (hard)** | `phase_c_plus_gnn` | 0.9579 | 0.8829 | 0.9245 | 0.8448 |

### 6.5 Val-Tuned Chain-Confidence Baseline
- Val-tuned threshold: `0.01`
- `test`: PR-AUC=0.4569, F1=0.4000, mean_pos=0.0499, mean_neg=0.0232, pct_below_0.1=0.9204
- `ood_platform_test`: PR-AUC=0.4939, F1=0.4158, mean_pos=0.0696, mean_neg=0.0547, pct_below_0.1=0.8714
- Explanation: Multiplicative confidence decay (`prod(score * 0.85)`) compresses ~87-92% of multi-hop scores below `0.1`, collapsing threshold separability.

### 6.6 Zero-Shot LLM Judge (2 Prompt Variants)
- Status: `ok` (provider: `g4f/GithubCopilot/gpt-4.1`)
- `minimal`: PR-AUC=0.9617, F1=0.8502, ECE=0.1085, tokens=65191, cost=$0.0, latency=10468.1384 ms/pair
- `field_by_field`: PR-AUC=0.9685, F1=0.8216, ECE=0.1144, tokens=77269, cost=$0.0, latency=6476.8015 ms/pair

## 7. Generated Figures & Tables
- PR Operating Points: `report/figures/pr_curves.png`
- Reliability Diagram: `report/figures/reliability_diagram.png`
- Budget vs Recall: `report/figures/budget_vs_recall.png`
- Ablation Table: `report/figures/ablation_table.md`

## 8. Threats to Validity
- **Small dataset**: Only 1,003 public profiles and 1,579 total pairs (574 positives / 1,005 negatives), below the 3k/6k stretch target, yielding wide bootstrap confidence intervals where several differences (e.g., GNN vs HistGBM on in-distribution `test`) are not statistically significant.
- **Developer-skewed platforms**: Seeds originate from open-source GitHub organizations and fediverse instances (`mastodon.social`, `fosstodon.org`, `hachyderm.io`), which over-represent technical maintainers.
- **Weak-supervision label noise**: Positives rely on self-declared links; stale/abandoned or org-shared links can introduce label noise (0/100 zero-signal pairs flagged in our 100-pair audit).
- **Platform bias**: `platform_pair_only` achieves `0.6156` PR-AUC on `test` (vs `0.3673` positive rate), showing that platform-pair frequencies carry shortcut signal.
- **Anchor-broadcast GNN sensitivity**: When `left`-anchor username broadcast edges are removed (`candidate_only_edges`), GNN PR-AUC drops from `0.9348` to `0.8532` (`test`) and `0.8585` to `0.6967` (`ood_platform_test`).
- **Ethics & Consent**: See [docs/ETHICS.md](../docs/ETHICS.md) for privacy, consent-gate, and breach-tool gating details.
