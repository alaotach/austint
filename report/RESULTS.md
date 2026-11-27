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
| `rule_baseline` | `test` | 0.8431 [0.7379, 0.9238] | 0.5181 [0.4144, 0.6095] | 0.6418 [0.5433, 0.7211] | 0.7414 [0.6530, 0.8151] | 0.1369 [0.0926, 0.1879] | 0.0021 |
| `logistic_regression` | `test` | 0.8955 [0.8068, 0.9600] | 0.7229 [0.6311, 0.8158] | 0.8000 [0.7211, 0.8722] | 0.8685 [0.8080, 0.9235] | 0.1323 [0.1001, 0.1800] | 0.0007 |
| `hist_gradient_boosting` | `test` | 0.9531 [0.8991, 1.0000] | 0.7349 [0.6410, 0.8256] | 0.8299 [0.7602, 0.8919] | 0.9174 [0.8705, 0.9558] | 0.0549 [0.0369, 0.1051] | 0.0048 |
| `rule_baseline` | `ood_platform_test` | 0.8333 [0.5833, 1.0000] | 0.3125 [0.1429, 0.4678] | 0.4545 [0.2284, 0.6087] | 0.6475 [0.4986, 0.7752] | 0.1069 [0.0680, 0.1893] | 0.0016 |
| `logistic_regression` | `ood_platform_test` | 0.9000 [0.7647, 1.0000] | 0.5625 [0.3968, 0.7188] | 0.6923 [0.5411, 0.8142] | 0.7372 [0.5959, 0.8606] | 0.1191 [0.0864, 0.1917] | 0.0013 |
| `hist_gradient_boosting` | `ood_platform_test` | 0.9474 [0.8440, 1.0000] | 0.5625 [0.3968, 0.7188] | 0.7059 [0.5555, 0.8264] | 0.7303 [0.6066, 0.8636] | 0.0792 [0.0335, 0.1452] | 0.0091 |

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
| New GNN (pair-links masked, split-isolated) | `test` | 0.9349 | 0.8315 | 0.7789 | 0.8916 | 0.0842 |
| New GNN (pair-links masked, split-isolated) | `ood_platform_test` | 0.8384 | 0.6747 | 0.5490 | 0.8750 | 0.1966 |
| New GNN (candidate-only edges, no anchor broadcast) | `test` | 0.8532 | 0.7123 | 0.8254 | 0.6265 | 0.0665 |
| New GNN (candidate-only edges, no anchor broadcast) | `ood_platform_test` | 0.6809 | 0.5902 | 0.6207 | 0.5625 | 0.0945 |

### 6.2 Person-Clustered Paired Bootstrap (1,000 iterations, seed `42`)

| Split | Comparison | ΔPR-AUC (95% CI) | PR-AUC Sig. | ΔF1 (95% CI) | F1 Sig. |
|---|---|---:|---|---:|---|
| `test` | GNN vs HistGBM | +0.0198 [-0.0217, +0.0646] | not significant | +0.0054 [-0.0747, +0.0881] | not significant |
| `test` | Neural vs HistGBM | -0.0948 [-0.1541, -0.0429] | significant | -0.1373 [-0.2089, -0.0699] | significant |
| `test` | GNN vs Neural | +0.1146 [+0.0740, +0.1594] | significant | +0.1427 [+0.0814, +0.2107] | significant |
| `ood_platform_test` | GNN vs HistGBM | +0.1097 [+0.0007, +0.2158] | significant | -0.0255 [-0.1427, +0.1025] | not significant |
| `ood_platform_test` | Neural vs HistGBM | -0.0098 [-0.1223, +0.1015] | not significant | -0.0345 [-0.1741, +0.1213] | not significant |
| `ood_platform_test` | GNN vs Neural | +0.1194 [+0.0658, +0.1747] | significant | +0.0090 [-0.0619, +0.0830] | not significant |

### 6.3 Shortcut Baselines (Dataset Bias Probe)

| Shortcut Model | Split | Pos Rate | PR-AUC | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| `platform_pair_only` | `test` | 0.3673 | 0.6156 | 0.7315 | 0.5940 | 0.9518 |
| `username_sim_only` | `test` | 0.3673 | 0.7621 | 0.7519 | 1.0000 | 0.6024 |
| `bio_len_has_bio_only` | `test` | 0.3673 | 0.4420 | 0.0465 | 0.6667 | 0.0241 |
| `exact_display_name_only` | `test` | 0.3673 | 0.5121 | 0.3725 | 1.0000 | 0.2289 |
| `exact_username_only` | `test` | 0.3673 | 0.5426 | 0.4340 | 1.0000 | 0.2771 |
| `has_bio_only` | `test` | 0.3673 | 0.3677 | 0.0000 | 0.0000 | 0.0000 |
| `platform_pair_only` | `ood_platform_test` | 0.2857 | 0.3661 | 0.4444 | 0.3673 | 0.5625 |
| `username_sim_only` | `ood_platform_test` | 0.2857 | 0.6137 | 0.5833 | 0.8750 | 0.4375 |
| `bio_len_has_bio_only` | `ood_platform_test` | 0.2857 | 0.3336 | 0.0000 | 0.0000 | 0.0000 |
| `exact_display_name_only` | `ood_platform_test` | 0.2857 | 0.8438 | 0.8772 | 1.0000 | 0.7812 |
| `exact_username_only` | `ood_platform_test` | 0.2857 | 0.5312 | 0.5116 | 1.0000 | 0.3438 |
| `has_bio_only` | `ood_platform_test` | 0.2857 | 0.2530 | 0.0000 | 0.0000 | 0.0000 |
- **Dataset Bias Flag**: platform_pair_only achieves PR-AUC=0.6156 vs random baseline=0.3673 on test

### 6.4 Per-Negative-Type Breakdown (`test` & `ood_platform_test`)

| Split | Negative Type | Hard Neg? | Model | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---:|---:|---:|---:|
| `test` | `diff_person_fallback` | no | `rule_baseline` | 0.9327 | 0.6825 | 1.0000 | 0.5181 |
| `test` | `diff_person_fallback` | no | `logistic_regression` | 0.9573 | 0.8333 | 0.9836 | 0.7229 |
| `test` | `diff_person_fallback` | no | `hist_gradient_boosting` | 0.9649 | 0.8356 | 0.9683 | 0.7349 |
| `test` | `diff_person_fallback` | no | `phase_c_neural` | 0.9366 | 0.7808 | 0.9048 | 0.6867 |
| `test` | `diff_person_fallback` | no | `phase_c_plus_gnn` | 0.9853 | 0.9136 | 0.9367 | 0.8916 |
| `test` | `similar_username_diff_person` | **YES (hard)** | `rule_baseline` | 0.8401 | 0.6466 | 0.8600 | 0.5181 |
| `test` | `similar_username_diff_person` | **YES (hard)** | `logistic_regression` | 0.9163 | 0.8219 | 0.9524 | 0.7229 |
| `test` | `similar_username_diff_person` | **YES (hard)** | `hist_gradient_boosting` | 0.9675 | 0.8472 | 1.0000 | 0.7349 |
| `test` | `similar_username_diff_person` | **YES (hard)** | `phase_c_neural` | 0.9017 | 0.7600 | 0.8507 | 0.6867 |
| `test` | `similar_username_diff_person` | **YES (hard)** | `phase_c_plus_gnn` | 0.9679 | 0.8916 | 0.8916 | 0.8916 |
| `test` | `same_name_diff_org_city` | **YES (hard)** | `rule_baseline` | 0.8735 | 0.6615 | 0.9149 | 0.5181 |
| `test` | `same_name_diff_org_city` | **YES (hard)** | `logistic_regression` | 0.9380 | 0.8219 | 0.9524 | 0.7229 |
| `test` | `same_name_diff_org_city` | **YES (hard)** | `hist_gradient_boosting` | 0.9575 | 0.8414 | 0.9839 | 0.7349 |
| `test` | `same_name_diff_org_city` | **YES (hard)** | `phase_c_neural` | 0.9412 | 0.7917 | 0.9344 | 0.6867 |
| `test` | `same_name_diff_org_city` | **YES (hard)** | `phase_c_plus_gnn` | 0.9843 | 0.9193 | 0.9487 | 0.8916 |
| `test` | `same_display_name_diff_links` | **YES (hard)** | `rule_baseline` | 0.9639 | 0.6772 | 0.9773 | 0.5181 |
| `test` | `same_display_name_diff_links` | **YES (hard)** | `logistic_regression` | 0.9819 | 0.8392 | 1.0000 | 0.7229 |
| `test` | `same_display_name_diff_links` | **YES (hard)** | `hist_gradient_boosting` | 0.9884 | 0.8472 | 1.0000 | 0.7349 |
| `test` | `same_display_name_diff_links` | **YES (hard)** | `phase_c_neural` | 0.9656 | 0.7862 | 0.9194 | 0.6867 |
| `test` | `same_display_name_diff_links` | **YES (hard)** | `phase_c_plus_gnn` | 0.9855 | 0.9250 | 0.9610 | 0.8916 |
| `ood_platform_test` | `diff_person_fallback` | no | `rule_baseline` | 0.8190 | 0.5455 | 1.0000 | 0.3750 |
| `ood_platform_test` | `diff_person_fallback` | no | `logistic_regression` | 0.8679 | 0.7200 | 1.0000 | 0.5625 |
| `ood_platform_test` | `diff_person_fallback` | no | `hist_gradient_boosting` | 0.8146 | 0.7200 | 1.0000 | 0.5625 |
| `ood_platform_test` | `diff_person_fallback` | no | `phase_c_neural` | 0.8520 | 0.7797 | 0.8519 | 0.7188 |
| `ood_platform_test` | `diff_person_fallback` | no | `phase_c_plus_gnn` | 0.9228 | 0.8000 | 0.7368 | 0.8750 |
| `ood_platform_test` | `similar_username_diff_person` | **YES (hard)** | `rule_baseline` | 0.7785 | 0.5333 | 0.9231 | 0.3750 |
| `ood_platform_test` | `similar_username_diff_person` | **YES (hard)** | `logistic_regression` | 0.8324 | 0.7059 | 0.9474 | 0.5625 |
| `ood_platform_test` | `similar_username_diff_person` | **YES (hard)** | `hist_gradient_boosting` | 0.8665 | 0.7200 | 1.0000 | 0.5625 |
| `ood_platform_test` | `similar_username_diff_person` | **YES (hard)** | `phase_c_neural` | 0.8378 | 0.7541 | 0.7931 | 0.7188 |
| `ood_platform_test` | `similar_username_diff_person` | **YES (hard)** | `phase_c_plus_gnn` | 0.9134 | 0.8116 | 0.7568 | 0.8750 |
| `ood_platform_test` | `same_name_diff_org_city` | **YES (hard)** | `rule_baseline` | 0.8008 | 0.5333 | 0.9231 | 0.3750 |
| `ood_platform_test` | `same_name_diff_org_city` | **YES (hard)** | `logistic_regression` | 0.8587 | 0.7059 | 0.9474 | 0.5625 |
| `ood_platform_test` | `same_name_diff_org_city` | **YES (hard)** | `hist_gradient_boosting` | 0.8467 | 0.7059 | 0.9474 | 0.5625 |
| `ood_platform_test` | `same_name_diff_org_city` | **YES (hard)** | `phase_c_neural` | 0.8800 | 0.7797 | 0.8519 | 0.7188 |
| `ood_platform_test` | `same_name_diff_org_city` | **YES (hard)** | `phase_c_plus_gnn` | 0.9480 | 0.8750 | 0.8750 | 0.8750 |

### 6.5 Val-Tuned Chain-Confidence Baseline
- Val-tuned threshold: `0.01`
- `test`: PR-AUC=0.4569, F1=0.4000, mean_pos=0.0499, mean_neg=0.0232, pct_below_0.1=0.9204
- `ood_platform_test`: PR-AUC=0.3379, F1=0.3125, mean_pos=0.0749, mean_neg=0.0561, pct_below_0.1=0.875
- Explanation: Multiplicative confidence decay (`prod(score * 0.85)`) compresses ~87-92% of multi-hop scores below `0.1`, collapsing threshold separability.

### 6.6 Zero-Shot LLM Judge (2 Prompt Variants)
- Status: `ok` (provider: `g4f/GithubCopilot/gpt-4.1 (+EryzaLabs)`)
- `minimal` (val threshold=`0.18`): test PR-AUC=0.9249 [0.8822, 0.9644], F1=0.8344 [0.7651, 0.9000], ECE=0.1135 [0.0885, 0.1470] (isotonic: 0.0806) | OOD PR-AUC=0.9982 [0.9857, 1.0000], F1=0.9538 [0.8710, 1.0000], ECE=0.0982 [0.0817, 0.1236] (isotonic: 0.0341) | tokens=65191, cost=$0.0, latency=9738.6852 ms/pair
- `field_by_field` (val threshold=`0.17`): test PR-AUC=0.9380 [0.8973, 0.9718], F1=0.8642 [0.7974, 0.9241], ECE=0.1292 [0.1073, 0.1646] (isotonic: 0.0644) | OOD PR-AUC=0.9991 [0.9939, 1.0000], F1=0.9846 [0.9508, 1.0000], ECE=0.1053 [0.0819, 0.1342] (isotonic: 0.0256) | tokens=77269, cost=$0.0, latency=6486.8105 ms/pair
- `raw_no_sim` (val threshold=`0.24`): test PR-AUC=0.9385 [0.9064, 0.9695], F1=0.8701 [0.8025, 0.9277], ECE=0.0815 [0.0618, 0.1141] (isotonic: 0.0589) | OOD PR-AUC=0.9853 [0.9566, 1.0000], F1=0.9153 [0.8260, 0.9831], ECE=0.1070 [0.0725, 0.1463] (isotonic: 0.0639) | tokens=356306, cost=$0.0, latency=9887.3253 ms/pair

| Split | Comparison (GNN - LLM) | ΔPR-AUC (95% CI) | PR-AUC Sig. | ΔF1 (95% CI) | F1 Sig. |
|---|---|---:|---|---:|---|
| `test` | `gnn_vs_llm_minimal` | +0.0119 [-0.0345, +0.0614] | not significant | +0.0002 [-0.0906, +0.0885] | not significant |
| `test` | `gnn_vs_llm_field_by_field` | -0.0010 [-0.0419, +0.0435] | not significant | -0.0305 [-0.1087, +0.0494] | not significant |
| `test` | `gnn_vs_llm_raw_no_sim` | -0.0015 [-0.0384, +0.0367] | not significant | -0.0357 [-0.1101, +0.0401] | not significant |
| `ood_platform_test` | `gnn_vs_llm_minimal` | -0.1537 [-0.2294, -0.0805] | significant | -0.2782 [-0.3913, -0.1459] | significant |
| `ood_platform_test` | `gnn_vs_llm_field_by_field` | -0.1555 [-0.2294, -0.0853] | significant | -0.3104 [-0.4021, -0.2144] | significant |
| `ood_platform_test` | `gnn_vs_llm_raw_no_sim` | -0.1410 [-0.2078, -0.0734] | significant | -0.2377 [-0.3338, -0.1289] | significant |

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
