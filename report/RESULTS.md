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
| `logistic_regression` | `test` | 0.8955 [0.8068, 0.9600] | 0.7229 [0.6311, 0.8158] | 0.8000 [0.7211, 0.8722] | 0.8685 [0.8080, 0.9235] | 0.1323 [0.1001, 0.1800] | 0.0008 |
| `hist_gradient_boosting` | `test` | 0.9531 [0.8991, 1.0000] | 0.7349 [0.6410, 0.8256] | 0.8299 [0.7602, 0.8919] | 0.9174 [0.8705, 0.9558] | 0.0549 [0.0369, 0.1051] | 0.0048 |
| `rule_baseline` | `ood_platform_test` | 0.8182 [0.5912, 1.0000] | 0.6429 [0.4000, 0.8711] | 0.7200 [0.5000, 0.8966] | 0.8414 [0.6583, 0.9818] | 0.1828 [0.1402, 0.2354] | 0.0014 |
| `logistic_regression` | `ood_platform_test` | 0.8571 [0.6667, 1.0000] | 0.8571 [0.6429, 1.0000] | 0.8571 [0.6776, 0.9655] | 0.8872 [0.7173, 1.0000] | 0.1742 [0.1366, 0.2102] | 0.0020 |
| `hist_gradient_boosting` | `ood_platform_test` | 0.9231 [0.7500, 1.0000] | 0.8571 [0.6429, 1.0000] | 0.8889 [0.7387, 1.0000] | 0.8605 [0.6803, 1.0000] | 0.0442 [0.0213, 0.0764] | 0.0115 |

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
| New GNN (pair-links masked, split-isolated) | `test` | 0.9340 | 0.8427 | 0.7895 | 0.9036 | 0.0707 |
| New GNN (pair-links masked, split-isolated) | `ood_platform_test` | 0.9018 | 0.4333 | 0.2826 | 0.9286 | 0.3250 |
| New GNN (candidate-only edges, no anchor broadcast) | `test` | 0.8532 | 0.7123 | 0.8254 | 0.6265 | 0.0665 |
| New GNN (candidate-only edges, no anchor broadcast) | `ood_platform_test` | 0.7897 | 0.6316 | 0.5000 | 0.8571 | 0.1975 |

### 6.2 Person-Clustered Paired Bootstrap (1,000 iterations, seed `42`)

| Split | Comparison | ΔPR-AUC (95% CI) | PR-AUC Sig. | ΔF1 (95% CI) | F1 Sig. |
|---|---|---:|---|---:|---|
| `test` | GNN vs HistGBM | +0.0188 [-0.0226, +0.0636] | not significant | +0.0161 [-0.0609, +0.0912] | not significant |
| `test` | Neural vs HistGBM | -0.0948 [-0.1541, -0.0429] | significant | -0.1373 [-0.2089, -0.0699] | significant |
| `test` | GNN vs Neural | +0.1136 [+0.0726, +0.1581] | significant | +0.1534 [+0.0915, +0.2233] | significant |
| `test` | gnn_vs_username_sim_only | +0.1746 [+0.1059, +0.2477] | significant | +0.0949 [+0.0145, +0.1798] | significant |
| `test` | gnn_vs_histgbm | +0.0188 [-0.0226, +0.0636] | not significant | +0.0161 [-0.0609, +0.0912] | not significant |
| `ood_platform_test` | GNN vs HistGBM | +0.0362 [-0.0434, +0.1161] | not significant | -0.4499 [-0.6300, -0.2701] | significant |
| `ood_platform_test` | Neural vs HistGBM | -0.0051 [-0.0985, +0.0695] | not significant | -0.2519 [-0.4024, -0.1221] | significant |
| `ood_platform_test` | GNN vs Neural | +0.0414 [+0.0064, +0.0897] | significant | -0.1980 [-0.3433, -0.0629] | significant |

### 6.3 Shortcut Baselines (Dataset Bias Probe)

| Shortcut Model | Split | Pos Rate | PR-AUC | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| `platform_pair_only` | `test` | 0.3673 | 0.6156 | 0.7315 | 0.5940 | 0.9518 |
| `username_sim_only` | `test` | 0.3673 | 0.7621 | 0.7519 | 1.0000 | 0.6024 |
| `bio_len_has_bio_only` | `test` | 0.3673 | 0.4420 | 0.0465 | 0.6667 | 0.0241 |
| `exact_display_name_only` | `test` | 0.3673 | 0.5121 | 0.3725 | 1.0000 | 0.2289 |
| `exact_username_only` | `test` | 0.3673 | 0.5426 | 0.4340 | 1.0000 | 0.2771 |
| `has_bio_only` | `test` | 0.3673 | 0.3677 | 0.0000 | 0.0000 | 0.0000 |
| `platform_pair_only` | `ood_platform_test` | 0.1489 | 0.1507 | 0.0000 | 0.0000 | 0.0000 |
| `username_sim_only` | `ood_platform_test` | 0.1489 | 0.8323 | 0.8148 | 0.8462 | 0.7857 |
| `bio_len_has_bio_only` | `ood_platform_test` | 0.1489 | 0.1489 | 0.0000 | 0.0000 | 0.0000 |
| `exact_display_name_only` | `ood_platform_test` | 0.1489 | 0.5745 | 0.6667 | 1.0000 | 0.5000 |
| `exact_username_only` | `ood_platform_test` | 0.1489 | 0.6960 | 0.7826 | 1.0000 | 0.6429 |
| `has_bio_only` | `ood_platform_test` | 0.1489 | 0.1220 | 0.0000 | 0.0000 | 0.0000 |
- **Dataset Bias Flag**: platform_pair_only achieves PR-AUC=0.6156 vs random baseline=0.3673 on test

### 6.4 Per-Negative-Type Breakdown (`test` & `ood_platform_test`)

| Split | Negative Type | Hard Neg? | Model | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---:|---:|---:|---:|
| `test` | `diff_person_fallback` | no | `rule_baseline` | 0.9327 | 0.6825 | 1.0000 | 0.5181 |
| `test` | `diff_person_fallback` | no | `logistic_regression` | 0.9573 | 0.8333 | 0.9836 | 0.7229 |
| `test` | `diff_person_fallback` | no | `hist_gradient_boosting` | 0.9649 | 0.8356 | 0.9683 | 0.7349 |
| `test` | `diff_person_fallback` | no | `phase_c_neural` | 0.9366 | 0.7808 | 0.9048 | 0.6867 |
| `test` | `diff_person_fallback` | no | `phase_c_plus_gnn` | 0.9867 | 0.9202 | 0.9375 | 0.9036 |
| `test` | `diff_person_fallback` | no | `llm_field_by_field` | 0.9834 | 0.9091 | 0.9859 | 0.8434 |
| `test` | `diff_person_fallback` | no | `llm_raw_no_sim` | 0.9781 | 0.8874 | 0.9853 | 0.8072 |
| `test` | `similar_username_diff_person` | **YES (hard)** | `rule_baseline` | 0.8401 | 0.6466 | 0.8600 | 0.5181 |
| `test` | `similar_username_diff_person` | **YES (hard)** | `logistic_regression` | 0.9163 | 0.8219 | 0.9524 | 0.7229 |
| `test` | `similar_username_diff_person` | **YES (hard)** | `hist_gradient_boosting` | 0.9675 | 0.8472 | 1.0000 | 0.7349 |
| `test` | `similar_username_diff_person` | **YES (hard)** | `phase_c_neural` | 0.9017 | 0.7600 | 0.8507 | 0.6867 |
| `test` | `similar_username_diff_person` | **YES (hard)** | `phase_c_plus_gnn` | 0.9671 | 0.9036 | 0.9036 | 0.9036 |
| `test` | `similar_username_diff_person` | **YES (hard)** | `llm_field_by_field` | 0.9627 | 0.8861 | 0.9333 | 0.8434 |
| `test` | `similar_username_diff_person` | **YES (hard)** | `llm_raw_no_sim` | 0.9653 | 0.8758 | 0.9571 | 0.8072 |
| `test` | `same_name_diff_org_city` | **YES (hard)** | `rule_baseline` | 0.8735 | 0.6615 | 0.9149 | 0.5181 |
| `test` | `same_name_diff_org_city` | **YES (hard)** | `logistic_regression` | 0.9380 | 0.8219 | 0.9524 | 0.7229 |
| `test` | `same_name_diff_org_city` | **YES (hard)** | `hist_gradient_boosting` | 0.9575 | 0.8414 | 0.9839 | 0.7349 |
| `test` | `same_name_diff_org_city` | **YES (hard)** | `phase_c_neural` | 0.9412 | 0.7917 | 0.9344 | 0.6867 |
| `test` | `same_name_diff_org_city` | **YES (hard)** | `phase_c_plus_gnn` | 0.9827 | 0.9259 | 0.9494 | 0.9036 |
| `test` | `same_name_diff_org_city` | **YES (hard)** | `llm_field_by_field` | 0.9761 | 0.9032 | 0.9722 | 0.8434 |
| `test` | `same_name_diff_org_city` | **YES (hard)** | `llm_raw_no_sim` | 0.9740 | 0.8933 | 1.0000 | 0.8072 |
| `test` | `same_display_name_diff_links` | **YES (hard)** | `rule_baseline` | 0.9639 | 0.6772 | 0.9773 | 0.5181 |
| `test` | `same_display_name_diff_links` | **YES (hard)** | `logistic_regression` | 0.9819 | 0.8392 | 1.0000 | 0.7229 |
| `test` | `same_display_name_diff_links` | **YES (hard)** | `hist_gradient_boosting` | 0.9884 | 0.8472 | 1.0000 | 0.7349 |
| `test` | `same_display_name_diff_links` | **YES (hard)** | `phase_c_neural` | 0.9656 | 0.7862 | 0.9194 | 0.6867 |
| `test` | `same_display_name_diff_links` | **YES (hard)** | `phase_c_plus_gnn` | 0.9848 | 0.9317 | 0.9615 | 0.9036 |
| `test` | `same_display_name_diff_links` | **YES (hard)** | `llm_field_by_field` | 0.9911 | 0.9091 | 0.9859 | 0.8434 |
| `test` | `same_display_name_diff_links` | **YES (hard)** | `llm_raw_no_sim` | 0.9899 | 0.8933 | 1.0000 | 0.8072 |
| `ood_platform_test` | `diff_person_fallback` | no | `rule_baseline` | 0.9236 | 0.8333 | 1.0000 | 0.7143 |
| `ood_platform_test` | `diff_person_fallback` | no | `logistic_regression` | 0.9344 | 0.9231 | 1.0000 | 0.8571 |
| `ood_platform_test` | `diff_person_fallback` | no | `hist_gradient_boosting` | 0.8964 | 0.9231 | 1.0000 | 0.8571 |
| `ood_platform_test` | `diff_person_fallback` | no | `phase_c_neural` | 0.9209 | 0.8387 | 0.7647 | 0.9286 |
| `ood_platform_test` | `diff_person_fallback` | no | `phase_c_plus_gnn` | 0.9458 | 0.6190 | 0.4643 | 0.9286 |
| `ood_platform_test` | `diff_person_fallback` | no | `llm_field_by_field` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `ood_platform_test` | `diff_person_fallback` | no | `llm_raw_no_sim` | 0.9874 | 0.9231 | 1.0000 | 0.8571 |
| `ood_platform_test` | `similar_username_diff_person` | **YES (hard)** | `rule_baseline` | 0.8904 | 0.8000 | 0.9091 | 0.7143 |
| `ood_platform_test` | `similar_username_diff_person` | **YES (hard)** | `logistic_regression` | 0.9169 | 0.8889 | 0.9231 | 0.8571 |
| `ood_platform_test` | `similar_username_diff_person` | **YES (hard)** | `hist_gradient_boosting` | 0.9127 | 0.9231 | 1.0000 | 0.8571 |
| `ood_platform_test` | `similar_username_diff_person` | **YES (hard)** | `phase_c_neural` | 0.9095 | 0.7879 | 0.6842 | 0.9286 |
| `ood_platform_test` | `similar_username_diff_person` | **YES (hard)** | `phase_c_plus_gnn` | 0.9394 | 0.6842 | 0.5417 | 0.9286 |
| `ood_platform_test` | `similar_username_diff_person` | **YES (hard)** | `llm_field_by_field` | 0.9952 | 0.9655 | 0.9333 | 1.0000 |
| `ood_platform_test` | `similar_username_diff_person` | **YES (hard)** | `llm_raw_no_sim` | 0.9812 | 0.9231 | 1.0000 | 0.8571 |
| `ood_platform_test` | `same_name_diff_org_city` | **YES (hard)** | `rule_baseline` | 0.8953 | 0.8000 | 0.9091 | 0.7143 |
| `ood_platform_test` | `same_name_diff_org_city` | **YES (hard)** | `logistic_regression` | 0.9299 | 0.8889 | 0.9231 | 0.8571 |
| `ood_platform_test` | `same_name_diff_org_city` | **YES (hard)** | `hist_gradient_boosting` | 0.8963 | 0.8889 | 0.9231 | 0.8571 |
| `ood_platform_test` | `same_name_diff_org_city` | **YES (hard)** | `phase_c_neural` | 0.9183 | 0.8387 | 0.7647 | 0.9286 |
| `ood_platform_test` | `same_name_diff_org_city` | **YES (hard)** | `phase_c_plus_gnn` | 0.9570 | 0.7647 | 0.6500 | 0.9286 |
| `ood_platform_test` | `same_name_diff_org_city` | **YES (hard)** | `llm_field_by_field` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `ood_platform_test` | `same_name_diff_org_city` | **YES (hard)** | `llm_raw_no_sim` | 0.9952 | 0.9231 | 1.0000 | 0.8571 |

### 6.5 Val-Tuned Chain-Confidence Baseline
- Val-tuned threshold: `0.01`
- `test`: PR-AUC=0.4569, F1=0.4000, mean_pos=0.0499, mean_neg=0.0232, pct_below_0.1=0.9204
- `ood_platform_test`: PR-AUC=0.2448, F1=0.2326, mean_pos=0.1351, mean_neg=0.0624, pct_below_0.1=0.8511
- Explanation: Multiplicative confidence decay (`prod(score * 0.85)`) compresses ~87-92% of multi-hop scores below `0.1`, collapsing threshold separability.

### 6.6 Zero-Shot LLM Judge (2 Prompt Variants)
- Status: `ok` (provider: `g4f/GithubCopilot/gpt-4.1 (+EryzaLabs)`)
- `minimal` (val threshold=`0.18`): test PR-AUC=0.9249 [0.8822, 0.9644], F1=0.8344 [0.7651, 0.9000], ECE=0.1135 [0.0885, 0.1470] (isotonic: 0.0806) | OOD PR-AUC=0.9911 [0.9429, 1.0000], F1=0.8966 [0.7273, 1.0000], ECE=0.0670 [0.0521, 0.0995] (isotonic: 0.0363) | tokens=61468, cost=$0.0, latency=9738.6852 ms/pair
- `field_by_field` (val threshold=`0.17`): test PR-AUC=0.9380 [0.8973, 0.9718], F1=0.8642 [0.7974, 0.9241], ECE=0.1292 [0.1073, 0.1646] (isotonic: 0.0644) | OOD PR-AUC=0.9952 [0.9792, 1.0000], F1=0.9655 [0.8966, 1.0000], ECE=0.0712 [0.0526, 0.0978] (isotonic: 0.0292) | tokens=72889, cost=$0.0, latency=6486.8105 ms/pair
- `raw_no_sim` (val threshold=`0.24`): test PR-AUC=0.9385 [0.9064, 0.9695], F1=0.8701 [0.8025, 0.9277], ECE=0.0815 [0.0618, 0.1141] (isotonic: 0.0589) | OOD PR-AUC=0.9720 [0.8968, 1.0000], F1=0.9231 [0.8000, 1.0000], ECE=0.0462 [0.0315, 0.0751] (isotonic: 0.0242) | tokens=326305, cost=$0.0, latency=9887.3253 ms/pair

| Split | Comparison (GNN - LLM) | ΔPR-AUC (95% CI) | PR-AUC Sig. | ΔF1 (95% CI) | F1 Sig. |
|---|---|---:|---|---:|---|
| `test` | `gnn_vs_llm_minimal` | +0.0108 [-0.0367, +0.0611] | not significant | +0.0109 [-0.0742, +0.0921] | not significant |
| `test` | `gnn_vs_llm_field_by_field` | -0.0021 [-0.0430, +0.0432] | not significant | -0.0198 [-0.0898, +0.0559] | not significant |
| `test` | `gnn_vs_llm_raw_no_sim` | -0.0025 [-0.0396, +0.0359] | not significant | -0.0250 [-0.0948, +0.0489] | not significant |
| `ood_platform_test` | `gnn_vs_llm_minimal` | -0.0874 [-0.2417, +0.0202] | not significant | -0.4610 [-0.6876, -0.2242] | significant |
| `ood_platform_test` | `gnn_vs_llm_field_by_field` | -0.0949 [-0.2473, +0.0000] | not significant | -0.5281 [-0.7028, -0.3422] | significant |
| `ood_platform_test` | `gnn_vs_llm_raw_no_sim` | -0.0703 [-0.2118, +0.0087] | not significant | -0.4830 [-0.6853, -0.2825] | significant |

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
