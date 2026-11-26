# Phase C Ablation Table

| Variant | Split | PR-AUC | F1 | ECE |
|---|---|---:|---:|---:|
| `full_model` | `test` | 0.8216 | 0.6909 | 0.0875 |
| `full_model` | `ood_platform_test` | 0.8012 | 0.7257 | 0.1426 |
| `no_username_char_cnn` | `test` | 0.8295 | 0.6797 | 0.0710 |
| `no_username_char_cnn` | `ood_platform_test` | 0.8151 | 0.7193 | 0.0630 |
| `no_bio_location` | `test` | 0.8849 | 0.7712 | 0.0528 |
| `no_bio_location` | `ood_platform_test` | 0.8568 | 0.7358 | 0.1148 |
| `no_photo_clip_phash` | `test` | 0.8367 | 0.7355 | 0.0885 |
| `no_photo_clip_phash` | `ood_platform_test` | 0.7569 | 0.5217 | 0.1793 |
| `no_links_overlap` | `test` | 0.8243 | 0.7013 | 0.1098 |
| `no_links_overlap` | `ood_platform_test` | 0.7812 | 0.6038 | 0.1360 |
| `no_contrastive_pretrain` | `test` | 0.8997 | 0.7735 | 0.1032 |
| `no_contrastive_pretrain` | `ood_platform_test` | 0.7742 | 0.7419 | 0.0971 |
| `no_hard_negatives` | `test` | 0.7947 | 0.6767 | 0.0983 |
| `no_hard_negatives` | `ood_platform_test` | 0.7600 | 0.6067 | 0.1454 |
