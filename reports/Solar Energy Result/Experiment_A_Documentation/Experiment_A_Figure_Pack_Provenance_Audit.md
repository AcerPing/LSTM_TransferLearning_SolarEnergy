# Experiment A Figure Pack Provenance Audit

## Audit identity

| Field | Value |
|---|---|
| Audit version | 1.0 |
| Audit date | 2026-10-05 |
| Experiment | A / internal experiment A2 |
| Direction | Plant1 → Plant2 |
| Run ID | `20260918T051937Z_seed1234` |
| Manifest | `reports/Solar Energy Result/Thesis_Figures/Experiment_A_Plant1_to_Plant2/figure_manifest.json` |
| Manifest SHA-256 | `32b0566695cc3269a6f48d455a4358c937d986da8900a02c8ca29c4f5afb5cee` |
| Audit result | PASS WITH GOVERNANCE NOTES |

This audit verifies existing file identity, manifest mapping, and recorded semantics. It does not regenerate figures, read raw Target Test observations, or recalculate metrics.

## Pack inventory

| # | Type | Candidate | Existing output | Size (bytes) | SHA-256 match |
|---:|---|---|---|---:|---|
| 1 | Learning curve | `WOTL_lr1e-4` | `learning_curve/wotl_learning_curve.png` | 171148 | PASS |
| 2 | Learning curve | `PFT_lr3e-5` | `learning_curve/pft_learning_curve.png` | 146282 | PASS |
| 3 | Prediction | `WOTL_lr1e-4` | `prediction/wotl_prediction_plot_original_scale.png` | 623686 | PASS |
| 4 | Prediction | `PFT_lr3e-5` | `prediction/pft_prediction_plot_original_scale.png` | 622785 | PASS |
| 5 | Observed vs. predicted | `WOTL_lr1e-4` | `yy_plot/wotl_yy_plot_original_scale.png` | 437963 | PASS |
| 6 | Observed vs. predicted | `PFT_lr3e-5` | `yy_plot/pft_yy_plot_original_scale.png` | 453091 | PASS |
| 7 | Residual | `WOTL_lr1e-4` | `residual/wotl_residual_plot_original_scale.png` | 438598 | PASS |
| 8 | Residual | `PFT_lr3e-5` | `residual/pft_residual_plot_original_scale.png` | 431896 | PASS |
| 9 | Error histogram | `WOTL_lr1e-4` | `histogram/wotl_error_histogram_original_scale.png` | 109741 | PASS |
| 10 | Error histogram | `PFT_lr3e-5` | `histogram/pft_error_histogram_original_scale.png` | 109777 | PASS |

Manifest-to-file verification: **10/10 SHA-256 matches; 0 missing; 0 mismatched**.

## Exact figure hashes

| Output | SHA-256 |
|---|---|
| `learning_curve/wotl_learning_curve.png` | `06d7a8d63aa636dc42c1a724ce01e7acc93f45be6d1516d5c38e6ed0c92b79bb` |
| `learning_curve/pft_learning_curve.png` | `0d197ad5b74600cd01c48b2c7ef3efd352f966bf1f818e4b8dd641dfabb4e9b5` |
| `prediction/wotl_prediction_plot_original_scale.png` | `777bf12a24d0fc3c428a30f72fd4f7839b1c393e792c02e74462c65f605b471c` |
| `prediction/pft_prediction_plot_original_scale.png` | `eee95cd2f95270156fd2669670e964f05f043259c7f271fbf3d8ee8b3fb5c460` |
| `yy_plot/wotl_yy_plot_original_scale.png` | `eac8420b426dc447cfc7d7531141cc02f906b4737753c40d07fae1fa86e121e1` |
| `yy_plot/pft_yy_plot_original_scale.png` | `9a835b0c45a05fcb24c47953fb70a0760f4ecce9f3326390e8ca3faea32785a1` |
| `residual/wotl_residual_plot_original_scale.png` | `395d2a72bf184a0946a4136322dc111a12696758f833ce701f73df79f30f0a04` |
| `residual/pft_residual_plot_original_scale.png` | `f6e8bbb31a5a0b84cbc090b3cd043a00403467aeb584721b050dc96a613d6a32` |
| `histogram/wotl_error_histogram_original_scale.png` | `424df4319653bed1163b2463b2e87251a98d3bba857411a5c82f89d4c63cf177` |
| `histogram/pft_error_histogram_original_scale.png` | `dcd5385d2ca22142a92306bbec868329840c2927b680833f504b0078cfd6a5b5` |

## Input binding

| Figure group | Formal input recorded by manifest |
|---|---|
| WOTL learning curve | `target_wotl_candidates/WOTL_lr1e-4/history.csv` |
| PFT learning curve | `target_partial_ft_candidates/PFT_lr3e-5/history.csv` |
| Prediction, Y–Y, residual, histogram | `final/predictions.csv` |

`predictions.csv` SHA-256 is `12fce7ec5852b428439a09c4fc3b2e3254a6345ef7948c17077b5e38bf08dcaa`. It was not reopened for row-level Test recomputation in DOC-A2A.

## Semantic review

| Control | Result | Manifest evidence |
|---|---|---|
| Experiment/run identity | PASS | A/A2, Plant1→Plant2, target Plant2, exact run ID |
| Candidate identity | PASS | `WOTL_lr1e-4` and `PFT_lr3e-5` only |
| Learning-curve scale | PASS | Y-axis `Compiled Loss (training scale)` |
| Prediction scale | PASS | Original scale, `DC_POWER (kW)` |
| Y–Y axes | PASS | Observed and predicted original-scale `DC_POWER` |
| Residual definition | PASS | `y_true_original - y_pred_original` |
| Histogram error definition | PASS | `True - Predicted` |
| Metrics-source boundary | PASS | Figures are visualization evidence; JSON remains the metrics authority |

## Governance notes

- The Figure Pack and manifest are locally present but Git-untracked.
- `tools/generate_solar_a_thesis_figures.py` is locally present but Git-untracked.
- F0/F1/F2 are accepted as PASS by project governance. A dedicated repository file containing the complete F2 narrative was not located during DOC-A0.
- This audit records existing figure semantics and hashes only. It does not claim that PNG pixels are a substitute for the formal metrics or prediction files.

## Conclusion

The existing Experiment A Figure Pack is internally bound to the formal A2 run and passes manifest/file SHA verification for all ten PNG files. Repository retention remains a separate governance action.

