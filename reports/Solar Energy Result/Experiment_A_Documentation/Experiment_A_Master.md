# Experiment A Master Record

## Document control

| Field | Value |
|---|---|
| Document version | 1.0 |
| Documentation date | 2026-10-05 |
| Experiment | A2 — Linear Formal |
| Direction | Plant1 → Plant2 |
| Source / Target | Plant1 / Plant2 |
| Formal run ID | `20260918T051937Z_seed1234` |
| Formal status | COMPLETED / ACCEPTED / CLOSED |
| Transfer judgment | Positive Transfer |
| Protocol | `solar-linear-v1.0` |

This record is the documentation entry point for Linear Formal Experiment A2. It does not replace or rewrite Historical Experiment A Corrected R2/R2.5. Formal measurements are cited from the existing run artifacts; this document does not contain new prediction, evaluation, inverse transformation, or Test recomputation.

## Evidence authority

The documentation package applies this precedence:

1. Formal JSON, CSV, checkpoint files, and their recorded SHA-256 values.
2. Validation selection, Test authorization, Test access, and final-state records.
3. Figure manifest and its existing PNG files.
4. Repository summaries and protocol documentation.
5. Historical and Legacy documents.

A thesis or exported report is a presentation surface, not a replacement for the formal experimental artifacts.

## Experiment scope

### Linear Formal A2

- Experiment ID: `A2`
- Direction: `Plant1_to_Plant2`
- Source: Plant1 `source_profile`
- Target: Plant2 `target_profile`
- Output activation: Linear
- Seed: 1234
- Device policy: CPU, deterministic operations enabled
- Target variable: `DC_POWER`
- Unit label recorded by the formal artifacts: `kW`

### Historical Experiment A

Historical Corrected R2/R2.5 used a different lifecycle and includes earlier Sigmoid and partial-fine-tuning evidence. Those artifacts remain historical evidence. Their metrics, checkpoints, activation settings, and transfer classifications are not A2 results and are not used to populate the A2 formal result tables below.

## Corrected R1 data contract

The formal protocol records five ordered input features:

1. `TIME_SIN`
2. `TIME_COS`
3. `IRRADIATION`
4. `AMBIENT_TEMPERATURE`
5. `MODULE_TEMPERATURE`

Window and horizon are `5` and `1`.

| Profile | Split | Rows | Sequences |
|---|---|---:|---:|
| Plant1 source | Training | 2,088 | 2,083 |
| Plant1 source | Validation | 523 | 518 |
| Plant1 source | Test | 653 | 648 |
| Plant2 target | Training | 521 | 516 |
| Plant2 target | Validation | 131 | 126 |
| Plant2 target | Test | 2,612 | 2,607 |

The accepted project state describes the Corrected R1 Data Handoff as `READY WITH NOTES`. DOC-A0 did not locate a standalone handoff report in the repository; this is retained as a governance note rather than silently promoted to repository-verified documentation.

## Scaler provenance

The A2 run does not contain local `.joblib` scaler snapshots. It records four external Corrected R1 scaler dependencies. DOC-A2A preflight confirmed that all four files exist and their current SHA-256 values match the values recorded in `protocol_manifest.json`.

| Role | Fit split / rows | SHA-256 | Location |
|---|---|---|---|
| Plant1 source feature scaler | Training / 2,088 | `6a66df27dc1358cd4890e632f09fec848edb3c571cfee36f2e141986d4159912` | `D:\HoChePing\北科大_碩班_AI學程\期刊研究\使用LSTM模型預測福壽雞隻重量\Code程式碼\PersonalNote\DataSet_For_TransferLearning\★★ Solar Power Generation Data ★★\preprocessed_profiles_corrected_r1\plant1\source_profile\feature_scaler.joblib` |
| Plant1 source target scaler | Training / 2,088 | `a9c0539b97e0faff07eb04e032b65f79dd45be5d0a6927c3b48fc67e4819f77e` | `D:\HoChePing\北科大_碩班_AI學程\期刊研究\使用LSTM模型預測福壽雞隻重量\Code程式碼\PersonalNote\DataSet_For_TransferLearning\★★ Solar Power Generation Data ★★\preprocessed_profiles_corrected_r1\plant1\source_profile\target_scaler.joblib` |
| Plant2 target feature scaler | Training / 521 | `d91ce120e26e45b540017859bf419ec35010927492cb7be2b02f275e87077d4d` | `D:\HoChePing\北科大_碩班_AI學程\期刊研究\使用LSTM模型預測福壽雞隻重量\Code程式碼\PersonalNote\DataSet_For_TransferLearning\★★ Solar Power Generation Data ★★\preprocessed_profiles_corrected_r1\plant2\target_profile\feature_scaler.joblib` |
| Plant2 target target scaler | Training / 521 | `6db35ad6a42939b687a7b0593920fb5a539f87c54de590f64daa335533c3680c` | `D:\HoChePing\北科大_碩班_AI學程\期刊研究\使用LSTM模型預測福壽雞隻重量\Code程式碼\PersonalNote\DataSet_For_TransferLearning\★★ Solar Power Generation Data ★★\preprocessed_profiles_corrected_r1\plant2\target_profile\target_scaler.joblib` |

The scalers are external dependencies, not repository-local snapshots. No scaler was refit during documentation creation.

## Formal model and candidate contract

Common settings recorded in the protocol:

- Linear output activation
- Adam optimizer
- MSE loss
- Batch size 128
- Maximum epochs 500
- `shuffle=False`
- Seed 1234
- CPU deterministic policy

Candidate registry:

| Lifecycle | Registered learning rates |
|---|---|
| Source | `1e-4`, `3e-5` |
| WOTL | `1e-4`, `3e-5` |
| Partial FT | `1e-5`, `3e-5` |

Partial FT uses strategy `partial_target_adapters_last_lstm`: transferred layers `(2,3,4,5)`, trainable layers `(1,4,6)`, frozen layers `(2,3,5)`, and frozen BatchNormalization. Selection evidence records 29,161 trainable parameters for the Partial FT candidates and 46,681 trainable parameters for Source/WOTL candidates.

No standalone `params.json` exists for A2. Parameter provenance is embedded in `protocol_manifest.json` and the candidate manifests; this document does not create a substitute params artifact.

## Validation-only selection

Selection was completed before Target Test authorization. `selection.json` records `test_accessed=false`, `test_authorized=false`, and `test_metrics_used_for_selection=false`.

| Lifecycle | Selected candidate | Best epoch | Validation loss | Validation original RMSE | Validation original MAE | Validation original R² |
|---|---|---:|---:|---:|---:|---:|
| Source | `SRC_lr1e-4` | 500 | 0.059372227638959885 | 24570.781656305102 | 13856.28931989663 | 0.9208654575851452 |
| WOTL | `WOTL_lr1e-4` | 500 | 0.3029176890850067 | 5284.82588479134 | 4475.544781851323 | 0.303895712863277 |
| Partial FT | `PFT_lr3e-5` | 500 | 0.15588022768497467 | 3847.744058385386 | 2336.8989073716184 | 0.6310009110302114 |

The table is transcribed from the validation comparison objects in `selection/selection.json`; it is not a new evaluation.

### Locked checkpoints

| Candidate | Checkpoint | SHA-256 |
|---|---|---|
| `SRC_lr1e-4` | `source_candidates/SRC_lr1e-4/checkpoint_epoch_0500.hdf5` | `6bbd2adebdc1652cf47665a97f7339466b13894df71270e93c397dc8ab1538c9` |
| `WOTL_lr1e-4` | `target_wotl_candidates/WOTL_lr1e-4/checkpoint_epoch_0500.hdf5` | `230ccb26aebb088ccb855ccdeaddc4f88f5a58f406a91632ce07b7ffda8a09d8` |
| `PFT_lr3e-5` | `target_partial_ft_candidates/PFT_lr3e-5/checkpoint_epoch_0500.hdf5` | `c95ce2c964cf75a65f24669a03c74bb43e1c212cd49040ffd516d0b3082e1a5b` |

DOC-A2A preflight verified all three current checkpoint hashes against their locked values.

## Selection, authorization, and final-state lifecycle

The earlier protocol manifest is a contract and selection-stage record. It retains:

- `protocol_stage=selection_locked_awaiting_test_authorization`
- `final_test_executed=false`
- `test_accessed=false`

It is not treated as the final lifecycle state and was not modified.

The formal evidence chain is:

1. `protocol_manifest.json` — locked configuration and pre-Test stage.
2. `source_dependency_selection.json` — Source candidate selected using validation only.
3. `selection/selection.json` — Source/WOTL/PFT candidates and checkpoints locked at `2026-09-19T07:44:20Z`.
4. `final/authorization.json` — comparison pair authorized at `2026-09-20T02:52:47Z`, state `TEST_AUTHORIZED`.
5. `final/access_log.json` — one Test access at `2026-09-20T02:52:48Z`, completed, with post-Test tuning prohibited.
6. `final/final_state.json` — state `TEST_COMPLETED`, `test_access_count=1`, `selection_locked=true`, and `post_test_tuning_allowed=false`.

## Formal original-scale results

The following values are transcribed directly from the two existing original-scale metrics JSON files.

| Method | Candidate | n | MAE (kW) | MSE (kW²) | RMSE (kW) | R² | Negative predictions |
|---|---|---:|---:|---:|---:|---:|---:|
| Without TL | `WOTL_lr1e-4` | 2,607 | 2908.859887498984 | 12556225.83245452 | 3543.4765178359116 | 0.6858374551041306 | 1,508 |
| Partial FT | `PFT_lr3e-5` | 2,607 | 1601.8604031239606 | 6617057.209112146 | 2572.3641284064247 | 0.8344381854646958 | 683 |

Primary sources:

- `final/wotl_metrics_original_scale.json`
- `final/partial_ft_metrics_original_scale.json`
- `final/comparison.json`

`comparison.json` records `Positive Transfer`. The existing `final/predictions.csv` is the formal row-level artifact, but it was not read as raw Target Test evidence or used to recompute metrics during DOC-A2A.

## Figure Pack

The thesis-only Figure Pack is stored at:

`reports/Solar Energy Result/Thesis_Figures/Experiment_A_Plant1_to_Plant2/`

It contains ten PNG files: two learning curves, two prediction plots, two observed-versus-predicted plots, two residual plots, and two error histograms. `figure_manifest.json` identifies A/A2, Plant1→Plant2, the formal run, and candidates `WOTL_lr1e-4` and `PFT_lr3e-5`.

DOC-A2A preflight confirmed 10/10 current PNG SHA-256 values match the manifest. No PNG was regenerated. Figure semantics include:

- Learning curves: `Compiled Loss (training scale)`.
- Prediction and observed-versus-predicted plots: original scale.
- Residual/error definition: `True - Predicted`.
- Recorded target label/unit: `DC_POWER (kW)`.

The figures visualize existing evidence and do not replace the metrics JSON or prediction CSV.

## Transfer judgment

Experiment A2 is recorded as Positive Transfer for the locked comparison between `WOTL_lr1e-4` and `PFT_lr3e-5`. This conclusion applies to the single seed, single split, single Test interval, fixed candidate registry, and recorded Corrected R1 profiles.

The Plant2 Test interval had been exposed in earlier Historical Experiment A work. Therefore A2 is a controlled follow-up result, not an untouched confirmatory Test. No causal, cross-direction, cross-seed, or general population claim is made.

## Thesis cross-reference

| Thesis chapter | Material supplied by this package |
|---|---|
| Chapter 3 | Corrected R1 contract, scaler flow, Linear candidate registry, Partial FT strategy, validation-only selection, Test gate |
| Chapter 4 | Locked candidates/checkpoints, formal original-scale metrics, comparison classification, Figure Pack |
| Chapter 5 | Positive Transfer interpretation, Historical-versus-A2 distinction, historical Test exposure, single-run and governance limitations |

## Governance notes

- The formal A2 run root is locally present but Git-untracked.
- The Figure Pack and generation script are locally present but Git-untracked.
- The four scalers are external Corrected R1 dependencies, not run-local snapshots.
- The current repository HEAD is later than both the selection and final-Test execution Git heads; the recorded chronology is retained rather than rewritten.
- F0/F1/F2 are accepted as PASS by project governance, but DOC-A0 did not locate standalone repository files for all three audit narratives.
- Existing `README.md`, `results_summary.md`, and `experiment_protocol.md` were not changed.

## Closure

Within the accepted evidence scope, Linear Formal Experiment A2 is `COMPLETED / ACCEPTED / CLOSED`. This documentation package records existing evidence only. It does not authorize another Test access, rerun, parameter adjustment, or alteration of Historical Experiment A artifacts.

