# Experiment A Result Provenance Audit

## Audit identity

| Field | Value |
|---|---|
| Audit version | 1.0 |
| Audit date | 2026-10-05 |
| Experiment | A2 — Linear Formal |
| Direction | Plant1 → Plant2 |
| Run ID | `20260918T051937Z_seed1234` |
| Audit mode | Documentation-only; no model/Test operations |
| Overall result | PASS WITH GOVERNANCE NOTES |

## Evidence boundary

This audit validates existing artifact identity, schema, lifecycle relationships, paths, and SHA-256 values. It does not perform prediction, evaluation, inverse transformation, metric recomputation, raw Target Test inspection, or checkpoint loading.

Historical Corrected R2/R2.5 evidence is outside the A2 result calculation and is retained only for separation and disclosure.

## Control results

| Control | Result | Evidence | Finding |
|---|---|---|---|
| Experiment identity | PASS | `protocol_manifest.json`, `final_state.json` | A2, Plant1→Plant2, run ID consistent |
| Protocol identity | PASS | `protocol_manifest.json` | `solar-linear-v1.0`, Linear activation, registered candidate grid present |
| Corrected R1 split contract | PASS | `protocol_manifest.json` | Source 2083/518/648; Target 516/126/2607 sequences |
| Feature/window/horizon contract | PASS | `protocol_manifest.json` | Five ordered features, window 5, horizon 1 |
| Formal candidate selection | PASS | `source_dependency_selection.json`, `selection.json` | Source `SRC_lr1e-4`, WOTL `WOTL_lr1e-4`, PFT `PFT_lr3e-5` |
| Validation-only selection | PASS | Selection records | `test_accessed=false`; `test_metrics_used_for_selection=false` |
| Checkpoint lock | PASS | Selection, best-checkpoint records, three `.hdf5` files | Recorded/current SHA matches 3/3 |
| Test authorization | PASS | `authorization.json` | Locked comparison authorized before access |
| Test access lifecycle | PASS | `access_log.json` | One access, completed, post-Test tuning prohibited |
| Final lifecycle | PASS | `final_state.json` | `TEST_COMPLETED`, access count 1, selection locked |
| Protocol lifecycle interpretation | PASS WITH NOTE | Protocol and final records | Early protocol remains pre-Test; final state is supplied by later immutable records |
| WOTL metric source | PASS | `wotl_metrics_original_scale.json` | Formal original-scale JSON exists and SHA matches DOC-A0 |
| Partial FT metric source | PASS | `partial_ft_metrics_original_scale.json` | Formal original-scale JSON exists and SHA matches DOC-A0 |
| Prediction artifact | PASS | `predictions.csv` | Existing schema/provenance recorded; no recomputation performed |
| Transfer judgment | PASS | `comparison.json` | Positive Transfer for the locked WOTL/PFT pair |
| Scaler identity | PASS WITH NOTE | Protocol scaler provenance and four external files | Recorded/current SHA matches 4/4; no run-local snapshots |
| Git provenance | PASS WITH NOTE | Protocol, selection, authorization, final state, current Git | Selection/execution/current HEAD chronology retained |
| Figure evidence | PASS WITH NOTE | Figure manifest and ten PNGs | Manifest SHA matches 10/10; pack is Git-untracked |
| Historical separation | PASS | Master/inventory evidence classification | No Historical R2/R2.5 metric is used as an A2 result |

## Lifecycle reconciliation

The lifecycle records are complementary rather than interchangeable:

| Sequence | Artifact | Recorded state |
|---:|---|---|
| 1 | `protocol_manifest.json` | `selection_locked_awaiting_test_authorization`; Test not accessed |
| 2 | `source_dependency_selection.json` | Source validation-only selection locked |
| 3 | `selection/selection.json` | Source/WOTL/PFT selection and checkpoint pair locked |
| 4 | `final/authorization.json` | `TEST_AUTHORIZED` |
| 5 | `final/access_log.json` | One Test access; completed |
| 6 | `final/final_state.json` | `TEST_COMPLETED` |

The protocol manifest was not edited to reflect later lifecycle events. Treating it alone as the final run state would be incorrect.

## Selection and checkpoint evidence

| Candidate | Role | Best epoch | Recorded/current checkpoint SHA-256 | Verification |
|---|---|---:|---|---|
| `SRC_lr1e-4` | Source | 500 | `6bbd2adebdc1652cf47665a97f7339466b13894df71270e93c397dc8ab1538c9` | MATCH |
| `WOTL_lr1e-4` | Without TL | 500 | `230ccb26aebb088ccb855ccdeaddc4f88f5a58f406a91632ce07b7ffda8a09d8` | MATCH |
| `PFT_lr3e-5` | Partial FT | 500 | `c95ce2c964cf75a65f24669a03c74bb43e1c212cd49040ffd516d0b3082e1a5b` | MATCH |

Selection Git HEAD: `51c496e40750591fefc007d972fcf5c34dbc30bc`.

Final-Test execution Git HEAD: `f6fe6821735a5356dd38617774edaf82c196671a`.

Repository HEAD at documentation preflight: `c2bc8b50761c9a69ffd62ef55459f0918f65ca10`.

## Original-scale result provenance

The formal result values are copied directly from the existing metrics JSON files.

| Method | Candidate | Metrics source | SHA-256 | n | MAE | MSE | RMSE | R² |
|---|---|---|---|---:|---:|---:|---:|---:|
| Without TL | `WOTL_lr1e-4` | `final/wotl_metrics_original_scale.json` | `0d8a125934d622de8e8a253384c254e44089008a701c708db968d41626640bfd` | 2607 | 2908.859887498984 | 12556225.83245452 | 3543.4765178359116 | 0.6858374551041306 |
| Partial FT | `PFT_lr3e-5` | `final/partial_ft_metrics_original_scale.json` | `cc292394519b7aa114ffd94f8a1c97c8989f210b107d7b788ff93eeeb8df17c2` | 2607 | 1601.8604031239606 | 6617057.209112146 | 2572.3641284064247 | 0.8344381854646958 |

The scale and unit fields in both sources are `original` and `kW`; MSE is documented as kW² in this package. No numerical values were recalculated from `predictions.csv`.

## Scaler evidence

The four external Corrected R1 scaler files were checked by file identity only. Current SHA matches protocol-recorded SHA for all four. They remain external dependencies:

- Plant1 source feature: `6a66df27dc1358cd4890e632f09fec848edb3c571cfee36f2e141986d4159912`
- Plant1 source target: `a9c0539b97e0faff07eb04e032b65f79dd45be5d0a6927c3b48fc67e4819f77e`
- Plant2 target feature: `d91ce120e26e45b540017859bf419ec35010927492cb7be2b02f275e87077d4d`
- Plant2 target target: `6db35ad6a42939b687a7b0593920fb5a539f87c54de590f64daa335533c3680c`

## Governance notes

1. The A2 formal run root is locally available but Git-untracked.
2. The Figure Pack is locally available but Git-untracked.
3. Scalers are referenced external evidence, not run-local snapshots.
4. No standalone A2 `params.json` exists; parameters are embedded in formal manifests.
5. The protocol manifest records the pre-Test lifecycle stage by design/history; the final lifecycle requires the later authorization/access/final records.
6. Plant2 Test had historical exposure before A2, so A2 is a controlled follow-up rather than an untouched confirmatory Test.
7. The result is single-seed, single-split, and single-Test-interval evidence.

## Audit conclusion

The existing evidence supports the accepted A2 status and Positive Transfer classification within the declared limitations. Governance notes do not alter the recorded metrics or lifecycle, but they must accompany repository/thesis use until retention and external-scaler handling receive separate authorization.

