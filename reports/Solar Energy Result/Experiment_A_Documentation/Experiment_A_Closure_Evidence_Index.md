# Experiment A Closure Evidence Index

## Document control

| Field | Value |
|---|---|
| Index version | 1.0 |
| Index date | 2026-10-05 |
| Experiment | A2 — Plant1 → Plant2 |
| Formal run | `20260918T051937Z_seed1234` |
| Accepted status | COMPLETED / ACCEPTED / CLOSED |
| Accepted judgment | Positive Transfer |

## Closure evidence map

| Closure topic | Status | Primary evidence | Documentation surface | Notes |
|---|---|---|---|---|
| Experiment identity | VERIFIED | `protocol_manifest.json`, `final_state.json` | Master, Result Audit | A2 distinct from Historical A |
| Corrected R1 contract | VERIFIED WITH NOTE | `protocol_manifest.json`, external scaler files | Master, Inventory | Handoff accepted READY WITH NOTES; standalone handoff report not found |
| Candidate registry | VERIFIED | `protocol_manifest.json` | Master, Result Audit | Six preregistered candidates |
| Source selection | VERIFIED | `source_dependency_selection.json` | Master, Result Audit | Validation-only |
| WOTL/PFT selection | VERIFIED | `selection/selection.json` | Master, Result Audit | Test not authorized/accessed at selection |
| Checkpoint lock | VERIFIED | Selection, best-checkpoint records, selected `.hdf5` | Inventory, Result Audit | 3/3 hashes match |
| Test authorization | VERIFIED | `final/authorization.json` | Master, Result Audit | Comparison pair locked before access |
| Test access | VERIFIED | `final/access_log.json` | Master, Result Audit | One completed access |
| Final lifecycle | VERIFIED | `final/final_state.json` | Master, Result Audit | `TEST_COMPLETED` |
| Formal metrics | VERIFIED | Two original-scale metrics JSON files | Master, Result Audit | No recomputation in documentation phase |
| Prediction artifact | VERIFIED | `final/predictions.csv` | Inventory, Result Audit | Existing artifact only |
| Transfer judgment | VERIFIED | `final/comparison.json` | Master, Result Audit | Positive Transfer |
| Scalers | VERIFIED WITH NOTE | Protocol-recorded external paths/SHA | Master, Inventory | 4/4 match; no run-local snapshots |
| Figure Pack | VERIFIED WITH NOTE | `figure_manifest.json`, ten PNGs | Figure Audit, Inventory | 10/10 match; untracked |
| F0/F1/F2 | HUMAN-CONFIRMED WITH NOTE | Accepted governance state | Figure Audit, this index | Full local audit narratives not all found |
| Historical separation | VERIFIED | Historical directories plus A2 identity | Master, Result Audit | Historical values not used for A2 |
| Repository retention | OPEN GOVERNANCE NOTE | Git status | All five documents | Formal A2 and Figure Pack remain untracked |

## Formal evidence paths

Run root:

`reports/Solar Energy Result/Linear_Formal/Experiment_A2/20260918T051937Z_seed1234/`

Core lifecycle files:

- `protocol_manifest.json`
- `source_dependency_selection.json`
- `selection/selection.json`
- `final/authorization.json`
- `final/access_log.json`
- `final/final_state.json`
- `final/comparison.json`
- `final/wotl_metrics_original_scale.json`
- `final/partial_ft_metrics_original_scale.json`
- `final/predictions.csv`

Figure root:

`reports/Solar Energy Result/Thesis_Figures/Experiment_A_Plant1_to_Plant2/`

## Git provenance chronology

| Event | Git HEAD |
|---|---|
| Training/selection evidence | `51c496e40750591fefc007d972fcf5c34dbc30bc` |
| Authorized Final-Test execution | `f6fe6821735a5356dd38617774edaf82c196671a` |
| DOC-A2A documentation preflight | `c2bc8b50761c9a69ffd62ef55459f0918f65ca10` |

These values record different lifecycle moments and are not normalized to the current HEAD.

## Documentation package index

| Document | Role |
|---|---|
| `Experiment_A_Master.md` | Canonical Experiment A/A2 narrative and thesis cross-reference |
| `Experiment_A_Final_Artifact_Inventory.csv` | Machine-readable artifact paths, hashes, roles, and governance state |
| `Experiment_A_Result_Provenance_Audit.md` | Contract, selection, authorization, final-state, metrics, and checkpoint audit |
| `Experiment_A_Figure_Pack_Provenance_Audit.md` | Figure input, semantics, path, and SHA audit |
| `Experiment_A_Closure_Evidence_Index.md` | Closure status and evidence navigation |

## Historical boundary

Historical Corrected R2/R2.5 remains sealed historical evidence. This closure package does not rename Historical runs as A2, copy their metrics into A2 tables, or replace their original manifests.

## Remaining governance items

1. Decide whether and how to retain the untracked A2 formal artifacts.
2. Decide whether and how to retain the untracked Figure Pack and generator.
3. Decide whether immutable external scaler snapshots are required in addition to path/SHA provenance.
4. Supply or formally waive standalone repository copies of the F0/F2 and Corrected R1 handoff narratives.
5. Resolve stale shared-document wording only under separate authorization.

## Closure statement

The evidence indexed here supports the accepted project decision that Linear Formal Experiment A2 is completed, accepted, and closed with a Positive Transfer classification. Open governance items concern retention and documentation completeness; they do not authorize changes to the recorded experiment or a new Test access.

