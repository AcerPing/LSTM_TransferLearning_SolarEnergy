# Solar LSTM Step 6A Exact Duplicate Cleanup Report

Generated UTC: `2026-09-16T10:31:16.1109311Z`

## Execution identity

- Branch: `AcerPing`
- HEAD: `c60bbb20e1ad4646251a0c9edaeeb67d4ac39666`
- Human authorization scope: delete exactly the three SHA256-proven duplicate files specified by the Step 6A authorization; no general cleanup, wildcard deletion, directory deletion, archive cleanup, checkpoint cleanup, or `NEEDS_REVIEW` cleanup.
- Pre-delete tracked modifications: `0`
- Pre-delete staged files: `0`

## Authorized duplicate reconciliation

| Group | Authorized duplicate | Canonical copy | Pre-delete Git state | Pre-delete duplicate exists | Pre-delete canonical exists | Duplicate SHA256 | Canonical SHA256 | SHA equality | Deleted | Post-delete duplicate exists | Post-delete canonical exists | Post-delete canonical SHA verified |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D007 | `notebook/修改紀錄/README_20260817.md` | `notebook/修改紀錄/README_Solar_v1_20260817.md` | Duplicate and canonical both `UNTRACKED` | True | True | `a04771ba2e5bfc8afcc0a52b778f0dc245602d7edc0a1f2f4d553ae09ec23071` | `a04771ba2e5bfc8afcc0a52b778f0dc245602d7edc0a1f2f4d553ae09ec23071` | True | True | False | True | True |
| D004 | `preprocess/Plant2第二號發電機組/.ipynb_checkpoints/Plant2 LSTM Prediction (Combine Generation & Weather)-checkpoint.ipynb` | `preprocess/Plant2第二號發電機組/Plant2 LSTM Prediction (Combine Generation & Weather).ipynb` | Duplicate and canonical both `TRACKED_CLEAN` | True | True | `24e7a91e8f96415b45d0cf180748f4c11420b7fe91f2e38440f4f27b844ef2c6` | `24e7a91e8f96415b45d0cf180748f4c11420b7fe91f2e38440f4f27b844ef2c6` | True | True | False | True | True |
| D001 | `reports/.ipynb_checkpoints/已條狀圖來表達模型性能-checkpoint.ipynb` | `reports/已條狀圖來表達模型性能.ipynb` | Duplicate and canonical both `TRACKED_CLEAN` | True | True | `0622104cd991681432cb340160e1391187022aae89ebfdf95efd283401e041f7` | `0622104cd991681432cb340160e1391187022aae89ebfdf95efd283401e041f7` | True | True | False | True | True |

## Git boundary

Expected and observed unstaged tracked deletions:

```text
D preprocess/Plant2第二號發電機組/.ipynb_checkpoints/Plant2 LSTM Prediction (Combine Generation & Weather)-checkpoint.ipynb
D reports/.ipynb_checkpoints/已條狀圖來表達模型性能-checkpoint.ipynb
```

- Expected tracked deletions: `2`
- Observed tracked modifications: `2`
- Unexpected tracked changes: `0`
- Staged files: `0`
- The untracked D007 deletion created no tracked Git diff.

## Protected evidence verification

| Protected artifact | Required and observed SHA256 | Verified unchanged |
|---|---|---|
| `reports/governance/Solar_LSTM_checkpoint_archive_inventory.csv` | `1302428fbebc6dd5fca70dd2d9d6d7d68f64b2fb4cd84ded05a1b4198cc86328` | True |
| `reports/governance/Solar_LSTM_Step6_notebook_archive_duplicate_decisions.csv` | `ed29cc04e4b1683a9ea8823c380b41308d1d7d479813fd6e5d482dd261531e3d` | True |
| `reports/governance/Solar_LSTM_Step6_notebook_archive_duplicate_report.md` | `c98707dda276aa700380d2919863729325dbf7568018e88443052e278045ee6e` | True |

- Remaining Step 6 source artifacts verified against the immutable Step 6 decision CSV: `76 / 76`
- Protected `NEEDS_REVIEW` artifacts present: `13 / 13`
- `README.md`, `results_summary.md`, `EVIDENCE_RETENTION_POLICY.md`, and `experiment_protocol.md` modified: `False`

## Operation accounting

- Authorized files deleted: `3`
- Unauthorized files deleted: `0`
- Directories deleted: `0`
- Files moved: `0`
- Files renamed: `0`
- Archives extracted or rewritten: `0`
- Notebooks executed or rewritten: `0`
- `desktop.ini` deleted: `0`
- Git add: `False`
- Git commit: `False`
- Git push: `False`
- Training: `0`
- Prediction: `0`
- Evaluation: `0`
- A2 execution: `0`
- Target Test access: `False`

The cleanup report itself is intentionally left untracked, unstaged, and uncommitted. Its SHA256 is calculated after the final byte is written and returned with the Step 6A execution result.

## Verdict

`STEP_6A_EXACT_DUPLICATE_CLEANUP_COMPLETE`
