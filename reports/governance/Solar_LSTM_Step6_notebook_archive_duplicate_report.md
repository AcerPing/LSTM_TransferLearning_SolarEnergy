# Solar LSTM Step 6 Notebook / Archive Duplicate Decision Report

Generated UTC: 2026-09-16T08:40:04.8778669Z

## Governance boundary

- Branch: `AcerPing`
- HEAD: `c60bbb20e1ad4646251a0c9edaeeb67d4ac39666`
- Analysis mode: read-only inspection plus the two explicitly authorized Step 6 report outputs.
- Existing artifacts changed, moved, renamed, extracted, executed, or deleted: **No**.
- Git add / commit / push: **No**.
- Notebook execution, archive extraction/repack, training, prediction, evaluation, A2, Target Test: **No**.
- Protected Step 5B inventory SHA256: `1302428fbebc6dd5fca70dd2d9d6d7d68f64b2fb4cd84ded05a1b4198cc86328` (approved value retained).

## Scope definition

- Included: all `.ipynb`; files in `notebook/`, `notebooks/`, or `.ipynb_checkpoints/`; archive-like extensions (`.zip`, `.7z`, `.rar`, `.tar`, `.tar.gz`, `.tgz`, `.bak`, `.backup`); and exact English archive/backup/snapshot/export path segments.
- Excluded: `.git`, `desktop.ini`, the protected Step 5B checkpoint inventory, and these Step 6 outputs.
- Exact duplicates are established only by full lowercase SHA256 equality. Similar names and related container contents are not treated as exact duplicates.

## Inventory summary

- In-scope files: **79**
- Total bytes: **130362257**
- Unique SHA256 values: **70**
- Exact duplicate groups: **7**
- Files participating in exact duplicate groups: **16**
- Decision CSV SHA256: `ed29cc04e4b1683a9ea8823c380b41308d1d7d479813fd6e5d482dd261531e3d`

| File family | Count | Bytes |
|---|---:|---:|
| ARCHIVE | 4 | 114464877 |
| NOTEBOOK | 13 | 13971081 |
| NOTEBOOK_EXPORT | 1 | 94993 |
| OTHER_IN_SCOPE | 61 | 1831306 |

## Disposition summary

| Recommended disposition | Count | Bytes |
|---|---:|---:|
| DUPLICATE_CANDIDATE_PENDING_HUMAN_APPROVAL | 3 | 3576870 |
| KEEP_EXTERNAL_ARCHIVE_ONLY | 1 | 111980488 |
| KEEP_IN_PROJECT | 10 | 9265035 |
| KEEP_IN_REPOSITORY_ARCHIVE_ONLY | 52 | 1971537 |
| NEEDS_REVIEW | 13 | 3568327 |

## Exact duplicate groups

| Group | SHA256 | Files | Canonical rule | Decision note |
|---|---|---:|---|---|
| D001 | `0622104cd991681432cb340160e1391187022aae89ebfdf95efd283401e041f7` | 2 | reports/已條狀圖來表達模型性能.ipynb | Canonical retained; byte-identical alternate remains pending human approval. |
| D002 | `07560ab0de6de47503dee321e1e52be31cb6140b43f064c2003534f79a722be1` | 2 | NOT_APPLICABLE_DISTINCT_LOGICAL_ROLES | Retain all candidate-role records; identical bytes do not collapse distinct formal roles. |
| D003 | `0d1a298d4d9941d59fccb07ccd19589c69187375f3576b5d0962ebaf3fa747d8` | 2 | NOT_APPLICABLE_DISTINCT_LOGICAL_ROLES | Retain all candidate-role records; identical bytes do not collapse distinct formal roles. |
| D004 | `24e7a91e8f96415b45d0cf180748f4c11420b7fe91f2e38440f4f27b844ef2c6` | 2 | preprocess/Plant2第二號發電機組/Plant2 LSTM Prediction (Combine Generation & Weather).ipynb | Canonical active notebook retained; byte-identical checkpoint copy remains pending human approval. |
| D005 | `4586c07df15b4b984bfdda3507063d70f4f7b2147badd4919bc8c62d0a44c434` | 2 | NOT_APPLICABLE_DISTINCT_LOGICAL_ROLES | Retain all candidate-role records; identical bytes do not collapse distinct formal roles. |
| D006 | `4ef606aac5878ec063024a37f3f51be8213952999da63b0e88fbc98f5279b1a4` | 4 | NOT_APPLICABLE_DISTINCT_LOGICAL_ROLES | Retain all candidate-role records; identical bytes do not collapse distinct formal roles. |
| D007 | `a04771ba2e5bfc8afcc0a52b778f0dc245602d7edc0a1f2f4d553ae09ec23071` | 2 | notebook/修改紀錄/README_Solar_v1_20260817.md | Canonical retained; byte-identical alternate remains pending human approval. |

The two Phase IV B audit containers (`.zip` and `.tar.gz`) each list 38 file members, but their container SHA256 values differ. They are related archive representations, not exact-SHA duplicates; neither was extracted or repacked.

## Items requiring review

| Path | Family | SHA256 | Reason |
|---|---|---|---|
| .ipynb_checkpoints/將資料集轉檔成pkl檔案-checkpoint.ipynb | NOTEBOOK | `e7d6b416bf971ff940f43b77f437891fc16c8deaa3f580e7a6580880c8b51c53` | Notebook autosave/checkpoint-area artifact has no exact-SHA active counterpart in scope; provenance value versus generated-state status requires human review. |
| .ipynb_checkpoints/讀取pkl檔案&原始資料-checkpoint.ipynb | NOTEBOOK | `ac182ca74918442ed8d8f2b1c9d989452427715c94e93e0c8fa2d628496f4862` | Notebook autosave/checkpoint-area artifact has no exact-SHA active counterpart in scope; provenance value versus generated-state status requires human review. |
| notebook/__pycache__/__init__.cpython-37.pyc | OTHER_IN_SCOPE | `b7532014dfbf6a1640b1f1a8cb027f00b9833c1a36dc8d0495270cc8e4529339` | Tracked generated Python bytecode is located in the notebook area; source/reproducibility value is unclear without a human retention decision. |
| notebook/__pycache__/bagging.cpython-37.pyc | OTHER_IN_SCOPE | `12a8c92e79474703b671010692b14c195de3e7c0b6929f5408c93f214f6130ac` | Tracked generated Python bytecode is located in the notebook area; source/reproducibility value is unclear without a human retention decision. |
| notebook/__pycache__/util.cpython-37.pyc | OTHER_IN_SCOPE | `26b4835fc8c7beb3a8d374cf7be10b6cf4e8fe62d4698c2f955bc5c433d2978c` | Tracked generated Python bytecode is located in the notebook area; source/reproducibility value is unclear without a human retention decision. |
| notebook/.ipynb_checkpoints/NoteForResult&Output-checkpoint.ipynb | NOTEBOOK | `c9499fbb8387aec72cbfccfd870756a82f6e23bb31781a716f2591a6fcdbad04` | Notebook autosave/checkpoint-area artifact has no exact-SHA active counterpart in scope; provenance value versus generated-state status requires human review. |
| notebook/.ipynb_checkpoints/NoteForUtil-checkpoint.ipynb | NOTEBOOK | `22b7331593bfd87805659dbd0dde09faddff572b524630d4b9c55a95e3db1b28` | Notebook autosave/checkpoint-area artifact has no exact-SHA active counterpart in scope; provenance value versus generated-state status requires human review. |
| notebook/.ipynb_checkpoints/result-checkpoint.ipynb | NOTEBOOK | `84513b802f4ff394bbe6a1690a1b370852d3c8ebb70c7bac2c089713b4c306e0` | Notebook autosave/checkpoint-area artifact has no exact-SHA active counterpart in scope; provenance value versus generated-state status requires human review. |
| notebook/.ipynb_checkpoints/wavelet-checkpoint.ipynb | NOTEBOOK | `0f1a0776655cd14213d89511683476048e83ddf266cada61d53795ec0798904f` | Notebook autosave/checkpoint-area artifact has no exact-SHA active counterpart in scope; provenance value versus generated-state status requires human review. |
| preprocess/.ipynb_checkpoints/資料切分-checkpoint.png | OTHER_IN_SCOPE | `f244d7822ed3f7bc392dc155c46cfea33a8eb40dab5a9db39bb0a1b6167c082a` | Notebook autosave/checkpoint-area artifact has no exact-SHA active counterpart in scope; provenance value versus generated-state status requires human review. |
| preprocess/.ipynb_checkpoints/觀察各池塘資料量(資料處理後)-checkpoint.ipynb | NOTEBOOK | `2ebd8acb3cfa15e178af569bc569008c578ae2ac91085f2fab9de252f7daeaa1` | Notebook autosave/checkpoint-area artifact has no exact-SHA active counterpart in scope; provenance value versus generated-state status requires human review. |
| preprocess/Plant1第一號發電機組/.ipynb_checkpoints/Plant_1_Weather_Sensor_Data-checkpoint.csv | OTHER_IN_SCOPE | `91325041328745c288cb809aebd8b1b59cde54284500902ead1fb393499d6806` | Notebook autosave/checkpoint-area artifact has no exact-SHA active counterpart in scope; provenance value versus generated-state status requires human review. |
| preprocess/Plant1第一號發電機組/.ipynb_checkpoints/Plant1 LSTM Prediction (Combine Generation & Weather)-checkpoint.ipynb | NOTEBOOK | `a943fb1a6275247a0c4aa9dfea5af4edfa3bcb21000647c438875a44455b4709` | Notebook autosave/checkpoint-area artifact has no exact-SHA active counterpart in scope; provenance value versus generated-state status requires human review. |

## Duplicate candidates pending human approval

| Path | Group | Canonical candidate |
|---|---|---|
| notebook/修改紀錄/README_20260817.md | D007 | notebook/修改紀錄/README_Solar_v1_20260817.md |
| preprocess/Plant2第二號發電機組/.ipynb_checkpoints/Plant2 LSTM Prediction (Combine Generation & Weather)-checkpoint.ipynb | D004 | preprocess/Plant2第二號發電機組/Plant2 LSTM Prediction (Combine Generation & Weather).ipynb |
| reports/.ipynb_checkpoints/已條狀圖來表達模型性能-checkpoint.ipynb | D001 | reports/已條狀圖來表達模型性能.ipynb |

## Formal-reference findings

- In-scope paths directly preserved in the sealed B2 run manifest's `untracked_paths`: **44**.
- These include the Phase IV B audit archives/snapshot records and `README_Solar_v1_20260817.md`; this provenance was used only to prioritize retention/canonical candidacy, never to modify source evidence.
- Historical project-file manifests reference both `preprocess/Solar Power Generation Data.zip` and `transfer-learning-LSTM-master.zip`; their active-versus-external recommendations also consider tracked/ignored state and working-tree role.

## Ambiguities and human decisions

- Autosave/checkpoint-area files without an exact active counterpart remain `NEEDS_REVIEW`; filename similarity alone was not used as proof of redundancy.
- Tracked `__pycache__` bytecode remains `NEEDS_REVIEW` because it is generated state but currently part of repository history.
- Exact duplicate autosaves and the generic historical README copy are only candidates; Step 6 does not authorize deletion, relocation, or Git changes.
- Candidate-specific formal audit files with identical content remain retained under every logical candidate path; no canonical collapse is recommended.

## Output governance

- Decision CSV: `reports/governance/Solar_LSTM_Step6_notebook_archive_duplicate_decisions.csv`
- This report: `reports/governance/Solar_LSTM_Step6_notebook_archive_duplicate_report.md`
- Intended state after creation: untracked, unstaged, uncommitted.
- No cleanup, archive transfer, Step 7, push, A2 execution, training, prediction, evaluation, or Target Test action is authorized by this map.

## Verdict

`STEP_6_DECISION_MAP_COMPLETE`
