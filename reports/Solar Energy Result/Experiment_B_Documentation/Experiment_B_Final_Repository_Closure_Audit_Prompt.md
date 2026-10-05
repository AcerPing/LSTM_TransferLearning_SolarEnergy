# Experiment B Final Repository Closure Audit Prompt

## Project
`LSTM_TransferLearning_SolarEnergy`

## Audit target
**Experiment B: Plant2 -> Plant1**

This round is a **read-only repository inventory and archival-closure audit**. Scientific/model-development work is already sealed. The audit must NOT reopen tuning or Test-guided model development.

---

## 1. HARD RESTRICTIONS

This round is strictly READ-ONLY.

Do NOT:
- modify any source code, data, JSON, CSV, Markdown, notebook, manifest, checkpoint, prediction, metric, figure, README or protocol;
- run `git add`, `git commit`, `git push`, `git reset`, `git checkout`, `git clean`, `git rm`, or any operation that changes the worktree/index/history;
- train any model;
- call model prediction to regenerate results;
- re-run Final Test;
- re-select checkpoints;
- clip predictions or recompute a new "better" official result;
- move, rename, delete, overwrite or archive files;
- modify `.gitignore`;
- create LFS tracking rules.

Allowed:
- list/read files and folders;
- `git status`, `git ls-files`, `git check-ignore`, `git log`, `git rev-parse`, `git diff --stat`, `git diff --name-only`;
- compute SHA-256 in read-only mode;
- inspect JSON/CSV/MD/log/manifest/checkpoint metadata;
- compare documented hashes to physical files;
- report recommended actions without executing them.

STOP after the audit report. Do not perform any recommended remediation.

---

## 2. KNOWN SCIENTIFIC GOVERNANCE TO PRESERVE

### Formal Experiment B
- Direction: Plant2 -> Plant1
- Protocol: `solar-linear-v1.0`
- Run: `reports/Solar Energy Result/Linear_Formal/Experiment_B/20260821T041718Z_seed1234/`
- Output activation: Linear
- Formal final classification: **Negative Transfer**
- Formal B must remain preserved and must not be replaced by B2.

### Experiment B2
- Run: `reports/Solar Energy Result/Linear_Supplementary/Experiment_B2/20260822T120239Z_seed1234/`
- Positioning: **Post-Test Supplementary / Exploratory Tuning**
- Selected candidate: **B2-P1**
- Strategy: Last LSTM + target-specific output head; Partial FT; BN frozen; Linear output; LR=1e-5
- Numerical classification: **Supplementary Positive Transfer**
- B2 does NOT constitute a new untouched confirmatory Final Test.

### Scientific closure
- `SEALED / CLOSED`
- `NO MORE TUNING`
- `NO MORE TEST ACCESS FOR MODEL DEVELOPMENT`

The repository audit must not change these scientific facts unless a concrete higher-priority artifact proves that the recorded evidence is wrong. If such a conflict is found, report it as **BLOCKING EVIDENCE CONFLICT** and STOP; do not repair it automatically.

---

## 3. REPOSITORY AREAS TO INVENTORY

Inspect at minimum:

```text
r2_config/
r2_helpers/
reports/r2/
reports/Solar Energy Result/_smoke_linear/
reports/Solar Energy Result/Linear_Formal/Experiment_B/
reports/Solar Energy Result/Linear_Supplementary/Experiment_B2/
reports/Solar Energy Result/Thesis_Figures/Experiment_B2_Plant2_to_Plant1/
reports/Solar Energy Result/實驗 B：Plant2 → Plant1/
notebook/執行紀錄/Phase_IV_B_Audit_20260821T041718Z_seed1234/
tests/
tools/
utils/
README.md
results_summary.md
experiment_protocol.md
```

If `Experiment_B_Documentation/` exists, inventory it too.

Do NOT infer equivalence merely from similar filenames. Distinguish Legacy / Corrected R2 / Linear Formal / B2 / thesis-derived files.

---

## 4. REQUIRED AUDIT QUESTIONS

Answer all ten questions with concrete repository evidence.

### Q1. Are all formal artifacts present?
For Formal B and B2 verify existence of, where applicable:
- run manifest(s)
- selection / selection lock
- source candidate artifacts
- WOTL candidate artifacts
- PFT/B2 candidate artifacts
- selected checkpoints
- training history
- validation metrics / predictions
- Final Test authorization/state
- original-scale Test metrics
- prediction CSV
- transfer classification
- Test reuse disclosure
- strategy/result cards
- thesis figure manifest / metrics recheck

Return `PRESENT`, `MISSING`, or `NOT APPLICABLE` for each.

### Q2. Which files are tracked, untracked, or ignored?
Use read-only Git commands to classify at least:
- canonical docs;
- Formal B run directory;
- B2 run directory;
- Thesis_Figures;
- `.hdf5` / checkpoint files;
- prediction CSV / metrics JSON / manifest JSON;
- notebooks and execution records.

Report counts and representative paths. Do not add anything to Git.

### Q3. Do selected checkpoints exist and do their SHA-256 values match?
Verify the physical files for the documented selected checkpoints, including at minimum:
- selected Formal Source checkpoint;
- selected Formal WOTL checkpoint;
- selected Formal PFT checkpoint;
- selected B2-P1 checkpoint.

For each report:
`path | exists | documented_sha256 | computed_sha256 | MATCH/MISMATCH`

A mismatch is a blocking issue.

### Q4. Are predictions / metrics / manifests mutually traceable?
For WOTL, Formal PFT, and B2-P1 verify whether:
- prediction file identifies the same run/candidate;
- sample count agrees with expected 2607 Test sequences;
- metrics file values agree with summaries/manifests;
- timestamp / y_true alignment evidence exists;
- no evidence shows clipping or an alternate Test interval.

Do not regenerate predictions.

### Q5. Are Formal B and B2 cleanly separated?
Check folder namespaces, run IDs, manifests, disclosure files and summaries.
Confirm that B2 does not overwrite Formal B and that Formal B Negative Transfer remains preserved.

### Q6. Are there stale README / results_summary / Master documents?
Search the repository for documentation copies that still state, for example:
- `Experiment B = Pending`;
- `Corrected Experiment B not completed`;
- Formal B/B2 conflated;
- Sigmoid incorrectly described as the Formal B output;
- B2 described as a new untouched Final Test.

List exact paths and stale statements. Do not edit them.

### Q7. Are any HDF5/checkpoint files insufficiently backed up?
From repository evidence only, identify large/important HDF5/checkpoint artifacts for Formal B/B2 and whether there is documented evidence of backup.
If backup status cannot be proven, mark `BACKUP STATUS UNVERIFIED` rather than guessing.

Prioritize at minimum selected checkpoints and any checkpoint necessary to reproduce a reported selected model.

### Q8. Which files should enter normal Git?
Recommend only lightweight reproducibility evidence, such as:
- canonical README / results_summary / experiment_protocol / Master;
- run manifests;
- params / selection / authorization / final state;
- metrics JSON;
- prediction CSV where practical;
- history CSV;
- figure manifests;
- transfer classification / disclosure documents.

Return recommendations only; do not stage files.

### Q9. Which checkpoints should use Git LFS or external backup?
Classify selected/formal `.hdf5` and large binary artifacts into:
- `KEEP LOCAL + EXTERNAL BACKUP`
- `GIT LFS CANDIDATE`
- `DISPOSABLE / NON-ESSENTIAL INTERMEDIATE` (only when evidence supports this; do not delete)

Do not configure LFS or move files.

### Q10. Is there any major gap that blocks repository archival SEALED status?
A blocker includes, for example:
- selected checkpoint missing;
- checkpoint hash mismatch;
- official metrics/prediction evidence missing or irreconcilable;
- Formal B overwritten by B2;
- current canonical docs materially contradict high-priority artifacts;
- selected artifacts exist only in an unverified ephemeral location with no recoverable copy.

Non-blocking notes may include large untracked intermediate checkpoints or stale historical docs that are clearly distinguishable and can be archived later.

---

## 5. REQUIRED GIT READ-ONLY COMMANDS

Run the equivalent of the following without changing repository state:

```bash
git rev-parse --show-toplevel
git rev-parse HEAD
git branch --show-current
git status --short
git status --ignored --short
git log --oneline -20
git diff --name-only
git diff --stat
```

Use `git ls-files` and `git check-ignore -v <path>` as needed.

Do not run destructive or mutating Git commands.

---

## 6. EVIDENCE PRIORITY

When evidence conflicts, apply:

1. raw data and actually executed code;
2. split / scaler / preprocessing code;
3. params / scaler / checkpoint / prediction artifacts;
4. recalculated or official original-scale metrics;
5. training log / history;
6. results summary;
7. experiment protocol;
8. README;
9. slides / thesis prose / notes.

If higher-priority evidence remains contradictory, label **【需人工確認】**.
If evidence is absent, label **【待確認】** or `NOT LOCATED IN EVIDENCE INSPECTED`.
Do not infer missing facts.

---

## 7. REQUIRED OUTPUT FORMAT

Return sections in exactly this order:

1. **Executive Verdict**
2. **Repository Identity** - root, branch, HEAD, dirty/clean
3. **Formal B Artifact Inventory**
4. **B2 Artifact Inventory**
5. **Tracked / Untracked / Ignored Audit**
6. **Checkpoint Existence + SHA-256 Audit**
7. **Prediction / Metrics / Manifest Traceability Audit**
8. **Formal B vs B2 Isolation Audit**
9. **Stale Documentation Audit**
10. **HDF5 / Backup Audit**
11. **Normal Git Recommendations**
12. **LFS / External Backup Recommendations**
13. **Blocking Issues**
14. **Non-Blocking Notes**
15. **Final Closure Classification**

Final classification must be exactly one of:

- `REPOSITORY ARCHIVAL CLOSURE: PASS`
- `REPOSITORY ARCHIVAL CLOSURE: PASS WITH NOTES`
- `REPOSITORY ARCHIVAL CLOSURE: BLOCKED`

Then STOP.

---

## 8. IMPORTANT STOP CONDITION

This is an audit only.

After reporting findings:
- do not modify any file;
- do not stage/commit/push;
- do not create archives;
- do not run training or Test;
- do not implement recommendations.

**STOP and wait for explicit human approval.**
