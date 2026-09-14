# Solar LSTM Evidence Retention Policy

## 1. Purpose and Scope

This policy governs source-controlled and externally retained research evidence for the Solar LSTM repository on the `AcerPing` research-development branch. Its purpose is to preserve reproducibility, provenance, auditability, and the separation between formal, supplementary, legacy, interrupted, smoke, and otherwise unverified evidence.

This policy defines retention and classification rules only. It does not change experiment metrics, predictions, checkpoints, manifests, lifecycle status, model conclusions, or formal authorization gates. It does not authorize training, prediction, evaluation, deletion, artifact rewriting, Git LFS migration, external transfer, or test-set access.

Unknown destinations, incomplete verification, and decisions that still require a person are explicitly recorded as `TBD`, `NOT_YET_VERIFIED`, or `HUMAN_DECISION_REQUIRED`.

## 2. Evidence Classes

Every retained experiment artifact must have an evidence class supported by its manifest, validation record, checksum, and lifecycle evidence. Similar filenames, directory names, timestamps, or model metrics are not sufficient grounds for classification.

- `FORMAL_LOCKED`: a formally selected artifact whose identity and checksum have been locked under an approved protocol. Locking must occur before any separately authorized final-test lifecycle.
- `FORMAL_SELECTED`: an artifact selected by the protocol's registered validation-only rule, with selection evidence preserved. This class alone does not imply that the artifact is locked, tested, sealed, or approved for Git LFS.
- `FORMAL_BEST`: the validation-selected best checkpoint from a completed formal run, supported by its formal manifest and training evidence.
- `SEALED_LEGACY_BEST`: the preserved best checkpoint from a completed and sealed legacy experiment. The legacy label and original provenance remain unchanged.
- `SUPPLEMENTARY_SELECTED`: a selected checkpoint from a supplementary experiment. It must remain clearly separate from the primary formal experiment and may be retained in Git LFS only when it is used as thesis evidence and approved by governance.
- `PER_EPOCH_CANDIDATE`: an intermediate epoch checkpoint that was not selected as the retained formal checkpoint. It belongs in the external immutable archive rather than normal Git or Git LFS.
- `INTERRUPTED_RUN_MIRROR`: a checkpoint from an interrupted run that is byte-identical by SHA256 to a canonical checkpoint from a completed PASS run. The mirror remains evidence of the interrupted lifecycle and is not promoted to formal status.
- `UNVERIFIED_ANOMALOUS_CHECKPOINT`: a checkpoint with unresolved provenance or structural evidence that differs from its expected mirror set. It must be preserved without being called corrupt, formal, selected, or best.
- `SMOKE_ONLY`: an artifact created solely by an explicitly identified smoke test. It is not formal experiment evidence and cannot support formal model-selection or test claims.
- `UNVERIFIED`: an artifact whose identity, role, lifecycle, or checksum support is insufficient for another class.

No artifact is automatically promoted because it exists, loads successfully, has favorable metrics, resembles another checkpoint, or is old. Promotion requires the evidence and human approval specified by the applicable formal protocol and governance step.

## 3. Git / Normal Git Policy

Normal Git is the default retention mechanism for reviewable, reasonably sized research records, including:

- source code, configuration, formal protocols, and approved documentation;
- experiment manifests, parameters, seed records, environment records, and selection records;
- metrics, per-sample predictions, result cards, training and evaluation logs, and figures;
- checksum inventories and external-archive inventories;
- small approved scaler artifacts when their preservation is required by the protocol.

Machine-generated sealed evidence must be committed as the authenticated artifact. It must not be rewritten merely to normalize whitespace, formatting, line endings, column order, or presentation. A formatting warning in sealed evidence must be documented rather than silently repaired when repair would change the authenticated bytes.

Large checkpoint collections, per-epoch candidates, archives, caches, temporary files, and unclassified binaries are not suitable for normal Git merely because they are research-related.

## 4. Git LFS Policy

Git LFS retention is limited to checkpoint binaries with evidence-supported roles of `FORMAL_LOCKED`, `FORMAL_BEST`, `SEALED_LEGACY_BEST`, or a thesis-used and governance-approved `SUPPLEMENTARY_SELECTED`. A `FORMAL_SELECTED` checkpoint must first receive its required locked or best evidence status before LFS retention.

The currently approved future Git LFS retention scope is exactly **14 logical paths representing 13 unique SHA256 objects, approximately 6.97 MiB by logical-path size**:

- six current sealed R2 `best_model` checkpoints;
- four historical R2 PASS formal-best checkpoints;
- three Linear Formal Experiment B selected checkpoints: Source, Without Transfer Learning (WOTL), and Partial Fine-Tuning;
- one Linear Supplementary B2 overall-selected P1 checkpoint.

This statement defines the approved scope; it does not assert that Git LFS is configured or that migration has occurred. Git LFS is a version-control transport and retention mechanism, not the sole backup. Every retained checkpoint must also satisfy the backup rules in Section 7.

## 5. External Immutable Archive Policy

`PER_EPOCH_CANDIDATE` checkpoint collections must not be placed in normal Git or Git LFS. They must be transferred to an immutable external archive with a complete SHA256 inventory and verified after transfer.

The locked per-epoch archive scope is **4,624 checkpoints totaling approximately 2.191 GiB**. The separately governed interrupted R2 evidence set in Section 9 also belongs in the external archive; it must not be silently folded into, substituted for, or confused with the 4,624-item per-epoch count.

Current external archive status:

- archive location: `TBD`;
- transfer completed: `NOT_YET_VERIFIED`;
- inventory verified at destination: `NOT_YET_VERIFIED`;
- human destination decision: `HUMAN_DECISION_REQUIRED`.

Creating an archive file alone does not establish immutability, completeness, or a verified backup. The source evidence remains protected until the archive, checksums, and backup copies have all been verified.

## 6. Checksum and Inventory Rules

The version-controlled inventory for retained checkpoint evidence must contain these fields:

1. `relative_path`
2. `filename`
3. `experiment`
4. `run_id`
5. `lifecycle`
6. `candidate_id`
7. `epoch`
8. `size_bytes`
9. `sha256`
10. `checkpoint_role`
11. `git_lfs_retained`
12. `archive_location`
13. `archive_verified`
14. `backup_copy_1`
15. `backup_copy_2`
16. `notes`

Every SHA256 value must be the full 64-character lowercase hexadecimal digest of the exact retained bytes. Checksums must be computed before transfer and recomputed at each destination after transfer. A copied file is not verified until the destination digest exactly matches the source inventory.

Inventory records must distinguish logical paths from unique content objects so that byte-identical mirrors remain auditable without inflating claims about unique checkpoint content.

## 7. Backup Policy

Every retained checkpoint or immutable archive must have at least two independent backup copies in addition to any working copy. Independence requires distinct failure domains; two paths on the same storage device are not presumed independent.

Current backup destinations are `TBD` and require `HUMAN_DECISION_REQUIRED`. Until the files at both destinations have been transferred, checksummed, and independently inspected, their inventory status must remain `NOT_YET_VERIFIED`. Repository presence, Git LFS presence, an archive file, or an untested sync indicator must not be represented as a verified backup.

## 8. Canonical and Mirror Rules

Canonical identity is established by exact SHA256 equality plus the higher-priority completed manifest and lifecycle evidence. Canonical selection must not be based only on filename, modification time, directory position, path naming, or file size.

The 246 byte-identical interrupted R2 checkpoint copies are classified as `INTERRUPTED_RUN_MIRROR`. Their canonical counterparts are the matching checkpoints supported by completed PASS evidence. The interrupted copies must retain their interrupted-run provenance in the inventory and must not be deleted before external-archive verification, two independent backup verifications, and explicit human approval.

Byte identity permits deduplicated storage only when logical-path provenance remains represented in the inventory. It does not erase the research history of the mirror path.

## 9. Interrupted / Anomalous Evidence Rules

Run `20260814T103322Z_seed1234` is classified as `INCOMPLETE/INTERRUPTED` and is not a formal completed run.

Within its 247-checkpoint evidence set:

- 246 byte-identical copies are `INTERRUPTED_RUN_MIRROR` and belong in the external immutable archive;
- epoch 271 is `UNVERIFIED_ANOMALOUS_CHECKPOINT` and belongs in the external immutable archive.

Epoch 271 is anomalous and unverified because there is no completed manifest, no PASS mirror supporting it, and its file size differs from the expected mirror pattern. These observations do **not** establish that the checkpoint is corrupt. It must not be classified as formal, selected, best, or approved for Git LFS, and it must not be deleted. Its original path, run identity, size, SHA256, and explanatory note must be preserved in the archive inventory.

## 10. Deletion Prohibition

No research evidence may be deleted until all of the following conditions are met:

1. its evidence class and lifecycle have been documented;
2. its full inventory record and source SHA256 have been recorded;
3. Git LFS retention has been verified when the artifact is in the approved LFS scope;
4. external archive transfer and destination SHA256 have been verified when external retention applies;
5. two independent backup copies have been verified;
6. a human has explicitly approved the exact deletion scope.

An artifact must not be deleted merely because it appears duplicated, is large, has unfavorable metrics, is from an older run, or is inconvenient to version. Suspected duplicates remain `REVIEW_DUPLICATE`, `INTERRUPTED_RUN_MIRROR`, or another evidence-supported class until the full governance process is complete.

## 11. Formal / Supplementary / Legacy Separation

Formal, supplementary, and legacy evidence must remain distinguishable in paths, manifests, inventories, labels, and thesis claims.

- Linear Supplementary B2 is supplementary evidence and must not replace, overwrite, or be presented as Linear Formal Experiment B.
- Legacy and Corrected R2 evidence must keep their authenticated experiment identity. Legacy R2 runs must not be renamed, overwritten, or retrospectively presented as Linear Formal results.
- A formal checkpoint, a supplementary selection, and a sealed legacy best may share architecture or checksum characteristics without becoming the same experimental claim.

## 12. A2 Governance Requirements

Experiment A2 must use its approved Linear Formal protocol and record, at minimum:

- experiment ID and direction;
- source and target profiles;
- activation and architecture contract;
- Corrected R1 split, scaler, feature order, window, and horizon evidence;
- registered candidate IDs and hyperparameters;
- seed, batch size, maximum epochs, shuffle policy, callbacks, and device;
- validation-only selection rule and selection evidence;
- locked checkpoint path and full SHA256;
- code, environment, Git commit, dirty status, and input-evidence provenance;
- lifecycle state and failure evidence when applicable.

Final Target Test access requires separate, explicit human authorization after validation-only selection and checkpoint locking. A2 execution must not overwrite, relabel, or modify Experiment A, Experiment B, R2, R2.5, supplementary, legacy, or sealed evidence. A new Git HEAD created by repository governance must replace the prior HEAD in a fresh zero-epoch A2 preflight and subsequent A2 provenance records.

## 13. Current Governance Status

- Step 1 — controlled source-code commits: `COMPLETE`.
- Step 2 — controlled lightweight research-evidence commits: `COMPLETE`.
- Step 3 — evidence retention policy: `THIS POLICY`; commit verification remains part of this governance step.
- Step 4 — selected/locked checkpoints to Git LFS: `NOT STARTED`.
- Step 5 — per-epoch checkpoints to external immutable archive with SHA256 inventory: `NOT STARTED`.
- Step 6 — notebook/archive duplicate decisions: `NOT STARTED`.
- Step 7 — commit and push remaining approved governance work: `NOT STARTED`.
- Step 8 — confirm clean Git status: `NOT STARTED`.
- Step 9 — rerun the A2 zero-epoch preflight using the new HEAD: `NOT STARTED`.

No status in this section implies that Git LFS, external archiving, backup verification, deletion approval, repository push, A2 training, or Target Test access has occurred.
