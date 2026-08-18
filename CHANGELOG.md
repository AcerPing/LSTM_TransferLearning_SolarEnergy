# Solar LSTM CHANGELOG

**文件版本：** v1.0

**建立日期：** 2026-08-18

**適用專案：** LSTM研究與實驗－SolarEnergy 太陽能發電預測（智慧能源）

**文件性質：** 專案重要變更紀錄（Project Change Log）

**目前狀態：** 已整理截至 2026-08-18 之主要 Corrected R2 / R2.5、文件化與後續 protocol 決策；Experiment A2 / Experiment B 尚未正式執行。

---

# 1. 文件目的

本文件記錄 Solar LSTM 專案中已經實際發生的重要變更、正式里程碑與研究決策，供：

- Git 歷史查核
- Legacy / Corrected 流程區隔
- 實驗復現差異追蹤
- 論文研究方法與實驗結果之版本追溯
- 後續 Experiment A2 / Experiment B 執行前確認

本文件只記錄：

```text
已經發生的變更
+
已正式建立的文件
+
已封存的實驗結果
+
已核准但尚未執行的 protocol decision
```

本文件不應把：

```text
planned
```

寫成：

```text
implemented
```

也不應把：

```text
research objective
```

寫成：

```text
achieved result
```

---

# 2. Changelog 使用原則

若本文件與較高優先序證據衝突，優先順序為：

```text
actual code / raw data
→ run manifest / params / scaler / checkpoint / prediction
→ original-scale metrics
→ training log
→ results_summary.md
→ experiment_protocol.md
→ CHANGELOG.md
→ README / slides / notes
```

若重大項目仍有衝突，標示：

```text
【需人工確認】
```

不得自行挑選較漂亮或較方便的一版。

---

# 3. 2026-08-16 — R2.5 Phase D 正式完成

**Git commit：**

```text
200baf627442582986bc0113b97aa6075a5973e4
```

**里程碑：**

```text
R2.5 Phase D
Validation-based candidate selection
```

主要完成事項：

- Experiment A（Plant1 → Plant2）之 R2.5 Partial Fine-tuning Candidate B 正式執行
- Strategy：
  ```text
  partial_target_adapters_last_lstm
  ```
- Trainable layers：
  ```text
  1, 4, 6
  ```
- Frozen layers：
  ```text
  2, 3, 5
  ```
- BatchNormalization layers 保持 frozen
- Target selection 僅依 Validation
- Target Test 尚未於 Phase D 用於 selection
- selected checkpoint 已鎖定
- selected checkpoint SHA-256 已記錄
- source checkpoint SHA-256 已記錄
- `selection_locked = true`
- `test_metrics_used_for_selection = false`

正式 Validation original-scale metrics：

```text
MAE  = 3702.2560237215926
MSE  = 39303185.752355754
RMSE = 6269.225291242591
R²   = 0.020418227602119643
n    = 126
```

Best epoch：

```text
500
```

此階段尚未形成 Final Target Test 結論。

---

# 4. 2026-08-17 — R2.5 Phase E Final Target Test

**Git commit：**

```text
af03be5
```

**Commit message：**

```text
2026/08/17 R2.5 Phase E: add final Target Test and formal evidence
```

主要完成事項：

- 對已鎖定之 R2.5 Candidate B 執行一次正式 Plant2 Target Test
- Target Test sequence count：
  ```text
  2607
  ```
- Test 不參與 training
- Test 不參與 checkpoint selection
- `test_access_count = 1`
- `post_test_tuning_allowed = false`

Final Target Test original-scale metrics：

```text
MAE  = 2252.923495328691
MSE  = 14216235.103203941
RMSE = 3770.4422954348393
R²   = 0.6443032597170582
n    = 2607
```

相較 Experiment A Without TL：

```text
MAE  improved
MSE  deteriorated
RMSE deteriorated
R²   deteriorated
```

正式 classification：

```text
observed_mixed
thesis_interpretation = Mixed Result
```

Experiment A 因此不得宣稱為 Positive Transfer。

此 Final Test 完成後，原 Experiment A 進入：

```text
SEALED
```

狀態。

---

# 5. 2026-08-17 — README 改為 Solar 專案正式指南

**Git commit：**

```text
53f349e
```

**Commit message：**

```text
2026/08/17 docs: replace generic README with Solar experiment guide
```

主要完成事項：

- 將 generic README 改為 Solar LSTM 專案專用說明
- 整理：
  - Dataset / Target / Features
  - Corrected preprocessing
  - Source / Target Profile
  - scaler policy
  - sequence setting
  - LSTM architecture
  - Experiment A / R2 / R2.5
  - Test integrity
  - original-scale metrics
  - DC_POWER unit caveat
  - Git / artifact policy
  - 後續 Experiment A2 / Experiment B 邊界

README 定位：

```text
Project overview / operating guide
```

不取代正式 run evidence。

---

# 6. 2026-08-18 — 建立正式 results_summary.md v1

**Git commit：**

```text
50c560073c56ce88f1b879bdc38c42e25036ee28
```

**Commit message：**

```text
2026/08/18 docs: add Solar formal results summary v1
```

主要完成事項：

- 建立 `results_summary.md`
- 正式整理 Corrected R2 / R2.5 已完成結果
- 保存 Experiment A Without TL / Freeze / Full Fine-tuning 指標
- 保存 R2.5 Partial FT Validation / Final Test 指標
- 保存 checkpoint provenance
- 保存 Final Test integrity
- 正式記錄：
  ```text
  Experiment A = Mixed Result
  ```
- 正式記錄：
  ```text
  Experiment A = SEALED
  ```
- 明確區分：
  ```text
  completed result
  vs
  future A2 / B plan
  ```

此文件定位：

```text
Formal results summary
```

---

# 7. 2026-08-18 — 建立 experiment_protocol.md v1

**Git commit：**

```text
2de09af4a343c7108c32588b49637c5dab6bb751
```

**Commit message：**

```text
2026/08/18 docs: add Solar experiment protocol v1
```

主要完成事項：

- 建立 `experiment_protocol.md`
- 正式鎖定後續 Experiment A2 / Experiment B 執行原則
- 正式研究目標：
  ```text
  A2 = Plant1 → Plant2
  Goal = Positive Transfer Learning

  B = Plant2 → Plant1
  Goal = Positive Transfer Learning
  ```
- Positive Transfer 判定標準固定：
  ```text
  MAE_TL  < MAE_WOTL
  MSE_TL  < MSE_WOTL
  RMSE_TL < RMSE_WOTL
  R²_TL   > R²_WOTL
  ```
- Test tuning 明確禁止
- same-activation fair comparison 明確要求
- Source / Target split、scaler、sequence、inverse transform、run artifacts 與 stop conditions 正式文件化

此文件定位：

```text
Formal future experiment execution protocol
```

---

# 8. 2026-08-18 — Linear Output Strategy 正式進入 Protocol

**狀態：**

```text
Protocol decision only
NOT YET IMPLEMENTED
NOT YET TRAINED
```

基於 Training-only MinMaxScaler 查核結果：

```text
Plant1 Target Test:
normalized DC_POWER max = 1.056099

Plant2 Target Test:
normalized DC_POWER max = 1.256463
```

而 Legacy LSTM output 為：

```text
Dense(1, activation="sigmoid")
```

Sigmoid output 受限於：

```text
0 < y_pred < 1
```

因此 `experiment_protocol.md v1` 將 A2 / B 的 revised primary output strategy 定義為：

```text
Dense(1, activation="linear")
```

但截至本 CHANGELOG v1 建立時：

```text
model.py 尚未因此正式改寫為 A2 / B Linear production implementation
尚未建立正式 A2 / B Linear source checkpoint
尚未完成 A2 / B Linear training
尚未產生 A2 / B Linear Final Test metrics
```

因此不得寫成：

```text
Changed activation to Linear
```

較正確說法為：

```text
Linear output strategy approved / documented for the revised A2 / B protocol.
```

---

# 9. Experiment A Final Historical Status

截至 2026-08-18：

```text
Experiment A
Direction = Plant1 → Plant2

Corrected R2
= completed

R2.5 Partial FT
= completed

Final classification
= Mixed Result

Status
= SEALED
```

其正式結果不得被 A2 覆寫。

Experiment A2 應視為：

```text
new revised controlled follow-up experiment
```

---

# 10. Experiment A2 Current Status

截至本 v1：

```text
Direction = Plant1 → Plant2
Status = PENDING
Primary revised activation = Linear
Primary TL strategy = Partial Fine-tuning principle
Goal = Positive Transfer Learning
```

尚未完成：

- read-only preflight
- Linear source pretraining implementation
- Linear Without-TL baseline
- Linear Partial FT formal run
- Validation selection
- Final Test
- Transfer classification

Plant2 Target Test 已於舊 Experiment A 揭露，因此 A2 必須保留 follow-up evidence limitation。

---

# 11. Experiment B Current Status

截至本 v1：

```text
Direction = Plant2 → Plant1
Status = PENDING
Primary revised activation = Linear
Primary TL strategy = Partial Fine-tuning principle
Goal = Positive Transfer Learning
```

尚未完成：

- read-only preflight
- Plant2 Source Linear pretraining
- Plant1 Linear Without-TL baseline
- Plant1 Linear Partial FT
- Validation selection
- Final Target Test
- Transfer classification

Experiment B 應優先保護 Plant1 Target Test integrity，直到 config / selection / checkpoint 鎖定。

---

# 12. Files / Artifacts Deliberately Not Changed

截至本 v1，下列既有 untracked items 尚未自動加入 Git：

```text
notebook/修改紀錄/
notebook/執行紀錄/
reports/Solar Energy Result/R2/Experiment_A/20260814T103322Z_seed1234/
```

目前政策：

```text
do not delete
do not blindly add
do not stage all
```

後續應先 read-only classification，再決定：

- keep local only
- ignore
- track selected evidence
- archive separately

不得僅為取得 clean `git status` 而刪除。

---

# 13. Artifact / Git Policy

正式建議：

```text
Code / tests
+
formal metrics
+
manifests
+
predictions
+
figures
+
history / documentation
→ Git
```

大型 per-epoch checkpoints：

```text
local / external backup
```

並透過：

```text
selected checkpoint path
+
SHA-256
+
run manifest
```

保存 provenance。

禁止使用：

```text
git add .
```

作為正式實驗 artifacts 的預設操作。

---

# 14. Pending Next Milestone

下一階段：

```text
Experiment A2 / Experiment B
Read-only Preflight
```

其後才進入：

```text
Linear implementation design
→ zero-epoch contract test
→ 2-epoch smoke
→ Source Linear pretraining
→ Target Linear Without TL
→ Target Linear Partial FT
→ Validation-only controlled selection
→ config / checkpoint lock
→ Final Target Test
→ Transfer classification
```

本 CHANGELOG 不代表上述項目已完成。

---

# 15. Version History

## v1.0 — 2026-08-18

首次建立。

涵蓋：

```text
R2.5 Phase D
R2.5 Phase E
Solar README
results_summary.md v1
experiment_protocol.md v1
Experiment A sealed status
A2 / B Linear protocol decision
pending next milestone
```

---

# 16. Final Principle

Solar LSTM 專案之變更紀錄應遵循：

> **只記錄真實發生的變更，不把規劃寫成完成，不把研究目標寫成研究結果。**

所有後續變更需保持：

```text
可追蹤
可重現
可查核
不覆寫舊證據
不依 Test 事後改寫研究故事
```

---

**End of `CHANGELOG.md` v1.0**
