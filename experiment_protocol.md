# Solar LSTM Experiment Protocol

**文件版本：** v1.0

**建立日期：** 2026-08-18

**適用專案：** LSTM研究與實驗－SolarEnergy 太陽能發電預測（智慧能源）

**文件性質：** 正式實驗執行規範（Formal Experiment Protocol）

**目前狀態：** Experiment A 原始 Corrected R2 / R2.5 已完成並封存；Experiment A2 與 Experiment B 待執行。

---

# 1. 文件目的

本文件定義 Solar LSTM 後續正式實驗規則，目的為確保：

- 實驗可重現
- scaler / inverse transform 正確
- Without TL 與 TL 公平比較
- Test 不作為反覆調參依據
- Positive Transfer 判定標準固定
- 舊 Experiment A 完整保留
- A2 / B 在執行前預先定義
- 每個正式 run 保留完整 evidence chain

本文件規範「接下來允許如何做實驗」，不取代 `results_summary.md`、run manifest、正式 metrics、prediction、checkpoint 或實際程式碼。

---

# 2. 核心研究目標

```text
Experiment A2
Plant1 → Plant2
Goal = Positive Transfer Learning

Experiment B
Plant2 → Plant1
Goal = Positive Transfer Learning
```

本研究可透過合理、有限且可重現的受控調整，提高取得 Positive Transfer 的機會。

但：

> Positive Transfer 是研究目標，不是預先保證的結論。

不得為達成 Positive Transfer 而改寫判定標準、刪除不利結果、反覆使用 Test 選參數，或混用不同 activation / scaler / test interval。

---

# 3. Experiment A 封存規則

既有 Experiment A：

```text
Direction = Plant1 → Plant2
Output activation = Sigmoid
R2 / R2.5 completed
Final Target Test = revealed
Final classification = Mixed Result
Status = SEALED
```

不得覆寫、刪除、重新命名為 Positive Transfer，亦不得用相同 run ID 重跑。

Experiment A2 為新的 controlled follow-up protocol，不取代 Experiment A。

---

# 4. Evidence Strength

## 4.1 Experiment A2

Plant2 Target Test 已在 Experiment A Phase E 被揭露，因此 A2 可建立新的 Linear-output controlled protocol，但其 Test evidence 應解讀為：

```text
revised controlled follow-up evidence
```

若四項正式指標均改善，可記錄：

```text
Observed Positive Transfer under the A2 revised protocol
```

但不得稱為完全獨立於既有 Plant2 Test information 的首次 confirmatory evidence。

## 4.2 Experiment B

若 Plant1 Target Test 未參與 B protocol 的選模，則可採：

```text
Training / Validation
→ lock config
→ lock checkpoint
→ one-time Target Test
```

作為 reciprocal-direction formal evidence。

---

# 5. Dataset / Prediction Task Contract

正式 Target：

```text
DC_POWER
```

正式 Features：

```text
TIME_SIN
TIME_COS
IRRADIATION
AMBIENT_TEMPERATURE
MODULE_TEMPERATURE
```

歷史 `DC_POWER` 不加入 X。

---

# 6. Data Split Contract

所有 split 均維持 chronological order。

## Source Profile

```text
Training   = 2088 rows
Validation = 523 rows
Test       = 653 rows
```

Sequence counts：

```text
2083 / 518 / 648
```

## Target Profile

```text
Training   = 521 rows
Validation = 131 rows
Test       = 2612 rows
```

Sequence counts：

```text
516 / 126 / 2607
```

Source Test 不得重新合併 Training 或 Validation。

---

# 7. Preprocessing Contract

正式使用 Corrected R1：

- Generation / Weather 先依 `DATE_TIME` 聚合
- 建立完整 15 分鐘時間軸
- 各 split 內獨立補值
- 不跨 split 插值
- `DC_POWER` daytime interpolation 不跨日期
- 夜間缺失 `DC_POWER` 補 0
- `TIME_SIN` / `TIME_COS` 由時間戳建立
- 不因 TL 結果回頭修改 preprocessing

---

# 8. Scaling Contract

Feature scaler 只處理：

```text
IRRADIATION
AMBIENT_TEMPERATURE
MODULE_TEMPERATURE
```

Target scaler 只處理：

```text
DC_POWER
```

每個 Plant / Profile：

```text
Training → fit
Validation → transform only
Test → transform only
```

禁止 full-data fit、Validation/Test refit、clipping。

Training-only MinMaxScaler 下，Validation/Test 出現 `<0` 或 `>1` 為合法 transform 結果。

---

# 9. Sequence Contract

```text
window = 5
horizon = 1
frequency = 15 minutes
```

定義：

```text
X[i:i+5] → y[i+5]
```

各 split 獨立建立 sequence，不跨 split。

---

# 10. Model Architecture Contract

保留既有 LSTM backbone：

```text
Input
→ TimeDistributed(Dense 10)
→ LSTM 60
→ BatchNormalization
→ LSTM 60
→ BatchNormalization
→ Dense 1
```

Legacy / R2 / R2.5：

```text
Dense(1, activation="sigmoid")
```

A2 / B revised protocol：

```text
Dense(1, activation="linear")
```

第一階段不得同時任意更換整個 backbone。

---

# 11. Activation Policy

Sigmoid 保留為歷史正式證據。

A2 / B 主要新 protocol 使用 Linear，原因：

- `DC_POWER` 是 continuous regression target
- Training-only MinMaxScaler 不保證未來值永遠落在 `[0,1]`
- Sigmoid 有 `(0,1)` 硬上限
- Linear 可輸出 `<0` 或 `>1`
- inverse transform 可正常處理超出 Training range 的 prediction

主 metrics 必須使用 raw / unclipped prediction。

若做 non-negative post-processing，須另列 supplementary branch。

---

# 12. Same-Activation Fair Comparison Rule

Positive Transfer 只能在相同 activation 下比較。

正確：

```text
Linear Without TL
vs
Linear Freeze
vs
Linear Partial FT
vs
Linear Full FT（若納入）
```

禁止：

```text
Sigmoid Without TL
vs
Linear TL
```

並把差異解讀成 TL 效果。

同一 Target 的正式比較固定：

- preprocessing
- split
- features
- target
- window / horizon
- scaler policy
- activation
- loss
- test interval
- original-scale metrics

---

# 13. Experiment A2 Protocol

```text
Source = Plant1
Target = Plant2
Goal = Positive Transfer Learning
Primary activation = Linear
```

第一階段主比較：

```text
A2-WOTL-Linear
vs
A2-PartialFT-Linear
```

若資源允許，再補 Freeze / Full FT。

A2 Linear TL 應使用：

```text
Plant1 Source Profile
+
Plant1 Source Linear pretrained checkpoint
```

不得默認以 Sigmoid source checkpoint 取代 Linear source pretraining。

Plant2 Test 已歷史揭露，因此不得用它選：

- learning rate
- trainable layers
- BN policy
- dropout / L2
- candidate

A2 selection 僅能依 Training / Validation。

---

# 14. Experiment B Protocol

```text
Source = Plant2
Target = Plant1
Goal = Positive Transfer Learning
Primary activation = Linear
```

主比較：

```text
B-WOTL-Linear
vs
B-PartialFT-Linear
```

建立：

```text
Plant2 Source Linear pretrained checkpoint
```

所有 B candidates 必須先完成 Training / Validation，再 lock config / selection / checkpoint，最後才開 Plant1 Target Test。

---

# 15. Transfer Learning Strategy Contract

## Without TL

Fresh initialization，不載入 source checkpoint。

## Freeze

Transferred layers 載入 source weights 並 frozen。

## Partial Fine-tuning

R2.5 Candidate B 歷史 principle：

```text
Topology:
0 Input
1 TimeDistributed(Dense10)
2 LSTM60
3 BatchNormalization
4 LSTM60
5 BatchNormalization
6 Dense1

Transferred:
2,3,4,5

Trainable:
1,4,6

Frozen:
2,3,5
```

A2 / B 第一個 Partial FT candidate 優先沿用：

```text
target-specific front adapter
+
last recurrent block fine-tuned
+
output head trainable
+
early recurrent block frozen
+
BN frozen
```

若修改 scope，建立新的 candidate ID。

## Full Fine-tuning

只有所有可遷移層均開放時才能稱 Full FT。

---

# 16. Hyperparameter Policy

第一輪優先固定：

```text
optimizer = Adam
loss = MSE
seed = 1234
batch_size = 128
maximum_epochs = 500
window = 5
horizon = 1
```

Linear protocol 可有限比較 learning rate，例如：

```text
1e-5
3e-5
1e-4
```

實際候選須在 run 前寫入 config / manifest。

所有 tuning 只用 Validation。

一次只改 1～2 個主要因素。

禁止無限制 hyperparameter search。

---

# 17. Validation Selection Rule

```text
Training
→ learn parameters

Validation
→ early stopping
→ ReduceLROnPlateau
→ checkpoint selection
→ candidate comparison
```

Target Test 不參與：

- fitting
- early stopping
- LR scheduling
- checkpoint selection
- activation selection
- layer-scope selection
- hyperparameter selection

正式選定後產生 `selection.json`，至少記錄：

```text
run_id
candidate_id
best_epoch
selection_basis
validation original-scale metrics
checkpoint path
checkpoint SHA-256
selection_locked = true
test_metrics_used_for_selection = false
```

---

# 18. Final Test Gate

只有全部 PASS 後才能測 Final Target Test：

```text
[PASS] preprocessing
[PASS] split manifest
[PASS] scaler fit rows
[PASS] sequence alignment
[PASS] candidate config locked
[PASS] trainable layers verified
[PASS] checkpoint selected
[PASS] checkpoint SHA-256 recorded
[PASS] selection.json locked
[PASS] no Target Test metric used for selection
```

Final Test 完成後：

```text
post_test_tuning_allowed = false
```

該 run 立即封存。

---

# 19. Formal Metrics Contract

正式論文比較：

```text
original-scale MAE
original-scale MSE
original-scale RMSE
original-scale R²
```

Normalized metrics 僅作 diagnostics。

Prediction CSV 至少保存：

```text
target timestamp
y_true_normalized
y_pred_normalized
y_true_original
y_pred_original
residual_original
```

---

# 20. Positive Transfer Classification

相較相同 Target / protocol / activation 之 Without TL：

```text
MAE_TL  < MAE_WOTL
MSE_TL  < MSE_WOTL
RMSE_TL < RMSE_WOTL
R²_TL   > R²_WOTL
```

四項同時成立：

```text
Positive Transfer
```

誤差改善但 R² 未改善：

```text
Partial Positive Transfer
```

不同 metrics 有好有壞：

```text
Mixed Result
```

主要 metrics 整體退步：

```text
Negative Transfer
```

不得因單一指標改善宣稱全面 Positive Transfer。

---

# 21. Positive Transfer Optimization Principle

允許：

```text
Baseline
→ Partial FT Candidate
→ Validation comparison
→ limited controlled adjustment
→ Validation comparison
→ lock candidate
→ Final Test
```

禁止：

```text
Final Test
→ 不 Positive
→ 改 LR
→ 再 Test
→ 改 layers
→ 再 Test
```

本研究目標可明確設定：

```text
A2 Positive
B Positive
```

但 Test 不能成為 optimizer。

---

# 22. Run ID Policy

建議：

```text
<Protocol>_<Direction>_<Method>_<Activation>_seed<SEED>_<timestamp>
```

例如：

```text
A2_P1toP2_WOTL_Linear_seed1234_YYYYMMDDTHHMMSSZ
A2_P1toP2_PartialFT_Linear_seed1234_YYYYMMDDTHHMMSSZ
B_P2toP1_WOTL_Linear_seed1234_YYYYMMDDTHHMMSSZ
B_P2toP1_PartialFT_Linear_seed1234_YYYYMMDDTHHMMSSZ
```

禁止覆寫既有 run directory。

---

# 23. Required Artifacts

每個正式 run 至少保存：

```text
run_id
model_name
source
target
experiment_mode
activation
transfer_strategy
trainable_layers
frozen_layers
feature_scaler
target_scaler
train / validation / test range
window / horizon
batch_size
epochs
learning_rate
optimizer
loss
seed
environment
Git commit hash
training log
history.csv
checkpoint
checkpoint SHA-256
selection.json
normalized metrics
original-scale metrics
prediction CSV
Learning Curve
Prediction Plot
YY Plot
Residual Plot
Error Histogram
transfer_classification.json
date
```

---

# 24. Source Checkpoint Contract

Transfer run 必須記錄：

```text
source_model_protocol
source_activation
source_checkpoint_path
source_checkpoint_sha256
source_training_split
source_validation_split
source_test_status
```

Linear Target TL 原則上使用 Linear Source pretrained weights。

若要跨 activation transfer，須另建 protocol。

---

# 25. BatchNormalization Policy

A2 / B 第一個 Partial FT candidate 優先維持：

```text
BN frozen
```

若改為 BN trainable，必須建立新的 candidate ID，且只能依 Validation 比較。

不得因 Test 結果不理想才解凍 BN。

---

# 26. Prediction / Inverse Transform Contract

```text
normalized y_true / y_pred
→ correct Target scaler inverse_transform
→ original-scale y_true / y_pred
→ formal metrics
```

必須驗證：

- shape
- target scaler
- column order
- round-trip
- timestamp alignment
- sample count

禁止使用 feature scaler inverse target、以 normalized aggregate metrics 推估 original metrics，或以尾端截取方式掩蓋 alignment bug。

---

# 27. DC_POWER Unit Policy

Target：

```text
DC_POWER
```

資料集欄位說明標示為 kW，但 Plant1 / Plant2 原始尺度存在一致性疑慮。

正式圖表優先保守標示：

```text
DC_POWER (raw dataset scale; documented unit: kW)
```

不得自行除以 10 或 1000，也不得以 DC/AC 比值直接推算逆變器效率。

---

# 28. Prohibited Practices

禁止：

- Target Test tuning
- Test 後反覆修改同一 run
- 覆寫舊 results
- 刪除不利 run
- 不同 activation 直接宣稱 TL effect
- 不同 scaler / test interval 直接比較
- Validation loss 當 Test metrics
- normalized / original-scale 混用
- Partial FT 誤稱 Full FT
- clipping 使 metrics 變漂亮
- 修改 Positive Transfer 定義
- 跨 Plant 直接比較 original-scale MAE / RMSE 絕對值並宣稱誰較佳

---

# 29. Stop Conditions

遇到任一項立即停止：

1. Train / Validation / Test index overlap
2. scaler fit rows ≠ Training rows
3. Validation / Test 被重新 fit
4. prediction / target timestamp mismatch
5. inverse transform round-trip FAIL
6. sequence count 不符
7. trainable / frozen layers 不符 contract
8. Source checkpoint provenance 不明
9. BN state 不符 protocol
10. output activation 與 run ID 不符
11. Formal comparison 使用不同 Target test interval
12. Formal comparison 使用不同 Target scaler
13. run directory 已存在
14. manifest 缺少核心欄位
15. Target Test 被用於 candidate selection
16. environment 無法穩定載入 checkpoint

---

# 30. Thesis Interpretation Rules

若只有 single seed / single split / single test interval，使用：

- 於本次資料切分與實驗設定下
- 呈現較低之 MAE / RMSE
- 初步呈現 Positive Transfer 趨勢
- 誤差層面之部分改善
- 本次結果顯示

不得使用：

- 穩定改善
- 統計顯著
- 一致優於
- 普遍有效
- 證明模型最佳

若 A2 / B 達成 Positive Transfer，可寫：

> 於本次資料切分與實驗設定下，Partial Fine-tuning 相較相同 Linear-output Without-TL baseline 呈現較低之 MAE、MSE 與 RMSE，且 R² 同步改善，因此本次實驗初步呈現 Positive Transfer。

A2 另須揭露 Plant2 Target Test 曾於較早 Experiment A 被檢視。

---

# 31. Protocol Change Rule

任何主要變更須記錄於 `CHANGELOG.md` 並提升 protocol version。

主要變更包括：

- activation
- features
- target
- split
- scaler policy
- window / horizon
- loss
- backbone
- transfer scope
- BN policy
- Positive Transfer definition

不得 silent change。

---

# 32. Recommended Execution Order

```text
P0  experiment_protocol.md locked
↓
P1  A2 / B read-only preflight
↓
P2  Linear architecture implementation
↓
P3  zero-epoch contract tests
↓
P4  2-epoch smoke
↓
P5  Source Linear pre-training
↓
P6  Target Linear Without TL
↓
P7  Target Linear Partial FT
↓
P8  Validation-only controlled tuning if needed
↓
P9  lock config / selection / checkpoint
↓
P10 Final Target Test
↓
P11 Transfer classification
↓
P12 results_summary.md update
```

---

# 33. Current Protocol Status

```text
results_summary.md v1
= completed

Experiment A original
= completed / sealed / Mixed Result

Experiment A2
= pending

Experiment B
= pending

A2 revised activation
= Linear

B revised activation
= Linear

Primary TL strategy
= Partial Fine-tuning principle

Positive Transfer goal
= A2 and B

Positive Transfer criteria
= fixed

Target Test tuning
= prohibited
```

---

# 34. Final Research Principle

> **追求可信的 Positive Transfer，而不是事後製造 Positive Transfer。**

正式成功標準：

```text
資料角色清楚
+
scaler 正確
+
比較公平
+
Validation 選模
+
Test gate 完整
+
original-scale metrics 正確
+
Positive Transfer 判定一致
+
結果可重現
+
論文可防守
```

---

**End of `experiment_protocol.md` v1.0**
