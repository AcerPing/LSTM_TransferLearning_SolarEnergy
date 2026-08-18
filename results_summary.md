# Solar LSTM Results Summary

**文件版本：** v1.0

**建立日期：** 2026-08-18

**適用專案：** LSTM研究與實驗－SolarEnergy 太陽能發電預測（智慧能源）

**文件性質：** 正式實驗結果摘要（Formal Results Summary）

**目前狀態：** Experiment A 之 Corrected R2 / R2.5 已完成整理；Experiment A2 與 Experiment B 尚未執行。

---

## 1. 文件目的

本文件集中整理 Solar Power Dataset 之 LSTM 與 Transfer Learning 正式實驗結果，作為：

- 論文第四章實驗結果與分析素材
- 後續 Experiment A2 / Experiment B 比較基準
- Git / run manifest / prediction / metrics 證據鏈索引
- Positive / Partial Positive / Mixed / Negative Transfer 判定依據
- Legacy 與 Corrected 實驗結果區隔

本文件不取代原始 `run_manifest.json`、`metrics_original.json`、prediction CSV、`selection.json`、
`transfer_classification.json` 或 checkpoint provenance。

若本文件與高優先序正式實驗檔案衝突，應以實際程式、正式 run manifest、
original-scale metrics 與逐筆 prediction 為準。

---

## 2. 正式結果使用原則

### 2.1 正式指標

正式比較以 Target scaler inverse transform 後之 **original-scale metrics** 為主：

- MAE
- MSE
- RMSE
- R²

Normalized-scale metrics 僅作訓練診斷與除錯，不與 original-scale metrics 混表。

### 2.2 Transfer Learning 判定

**Positive Transfer：**

相較相同 Target Dataset、相同資料切分與相同 experimental protocol 之 Without-TL baseline：

- MAE 下降
- MSE 下降
- RMSE 下降
- R² 改善

四項同時成立時，判定為 **Positive Transfer**。

**Partial Positive Transfer：** 誤差指標改善，但 R² 未同步改善或仍為負值。

**Negative Transfer：** 主要評估指標整體退步。

**Mixed Result：** 不同 metrics 呈現不同方向，無法支持整體 Positive 或 Negative 判定。

---

## 3. Dataset 與 Formal Prediction Task

### 3.1 Dataset

- Plant 1
- Plant 2

### 3.2 正式預測目標

```text
DC_POWER
```

本研究以太陽光電系統之直流側發電功率作為預測目標。

### 3.3 正式模型輸入特徵

```text
TIME_SIN
TIME_COS
IRRADIATION
AMBIENT_TEMPERATURE
MODULE_TEMPERATURE
```

### 3.4 Sequence 設定

```text
window  = 5
horizon = 1
frequency = 15 minutes
```

即：

```text
過去 5 筆資料（75 分鐘）
→
預測下一個 15 分鐘時間點
```

---

## 4. Corrected Data / Scaling Contract

正式 Corrected 流程採：

- chronological split
- Source / Target Profile 分離
- split-isolated missing-value handling
- feature scaler 與 target scaler 分離
- scaler 僅使用 Training subset fit
- Validation / Test 只 transform
- no clipping
- 各 split 獨立建立 sequence
- Target scaler inverse transform 後計算正式 metrics
- Source Test 保持獨立，不回併 Training / Validation

### 4.1 Source Profile

```text
Training   : 2088 rows
Validation : 523 rows
Test       : 653 rows
```

Sequence counts：

```text
Training   : 2083
Validation : 518
Test       : 648
```

### 4.2 Target Profile

```text
Training   : 521 rows
Validation : 131 rows
Test       : 2612 rows
```

Sequence counts：

```text
Training   : 516
Validation : 126
Test       : 2607
```

---

# 5. Experiment A — Plant1 → Plant2

## 5.1 實驗方向

```text
Source = Plant1
Target = Plant2
```

目前已完成之正式結果包含 Corrected R2 與 R2.5 Partial Fine-tuning。

---

## 5.2 R2 Formal Run

**Run ID：**

```text
20260814T150304Z_seed1234
```

R2 為 Corrected Legacy Reproduction，主要保留既有 Legacy LSTM 架構與歷史策略，
同時採用 Corrected preprocessing、Training-only scaler、正確 prediction alignment
與 original-scale evaluation。

R2 Target 方法：

- Without Transfer Learning
- TL Freeze
- TL Full Fine-tuning

---

## 5.3 R2 Target Test — Original-Scale Metrics

| Method | MAE | MSE | RMSE | R² | Best Epoch | Epochs Completed | Result Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| Without TL | 2866.104160 | 13705909.923007 | 3702.149365 | 0.657072 | 292 | 302 | Baseline |
| TL Freeze | 2879.526879 | 24312780.909834 | 4930.799216 | 0.391683 | 500 | 500 | Negative / no overall improvement |
| TL Full Fine-tuning | 2405.355655 | 16887667.836972 | 4109.460772 | 0.577463 | 83 | 93 | Not Positive overall |

### 5.3.1 Without TL

```text
MAE  = 2866.104160
MSE  = 13705909.923007
RMSE = 3702.149365
R²   = 0.657072
```

此組作為 Experiment A 主要 baseline。

### 5.3.2 TL Freeze

```text
MAE  = 2879.526879
MSE  = 24312780.909834
RMSE = 4930.799216
R²   = 0.391683
```

相較 Without TL，四項主要指標均未呈現整體改善，因此不支持 Positive Transfer。

### 5.3.3 TL Full Fine-tuning

```text
MAE  = 2405.355655
MSE  = 16887667.836972
RMSE = 4109.460772
R²   = 0.577463
```

MAE 低於 Without TL，但 MSE、RMSE 與 R² 未同步改善，因此不判定為 Positive Transfer。

---

# 6. R2.5 Partial Fine-tuning — Experiment A

## 6.1 Formal Run

**Run ID：**

```text
20260816T073233Z_seed1234
```

**Strategy ID：**

```text
partial_target_adapters_last_lstm
```

正式名稱：

```text
Partial Fine-tuning / Partial FT
```

而非 Full Fine-tuning。

---

## 6.2 Layer / Parameter Contract

Layer topology：

```text
0 Input
1 TimeDistributed(Dense10)
2 LSTM60
3 BatchNormalization
4 LSTM60
5 BatchNormalization
6 Dense1(sigmoid)
```

Trainable layers：

```text
1, 4, 6
```

Frozen layers：

```text
2, 3, 5
```

BatchNormalization：

```text
Layer 3 frozen
Layer 5 frozen
```

Parameter counts：

```text
Trainable params     = 29161
Non-trainable params = 17520
Total params         = 46681
Trainable percentage = 62.468670%
```

Run manifest 已驗證：

- trainable layers 有實際更新
- frozen LSTM layer 未更新
- frozen BatchNormalization states 未更新
- Test 未在 Phase D model selection 中使用

---

## 6.3 Phase D — Validation Selection

```text
Status = PASS
Best epoch = 500
Epochs completed = 500
Best validation loss = 0.1819697022
Final learning rate ≈ 1.0e-7
```

Validation original-scale metrics：

| Metric | Value |
|---|---:|
| MAE | 3702.256024 |
| MSE | 39303185.752356 |
| RMSE | 6269.225291 |
| R² | 0.020418 |
| Samples | 126 |

Selection contract：

```text
selection_locked = true
selection_basis = original_scale_validation
test_metrics_used_for_selection = false
```

Selected checkpoint：

```text
candidate/partial_target_adapters_last_lstm/checkpoint_epoch_0500.hdf5
```

Selected checkpoint SHA-256：

```text
4dca61375c80d609f73ffc44e4cf829a4d1b20aab7e91e22f38072257e4679a6
```

Source checkpoint SHA-256：

```text
8aa299e09c58c2ce77473ba317bdaf9fe374310a453b61718b605022356ad6aa
```

---

# 7. R2.5 Phase E — Final Target Test

## 7.1 Test Integrity

```text
Status = PASS
Target Test accessed = true
Test access count = 1
Test metrics used for selection = false
Test metrics used for training = false
Post-test tuning allowed = false
Test sequence count = 2607
```

Experiment A 之 Plant2 Target Test 已於此階段揭露。

## 7.2 Partial FT Final Test — Original-Scale Metrics

| Metric | Value |
|---|---:|
| MAE | 2252.923495 |
| MSE | 14216235.103204 |
| RMSE | 3770.442295 |
| R² | 0.644303 |
| Samples | 2607 |

## 7.3 Partial FT vs Without TL

| Metric | Without TL | Partial FT | Direction | Relative Change |
|---|---:|---:|---|---:|
| MAE | 2866.104160 | 2252.923495 | Improved | 21.394221% improvement |
| MSE | 13705909.923007 | 14216235.103204 | Deteriorated | 3.723395% worse |
| RMSE | 3702.149365 | 3770.442295 | Deteriorated | 1.844683% worse |
| R² | 0.657072 | 0.644303 | Deteriorated | 1.943253% decrease |

Numerical delta：

```text
ΔMAE  = -613.180665
ΔMSE  = +510325.180197
ΔRMSE = +68.292930
ΔR²   = -0.012769
```

---

# 8. Experiment A Transfer Classification

正式 Phase E classification：

```text
classification = observed_mixed
rules_applied_to = original_scale_test
thesis_interpretation = Mixed Result
```

因此目前 Experiment A 原始 Corrected Legacy / R2.5 結果正式判定為：

> **Mixed Result**

原因：

- MAE 有明顯下降
- MSE 未下降
- RMSE 未下降
- R² 未改善

因此不符合本專案 Positive Transfer 的四項同步改善標準。

---

# 9. Experiment A Current Interpretation

於本次資料切分與實驗設定下，
Partial Fine-tuning Candidate B 相較 Without-TL baseline 呈現較低 MAE，
但 MSE、RMSE 與 R² 未同步改善。

因此目前只能確認：

> Partial Fine-tuning 在平均絕對誤差層面呈現改善，
> 但尚不足以支持整體 Positive Transfer Learning。

不得寫成：

- Transfer Learning 全面成功
- TL 穩定優於 Without TL
- 統計顯著改善
- Experiment A 已證明 Positive Transfer

---

# 10. Experiment A Seal Status

目前原始 Experiment A：

```text
Direction = Plant1 → Plant2
Output activation = Sigmoid
Final Test = revealed
Final classification = Mixed Result
Status = SEALED
```

Phase E 明確記錄：

```text
post_test_tuning_allowed = false
```

因此不得以既有 Plant2 Test 作為反覆調整 learning rate、trainable layers、
BN strategy、epoch count、activation、scaler 或 architecture 之模型選擇依據。

舊 Experiment A 結果應完整保留，不覆寫、不刪除、不重新命名為 Positive Transfer。

---

# 11. Output Activation Audit

目前 R2 / R2.5 實際執行之 Legacy architecture output：

```text
Dense(1, activation="sigmoid")
```

Training-only MinMax scaling 下，Validation / Test target 可能合法超出 `[0,1]`。

因此 Sigmoid 的輸出上限可能形成高峰預測限制。

目前只能視為：

> 已確認之 architecture limitation / future protocol consideration

不得直接宣稱：

> Experiment A Mixed Result 是由 Sigmoid 所造成。

因目前尚未完成 activation-controlled comparison。

---

# 12. Next Formal Research Targets

## 12.1 Experiment A2 — Plant1 → Plant2

目前狀態：

```text
Pending
```

研究目標：

> 建立新的受控 experimental protocol，以可信方式爭取取得 Positive Transfer Learning。

預定方向：

```text
Output activation = Linear（待 experiment_protocol.md 正式核准）
Without TL = same Linear protocol
TL / Partial FT = same Linear protocol
```

舊 Experiment A Mixed Result 保留為歷史正式結果；
A2 必須使用新的 run ID、manifest 與完整實驗證據鏈。

## 12.2 Experiment B — Plant2 → Plant1

目前狀態：

```text
Pending
```

方向：

```text
Source = Plant2
Target = Plant1
```

研究目標：

> 在 reciprocal transfer direction 下，取得可信之 Positive Transfer Learning 證據。

Activation、Partial FT、Learning Rate 與 Test gate
應於 `experiment_protocol.md` 中預先定義。

---

# 13. Positive Transfer Research Goal

本專案後續研究目標：

```text
Experiment A2：Plant1 → Plant2
→ Positive Transfer

Experiment B：Plant2 → Plant1
→ Positive Transfer
```

正式判定標準不得因研究目標而降低：

```text
MAE_TL  < MAE_WithoutTL
MSE_TL  < MSE_WithoutTL
RMSE_TL < RMSE_WithoutTL
R²_TL   > R²_WithoutTL
```

若實際結果未達標，應依 metrics 如實標記為 Partial Positive Transfer、
Mixed Result 或 Negative Transfer。

---

# 14. Evidence Index

## 14.1 R2 Formal Experiment A

```text
reports/Solar Energy Result/R2/Experiment_A/
└─ 20260814T150304Z_seed1234/
   ├─ run_manifest.json
   ├─ source/
   ├─ target/
   └─ target_comparison.csv
```

R2 principal comparison evidence：

```text
target_comparison.csv
```

## 14.2 R2.5 Partial FT

```text
reports/Solar Energy Result/R2.5/Experiment_A/Partial_FT/
└─ 20260816T073233Z_seed1234/
   ├─ run_manifest.json
   ├─ selection.json
   ├─ candidate/
   │  └─ partial_target_adapters_last_lstm/
   │     ├─ history.csv
   │     ├─ validation_metrics_original.json
   │     ├─ validation_predictions.csv
   │     └─ training_curve.png
   └─ final_test/
      ├─ metrics_original.json
      ├─ metrics_normalized.json
      ├─ predictions_test.csv
      ├─ comparison_to_without_tl.csv
      ├─ transfer_classification.json
      └─ final_test_manifest.json
```

---

# 15. Git Provenance

重要 milestone：

```text
200baf6
2026/08/16 R2.5 Phase D: formal Candidate B train validation
```

```text
af03be5
2026/08/17 R2.5 Phase E: add final Target Test and formal evidence
```

```text
53f349e
2026/08/17 docs: replace generic README with Solar experiment guide
```

---

# 16. Current Formal Conclusion

截至 v1.0：

### 已完成

```text
Experiment A
Plant1 → Plant2
Corrected R2 / R2.5
Sigmoid output
Partial Fine-tuning Candidate B
Final Target Test completed
```

### 正式結果

```text
Without TL
MAE  = 2866.104160
MSE  = 13705909.923007
RMSE = 3702.149365
R²   = 0.657072
```

```text
Partial FT
MAE  = 2252.923495
MSE  = 14216235.103204
RMSE = 3770.442295
R²   = 0.644303
```

### 正式判定

```text
Mixed Result
```

### 尚未完成

```text
Experiment A2
Experiment B
Linear-output controlled protocol
Positive Transfer formal evidence
```

---

# 17. Research Boundary

本文件不得：

- 將未完成的 A2 / B 寫成完成
- 將 Mixed Result 改稱 Positive Transfer
- 使用 normalized metrics 取代 formal original-scale metrics
- 自行推估不存在的正式數值
- 覆寫舊 run
- 省略不利結果
- 將 single seed / single split 描述為穩定、普遍或統計顯著

正式論文語氣應採：

- 「於本次資料切分與實驗設定下」
- 「呈現較低之 MAE」
- 「誤差層面之部分改善」
- 「Mixed Result」
- 「初步呈現……趨勢」

---

# 18. 後續更新規則

當 Experiment A2 或 Experiment B 產生新正式 run 時：

1. 保留 v1.0 既有 Experiment A 結果。
2. 新增新的 Experiment section，不覆寫舊結果。
3. 填入新的 run ID。
4. 填入 original-scale MAE / MSE / RMSE / R²。
5. 填入 Validation selection evidence。
6. 填入 Final Test gate 狀態。
7. 填入 Transfer classification。
8. 更新 Git commit hash。
9. 若為 Positive Transfer，明確標示所屬 protocol 與 Target Dataset。
10. 不跨不同 Target Dataset 直接比較 MAE / RMSE 絕對值大小。

---

**End of `results_summary.md` v1.0**
