# LSTM Transfer Learning for Solar Power Forecasting

> Solar Power Generation Dataset × LSTM × Transfer Learning × Reproducible Time-Series Experimentation

## Project Overview

本專案為碩士論文研究中 **Solar Power Dataset 的 LSTM 實驗分支**，主要用於建立、復現與查核太陽能發電功率預測之 LSTM 與 Transfer Learning（TL）流程。

本 repository 已由原始通用的 time-series transfer learning 專案，逐步重構為 Solar Power Generation Dataset 專屬實驗環境，研究內容包含：

- LSTM Without Transfer Learning baseline
- LSTM Transfer Learning
- Plant1 → Plant2
- Plant2 → Plant1
- Freeze
- Full Fine-tuning
- Partial Fine-tuning
- chronological split
- Training-only scaler
- feature / target scaler separation
- inverse transform
- normalized-scale / original-scale metrics
- Test-set integrity
- run manifest / prediction / metrics / figures / Git provenance
- Legacy 與 Corrected 實驗差異保存

本 repository 目前只處理 **Solar Power Dataset 的 LSTM 實驗分支**；xLSTM 正式實驗由其他研究分支處理。

---

## Research Context

本專案隸屬碩士論文：

**中文題目**

《基於 xLSTM 模型之遷移學習應用研究：以 AI 智慧養殖與綠色能源為例》

**English Title**

*A Study on the Application of Transfer Learning Based on the xLSTM Model:  
Case Studies in AI-Driven Animal Production and Green Energy*

Solar LSTM 分支主要提供：

- 實驗方法設定
- 實驗復現紀錄
- scaler / inverse-transform 查核
- original-scale metrics
- Transfer Learning 比較結果
- prediction / learning curve / residual 等圖表
- Transfer Learning 判定
- 可供論文第三章、第四章與第五章使用之實驗素材

---

# Dataset

本研究使用公開 **Solar Power Generation Dataset** 中的兩座太陽能電廠：

- Plant 1
- Plant 2

兩座場站均包含：

1. Generation Data
2. Weather Sensor Data

## Raw Generation Fields

```text
DATE_TIME
PLANT_ID
SOURCE_KEY
DC_POWER
AC_POWER
DAILY_YIELD
TOTAL_YIELD
```

## Raw Weather Fields

```text
DATE_TIME
PLANT_ID
SOURCE_KEY
AMBIENT_TEMPERATURE
MODULE_TEMPERATURE
IRRADIATION
```

---

# Prediction Task

正式預測目標為：

```text
DC_POWER
```

亦即：

> 太陽光電系統之直流側發電功率。

正式模型輸入特徵為：

```text
TIME_SIN
TIME_COS
IRRADIATION
AMBIENT_TEMPERATURE
MODULE_TEMPERATURE
```

目前 fixed R2 contract 中，歷史 `DC_POWER` **不是**模型輸入特徵。

---

# Corrected Data Preprocessing

Legacy Solar 實驗曾存在：

- full-data scaler fit
- Source Test 使用方式不嚴謹
- prediction alignment
- normalized / original scale 混淆

等問題。

Corrected preprocessing 採以下流程：

```text
Raw Generation Data
        +
Raw Weather Sensor Data
        ↓
Generation 依 DATE_TIME 聚合
Weather 依 DATE_TIME 聚合
        ↓
Merge
        ↓
建立固定 15-minute timeline
        ↓
建立 Source / Target Profile
        ↓
Chronological Train / Validation / Test Split
        ↓
Split-isolated Missing-Value Handling
        ↓
Training-only Feature Scaler
Training-only Target Scaler
        ↓
Validation / Test Transform Only
        ↓
Sliding Window / Sequence Construction
        ↓
LSTM Training / Validation
        ↓
Selected Checkpoint
        ↓
Final Test
        ↓
Target Scaler Inverse Transform
        ↓
Original-Scale Metrics
```

---

# Timeline

資料時間解析度：

```text
15 minutes
```

完整時間範圍：

```text
2020-05-15 00:00
→
2020-06-17 23:45
```

完整 timeline：

```text
3264 rows
```

---

# Source / Target Profiles

Plant1 與 Plant2 皆可依 Transfer Learning 方向扮演 Source 或 Target。

## Source Profile

```text
Training   : 2088 rows
Validation : 523 rows
Test       : 653 rows
```

## Target Profile

```text
Training   : 521 rows
Validation : 131 rows
Test       : 2612 rows
```

Corrected protocol 中：

> Source Test 保持為獨立 holdout，不再重新合併回 Source Training / Validation。

---

# Missing-Value Handling

Missing-value handling 僅能在各 split 內完成，不跨越：

```text
Training
Validation
Test
```

邊界。

## Weather Features

各 split 內依序：

```text
Time interpolation
→ Forward fill
→ Backward fill
```

## DC_POWER

- 夜間 missing value → `0`
- 日間 missing value → 同一日期內 interpolation
- 不跨日期 interpolation
- 不跨 split interpolation
- residual missing value → `0`

---

# Scaling Policy

Feature scaler 與 target scaler 分開建立。

## Feature Scaler

只縮放：

```text
IRRADIATION
AMBIENT_TEMPERATURE
MODULE_TEMPERATURE
```

## Target Scaler

只縮放：

```text
DC_POWER
```

## Cyclic Time Features

```text
TIME_SIN
TIME_COS
```

不進行 MinMax scaling。

## Fit Rule

```text
Scaler.fit()
→ Training only
```

```text
Scaler.transform()
→ Validation / Test only
```

Validation / Test normalized values 若出現：

```text
< 0
或
> 1
```

不進行 clipping。

這代表資料超出 Training scaler 所觀察到的範圍，而不是 scaler 錯誤。

---

# Sequence Construction

正式 R2 / R2.5：

```text
window  = 5
horizon = 1
```

15 分鐘資料下可解讀為：

```text
過去 5 筆資料（75 分鐘）
→
預測下一個 15 分鐘時間點
```

正式 alignment：

```text
X[i : i + window]
→
y[i + window + horizon - 1]
```

Sequence 在每個 split 中獨立建立，不跨 split boundary。

## Sequence Counts

### Source

```text
Training   : 2083
Validation : 518
Test       : 648
```

### Target

```text
Training   : 516
Validation : 126
Test       : 2607
```

---

# LSTM Architecture

R2 / R2.5 為 **Corrected Legacy Reproduction**，因此保留 Legacy LSTM 架構：

```text
Input
↓
TimeDistributed Dense(10)
↓
LSTM(60, return_sequences=True)
↓
BatchNormalization
↓
LSTM(60)
↓
BatchNormalization
↓
Dense(1, activation="sigmoid")
```

主要設定包含：

- L2 regularization
- Glorot initialization
- Orthogonal recurrent initialization
- Adam optimizer
- MSE loss
- Sigmoid output

---

# Output Activation Note

R2 / R2.5 **實際執行模型使用 Sigmoid output**。

Training-only MinMaxScaler 不保證未來 Validation / Test target 一定落在：

```text
[0, 1]
```

因此 Test target 超出 Training target 最大值時，Sigmoid 存在理論上的輸出上限。

這項限制屬於：

> R2 / R2.5 Corrected Legacy Reproduction 的既有架構限制。

若後續建立新的 Formal Fair Comparison protocol，可事先定義 Linear regression head；但不得用 Linear 回頭改寫已完成之 Experiment A。

---

# Transfer Learning Strategies

## Without Transfer Learning

Target model 由隨機初始化開始訓練，不載入 Source pretrained weights。

## TL Freeze

載入 Source pretrained weights 後，Transferred layers 維持 frozen。

## TL Full Fine-tuning

載入 Source pretrained weights 後，既定 transferred layers 全部允許更新。

## TL Partial Fine-tuning

R2.5 Candidate B：

```text
partial_target_adapters_last_lstm
```

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

Transferred：

```text
Layer 2–5
```

Trainable：

```text
Layer 1
Layer 4
Layer 6
```

Frozen：

```text
Layer 2
Layer 3
Layer 5
```

BatchNormalization：

```text
Layer 3 frozen
Layer 5 frozen
```

Trainable parameters：

```text
29161 / 46681
≈ 62.47%
```

因此此策略稱為：

> **Partial Fine-tuning / Partial FT**

而不是 Full Fine-tuning。

---

# Experiment Design

## Experiment A

```text
Plant1 → Plant2
```

```text
Source = Plant1
Target = Plant2
```

## Experiment B

```text
Plant2 → Plant1
```

```text
Source = Plant2
Target = Plant1
```

---

# Fair Comparison Principles

同一 Target Dataset 進行正式比較時，須固定：

1. Target Dataset
2. Training / Validation / Test
3. Input features
4. window / horizon
5. scaler flow
6. Test interval
7. evaluation scale
8. metric definitions
9. Test-set usage policy

Without TL 與 TL 原則上主要只在：

```text
Initial weights
Trainable layers
```

產生差異。

若後續再調整：

- Learning rate
- Batch size
- Dropout
- L2
- Fine-tuning strategy

應標示為：

> supplementary controlled tuning

不得與主要公平比較混為一談。

---

# Evaluation Metrics

正式論文結果以 **inverse transform 後的 original-scale metrics** 為主：

```text
MAE
MSE
RMSE
R²
```

Normalized-scale metrics 僅作：

- training diagnostics
- scaler diagnostics
- debugging

不得與 original-scale metrics 混表。

另外：

```text
Normalized-scale RMSE
≠
nRMSE
```

若本研究沒有另外定義 nRMSE normalization factor，不應將 normalized data 上直接計算的 RMSE 稱為 nRMSE。

---

# Experiment A — Formal Results

## R2 Formal Run

Run ID：

```text
20260814T150304Z_seed1234
```

Direction：

```text
Plant1 → Plant2
```

## Target Original-Scale Test Metrics

| Method | MAE | RMSE | R² | Interpretation |
|---|---:|---:|---:|---|
| Without TL | 2866.104160 | 3702.149365 | 0.657072 | Baseline |
| TL Freeze | 2879.526879 | 4930.799216 | 0.391683 | Negative / no improvement |
| TL Full Fine-tuning | 2405.355655 | 4109.460772 | 0.577463 | Not positive overall |

> README 僅列目前已正式確認的 original-scale values，不由摘要值推估缺少的正式 metrics。

---

# R2.5 Partial Fine-tuning

## Candidate B

```text
partial_target_adapters_last_lstm
```

## Phase A — Contract / Design

```text
PASS
```

## Phase B — Zero-Epoch Structural Check

```text
PASS
```

## Phase C — Smoke Validation

```text
PASS
```

已確認：

- 指定 trainable layers 才會更新
- Frozen LSTM 不更新
- Frozen BatchNormalization 不更新
- checkpoint reload PASS
- Test untouched

## Phase D — Formal Training / Validation

Run：

```text
20260816T073233Z_seed1234
```

設定：

```text
Seed       : 1234
Batch size : 128
Initial LR : 1e-5
Loss       : MSE
Max epochs : 500
Device     : CPU
```

Best epoch：

```text
500
```

Best validation loss：

```text
0.1819697022
```

Validation original-scale：

| Metric | Value |
|---|---:|
| MAE | 3702.256024 |
| MSE | 39303185.752356 |
| RMSE | 6269.225291 |
| R² | 0.020418 |

Phase D 完成後先鎖定 Validation selection，再進入 Test。

---

# Phase E — Final Target Test

Test sequences：

```text
2607
```

Original-scale metrics：

| Metric | Value |
|---|---:|
| MAE | 2252.923495 |
| MSE | 14216235.103204 |
| RMSE | 3770.442295 |
| R² | 0.644303 |

相較 Without TL：

| Metric | Change |
|---|---:|
| MAE | improved approximately 21.39% |
| MSE | worsened approximately 3.72% |
| RMSE | worsened approximately 1.84% |
| R² | decreased |

## Transfer Classification

```text
Mixed Result
```

Interpretation：

> Under the current data split and experimental configuration, Partial Fine-tuning Candidate B reduced MAE relative to the Without-TL baseline, while MSE, RMSE, and R² did not improve simultaneously. Therefore, this run is classified as a **Mixed Result**, rather than Positive Transfer.

Experiment A 在 Phase E one-time Target Test 後視為：

```text
SEALED
```

Plant2 Test 已被揭露，因此不得再依 Phase E Test 結果修改：

- learning rate
- trainable layers
- BatchNormalization policy
- epochs
- activation
- scaler
- architecture

後再將同一 Test 當作新的 confirmatory Test。

---

# Experiment B — Status

Direction：

```text
Plant2 → Plant1
```

目前 Corrected reciprocal formal experiment：

```text
Pending
```

Repository 中即使存在 Legacy Experiment B 結果，也不得因此宣稱目前 Corrected reciprocal Experiment B 已完成。

---

# Test-Set Integrity

正式流程：

```text
Training
↓
Validation
↓
Best Checkpoint Selection
↓
Lock Selection
↓
Final Test
```

Test 不得用於：

- Learning-rate selection
- Epoch selection
- Activation selection
- Trainable-layer selection
- Scaler fitting
- Early stopping
- Checkpoint selection

---

# Transfer Learning Classification

## Positive Transfer

```text
MAE ↓
MSE ↓
RMSE ↓
R² ↑
```

## Partial Positive Transfer

主要 error metrics 改善，但 R² 未同步改善或仍為負值。

## Negative Transfer

主要評估指標整體退步。

## Mixed Result

不同 metrics 呈現不同方向，無法支持整體 Positive 或 Negative 判定。

單一 metric 改善不足以宣稱 Transfer Learning 全面成功。

---

# DC_POWER Scale / Unit Note

資料集欄位說明通常將 `DC_POWER` 標示為功率量，並以 kW 描述。

但 Plant1 與 Plant2 raw values 在 `DC_POWER / AC_POWER` 關係上存在明顯尺度差異，因此目前：

- 保留 raw `DC_POWER`
- 不自行 `/10`
- 不自行 `/1000`
- 不直接將 DC / AC ratio 解釋為 inverter efficiency

保守圖表文字建議：

```text
DC_POWER (raw dataset scale; documented unit: kW)
```

正式 Transfer Learning 效果應在：

> **相同 Target Dataset**

內比較，而不是直接跨 Plant 比較 MAE / RMSE 絕對數值。

---

# Repository Structure

Representative structure：

```text
LSTM_TransferLearning_SolarEnergy/
│
├─ dataset/
│
├─ preprocess/
│
├─ notebook/
│
├─ r2_config/
│   ├─ solar_r2.py
│   └─ solar_r25.py
│
├─ r2_helpers/
│   ├─ solar_data.py
│   ├─ solar_formal.py
│   ├─ solar_partial_formal.py
│   ├─ solar_partial_ft.py
│   ├─ solar_partial_smoke.py
│   ├─ solar_partial_final.py
│   ├─ solar_runtime.py
│   └─ solar_smoke.py
│
├─ reports/
│   └─ Solar Energy Result/
│       ├─ R2/
│       └─ R2.5/
│
├─ tests/
│
├─ main.py
├─ model.py
├─ r2_solar.py
├─ r25_solar.py
├─ .gitignore
└─ README.md
```

若實際 repository 與 README 日後不同，應以實際 repository 為準。

---

# How to Run

## Legacy Workflow

```text
main.py
```

保留作為 Legacy / upstream compatibility 與歷史復現用途。

舊 `main.py` 執行方式不得直接視為目前 Corrected Solar Formal Protocol。

## Corrected R2.5 Workflow

目前 R2.5 CLI 包含：

```bash
python r25_solar.py contract
```

```bash
python r25_solar.py smoke
```

```bash
python r25_solar.py formal-validation
```

```bash
python r25_solar.py final-test
```

`final-test` 為正式 Test Gate。

不得：

```text
Test
→ 看結果
→ 改參數
→ 再 Test
→ 再把 Test 當正式確認結果
```

---

# Formal Experiment Outputs

Formal run 原則上保存：

- `run_id`
- model name
- Source / Target
- experiment mode
- scaler references
- Train / Validation / Test ranges
- parameters
- environment
- training history
- checkpoint provenance
- normalized metrics
- original-scale metrics
- prediction CSV
- figures
- transfer classification
- date
- seed
- Git commit hash

舊結果不得靜默覆寫。

---

# Artifact Policy

Git repository 優先保存 lightweight reproducibility evidence：

- run manifests
- metrics
- prediction CSV
- training history
- validation selection
- figures
- transfer classification

大量 binary checkpoints：

```text
checkpoint_epoch_*.hdf5
```

與 disposable R2.5 smoke artifacts 不納入一般 Git history。

其實體檔案應保留於：

- Local research storage
- Research workstation
- Formal backup storage

Legacy 已被 Git tracking 的 HDF5 不因新 `.gitignore` 規則而刪除。

---

# Current Git Milestone

Phase E implementation 與 formal evidence 已 Commit：

```text
af03be5
2026/08/17 R2.5 Phase E: add final Target Test and formal evidence
```

前一個 Phase D milestone：

```text
200baf6
2026/08/16 R2.5 Phase D: formal Candidate B train validation
```

此 commit hash 用於研究 provenance；未來 README 更新後 HEAD 可能繼續前進。

---

# Known Limitations

目前限制包括：

1. Current formal evidence 為 single seed / single chronological split。
2. Experiment A Plant2 Test 已揭露，後續不得用其調參。
3. R2 / R2.5 為 Legacy architecture reproduction，因此仍使用 Sigmoid。
4. Training-only MinMax scaling 允許 Validation / Test 出現 `< 0` 或 `> 1`。
5. Sigmoid 無法表示 normalized prediction > 1。
6. Plant1 / Plant2 raw `DC_POWER` 實際物理尺度一致性仍有疑義。
7. Candidate B 的正式結果為 Mixed Result，而不是全面 Positive Transfer。
8. Corrected Experiment B 尚未完成。
9. Legacy 與 Corrected results 同時存在，不得混用。
10. Normalized-scale metrics 與 original-scale metrics 必須明確區分。

---

# Research Interpretation Boundary

目前只有 single seed / single split，因此適合使用：

- 「於本次資料切分與實驗設定下」
- 「呈現較低之 MAE」
- 「初步呈現改善趨勢」
- 「誤差層面之部分改善」
- 「Mixed Result」

不應使用：

- statistically significant
- consistently superior
- universally effective
- proven best
- stable improvement

除非後續已有 multi-seed、平均值、標準差或統計檢定支持。

---

# Upstream Project / Acknowledgment

本 repository 最初源自：

**Transfer Learning LSTM for Time-Series Regression**

Upstream repository：

```text
dainnovation722/transfer-learning-LSTM
```

Original author：

```text
dainnovation722
```

目前 repository 已針對 Solar Power Generation Dataset 進行大幅調整，包括：

- Corrected preprocessing
- Training-only scaling
- feature / target scaler separation
- inverse-transform evaluation
- run manifest
- formal Test gating
- Partial Fine-tuning
- experiment provenance
- original-scale evaluation

原始專案 attribution 保留，以維持 repository provenance 與研究倫理。

---

# Current Project Status

## Completed

- Legacy Solar workflow audit
- Corrected preprocessing
- Training-only scaler
- Source / Target Profile
- Original-scale evaluation
- Experiment A R2 formal reproduction
- R2.5 Partial Fine-tuning Candidate B
- Phase C smoke validation
- Phase D formal Train / Validation
- Phase E one-time Target Test
- Experiment A transfer classification
- Git artifact policy
- Phase E Git provenance

## Next

- Corrected reciprocal Experiment B：Plant2 → Plant1
- Reciprocal fair-comparison review
- Solar formal figures consolidation
- `experiment_protocol`
- `run_manifest`
- `results_summary`
- Thesis Chapter 4 experiment-material consolidation

---

# Research Principle

本 repository 的目標不是：

> 尋找最好看的數據。

而是：

```text
Audit
→
Reproduce
→
Verify Split / Scaler
→
Verify Inverse Transform
→
Produce Trustworthy Original-Scale Results
→
Perform Fair Comparison
→
Apply Limited Controlled Tuning if Necessary
→
Organize Thesis Evidence
→
Seal the Experiment
```

完成標準為：

> **Experiments are reproducible, scales are correct, comparisons are fair, results are interpretable, and the research evidence is defensible.**
