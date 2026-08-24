# Solar LSTM — Optimized Transfer Learning Strategy v1｜正式結果卡

**Project:** LSTM研究與實驗－SolarEnergy 太陽能發電預測（智慧能源）
**Document type:** Formal Result Card / Method Consolidation
**Version:** v1.0
**Experiment:** B2 — Post-Test Supplementary / Exploratory Tuning
**Run ID:** `20260822T120239Z_seed1234`
**Direction:** `Plant2 → Plant1`
**Selected candidate:** `B2-P1`
**Strategy:** `Last LSTM + Output Fine-tuning`
**Execution Git HEAD:** `2148396d00a904f457ec5a64d6b80369303970ca`
**Status:** **Supplementary Positive Transfer / Solar B2 consolidated; no additional Solar training required**

---

## 1. 問題與比較目的

本結果卡用於彙整 Solar Power Dataset 之 `Plant2 → Plant1` LSTM 遷移學習受控調整結果，核心比較為：

> **LSTM Without Transfer Learning (WOTL)**
> vs.
> **LSTM + Optimized Transfer Learning (B2-P1)**

目標是在相同 Target Dataset、相同資料切分、相同特徵、相同 window / horizon、相同 scaler 流程與相同 Test 區間下，檢查經受控 Partial Fine-tuning 後之 Transfer Learning 模型，是否同時達成：

- `MAE_TL < MAE_WOTL`
- `MSE_TL < MSE_WOTL`
- `RMSE_TL < RMSE_WOTL`
- `R²_TL > R²_WOTL`

四項條件同時成立，依本專案判定規則視為正向遷移。

---

## 2. Dataset 與資料流程

| Item | Setting |
|---|---|
| Source Dataset | Plant2 |
| Target Dataset | Plant1 |
| Direction | Plant2 → Plant1 |
| Target | `DC_POWER` |
| Frequency | 15 minutes |
| Window | 5 |
| Horizon | 1 |
| Input duration | Past 75 minutes |
| Forecast point | Next 15-minute point |
| Train / Validation / Test | Chronological split; no shuffle; no split crossing |
| Target sequences | Train 516 / Validation 126 / Test 2,607 |
| Feature scaler | Training-only fit |
| Target scaler | Training-only fit, separated from feature scaler |
| Validation / Test | Transform only |
| Official metrics | Target inverse-transform 後之 original-scale MAE / MSE / RMSE / R² |
| Metric output policy | Raw / unclipped prediction |
| Unit label | `DC_POWER (raw dataset scale; documented unit: kW)` |

### Input features

- `TIME_SIN`
- `TIME_COS`
- `IRRADIATION`
- `AMBIENT_TEMPERATURE`
- `MODULE_TEMPERATURE`

---

## 3. Optimized Transfer Learning 方法設定

### 3.1 Model structure

| Layer Index | Layer | B2-P1 role |
|---:|---|---|
| 0 | Input | Input |
| 1 | TimeDistributed(Dense, 10 units) | Transferred, Frozen |
| 2 | LSTM, 60 units | Transferred, Frozen |
| 3 | BatchNormalization | Transferred, Frozen |
| 4 | LSTM, 60 units | Transferred, **Trainable** |
| 5 | BatchNormalization | Transferred, Frozen |
| 6 | Dense, 1 unit, **Linear activation** | Target output, **Trainable** |

**Transferred layers:** 1, 2, 3, 4, 5
**Trainable layers:** 4, 6
**Frozen layers:** 1, 2, 3, 5
**BatchNormalization:** Frozen
**Output activation:** Linear

此設定僅開放最後一個 recurrent block 與 output head，因此正式稱為：

> **Partial Fine-tuning / Partial FT**

不得稱為 Full Fine-tuning。

### 3.2 Parameter contract

- Total parameters: **46,681**
- Trainable parameters: **29,101**
- Trainable ratio: **62.340138%**

### 3.3 Training setting

| Item | B2-P1 |
|---|---:|
| Learning rate | `1e-5` |
| Batch size | 128 |
| Loss | MSE |
| Maximum epochs | 500 |
| Best epoch | **500** |
| Best Validation loss | **0.096119724214077** |
| Shuffle | False |
| Seed | 1234 |
| Selection split | Validation |
| Test used for epoch selection | False |

---

## 4. Validation 結果（Original Scale）

| Metric | B2-P1 |
|---|---:|
| MAE ↓ | **19,442.8612** |
| MSE ↓ | **810,674,279.2778** |
| RMSE ↓ | **28,472.3424** |
| R² ↑ | **0.920729** |
| n | 126 |

本次 Validation 設定下，B2-P1 為 B2-P1 / B2-P2 / B2-P3 三個候選中四項主要指標表現最佳之候選；checkpoint 依 Validation 鎖定後再進行 supplementary Test comparison。

---

## 5. Test Original-Scale 正式比較

| Method | MAE ↓ | MSE ↓ | RMSE ↓ | R² ↑ | n |
|---|---:|---:|---:|---:|---:|
| LSTM Without TL | 28,769.9078 | 1,392,464,754.3387 | 37,315.7441 | 0.820814 | 2,607 |
| **LSTM + Optimized TL (B2-P1)** | **12,964.0142** | **550,250,743.1286** | **23,457.4241** | **0.929192** | 2,607 |

### Relative change vs. WOTL

| Metric | Change | Better? |
|---|---:|---|
| MAE | **−54.939%** | Yes |
| MSE | **−60.484%** | Yes |
| RMSE | **−37.138%** | Yes |
| R² | **+0.108378 absolute** | Yes |

四項主要指標均符合本專案 Positive Transfer 判定條件。

> **Result classification: Supplementary Positive Transfer**

---

## 6. 重要診斷與限制

### 6.1 Test reuse disclosure

本次 Experiment B2 屬於：

> **Post-Test Supplementary / Exploratory Tuning**

Plant1 Test 在先前 Formal Experiment B 已經揭露，B2 再次使用相同 Test 區間進行 supplementary comparison。因此，本結果：

- **不取代**已封存之 Formal Experiment B 結果；
- **不是**新的 untouched confirmatory Final Test；
- 應主要定位為 **method-development / optimized-strategy evidence**；
- 可用來定義後續跨資料集驗證之方法原則，但不可單憑此 Solar B2 結果宣稱普遍泛化能力。

### 6.2 Negative prediction diagnostic

Raw / unclipped Test prediction 中：

- WOTL negative prediction count: **21**
- B2-P1 negative prediction count: **610**

因此，雖然 B2-P1 在 MAE、MSE、RMSE 與 R² 四項主要指標均顯著優於 WOTL，但 Linear output 仍產生較多負值預測。此現象不改變四項主要評估指標所呈現的正向遷移結果，但屬於太陽能發電預測之**物理合理性限制**，正式論文中應保留為限制／討論事項，不應透過事後 clipping 改寫本次正式 original-scale metrics。

### 6.3 Single-run interpretation

本次為單一 seed、單次資料切分與單一測試區間，因此論文用語應採：

> 「於本次資料切分與實驗設定下，Optimized Transfer Learning 呈現較低之 MAE、MSE 與 RMSE，且 R² 較高，呈現正向遷移結果。」

不使用「穩定改善」、「統計顯著」、「普遍有效」或「證明模型最佳」等敘述。

---

## 7. 論文可用結果敘述

### Chapter 4 — Solar result

> 於本次 Solar Plant2→Plant1 資料設定下，Optimized Transfer Learning 模型之 MAE、MSE 與 RMSE 均低於 Without Transfer Learning，且 R² 較高。其中 MAE 由 28,769.91 降至 12,964.01，RMSE 由 37,315.74 降至 23,457.42，R² 則由 0.82081 提升至 0.92919，顯示於本次資料切分與實驗設定下，經受控 Partial Fine-tuning 後之遷移學習模型在誤差與解釋能力兩方面均呈現改善，形成正向遷移結果。惟本次 B2 為既有 Test 揭露後之 supplementary tuning，因此此結果主要作為 optimized-strategy evidence，而非新的 untouched confirmatory Test。

### Recommended main table

| Method | MAE ↓ | MSE ↓ | RMSE ↓ | R² ↑ | Result |
|---|---:|---:|---:|---:|---|
| LSTM Without TL | 28,769.91 | 1.392 × 10⁹ | 37,315.74 | 0.82081 | Baseline |
| **LSTM + Optimized TL** | **12,964.01** | **5.503 × 10⁸** | **23,457.42** | **0.92919** | **Positive Transfer*** |

\* Internal audit note: Solar B2 為 post-Test supplementary comparison；四項數值型判定條件皆成立，但不是新的 untouched Final Test。

---

## 8. 方法意義與後續驗證定位

Solar B2-P1 所要保留的不是單一數值超參數，而是以下方法原則：

1. Source pre-training 後遷移可重用之 temporal representation。
2. Target adaptation 時凍結較早期 representation layers。
3. BatchNormalization layers 初始維持 frozen。
4. 僅 fine-tune 最後 temporal / recurrent block。
5. Target output head 維持 trainable / target-specific。
6. Learning rate 採較保守的 fine-tuning 級距，由 Validation 選擇。
7. Scaler 僅在 Training fit，Validation / Test 僅 transform。
8. Model / checkpoint 以 Validation 鎖定後再進行 Test。
9. 正式指標以 target inverse-transform 後之 original scale 計算。
10. 下一資料集驗證應移植「方法原則」，而不是機械複製 Solar 的 layer index 或 `1e-5`。

後續若相同方法原則於 **Aquaponics** 與 **xLSTM / xLSTMTime** 的預先鎖定 Test 上亦呈現 TL > WOTL，才可進一步支持：

> 「於本研究所評估之多個資料集與實驗設定下，所提出之 optimized transfer learning strategy 呈現一定程度之跨資料集泛化能力。」

---

## 9. Provenance

### Source checkpoint

- Candidate: `SRC_lr1e-4`
- Epoch: **500**
- SHA-256:
  `5cd3db4501d6c6720fbdd8ee3cce692b01cecb82619574f8c6944a8795182d91`

### Selected B2-P1 checkpoint

- Candidate: `B2-P1`
- Best epoch: **500**
- SHA-256:
  `a666b15cdbc3af8cf25d79f3af2080f4869ac5658a2d15d0370033da20fbabf1`

### Execution identity

- Run ID: `20260822T120239Z_seed1234`
- Git HEAD: `2148396d00a904f457ec5a64d6b80369303970ca`

---

# Final Decision

**Solar LSTM — Optimized Transfer Learning Strategy v1 = CONSOLIDATED**

- B2-P1 selected.
- Four primary Test metrics outperform fixed WOTL.
- Numerical transfer classification: **Supplementary Positive Transfer**.
- Formal Experiment B remains preserved and sealed.
- Test reuse disclosure retained.
- No additional Solar training is required for this method specification.
- Next research value should come from pre-locked cross-dataset validation rather than further Solar tuning.
