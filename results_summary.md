# Solar LSTM Results Summary

**文件版本：** v2.0

**更新日期：** 2026-09-14

**適用專案：** LSTM研究與實驗－SolarEnergy 太陽能發電預測（智慧能源）

**文件性質：** Formal Results Summary / Experiment Governance Summary

**目前狀態：** Experiment A（Plant1→Plant2）舊 Corrected R2/R2.5 已封存為 Mixed Result；Experiment B（Plant2→Plant1）Formal Linear run 已完成並封存為 Negative Transfer；Experiment B2 已完成 supplementary optimization，B2-P1 為 selected optimized strategy，數值型結果為 Supplementary Positive Transfer。Experiment B 模型開發已 CLOSED。Repository governance Steps 1–4 與 Step 5A 已完成；Step 5B external archive transfer 尚未開始。

---

## 1. 文件目的與證據優先序

本文件集中整理 Solar Power Dataset 之 LSTM / Transfer Learning 正式結果，並區分 Legacy、Corrected R2/R2.5、Linear Formal Experiment B 與 Post-Test Supplementary Experiment B2。

本文件不取代 raw data、實際執行程式、run manifest、scaler/checkpoint artifacts、prediction CSV、metrics JSON、selection / authorization / final state 或 test reuse disclosure。若內容衝突，優先依據：實際程式與資料 → split/scaler 程式 → params/scaler/checkpoint/predictions → original-scale metrics → training log → results summary / protocol / README。

---

## 2. Current Data Contract

正式預測目標：`DC_POWER`

正式輸入特徵：

- `TIME_SIN`
- `TIME_COS`
- `IRRADIATION`
- `AMBIENT_TEMPERATURE`
- `MODULE_TEMPERATURE`

Sequence：`window=5`、`horizon=1`、frequency=`15 minutes`，亦即過去 75 分鐘預測下一個 15 分鐘時間點。

Source profile rows：Train 2088 / Validation 523 / Test 653；sequences：2083 / 518 / 648。

Target profile rows：Train 521 / Validation 131 / Test 2612；sequences：516 / 126 / 2607。

Feature scaler 與 target scaler 分開，皆只使用各 profile 的 Training split fit；Validation/Test transform only；各 split 內獨立建 sequence；正式 metrics 為 target inverse-transform 後的 original-scale MAE / MSE / RMSE / R²。

---

## 3. Experiment A — Plant1 → Plant2（Historical Corrected R2 / R2.5）

### 3.1 R2 Formal Run

Run ID：`20260814T150304Z_seed1234`

WOTL：MAE 2866.104160；MSE 13705909.923007；RMSE 3702.149365；R² 0.657072。

TL Freeze：MAE 2879.526879；MSE 24312780.909834；RMSE 4930.799216；R² 0.391683。

TL Full Fine-tuning：MAE 2405.355655；MSE 16887667.836972；RMSE 4109.460772；R² 0.577463。

### 3.2 R2.5 Partial FT

Run ID：`20260816T073233Z_seed1234`

Strategy：`partial_target_adapters_last_lstm`；Legacy output activation = Sigmoid。

Final Test：MAE 2252.923495；MSE 14216235.103204；RMSE 3770.442295；R² 0.644303；n=2607。

相較 WOTL：MAE 改善，但 MSE / RMSE / R² 未同步改善，因此正式分類為 **Mixed Result**。

Experiment A Test 已揭露，原 run 保留、不得覆寫；不得再以同一 Test 反覆調整 activation / LR / layers / BN / scaler / architecture 後重新宣稱為 untouched confirmatory result。

### 3.3 Experiment A2 — Linear Formal Status

Direction：Plant1 → Plant2

Protocol：Linear Formal / `solar-linear-v1.0`

Status：**NOT STARTED / PENDING**

目前沒有 A2 formal training result、validation-selected result 或 Final Test result；不得新增 A2 metrics，也不得將 Historical Experiment A / R2 / R2.5 指標改寫成 A2 Formal Linear 結果。

---

## 4. Linear Formal Experiment B — Plant2 → Plant1

### 4.1 Formal Contract

Protocol：`solar-linear-v1.0`

Direction：Plant2 → Plant1

Source：Plant2 `source_profile`

Target：Plant1 `target_profile`

Output activation：**Linear**

Formal run ID：`20260821T041718Z_seed1234`

Formal namespace：`reports/Solar Energy Result/Linear_Formal/Experiment_B/20260821T041718Z_seed1234/`

Source / WOTL / Partial FT 均使用相同 Linear output、相同 Target split、features、window/horizon、scaler flow、batch size、optimizer/loss、callback policy、seed 與 original-scale evaluation；主要差異限於初始化權重、trainable layers 與預先登錄之 learning-rate candidate。

### 4.2 Locked Validation Selection

Selected Source：`SRC_lr1e-4`，epoch 500，SHA-256 `5cd3db4501d6c6720fbdd8ee3cce692b01cecb82619574f8c6944a8795182d91`。

Selected WOTL：`WOTL_lr1e-4`，epoch 499，SHA-256 `b8162994b7138cc619a79811b470c05087e329e5087813f4229cf0819a3a56b6`。

Selected Formal PFT：`PFT_lr3e-5`，epoch 4，SHA-256 `1d14bdb481a362e8e974d42f44bc83dffa7034b4ea3e19ac30196c316463dc37`。

Test 在 selection 完成後才經 one-time gate 存取；`test_metrics_used_for_selection=false`。

### 4.3 Formal Final Test

WOTL：MAE **28,769.9078**；MSE **1,392,464,754.3387**；RMSE **37,315.7441**；R² **0.820814**。

Formal PFT：MAE **36,009.4171**；MSE **2,149,426,000.9080**；RMSE **46,361.9025**；R² **0.723406**。

四項主要指標相較 WOTL 整體退步，因此：

> **Formal Experiment B classification = Negative Transfer**

Formal B 為歷史正式 Final Test，必須永久保留，不得被 B2 覆寫或重新命名。

---

## 5. Experiment B2 — Post-Test Supplementary / Exploratory Tuning

### 5.1 Governance

Run ID：`20260822T120239Z_seed1234`

Plant1 Test 已於 Formal B 揭露，因此 B2 明確定位為 **Post-Test Supplementary / Exploratory Tuning**；`formal_final_test_replacement=false`、`plant1_test_reused=true`。

B2 的研究用途是 method-development / optimized-strategy evidence，而非新的 untouched confirmatory Final Test。

### 5.2 Candidate Design

- B2-P1：Last LSTM + target output head，LR=`1e-5`
- B2-P2：Last LSTM + target output head，LR=`3e-6`
- B2-P3：Output head only，LR=`1e-5`

共同設定：Linear output、BatchNormalization frozen、batch=128、MSE、max epochs=500、shuffle=False、seed=1234。

### 5.3 Selected Candidate — B2-P1

Transferred layers：1,2,3,4,5

Trainable layers：4,6

Frozen layers：1,2,3,5

BatchNormalization：Frozen

Output activation：Linear

正式名稱：**Partial Fine-tuning / Partial FT**（不得稱 Full Fine-tuning）。

Total params：46,681；trainable params：29,101；trainable ratio：約 62.34%。

Best epoch：500；best validation loss：0.096119724214077。

Validation original-scale：MAE 19,442.8612；MSE 810,674,279.2778；RMSE 28,472.3424；R² 0.920729；n=126。

Selected B2-P1 checkpoint SHA-256：`a666b15cdbc3af8cf25d79f3af2080f4869ac5658a2d15d0370033da20fbabf1`。

Execution Git HEAD：`2148396d00a904f457ec5a64d6b80369303970ca`。

### 5.4 Supplementary Test Comparison

| Method | MAE ↓ | MSE ↓ | RMSE ↓ | R² ↑ | n |
|---|---:|---:|---:|---:|---:|
| WOTL | 28,769.9078 | 1,392,464,754.3387 | 37,315.7441 | 0.820814 | 2607 |
| B2-P1 Optimized TL | 12,964.0142 | 550,250,743.1286 | 23,457.4241 | 0.929192 | 2607 |

Relative change：MAE −54.939%；MSE −60.484%；RMSE −37.138%；R² +0.108378 absolute。

數值型四項判定條件均成立，因此：

> **B2-P1 classification = Supplementary Positive Transfer**

### 5.5 Diagnostic Limitations

Raw / unclipped prediction：WOTL negative count=21；B2-P1 negative count=610。此為 Linear output 缺乏非負物理限制之模型限制；不得透過事後 clipping 改寫正式 metrics。

本結果為 single seed / single split / single Test interval，論文不得宣稱 statistically significant、stable improvement、universally effective 或 proven best。

---

## 6. Figure / Thesis Evidence

Read-only thesis figure post-processing 已完成；未重新 training、未 model.predict、未重新選 checkpoint、未 clipping、未修改 Formal artifacts。

十張 raw figures：WOTL/B2-P1 Learning Curve、Prediction Plot、YY Plot、Residual Plot、Error Histogram；另有 `metrics_recheck.json` 與 `figure_manifest.json`。

Test alignment：2,607 rows；timestamp identical；y_true identical；metrics recalculation PASS。

第四章 Solar Plant2→Plant1 已完成 WOTL、Optimized TL、original-scale metrics、B2-P1 Supplementary Positive Transfer、Learning/Prediction/YY/Residual/Histogram 分析與 Cross-reference audit。

---

## 7. Current Closure Status

### Scientific / Model-Development Closure

**Experiment B = SEALED / CLOSED**

- Formal B preserved = Negative Transfer
- B2 selected strategy = B2-P1
- B2 numerical result = Supplementary Positive Transfer
- NO MORE TUNING
- NO MORE TEST ACCESS FOR MODEL DEVELOPMENT
- NO MORE B2 CANDIDATE SEARCH

允許：read-only audit、論文引用、已有 prediction descriptive analysis、SHA/provenance verification、documentation、oral-defense QA。

### Repository / Archival Closure

- Step 1 = COMPLETE
- Step 2 = COMPLETE
- Step 3 = COMPLETE
- Step 4 = COMPLETE
- Step 5A = COMPLETE
- Step 5B transfer = NOT STARTED

本次變更前，Step 5B 的唯一阻塞為 documentation dirty-file gate repair。本次 controlled commit 完成後該文件阻塞即解除，但 external archive transfer 仍須另行執行與驗證；本文件修復本身不代表 archive backup、Google Drive upload 或外接 HDD archive 已完成。

---

## 8. Evidence Index — Experiment B

Formal B：

`reports/Solar Energy Result/Linear_Formal/Experiment_B/20260821T041718Z_seed1234/`

主要 evidence：`run_manifest.json`、`selection/`、Source/WOTL/PFT candidate records、`final_test/` predictions / metrics / authorization / final state。

B2：

`reports/Solar Energy Result/Linear_Supplementary/Experiment_B2/20260822T120239Z_seed1234/`

主要 evidence：`run_manifest.json`、`test_reuse_disclosure.json`、`baseline/`、`candidates/B2-P1/`、`summary/`、Optimized Strategy/Result Card。

Thesis figures：

`reports/Solar Energy Result/Thesis_Figures/Experiment_B2_Plant2_to_Plant1/`

---

## 9. Update Rules

1. Formal B 與 B2 永久分開記錄。
2. 不刪除、不覆寫不利結果。
3. 不把 B2 說成新的 untouched Final Test。
4. 不以 Plant1 Test 再做 model development。
5. 若 repository audit 發現 artifact 缺漏，只做 archival / documentation 修復；任何會改變模型、prediction 或 metrics 的操作須另行核准。
6. Experiment A 後續新工作不得自動沿用 B2 的 Test-guided tuning；應先完成其資料與 scaler handoff / preflight。

---

**End of `results_summary.md` v2.0**
