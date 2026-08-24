# Solar LSTM — Optimized Transfer Learning Strategy v1

**Document type:** Method Specification / Result Card
**Project:** LSTM研究與實驗－SolarEnergy 太陽能發電預測（智慧能源）
**Version:** v1.0
**Status:** Solar B2-P1 method consolidated; no additional Solar tuning required for this specification
**Run ID:** `20260822T120239Z_seed1234`
**Direction:** `Plant2 → Plant1`
**Selected candidate:** `B2-P1`
**Strategy:** `Last LSTM + Output Fine-tuning`
**Execution Git HEAD:** `2148396d00a904f457ec5a64d6b80369303970ca`

> Purpose: extract the successful Solar B2-P1 configuration into a reusable Transfer Learning method specification. This document separates **generalizable method principles** from **Solar-specific settings**, so the same methodological idea can later be evaluated on Aquaponics and xLSTM/xLSTMTime without blindly copying Solar-specific numeric hyperparameters.

---

# A. Method Core

## A.1 Core idea

The method preserves transferable temporal representations learned from the Source domain while limiting Target-domain adaptation to the later temporal block and the Target output head.

```text
Source pre-training
        ↓
Transfer reusable temporal representation
        ↓
Freeze earlier representation layers
        ↓
Freeze BatchNormalization layers
        ↓
Fine-tune final recurrent block
        +
Target output head
        ↓
Validation-based checkpoint selection
        ↓
Target-specific inverse transform
        ↓
Original-scale evaluation
```

## A.2 Solar B2-P1 implementation

| Layer Index | Layer | Role in B2-P1 |
|---:|---|---|
| 0 | Input | Input |
| 1 | TimeDistributed(Dense, 10 units) | Transferred, Frozen |
| 2 | LSTM, 60 units | Transferred, Frozen |
| 3 | BatchNormalization | Transferred, Frozen |
| 4 | LSTM, 60 units | Transferred, **Trainable** |
| 5 | BatchNormalization | Transferred, Frozen |
| 6 | Dense, 1 unit, **Linear activation** | Target output, **Trainable** |

```text
Transferred layers = 1, 2, 3, 4, 5
Trainable layers   = 4, 6
Frozen layers      = 1, 2, 3, 5
BatchNormalization = Frozen
Output activation  = Linear
```

Parameter contract:

```text
Total parameters     = 46,681
Trainable parameters = 29,101
Trainable ratio      = 62.340138%
```

## A.3 Terminology

Only the later recurrent layer and output layer are trainable. Earlier transferred representations and normalization states remain fixed. The correct term is:

> **Partial Fine-tuning / Partial FT**

Do **not** call this Full Fine-tuning.

---

# B. Solar-specific Configuration

## B.1 Dataset mapping

| Item | Setting |
|---|---|
| Source | Plant2 |
| Target | Plant1 |
| Direction | Plant2 → Plant1 |
| Target variable | `DC_POWER` |
| Frequency | 15 minutes |
| Window | 5 |
| Horizon | 1 |
| Input duration | Past 75 minutes |
| Forecast target | Next 15-minute point |

## B.2 Features

```text
TIME_SIN
TIME_COS
IRRADIATION
AMBIENT_TEMPERATURE
MODULE_TEMPERATURE
```

Target:

```text
DC_POWER
```

No lagged `DC_POWER` is used as an input feature.

## B.3 Corrected Target split

| Split | Rows | Effective Sequences |
|---|---:|---:|
| Training | 521 | 516 |
| Validation | 131 | 126 |
| Test | 2,612 | 2,607 |

```text
Chronological split
No shuffle
No split crossing
```

## B.4 Scaler contract

```text
Feature scaler and Target scaler are separate.
Scaler fit split = Training only.
Validation = transform only.
Test = transform only.
Official metrics = after Target scaler inverse_transform.
```

Formal result scale:

> `DC_POWER (raw dataset scale; documented unit: kW)`

The dataset raw magnitude should not be manually divided by 10 or 1000 without provider-side evidence.

## B.5 Training contract

| Hyperparameter | B2-P1 setting |
|---|---|
| Optimizer | Adam |
| Loss | MSE |
| Learning rate | `1e-5` |
| Batch size | 128 |
| Maximum epochs | 500 |
| Shuffle | False |
| Seed | 1234 |
| Device policy | CPU |
| Output activation | Linear |

Callback policy:

```text
ModelCheckpoint:
  monitor = val_loss
  save_best_only = True

ReduceLROnPlateau:
  factor = 0.1
  patience = 20
  min_lr = 1e-7

EarlyStopping:
  patience = 50
  min_delta = 0
  restore_best_weights = False

TerminateOnNaN:
  Enabled
```

Best checkpoint is reloaded for evaluation.

---

# C. Transferable Principles

| Principle | Transferable? | Meaning |
|---|---|---|
| Source pre-training | Yes | Learn reusable temporal representation from Source |
| Transfer hidden temporal representation | Yes | Reuse representation rather than only training from scratch |
| Target-specific output head | Yes | Final prediction head represents Target task/output domain |
| Freeze earlier representation | Yes | Reduce destructive updates to reusable features |
| Fine-tune later temporal block | Yes | Adapt higher-level temporal representation to Target |
| Freeze normalization states initially | Yes, where applicable | Reduce instability under limited Target data |
| Conservative fine-tuning LR | Yes as a principle | Target adaptation LR should be conservative |
| Validation-based selection | Yes | Checkpoint/hyperparameter selection must not use Test |
| Target-specific scaler | Yes | Each Target domain uses its own Training-fitted scaler |
| Original-scale final metrics | Yes | Formal MAE/MSE/RMSE/R² are computed after inverse transform |

## C.1 Semantic transfer rule

When moving to another architecture, do **not** mechanically reproduce:

```text
Layer 4 + Layer 6
```

Instead reproduce the semantic idea:

```text
Final temporal/recurrent block
+
Target-specific prediction head
```

For xLSTM/xLSTMTime, layer indices and block names may differ.

---

# D. Dataset-specific Parameters

| Parameter | Solar value | Cross-dataset rule |
|---|---:|---|
| Source / Target | Plant2 → Plant1 | Dataset-specific |
| Window | 5 | Re-determine from dataset sampling/task |
| Horizon | 1 | Re-determine from forecast task |
| Frequency | 15 min | Dataset-specific |
| LSTM units | 60 / 60 | Architecture-specific |
| TimeDistributed Dense units | 10 | Architecture-specific |
| Batch size | 128 | May be tuned on Validation |
| Learning rate | `1e-5` | Starting candidate, not universal constant |
| Epoch cap | 500 | May be adapted before Test |
| Feature set | Solar weather/time features | Dataset-specific |
| Target | `DC_POWER` | Dataset-specific |
| Scaler numeric range | Target-specific MinMax fit | Refit on each dataset Training split |

Allowed cross-dataset adaptation must be decided **before Test evaluation** and selected using Validation only.

---

# E. Evaluation Rule

## E.1 Fair comparison requirement

A TL vs WOTL comparison is valid only when both methods use the same:

```text
Target Dataset
Train / Validation / Test split
Features
Window
Horizon
Scaler procedure
Test interval
Original-scale metric definitions
```

The primary methodological difference should be:

```text
Initialization / transferred weights
+
Trainable / frozen layer policy
```

## E.2 Positive Transfer decision rule

```text
MAE_TL  < MAE_WOTL
MSE_TL  < MSE_WOTL
RMSE_TL < RMSE_WOTL
R²_TL   > R²_WOTL
```

All four simultaneously satisfied:

> **Positive Transfer**

Errors improve but R² does not improve:

> **Partial Positive Transfer**

Overall metrics deteriorate:

> **Negative Transfer**

## E.3 B2-P1 Validation evidence

```text
Best epoch = 500
Best Validation loss = 0.096119724214077
```

| Metric | B2-P1 |
|---|---:|
| MAE | 19,442.8612 |
| MSE | 810,674,279.2778 |
| RMSE | 28,472.3424 |
| R² | 0.920729 |
| n | 126 |

Selected checkpoint SHA-256:

```text
a666b15cdbc3af8cf25d79f3af2080f4869ac5658a2d15d0370033da20fbabf1
```

## E.4 B2-P1 Test original-scale comparison

| Metric | WOTL | Optimized TL — B2-P1 |
|---|---:|---:|
| MAE ↓ | 28,769.9078 | **12,964.0142** |
| MSE ↓ | 1,392,464,754.3387 | **550,250,743.1286** |
| RMSE ↓ | 37,315.7441 | **23,457.4241** |
| R² ↑ | 0.820814 | **0.929192** |
| n | 2,607 | 2,607 |

Improvement relative to WOTL:

| Metric | Change |
|---|---:|
| MAE | **−54.939%** |
| MSE | **−60.484%** |
| RMSE | **−37.138%** |
| R² | **+0.108378 absolute** |

```text
MAE  better = True
MSE  better = True
RMSE better = True
R²   better = True
```

Therefore, for this B2 supplementary comparison:

> **Supplementary Positive Transfer**

Internal evidence records that the Plant1 Test had been revealed previously and was reused for B2 supplementary comparison. This does not change the observed numerical comparison, but it means this Solar B2 result should be treated primarily as **method-development / optimized-strategy evidence**, not as a new untouched confirmatory Test.

---

# F. Cross-Dataset Validation Rule

The purpose of Strategy v1 is not to keep optimizing Solar, but to test whether the same **method principle** can produce improvements on other datasets.

## F.1 Pre-lock before next dataset Test

1. Define Source and Target.
2. Define WOTL baseline.
3. Map the semantic transfer blocks.
4. Define frozen and trainable blocks.
5. Define a small Validation-only LR candidate set if needed.
6. Define window/horizon based on the new dataset.
7. Fit scalers on Training only.
8. Select model/checkpoint using Validation only.
9. Lock configuration.
10. Evaluate Test only after lock.

## F.2 Aquaponics validation target

Use the same research question:

> Does the optimized Transfer Learning strategy produce lower prediction error and higher predictive performance than Without Transfer Learning under the same Target split?

Success rule:

```text
MAE_TL  < MAE_WOTL
MSE_TL  < MSE_WOTL
RMSE_TL < RMSE_WOTL
R²_TL   > R²_WOTL
```

If reproduced on an independently locked Aquaponics Test, it provides stronger evidence that the method is not Solar-only.

## F.3 xLSTM / xLSTMTime adaptation

Transfer the **principle**, not LSTM layer numbers:

```text
Earlier temporal representation:
  Frozen initially

Final temporal / xLSTM block:
  Fine-tuned

Target prediction head:
  Trainable / target-specific

Normalization-related states:
  Frozen initially where architecture permits

Model selection:
  Validation only
```

Any xLSTM-specific structural change must be documented separately from the Solar LSTM implementation.

## F.4 Generalization claim threshold

One Solar result alone supports:

> The proposed strategy is effective under the current Solar setting.

Multiple datasets showing the same direction of improvement can support the more conservative thesis statement:

> The proposed strategy demonstrates a certain degree of cross-dataset generalizability under the evaluated datasets and experimental settings.

Do not claim universal effectiveness from single-seed / single-split experiments.

---

# G. Thesis Positioning

## G.1 Chapter 3 — Method

> 本研究採用受控之遷移學習微調策略，先利用來源資料域進行模型預訓練，再將可重用之時間序列表徵遷移至目標模型。目標域訓練階段凍結前段表徵層與正規化層，並僅針對後段時間序列層及目標輸出層進行 Partial Fine-tuning，以降低有限目標資料下大幅更新模型參數所造成之不穩定性。模型選擇以 Validation set 為依據，正式評估則於 target scaler inverse transform 後，以 original-scale MAE、MSE、RMSE 與 R² 進行比較。

## G.2 Chapter 4 — Solar result

Main presentation:

```text
LSTM Without Transfer Learning
vs.
LSTM + Optimized Transfer Learning
```

> 於本次 Solar Plant2→Plant1 資料設定下，Optimized Transfer Learning 模型之 MAE、MSE 與 RMSE 均低於 Without Transfer Learning，且 R² 較高，顯示遷移學習在此設定下能降低預測誤差並提升模型對目標資料變異之解釋能力，呈現正向遷移結果。

Recommended main table:

| Method | MAE ↓ | MSE ↓ | RMSE ↓ | R² ↑ | Result |
|---|---:|---:|---:|---:|---|
| LSTM Without TL | 28,769.91 | 1.392×10⁹ | 37,315.74 | 0.82081 | Baseline |
| **LSTM + Optimized TL** | **12,964.01** | **5.503×10⁸** | **23,457.42** | **0.92919** | **Positive Transfer*** |

*Internal audit note: the current Solar B2 comparison is a post-Test supplementary comparison; the observed four-metric pattern satisfies the project Positive Transfer rule, but it is not a new untouched confirmatory Test.

Development / preliminary runs do not need to appear in the main thesis result table or main presentation. They should remain archived in Local experiment records for traceability.

## G.3 Chapter 5 — Contribution positioning

Do not make the contribution claim from Solar alone.

```text
Solar:
Optimized TL > WOTL
        ↓
Aquaponics:
Evaluate same method principle
        ↓
xLSTM / xLSTMTime:
Evaluate architecture-level adaptation
        ↓
Multiple datasets show improvement
        ↓
Support a certain degree of cross-dataset generalizability
```

> 本研究之貢獻不僅在於單一資料集之參數調整，而在於將來源模型之可遷移時間序列表徵、受控凍結策略與後段 Partial Fine-tuning 結合為一套可跨資料情境驗證之遷移學習方法。若該策略於 Solar 與其他主要公開資料集均呈現較低預測誤差及較高 R²，則可支持所提出方法於本研究資料條件下具有一定之跨資料集泛化能力。

---

# Provenance

```text
Run ID:
20260822T120239Z_seed1234

Execution Git HEAD:
2148396d00a904f457ec5a64d6b80369303970ca

Source checkpoint:
SRC_lr1e-4
Epoch = 500
SHA-256 =
5cd3db4501d6c6720fbdd8ee3cce692b01cecb82619574f8c6944a8795182d91

Selected B2-P1 checkpoint:
Epoch = 500
SHA-256 =
a666b15cdbc3af8cf25d79f3af2080f4869ac5658a2d15d0370033da20fbabf1
```

Recommended repository location:

```text
reports/Solar Energy Result/Linear_Supplementary/Experiment_B2/
20260822T120239Z_seed1234/summary/
Solar_LSTM_Optimized_Transfer_Learning_Strategy_v1.md
```

---

# Phase B2-1D Completion Criteria

- [x] B2-P1 method core is explicitly defined.
- [x] Solar-specific parameters are separated from transferable principles.
- [x] Positive Transfer evaluation rule is fixed.
- [x] Cross-dataset validation rule is defined before the next dataset Test.
- [x] Thesis positioning is defined.
- [x] Provenance and checkpoint identity are recorded.
- [x] Copy this Markdown file into the repository path above.
- [x] Review `git status` before any optional commit.

**No additional Solar training is required for Phase B2-1D.**
