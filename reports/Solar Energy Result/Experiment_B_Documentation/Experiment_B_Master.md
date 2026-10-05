Experiment B — Plant2 → Plant1｜Master Closure Record

Project: LSTM研究與實驗－SolarEnergy 太陽能發電預測（智慧能源）  
Document role: Canonical experiment closure record  
Status: SEALED / CLOSED  
Model-development policy: NO MORE TUNING; NO MORE TEST ACCESS FOR MODEL DEVELOPMENT

1\. Executive Summary

Experiment B evaluates LSTM Transfer Learning from Plant2 (Source) to Plant1 (Target) for DC\_POWER forecasting. The corrected pipeline uses chronological splitting, Training-only feature/target scaler fitting, split-isolated sequence construction, window=5, horizon=1, 15-minute sampling, and original-scale evaluation after target inverse\_transform.

The formal Linear protocol (solar-linear-v1.0) preserved an untouched Plant1 Target Test until authorization. Its formal WOTL baseline became the fixed baseline later reused for the supplementary comparison. Formal Experiment B itself is preserved as a sealed historical result and is not overwritten by later tuning.

After the formal Test had been revealed, Experiment B2 was explicitly opened as Post-Test Supplementary / Exploratory Tuning. Validation-only comparison selected B2-P1: Partial Fine-tuning of the final LSTM block plus a target-specific Linear output head, with earlier transferred layers and BatchNormalization frozen. B2-P1 produced lower MAE, MSE, RMSE and higher R² than the fixed WOTL baseline on the same 2,607 aligned Plant1 Test sequences. Numerically this satisfies the project Positive Transfer rule, but because the Test had already been revealed, the B2 result is classified as Supplementary Positive Transfer / method-development evidence, not a new untouched confirmatory Final Test.

2\. Experiment Identity

Direction: Plant2 → Plant1  
Source: Plant2 source\_profile  
Target: Plant1 target\_profile  
Target variable: DC\_POWER  
Features: TIME\_SIN, TIME\_COS, IRRADIATION, AMBIENT\_TEMPERATURE, MODULE\_TEMPERATURE  
Frequency: 15 minutes  
Window: 5  
Horizon: 1  
Input duration: past 75 minutes  
Forecast point: next 15-minute point  
Output activation: Linear  
Loss: MSE  
Optimizer: Adam  
Seed: 1234  
Shuffle: False

3\. Corrected Data Contract

Source profile rows: Training 2,088 / Validation 523 / Test 653  
Source sequences: Training 2,083 / Validation 518 / Test 648  
Target profile rows: Training 521 / Validation 131 / Test 2,612  
Target sequences: Training 516 / Validation 126 / Test 2,607

Data rules:  
\- chronological split  
\- no split crossing  
\- feature scaler and target scaler are separate  
\- scalers fit Training only  
\- Validation/Test transform only  
\- Source Test remains independent  
\- official metrics are computed after Target target\_scaler inverse\_transform  
\- raw / unclipped prediction policy

4\. Formal Linear Protocol

Protocol version: solar-linear-v1.0  
Formal Experiment B namespace: reports/Solar Energy Result/Linear\_Formal/Experiment\_B/\<run\_id\>/  
Formal B run used for fixed WOTL baseline: 20260821T041718Z\_seed1234

Protocol governance included config lock, Validation-only candidate selection, checkpoint SHA verification, selection lock, explicit Final Test authorization, and post-test tuning disabled for the sealed formal run.

5\. Optimized Supplementary Strategy — B2-P1

Run ID: 20260822T120239Z\_seed1234  
Execution Git HEAD: 2148396d00a904f457ec5a64d6b80369303970ca  
Selected candidate: B2-P1  
Strategy: Last LSTM \+ Target Output Fine-tuning  
Terminology: Partial Fine-tuning / Partial FT  
Transferred layers: 1,2,3,4,5  
Trainable layers: 4,6  
Frozen layers: 1,2,3,5  
BatchNormalization: Frozen  
Output activation: Linear  
Learning rate: 1e-5  
Batch size: 128  
Maximum epochs: 500  
Best epoch: 500  
Best Validation loss: 0.096119724214077  
Total parameters: 46,681  
Trainable parameters: 29,101  
Trainable ratio: 62.340138%

Source checkpoint SHA-256:  
5cd3db4501d6c6720fbdd8ee3cce692b01cecb82619574f8c6944a8795182d91

Selected B2-P1 checkpoint SHA-256:  
a666b15cdbc3af8cf25d79f3af2080f4869ac5658a2d15d0370033da20fbabf1

6\. Validation Evidence

B2-P1 Validation, original scale, n=126:  
MAE  \= 19,442.8612  
MSE  \= 810,674,279.2778  
RMSE \= 28,472.3424  
R²   \= 0.920729

B2-P1 was selected over B2-P2 and B2-P3 using Validation evidence before the supplementary Test comparison.

7\. Test Original-Scale Comparison

Fixed WOTL baseline, n=2,607:  
MAE  \= 28,769.9078  
MSE  \= 1,392,464,754.3387  
RMSE \= 37,315.7441  
R²   \= 0.820814

Optimized TL B2-P1, n=2,607:  
MAE  \= 12,964.0142  
MSE  \= 550,250,743.1286  
RMSE \= 23,457.4241  
R²   \= 0.929192

Relative change vs WOTL:  
MAE  \-54.939%  
MSE  \-60.484%  
RMSE \-37.138%  
R²   \+0.108378 absolute

Observed numerical classification: Supplementary Positive Transfer.

8\. Alignment and Recalculation Audit

Target Test rows: 2,607 effective sequences  
Timestamp alignment: PASS  
WOTL/B2-P1 y\_true identity: PASS  
Residual definition: actual \- predicted  
Residual verification: PASS  
Metrics recalculation against official JSON: PASS

The thesis-figure post-processing stage performed no training, no model loading, no model.predict, no checkpoint reselection, no clipping, and no formal artifact overwrite.

9\. Physical-Plausibility Limitation

Raw/unclipped negative prediction count:  
WOTL: 21  
B2-P1: 610

B2-P1 therefore improves the four primary error/explanatory metrics while exposing a physical-plausibility limitation of unconstrained Linear output. The official metrics must remain raw/unclipped; retrospective clipping must not be used to rewrite this experiment.

10\. Test Reuse Disclosure

Experiment B2 is Post-Test Supplementary / Exploratory Tuning.  
Plant1 Test had already been revealed by Formal Experiment B before B2 model development.  
Therefore:  
\- B2 does not replace Formal Experiment B.  
\- B2 is not a new untouched confirmatory Final Test.  
\- B2 is method-development / optimized-strategy evidence.  
\- Solar B2 alone cannot establish universal generalization.

11\. Thesis Positioning

Chapter 4 may present WOTL vs Optimized TL as the observed Plant2→Plant1 result under the current split/settings, with explicit methodological provenance retained in project records. Thesis-safe language: “於本次資料切分與實驗設定下，Optimized Transfer Learning 呈現較低之 MAE、MSE 與 RMSE，且 R² 較高，呈現正向遷移結果。”

Single-run language restrictions: do not claim stable improvement, statistical significance, universal effectiveness, or proof of global optimality.

12\. Evidence Index

Formal evidence root:  
reports/Solar Energy Result/Linear\_Formal/Experiment\_B/20260821T041718Z\_seed1234/

Supplementary B2 root:  
reports/Solar Energy Result/Linear\_Supplementary/Experiment\_B2/20260822T120239Z\_seed1234/

Key B2 artifacts:  
\- run\_manifest.json  
\- test\_reuse\_disclosure.json  
\- summary/comparison.json  
\- summary/results\_summary.json  
\- summary/validation\_registry.json  
\- candidates/B2-P1/training\_history.csv  
\- candidates/B2-P1/predictions.csv  
\- candidates/B2-P1/test\_metrics\_original\_scale.json  
\- summary/Solar\_LSTM\_Optimized\_Transfer\_Learning\_Strategy\_v1.md  
\- summary/Solar\_LSTM\_Optimized\_Transfer\_Learning\_Result\_Card\_v1.md

Thesis figure root:  
reports/Solar Energy Result/Thesis\_Figures/Experiment\_B2\_Plant2\_to\_Plant1/

Verification artifacts:  
\- metrics\_recheck.json  
\- figure\_manifest.json  
\- 10 raw diagnostic figures

13\. Closure Decision

Experiment B model development status: CLOSED  
Formal Experiment B: PRESERVED / SEALED  
B2 optimized strategy: CONSOLIDATED  
Further Solar-B tuning: PROHIBITED  
Further Plant1 Test use for model-development decisions: PROHIBITED  
Read-only audit / thesis writing / provenance verification: ALLOWED

FINAL STATUS: SEALED / CLOSED  
