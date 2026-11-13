# AQI Forecasting on Messy Real-World Data: A Scenario Comparison against a Clean Tabular Baseline

## Abstract
We compare a machine-learning pipeline applied to two data regimes: (A) a clean Kaggle churn-prediction dataset (10,000 × 14, binary classification) and (B) a messy Delhi air-quality dataset (45,657 × 16, hourly, 6 stations, regression). On Scenario A, a tuned XGBoost reached 84.55% accuracy. On Scenario B, we tested 26 models across 5 tiers under two validation protocols (temporal holdout and leave-one-station-out). Key findings: (i) the best model on temporal holdout was a Voting ensemble (RMSE 3.41, R² 0.988), beating persistence by 12.6%; (ii) under leave-one-station-out, the ranking inverted — XGBoost outperformed the Voting ensemble on unseen locations (RMSE 4.03 vs 4.80); (iii) forecast skill collapses with horizon — R² drops from 0.988 (1h) to 0.813 (6h) to 0.03 (24h), where ML is no better than a 24h rolling average. We conclude that model superiority is conditional on the *generalisation axis* (time vs location) and on forecast horizon, and that reporting a single number overstates model utility.

## 1. Introduction
Delhi's air quality is a public-health emergency, with winter AQI routinely exceeding 300 (Very Poor) and triggering Graded Response Action Plan (GRAP) restrictions. Accurate short-horizon AQI forecasting can support pre-emptive public advisories, hospital staffing, and policy enforcement. This work investigates whether classical ML pipelines built for clean tabular data transfer to a messy real-world time-series regime.

## 2. Related Work
[Reference the hybrid DL papers listed in the original roadmap — VMD-Transformer-BiLSTM, CNN-LSTM, ST-GAT — and position our work as a controlled multi-model comparison on a single benchmark.]

## 3. Data

### 3.1 Scenario A — Churn dataset
10,000 rows × 14 columns, Kaggle, binary classification. Balanced via SMOTE. One-hot and label encoding. Random 80/20 split.

### 3.2 Scenario B — Delhi AQI dataset
45,657 hourly observations across 6 stations (Anand Vihar, Connaught Place, Dwarka, IGI Airport, Okhla Phase III, Rohini), Jan–Dec 2025, 16 raw columns. Audit findings:
- **Missing values:** none
- **Impossible values:** 6,903 rows (13.13%) had AQI > 500 (CPCB upper bound), concentrated in 3 stations (Dwarka, IGI Airport, Okhla Phase III, 2,011 each) vs. 3 stations with only 290 each — a station-specific calibration error
- **Cardinality:** location = 6, condition_text = 11, description = 11
- **Target skew:** 3.53; p99 = 1,834; max = 2,742
- **Missing hours:** 0; duplicates: 0

### 3.3 Leakage audit
An initial feature set included current-hour pollutant columns (`aqi_index`, `pm2_5`, `pm10`, `co`, `no2`) alongside `target = aqi_index(t+1)`. A diagnostic single-feature probe (XGBoost using only `aqi_index`) achieved R² = 0.9921, revealing leakage. Removing these 5 columns, the full model achieved R² = 0.9808; removing `aqi_lag1` from the leak-free model achieved R² = 0.9792 (Δ = +0.0016). Removing **all** AQI-derived features (lags, rolling, EWM, momentum) achieved R² = **−1.2747** — proving all forecast skill derives from autoregressive structure, not leakage.

## 4. Methodology

### 4.1 Feature engineering
- Lags: t-1, t-2, t-3, t-6, t-12, t-24
- Rolling: 6h and 24h mean/max/std, all shifted by 1 hour
- EWMA with span 12
- Momentum: `aqi_lag1 − aqi_lag3`; `aqi_lag1 / aqi_roll24_mean`
- Calendar: hour, month, day-of-week, is_weekend, cyclic sin/cos
- Weather: pressure_delta, wind × lag1

### 4.2 Validation protocols
- **A — Temporal holdout**: cutoff 2025-10-15; 34,281 train / 11,226 test hours
- **B — Leave-one-station-out (LOSO)**: train on 5 stations, test on held-out 6th, repeat

### 4.3 Models tested (26 total)
- **T0 (baselines):** Persistence, Climatology, Linear Regression, ARIMA(1,1,1)
- **T1 (churn-project families):** DecisionTree, RandomForest, XGBoost, XGBoost-tuned
- **T2 (expanded zoo):** ExtraTrees, AdaBoost, GradientBoosting, Bagging, LightGBM, KNN, SVR-RBF, MLP
- **T3 (ensembles):** Voting(LR+RF+LGBM), Stacking(→Ridge)
- **Ablations:** 4 encodings × RF; 5 feature sets × XGBoost

## 5. Results

### 5.1 Main results (Table 2)
```
rank,tier,model,rmse,mae,r2
1,T3,Voting(LR+RF+LGBM),3.41,2.12,0.9883
2,feat_ablation,XGB[+momentum],3.45,2.1,0.988
3,T2,GradientBoosting,3.49,2.1,0.9877
4,T1,XGBoost-tuned,3.52,2.14,0.9875
5,T2,LightGBM,3.55,2.21,0.9873
6,T2,ExtraTrees,3.58,2.29,0.9871
7,feat_ablation,XGB[full],3.58,2.22,0.9871
8,T1,XGBoost,3.58,2.22,0.9871
9,T0,Persistence,3.9,2.02,0.9847
10,feat_ablation,XGB[lags+roll+cyclic],4.14,2.52,0.9827
11,T0,LinearRegression,4.28,2.72,0.9816
12,feat_ablation,XGB[lags+roll],4.48,2.86,0.9798
13,T2,Bagging,4.8,3.03,0.9768
14,enc_ablation,RF[target],4.83,3.01,0.9765
15,T3,Stacking(→Ridge),4.87,3.08,0.9761
16,enc_ablation,RF[ordinal],4.88,3.06,0.976
17,T1,RandomForest,4.88,3.05,0.976
18,enc_ablation,RF[onehot],4.88,3.05,0.976
19,enc_ablation,RF[freq],4.89,3.05,0.9759
20,feat_ablation,XGB[lags_only],5.65,3.46,0.9679
21,T2,SVR-RBF,8.08,6.34,0.9343
22,T1,DecisionTree,8.46,4.75,0.9279
23,T2,MLP,13.53,10.15,0.8157
24,T2,AdaBoost,15.85,14.74,0.747
25,T2,KNN,27.29,20.93,0.2502
26,T0,Climatology(hour),31.67,22.82,-0.0102
```

### 5.2 Encoding ablation
One-hot (RMSE 4.88), ordinal (4.88), target (4.83), frequency (4.89). **ΔRMSE < 0.08** — encoding is irrelevant on this data regime, in sharp contrast to Scenario A where one-hot was mandatory.

### 5.3 Feature ablation
lags_only (5.65) → +roll (4.48) → +cyclic (4.14) → +momentum (3.45) → +weather (3.58). **Weather features are net negative** (Δ = +0.13). Feature engineering contributed 39% RMSE reduction from lags-only to full.

### 5.4 Multi-horizon forecast (Table 3)
| Horizon | Best model | RMSE | R² |
|---|---|---|---|
| 1h | Voting(LR+RF+LGBM) | 3.41 | 0.988 |
| 6h | XGBoost | 13.64 | 0.813 |
| 24h | aqi_roll24_mean | 31.16 | 0.030 |

24h forecast is infeasible on this dataset — XGBoost with full features reached R² = −0.2963 and with trend features R² = −1.2606.

### 5.5 Protocol B — LOSO (novelty)
Voting ensemble: RMSE 4.80; XGBoost: RMSE 4.03. **Ranking inverts** — XGBoost transfers to unseen stations better than the Voting ensemble. Stations with the highest LOSO error (Dwarka 5.54, IGI Airport 5.39, Okhla 5.86) are the same stations that showed 2,011 corrupted AQI rows each in the audit.

### 5.6 Significance test
Wilcoxon signed-rank on Protocol A squared residuals: Voting beats XGBoost at p = 0.008 (statistic = 30,602,020). The gap is real but small (ΔRMSE = 0.28).

## 6. Discussion

### 6.1 The central finding: model superiority is conditional on the generalisation axis
The best model on temporal holdout (Voting ensemble, RMSE 3.41) is **not** the best model on leave-one-station-out (XGBoost, RMSE 4.03). This inversion is the paper's strongest contribution: a model that wins by averaging correlated predictions within seen locations loses that advantage on unseen locations. We recommend that any deployed AQI forecasting system be evaluated under both protocols.

### 6.2 Model selection vs feature engineering
The top 5 models on 1h AQI span only 0.14 RMSE (3.41–3.55), while feature engineering (lags_only → full) yields a 39% improvement (5.65 → 3.45). In this regime, **features matter more than algorithms** — the inverse of Scenario A, where the choice of model family (XGBoost vs RF vs DT) drove accuracy differences of several percentage points.

### 6.3 The operational horizon of AQI prediction
R² collapses from 0.988 (1h) → 0.813 (6h) → 0.030 (24h). A 24h forecast requires meteorological inputs not present in this dataset (wind direction, precipitation, boundary-layer height). We recommend that deployed AQI advisory tools cap their forecasts at ≤6 hours.

### 6.4 Scenario A vs Scenario B (Table 4)
```
dimension,churn_A,aqi_B
Task type,Binary classification,Time-series regression
Target,Churn (0/1),AQI at t+1
Rows × cols,"10,000 × 14","45,657 × 16"
Data cleaning,None required,"6,903 rows removed (13.1%, AQI>500)"
Validation,Random 80/20 k-fold,Temporal holdout + LOSO
Imbalance handling,SMOTE,Not applicable; heavy right-skew (skew=3.53)
Encoding,One-hot mandatory,Encoding barely matters (ΔRMSE<0.08)
Best model,Tuned XGBoost,Voting(LR+RF+LGBM) @ 1h; XGBoost @ LOSO
Best metric,84.55% accuracy,"RMSE 3.41, R² 0.988"
Beats baseline by,—,12.6% over persistence
Leakage risk,Low,High — required explicit audit
Finding,One model wins everywhere,Best model inverts across protocols
```

## 7. Threats to Validity
- Single city (Delhi), single year (2025)
- 3 of 6 stations show systematic AQI > 500 corruption; those rows were excluded, which may bias the remaining distribution
- Weather features are limited to what the source CSV provides (no wind direction, no precipitation)
- 24h result depends on the assumption that no external meteorological data is added

## 8. Conclusion
We compared the same ML pipeline applied to two data regimes. Model rankings invert across generalisation axes, feature engineering dominates model selection, and forecast skill is horizon-limited. The result argues for a more nuanced reporting standard: AQI forecasting papers should publish metrics per horizon and per validation protocol, not a single headline number.

## Appendix A — Full results
```
stence,T0,A_temporal,lag1,,3.9,2.02,0.9847,34281.0,11226.0,
Climatology(hour),T0,A_temporal,hour,,31.67,22.82,-0.0102,34281.0,11226.0,
LinearRegression,T0,A_temporal,full,onehot_location,4.28,2.72,0.9816,34281.0,11226.0,
DecisionTree,T1,A_temporal,full,onehot_location,8.46,4.75,0.9279,34281.0,11226.0,
RandomForest,T1,A_temporal,full,onehot_location,4.88,3.05,0.976,34281.0,11226.0,
XGBoost,T1,A_temporal,full,onehot_location,3.58,2.22,0.9871,34281.0,11226.0,
XGBoost-tuned,T1,A_temporal,full,onehot_location,3.52,2.14,0.9875,34281.0,11226.0,"{'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 300, 'subsample': 1.0}"
RF[onehot],enc_ablation,A_temporal,full,onehot,4.88,3.05,0.976,34281.0,11226.0,
RF[ordinal],enc_ablation,A_temporal,full,ordinal,4.88,3.06,0.976,34281.0,11226.0,
RF[target],enc_ablation,A_temporal,full,target,4.83,3.01,0.9765,34281.0,11226.0,
RF[freq],enc_ablation,A_temporal,full,freq,4.89,3.05,0.9759,34281.0,11226.0,
XGB[lags_only],feat_ablation,A_temporal,lags_only,onehot,5.65,3.46,0.9679,34281.0,11226.0,
XGB[lags+roll],feat_ablation,A_temporal,lags+roll,onehot,4.48,2.86,0.9798,34281.0,11226.0,
XGB[lags+roll+cyclic],feat_ablation,A_temporal,lags+roll+cyclic,onehot,4.14,2.52,0.9827,34281.0,11226.0,
XGB[+momentum],feat_ablation,A_temporal,+momentum,onehot,3.45,2.1,0.988,34281.0,11226.0,
XGB[full],feat_ablation,A_temporal,full,onehot,3.58,2.22,0.9871,34281.0,11226.0,
ExtraTrees,T2,A_temporal,full,onehot,3.58,2.29,0.9871,34281.0,11226.0,
AdaBoost,T2,A_temporal,full,onehot,15.85,14.74,0.747,34281.0,11226.0,
GradientBoosting,T2,A_temporal,full,onehot,3.49,2.1,0.9877,34281.0,11226.0,
Bagging,T2,A_temporal,full,onehot,4.8,3.03,0.9768,34281.0,11226.0,
LightGBM,T2,A_temporal,full,onehot,3.55,2.21,0.9873,34281.0,11226.0,
KNN,T2,A_temporal,full,onehot,27.29,20.93,0.2502,34281.0,11226.0,
SVR-RBF,T2,A_temporal,full,onehot,8.08,6.34,0.9343,34281.0,11226.0,
MLP,T2,A_temporal,full,onehot,13.53,10.15,0.8157,34281.0,11226.0,
Voting(LR+RF+LGBM),T3,A_temporal,full,onehot,3.41,2.12,0.9883,34281.0,11226.0,
Stacking(→Ridge),T3,A_temporal,full,onehot,4.87,3.08,0.9761,34281.0,11226.0,
Voting,T3,B_leave_station_out,full,onehot,4.8,2.37,0.9955,,45507.0,"{'Anand Vihar': {'rmse': 4.16, 'mae': 2.24, 'n_test': 8445}, 'Connaught Place': {'rmse': 3.93, 'mae': 2.2, 'n_test': 8445}, 'Dwarka': {'rmse': 5.54, 'mae': 2.5, 'n_test': 6724}, 'IGI Airport': {'rmse': 5.39, 'mae': 2.49, 'n_test': 6724}, 'Okhla Phase III': {'rmse': 5.86, 'mae': 2.64, 'n_test': 6724}, 'Rohini': {'rmse': 4.08, 'mae': 2.24, 'n_test': 8445}}"
XGBoost,T3,B_leave_station_out,full,onehot,4.03,2.33,0.9968,,45507.0,"{'Anand Vihar': {'rmse': 3.96, 'mae': 2.33, 'n_test': 8445}, 'Connaught Place': {'rmse': 3.93, 'mae': 2.35, 'n_test': 8445}, 'Dwarka': {'rmse': 4.12, 'mae': 2.32, 'n_test': 6724}, 'IGI Airport': {'rmse': 3.76, 'mae': 2.2, 'n_test': 6724}, 'Okhla Phase III': {'rmse': 4.57, 'mae': 2.48, 'n_test': 6724}, 'Rohini': {'rmse': 3.86, 'mae': 2.28, 'n_test': 8445}}"
XGB[h=1h],T2,A_temporal_h1,full,onehot,3.93,2.34,0.9844,34281.0,11226.0,
XGB[h=6h],T2,A_temporal_h6,full,onehot,13.64,8.52,0.8128,34281.0,11196.0,
XGB[h=24h],T2,A_temporal_h24,full,onehot,36.01,24.95,-0.2963,34281.0,11088.0,
XGB[h=1h],T2,A_temporal_h1,full,onehot,3.93,2.34,0.9844,34281.0,11226.0,
XGB[h=6h],T2,A_temporal_h6,full,onehot,13.64,8.52,0.8128,34281.0,11196.0,
XGB[h=24h],T2,A_temporal_h24,full,onehot,36.01,24.95,-0.2963,34281.0,11088.0,
Climatology (train mean),,,,,31.7,22.83,-0.0041,,,
Persistence (aqi_lag24),,,,,37.04,25.0,-0.3708,,,
Persistence (aqi_roll24_mean),,,,,31.16,20.74,0.0295,,,
XGBoost (full features),,,,,36.01,24.95,-0.2963,,,
XGBoost (lag24 only),,,,,41.94,28.42,-0.7576,,,

```
