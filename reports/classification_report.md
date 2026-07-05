# Kalshi Weather Market: Classification Report (Log Loss)

_Generated: 2026-07-05 22:23 UTC_

**Train**: 2025-06-01 – 2025-08-31 (279,538 bars)  **Test**: 2025-09-01 – 2025-09-30 (134,786 bars)

## Methodology

- **All bars evaluated** — flat-move bars (~35%) score as negative class; matches live trading where future outcome is unknown
- **Primary metric: log loss** — measures calibration of predicted probabilities. Lower is better; 0 = perfect. Naive baseline = always predict the class prior.
- **Label (up_vs_rest)**: 1 if `target_move_30 > 0`, else 0 (~32% positive)
- **Label (down_vs_rest)**: 1 if `target_move_30 < 0`, else 0 (~33% positive)
- **LightGBM**: `objective='binary'` (binary cross-entropy), `is_unbalance=True`
- **Logistic Regression**: log loss is the default objective
- **Train**: 2025-06-01 – 2025-08-31 | **Test**: 2025-09-01 – 2025-09-30

## Results — Predict UP (move > 0) vs Flat-or-Down

Positive rate: 32.7%  |  Majority-class accuracy baseline: 67.3%  |  Naive log loss (predict prior): **0.6324**

| Model | Log Loss | vs Naive | AUC | Accuracy | vs Maj | Precision | Recall |
|---|---|---|---|---|---|---|---|
| baseline_lr | **0.6166** | +2.5% | 0.6290 | 67.4% | +0.1pp | 51.3% | 7.8% |
| model1_lr | **0.6103** | +3.5% | 0.6443 | 67.5% | +0.2pp | 51.6% | 10.5% |
| model2_lr | **0.6102** | +3.5% | 0.6429 | 67.5% | +0.2pp | 51.7% | 10.5% |
| model3_lr | **0.6070** | +4.0% | 0.6454 | 67.6% | +0.4pp | 52.2% | 14.2% |
| model4_lr | **0.6067** | +4.1% | 0.6459 | 67.7% | +0.4pp | 52.3% | 14.5% |
| model5_lr | **0.6065** | +4.1% | 0.6450 | 67.9% | +0.7pp | 53.8% | 15.0% |
| model6_gbm | **0.6198** | +2.0% | 0.6877 | 63.1% | -4.2pp | 45.3% | 62.1% |

## Results — Predict DOWN (move < 0) vs Flat-or-Up

Positive rate: 34.1%  |  Majority-class accuracy baseline: 65.9%  |  Naive log loss (predict prior): **0.6414**

| Model | Log Loss | vs Naive | AUC | Accuracy | vs Maj | Precision | Recall |
|---|---|---|---|---|---|---|---|
| model6_gbm_down | **0.5877** | +8.4% | 0.7166 | 65.3% | -0.7pp | 49.2% | 64.1% |

## Feature Importance: Logistic Regression (model5_lr — all features, up_vs_rest)

Coefficients normalized to % of total absolute weight after StandardScaler.

| Rank | Feature | Importance % | Coef | Category |
|---|---|---|---|---|
| 1 | `ask_lag1` | 10.30% | +0.4830 | momentum |
| 2 | `frac_day_elapsed` | 9.93% | -0.4654 | resolution |
| 3 | `bar_hour` | 6.46% | +0.3029 | time |
| 4 | `ask_lag15` | 6.15% | +0.2882 | momentum |
| 5 | `yes_ask_close` | 6.11% | -0.2863 | anchor |
| 6 | `mid_close` | 6.08% | -0.2851 | anchor |
| 7 | `vol_zscore` | 5.64% | -0.2645 | burst |
| 8 | `tod_sin` | 5.19% | -0.2431 | time |
| 9 | `vol_ratio` | 4.02% | +0.1886 | burst |
| 10 | `ask_lag5` | 3.83% | +0.1795 | momentum |
| 11 | `tod_cos` | 3.79% | +0.1777 | time |
| 12 | `price_polarization` | 3.77% | +0.1768 | resolution |
| 13 | `spread_chg_5` | 2.51% | -0.1177 | spread |
| 14 | `spread` | 2.38% | -0.1117 | spread |
| 15 | `mid_vol_15` | 1.99% | +0.0932 | volatility |
| 16 | `vol_burst_flag` | 1.44% | +0.0676 | burst |
| 17 | `mins_to_peak` | 1.40% | -0.0656 | resolution |
| 18 | `spread_lag1` | 1.29% | +0.0603 | spread |
| 19 | `bars_since_metar` | 1.21% | -0.0568 | metar |
| 20 | `ask_intrabar_range` | 1.18% | +0.0553 | volatility |

### Category Summary (Logistic Regression)

| Category | Total % |
|---|---|
| momentum | 22.3% |
| time | 16.5% |
| resolution | 15.7% |
| anchor | 13.2% |
| burst | 11.1% |
| spread | 7.2% |
| cross | 4.9% |
| volatility | 3.9% |
| flow | 3.0% |
| metar | 2.3% |

## Feature Importance: LightGBM (model6_gbm — up_vs_rest)

Gain importance (total log-loss reduction from splits on this feature).

| Rank | Feature | Gain | Share | Category | Bar |
|---|---|---|---|---|---|
| 1 | `tod_sin` | 1,975 | 6.4% | time | `█████████████████████████` |
| 2 | `tod_cos` | 1,888 | 6.1% | time | `████████████████████████░` |
| 3 | `mid_ret_30` | 1,746 | 5.6% | momentum | `██████████████████████░░░` |
| 4 | `price_previous` | 1,670 | 5.4% | anchor | `█████████████████████░░░░` |
| 5 | `oi_change_15` | 1,421 | 4.6% | flow | `██████████████████░░░░░░░` |
| 6 | `vol_sum_15` | 1,420 | 4.6% | flow | `██████████████████░░░░░░░` |
| 7 | `mid_vol_15` | 1,331 | 4.3% | volatility | `█████████████████░░░░░░░░` |
| 8 | `rel_mid` | 1,224 | 3.9% | cross | `███████████████░░░░░░░░░░` |
| 9 | `mid_close` | 1,093 | 3.5% | anchor | `██████████████░░░░░░░░░░░` |
| 10 | `ask_lag15` | 1,016 | 3.3% | momentum | `█████████████░░░░░░░░░░░░` |
| 11 | `mins_to_peak` | 952 | 3.1% | resolution | `████████████░░░░░░░░░░░░░` |
| 12 | `yes_ask_close` | 939 | 3.0% | anchor | `████████████░░░░░░░░░░░░░` |
| 13 | `relative_spread` | 910 | 2.9% | cross | `████████████░░░░░░░░░░░░░` |
| 14 | `mid_ret_15` | 899 | 2.9% | momentum | `███████████░░░░░░░░░░░░░░` |
| 15 | `spread` | 863 | 2.8% | spread | `███████████░░░░░░░░░░░░░░` |
| 16 | `avg_spread_all` | 850 | 2.7% | cross | `███████████░░░░░░░░░░░░░░` |
| 17 | `sum_mid_all` | 818 | 2.6% | cross | `██████████░░░░░░░░░░░░░░░` |
| 18 | `spread_dispersion` | 778 | 2.5% | cross | `██████████░░░░░░░░░░░░░░░` |
| 19 | `prob_sum_deviation` | 723 | 2.3% | cross | `█████████░░░░░░░░░░░░░░░░` |
| 20 | `frac_day_elapsed` | 645 | 2.1% | resolution | `████████░░░░░░░░░░░░░░░░░` |

### Category Summary (GBM)

| Category | Total Gain % |
|---|---|
| cross | 18.5% |
| momentum | 17.2% |
| time | 14.9% |
| anchor | 11.9% |
| flow | 11.6% |
| resolution | 7.8% |
| spread | 7.0% |
| volatility | 6.6% |
| metar | 2.4% |
| burst | 2.1% |

## Per-Ticker Log Loss (GBM up_vs_rest, Test Set)

Sorted by log loss ascending (best first). High flat% tickers trivially predict the majority class.

| Ticker | Log Loss | Accuracy | N Rows | % Flat |
|---|---|---|---|---|
| `KXHIGHLAX-25SEP13-T71` | 0.1420 | 98.8% | 255 | 96.9% |
| `KXHIGHLAX-25SEP09-T71` | 0.1573 | 98.4% | 253 | 93.7% |
| `KXHIGHLAX-25SEP22-T74` | 0.2148 | 95.2% | 188 | 89.4% |
| `KXHIGHLAX-25SEP10-B70.5` | 0.2214 | 92.6% | 660 | 83.9% |
| `KXHIGHLAX-25SEP15-T74` | 0.2334 | 97.7% | 397 | 89.7% |
| `KXHIGHLAX-25SEP19-B71.5` | 0.2390 | 97.1% | 173 | 87.3% |
| `KXHIGHLAX-25SEP10-T70` | 0.2497 | 90.0% | 673 | 78.6% |
| `KXHIGHLAX-25SEP06-B72.5` | 0.2509 | 95.4% | 500 | 81.8% |
| `KXHIGHLAX-25SEP05-T73` | 0.2636 | 96.1% | 769 | 87.5% |
| `KXHIGHLAX-25SEP20-T72` | 0.3083 | 90.6% | 160 | 75.0% |
| `KXHIGHLAX-25SEP08-B83.5` | 0.3172 | 90.4% | 903 | 75.6% |
| `KXHIGHLAX-25SEP14-T72` | 0.3301 | 92.8% | 111 | 71.2% |
| `KXHIGHLAX-25SEP05-T80` | 0.3318 | 90.8% | 633 | 78.5% |
| `KXHIGHLAX-25SEP27-B69.5` | 0.3351 | 89.0% | 608 | 76.2% |
| `KXHIGHLAX-25SEP14-B72.5` | 0.3384 | 89.3% | 410 | 75.9% |
| `KXHIGHLAX-25SEP04-B80.5` | 0.3492 | 82.8% | 854 | 59.8% |
| `KXHIGHLAX-25SEP21-T73` | 0.3663 | 88.2% | 508 | 73.6% |
| `KXHIGHLAX-25SEP08-T84` | 0.3809 | 83.8% | 779 | 69.8% |
| `KXHIGHLAX-25SEP09-B71.5` | 0.4134 | 85.2% | 453 | 64.7% |
| `KXHIGHLAX-25SEP04-B82.5` | 0.4311 | 88.1% | 495 | 65.7% |
| `KXHIGHLAX-25SEP24-B72.5` | 0.4328 | 84.6% | 428 | 61.4% |
| `KXHIGHLAX-25SEP17-B84.5` | 0.4371 | 84.5% | 588 | 63.3% |
| `KXHIGHLAX-25SEP19-B73.5` | 0.4435 | 78.2% | 349 | 49.6% |
| `KXHIGHLAX-25SEP20-B72.5` | 0.4445 | 87.7% | 219 | 70.8% |
| `KXHIGHLAX-25SEP28-B70.5` | 0.4537 | 79.1% | 555 | 61.3% |
| `KXHIGHLAX-25SEP25-T70` | 0.4550 | 87.1% | 526 | 70.0% |
| `KXHIGHLAX-25SEP19-B75.5` | 0.4672 | 72.8% | 1,007 | 46.7% |
| `KXHIGHLAX-25SEP05-B73.5` | 0.4720 | 70.7% | 1,197 | 54.8% |
| `KXHIGHLAX-25SEP16-T77` | 0.4725 | 72.6% | 430 | 65.8% |
| `KXHIGHLAX-25SEP04-T76` | 0.4931 | 71.6% | 662 | 45.2% |

## Key Findings

1. **Naive log loss baseline**: 0.6324 (always predict 32.7% probability of up-move)
2. **Best LR model**: `model5_lr` — log loss 0.6065 (+4.1% vs naive), AUC 0.6450
3. **GBM up_vs_rest**: log loss 0.6198 (+2.0% vs naive), AUC 0.6877, accuracy 63.1% (-4.2pp vs majority)
4. **GBM down_vs_rest**: log loss 0.5877 (+8.4% vs naive), AUC 0.7166
5. **Top GBM features**: `tod_sin`, `tod_cos`, `mid_ret_30`
6. **Dominant feature categories** (GBM gain): cross, momentum, time — combined 50.6% of total gain

## Limitations

- Flat bars (~35%) are structural noise: any directional prediction is wrong for them, setting a floor on log loss that no binary model can escape
- `is_unbalance=True` in LightGBM reweights classes but does not change the label space; precision/recall tradeoffs still apply
- Predicted probabilities near 0.5 carry little trading value; consider filtering to high-confidence predictions (e.g. prob > 0.60) for live use
- No transaction cost model — bid/ask spread (~1¢–2¢) must be overcome
