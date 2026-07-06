# Kalshi Weather Market: 3-Class Classification (UP / NEUTRAL / DOWN)

_Generated: 2026-07-06 01:30 UTC_

**Train**: 2025-06-01 – 2025-08-31 (279,538 bars)  **Test**: 2025-09-01 – 2025-09-30 (134,786 bars)

## Methodology

- **Classes**: DOWN (`move < -0.05`, 9.0%), NEUTRAL (`|move| ≤ 0.05`, 81.8%), UP (`move > +0.05`, 9.1%)
- **All bars included** — model predicts on every bar as in live trading
- **Loss**: negative log loss (categorical cross-entropy), lower = better
- **Naive baseline**: predict class priors on every bar → log loss = 0.5999
- **LightGBM**: `objective='multiclass'`, `num_class=3`, `class_weight='balanced'`
- **Logistic Regression**: `multi_class='multinomial'`, `solver='lbfgs'`

## Model Comparison

Naive log loss: **0.5999**  |  Majority-class accuracy baseline: **81.8%** (always predict NEUTRAL)

| Model | Log Loss | vs Naive | Accuracy | vs Majority | Prec↑up | Rec↑up | Prec↓down | Rec↓down |
|---|---|---|---|---|---|---|---|---|
| baseline_lr | **0.5787** | +3.5% | 81.8% | +0.0pp | 0.0% | 0.0% | 0.0% | 0.0% |
| model1_lr | **0.5260** | +12.3% | 81.9% | +0.1pp | 37.1% | 2.4% | 41.1% | 7.2% |
| model2_lr | **0.5213** | +13.1% | 82.0% | +0.2pp | 38.1% | 2.3% | 43.9% | 7.6% |
| model3_lr | **0.5184** | +13.6% | 82.0% | +0.2pp | 40.0% | 2.6% | 45.6% | 7.2% |
| model4_lr | **0.5179** | +13.7% | 82.0% | +0.2pp | 39.0% | 2.6% | 45.3% | 7.3% |
| model5_lr | **0.5086** | +15.2% | 82.2% | +0.4pp | 40.7% | 3.9% | 44.6% | 9.0% |
| model6_gbm | **0.6348** | -5.8% | 72.4% | -9.4pp | 25.0% | 45.1% | 30.2% | 49.0% |

### Confusion Matrix — model5_lr

Rows = actual, columns = predicted (DOWN / NEUTRAL / UP)

| Actual \ Predicted | DOWN | NEUTRAL | UP |
|---|---|---|---|
| **DOWN** | 1,088 (9%) | 10,692 (88%) | 376 (3%) |
| **NEUTRAL** | 732 (1%) | 109,233 (99%) | 333 (0%) |
| **UP** | 622 (5%) | 11,223 (91%) | 487 (4%) |

## Feature Importance: Logistic Regression (model5_lr — up_vs_rest)

Mean absolute coefficient across DOWN / NEUTRAL / UP planes, normalised. Also shows per-class coefficient direction.

| Rank | Feature | Importance % | Coef↓down | Coef~neutral | Coef↑up | Category |
|---|---|---|---|---|---|---|
| 1 | `price_polarization` | 8.41% | +0.302 | -0.440 | +0.138 | resolution |
| 2 | `ask_lag1` | 6.78% | -0.355 | +0.014 | +0.340 | momentum |
| 3 | `bar_hour` | 6.68% | -0.128 | -0.222 | +0.350 | time |
| 4 | `relative_spread` | 5.34% | +0.279 | -0.162 | -0.118 | cross |
| 5 | `ask_lag15` | 5.29% | +0.045 | -0.277 | +0.232 | momentum |
| 6 | `mins_to_peak` | 4.95% | -0.182 | -0.077 | +0.259 | resolution |
| 7 | `mid_close` | 4.87% | +0.125 | +0.130 | -0.255 | anchor |
| 8 | `yes_ask_close` | 4.79% | +0.137 | +0.114 | -0.250 | anchor |
| 9 | `vol_zscore` | 3.74% | -0.056 | +0.195 | -0.139 | burst |
| 10 | `tod_sin` | 3.69% | +0.063 | +0.130 | -0.193 | time |
| 11 | `ask_lag5` | 3.64% | -0.190 | +0.042 | +0.148 | momentum |
| 12 | `avg_spread_all` | 3.28% | +0.171 | -0.107 | -0.064 | cross |
| 13 | `rel_mid` | 3.00% | +0.157 | -0.006 | -0.151 | cross |
| 14 | `mid_vol_15` | 2.73% | +0.045 | -0.143 | +0.098 | volatility |
| 15 | `spread` | 2.68% | +0.140 | -0.087 | -0.053 | spread |
| 16 | `price_previous` | 2.66% | -0.139 | +0.120 | +0.019 | anchor |
| 17 | `vol_sum_15` | 2.55% | +0.062 | +0.071 | -0.133 | flow |
| 18 | `vol_ratio` | 2.39% | +0.041 | -0.125 | +0.085 | burst |
| 19 | `spread_chg_5` | 2.34% | -0.062 | +0.123 | -0.060 | spread |
| 20 | `frac_day_elapsed` | 2.32% | +0.063 | -0.121 | +0.058 | resolution |

### Category Summary (Logistic Regression)

| Category | Total % |
|---|---|
| momentum | 17.2% |
| cross | 17.1% |
| resolution | 16.1% |
| anchor | 12.3% |
| time | 11.6% |
| spread | 8.1% |
| burst | 7.1% |
| flow | 4.8% |
| volatility | 4.5% |
| metar | 1.2% |

## Feature Importance: LightGBM (model6_gbm)

Gain importance (total log-loss reduction from splits).

| Rank | Feature | Gain | Share | Category | Bar |
|---|---|---|---|---|---|
| 1 | `tod_cos` | 5,303 | 5.7% | time | `█████████████████████████` |
| 2 | `tod_sin` | 5,163 | 5.6% | time | `████████████████████████░` |
| 3 | `vol_sum_15` | 4,713 | 5.1% | flow | `██████████████████████░░░` |
| 4 | `mid_ret_30` | 4,560 | 4.9% | momentum | `█████████████████████░░░░` |
| 5 | `price_previous` | 4,335 | 4.7% | anchor | `████████████████████░░░░░` |
| 6 | `oi_change_15` | 4,305 | 4.6% | flow | `████████████████████░░░░░` |
| 7 | `mid_vol_15` | 4,129 | 4.4% | volatility | `███████████████████░░░░░░` |
| 8 | `ask_lag15` | 3,302 | 3.6% | momentum | `████████████████░░░░░░░░░` |
| 9 | `mid_close` | 3,288 | 3.5% | anchor | `████████████████░░░░░░░░░` |
| 10 | `rel_mid` | 3,171 | 3.4% | cross | `███████████████░░░░░░░░░░` |
| 11 | `mins_to_peak` | 2,782 | 3.0% | resolution | `█████████████░░░░░░░░░░░░` |
| 12 | `mid_ret_15` | 2,667 | 2.9% | momentum | `█████████████░░░░░░░░░░░░` |
| 13 | `yes_ask_close` | 2,644 | 2.8% | anchor | `████████████░░░░░░░░░░░░░` |
| 14 | `spread_dispersion` | 2,613 | 2.8% | cross | `████████████░░░░░░░░░░░░░` |
| 15 | `avg_spread_all` | 2,420 | 2.6% | cross | `███████████░░░░░░░░░░░░░░` |
| 16 | `spread` | 2,295 | 2.5% | spread | `███████████░░░░░░░░░░░░░░` |
| 17 | `relative_spread` | 2,231 | 2.4% | cross | `███████████░░░░░░░░░░░░░░` |
| 18 | `minute_of_hour` | 2,086 | 2.2% | time | `██████████░░░░░░░░░░░░░░░` |
| 19 | `vol_zscore` | 2,076 | 2.2% | burst | `██████████░░░░░░░░░░░░░░░` |
| 20 | `sum_mid_all` | 2,066 | 2.2% | cross | `██████████░░░░░░░░░░░░░░░` |

### Category Summary (GBM)

| Category | Total Gain % |
|---|---|
| momentum | 17.3% |
| cross | 17.0% |
| time | 14.1% |
| flow | 13.1% |
| anchor | 11.0% |
| resolution | 8.4% |
| spread | 7.1% |
| volatility | 6.6% |
| metar | 2.8% |
| burst | 2.6% |

## Per-Ticker Log Loss (GBM, Test Set)

Best 30 tickers by log loss. High-flat tickers score trivially well.

| Ticker | Log Loss | Accuracy | N Rows | % Flat |
|---|---|---|---|---|
| `KXHIGHLAX-25SEP10-B70.5` | 0.0414 | 99.4% | 660 | 83.9% |
| `KXHIGHLAX-25SEP25-T70` | 0.0800 | 98.3% | 526 | 70.0% |
| `KXHIGHLAX-25SEP14-B72.5` | 0.1143 | 96.3% | 410 | 75.9% |
| `KXHIGHLAX-25SEP19-B73.5` | 0.1703 | 95.4% | 349 | 49.6% |
| `KXHIGHLAX-25SEP22-B80.5` | 0.1768 | 97.8% | 670 | 31.6% |
| `KXHIGHLAX-25SEP04-B80.5` | 0.1810 | 94.3% | 854 | 59.8% |
| `KXHIGHLAX-25SEP26-B78.5` | 0.2111 | 92.4% | 766 | 37.1% |
| `KXHIGHLAX-25SEP17-B84.5` | 0.2235 | 94.7% | 588 | 63.3% |
| `KXHIGHLAX-25SEP28-B70.5` | 0.2309 | 93.7% | 555 | 61.3% |
| `KXHIGHLAX-25SEP11-B71.5` | 0.2325 | 92.6% | 659 | 53.6% |
| `KXHIGHLAX-25SEP07-B83.5` | 0.2733 | 87.8% | 760 | 40.7% |
| `KXHIGHLAX-25SEP26-T72` | 0.2869 | 91.1% | 1,096 | 57.4% |
| `KXHIGHLAX-25SEP30-B69.5` | 0.3344 | 88.0% | 768 | 48.2% |
| `KXHIGHLAX-25SEP05-B79.5` | 0.3412 | 88.1% | 933 | 56.6% |
| `KXHIGHLAX-25SEP13-B71.5` | 0.3518 | 86.7% | 655 | 58.6% |
| `KXHIGHLAX-25SEP04-T76` | 0.3901 | 80.5% | 662 | 45.2% |
| `KXHIGHLAX-25SEP27-T76` | 0.3957 | 89.4% | 936 | 29.9% |
| `KXHIGHLAX-25SEP23-B74.5` | 0.4100 | 89.7% | 437 | 33.4% |
| `KXHIGHLAX-25SEP05-B73.5` | 0.4155 | 82.0% | 1,197 | 54.8% |
| `KXHIGHLAX-25SEP10-T77` | 0.4223 | 83.6% | 903 | 35.1% |
| `KXHIGHLAX-25SEP24-T79` | 0.4481 | 88.1% | 1,130 | 30.1% |
| `KXHIGHLAX-25SEP09-B73.5` | 0.4712 | 83.9% | 1,135 | 45.8% |
| `KXHIGHLAX-25SEP15-B80.5` | 0.4764 | 76.9% | 720 | 34.4% |
| `KXHIGHLAX-25SEP19-B75.5` | 0.4766 | 81.1% | 1,007 | 46.7% |
| `KXHIGHLAX-25SEP06-B76.5` | 0.4881 | 79.0% | 794 | 36.1% |
| `KXHIGHLAX-25SEP21-B79.5` | 0.4950 | 84.7% | 531 | 40.1% |
| `KXHIGHLAX-25SEP16-B83.5` | 0.5092 | 80.6% | 816 | 34.1% |
| `KXHIGHLAX-25SEP25-B70.5` | 0.5192 | 79.1% | 965 | 33.9% |
| `KXHIGHLAX-25SEP06-T79` | 0.5276 | 83.4% | 730 | 24.8% |
| `KXHIGHLAX-25SEP06-B78.5` | 0.5279 | 77.8% | 966 | 28.1% |

## Key Findings

1. **Naive baseline**: log loss 0.5999 (predicting class priors 9%/82%/9% always)
2. **Best LR**: `model5_lr` — log loss 0.5086 (+15.2% vs naive), accuracy 82.2%
3. **GBM**: log loss 0.6348 (-5.8% vs naive), accuracy 72.4% (-9.4pp vs majority)
4. **Top GBM features**: `tod_cos`, `tod_sin`, `vol_sum_15`
5. **UP recall**: GBM 45.1% vs LR 3.9%  |  **DOWN recall**: GBM 49.0% vs LR 9.0%

## Limitations

- NEUTRAL class (~35%) is structurally unpredictable from prior bars alone; it sets a log loss floor no model can escape
- Three-class log loss penalises confident wrong predictions heavily — a model that says 90% UP when the outcome is DOWN is punished severely
- Precision on UP/DOWN is what matters for trading; filter on high predicted probability (e.g. P(UP) > 0.45) to find actionable bars
- No transaction cost model — bid/ask spread must be overcome for profitability
