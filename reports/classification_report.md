# Kalshi Weather Market: 3-Class Classification (UP / NEUTRAL / DOWN)

_Generated: 2026-07-06 07:05 UTC_

**Train**: 2025-06-01 – 2025-08-31 (244,076 bars)  **Test**: 2025-09-01 – 2025-09-30 (127,245 bars)

## Methodology

- **Classes**: DOWN (`move < -0.05`, 16.1%), NEUTRAL (`|move| ≤ 0.05`, 67.2%), UP (`move > +0.05`, 16.7%)
- **All bars included** — model predicts on every bar as in live trading
- **Loss**: negative log loss (categorical cross-entropy), lower = better
- **Naive baseline**: predict class priors on every bar → log loss = 0.8595
- **LightGBM**: `objective='multiclass'`, `num_class=3`, `class_weight='balanced'`
- **Logistic Regression**: `multi_class='multinomial'`, `solver='lbfgs'`

## Model Comparison

Naive log loss: **0.8595**  |  Majority-class accuracy baseline: **67.2%** (always predict NEUTRAL)

| Model | Log Loss | vs Naive | Accuracy | vs Majority | Prec↑up | Rec↑up | Prec↓down | Rec↓down |
|---|---|---|---|---|---|---|---|---|
| baseline_lr | **0.8211** | +4.5% | 66.8% | -0.4pp | 33.6% | 7.5% | 0.0% | 0.0% |
| model1_lr | **0.7609** | +11.5% | 68.7% | +1.5pp | 38.3% | 11.5% | 51.3% | 13.9% |
| model2_lr | **0.7573** | +11.9% | 68.8% | +1.5pp | 39.4% | 11.3% | 51.3% | 15.3% |
| model3_lr | **0.7505** | +12.7% | 69.1% | +1.8pp | 40.6% | 15.7% | 51.6% | 13.7% |
| model4_lr | **0.7501** | +12.7% | 69.0% | +1.8pp | 40.2% | 15.8% | 51.4% | 13.5% |
| model5_lr | **0.7328** | +14.7% | 70.4% | +3.1pp | 48.5% | 15.1% | 52.7% | 16.3% |
| model6_gbm | **0.7569** | +11.9% | 64.4% | -2.9pp | 34.6% | 44.3% | 38.3% | 49.6% |

### Confusion Matrix — model5_lr

Rows = actual, columns = predicted (DOWN / NEUTRAL / UP)

| Actual \ Predicted | DOWN | NEUTRAL | UP |
|---|---|---|---|
| **DOWN** | 3,329 (16%) | 15,078 (74%) | 2,072 (10%) |
| **NEUTRAL** | 1,235 (1%) | 82,997 (97%) | 1,324 (2%) |
| **UP** | 1,754 (8%) | 16,253 (77%) | 3,203 (15%) |

## Feature Importance: Logistic Regression (model5_lr — up_vs_rest)

Mean absolute coefficient across DOWN / NEUTRAL / UP planes, normalised. Also shows per-class coefficient direction.

| Rank | Feature | Importance % | Coef↓down | Coef~neutral | Coef↑up | Category |
|---|---|---|---|---|---|---|
| 1 | `price_polarization` | 10.77% | +0.440 | -0.530 | +0.090 | resolution |
| 2 | `bar_hour` | 8.13% | -0.168 | -0.232 | +0.400 | time |
| 3 | `frac_day_elapsed` | 7.92% | +0.274 | +0.116 | -0.390 | resolution |
| 4 | `ask_lag15` | 7.38% | +0.092 | -0.363 | +0.271 | momentum |
| 5 | `mins_to_peak` | 5.46% | -0.147 | +0.269 | -0.122 | resolution |
| 6 | `tod_cos` | 5.11% | -0.083 | -0.168 | +0.251 | time |
| 7 | `ask_lag5` | 4.60% | -0.226 | +0.116 | +0.111 | momentum |
| 8 | `relative_spread` | 4.07% | +0.200 | -0.059 | -0.142 | cross |
| 9 | `tod_sin` | 4.00% | +0.154 | +0.043 | -0.197 | time |
| 10 | `vol_zscore` | 3.08% | -0.033 | +0.152 | -0.119 | burst |
| 11 | `avg_spread_all` | 2.96% | +0.146 | -0.018 | -0.127 | cross |
| 12 | `price_previous` | 2.85% | -0.140 | +0.126 | +0.015 | anchor |
| 13 | `mid_vol_15` | 2.82% | +0.059 | -0.139 | +0.079 | volatility |
| 14 | `rel_mid` | 2.50% | +0.096 | +0.028 | -0.123 | cross |
| 15 | `mid_close` | 2.38% | -0.041 | +0.117 | -0.076 | anchor |
| 16 | `spread` | 2.10% | +0.086 | -0.103 | +0.017 | spread |
| 17 | `yes_ask_close` | 2.03% | -0.029 | +0.100 | -0.071 | anchor |
| 18 | `vol_ratio` | 1.85% | +0.019 | -0.091 | +0.072 | burst |
| 19 | `vol_sum_15` | 1.76% | +0.014 | +0.072 | -0.087 | flow |
| 20 | `spread_dispersion` | 1.57% | -0.055 | +0.077 | -0.022 | cross |

### Category Summary (Logistic Regression)

| Category | Total % |
|---|---|
| resolution | 24.5% |
| time | 17.7% |
| momentum | 16.4% |
| cross | 14.2% |
| anchor | 7.3% |
| burst | 6.1% |
| spread | 5.5% |
| volatility | 4.2% |
| flow | 3.3% |
| metar | 0.9% |

## Feature Importance: LightGBM (model6_gbm)

Gain importance (total log-loss reduction from splits).

| Rank | Feature | Gain | Share | Category | Bar |
|---|---|---|---|---|---|
| 1 | `tod_sin` | 5,576 | 6.0% | time | `█████████████████████████` |
| 2 | `tod_cos` | 5,289 | 5.7% | time | `████████████████████████░` |
| 3 | `price_previous` | 5,284 | 5.7% | anchor | `████████████████████████░` |
| 4 | `vol_sum_15` | 4,955 | 5.3% | flow | `██████████████████████░░░` |
| 5 | `oi_change_15` | 4,743 | 5.1% | flow | `█████████████████████░░░░` |
| 6 | `mid_ret_30` | 4,547 | 4.9% | momentum | `████████████████████░░░░░` |
| 7 | `mid_vol_15` | 3,958 | 4.3% | volatility | `██████████████████░░░░░░░` |
| 8 | `mid_close` | 3,947 | 4.2% | anchor | `██████████████████░░░░░░░` |
| 9 | `ask_lag15` | 3,653 | 3.9% | momentum | `████████████████░░░░░░░░░` |
| 10 | `rel_mid` | 3,158 | 3.4% | cross | `██████████████░░░░░░░░░░░` |
| 11 | `mins_to_peak` | 2,926 | 3.1% | resolution | `█████████████░░░░░░░░░░░░` |
| 12 | `yes_ask_close` | 2,856 | 3.1% | anchor | `█████████████░░░░░░░░░░░░` |
| 13 | `spread_dispersion` | 2,688 | 2.9% | cross | `████████████░░░░░░░░░░░░░` |
| 14 | `avg_spread_all` | 2,453 | 2.6% | cross | `███████████░░░░░░░░░░░░░░` |
| 15 | `relative_spread` | 2,408 | 2.6% | cross | `███████████░░░░░░░░░░░░░░` |
| 16 | `mid_ret_15` | 2,292 | 2.5% | momentum | `██████████░░░░░░░░░░░░░░░` |
| 17 | `spread` | 2,256 | 2.4% | spread | `██████████░░░░░░░░░░░░░░░` |
| 18 | `price_polarization` | 2,030 | 2.2% | resolution | `█████████░░░░░░░░░░░░░░░░` |
| 19 | `ask_lag5` | 2,028 | 2.2% | momentum | `█████████░░░░░░░░░░░░░░░░` |
| 20 | `frac_day_elapsed` | 1,978 | 2.1% | resolution | `█████████░░░░░░░░░░░░░░░░` |

### Category Summary (GBM)

| Category | Total Gain % |
|---|---|
| cross | 17.0% |
| momentum | 16.8% |
| time | 14.0% |
| flow | 13.3% |
| anchor | 13.0% |
| resolution | 8.3% |
| spread | 6.8% |
| volatility | 6.3% |
| burst | 2.3% |
| metar | 2.2% |

## Per-Ticker Log Loss (GBM, Test Set)

Best 30 tickers by log loss. High-flat tickers score trivially well.

| Ticker | Log Loss | Accuracy | N Rows | % Flat |
|---|---|---|---|---|
| `KXHIGHLAX-25SEP25-T70` | 0.0609 | 98.7% | 459 | 54.0% |
| `KXHIGHLAX-25SEP14-B72.5` | 0.0850 | 97.0% | 395 | 60.5% |
| `KXHIGHLAX-25SEP07-T84` | 0.0954 | 99.1% | 633 | 35.9% |
| `KXHIGHLAX-25SEP21-T73` | 0.1788 | 96.4% | 467 | 50.5% |
| `KXHIGHLAX-25SEP13-T78` | 0.2279 | 94.4% | 1,127 | 26.7% |
| `KXHIGHLAX-25SEP26-B78.5` | 0.2457 | 93.1% | 735 | 23.1% |
| `KXHIGHLAX-25SEP30-B69.5` | 0.2797 | 86.8% | 752 | 39.0% |
| `KXHIGHLAX-25SEP22-B80.5` | 0.2825 | 93.8% | 645 | 16.3% |
| `KXHIGHLAX-25SEP07-B83.5` | 0.2954 | 87.3% | 714 | 29.6% |
| `KXHIGHLAX-25SEP04-B80.5` | 0.3085 | 88.9% | 739 | 47.4% |
| `KXHIGHLAX-25SEP28-B70.5` | 0.3450 | 85.9% | 516 | 43.8% |
| `KXHIGHLAX-25SEP26-T72` | 0.4005 | 87.2% | 1,051 | 25.5% |
| `KXHIGHLAX-25SEP13-B71.5` | 0.4345 | 84.4% | 596 | 15.1% |
| `KXHIGHLAX-25SEP04-T76` | 0.4382 | 81.1% | 576 | 30.6% |
| `KXHIGHLAX-25SEP11-B71.5` | 0.4465 | 85.2% | 583 | 27.6% |
| `KXHIGHLAX-25SEP16-T77` | 0.4485 | 82.8% | 383 | 43.1% |
| `KXHIGHLAX-25SEP19-B73.5` | 0.4524 | 85.5% | 331 | 34.4% |
| `KXHIGHLAX-25SEP17-B84.5` | 0.4755 | 86.2% | 545 | 43.7% |
| `KXHIGHLAX-25SEP05-B79.5` | 0.4937 | 83.8% | 885 | 39.7% |
| `KXHIGHLAX-25SEP24-T79` | 0.5140 | 83.4% | 1,099 | 20.0% |
| `KXHIGHLAX-25SEP05-B73.5` | 0.5803 | 74.9% | 1,142 | 38.9% |
| `KXHIGHLAX-25SEP10-T77` | 0.5897 | 72.5% | 877 | 16.8% |
| `KXHIGHLAX-25SEP21-B79.5` | 0.6057 | 80.3% | 472 | 19.9% |
| `KXHIGHLAX-25SEP15-B80.5` | 0.6086 | 75.1% | 668 | 14.2% |
| `KXHIGHLAX-25SEP06-B76.5` | 0.6203 | 75.7% | 779 | 27.3% |
| `KXHIGHLAX-25SEP27-T76` | 0.6207 | 78.8% | 910 | 12.9% |
| `KXHIGHLAX-25SEP10-B72.5` | 0.6392 | 73.3% | 972 | 23.8% |
| `KXHIGHLAX-25SEP01-B84.5` | 0.6495 | 74.7% | 972 | 35.1% |
| `KXHIGHLAX-25SEP16-B83.5` | 0.6516 | 77.0% | 791 | 13.0% |
| `KXHIGHLAX-25SEP14-B76.5` | 0.6853 | 67.3% | 1,190 | 16.7% |

## Key Findings

1. **Naive baseline**: log loss 0.8595 (predicting class priors 16%/67%/17% always)
2. **Best LR**: `model5_lr` — log loss 0.7328 (+14.7% vs naive), accuracy 70.4%
3. **GBM**: log loss 0.7569 (+11.9% vs naive), accuracy 64.4% (-2.9pp vs majority)
4. **Top GBM features**: `tod_sin`, `tod_cos`, `price_previous`
5. **UP recall**: GBM 44.3% vs LR 15.1%  |  **DOWN recall**: GBM 49.6% vs LR 16.3%

## Limitations

- NEUTRAL class (~35%) is structurally unpredictable from prior bars alone; it sets a log loss floor no model can escape
- Three-class log loss penalises confident wrong predictions heavily — a model that says 90% UP when the outcome is DOWN is punished severely
- Precision on UP/DOWN is what matters for trading; filter on high predicted probability (e.g. P(UP) > 0.45) to find actionable bars
- No transaction cost model — bid/ask spread must be overcome for profitability
