# Kalshi Weather Market: 3-Class Classification (UP / NEUTRAL / DOWN)

_Generated: 2026-07-05 22:32 UTC_

**Train**: 2025-06-01 – 2025-08-31 (279,538 bars)  **Test**: 2025-09-01 – 2025-09-30 (134,786 bars)

## Methodology

- **Classes**: DOWN (`move < 0`, 34.1%), NEUTRAL (`move = 0`, 33.2%), UP (`move > 0`, 32.7%)
- **All bars included** — model predicts on every bar as in live trading
- **Loss**: negative log loss (categorical cross-entropy), lower = better
- **Naive baseline**: predict class priors on every bar → log loss = 1.0985
- **LightGBM**: `objective='multiclass'`, `num_class=3`, `class_weight='balanced'`
- **Logistic Regression**: `multi_class='multinomial'`, `solver='lbfgs'`

## Model Comparison

Naive log loss: **1.0985**  |  Majority-class accuracy baseline: **34.1%** (always predict NEUTRAL)

| Model | Log Loss | vs Naive | Accuracy | vs Majority | Prec↑up | Rec↑up | Prec↓down | Rec↓down |
|---|---|---|---|---|---|---|---|---|
| baseline_lr | **1.0565** | +3.8% | 43.4% | +9.3pp | 42.7% | 37.5% | 42.7% | 20.6% |
| model1_lr | **1.0079** | +8.3% | 47.0% | +12.9pp | 44.0% | 37.1% | 55.0% | 27.3% |
| model2_lr | **1.0030** | +8.7% | 48.2% | +14.2pp | 44.8% | 34.9% | 53.6% | 35.5% |
| model3_lr | **0.9982** | +9.1% | 48.3% | +14.3pp | 46.3% | 33.5% | 52.2% | 36.5% |
| model4_lr | **0.9973** | +9.2% | 48.4% | +14.4pp | 46.4% | 33.8% | 52.3% | 36.7% |
| model5_lr | **0.9879** | +10.1% | 49.4% | +15.3pp | 46.7% | 34.6% | 52.5% | 37.8% |
| model6_gbm | **0.9120** | +17.0% | 53.9% | +19.9pp | 47.8% | 48.7% | 51.6% | 50.3% |

### Confusion Matrix — model6_gbm

Rows = actual, columns = predicted (DOWN / NEUTRAL / UP)

| Actual \ Predicted | DOWN | NEUTRAL | UP |
|---|---|---|---|
| **DOWN** | 23,108 (50%) | 8,005 (17%) | 14,784 (32%) |
| **NEUTRAL** | 7,985 (18%) | 28,079 (63%) | 8,685 (19%) |
| **UP** | 13,714 (31%) | 8,912 (20%) | 21,514 (49%) |

## Feature Importance: Logistic Regression (model5_lr — up_vs_rest)

Mean absolute coefficient across DOWN / NEUTRAL / UP planes, normalised. Also shows per-class coefficient direction.

| Rank | Feature | Importance % | Coef↓down | Coef~neutral | Coef↑up | Category |
|---|---|---|---|---|---|---|
| 1 | `frac_day_elapsed` | 12.66% | -0.451 | +0.733 | -0.282 | resolution |
| 2 | `bar_hour` | 6.95% | +0.203 | -0.402 | +0.199 | time |
| 3 | `price_polarization` | 6.43% | +0.237 | -0.372 | +0.135 | resolution |
| 4 | `vol_zscore` | 6.18% | -0.148 | +0.358 | -0.210 | burst |
| 5 | `tod_cos` | 5.29% | +0.191 | -0.306 | +0.115 | time |
| 6 | `mins_to_peak` | 5.05% | -0.240 | +0.292 | -0.052 | resolution |
| 7 | `relative_spread` | 4.25% | +0.246 | -0.153 | -0.093 | cross |
| 8 | `vol_ratio` | 3.99% | +0.093 | -0.231 | +0.138 | burst |
| 9 | `price_previous` | 3.90% | -0.149 | +0.226 | -0.077 | anchor |
| 10 | `ask_lag1` | 3.48% | -0.170 | -0.031 | +0.201 | momentum |
| 11 | `avg_spread_all` | 3.24% | +0.188 | -0.123 | -0.064 | cross |
| 12 | `ask_lag15` | 3.02% | +0.009 | -0.175 | +0.166 | momentum |
| 13 | `ask_intrabar_range` | 2.52% | +0.058 | -0.146 | +0.088 | volatility |
| 14 | `mid_vol_15` | 2.43% | +0.047 | -0.141 | +0.094 | volatility |
| 15 | `tod_sin` | 2.43% | +0.008 | +0.133 | -0.141 | time |
| 16 | `ask_lag5` | 2.12% | -0.123 | +0.064 | +0.059 | momentum |
| 17 | `yes_ask_close` | 1.85% | +0.107 | -0.035 | -0.072 | anchor |
| 18 | `mid_close` | 1.79% | +0.103 | -0.027 | -0.076 | anchor |
| 19 | `vol_burst_flag` | 1.77% | +0.047 | -0.102 | +0.055 | burst |
| 20 | `spread_lag1` | 1.65% | +0.041 | -0.095 | +0.055 | spread |

### Category Summary (Logistic Regression)

| Category | Total % |
|---|---|
| resolution | 24.7% |
| time | 15.2% |
| cross | 12.5% |
| burst | 12.1% |
| momentum | 10.8% |
| anchor | 7.5% |
| volatility | 6.6% |
| spread | 5.1% |
| flow | 3.1% |
| metar | 2.3% |

## Feature Importance: LightGBM (model6_gbm)

Gain importance (total log-loss reduction from splits).

| Rank | Feature | Gain | Share | Category | Bar |
|---|---|---|---|---|---|
| 1 | `tod_sin` | 5,448 | 5.9% | time | `█████████████████████████` |
| 2 | `tod_cos` | 5,207 | 5.6% | time | `████████████████████████░` |
| 3 | `mid_ret_30` | 4,656 | 5.0% | momentum | `█████████████████████░░░░` |
| 4 | `vol_sum_15` | 4,553 | 4.9% | flow | `█████████████████████░░░░` |
| 5 | `price_previous` | 4,445 | 4.8% | anchor | `████████████████████░░░░░` |
| 6 | `oi_change_15` | 4,177 | 4.5% | flow | `███████████████████░░░░░░` |
| 7 | `mid_vol_15` | 4,115 | 4.4% | volatility | `███████████████████░░░░░░` |
| 8 | `rel_mid` | 3,590 | 3.9% | cross | `████████████████░░░░░░░░░` |
| 9 | `mid_close` | 3,180 | 3.4% | anchor | `███████████████░░░░░░░░░░` |
| 10 | `mins_to_peak` | 2,871 | 3.1% | resolution | `█████████████░░░░░░░░░░░░` |
| 11 | `spread_dispersion` | 2,857 | 3.1% | cross | `█████████████░░░░░░░░░░░░` |
| 12 | `mid_ret_15` | 2,834 | 3.0% | momentum | `█████████████░░░░░░░░░░░░` |
| 13 | `relative_spread` | 2,824 | 3.0% | cross | `█████████████░░░░░░░░░░░░` |
| 14 | `yes_ask_close` | 2,764 | 3.0% | anchor | `█████████████░░░░░░░░░░░░` |
| 15 | `avg_spread_all` | 2,758 | 3.0% | cross | `█████████████░░░░░░░░░░░░` |
| 16 | `ask_lag15` | 2,682 | 2.9% | momentum | `████████████░░░░░░░░░░░░░` |
| 17 | `spread` | 2,582 | 2.8% | spread | `████████████░░░░░░░░░░░░░` |
| 18 | `sum_mid_all` | 2,281 | 2.5% | cross | `██████████░░░░░░░░░░░░░░░` |
| 19 | `prob_sum_deviation` | 2,097 | 2.3% | cross | `██████████░░░░░░░░░░░░░░░` |
| 20 | `price_polarization` | 2,008 | 2.2% | resolution | `█████████░░░░░░░░░░░░░░░░` |

### Category Summary (GBM)

| Category | Total Gain % |
|---|---|
| cross | 19.2% |
| momentum | 16.3% |
| time | 14.2% |
| flow | 12.1% |
| anchor | 11.2% |
| resolution | 8.0% |
| spread | 7.1% |
| volatility | 6.7% |
| metar | 2.7% |
| burst | 2.4% |

## Per-Ticker Log Loss (GBM, Test Set)

Best 30 tickers by log loss. High-flat tickers score trivially well.

| Ticker | Log Loss | Accuracy | N Rows | % Flat |
|---|---|---|---|---|
| `KXHIGHLAX-25SEP13-T71` | 0.1510 | 97.3% | 255 | 96.9% |
| `KXHIGHLAX-25SEP09-T71` | 0.2227 | 93.7% | 253 | 93.7% |
| `KXHIGHLAX-25SEP22-T74` | 0.2536 | 86.2% | 188 | 89.4% |
| `KXHIGHLAX-25SEP10-B70.5` | 0.3686 | 84.7% | 660 | 83.9% |
| `KXHIGHLAX-25SEP15-T74` | 0.3947 | 90.9% | 397 | 89.7% |
| `KXHIGHLAX-25SEP10-T70` | 0.3969 | 82.8% | 673 | 78.6% |
| `KXHIGHLAX-25SEP05-T73` | 0.4154 | 88.8% | 769 | 87.5% |
| `KXHIGHLAX-25SEP06-B72.5` | 0.4866 | 84.2% | 500 | 81.8% |
| `KXHIGHLAX-25SEP05-T80` | 0.4930 | 81.8% | 633 | 78.5% |
| `KXHIGHLAX-25SEP08-B83.5` | 0.5163 | 78.3% | 903 | 75.6% |
| `KXHIGHLAX-25SEP19-B71.5` | 0.5480 | 87.3% | 173 | 87.3% |
| `KXHIGHLAX-25SEP27-B69.5` | 0.5551 | 79.1% | 608 | 76.2% |
| `KXHIGHLAX-25SEP20-T72` | 0.5891 | 75.6% | 160 | 75.0% |
| `KXHIGHLAX-25SEP25-T70` | 0.6379 | 76.2% | 526 | 70.0% |
| `KXHIGHLAX-25SEP04-B80.5` | 0.6582 | 69.0% | 854 | 59.8% |
| `KXHIGHLAX-25SEP14-B72.5` | 0.6606 | 79.5% | 410 | 75.9% |
| `KXHIGHLAX-25SEP08-T84` | 0.6676 | 73.2% | 779 | 69.8% |
| `KXHIGHLAX-25SEP19-B75.5` | 0.6815 | 68.2% | 1,007 | 46.7% |
| `KXHIGHLAX-25SEP21-T73` | 0.7054 | 80.3% | 508 | 73.6% |
| `KXHIGHLAX-25SEP17-B84.5` | 0.7065 | 71.8% | 588 | 63.3% |
| `KXHIGHLAX-25SEP09-B71.5` | 0.7131 | 73.3% | 453 | 64.7% |
| `KXHIGHLAX-25SEP09-B73.5` | 0.7214 | 63.3% | 1,135 | 45.8% |
| `KXHIGHLAX-25SEP20-B72.5` | 0.7457 | 69.4% | 219 | 70.8% |
| `KXHIGHLAX-25SEP16-T77` | 0.7484 | 62.8% | 430 | 65.8% |
| `KXHIGHLAX-25SEP14-T72` | 0.7714 | 71.2% | 111 | 71.2% |
| `KXHIGHLAX-25SEP05-B73.5` | 0.7781 | 59.1% | 1,197 | 54.8% |
| `KXHIGHLAX-25SEP29-B70.5` | 0.7816 | 61.1% | 999 | 32.2% |
| `KXHIGHLAX-25SEP04-B82.5` | 0.7867 | 69.5% | 495 | 65.7% |
| `KXHIGHLAX-25SEP10-B74.5` | 0.7957 | 60.7% | 1,187 | 27.2% |
| `KXHIGHLAX-25SEP24-B72.5` | 0.7966 | 67.3% | 428 | 61.4% |

## Key Findings

1. **Naive baseline**: log loss 1.0985 (predicting class priors 34%/33%/33% always)
2. **Best LR**: `model5_lr` — log loss 0.9879 (+10.1% vs naive), accuracy 49.4%
3. **GBM**: log loss 0.9120 (+17.0% vs naive), accuracy 53.9% (+19.9pp vs majority)
4. **Top GBM features**: `tod_sin`, `tod_cos`, `mid_ret_30`
5. **UP recall**: GBM 48.7% vs LR 34.6%  |  **DOWN recall**: GBM 50.3% vs LR 37.8%

## Limitations

- NEUTRAL class (~35%) is structurally unpredictable from prior bars alone; it sets a log loss floor no model can escape
- Three-class log loss penalises confident wrong predictions heavily — a model that says 90% UP when the outcome is DOWN is punished severely
- Precision on UP/DOWN is what matters for trading; filter on high predicted probability (e.g. P(UP) > 0.45) to find actionable bars
- No transaction cost model — bid/ask spread must be overcome for profitability
