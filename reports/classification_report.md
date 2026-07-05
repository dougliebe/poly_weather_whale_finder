# Kalshi Weather Market: Price Direction Classification Report

_Generated: 2026-07-05 16:07 UTC_

**Train**: 2025-06-01 – 2025-08-31 (279,538 bars)  **Test**: 2025-09-01 – 2025-09-30 (134,786 bars)

## Methodology

- **All bars included in evaluation** — flat-move bars (~35%) are scored as incorrect for any directional prediction, matching real trading conditions where we cannot pre-filter bars by their future outcome
- **Label (up_vs_rest)**: 1 if `target_move_30 > 0`, else 0 (~32% positive)
- **Label (down_vs_rest)**: 1 if `target_move_30 < 0`, else 0 (~33% positive)
- **Majority-class baseline**: predict 0 always — equals the negative class frequency
- **Train**: 2025-06-01 – 2025-08-31 (all bars, no filtering)
- **Test**: 2025-09-01 – 2025-09-30 (all bars, no filtering)
- **NaN handling**: median imputation fit on training set only

## Model Comparison — Predict UP (move > 0) vs Flat-or-Down

Positive class frequency: 32.7%  →  Majority-class baseline: **67.3%**

| Model | Accuracy | vs Baseline | AUC | Precision | Recall |
|---|---|---|---|---|---|
| baseline_lr | **67.4%** | +0.1pp | 0.6290 | 51.3% | 7.8% |
| model1_lr | **67.5%** | +0.2pp | 0.6443 | 51.6% | 10.5% |
| model2_lr | **67.5%** | +0.2pp | 0.6430 | 51.8% | 10.6% |
| model3_lr | **67.7%** | +0.4pp | 0.6454 | 52.3% | 14.2% |
| model4_gbm_up | **68.4%** | +1.2pp | 0.6868 | 53.8% | 26.0% |

## Model Comparison — Predict DOWN (move < 0) vs Flat-or-Up

Positive class frequency: 34.1%  →  Majority-class baseline: **65.9%**

| Model | Accuracy | vs Baseline | AUC | Precision | Recall |
|---|---|---|---|---|---|
| model4_gbm_down | **68.8%** | +2.8pp | 0.7155 | 58.7% | 28.1% |

## Feature Importance: Logistic Regression (Model 3 — up_vs_rest)

Coefficients normalized to % of total absolute weight after StandardScaler.

| Rank | Feature | Importance % | Coef | Category |
|---|---|---|---|---|
| 1 | `ask_lag1` | 17.65% | +0.5866 | momentum |
| 2 | `ask_lag15` | 9.58% | +0.3183 | momentum |
| 3 | `tod_sin` | 8.92% | -0.2966 | time |
| 4 | `yes_ask_close` | 8.30% | -0.2759 | anchor |
| 5 | `mid_close` | 7.84% | -0.2605 | anchor |
| 6 | `spread` | 6.64% | -0.2207 | spread |
| 7 | `bar_hour` | 5.08% | -0.1688 | time |
| 8 | `ask_lag5` | 4.67% | +0.1553 | momentum |
| 9 | `tod_cos` | 3.35% | -0.1114 | time |
| 10 | `spread_lag5` | 3.27% | +0.1086 | spread |
| 11 | `mid_vol_15` | 2.88% | +0.0959 | volatility |
| 12 | `price_previous` | 2.30% | -0.0764 | anchor |
| 13 | `ask_intrabar_range` | 1.83% | +0.0608 | volatility |
| 14 | `relative_spread` | 1.67% | -0.0554 | cross |
| 15 | `spread_lag1` | 1.63% | +0.0541 | spread |
| 16 | `avg_spread_all` | 1.45% | -0.0483 | cross |
| 17 | `vol_sum_15` | 1.44% | -0.0480 | flow |
| 18 | `rel_mid` | 1.23% | +0.0408 | cross |
| 19 | `minute_of_hour` | 1.22% | +0.0405 | time |
| 20 | `mid_ret_1` | 1.22% | -0.0405 | momentum |

### Category Summary (Logistic Regression)

| Category | Total Importance % |
|---|---|
| momentum | 34.2% |
| time | 18.6% |
| anchor | 18.4% |
| spread | 11.5% |
| cross | 6.7% |
| volatility | 5.7% |
| flow | 4.8% |

## Feature Importance: LightGBM (up_vs_rest)

| Rank | Feature | Gain | Share | Category | Bar |
|---|---|---|---|---|---|
| 1 | `tod_cos` | 2,399 | 7.7% | time | `█████████████████████████` |
| 2 | `tod_sin` | 2,355 | 7.6% | time | `█████████████████████████` |
| 3 | `mid_ret_30` | 1,783 | 5.8% | momentum | `███████████████████░░░░░░` |
| 4 | `price_previous` | 1,722 | 5.6% | anchor | `██████████████████░░░░░░░` |
| 5 | `oi_change_15` | 1,686 | 5.4% | flow | `██████████████████░░░░░░░` |
| 6 | `vol_sum_15` | 1,675 | 5.4% | flow | `█████████████████░░░░░░░░` |
| 7 | `mid_vol_15` | 1,452 | 4.7% | volatility | `███████████████░░░░░░░░░░` |
| 8 | `mid_close` | 1,328 | 4.3% | anchor | `██████████████░░░░░░░░░░░` |
| 9 | `rel_mid` | 1,319 | 4.3% | cross | `██████████████░░░░░░░░░░░` |
| 10 | `yes_ask_close` | 1,145 | 3.7% | anchor | `████████████░░░░░░░░░░░░░` |
| 11 | `ask_lag15` | 1,104 | 3.6% | momentum | `████████████░░░░░░░░░░░░░` |
| 12 | `mid_ret_15` | 1,066 | 3.4% | momentum | `███████████░░░░░░░░░░░░░░` |
| 13 | `spread` | 1,065 | 3.4% | spread | `███████████░░░░░░░░░░░░░░` |
| 14 | `relative_spread` | 1,004 | 3.2% | cross | `██████████░░░░░░░░░░░░░░░` |
| 15 | `minute_of_hour` | 932 | 3.0% | time | `██████████░░░░░░░░░░░░░░░` |
| 16 | `avg_spread_all` | 898 | 2.9% | cross | `█████████░░░░░░░░░░░░░░░░` |
| 17 | `spread_dispersion` | 878 | 2.8% | cross | `█████████░░░░░░░░░░░░░░░░` |
| 18 | `sum_mid_all` | 820 | 2.6% | cross | `█████████░░░░░░░░░░░░░░░░` |
| 19 | `prob_sum_deviation` | 752 | 2.4% | cross | `████████░░░░░░░░░░░░░░░░░` |
| 20 | `bar_hour` | 703 | 2.3% | time | `███████░░░░░░░░░░░░░░░░░░` |

### Category Summary (GBM)

| Category | Total Gain % |
|---|---|
| time | 20.6% |
| cross | 19.2% |
| momentum | 18.8% |
| flow | 13.6% |
| anchor | 13.5% |
| spread | 7.1% |
| volatility | 7.1% |

## Per-Ticker Accuracy (GBM up_vs_rest, Test Set)

Sorted by accuracy descending.

| Ticker | Accuracy | N Rows | Mean |move| | % Flat |
|---|---|---|---|---|
| `KXHIGHLAX-25SEP06-T72` | 100.0% | 219 | 0.0001 | 98.6% |
| `KXHIGHLAX-25SEP28-T70` | 100.0% | 426 | 0.0001 | 99.1% |
| `KXHIGHLAX-25SEP26-T79` | 100.0% | 549 | 0.0010 | 90.3% |
| `KXHIGHLAX-25SEP12-B70.5` | 100.0% | 161 | 0.0006 | 93.8% |
| `KXHIGHLAX-25SEP29-B68.5` | 100.0% | 398 | 0.0010 | 91.2% |
| `KXHIGHLAX-25SEP13-T71` | 98.8% | 255 | 0.0004 | 96.9% |
| `KXHIGHLAX-25SEP09-T71` | 98.4% | 253 | 0.0010 | 93.7% |
| `KXHIGHLAX-25SEP15-T74` | 98.0% | 397 | 0.0013 | 89.7% |
| `KXHIGHLAX-25SEP05-T73` | 97.8% | 769 | 0.0013 | 87.5% |
| `KXHIGHLAX-25SEP19-B71.5` | 97.7% | 173 | 0.0014 | 87.3% |
| `KXHIGHLAX-25SEP06-B72.5` | 96.0% | 500 | 0.0020 | 81.8% |
| `KXHIGHLAX-25SEP22-T74` | 95.7% | 188 | 0.0033 | 89.4% |
| `KXHIGHLAX-25SEP10-B70.5` | 94.5% | 660 | 0.0028 | 83.9% |
| `KXHIGHLAX-25SEP14-T72` | 93.7% | 111 | 0.0054 | 71.2% |
| `KXHIGHLAX-25SEP27-B69.5` | 93.1% | 608 | 0.0048 | 76.2% |
| `KXHIGHLAX-25SEP21-T73` | 91.9% | 508 | 0.0055 | 73.6% |
| `KXHIGHLAX-25SEP04-B82.5` | 91.5% | 495 | 0.0066 | 65.7% |
| `KXHIGHLAX-25SEP14-B72.5` | 91.2% | 410 | 0.0057 | 75.9% |
| `KXHIGHLAX-25SEP20-T72` | 90.6% | 160 | 0.0026 | 75.0% |
| `KXHIGHLAX-25SEP05-T80` | 90.5% | 633 | 0.0030 | 78.5% |
| `KXHIGHLAX-25SEP10-T70` | 90.0% | 673 | 0.0057 | 78.6% |
| `KXHIGHLAX-25SEP08-B83.5` | 89.9% | 903 | 0.0031 | 75.6% |
| `KXHIGHLAX-25SEP20-B72.5` | 89.5% | 219 | 0.0032 | 70.8% |
| `KXHIGHLAX-25SEP25-T70` | 88.2% | 526 | 0.0063 | 70.0% |
| `KXHIGHLAX-25SEP09-B71.5` | 87.9% | 453 | 0.0063 | 64.7% |
| `KXHIGHLAX-25SEP04-B80.5` | 87.1% | 854 | 0.0120 | 59.8% |
| `KXHIGHLAX-25SEP24-B72.5` | 86.7% | 428 | 0.0073 | 61.4% |
| `KXHIGHLAX-25SEP08-T84` | 85.6% | 779 | 0.0042 | 69.8% |
| `KXHIGHLAX-25SEP17-B84.5` | 85.5% | 588 | 0.0097 | 63.3% |
| `KXHIGHLAX-25SEP19-B73.5` | 85.4% | 349 | 0.0087 | 49.6% |
| `KXHIGHLAX-25SEP18-T76` | 85.3% | 245 | 0.0064 | 60.0% |
| `KXHIGHLAX-25SEP28-B70.5` | 84.3% | 555 | 0.0113 | 61.3% |
| `KXHIGHLAX-25SEP21-T80` | 83.9% | 385 | 0.0047 | 63.4% |
| `KXHIGHLAX-25SEP05-B79.5` | 83.7% | 933 | 0.0118 | 56.6% |
| `KXHIGHLAX-25SEP16-T77` | 82.6% | 430 | 0.0064 | 65.8% |
| `KXHIGHLAX-25SEP07-T77` | 81.9% | 744 | 0.0066 | 54.3% |
| `KXHIGHLAX-25SEP14-B78.5` | 81.6% | 832 | 0.0160 | 34.6% |
| `KXHIGHLAX-25SEP07-T84` | 81.2% | 691 | 0.0072 | 57.9% |
| `KXHIGHLAX-25SEP21-B79.5` | 81.0% | 531 | 0.0143 | 40.1% |
| `KXHIGHLAX-25SEP14-T79` | 80.6% | 1,301 | 0.0055 | 61.0% |
| `KXHIGHLAX-25SEP16-T84` | 80.5% | 724 | 0.0063 | 62.4% |
| `KXHIGHLAX-25SEP11-B71.5` | 80.4% | 659 | 0.0151 | 53.6% |
| `KXHIGHLAX-25SEP30-B69.5` | 80.1% | 768 | 0.0143 | 48.2% |
| `KXHIGHLAX-25SEP04-T76` | 79.6% | 662 | 0.0158 | 45.2% |
| `KXHIGHLAX-25SEP13-T78` | 79.3% | 1,184 | 0.0077 | 53.6% |
| `KXHIGHLAX-25SEP28-T77` | 78.7% | 717 | 0.0086 | 41.6% |
| `KXHIGHLAX-25SEP19-B75.5` | 77.9% | 1,007 | 0.0191 | 46.7% |
| `KXHIGHLAX-25SEP26-T72` | 77.4% | 1,096 | 0.0131 | 57.4% |
| `KXHIGHLAX-25SEP05-B73.5` | 77.4% | 1,197 | 0.0112 | 54.8% |
| `KXHIGHLAX-25SEP01-B84.5` | 77.3% | 1,018 | 0.0216 | 48.3% |
| `KXHIGHLAX-25SEP12-B72.5` | 77.1% | 809 | 0.0079 | 55.7% |
| `KXHIGHLAX-25SEP27-T76` | 76.5% | 936 | 0.0180 | 29.9% |
| `KXHIGHLAX-25SEP09-T78` | 75.4% | 788 | 0.0077 | 49.5% |
| `KXHIGHLAX-25SEP15-B80.5` | 75.3% | 720 | 0.0183 | 34.4% |
| `KXHIGHLAX-25SEP13-B71.5` | 75.3% | 655 | 0.0108 | 58.6% |
| `KXHIGHLAX-25SEP15-B78.5` | 74.8% | 1,265 | 0.0370 | 18.1% |
| `KXHIGHLAX-25SEP03-B84.5` | 74.7% | 867 | 0.0268 | 38.9% |
| `KXHIGHLAX-25SEP26-B78.5` | 74.0% | 766 | 0.0146 | 37.1% |
| `KXHIGHLAX-25SEP09-B73.5` | 73.9% | 1,135 | 0.0241 | 45.8% |
| `KXHIGHLAX-25SEP25-T77` | 73.6% | 959 | 0.0090 | 45.5% |
| `KXHIGHLAX-25SEP24-T79` | 72.5% | 1,130 | 0.0178 | 30.1% |
| `KXHIGHLAX-25SEP06-B74.5` | 71.9% | 715 | 0.0233 | 33.6% |
| `KXHIGHLAX-25SEP29-B70.5` | 71.8% | 999 | 0.0394 | 32.2% |
| `KXHIGHLAX-25SEP17-B82.5` | 71.2% | 809 | 0.0251 | 37.1% |
| `KXHIGHLAX-25SEP13-B77.5` | 71.1% | 938 | 0.0189 | 28.5% |
| `KXHIGHLAX-25SEP04-B78.5` | 70.8% | 850 | 0.0398 | 22.5% |
| `KXHIGHLAX-25SEP11-T76` | 70.4% | 866 | 0.0235 | 25.8% |
| `KXHIGHLAX-25SEP10-B72.5` | 70.4% | 1,001 | 0.0322 | 36.6% |
| `KXHIGHLAX-25SEP16-B83.5` | 70.1% | 816 | 0.0230 | 34.1% |
| `KXHIGHLAX-25SEP07-B77.5` | 69.6% | 918 | 0.0267 | 31.2% |
| `KXHIGHLAX-25SEP23-B80.5` | 69.3% | 982 | 0.0469 | 29.1% |
| `KXHIGHLAX-25SEP07-B83.5` | 68.9% | 760 | 0.0150 | 40.7% |
| `KXHIGHLAX-25SEP26-B76.5` | 68.9% | 961 | 0.0332 | 14.5% |
| `KXHIGHLAX-25SEP22-B80.5` | 68.8% | 670 | 0.0122 | 31.6% |
| `KXHIGHLAX-25SEP16-B77.5` | 68.2% | 785 | 0.0330 | 36.8% |
| `KXHIGHLAX-25SEP08-B81.5` | 68.1% | 1,022 | 0.0238 | 32.1% |
| `KXHIGHLAX-25SEP11-B73.5` | 68.1% | 1,080 | 0.0383 | 31.9% |
| `KXHIGHLAX-25SEP07-B81.5` | 67.7% | 1,033 | 0.0375 | 31.8% |
| `KXHIGHLAX-25SEP06-B78.5` | 67.6% | 966 | 0.0369 | 28.1% |
| `KXHIGHLAX-25SEP02-B85.5` | 67.5% | 906 | 0.0212 | 28.6% |
| `KXHIGHLAX-25SEP03-B82.5` | 67.2% | 990 | 0.0306 | 27.2% |
| `KXHIGHLAX-25SEP27-B71.5` | 67.1% | 1,052 | 0.0560 | 20.5% |
| `KXHIGHLAX-25SEP11-B75.5` | 66.9% | 1,166 | 0.0387 | 31.8% |
| `KXHIGHLAX-25SEP23-B78.5` | 66.9% | 803 | 0.0463 | 22.7% |
| `KXHIGHLAX-25SEP17-B78.5` | 66.7% | 893 | 0.0509 | 23.6% |
| `KXHIGHLAX-25SEP14-B74.5` | 66.3% | 913 | 0.0398 | 30.9% |
| `KXHIGHLAX-25SEP10-B76.5` | 66.2% | 1,235 | 0.0588 | 17.4% |
| `KXHIGHLAX-25SEP29-B74.5` | 66.1% | 1,114 | 0.0649 | 13.6% |
| `KXHIGHLAX-25SEP24-B78.5` | 65.9% | 1,221 | 0.0454 | 22.7% |
| `KXHIGHLAX-25SEP25-B70.5` | 65.8% | 965 | 0.0690 | 33.9% |
| `KXHIGHLAX-25SEP25-B74.5` | 65.6% | 1,100 | 0.0472 | 18.9% |
| `KXHIGHLAX-25SEP12-B76.5` | 65.6% | 1,388 | 0.0240 | 33.9% |
| `KXHIGHLAX-25SEP10-B74.5` | 65.5% | 1,187 | 0.0443 | 27.2% |
| `KXHIGHLAX-25SEP22-B76.5` | 65.0% | 1,147 | 0.0358 | 27.5% |
| `KXHIGHLAX-25SEP01-B82.5` | 65.0% | 1,106 | 0.0266 | 26.0% |
| `KXHIGHLAX-25SEP23-B74.5` | 65.0% | 437 | 0.0170 | 33.4% |
| `KXHIGHLAX-25SEP01-B80.5` | 65.0% | 1,036 | 0.0306 | 22.2% |
| `KXHIGHLAX-25SEP01-T80` | 64.9% | 1,067 | 0.0368 | 29.3% |
| `KXHIGHLAX-25SEP28-B72.5` | 64.7% | 1,114 | 0.0408 | 13.2% |
| `KXHIGHLAX-25SEP20-T79` | 64.7% | 946 | 0.0483 | 18.0% |
| `KXHIGHLAX-25SEP02-B83.5` | 64.7% | 988 | 0.0380 | 16.5% |
| `KXHIGHLAX-25SEP18-B82.5` | 64.6% | 724 | 0.0293 | 20.7% |
| `KXHIGHLAX-25SEP06-T79` | 64.4% | 730 | 0.0450 | 24.8% |
| `KXHIGHLAX-25SEP18-B76.5` | 64.2% | 707 | 0.0450 | 20.7% |
| `KXHIGHLAX-25SEP15-B74.5` | 64.1% | 973 | 0.0415 | 26.1% |
| `KXHIGHLAX-25SEP05-B77.5` | 64.0% | 1,118 | 0.0434 | 27.3% |
| `KXHIGHLAX-25SEP20-B78.5` | 63.7% | 920 | 0.0408 | 14.5% |
| `KXHIGHLAX-25SEP12-T77` | 63.7% | 1,382 | 0.0255 | 35.4% |
| `KXHIGHLAX-25SEP19-B77.5` | 63.5% | 931 | 0.0678 | 21.1% |
| `KXHIGHLAX-25SEP16-B81.5` | 63.4% | 1,046 | 0.0299 | 29.3% |
| `KXHIGHLAX-25SEP30-B71.5` | 63.1% | 1,149 | 0.0307 | 29.0% |
| `KXHIGHLAX-25SEP24-B74.5` | 63.0% | 817 | 0.0379 | 28.0% |
| `KXHIGHLAX-25SEP10-T77` | 63.0% | 903 | 0.0194 | 35.1% |
| `KXHIGHLAX-25SEP13-B75.5` | 62.8% | 1,378 | 0.0336 | 29.2% |
| `KXHIGHLAX-25SEP21-B73.5` | 62.5% | 656 | 0.0346 | 32.0% |
| `KXHIGHLAX-25SEP09-B77.5` | 62.4% | 1,000 | 0.0266 | 23.0% |
| `KXHIGHLAX-25SEP13-B73.5` | 61.8% | 995 | 0.0460 | 23.1% |
| `KXHIGHLAX-25SEP18-B78.5` | 61.6% | 865 | 0.0528 | 14.7% |
| `KXHIGHLAX-25SEP18-B80.5` | 61.6% | 880 | 0.0555 | 19.2% |
| `KXHIGHLAX-25SEP08-B77.5` | 61.5% | 1,212 | 0.0436 | 15.5% |
| `KXHIGHLAX-25SEP30-B73.5` | 61.3% | 1,349 | 0.0389 | 14.5% |
| `KXHIGHLAX-25SEP20-B74.5` | 61.0% | 857 | 0.0640 | 12.3% |
| `KXHIGHLAX-25SEP06-B76.5` | 61.0% | 794 | 0.0362 | 36.1% |
| `KXHIGHLAX-25SEP21-B77.5` | 60.6% | 967 | 0.0599 | 17.4% |
| `KXHIGHLAX-25SEP22-B78.5` | 60.5% | 1,047 | 0.0433 | 17.7% |
| `KXHIGHLAX-25SEP03-T82` | 60.1% | 1,030 | 0.0386 | 16.4% |
| `KXHIGHLAX-25SEP28-B74.5` | 59.8% | 1,265 | 0.0481 | 10.5% |
| `KXHIGHLAX-25SEP08-B79.5` | 59.8% | 1,208 | 0.0514 | 14.4% |
| `KXHIGHLAX-25SEP16-B79.5` | 59.7% | 1,011 | 0.0359 | 22.6% |
| `KXHIGHLAX-25SEP30-T74` | 59.7% | 1,150 | 0.0467 | 17.5% |
| `KXHIGHLAX-25SEP02-B81.5` | 59.7% | 1,108 | 0.0647 | 7.6% |
| `KXHIGHLAX-25SEP24-B76.5` | 59.6% | 1,142 | 0.0538 | 21.8% |
| `KXHIGHLAX-25SEP08-T77` | 58.9% | 1,208 | 0.0673 | 8.9% |
| `KXHIGHLAX-25SEP05-B75.5` | 58.5% | 1,129 | 0.0359 | 20.7% |
| `KXHIGHLAX-25SEP26-B74.5` | 58.2% | 1,049 | 0.0549 | 13.3% |
| `KXHIGHLAX-25SEP17-B80.5` | 58.1% | 1,102 | 0.0480 | 18.6% |
| `KXHIGHLAX-25SEP14-B76.5` | 58.1% | 1,252 | 0.0338 | 37.7% |
| `KXHIGHLAX-25SEP25-B76.5` | 58.0% | 986 | 0.0301 | 26.6% |
| `KXHIGHLAX-25SEP27-B75.5` | 57.9% | 1,018 | 0.0352 | 24.4% |
| `KXHIGHLAX-25SEP28-B76.5` | 57.7% | 1,102 | 0.0451 | 11.7% |
| `KXHIGHLAX-25SEP19-T78` | 57.2% | 1,045 | 0.0662 | 13.2% |
| `KXHIGHLAX-25SEP29-B72.5` | 57.2% | 1,115 | 0.0600 | 10.1% |
| `KXHIGHLAX-25SEP21-B75.5` | 56.9% | 939 | 0.0387 | 28.3% |
| `KXHIGHLAX-25SEP12-B74.5` | 56.7% | 1,120 | 0.0264 | 18.4% |
| `KXHIGHLAX-25SEP17-T78` | 56.2% | 821 | 0.0487 | 12.8% |
| `KXHIGHLAX-25SEP29-T75` | 55.7% | 1,149 | 0.0385 | 19.9% |
| `KXHIGHLAX-25SEP27-B73.5` | 55.6% | 1,186 | 0.0424 | 13.1% |
| `KXHIGHLAX-25SEP15-B76.5` | 55.5% | 1,223 | 0.0401 | 24.5% |
| `KXHIGHLAX-25SEP09-B75.5` | 54.5% | 1,042 | 0.0260 | 20.3% |
| `KXHIGHLAX-25SEP22-B74.5` | 54.2% | 909 | 0.0295 | 23.8% |
| `KXHIGHLAX-25SEP20-B76.5` | 54.0% | 923 | 0.0589 | 12.2% |
| `KXHIGHLAX-25SEP26-B72.5` | 52.2% | 926 | 0.0443 | 14.6% |
| `KXHIGHLAX-25SEP04-B76.5` | 51.8% | 825 | 0.0322 | 18.4% |
| `KXHIGHLAX-25SEP25-B72.5` | 50.1% | 1,136 | 0.0757 | 10.2% |
| `KXHIGHLAX-25SEP07-B79.5` | 49.4% | 1,090 | 0.0397 | 24.5% |
| `KXHIGHLAX-25SEP23-B76.5` | 47.7% | 878 | 0.0424 | 27.6% |

## Key Findings

1. **Realistic baseline**: ~67% accuracy for always predicting flat-or-down (majority class). Any model must clear this bar.
2. **Best linear model**: `model3_lr` at 67.7% (+0.4pp vs baseline, AUC 0.6454)
3. **GBM (up_vs_rest)**: 68.4% accuracy (+1.2pp vs baseline, AUC 0.6868). Precision 53.8% means 54% of predicted-up bars actually rise — above coin-flip.
4. **GBM (down_vs_rest)**: 68.8% accuracy (+2.8pp vs baseline). Up and down signals are asymmetric.
5. **Top GBM features**: `tod_cos`, `tod_sin`, `mid_ret_30`

## Limitations

- Flat bars (~35% of test) mechanically cap accuracy — even a perfect signal for non-flat bars yields only ~65% overall accuracy if all flat bars are wrong
- Precision/recall tradeoff: a threshold on predicted probability can trade recall for higher-precision signals (fewer but more reliable trades)
- No transaction cost model — bid/ask spread (~1¢–2¢) must be overcome
- Model trained on summer 2025; seasonal/regime shift may affect Sep performance
