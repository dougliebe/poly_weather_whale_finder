# Kalshi Weather Market: Price Prediction Report

_Generated: 2026-07-06 02:44 UTC_

**Train**: 2025-06-01 – 2025-08-31 (244,076 rows)  **Test**: 2025-09-01 – 2025-09-30 (127,245 rows)

## Methodology

- **Target**: `target_move_90` = `yes_ask_open` at T+90 minus `yes_ask_close` at T — the 90-min price move
- **Train/Test split**: purely temporal (no shuffling) — 2025-06-01–2025-08-31 train, 2025-09-01–2025-09-30 test
- **NaN handling**: median imputation fit on training set only, applied to test
- **Prices**: decimal scale (0–1), where 0.72 = 72¢ / 72% implied probability
- **Naive benchmark**: predict zero price change (move = 0)
- **Evaluation**: MAE, RMSE, R², directional accuracy, % improvement vs naive

## Model Comparison

| Model | MAE | RMSE | R² | Dir Acc | vs Naive |
|---|---|---|---|---|---|
| Naive (no change) | 0.05891 | — | — | — | 0.0% |
| baseline | 0.06004 | 0.10919 | 0.0007 | 39.1% | -1.9% |
| model1_momentum | 0.05941 | 0.10838 | 0.0155 | 45.7% | -0.8% |
| model2_cross | 0.05947 | 0.10839 | 0.0154 | 45.9% | -0.9% |
| model3_tod | 0.05988 | 0.10829 | 0.0170 | 50.5% | -1.6% |
| model4_burst | 0.05988 | 0.10829 | 0.0170 | 50.5% | -1.6% |
| model5_full | 0.06018 | 0.10829 | 0.0171 | 50.2% | -2.1% |
| model6_gbm | 0.05969 | 0.11030 | -0.0197 | 51.3% | -1.3% |

## Feature Importance: Linear Model (Model 3 — all linear features)

Coefficients are normalized to % of total absolute weight after StandardScaler (i.e., features are on the same scale — higher % = stronger linear influence).

| Rank | Feature | Importance % | Coef | Category |
|---|---|---|---|---|
| 1 | `bar_hour` | 11.78% | +0.0263 | time |
| 2 | `yes_ask_close` | 11.27% | -0.0252 | anchor |
| 3 | `mid_close` | 11.05% | -0.0247 | anchor |
| 4 | `ask_lag1` | 10.75% | +0.0240 | momentum |
| 5 | `mins_to_peak` | 10.11% | +0.0226 | resolution |
| 6 | `ask_lag15` | 5.73% | +0.0128 | momentum |
| 7 | `spread` | 5.67% | -0.0127 | spread |
| 8 | `ask_lag5` | 4.41% | +0.0099 | momentum |
| 9 | `price_polarization` | 3.57% | +0.0080 | resolution |
| 10 | `tod_sin` | 2.91% | -0.0065 | time |
| 11 | `spread_lag5` | 2.44% | -0.0054 | spread |
| 12 | `vol_zscore` | 2.13% | -0.0048 | burst |
| 13 | `spread_chg_5` | 1.97% | -0.0044 | spread |
| 14 | `rel_mid` | 1.75% | -0.0039 | cross |
| 15 | `vol_ratio` | 1.67% | +0.0037 | burst |
| 16 | `tod_cos` | 1.35% | +0.0030 | time |
| 17 | `price_previous` | 1.28% | +0.0029 | anchor |
| 18 | `n_tickers_at_bar` | 1.12% | -0.0025 | cross |
| 19 | `mid_vol_15` | 0.85% | +0.0019 | volatility |
| 20 | `bid_intrabar_range` | 0.77% | +0.0017 | volatility |

### Category Summary (Linear)

| Category | Total Importance % |
|---|---|
| anchor | 23.6% |
| momentum | 22.2% |
| time | 16.5% |
| resolution | 14.1% |
| spread | 10.3% |
| cross | 4.7% |
| burst | 4.3% |
| volatility | 2.0% |
| metar | 1.5% |
| flow | 0.8% |

## Feature Importance: LightGBM (Model 4 — all features)

Importance by gain (total reduction in loss from splits on this feature).

| Rank | Feature | Gain | Share | Category | Bar |
|---|---|---|---|---|---|
| 1 | `tod_sin` | 1,864 | 6.0% | time | `█████████████████████████` |
| 2 | `oi_change_15` | 1,748 | 5.6% | flow | `███████████████████████░░` |
| 3 | `price_previous` | 1,730 | 5.6% | anchor | `███████████████████████░░` |
| 4 | `vol_sum_15` | 1,495 | 4.8% | flow | `████████████████████░░░░░` |
| 5 | `mid_ret_30` | 1,461 | 4.7% | momentum | `████████████████████░░░░░` |
| 6 | `mid_vol_15` | 1,347 | 4.3% | volatility | `██████████████████░░░░░░░` |
| 7 | `ask_lag15` | 1,303 | 4.2% | momentum | `█████████████████░░░░░░░░` |
| 8 | `mid_close` | 1,167 | 3.8% | anchor | `████████████████░░░░░░░░░` |
| 9 | `tod_cos` | 1,147 | 3.7% | time | `███████████████░░░░░░░░░░` |
| 10 | `frac_day_elapsed` | 1,114 | 3.6% | resolution | `███████████████░░░░░░░░░░` |
| 11 | `mins_to_peak` | 985 | 3.2% | resolution | `█████████████░░░░░░░░░░░░` |
| 12 | `ask_lag5` | 885 | 2.9% | momentum | `████████████░░░░░░░░░░░░░` |
| 13 | `yes_ask_close` | 831 | 2.7% | anchor | `███████████░░░░░░░░░░░░░░` |
| 14 | `rel_mid` | 807 | 2.6% | cross | `███████████░░░░░░░░░░░░░░` |
| 15 | `price_polarization` | 753 | 2.4% | resolution | `██████████░░░░░░░░░░░░░░░` |
| 16 | `mid_ret_15` | 734 | 2.4% | momentum | `██████████░░░░░░░░░░░░░░░` |
| 17 | `spread_lag5` | 731 | 2.4% | spread | `██████████░░░░░░░░░░░░░░░` |
| 18 | `avg_spread_all` | 714 | 2.3% | cross | `██████████░░░░░░░░░░░░░░░` |
| 19 | `spread` | 697 | 2.2% | spread | `█████████░░░░░░░░░░░░░░░░` |
| 20 | `vol_zscore` | 697 | 2.2% | burst | `█████████░░░░░░░░░░░░░░░░` |

### Category Summary (GBM)

| Category | Total Gain % |
|---|---|
| momentum | 17.9% |
| flow | 14.4% |
| cross | 13.3% |
| anchor | 12.0% |
| time | 12.0% |
| resolution | 10.4% |
| spread | 7.6% |
| volatility | 7.4% |
| burst | 2.7% |
| metar | 2.4% |

## Cross-Ticker Signal Analysis

- Mean `sum_mid_all` (test set): **0.9762** (expect ≈1.0; deviation = arbitrage / liquidity imbalance)
- Std `sum_mid_all`: **0.1581**
- Mean `prob_sum_deviation`: **-0.0238**
- `rel_mid` rank in GBM: **#14** of 48
- `prob_sum_deviation` rank in GBM: **#30** of 48

Model 2 (+ cross features) vs Model 1 MAE delta: **-0.00006** (degradation)

## Per-Ticker MAE (Test Set)

Tail bins (T-prefix) often have wider spreads and sparser trading.

| Ticker | MAE | Rows | Mean Ask | Mean Spread |
|---|---|---|---|---|
| `KXHIGHLAX-25SEP02-B81.5` | 0.12787 | 1,055 | 0.3953 | 0.0766 |
| `KXHIGHLAX-25SEP19-T78` | 0.12714 | 1,017 | 0.3913 | 0.0626 |
| `KXHIGHLAX-25SEP21-B77.5` | 0.12110 | 920 | 0.3364 | 0.0650 |
| `KXHIGHLAX-25SEP08-T77` | 0.12047 | 1,163 | 0.3842 | 0.0670 |
| `KXHIGHLAX-25SEP19-B77.5` | 0.11692 | 889 | 0.5116 | 0.0682 |
| `KXHIGHLAX-25SEP27-B71.5` | 0.11625 | 1,014 | 0.2123 | 0.0303 |
| `KXHIGHLAX-25SEP20-B76.5` | 0.11466 | 876 | 0.5193 | 0.1058 |
| `KXHIGHLAX-25SEP10-B76.5` | 0.11352 | 1,174 | 0.4243 | 0.0687 |
| `KXHIGHLAX-25SEP18-B80.5` | 0.11212 | 809 | 0.2692 | 0.0811 |
| `KXHIGHLAX-25SEP20-B74.5` | 0.10682 | 820 | 0.2776 | 0.0742 |
| `KXHIGHLAX-25SEP23-B80.5` | 0.10652 | 878 | 0.3022 | 0.0518 |
| `KXHIGHLAX-25SEP18-B78.5` | 0.10569 | 794 | 0.4615 | 0.0931 |
| `KXHIGHLAX-25SEP29-B74.5` | 0.10491 | 1,059 | 0.4766 | 0.0590 |
| `KXHIGHLAX-25SEP17-B80.5` | 0.10312 | 1,043 | 0.3264 | 0.0572 |
| `KXHIGHLAX-25SEP25-B72.5` | 0.10170 | 1,041 | 0.3972 | 0.0430 |
| `KXHIGHLAX-25SEP10-B74.5` | 0.09988 | 1,132 | 0.5153 | 0.0508 |
| `KXHIGHLAX-25SEP06-B78.5` | 0.09924 | 913 | 0.3970 | 0.0419 |
| `KXHIGHLAX-25SEP25-B70.5` | 0.09921 | 883 | 0.1772 | 0.0374 |
| `KXHIGHLAX-25SEP17-B78.5` | 0.09818 | 834 | 0.3567 | 0.0663 |
| `KXHIGHLAX-25SEP26-B74.5` | 0.09581 | 1,006 | 0.4275 | 0.0311 |
| `KXHIGHLAX-25SEP17-T78` | 0.09351 | 760 | 0.4753 | 0.0624 |
| `KXHIGHLAX-25SEP26-B72.5` | 0.09133 | 885 | 0.3490 | 0.0393 |
| `KXHIGHLAX-25SEP11-B73.5` | 0.09073 | 975 | 0.3790 | 0.0538 |
| `KXHIGHLAX-25SEP29-B72.5` | 0.08834 | 1,056 | 0.5406 | 0.0595 |
| `KXHIGHLAX-25SEP23-B76.5` | 0.08818 | 790 | 0.4108 | 0.0317 |
| `KXHIGHLAX-25SEP25-B74.5` | 0.08807 | 995 | 0.3352 | 0.0398 |
| `KXHIGHLAX-25SEP08-B79.5` | 0.08696 | 1,158 | 0.2729 | 0.0551 |
| `KXHIGHLAX-25SEP24-B78.5` | 0.08636 | 1,170 | 0.2915 | 0.0418 |
| `KXHIGHLAX-25SEP23-B78.5` | 0.08545 | 725 | 0.3799 | 0.0292 |
| `KXHIGHLAX-25SEP05-B77.5` | 0.08341 | 1,075 | 0.2618 | 0.0398 |
| `KXHIGHLAX-25SEP24-B76.5` | 0.08333 | 1,107 | 0.4927 | 0.0362 |
| `KXHIGHLAX-25SEP03-T82` | 0.08331 | 987 | 0.5437 | 0.0478 |
| `KXHIGHLAX-25SEP11-B75.5` | 0.08199 | 1,059 | 0.5442 | 0.0417 |
| `KXHIGHLAX-25SEP20-T79` | 0.08038 | 910 | 0.1982 | 0.0811 |
| `KXHIGHLAX-25SEP06-B76.5` | 0.08036 | 779 | 0.5484 | 0.0350 |
| `KXHIGHLAX-25SEP27-B73.5` | 0.08005 | 1,160 | 0.5098 | 0.0256 |
| `KXHIGHLAX-25SEP07-B81.5` | 0.07992 | 988 | 0.2627 | 0.0504 |
| `KXHIGHLAX-25SEP28-B74.5` | 0.07989 | 1,208 | 0.6074 | 0.0634 |
| `KXHIGHLAX-25SEP21-B73.5` | 0.07952 | 633 | 0.1477 | 0.0356 |
| `KXHIGHLAX-25SEP14-B74.5` | 0.07875 | 885 | 0.3025 | 0.0454 |
| `KXHIGHLAX-25SEP21-B75.5` | 0.07790 | 889 | 0.5485 | 0.0568 |
| `KXHIGHLAX-25SEP15-B76.5` | 0.07789 | 1,179 | 0.5749 | 0.0483 |
| `KXHIGHLAX-25SEP18-B76.5` | 0.07754 | 621 | 0.2430 | 0.0715 |
| `KXHIGHLAX-25SEP08-B77.5` | 0.07660 | 1,163 | 0.4126 | 0.0731 |
| `KXHIGHLAX-25SEP13-B73.5` | 0.07615 | 962 | 0.3409 | 0.0421 |
| `KXHIGHLAX-25SEP15-B78.5` | 0.07584 | 1,216 | 0.3016 | 0.0529 |
| `KXHIGHLAX-25SEP28-B76.5` | 0.07573 | 1,045 | 0.2421 | 0.0601 |
| `KXHIGHLAX-25SEP04-B78.5` | 0.07319 | 755 | 0.3581 | 0.0493 |
| `KXHIGHLAX-25SEP30-T74` | 0.07272 | 1,107 | 0.2136 | 0.0298 |
| `KXHIGHLAX-25SEP22-B78.5` | 0.07244 | 983 | 0.3155 | 0.0439 |
| `KXHIGHLAX-25SEP15-B74.5` | 0.07236 | 930 | 0.1457 | 0.0412 |
| `KXHIGHLAX-25SEP16-B79.5` | 0.07224 | 978 | 0.5651 | 0.0502 |
| `KXHIGHLAX-25SEP06-T79` | 0.07067 | 704 | 0.1069 | 0.0396 |
| `KXHIGHLAX-25SEP05-B75.5` | 0.06933 | 1,086 | 0.6279 | 0.0386 |
| `KXHIGHLAX-25SEP27-B75.5` | 0.06923 | 979 | 0.2573 | 0.0289 |
| `KXHIGHLAX-25SEP01-T80` | 0.06830 | 1,017 | 0.2745 | 0.0475 |
| `KXHIGHLAX-25SEP16-B81.5` | 0.06767 | 1,005 | 0.2362 | 0.0467 |
| `KXHIGHLAX-25SEP01-B80.5` | 0.06722 | 977 | 0.4673 | 0.0369 |
| `KXHIGHLAX-25SEP07-B79.5` | 0.06652 | 1,041 | 0.5444 | 0.0429 |
| `KXHIGHLAX-25SEP24-B74.5` | 0.06553 | 771 | 0.2168 | 0.0439 |
| `KXHIGHLAX-25SEP16-B77.5` | 0.06482 | 755 | 0.1899 | 0.0384 |
| `KXHIGHLAX-25SEP13-B75.5` | 0.06441 | 1,318 | 0.5899 | 0.0508 |
| `KXHIGHLAX-25SEP07-B77.5` | 0.06408 | 869 | 0.2193 | 0.0375 |
| `KXHIGHLAX-25SEP04-B76.5` | 0.06385 | 727 | 0.5996 | 0.0345 |
| `KXHIGHLAX-25SEP30-B73.5` | 0.06367 | 1,296 | 0.5849 | 0.0339 |
| `KXHIGHLAX-25SEP14-B76.5` | 0.06317 | 1,190 | 0.5662 | 0.0536 |
| `KXHIGHLAX-25SEP20-B78.5` | 0.06263 | 893 | 0.2618 | 0.0816 |
| `KXHIGHLAX-25SEP25-B76.5` | 0.06130 | 893 | 0.1721 | 0.0291 |
| `KXHIGHLAX-25SEP30-B71.5` | 0.06092 | 1,118 | 0.2698 | 0.0305 |
| `KXHIGHLAX-25SEP22-B76.5` | 0.06080 | 1,089 | 0.5197 | 0.0359 |
| `KXHIGHLAX-25SEP18-B82.5` | 0.06015 | 642 | 0.1760 | 0.0880 |
| `KXHIGHLAX-25SEP28-B72.5` | 0.05959 | 1,073 | 0.2673 | 0.0641 |
| `KXHIGHLAX-25SEP29-B70.5` | 0.05687 | 946 | 0.1294 | 0.0640 |
| `KXHIGHLAX-25SEP09-B75.5` | 0.05682 | 1,011 | 0.6126 | 0.0519 |
| `KXHIGHLAX-25SEP03-B82.5` | 0.05682 | 936 | 0.4056 | 0.0517 |
| `KXHIGHLAX-25SEP02-B83.5` | 0.05619 | 938 | 0.1765 | 0.0435 |
| `KXHIGHLAX-25SEP26-B76.5` | 0.05584 | 919 | 0.2332 | 0.0358 |
| `KXHIGHLAX-25SEP12-B76.5` | 0.05552 | 1,326 | 0.5948 | 0.0562 |
| `KXHIGHLAX-25SEP09-B73.5` | 0.05453 | 1,099 | 0.2599 | 0.0305 |
| `KXHIGHLAX-25SEP29-T75` | 0.05450 | 1,094 | 0.1005 | 0.0497 |
| `KXHIGHLAX-25SEP01-B82.5` | 0.05338 | 1,066 | 0.3156 | 0.0648 |
| `KXHIGHLAX-25SEP19-B75.5` | 0.05205 | 969 | 0.1393 | 0.0278 |
| `KXHIGHLAX-25SEP09-B77.5` | 0.05151 | 964 | 0.1567 | 0.0474 |
| `KXHIGHLAX-25SEP12-T77` | 0.04973 | 1,320 | 0.1576 | 0.0417 |
| `KXHIGHLAX-25SEP06-B74.5` | 0.04907 | 693 | 0.2118 | 0.0329 |
| `KXHIGHLAX-25SEP17-B82.5` | 0.04645 | 742 | 0.0977 | 0.0521 |
| `KXHIGHLAX-25SEP10-B72.5` | 0.04467 | 972 | 0.1304 | 0.0354 |
| `KXHIGHLAX-25SEP16-B83.5` | 0.04265 | 791 | 0.0957 | 0.0487 |
| `KXHIGHLAX-25SEP12-B74.5` | 0.04160 | 1,065 | 0.3419 | 0.0355 |
| `KXHIGHLAX-25SEP22-B74.5` | 0.04143 | 868 | 0.1838 | 0.0471 |
| `KXHIGHLAX-25SEP23-B74.5` | 0.04109 | 385 | 0.1103 | 0.0354 |
| `KXHIGHLAX-25SEP08-B81.5` | 0.03789 | 970 | 0.1004 | 0.0455 |
| `KXHIGHLAX-25SEP03-B84.5` | 0.03704 | 810 | 0.1377 | 0.0441 |
| `KXHIGHLAX-25SEP02-B85.5` | 0.03592 | 853 | 0.1265 | 0.0432 |
| `KXHIGHLAX-25SEP27-T76` | 0.03259 | 910 | 0.0762 | 0.0304 |
| `KXHIGHLAX-25SEP04-T76` | 0.03112 | 576 | 0.0778 | 0.0393 |
| `KXHIGHLAX-25SEP15-B80.5` | 0.03097 | 668 | 0.0790 | 0.0542 |
| `KXHIGHLAX-25SEP14-B78.5` | 0.02991 | 798 | 0.1956 | 0.0318 |
| `KXHIGHLAX-25SEP13-B77.5` | 0.02973 | 921 | 0.1339 | 0.0393 |
| `KXHIGHLAX-25SEP21-B79.5` | 0.02959 | 472 | 0.0852 | 0.0352 |
| `KXHIGHLAX-25SEP11-B71.5` | 0.02887 | 583 | 0.0579 | 0.0333 |
| `KXHIGHLAX-25SEP01-B84.5` | 0.02887 | 972 | 0.0884 | 0.0390 |
| `KXHIGHLAX-25SEP11-T76` | 0.02873 | 778 | 0.1292 | 0.0511 |
| `KXHIGHLAX-25SEP26-T72` | 0.02758 | 1,051 | 0.0573 | 0.0187 |
| `KXHIGHLAX-25SEP10-T77` | 0.02730 | 877 | 0.0689 | 0.0441 |
| `KXHIGHLAX-25SEP05-B73.5` | 0.02610 | 1,142 | 0.1190 | 0.0253 |
| `KXHIGHLAX-25SEP26-B78.5` | 0.02538 | 735 | 0.0546 | 0.0226 |
| `KXHIGHLAX-25SEP24-T79` | 0.02537 | 1,099 | 0.1045 | 0.0357 |
| `KXHIGHLAX-25SEP13-B71.5` | 0.02432 | 596 | 0.0489 | 0.0362 |
| `KXHIGHLAX-25SEP07-B83.5` | 0.02282 | 714 | 0.0618 | 0.0295 |
| `KXHIGHLAX-25SEP05-B79.5` | 0.02263 | 885 | 0.0612 | 0.0270 |
| `KXHIGHLAX-25SEP19-B73.5` | 0.02230 | 331 | 0.0581 | 0.0265 |
| `KXHIGHLAX-25SEP28-B70.5` | 0.02179 | 516 | 0.0501 | 0.0349 |
| `KXHIGHLAX-25SEP22-B80.5` | 0.02033 | 645 | 0.0668 | 0.0344 |
| `KXHIGHLAX-25SEP16-T77` | 0.02006 | 383 | 0.0418 | 0.0216 |
| `KXHIGHLAX-25SEP30-B69.5` | 0.02005 | 752 | 0.0527 | 0.0272 |
| `KXHIGHLAX-25SEP28-T77` | 0.02004 | 673 | 0.0482 | 0.0277 |
| `KXHIGHLAX-25SEP04-B80.5` | 0.01913 | 739 | 0.0652 | 0.0181 |
| `KXHIGHLAX-25SEP13-T78` | 0.01882 | 1,127 | 0.0425 | 0.0241 |
| `KXHIGHLAX-25SEP17-B84.5` | 0.01874 | 545 | 0.0506 | 0.0399 |
| `KXHIGHLAX-25SEP24-B72.5` | 0.01819 | 397 | 0.0320 | 0.0241 |
| `KXHIGHLAX-25SEP12-B72.5` | 0.01627 | 774 | 0.0563 | 0.0261 |
| `KXHIGHLAX-25SEP21-T73` | 0.01608 | 467 | 0.0233 | 0.0225 |
| `KXHIGHLAX-25SEP16-T84` | 0.01591 | 678 | 0.0444 | 0.0249 |
| `KXHIGHLAX-25SEP07-T84` | 0.01590 | 633 | 0.0360 | 0.0311 |
| `KXHIGHLAX-25SEP09-T78` | 0.01491 | 744 | 0.0413 | 0.0166 |
| `KXHIGHLAX-25SEP07-T77` | 0.01456 | 698 | 0.0457 | 0.0241 |
| `KXHIGHLAX-25SEP25-T77` | 0.01434 | 874 | 0.0406 | 0.0276 |
| `KXHIGHLAX-25SEP18-T76` | 0.01364 | 205 | 0.0331 | 0.0176 |
| `KXHIGHLAX-25SEP21-T80` | 0.01347 | 355 | 0.0286 | 0.0240 |
| `KXHIGHLAX-25SEP09-B71.5` | 0.01241 | 404 | 0.0493 | 0.0292 |
| `KXHIGHLAX-25SEP04-B82.5` | 0.01165 | 401 | 0.0283 | 0.0253 |
| `KXHIGHLAX-25SEP27-B69.5` | 0.01158 | 569 | 0.0297 | 0.0204 |
| `KXHIGHLAX-25SEP14-T79` | 0.01145 | 1,231 | 0.0400 | 0.0267 |
| `KXHIGHLAX-25SEP25-T70` | 0.01080 | 459 | 0.0283 | 0.0278 |
| `KXHIGHLAX-25SEP22-T74` | 0.01062 | 160 | 0.0161 | 0.0149 |
| `KXHIGHLAX-25SEP14-T72` | 0.01023 | 97 | 0.0275 | 0.0275 |
| `KXHIGHLAX-25SEP10-T70` | 0.00993 | 637 | 0.0230 | 0.0230 |
| `KXHIGHLAX-25SEP05-T80` | 0.00978 | 597 | 0.0277 | 0.0224 |
| `KXHIGHLAX-25SEP14-B72.5` | 0.00857 | 395 | 0.0362 | 0.0266 |
| `KXHIGHLAX-25SEP19-B71.5` | 0.00787 | 147 | 0.0301 | 0.0300 |
| `KXHIGHLAX-25SEP20-B72.5` | 0.00776 | 183 | 0.0333 | 0.0333 |
| `KXHIGHLAX-25SEP26-T79` | 0.00756 | 532 | 0.0289 | 0.0189 |
| `KXHIGHLAX-25SEP10-B70.5` | 0.00699 | 615 | 0.0180 | 0.0180 |
| `KXHIGHLAX-25SEP08-B83.5` | 0.00693 | 858 | 0.0296 | 0.0291 |
| `KXHIGHLAX-25SEP20-T72` | 0.00671 | 147 | 0.0316 | 0.0316 |
| `KXHIGHLAX-25SEP29-B68.5` | 0.00647 | 360 | 0.0264 | 0.0236 |
| `KXHIGHLAX-25SEP08-T84` | 0.00622 | 744 | 0.0280 | 0.0280 |
| `KXHIGHLAX-25SEP15-T74` | 0.00619 | 367 | 0.0346 | 0.0301 |
| `KXHIGHLAX-25SEP06-B72.5` | 0.00609 | 480 | 0.0330 | 0.0259 |
| `KXHIGHLAX-25SEP12-B70.5` | 0.00578 | 141 | 0.0228 | 0.0228 |
| `KXHIGHLAX-25SEP05-T73` | 0.00551 | 720 | 0.0291 | 0.0222 |
| `KXHIGHLAX-25SEP09-T71` | 0.00516 | 210 | 0.0147 | 0.0147 |
| `KXHIGHLAX-25SEP13-T71` | 0.00508 | 227 | 0.0124 | 0.0124 |
| `KXHIGHLAX-25SEP06-T72` | 0.00398 | 216 | 0.0131 | 0.0131 |
| `KXHIGHLAX-25SEP28-T70` | 0.00355 | 402 | 0.0138 | 0.0138 |

## Key Findings

1. **Top linear predictors**: `bar_hour`, `yes_ask_close`, `mid_close` — features ranked by normalized coefficient weight after scaling. With a relative (move) target, price-level features carry less weight than momentum/flow signals.
2. **Momentum features provide modest lift**: Model 1 (momentum) reduces MAE by 0.00063 vs baseline, suggesting lag returns and spread carry incremental signal beyond the current price level alone.
3. **Cross-ticker signals**: Adding probability conservation features shifts MAE by -0.00006. The sum-of-mids deviation from 1.0 reflects cross-bin arbitrage opportunities that briefly predict price moves.
4. **Time-of-day**: Adding time features shifts MAE by -0.00041. The cyclical encoding captures METAR observation windows (fires at :53 past each hour) when informed traders update positions.
5. **Non-linearity**: LightGBM (MAE=0.05969) vs best linear model (MAE=0.06018) — delta=+0.00049. GBM captures meaningful non-linear interactions.

## Limitations

- Only 11.5% of bars have trade OHLC (`price_*`); models rely primarily on quote OHLC
- Same-day bars only — no inter-day momentum or regime features
- 122 days of training data; seasonal effects within summer may be underweighted
- `price_previous` (2.1% null) imputed with median — may understate signal in thin markets
- Tail bins (T-prefix) have wider spreads; a per-bin model may outperform a pooled one
- The T+30 target assumes a bar exists 30 minutes later (dropped ≈20% of bars near day-end)
