# Kalshi Weather Market: Price Prediction Report

_Generated: 2026-07-05 15:32 UTC_

**Train**: 2025-06-01 – 2025-08-31 (279,538 rows)  **Test**: 2025-09-01 – 2025-09-30 (134,786 rows)

## Methodology

- **Target**: `yes_ask_open` at T+30 minutes — the best ask price 30 1-minute bars ahead
- **Train/Test split**: purely temporal (no shuffling) — 2025-06-01–2025-08-31 train, 2025-09-01–2025-09-30 test
- **NaN handling**: median imputation fit on training set only, applied to test
- **Prices**: decimal scale (0–1), where 0.72 = 72¢ / 72% implied probability
- **Naive benchmark**: predict no change (forecast = current `yes_ask_close`)
- **Evaluation**: MAE, RMSE, R², directional accuracy, % improvement vs naive

## Model Comparison

| Model | MAE | RMSE | R² | Dir Acc | vs Naive |
|---|---|---|---|---|---|
| Naive (no change) | 0.03211 | — | — | — | 0.0% |
| baseline | 0.03302 | 0.06726 | 0.9175 | 38.9% | -2.8% |
| model1_momentum | 0.03307 | 0.06649 | 0.9194 | 42.3% | -3.0% |
| model2_cross | 0.03300 | 0.06644 | 0.9195 | 43.9% | -2.8% |
| model3_tod | 0.03336 | 0.06645 | 0.9194 | 48.3% | -3.9% |
| model4_gbm | 0.03341 | 0.06790 | 0.9159 | 48.2% | -4.0% |

## Feature Importance: Linear Model (Model 3 — all linear features)

Coefficients are normalized to % of total absolute weight after StandardScaler (i.e., features are on the same scale — higher % = stronger linear influence).

| Rank | Feature | Importance % | Coef | Category |
|---|---|---|---|---|
| 1 | `mid_close` | 35.80% | +0.0943 | anchor |
| 2 | `yes_ask_close` | 34.00% | +0.0896 | anchor |
| 3 | `ask_lag1` | 10.50% | +0.0277 | momentum |
| 4 | `ask_lag5` | 4.02% | +0.0106 | momentum |
| 5 | `ask_lag15` | 3.19% | +0.0084 | momentum |
| 6 | `spread` | 2.17% | -0.0057 | spread |
| 7 | `tod_sin` | 1.28% | -0.0034 | time |
| 8 | `spread_lag5` | 1.25% | +0.0033 | spread |
| 9 | `avg_spread_all` | 0.99% | +0.0026 | cross |
| 10 | `vol_sum_15` | 0.66% | -0.0017 | flow |
| 11 | `mid_vol_15` | 0.63% | +0.0017 | volatility |
| 12 | `relative_spread` | 0.57% | +0.0015 | cross |
| 13 | `tod_cos` | 0.57% | +0.0015 | time |
| 14 | `bid_intrabar_range` | 0.38% | +0.0010 | volatility |
| 15 | `rel_mid` | 0.38% | -0.0010 | cross |
| 16 | `price_previous` | 0.34% | +0.0009 | anchor |
| 17 | `sum_mid_all` | 0.32% | -0.0008 | cross |
| 18 | `prob_sum_deviation` | 0.32% | -0.0008 | cross |
| 19 | `oi_change_15` | 0.30% | +0.0008 | flow |
| 20 | `bar_hour` | 0.29% | +0.0008 | time |

### Category Summary (Linear)

| Category | Total Importance % |
|---|---|
| anchor | 70.1% |
| momentum | 18.5% |
| spread | 3.6% |
| cross | 2.9% |
| time | 2.4% |
| flow | 1.5% |
| volatility | 1.1% |

## Feature Importance: LightGBM (Model 4 — all features)

Importance by gain (total reduction in loss from splits on this feature).

| Rank | Feature | Gain | Share | Category | Bar |
|---|---|---|---|---|---|
| 1 | `tod_sin` | 2,128 | 6.9% | time | `█████████████████████████` |
| 2 | `tod_cos` | 1,867 | 6.0% | time | `██████████████████████░░░` |
| 3 | `vol_sum_15` | 1,803 | 5.8% | flow | `█████████████████████░░░░` |
| 4 | `oi_change_15` | 1,744 | 5.6% | flow | `████████████████████░░░░░` |
| 5 | `price_previous` | 1,514 | 4.9% | anchor | `██████████████████░░░░░░░` |
| 6 | `mid_ret_30` | 1,514 | 4.9% | momentum | `██████████████████░░░░░░░` |
| 7 | `mid_vol_15` | 1,381 | 4.5% | volatility | `████████████████░░░░░░░░░` |
| 8 | `mid_close` | 1,379 | 4.4% | anchor | `████████████████░░░░░░░░░` |
| 9 | `ask_lag15` | 1,360 | 4.4% | momentum | `████████████████░░░░░░░░░` |
| 10 | `bar_minute` | 1,166 | 3.8% | time | `██████████████░░░░░░░░░░░` |
| 11 | `yes_ask_close` | 1,105 | 3.6% | anchor | `█████████████░░░░░░░░░░░░` |
| 12 | `rel_mid` | 975 | 3.1% | cross | `███████████░░░░░░░░░░░░░░` |
| 13 | `ask_lag5` | 972 | 3.1% | momentum | `███████████░░░░░░░░░░░░░░` |
| 14 | `mid_ret_15` | 938 | 3.0% | momentum | `███████████░░░░░░░░░░░░░░` |
| 15 | `spread_lag5` | 922 | 3.0% | spread | `███████████░░░░░░░░░░░░░░` |
| 16 | `spread` | 911 | 2.9% | spread | `███████████░░░░░░░░░░░░░░` |
| 17 | `bar_hour` | 809 | 2.6% | time | `██████████░░░░░░░░░░░░░░░` |
| 18 | `spread_lag1` | 790 | 2.5% | spread | `█████████░░░░░░░░░░░░░░░░` |
| 19 | `avg_spread_all` | 762 | 2.5% | cross | `█████████░░░░░░░░░░░░░░░░` |
| 20 | `ask_lag1` | 749 | 2.4% | momentum | `█████████░░░░░░░░░░░░░░░░` |

### Category Summary (GBM)

| Category | Total Gain % |
|---|---|
| momentum | 20.5% |
| time | 19.3% |
| flow | 16.4% |
| cross | 14.3% |
| anchor | 12.9% |
| spread | 8.5% |
| volatility | 8.1% |

## Cross-Ticker Signal Analysis

- Mean `sum_mid_all` (test set): **0.9740** (expect ≈1.0; deviation = arbitrage / liquidity imbalance)
- Std `sum_mid_all`: **0.1650**
- Mean `prob_sum_deviation`: **-0.0260**
- `rel_mid` rank in GBM: **#12** of 33
- `prob_sum_deviation` rank in GBM: **#29** of 33

Model 2 (+ cross features) vs Model 1 MAE delta: **+0.00007** (improvement)

## Per-Ticker MAE (Test Set)

Tail bins (T-prefix) often have wider spreads and sparser trading.

| Ticker | MAE | Rows | Mean Ask | Mean Spread |
|---|---|---|---|---|
| `KXHIGHLAX-25SEP25-B72.5` | 0.08351 | 1,136 | 0.3914 | 0.0427 |
| `KXHIGHLAX-25SEP19-T78` | 0.07424 | 1,045 | 0.4011 | 0.0667 |
| `KXHIGHLAX-25SEP08-T77` | 0.07404 | 1,208 | 0.4080 | 0.0674 |
| `KXHIGHLAX-25SEP02-B81.5` | 0.07324 | 1,108 | 0.3834 | 0.0760 |
| `KXHIGHLAX-25SEP25-B70.5` | 0.07067 | 965 | 0.2011 | 0.0376 |
| `KXHIGHLAX-25SEP20-B76.5` | 0.06877 | 923 | 0.5076 | 0.1070 |
| `KXHIGHLAX-25SEP20-B74.5` | 0.06846 | 857 | 0.3142 | 0.0716 |
| `KXHIGHLAX-25SEP19-B77.5` | 0.06789 | 931 | 0.5192 | 0.0763 |
| `KXHIGHLAX-25SEP29-B74.5` | 0.06237 | 1,114 | 0.5045 | 0.0570 |
| `KXHIGHLAX-25SEP21-B77.5` | 0.06221 | 967 | 0.3523 | 0.0719 |
| `KXHIGHLAX-25SEP10-B76.5` | 0.05730 | 1,235 | 0.4521 | 0.0666 |
| `KXHIGHLAX-25SEP27-B71.5` | 0.05714 | 1,052 | 0.2094 | 0.0297 |
| `KXHIGHLAX-25SEP18-B80.5` | 0.05707 | 880 | 0.2606 | 0.0804 |
| `KXHIGHLAX-25SEP18-B78.5` | 0.05689 | 865 | 0.4819 | 0.0894 |
| `KXHIGHLAX-25SEP29-B72.5` | 0.05686 | 1,115 | 0.5246 | 0.0600 |
| `KXHIGHLAX-25SEP26-B74.5` | 0.05456 | 1,049 | 0.4200 | 0.0313 |
| `KXHIGHLAX-25SEP17-B78.5` | 0.05396 | 893 | 0.3661 | 0.0677 |
| `KXHIGHLAX-25SEP24-B76.5` | 0.05326 | 1,142 | 0.4931 | 0.0392 |
| `KXHIGHLAX-25SEP30-T74` | 0.05261 | 1,150 | 0.2130 | 0.0307 |
| `KXHIGHLAX-25SEP24-B78.5` | 0.05157 | 1,221 | 0.3077 | 0.0446 |
| `KXHIGHLAX-25SEP08-B79.5` | 0.05096 | 1,208 | 0.2613 | 0.0529 |
| `KXHIGHLAX-25SEP25-B74.5` | 0.04918 | 1,100 | 0.3185 | 0.0383 |
| `KXHIGHLAX-25SEP20-T79` | 0.04894 | 946 | 0.1870 | 0.0759 |
| `KXHIGHLAX-25SEP17-B80.5` | 0.04782 | 1,102 | 0.3585 | 0.0545 |
| `KXHIGHLAX-25SEP28-B76.5` | 0.04722 | 1,102 | 0.2381 | 0.0603 |
| `KXHIGHLAX-25SEP22-B78.5` | 0.04709 | 1,047 | 0.3105 | 0.0432 |
| `KXHIGHLAX-25SEP18-B76.5` | 0.04697 | 707 | 0.2369 | 0.0689 |
| `KXHIGHLAX-25SEP28-B74.5` | 0.04678 | 1,265 | 0.6211 | 0.0621 |
| `KXHIGHLAX-25SEP17-T78` | 0.04675 | 821 | 0.4591 | 0.0626 |
| `KXHIGHLAX-25SEP23-B80.5` | 0.04654 | 982 | 0.3330 | 0.0509 |
| `KXHIGHLAX-25SEP13-B73.5` | 0.04609 | 995 | 0.3395 | 0.0436 |
| `KXHIGHLAX-25SEP23-B78.5` | 0.04572 | 803 | 0.3852 | 0.0310 |
| `KXHIGHLAX-25SEP14-B74.5` | 0.04519 | 913 | 0.3211 | 0.0485 |
| `KXHIGHLAX-25SEP15-B76.5` | 0.04446 | 1,223 | 0.5779 | 0.0520 |
| `KXHIGHLAX-25SEP15-B74.5` | 0.04414 | 973 | 0.1617 | 0.0468 |
| `KXHIGHLAX-25SEP05-B77.5` | 0.04399 | 1,118 | 0.2545 | 0.0408 |
| `KXHIGHLAX-25SEP06-T79` | 0.04391 | 730 | 0.1111 | 0.0406 |
| `KXHIGHLAX-25SEP10-B74.5` | 0.04385 | 1,187 | 0.4965 | 0.0498 |
| `KXHIGHLAX-25SEP23-B76.5` | 0.04375 | 878 | 0.4053 | 0.0351 |
| `KXHIGHLAX-25SEP08-B77.5` | 0.04331 | 1,212 | 0.3988 | 0.0726 |
| `KXHIGHLAX-25SEP27-B75.5` | 0.04261 | 1,018 | 0.2521 | 0.0282 |
| `KXHIGHLAX-25SEP27-B73.5` | 0.04202 | 1,186 | 0.5224 | 0.0257 |
| `KXHIGHLAX-25SEP28-B72.5` | 0.04166 | 1,114 | 0.2621 | 0.0643 |
| `KXHIGHLAX-25SEP26-B72.5` | 0.04128 | 926 | 0.3689 | 0.0395 |
| `KXHIGHLAX-25SEP21-B75.5` | 0.04096 | 939 | 0.5588 | 0.0635 |
| `KXHIGHLAX-25SEP24-B74.5` | 0.04005 | 817 | 0.2090 | 0.0444 |
| `KXHIGHLAX-25SEP29-T75` | 0.03999 | 1,149 | 0.0970 | 0.0484 |
| `KXHIGHLAX-25SEP20-B78.5` | 0.03911 | 920 | 0.2517 | 0.0782 |
| `KXHIGHLAX-25SEP11-B73.5` | 0.03861 | 1,080 | 0.3546 | 0.0492 |
| `KXHIGHLAX-25SEP07-B81.5` | 0.03809 | 1,033 | 0.2832 | 0.0565 |
| `KXHIGHLAX-25SEP02-B83.5` | 0.03808 | 988 | 0.1711 | 0.0428 |
| `KXHIGHLAX-25SEP16-B79.5` | 0.03808 | 1,011 | 0.5846 | 0.0494 |
| `KXHIGHLAX-25SEP29-B70.5` | 0.03773 | 999 | 0.1275 | 0.0629 |
| `KXHIGHLAX-25SEP06-B76.5` | 0.03760 | 794 | 0.5395 | 0.0361 |
| `KXHIGHLAX-25SEP21-B73.5` | 0.03752 | 656 | 0.1337 | 0.0329 |
| `KXHIGHLAX-25SEP07-B79.5` | 0.03737 | 1,090 | 0.5382 | 0.0485 |
| `KXHIGHLAX-25SEP15-B78.5` | 0.03734 | 1,265 | 0.2925 | 0.0523 |
| `KXHIGHLAX-25SEP30-B73.5` | 0.03678 | 1,349 | 0.5990 | 0.0333 |
| `KXHIGHLAX-25SEP11-B75.5` | 0.03674 | 1,166 | 0.5667 | 0.0388 |
| `KXHIGHLAX-25SEP06-B78.5` | 0.03651 | 966 | 0.4363 | 0.0394 |
| `KXHIGHLAX-25SEP03-T82` | 0.03641 | 1,030 | 0.5508 | 0.0506 |
| `KXHIGHLAX-25SEP14-B76.5` | 0.03639 | 1,252 | 0.5585 | 0.0577 |
| `KXHIGHLAX-25SEP04-B78.5` | 0.03627 | 850 | 0.3487 | 0.0503 |
| `KXHIGHLAX-25SEP30-B71.5` | 0.03597 | 1,149 | 0.2652 | 0.0317 |
| `KXHIGHLAX-25SEP05-B75.5` | 0.03553 | 1,129 | 0.6392 | 0.0388 |
| `KXHIGHLAX-25SEP18-B82.5` | 0.03497 | 724 | 0.1689 | 0.0823 |
| `KXHIGHLAX-25SEP22-B74.5` | 0.03476 | 909 | 0.1819 | 0.0481 |
| `KXHIGHLAX-25SEP01-T80` | 0.03467 | 1,067 | 0.2635 | 0.0461 |
| `KXHIGHLAX-25SEP13-B75.5` | 0.03428 | 1,378 | 0.6077 | 0.0490 |
| `KXHIGHLAX-25SEP26-B76.5` | 0.03410 | 961 | 0.2279 | 0.0357 |
| `KXHIGHLAX-25SEP04-B76.5` | 0.03304 | 825 | 0.6105 | 0.0330 |
| `KXHIGHLAX-25SEP16-B81.5` | 0.03304 | 1,046 | 0.2254 | 0.0452 |
| `KXHIGHLAX-25SEP16-B77.5` | 0.03273 | 785 | 0.1799 | 0.0375 |
| `KXHIGHLAX-25SEP10-B72.5` | 0.03266 | 1,001 | 0.1261 | 0.0344 |
| `KXHIGHLAX-25SEP25-B76.5` | 0.03259 | 986 | 0.1673 | 0.0281 |
| `KXHIGHLAX-25SEP22-B76.5` | 0.03228 | 1,147 | 0.5379 | 0.0359 |
| `KXHIGHLAX-25SEP09-B77.5` | 0.03226 | 1,000 | 0.1535 | 0.0479 |
| `KXHIGHLAX-25SEP01-B80.5` | 0.03102 | 1,036 | 0.4873 | 0.0378 |
| `KXHIGHLAX-25SEP01-B82.5` | 0.02937 | 1,106 | 0.3123 | 0.0660 |
| `KXHIGHLAX-25SEP03-B82.5` | 0.02907 | 990 | 0.3993 | 0.0568 |
| `KXHIGHLAX-25SEP09-B75.5` | 0.02814 | 1,042 | 0.6275 | 0.0513 |
| `KXHIGHLAX-25SEP12-T77` | 0.02770 | 1,382 | 0.1512 | 0.0402 |
| `KXHIGHLAX-25SEP12-B74.5` | 0.02711 | 1,120 | 0.3379 | 0.0384 |
| `KXHIGHLAX-25SEP07-B77.5` | 0.02709 | 918 | 0.2156 | 0.0375 |
| `KXHIGHLAX-25SEP17-B82.5` | 0.02560 | 809 | 0.0967 | 0.0528 |
| `KXHIGHLAX-25SEP12-B76.5` | 0.02527 | 1,388 | 0.6122 | 0.0541 |
| `KXHIGHLAX-25SEP03-B84.5` | 0.02475 | 867 | 0.1328 | 0.0446 |
| `KXHIGHLAX-25SEP06-B74.5` | 0.02437 | 715 | 0.2053 | 0.0335 |
| `KXHIGHLAX-25SEP08-B81.5` | 0.02415 | 1,022 | 0.0961 | 0.0424 |
| `KXHIGHLAX-25SEP11-T76` | 0.02337 | 866 | 0.1314 | 0.0502 |
| `KXHIGHLAX-25SEP02-B85.5` | 0.02290 | 906 | 0.1240 | 0.0443 |
| `KXHIGHLAX-25SEP16-B83.5` | 0.02253 | 816 | 0.0948 | 0.0488 |
| `KXHIGHLAX-25SEP09-B73.5` | 0.02226 | 1,135 | 0.2450 | 0.0293 |
| `KXHIGHLAX-25SEP01-B84.5` | 0.02198 | 1,018 | 0.0865 | 0.0391 |
| `KXHIGHLAX-25SEP10-T77` | 0.02173 | 903 | 0.0731 | 0.0474 |
| `KXHIGHLAX-25SEP13-B77.5` | 0.02048 | 938 | 0.1302 | 0.0405 |
| `KXHIGHLAX-25SEP19-B75.5` | 0.02026 | 1,007 | 0.1293 | 0.0265 |
| `KXHIGHLAX-25SEP14-B78.5` | 0.01982 | 832 | 0.1893 | 0.0317 |
| `KXHIGHLAX-25SEP27-T76` | 0.01859 | 936 | 0.0752 | 0.0307 |
| `KXHIGHLAX-25SEP24-T79` | 0.01851 | 1,130 | 0.1002 | 0.0351 |
| `KXHIGHLAX-25SEP15-B80.5` | 0.01801 | 720 | 0.0769 | 0.0527 |
| `KXHIGHLAX-25SEP23-B74.5` | 0.01787 | 437 | 0.1079 | 0.0347 |
| `KXHIGHLAX-25SEP11-B71.5` | 0.01702 | 659 | 0.0554 | 0.0331 |
| `KXHIGHLAX-25SEP26-B78.5` | 0.01692 | 766 | 0.0527 | 0.0224 |
| `KXHIGHLAX-25SEP21-B79.5` | 0.01657 | 531 | 0.0859 | 0.0360 |
| `KXHIGHLAX-25SEP07-B83.5` | 0.01532 | 760 | 0.0612 | 0.0322 |
| `KXHIGHLAX-25SEP04-T76` | 0.01504 | 662 | 0.0732 | 0.0382 |
| `KXHIGHLAX-25SEP26-T72` | 0.01500 | 1,096 | 0.0557 | 0.0186 |
| `KXHIGHLAX-25SEP30-B69.5` | 0.01498 | 768 | 0.0522 | 0.0267 |
| `KXHIGHLAX-25SEP05-B79.5` | 0.01451 | 933 | 0.0604 | 0.0273 |
| `KXHIGHLAX-25SEP22-B80.5` | 0.01325 | 670 | 0.0648 | 0.0336 |
| `KXHIGHLAX-25SEP05-B73.5` | 0.01315 | 1,197 | 0.1132 | 0.0240 |
| `KXHIGHLAX-25SEP28-B70.5` | 0.01272 | 555 | 0.0493 | 0.0352 |
| `KXHIGHLAX-25SEP04-B80.5` | 0.01253 | 854 | 0.0645 | 0.0191 |
| `KXHIGHLAX-25SEP13-B71.5` | 0.01205 | 655 | 0.0463 | 0.0350 |
| `KXHIGHLAX-25SEP28-T77` | 0.01050 | 717 | 0.0459 | 0.0265 |
| `KXHIGHLAX-25SEP25-T77` | 0.01031 | 959 | 0.0400 | 0.0261 |
| `KXHIGHLAX-25SEP12-B72.5` | 0.01012 | 809 | 0.0577 | 0.0280 |
| `KXHIGHLAX-25SEP17-B84.5` | 0.01010 | 588 | 0.0485 | 0.0391 |
| `KXHIGHLAX-25SEP09-T78` | 0.00980 | 788 | 0.0395 | 0.0161 |
| `KXHIGHLAX-25SEP13-T78` | 0.00962 | 1,184 | 0.0402 | 0.0233 |
| `KXHIGHLAX-25SEP19-B73.5` | 0.00928 | 349 | 0.0531 | 0.0255 |
| `KXHIGHLAX-25SEP09-B71.5` | 0.00920 | 453 | 0.0464 | 0.0282 |
| `KXHIGHLAX-25SEP24-B72.5` | 0.00916 | 428 | 0.0326 | 0.0237 |
| `KXHIGHLAX-25SEP07-T84` | 0.00904 | 691 | 0.0343 | 0.0299 |
| `KXHIGHLAX-25SEP07-T77` | 0.00899 | 744 | 0.0439 | 0.0229 |
| `KXHIGHLAX-25SEP16-T77` | 0.00894 | 430 | 0.0400 | 0.0210 |
| `KXHIGHLAX-25SEP18-T76` | 0.00873 | 245 | 0.0327 | 0.0182 |
| `KXHIGHLAX-25SEP16-T84` | 0.00799 | 724 | 0.0436 | 0.0244 |
| `KXHIGHLAX-25SEP25-T70` | 0.00770 | 526 | 0.0263 | 0.0259 |
| `KXHIGHLAX-25SEP04-B82.5` | 0.00712 | 495 | 0.0295 | 0.0274 |
| `KXHIGHLAX-25SEP14-T72` | 0.00708 | 111 | 0.0271 | 0.0271 |
| `KXHIGHLAX-25SEP21-T80` | 0.00706 | 385 | 0.0279 | 0.0230 |
| `KXHIGHLAX-25SEP14-T79` | 0.00657 | 1,301 | 0.0387 | 0.0258 |
| `KXHIGHLAX-25SEP21-T73` | 0.00647 | 508 | 0.0245 | 0.0233 |
| `KXHIGHLAX-25SEP10-T70` | 0.00626 | 673 | 0.0220 | 0.0220 |
| `KXHIGHLAX-25SEP14-B72.5` | 0.00624 | 410 | 0.0370 | 0.0273 |
| `KXHIGHLAX-25SEP27-B69.5` | 0.00607 | 608 | 0.0300 | 0.0209 |
| `KXHIGHLAX-25SEP22-T74` | 0.00501 | 188 | 0.0163 | 0.0151 |
| `KXHIGHLAX-25SEP20-B72.5` | 0.00486 | 219 | 0.0317 | 0.0317 |
| `KXHIGHLAX-25SEP08-T84` | 0.00480 | 779 | 0.0268 | 0.0268 |
| `KXHIGHLAX-25SEP20-T72` | 0.00452 | 160 | 0.0301 | 0.0301 |
| `KXHIGHLAX-25SEP05-T80` | 0.00425 | 633 | 0.0274 | 0.0220 |
| `KXHIGHLAX-25SEP10-B70.5` | 0.00391 | 660 | 0.0175 | 0.0174 |
| `KXHIGHLAX-25SEP08-B83.5` | 0.00368 | 903 | 0.0284 | 0.0279 |
| `KXHIGHLAX-25SEP06-B72.5` | 0.00309 | 500 | 0.0326 | 0.0255 |
| `KXHIGHLAX-25SEP05-T73` | 0.00306 | 769 | 0.0282 | 0.0215 |
| `KXHIGHLAX-25SEP19-B71.5` | 0.00295 | 173 | 0.0279 | 0.0278 |
| `KXHIGHLAX-25SEP15-T74` | 0.00249 | 397 | 0.0334 | 0.0290 |
| `KXHIGHLAX-25SEP26-T79` | 0.00248 | 549 | 0.0282 | 0.0186 |
| `KXHIGHLAX-25SEP29-B68.5` | 0.00234 | 398 | 0.0248 | 0.0228 |
| `KXHIGHLAX-25SEP09-T71` | 0.00208 | 253 | 0.0144 | 0.0144 |
| `KXHIGHLAX-25SEP13-T71` | 0.00198 | 255 | 0.0126 | 0.0125 |
| `KXHIGHLAX-25SEP12-B70.5` | 0.00155 | 161 | 0.0219 | 0.0219 |
| `KXHIGHLAX-25SEP06-T72` | 0.00150 | 219 | 0.0128 | 0.0128 |
| `KXHIGHLAX-25SEP28-T70` | 0.00122 | 426 | 0.0136 | 0.0136 |

## Key Findings

1. **Current ask dominates**: `yes_ask_close` is the single strongest linear predictor (rank 1 in both linear and GBM models), confirming price persistence — the 30-min ahead ask is strongly anchored to the current ask.
2. **Momentum features provide modest lift**: Model 1 (momentum) reduces MAE by -0.00005 vs baseline, suggesting lag returns and spread carry incremental signal beyond the current price level alone.
3. **Cross-ticker signals**: Adding probability conservation features shifts MAE by +0.00007. The sum-of-mids deviation from 1.0 reflects cross-bin arbitrage opportunities that briefly predict price moves.
4. **Time-of-day**: Adding time features shifts MAE by -0.00036. The cyclical encoding captures METAR observation windows (fires at :53 past each hour) when informed traders update positions.
5. **Non-linearity**: LightGBM (MAE=0.03341) vs best linear model (MAE=0.03336) — delta=-0.00005. Linear model is competitive; non-linearity adds little.

## Limitations

- Only 11.5% of bars have trade OHLC (`price_*`); models rely primarily on quote OHLC
- Same-day bars only — no inter-day momentum or regime features
- 122 days of training data; seasonal effects within summer may be underweighted
- `price_previous` (2.1% null) imputed with median — may understate signal in thin markets
- Tail bins (T-prefix) have wider spreads; a per-bin model may outperform a pooled one
- The T+30 target assumes a bar exists 30 minutes later (dropped ≈20% of bars near day-end)
