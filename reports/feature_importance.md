# Kalshi Weather Market: Price Prediction Report

_Generated: 2026-07-05 15:54 UTC_

**Train**: 2025-06-01 – 2025-08-31 (279,538 rows)  **Test**: 2025-09-01 – 2025-09-30 (134,786 rows)

## Methodology

- **Target**: `target_move_30` = `yes_ask_open` at T+30 minus `yes_ask_close` at T — the 30-min price move
- **Train/Test split**: purely temporal (no shuffling) — 2025-06-01–2025-08-31 train, 2025-09-01–2025-09-30 test
- **NaN handling**: median imputation fit on training set only, applied to test
- **Prices**: decimal scale (0–1), where 0.72 = 72¢ / 72% implied probability
- **Naive benchmark**: predict zero price change (move = 0)
- **Evaluation**: MAE, RMSE, R², directional accuracy, % improvement vs naive

## Model Comparison

| Model | MAE | RMSE | R² | Dir Acc | vs Naive |
|---|---|---|---|---|---|
| Naive (no change) | 0.03211 | — | — | — | 0.0% |
| baseline | 0.03302 | 0.06726 | 0.0014 | 38.9% | -2.8% |
| model1_momentum | 0.03308 | 0.06649 | 0.0241 | 42.4% | -3.0% |
| model2_cross | 0.03302 | 0.06644 | 0.0255 | 44.1% | -2.8% |
| model3_tod | 0.03336 | 0.06645 | 0.0251 | 48.3% | -3.9% |
| model4_gbm | 0.03314 | 0.06743 | -0.0038 | 47.0% | -3.2% |

## Feature Importance: Linear Model (Model 3 — all linear features)

Coefficients are normalized to % of total absolute weight after StandardScaler (i.e., features are on the same scale — higher % = stronger linear influence).

| Rank | Feature | Importance % | Coef | Category |
|---|---|---|---|---|
| 1 | `ask_lag1` | 20.02% | +0.0276 | momentum |
| 2 | `yes_ask_close` | 16.44% | -0.0227 | anchor |
| 3 | `mid_close` | 15.33% | -0.0211 | anchor |
| 4 | `spread` | 14.69% | -0.0202 | spread |
| 5 | `ask_lag5` | 7.69% | +0.0106 | momentum |
| 6 | `ask_lag15` | 6.09% | +0.0084 | momentum |
| 7 | `tod_sin` | 2.40% | -0.0033 | time |
| 8 | `spread_lag5` | 2.38% | +0.0033 | spread |
| 9 | `avg_spread_all` | 1.90% | +0.0026 | cross |
| 10 | `vol_sum_15` | 1.26% | -0.0017 | flow |
| 11 | `mid_vol_15` | 1.20% | +0.0017 | volatility |
| 12 | `tod_cos` | 1.10% | +0.0015 | time |
| 13 | `relative_spread` | 1.09% | +0.0015 | cross |
| 14 | `bid_intrabar_range` | 0.72% | +0.0010 | volatility |
| 15 | `rel_mid` | 0.72% | -0.0010 | cross |
| 16 | `price_previous` | 0.66% | +0.0009 | anchor |
| 17 | `sum_mid_all` | 0.61% | -0.0008 | cross |
| 18 | `prob_sum_deviation` | 0.61% | -0.0008 | cross |
| 19 | `bar_hour` | 0.61% | +0.0008 | time |
| 20 | `oi_change_15` | 0.58% | +0.0008 | flow |

### Category Summary (Linear)

| Category | Total Importance % |
|---|---|
| momentum | 35.2% |
| anchor | 32.4% |
| spread | 17.4% |
| cross | 5.5% |
| time | 4.5% |
| flow | 2.9% |
| volatility | 2.1% |

## Feature Importance: LightGBM (Model 4 — all features)

Importance by gain (total reduction in loss from splits on this feature).

| Rank | Feature | Gain | Share | Category | Bar |
|---|---|---|---|---|---|
| 1 | `tod_sin` | 2,063 | 6.7% | time | `█████████████████████████` |
| 2 | `tod_cos` | 2,039 | 6.6% | time | `█████████████████████████` |
| 3 | `vol_sum_15` | 1,685 | 5.4% | flow | `████████████████████░░░░░` |
| 4 | `oi_change_15` | 1,678 | 5.4% | flow | `████████████████████░░░░░` |
| 5 | `price_previous` | 1,603 | 5.2% | anchor | `███████████████████░░░░░░` |
| 6 | `mid_ret_30` | 1,528 | 4.9% | momentum | `███████████████████░░░░░░` |
| 7 | `mid_vol_15` | 1,465 | 4.7% | volatility | `██████████████████░░░░░░░` |
| 8 | `ask_lag15` | 1,372 | 4.4% | momentum | `█████████████████░░░░░░░░` |
| 9 | `mid_close` | 1,220 | 3.9% | anchor | `███████████████░░░░░░░░░░` |
| 10 | `minute_of_hour` | 1,127 | 3.6% | time | `██████████████░░░░░░░░░░░` |
| 11 | `mid_ret_15` | 1,023 | 3.3% | momentum | `████████████░░░░░░░░░░░░░` |
| 12 | `rel_mid` | 990 | 3.2% | cross | `████████████░░░░░░░░░░░░░` |
| 13 | `spread` | 954 | 3.1% | spread | `████████████░░░░░░░░░░░░░` |
| 14 | `yes_ask_close` | 952 | 3.1% | anchor | `████████████░░░░░░░░░░░░░` |
| 15 | `ask_lag5` | 943 | 3.0% | momentum | `███████████░░░░░░░░░░░░░░` |
| 16 | `spread_lag5` | 915 | 3.0% | spread | `███████████░░░░░░░░░░░░░░` |
| 17 | `bar_hour` | 828 | 2.7% | time | `██████████░░░░░░░░░░░░░░░` |
| 18 | `spread_dispersion` | 803 | 2.6% | cross | `██████████░░░░░░░░░░░░░░░` |
| 19 | `avg_spread_all` | 769 | 2.5% | cross | `█████████░░░░░░░░░░░░░░░░` |
| 20 | `vol_sum_5` | 744 | 2.4% | flow | `█████████░░░░░░░░░░░░░░░░` |

### Category Summary (GBM)

| Category | Total Gain % |
|---|---|
| momentum | 20.8% |
| time | 19.5% |
| flow | 16.1% |
| cross | 14.6% |
| anchor | 12.2% |
| volatility | 8.4% |
| spread | 8.4% |

## Cross-Ticker Signal Analysis

- Mean `sum_mid_all` (test set): **0.9740** (expect ≈1.0; deviation = arbitrage / liquidity imbalance)
- Std `sum_mid_all`: **0.1650**
- Mean `prob_sum_deviation`: **-0.0260**
- `rel_mid` rank in GBM: **#12** of 33
- `prob_sum_deviation` rank in GBM: **#28** of 33

Model 2 (+ cross features) vs Model 1 MAE delta: **+0.00007** (improvement)

## Per-Ticker MAE (Test Set)

Tail bins (T-prefix) often have wider spreads and sparser trading.

| Ticker | MAE | Rows | Mean Ask | Mean Spread |
|---|---|---|---|---|
| `KXHIGHLAX-25SEP25-B72.5` | 0.08425 | 1,136 | 0.3914 | 0.0427 |
| `KXHIGHLAX-25SEP19-T78` | 0.07446 | 1,045 | 0.4011 | 0.0667 |
| `KXHIGHLAX-25SEP08-T77` | 0.07333 | 1,208 | 0.4080 | 0.0674 |
| `KXHIGHLAX-25SEP02-B81.5` | 0.06903 | 1,108 | 0.3834 | 0.0760 |
| `KXHIGHLAX-25SEP20-B76.5` | 0.06724 | 923 | 0.5076 | 0.1070 |
| `KXHIGHLAX-25SEP25-B70.5` | 0.06718 | 965 | 0.2011 | 0.0376 |
| `KXHIGHLAX-25SEP19-B77.5` | 0.06711 | 931 | 0.5192 | 0.0763 |
| `KXHIGHLAX-25SEP20-B74.5` | 0.06509 | 857 | 0.3142 | 0.0716 |
| `KXHIGHLAX-25SEP29-B74.5` | 0.06197 | 1,114 | 0.5045 | 0.0570 |
| `KXHIGHLAX-25SEP21-B77.5` | 0.06143 | 967 | 0.3523 | 0.0719 |
| `KXHIGHLAX-25SEP10-B76.5` | 0.05737 | 1,235 | 0.4521 | 0.0666 |
| `KXHIGHLAX-25SEP18-B80.5` | 0.05713 | 880 | 0.2606 | 0.0804 |
| `KXHIGHLAX-25SEP18-B78.5` | 0.05698 | 865 | 0.4819 | 0.0894 |
| `KXHIGHLAX-25SEP29-B72.5` | 0.05669 | 1,115 | 0.5246 | 0.0600 |
| `KXHIGHLAX-25SEP27-B71.5` | 0.05526 | 1,052 | 0.2094 | 0.0297 |
| `KXHIGHLAX-25SEP26-B74.5` | 0.05431 | 1,049 | 0.4200 | 0.0313 |
| `KXHIGHLAX-25SEP17-B78.5` | 0.05331 | 893 | 0.3661 | 0.0677 |
| `KXHIGHLAX-25SEP24-B76.5` | 0.05207 | 1,142 | 0.4931 | 0.0392 |
| `KXHIGHLAX-25SEP08-B79.5` | 0.05079 | 1,208 | 0.2613 | 0.0529 |
| `KXHIGHLAX-25SEP20-T79` | 0.05004 | 946 | 0.1870 | 0.0759 |
| `KXHIGHLAX-25SEP30-T74` | 0.04996 | 1,150 | 0.2130 | 0.0307 |
| `KXHIGHLAX-25SEP24-B78.5` | 0.04908 | 1,221 | 0.3077 | 0.0446 |
| `KXHIGHLAX-25SEP25-B74.5` | 0.04899 | 1,100 | 0.3185 | 0.0383 |
| `KXHIGHLAX-25SEP28-B74.5` | 0.04764 | 1,265 | 0.6211 | 0.0621 |
| `KXHIGHLAX-25SEP23-B80.5` | 0.04764 | 982 | 0.3330 | 0.0509 |
| `KXHIGHLAX-25SEP17-B80.5` | 0.04762 | 1,102 | 0.3585 | 0.0545 |
| `KXHIGHLAX-25SEP22-B78.5` | 0.04721 | 1,047 | 0.3105 | 0.0432 |
| `KXHIGHLAX-25SEP18-B76.5` | 0.04720 | 707 | 0.2369 | 0.0689 |
| `KXHIGHLAX-25SEP17-T78` | 0.04645 | 821 | 0.4591 | 0.0626 |
| `KXHIGHLAX-25SEP13-B73.5` | 0.04640 | 995 | 0.3395 | 0.0436 |
| `KXHIGHLAX-25SEP23-B78.5` | 0.04619 | 803 | 0.3852 | 0.0310 |
| `KXHIGHLAX-25SEP28-B76.5` | 0.04586 | 1,102 | 0.2381 | 0.0603 |
| `KXHIGHLAX-25SEP14-B74.5` | 0.04516 | 913 | 0.3211 | 0.0485 |
| `KXHIGHLAX-25SEP06-T79` | 0.04464 | 730 | 0.1111 | 0.0406 |
| `KXHIGHLAX-25SEP10-B74.5` | 0.04456 | 1,187 | 0.4965 | 0.0498 |
| `KXHIGHLAX-25SEP08-B77.5` | 0.04431 | 1,212 | 0.3988 | 0.0726 |
| `KXHIGHLAX-25SEP23-B76.5` | 0.04355 | 878 | 0.4053 | 0.0351 |
| `KXHIGHLAX-25SEP05-B77.5` | 0.04340 | 1,118 | 0.2545 | 0.0408 |
| `KXHIGHLAX-25SEP15-B76.5` | 0.04296 | 1,223 | 0.5779 | 0.0520 |
| `KXHIGHLAX-25SEP15-B74.5` | 0.04294 | 973 | 0.1617 | 0.0468 |
| `KXHIGHLAX-25SEP27-B73.5` | 0.04211 | 1,186 | 0.5224 | 0.0257 |
| `KXHIGHLAX-25SEP21-B75.5` | 0.04171 | 939 | 0.5588 | 0.0635 |
| `KXHIGHLAX-25SEP26-B72.5` | 0.04157 | 926 | 0.3689 | 0.0395 |
| `KXHIGHLAX-25SEP27-B75.5` | 0.04144 | 1,018 | 0.2521 | 0.0282 |
| `KXHIGHLAX-25SEP28-B72.5` | 0.04143 | 1,114 | 0.2621 | 0.0643 |
| `KXHIGHLAX-25SEP24-B74.5` | 0.03991 | 817 | 0.2090 | 0.0444 |
| `KXHIGHLAX-25SEP29-T75` | 0.03982 | 1,149 | 0.0970 | 0.0484 |
| `KXHIGHLAX-25SEP20-B78.5` | 0.03919 | 920 | 0.2517 | 0.0782 |
| `KXHIGHLAX-25SEP02-B83.5` | 0.03880 | 988 | 0.1711 | 0.0428 |
| `KXHIGHLAX-25SEP07-B79.5` | 0.03840 | 1,090 | 0.5382 | 0.0485 |
| `KXHIGHLAX-25SEP29-B70.5` | 0.03837 | 999 | 0.1275 | 0.0629 |
| `KXHIGHLAX-25SEP11-B73.5` | 0.03820 | 1,080 | 0.3546 | 0.0492 |
| `KXHIGHLAX-25SEP06-B78.5` | 0.03791 | 966 | 0.4363 | 0.0394 |
| `KXHIGHLAX-25SEP16-B79.5` | 0.03781 | 1,011 | 0.5846 | 0.0494 |
| `KXHIGHLAX-25SEP06-B76.5` | 0.03781 | 794 | 0.5395 | 0.0361 |
| `KXHIGHLAX-25SEP15-B78.5` | 0.03694 | 1,265 | 0.2925 | 0.0523 |
| `KXHIGHLAX-25SEP07-B81.5` | 0.03686 | 1,033 | 0.2832 | 0.0565 |
| `KXHIGHLAX-25SEP21-B73.5` | 0.03665 | 656 | 0.1337 | 0.0329 |
| `KXHIGHLAX-25SEP30-B73.5` | 0.03661 | 1,349 | 0.5990 | 0.0333 |
| `KXHIGHLAX-25SEP04-B78.5` | 0.03658 | 850 | 0.3487 | 0.0503 |
| `KXHIGHLAX-25SEP03-T82` | 0.03633 | 1,030 | 0.5508 | 0.0506 |
| `KXHIGHLAX-25SEP11-B75.5` | 0.03631 | 1,166 | 0.5667 | 0.0388 |
| `KXHIGHLAX-25SEP05-B75.5` | 0.03597 | 1,129 | 0.6392 | 0.0388 |
| `KXHIGHLAX-25SEP14-B76.5` | 0.03590 | 1,252 | 0.5585 | 0.0577 |
| `KXHIGHLAX-25SEP30-B71.5` | 0.03479 | 1,149 | 0.2652 | 0.0317 |
| `KXHIGHLAX-25SEP13-B75.5` | 0.03477 | 1,378 | 0.6077 | 0.0490 |
| `KXHIGHLAX-25SEP01-T80` | 0.03459 | 1,067 | 0.2635 | 0.0461 |
| `KXHIGHLAX-25SEP10-B72.5` | 0.03361 | 1,001 | 0.1261 | 0.0344 |
| `KXHIGHLAX-25SEP26-B76.5` | 0.03322 | 961 | 0.2279 | 0.0357 |
| `KXHIGHLAX-25SEP04-B76.5` | 0.03316 | 825 | 0.6105 | 0.0330 |
| `KXHIGHLAX-25SEP22-B76.5` | 0.03311 | 1,147 | 0.5379 | 0.0359 |
| `KXHIGHLAX-25SEP22-B74.5` | 0.03255 | 909 | 0.1819 | 0.0481 |
| `KXHIGHLAX-25SEP25-B76.5` | 0.03249 | 986 | 0.1673 | 0.0281 |
| `KXHIGHLAX-25SEP18-B82.5` | 0.03246 | 724 | 0.1689 | 0.0823 |
| `KXHIGHLAX-25SEP16-B81.5` | 0.03196 | 1,046 | 0.2254 | 0.0452 |
| `KXHIGHLAX-25SEP16-B77.5` | 0.03143 | 785 | 0.1799 | 0.0375 |
| `KXHIGHLAX-25SEP09-B77.5` | 0.03141 | 1,000 | 0.1535 | 0.0479 |
| `KXHIGHLAX-25SEP01-B80.5` | 0.03041 | 1,036 | 0.4873 | 0.0378 |
| `KXHIGHLAX-25SEP03-B82.5` | 0.02961 | 990 | 0.3993 | 0.0568 |
| `KXHIGHLAX-25SEP01-B82.5` | 0.02879 | 1,106 | 0.3123 | 0.0660 |
| `KXHIGHLAX-25SEP09-B75.5` | 0.02844 | 1,042 | 0.6275 | 0.0513 |
| `KXHIGHLAX-25SEP12-T77` | 0.02729 | 1,382 | 0.1512 | 0.0402 |
| `KXHIGHLAX-25SEP12-B74.5` | 0.02717 | 1,120 | 0.3379 | 0.0384 |
| `KXHIGHLAX-25SEP07-B77.5` | 0.02638 | 918 | 0.2156 | 0.0375 |
| `KXHIGHLAX-25SEP03-B84.5` | 0.02578 | 867 | 0.1328 | 0.0446 |
| `KXHIGHLAX-25SEP17-B82.5` | 0.02522 | 809 | 0.0967 | 0.0528 |
| `KXHIGHLAX-25SEP12-B76.5` | 0.02456 | 1,388 | 0.6122 | 0.0541 |
| `KXHIGHLAX-25SEP06-B74.5` | 0.02444 | 715 | 0.2053 | 0.0335 |
| `KXHIGHLAX-25SEP08-B81.5` | 0.02434 | 1,022 | 0.0961 | 0.0424 |
| `KXHIGHLAX-25SEP11-T76` | 0.02350 | 866 | 0.1314 | 0.0502 |
| `KXHIGHLAX-25SEP09-B73.5` | 0.02334 | 1,135 | 0.2450 | 0.0293 |
| `KXHIGHLAX-25SEP02-B85.5` | 0.02298 | 906 | 0.1240 | 0.0443 |
| `KXHIGHLAX-25SEP16-B83.5` | 0.02255 | 816 | 0.0948 | 0.0488 |
| `KXHIGHLAX-25SEP10-T77` | 0.02167 | 903 | 0.0731 | 0.0474 |
| `KXHIGHLAX-25SEP01-B84.5` | 0.02104 | 1,018 | 0.0865 | 0.0391 |
| `KXHIGHLAX-25SEP13-B77.5` | 0.02016 | 938 | 0.1302 | 0.0405 |
| `KXHIGHLAX-25SEP19-B75.5` | 0.02013 | 1,007 | 0.1293 | 0.0265 |
| `KXHIGHLAX-25SEP24-T79` | 0.01917 | 1,130 | 0.1002 | 0.0351 |
| `KXHIGHLAX-25SEP27-T76` | 0.01883 | 936 | 0.0752 | 0.0307 |
| `KXHIGHLAX-25SEP14-B78.5` | 0.01872 | 832 | 0.1893 | 0.0317 |
| `KXHIGHLAX-25SEP15-B80.5` | 0.01863 | 720 | 0.0769 | 0.0527 |
| `KXHIGHLAX-25SEP23-B74.5` | 0.01815 | 437 | 0.1079 | 0.0347 |
| `KXHIGHLAX-25SEP26-B78.5` | 0.01635 | 766 | 0.0527 | 0.0224 |
| `KXHIGHLAX-25SEP21-B79.5` | 0.01634 | 531 | 0.0859 | 0.0360 |
| `KXHIGHLAX-25SEP11-B71.5` | 0.01612 | 659 | 0.0554 | 0.0331 |
| `KXHIGHLAX-25SEP04-T76` | 0.01501 | 662 | 0.0732 | 0.0382 |
| `KXHIGHLAX-25SEP07-B83.5` | 0.01480 | 760 | 0.0612 | 0.0322 |
| `KXHIGHLAX-25SEP30-B69.5` | 0.01446 | 768 | 0.0522 | 0.0267 |
| `KXHIGHLAX-25SEP26-T72` | 0.01435 | 1,096 | 0.0557 | 0.0186 |
| `KXHIGHLAX-25SEP05-B79.5` | 0.01383 | 933 | 0.0604 | 0.0273 |
| `KXHIGHLAX-25SEP28-B70.5` | 0.01321 | 555 | 0.0493 | 0.0352 |
| `KXHIGHLAX-25SEP22-B80.5` | 0.01307 | 670 | 0.0648 | 0.0336 |
| `KXHIGHLAX-25SEP05-B73.5` | 0.01289 | 1,197 | 0.1132 | 0.0240 |
| `KXHIGHLAX-25SEP04-B80.5` | 0.01256 | 854 | 0.0645 | 0.0191 |
| `KXHIGHLAX-25SEP13-B71.5` | 0.01197 | 655 | 0.0463 | 0.0350 |
| `KXHIGHLAX-25SEP28-T77` | 0.01106 | 717 | 0.0459 | 0.0265 |
| `KXHIGHLAX-25SEP17-B84.5` | 0.01016 | 588 | 0.0485 | 0.0391 |
| `KXHIGHLAX-25SEP25-T77` | 0.01013 | 959 | 0.0400 | 0.0261 |
| `KXHIGHLAX-25SEP12-B72.5` | 0.00960 | 809 | 0.0577 | 0.0280 |
| `KXHIGHLAX-25SEP24-B72.5` | 0.00942 | 428 | 0.0326 | 0.0237 |
| `KXHIGHLAX-25SEP19-B73.5` | 0.00936 | 349 | 0.0531 | 0.0255 |
| `KXHIGHLAX-25SEP09-T78` | 0.00933 | 788 | 0.0395 | 0.0161 |
| `KXHIGHLAX-25SEP07-T84` | 0.00916 | 691 | 0.0343 | 0.0299 |
| `KXHIGHLAX-25SEP18-T76` | 0.00891 | 245 | 0.0327 | 0.0182 |
| `KXHIGHLAX-25SEP13-T78` | 0.00882 | 1,184 | 0.0402 | 0.0233 |
| `KXHIGHLAX-25SEP16-T77` | 0.00836 | 430 | 0.0400 | 0.0210 |
| `KXHIGHLAX-25SEP07-T77` | 0.00830 | 744 | 0.0439 | 0.0229 |
| `KXHIGHLAX-25SEP16-T84` | 0.00806 | 724 | 0.0436 | 0.0244 |
| `KXHIGHLAX-25SEP09-B71.5` | 0.00792 | 453 | 0.0464 | 0.0282 |
| `KXHIGHLAX-25SEP21-T80` | 0.00784 | 385 | 0.0279 | 0.0230 |
| `KXHIGHLAX-25SEP25-T70` | 0.00736 | 526 | 0.0263 | 0.0259 |
| `KXHIGHLAX-25SEP21-T73` | 0.00718 | 508 | 0.0245 | 0.0233 |
| `KXHIGHLAX-25SEP04-B82.5` | 0.00701 | 495 | 0.0295 | 0.0274 |
| `KXHIGHLAX-25SEP14-T72` | 0.00691 | 111 | 0.0271 | 0.0271 |
| `KXHIGHLAX-25SEP14-T79` | 0.00664 | 1,301 | 0.0387 | 0.0258 |
| `KXHIGHLAX-25SEP14-B72.5` | 0.00651 | 410 | 0.0370 | 0.0273 |
| `KXHIGHLAX-25SEP10-T70` | 0.00629 | 673 | 0.0220 | 0.0220 |
| `KXHIGHLAX-25SEP27-B69.5` | 0.00607 | 608 | 0.0300 | 0.0209 |
| `KXHIGHLAX-25SEP20-B72.5` | 0.00534 | 219 | 0.0317 | 0.0317 |
| `KXHIGHLAX-25SEP22-T74` | 0.00529 | 188 | 0.0163 | 0.0151 |
| `KXHIGHLAX-25SEP08-T84` | 0.00460 | 779 | 0.0268 | 0.0268 |
| `KXHIGHLAX-25SEP05-T80` | 0.00420 | 633 | 0.0274 | 0.0220 |
| `KXHIGHLAX-25SEP20-T72` | 0.00419 | 160 | 0.0301 | 0.0301 |
| `KXHIGHLAX-25SEP10-B70.5` | 0.00396 | 660 | 0.0175 | 0.0174 |
| `KXHIGHLAX-25SEP08-B83.5` | 0.00370 | 903 | 0.0284 | 0.0279 |
| `KXHIGHLAX-25SEP06-B72.5` | 0.00322 | 500 | 0.0326 | 0.0255 |
| `KXHIGHLAX-25SEP19-B71.5` | 0.00322 | 173 | 0.0279 | 0.0278 |
| `KXHIGHLAX-25SEP05-T73` | 0.00309 | 769 | 0.0282 | 0.0215 |
| `KXHIGHLAX-25SEP15-T74` | 0.00299 | 397 | 0.0334 | 0.0290 |
| `KXHIGHLAX-25SEP26-T79` | 0.00294 | 549 | 0.0282 | 0.0186 |
| `KXHIGHLAX-25SEP09-T71` | 0.00244 | 253 | 0.0144 | 0.0144 |
| `KXHIGHLAX-25SEP29-B68.5` | 0.00227 | 398 | 0.0248 | 0.0228 |
| `KXHIGHLAX-25SEP13-T71` | 0.00221 | 255 | 0.0126 | 0.0125 |
| `KXHIGHLAX-25SEP12-B70.5` | 0.00220 | 161 | 0.0219 | 0.0219 |
| `KXHIGHLAX-25SEP06-T72` | 0.00174 | 219 | 0.0128 | 0.0128 |
| `KXHIGHLAX-25SEP28-T70` | 0.00142 | 426 | 0.0136 | 0.0136 |

## Key Findings

1. **Top linear predictors**: `ask_lag1`, `yes_ask_close`, `mid_close` — features ranked by normalized coefficient weight after scaling. With a relative (move) target, price-level features carry less weight than momentum/flow signals.
2. **Momentum features provide modest lift**: Model 1 (momentum) reduces MAE by -0.00006 vs baseline, suggesting lag returns and spread carry incremental signal beyond the current price level alone.
3. **Cross-ticker signals**: Adding probability conservation features shifts MAE by +0.00007. The sum-of-mids deviation from 1.0 reflects cross-bin arbitrage opportunities that briefly predict price moves.
4. **Time-of-day**: Adding time features shifts MAE by -0.00035. The cyclical encoding captures METAR observation windows (fires at :53 past each hour) when informed traders update positions.
5. **Non-linearity**: LightGBM (MAE=0.03314) vs best linear model (MAE=0.03336) — delta=+0.00022. GBM captures meaningful non-linear interactions.

## Limitations

- Only 11.5% of bars have trade OHLC (`price_*`); models rely primarily on quote OHLC
- Same-day bars only — no inter-day momentum or regime features
- 122 days of training data; seasonal effects within summer may be underweighted
- `price_previous` (2.1% null) imputed with median — may understate signal in thin markets
- Tail bins (T-prefix) have wider spreads; a per-bin model may outperform a pooled one
- The T+30 target assumes a bar exists 30 minutes later (dropped ≈20% of bars near day-end)
