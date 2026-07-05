# Kalshi Weather Market: Price Direction Classification Report

_Generated: 2026-07-05 16:01 UTC_

**Train**: 2025-06-01 – 2025-08-31  **Test**: 2025-09-01 – 2025-09-30

## Methodology

- **Target**: binary — did `yes_ask_open` at T+30 go UP (1) or DOWN (0)?
- **Flat bars excluded**: rows where `target_move_30 == 0` (~35% of data) are dropped; they carry no directional signal
- **Strict split** (no flat): train 177,154 rows, test 90,037 rows
- **Threshold split** (|move| ≥ 0.005): train 177,154 rows, test 90,037 rows
- **Train/Test split**: purely temporal — 2025-06-01–2025-08-31 train, 2025-09-01–2025-09-30 test
- **Baseline**: predict majority class (50/50 after flat removal → 50% accuracy)

## Model Comparison — Strict (exclude flat bars)

| Model | Accuracy | AUC | Precision↑ | Recall↑ |
|---|---|---|---|---|
| Majority-class baseline | 50.0% | 0.500 | — | — |
| baseline_lr | **53.1%** | 0.5297 | 52.9% | 38.9% |
| model1_lr | **56.5%** | 0.6019 | 55.3% | 59.0% |
| model2_lr | **58.7%** | 0.6194 | 57.5% | 60.0% |
| model3_lr | **58.5%** | 0.6182 | 57.7% | 57.3% |
| model4_gbm | **60.6%** | 0.6520 | 59.7% | 60.6% |

## Model Comparison — Threshold (|move| ≥ 0.005, filters microstructure noise)

| Model | Accuracy | AUC | Precision↑ | Recall↑ |
|---|---|---|---|---|
| Majority-class baseline | 50.0% | 0.500 | — | — |
| model4_gbm_thresh | **60.6%** | 0.6520 | 59.7% | 60.6% |

## Feature Importance: Logistic Regression (Model 3 — all linear features)

Coefficients normalized to % of total absolute weight (StandardScaler applied first).

| Rank | Feature | Importance % | Coef | Category |
|---|---|---|---|---|
| 1 | `ask_lag1` | 15.15% | +0.4326 | momentum |
| 2 | `relative_spread` | 11.37% | -0.3245 | cross |
| 3 | `yes_ask_close` | 8.48% | -0.2421 | anchor |
| 4 | `mid_close` | 8.15% | -0.2327 | anchor |
| 5 | `avg_spread_all` | 8.13% | -0.2320 | cross |
| 6 | `ask_lag15` | 5.94% | +0.1695 | momentum |
| 7 | `spread` | 4.72% | -0.1348 | spread |
| 8 | `ask_lag5` | 4.23% | +0.1207 | momentum |
| 9 | `price_previous` | 3.28% | +0.0937 | anchor |
| 10 | `mid_vol_15` | 2.77% | +0.0789 | volatility |
| 11 | `mid_ret_1` | 2.70% | -0.0770 | momentum |
| 12 | `spread_dispersion` | 2.53% | +0.0721 | cross |
| 13 | `spread_lag5` | 2.28% | +0.0652 | spread |
| 14 | `rel_mid` | 2.27% | -0.0649 | cross |
| 15 | `mid_ret_15` | 2.10% | -0.0601 | momentum |
| 16 | `mid_ret_5` | 1.78% | -0.0508 | momentum |
| 17 | `minute_of_hour` | 1.75% | +0.0499 | time |
| 18 | `mid_ret_30` | 1.66% | -0.0475 | momentum |
| 19 | `sum_mid_all` | 1.55% | -0.0441 | cross |
| 20 | `prob_sum_deviation` | 1.55% | -0.0441 | cross |

### Category Summary (Logistic Regression)

| Category | Total Importance % |
|---|---|
| momentum | 33.6% |
| cross | 28.9% |
| anchor | 19.9% |
| spread | 8.0% |
| volatility | 4.5% |
| flow | 3.3% |
| time | 1.7% |

## Feature Importance: LightGBM (Model 4)

Importance by gain.

| Rank | Feature | Gain | Share | Category | Bar |
|---|---|---|---|---|---|
| 1 | `tod_sin` | 2,343 | 7.6% | time | `█████████████████████████` |
| 2 | `tod_cos` | 2,339 | 7.5% | time | `█████████████████████████` |
| 3 | `price_previous` | 1,845 | 6.0% | anchor | `████████████████████░░░░░` |
| 4 | `vol_sum_15` | 1,741 | 5.6% | flow | `███████████████████░░░░░░` |
| 5 | `oi_change_15` | 1,680 | 5.4% | flow | `██████████████████░░░░░░░` |
| 6 | `mid_ret_30` | 1,664 | 5.4% | momentum | `██████████████████░░░░░░░` |
| 7 | `mid_vol_15` | 1,495 | 4.8% | volatility | `████████████████░░░░░░░░░` |
| 8 | `mid_close` | 1,337 | 4.3% | anchor | `██████████████░░░░░░░░░░░` |
| 9 | `rel_mid` | 1,249 | 4.0% | cross | `█████████████░░░░░░░░░░░░` |
| 10 | `yes_ask_close` | 1,235 | 4.0% | anchor | `█████████████░░░░░░░░░░░░` |
| 11 | `ask_lag15` | 1,112 | 3.6% | momentum | `████████████░░░░░░░░░░░░░` |
| 12 | `spread` | 1,068 | 3.4% | spread | `███████████░░░░░░░░░░░░░░` |
| 13 | `mid_ret_15` | 994 | 3.2% | momentum | `███████████░░░░░░░░░░░░░░` |
| 14 | `avg_spread_all` | 879 | 2.8% | cross | `█████████░░░░░░░░░░░░░░░░` |
| 15 | `minute_of_hour` | 868 | 2.8% | time | `█████████░░░░░░░░░░░░░░░░` |
| 16 | `relative_spread` | 840 | 2.7% | cross | `█████████░░░░░░░░░░░░░░░░` |
| 17 | `bar_hour` | 810 | 2.6% | time | `█████████░░░░░░░░░░░░░░░░` |
| 18 | `spread_dispersion` | 771 | 2.5% | cross | `████████░░░░░░░░░░░░░░░░░` |
| 19 | `prob_sum_deviation` | 723 | 2.3% | cross | `████████░░░░░░░░░░░░░░░░░` |
| 20 | `sum_mid_all` | 720 | 2.3% | cross | `████████░░░░░░░░░░░░░░░░░` |

### Category Summary (GBM)

| Category | Total Gain % |
|---|---|
| time | 20.5% |
| momentum | 18.7% |
| cross | 17.5% |
| anchor | 14.2% |
| flow | 14.2% |
| volatility | 7.4% |
| spread | 7.4% |

## Per-Ticker Accuracy (GBM, Strict, Test Set)

| Ticker | Accuracy | N Rows | Mean |move| |
|---|---|---|---|
| `KXHIGHLAX-25SEP06-T72` | 100.0% | 3 | 0.0100 |
| `KXHIGHLAX-25SEP05-T80` | 96.3% | 136 | 0.0138 |
| `KXHIGHLAX-25SEP26-T79` | 96.2% | 53 | 0.0100 |
| `KXHIGHLAX-25SEP25-T70` | 85.4% | 158 | 0.0209 |
| `KXHIGHLAX-25SEP06-B72.5` | 84.6% | 91 | 0.0111 |
| `KXHIGHLAX-25SEP08-T84` | 83.8% | 235 | 0.0140 |
| `KXHIGHLAX-25SEP20-T72` | 82.5% | 40 | 0.0105 |
| `KXHIGHLAX-25SEP15-T74` | 80.5% | 41 | 0.0129 |
| `KXHIGHLAX-25SEP04-B82.5` | 79.4% | 170 | 0.0191 |
| `KXHIGHLAX-25SEP14-T72` | 78.1% | 32 | 0.0187 |
| `KXHIGHLAX-25SEP13-T71` | 75.0% | 8 | 0.0125 |
| `KXHIGHLAX-25SEP28-T70` | 75.0% | 4 | 0.0100 |
| `KXHIGHLAX-25SEP18-B82.5` | 74.4% | 574 | 0.0369 |
| `KXHIGHLAX-25SEP07-T84` | 73.2% | 291 | 0.0170 |
| `KXHIGHLAX-25SEP27-B69.5` | 73.1% | 145 | 0.0201 |
| `KXHIGHLAX-25SEP04-B80.5` | 72.9% | 343 | 0.0300 |
| `KXHIGHLAX-25SEP14-B72.5` | 72.7% | 99 | 0.0236 |
| `KXHIGHLAX-25SEP13-B71.5` | 72.7% | 271 | 0.0261 |
| `KXHIGHLAX-25SEP04-T76` | 72.5% | 363 | 0.0288 |
| `KXHIGHLAX-25SEP14-B78.5` | 72.2% | 544 | 0.0244 |
| `KXHIGHLAX-25SEP24-B72.5` | 72.1% | 165 | 0.0190 |
| `KXHIGHLAX-25SEP15-B80.5` | 72.0% | 472 | 0.0279 |
| `KXHIGHLAX-25SEP03-B84.5` | 71.7% | 530 | 0.0438 |
| `KXHIGHLAX-25SEP10-T70` | 71.5% | 144 | 0.0266 |
| `KXHIGHLAX-25SEP17-B84.5` | 71.3% | 216 | 0.0265 |
| `KXHIGHLAX-25SEP10-B70.5` | 70.8% | 106 | 0.0177 |
| `KXHIGHLAX-25SEP09-B73.5` | 70.7% | 615 | 0.0445 |
| `KXHIGHLAX-25SEP26-T72` | 69.6% | 467 | 0.0306 |
| `KXHIGHLAX-25SEP25-T77` | 68.3% | 523 | 0.0165 |
| `KXHIGHLAX-25SEP09-B71.5` | 68.1% | 160 | 0.0179 |
| `KXHIGHLAX-25SEP11-B75.5` | 67.9% | 795 | 0.0568 |
| `KXHIGHLAX-25SEP12-B72.5` | 67.6% | 358 | 0.0179 |
| `KXHIGHLAX-25SEP14-T79` | 67.3% | 507 | 0.0141 |
| `KXHIGHLAX-25SEP20-B72.5` | 67.2% | 64 | 0.0111 |
| `KXHIGHLAX-25SEP01-B84.5` | 67.1% | 526 | 0.0418 |
| `KXHIGHLAX-25SEP10-B76.5` | 67.0% | 1,020 | 0.0712 |
| `KXHIGHLAX-25SEP20-B78.5` | 66.8% | 787 | 0.0477 |
| `KXHIGHLAX-25SEP05-B75.5` | 66.6% | 895 | 0.0453 |
| `KXHIGHLAX-25SEP28-T77` | 66.6% | 419 | 0.0147 |
| `KXHIGHLAX-25SEP26-B76.5` | 66.5% | 822 | 0.0388 |
| `KXHIGHLAX-25SEP16-T84` | 66.5% | 272 | 0.0167 |
| `KXHIGHLAX-25SEP03-B82.5` | 66.4% | 721 | 0.0420 |
| `KXHIGHLAX-25SEP06-B78.5` | 66.3% | 695 | 0.0514 |
| `KXHIGHLAX-25SEP27-B71.5` | 66.3% | 836 | 0.0705 |
| `KXHIGHLAX-25SEP30-B73.5` | 66.3% | 1,153 | 0.0455 |
| `KXHIGHLAX-25SEP19-B75.5` | 66.1% | 537 | 0.0358 |
| `KXHIGHLAX-25SEP08-B83.5` | 65.9% | 220 | 0.0127 |
| `KXHIGHLAX-25SEP04-B76.5` | 65.7% | 673 | 0.0395 |
| `KXHIGHLAX-25SEP10-B74.5` | 65.4% | 864 | 0.0609 |
| `KXHIGHLAX-25SEP01-T80` | 65.4% | 754 | 0.0521 |
| `KXHIGHLAX-25SEP11-B71.5` | 65.4% | 306 | 0.0325 |
| `KXHIGHLAX-25SEP30-B69.5` | 65.3% | 398 | 0.0276 |
| `KXHIGHLAX-25SEP24-T79` | 65.3% | 790 | 0.0254 |
| `KXHIGHLAX-25SEP22-B76.5` | 65.3% | 832 | 0.0494 |
| `KXHIGHLAX-25SEP04-B78.5` | 65.3% | 659 | 0.0513 |
| `KXHIGHLAX-25SEP29-B74.5` | 65.1% | 962 | 0.0751 |
| `KXHIGHLAX-25SEP22-B80.5` | 65.1% | 458 | 0.0179 |
| `KXHIGHLAX-25SEP22-T74` | 65.0% | 20 | 0.0310 |
| `KXHIGHLAX-25SEP28-B74.5` | 64.7% | 1,132 | 0.0538 |
| `KXHIGHLAX-25SEP03-T82` | 64.6% | 861 | 0.0462 |
| `KXHIGHLAX-25SEP08-B81.5` | 64.6% | 694 | 0.0350 |
| `KXHIGHLAX-25SEP11-T76` | 64.5% | 643 | 0.0316 |
| `KXHIGHLAX-25SEP12-B76.5` | 64.4% | 917 | 0.0364 |
| `KXHIGHLAX-25SEP17-B82.5` | 64.4% | 509 | 0.0399 |
| `KXHIGHLAX-25SEP15-B76.5` | 63.9% | 923 | 0.0532 |
| `KXHIGHLAX-25SEP13-B73.5` | 63.8% | 765 | 0.0598 |
| `KXHIGHLAX-25SEP20-T79` | 63.8% | 776 | 0.0589 |
| `KXHIGHLAX-25SEP05-B79.5` | 63.7% | 405 | 0.0272 |
| `KXHIGHLAX-25SEP23-B80.5` | 63.1% | 696 | 0.0661 |
| `KXHIGHLAX-25SEP29-B68.5` | 62.9% | 35 | 0.0109 |
| `KXHIGHLAX-25SEP07-B83.5` | 62.7% | 451 | 0.0253 |
| `KXHIGHLAX-25SEP09-B75.5` | 62.5% | 830 | 0.0327 |
| `KXHIGHLAX-25SEP05-T73` | 62.5% | 96 | 0.0103 |
| `KXHIGHLAX-25SEP15-B78.5` | 62.4% | 1,036 | 0.0452 |
| `KXHIGHLAX-25SEP01-B80.5` | 62.3% | 806 | 0.0393 |
| `KXHIGHLAX-25SEP28-B72.5` | 62.3% | 967 | 0.0470 |
| `KXHIGHLAX-25SEP26-B78.5` | 62.2% | 482 | 0.0232 |
| `KXHIGHLAX-25SEP10-B72.5` | 62.2% | 635 | 0.0508 |
| `KXHIGHLAX-25SEP27-T76` | 62.0% | 656 | 0.0256 |
| `KXHIGHLAX-25SEP02-B83.5` | 61.9% | 825 | 0.0455 |
| `KXHIGHLAX-25SEP16-T77` | 61.9% | 147 | 0.0187 |
| `KXHIGHLAX-25SEP16-B79.5` | 61.3% | 783 | 0.0464 |
| `KXHIGHLAX-25SEP28-B70.5` | 60.9% | 215 | 0.0291 |
| `KXHIGHLAX-25SEP07-B79.5` | 60.9% | 823 | 0.0525 |
| `KXHIGHLAX-25SEP26-B74.5` | 60.6% | 909 | 0.0634 |
| `KXHIGHLAX-25SEP13-T78` | 60.5% | 549 | 0.0165 |
| `KXHIGHLAX-25SEP29-B70.5` | 60.3% | 677 | 0.0582 |
| `KXHIGHLAX-25SEP19-B73.5` | 60.2% | 176 | 0.0173 |
| `KXHIGHLAX-25SEP20-B74.5` | 60.1% | 752 | 0.0730 |
| `KXHIGHLAX-25SEP29-T75` | 59.8% | 920 | 0.0481 |
| `KXHIGHLAX-25SEP21-B75.5` | 59.7% | 673 | 0.0540 |
| `KXHIGHLAX-25SEP12-B74.5` | 59.6% | 914 | 0.0324 |
| `KXHIGHLAX-25SEP24-B76.5` | 59.6% | 893 | 0.0688 |
| `KXHIGHLAX-25SEP23-B74.5` | 59.5% | 291 | 0.0255 |
| `KXHIGHLAX-25SEP11-B73.5` | 59.3% | 735 | 0.0563 |
| `KXHIGHLAX-25SEP16-B83.5` | 59.3% | 538 | 0.0349 |
| `KXHIGHLAX-25SEP25-B74.5` | 59.2% | 892 | 0.0582 |
| `KXHIGHLAX-25SEP19-B77.5` | 58.9% | 735 | 0.0859 |
| `KXHIGHLAX-25SEP29-B72.5` | 58.9% | 1,002 | 0.0668 |
| `KXHIGHLAX-25SEP07-B77.5` | 58.9% | 632 | 0.0387 |
| `KXHIGHLAX-25SEP18-B78.5` | 58.8% | 738 | 0.0619 |
| `KXHIGHLAX-25SEP21-B79.5` | 58.8% | 318 | 0.0239 |
| `KXHIGHLAX-25SEP09-B77.5` | 58.7% | 770 | 0.0345 |
| `KXHIGHLAX-25SEP18-B76.5` | 58.6% | 561 | 0.0567 |
| `KXHIGHLAX-25SEP06-B76.5` | 58.4% | 507 | 0.0567 |
| `KXHIGHLAX-25SEP06-T79` | 58.3% | 549 | 0.0598 |
| `KXHIGHLAX-25SEP30-T74` | 58.3% | 949 | 0.0566 |
| `KXHIGHLAX-25SEP21-T80` | 58.2% | 141 | 0.0129 |
| `KXHIGHLAX-25SEP17-B80.5` | 58.1% | 897 | 0.0590 |
| `KXHIGHLAX-25SEP25-B76.5` | 57.9% | 724 | 0.0410 |
| `KXHIGHLAX-25SEP01-B82.5` | 57.8% | 818 | 0.0360 |
| `KXHIGHLAX-25SEP06-B74.5` | 57.7% | 475 | 0.0350 |
| `KXHIGHLAX-25SEP09-T78` | 57.5% | 398 | 0.0152 |
| `KXHIGHLAX-25SEP18-B80.5` | 57.2% | 711 | 0.0687 |
| `KXHIGHLAX-25SEP23-B76.5` | 57.2% | 636 | 0.0585 |
| `KXHIGHLAX-25SEP27-B75.5` | 57.0% | 770 | 0.0466 |
| `KXHIGHLAX-25SEP10-T77` | 57.0% | 586 | 0.0299 |
| `KXHIGHLAX-25SEP08-B79.5` | 57.0% | 1,034 | 0.0600 |
| `KXHIGHLAX-25SEP08-T77` | 56.8% | 1,100 | 0.0739 |
| `KXHIGHLAX-25SEP28-B76.5` | 56.7% | 973 | 0.0511 |
| `KXHIGHLAX-25SEP23-B78.5` | 56.7% | 621 | 0.0599 |
| `KXHIGHLAX-25SEP12-T77` | 56.7% | 893 | 0.0395 |
| `KXHIGHLAX-25SEP27-B73.5` | 56.6% | 1,031 | 0.0487 |
| `KXHIGHLAX-25SEP26-B72.5` | 56.6% | 791 | 0.0519 |
| `KXHIGHLAX-25SEP07-B81.5` | 56.6% | 705 | 0.0549 |
| `KXHIGHLAX-25SEP05-B77.5` | 56.6% | 813 | 0.0597 |
| `KXHIGHLAX-25SEP13-B75.5` | 56.5% | 976 | 0.0474 |
| `KXHIGHLAX-25SEP08-B77.5` | 56.4% | 1,024 | 0.0516 |
| `KXHIGHLAX-25SEP15-B74.5` | 56.2% | 719 | 0.0561 |
| `KXHIGHLAX-25SEP17-T78` | 56.1% | 716 | 0.0559 |
| `KXHIGHLAX-25SEP02-B85.5` | 55.8% | 647 | 0.0297 |
| `KXHIGHLAX-25SEP20-B76.5` | 55.4% | 810 | 0.0671 |
| `KXHIGHLAX-25SEP05-B73.5` | 54.7% | 541 | 0.0247 |
| `KXHIGHLAX-25SEP14-B76.5` | 54.5% | 780 | 0.0543 |
| `KXHIGHLAX-25SEP24-B74.5` | 54.3% | 588 | 0.0527 |
| `KXHIGHLAX-25SEP25-B72.5` | 54.1% | 1,020 | 0.0844 |
| `KXHIGHLAX-25SEP02-B81.5` | 53.6% | 1,024 | 0.0701 |
| `KXHIGHLAX-25SEP14-B74.5` | 53.2% | 631 | 0.0576 |
| `KXHIGHLAX-25SEP22-B78.5` | 53.2% | 862 | 0.0526 |
| `KXHIGHLAX-25SEP30-B71.5` | 53.2% | 816 | 0.0433 |
| `KXHIGHLAX-25SEP18-T76` | 53.1% | 98 | 0.0159 |
| `KXHIGHLAX-25SEP16-B81.5` | 52.6% | 740 | 0.0423 |
| `KXHIGHLAX-25SEP21-T73` | 52.2% | 134 | 0.0209 |
| `KXHIGHLAX-25SEP16-B77.5` | 51.4% | 496 | 0.0522 |
| `KXHIGHLAX-25SEP07-T77` | 51.2% | 340 | 0.0144 |
| `KXHIGHLAX-25SEP17-B78.5` | 51.2% | 682 | 0.0666 |
| `KXHIGHLAX-25SEP25-B70.5` | 49.4% | 638 | 0.1043 |
| `KXHIGHLAX-25SEP21-B73.5` | 48.9% | 446 | 0.0508 |
| `KXHIGHLAX-25SEP24-B78.5` | 47.7% | 944 | 0.0588 |
| `KXHIGHLAX-25SEP21-B77.5` | 47.3% | 799 | 0.0724 |
| `KXHIGHLAX-25SEP22-B74.5` | 46.9% | 693 | 0.0387 |
| `KXHIGHLAX-25SEP13-B77.5` | 46.6% | 671 | 0.0264 |
| `KXHIGHLAX-25SEP19-T78` | 45.3% | 907 | 0.0762 |
| `KXHIGHLAX-25SEP09-T71` | 43.8% | 16 | 0.0163 |
| `KXHIGHLAX-25SEP19-B71.5` | 40.9% | 22 | 0.0109 |
| `KXHIGHLAX-25SEP12-B70.5` | 10.0% | 10 | 0.0100 |

## Key Findings

1. **Best linear accuracy**: `model2_lr` at **58.7%** (AUC 0.6194) — logistic regression on all features
2. **GBM strict**: **60.6%** accuracy (AUC 0.6520), using all features on bars with any non-zero move
3. **GBM threshold**: **60.6%** accuracy (AUC 0.6520), restricting to |move| ≥ 0.005 — lower than strict, suggesting threshold filter does not help
4. **GBM top features**: `tod_sin`, `tod_cos`, `price_previous` — time-of-day signals dominate

## Limitations

- Flat bars (~35%) excluded — live trading must also handle the no-move case
- No position sizing: accuracy does not account for magnitude of wrong predictions
- Bid/ask spread (≈1¢–2¢) means directional accuracy must be sustained to be profitable
- Same-day only; no cross-day regime features
- Model may overfit to time-of-day patterns specific to summer 2025
