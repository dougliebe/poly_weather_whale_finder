"""
train_classification_model.py
────────────────────────────────────────────────────────────────────────────
Trains binary classifiers (log loss objective) to predict whether
yes_ask_open will be HIGHER 30 minutes from now vs flat-or-lower,
and separately whether it will be LOWER vs flat-or-higher.

Label construction
------------------
  All bars kept — in live trading we cannot filter by future outcome.
    up_vs_rest   1 if move > 0, else 0  (~32% positive)
    down_vs_rest 1 if move < 0, else 0  (~33% positive)

Loss function
-------------
  Logistic Regression: log loss by default (sklearn).
  LightGBM: objective='binary' (binary cross-entropy / log loss) with
            is_unbalance=True to handle the ~68/32 class split without
            manual sample weighting.

  Log loss measures calibration quality of predicted probabilities, not
  just hard-threshold accuracy — a lower log loss means the model's
  confidence scores are more trustworthy for sizing trades.

Models
------
  baseline_lr  — Logistic Regression, yes_ask_close only
  model1_lr    — + momentum (lags, spread, OI, volume, minute_of_hour)
  model2_lr    — + cross-ticker probability conservation
  model3_lr    — + time-of-day (tod_sin/cos, bar_hour)
  model4_lr    — + volume burst signals (z-score, ratio, OI accel)
  model5_lr    — + METAR timing + time-to-resolution
  model6_gbm   — LightGBM (binary logloss), all features, up_vs_rest
  model7_gbm_d — LightGBM (binary logloss), all features, down_vs_rest

Usage
-----
    python scripts/train_classification_model.py
"""

import argparse
import datetime
import logging
import pathlib
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TRAIN_END  = "2025-08-31"
TEST_START = "2025-09-01"
TARGET_COL = "target_move_30"

# ── feature groups (mirrors train_price_model.py) ─────────────────────────────
_MOMENTUM = [
    "yes_ask_close", "mid_close", "spread",
    "mid_ret_1", "mid_ret_5", "mid_ret_15", "mid_ret_30",
    "ask_lag1", "ask_lag5", "ask_lag15",
    "ask_intrabar_range", "bid_intrabar_range",
    "spread_lag1", "spread_lag5",
    "has_trade", "vol_sum_5", "vol_sum_15",
    "oi_change_1", "oi_change_5", "oi_change_15",
    "mid_vol_15", "price_previous",
    "minute_of_hour",
]
_CROSS = [
    "sum_mid_all", "rel_mid", "prob_sum_deviation",
    "relative_spread", "avg_spread_all", "spread_dispersion",
    "n_tickers_at_bar", "total_vol_cross", "vol_vs_cross",
]
_TOD = [
    "bar_hour", "minute_of_hour", "tod_sin", "tod_cos",
]
_BURST = [
    "vol_zscore", "vol_ratio", "vol_burst_flag",
    "oi_accel", "spread_chg_5",
]
_RESOLUTION = [
    "mins_to_peak", "frac_day_elapsed",
    "price_polarization", "polar_chg_5",
]
_METAR = [
    "bars_since_metar", "minutes_to_next_metar",
    "is_pre_metar", "is_post_metar",
]

FEATURE_SETS = {
    "baseline_lr": ["yes_ask_close"],
    "model1_lr":   _MOMENTUM,
    "model2_lr":   _MOMENTUM + _CROSS,
    "model3_lr":   _MOMENTUM + _CROSS + _TOD,
    "model4_lr":   _MOMENTUM + _CROSS + _TOD + _BURST,
    "model5_lr":   _MOMENTUM + _CROSS + _TOD + _BURST + _RESOLUTION + _METAR,
}

ALL_FEATURES = list(dict.fromkeys(
    _MOMENTUM + _CROSS + _TOD + _BURST + _RESOLUTION + _METAR
))

FEATURE_CATEGORIES = {
    "yes_ask_close": "anchor", "mid_close": "anchor", "price_previous": "anchor",
    "mid_ret_1": "momentum", "mid_ret_5": "momentum",
    "mid_ret_15": "momentum", "mid_ret_30": "momentum",
    "ask_lag1": "momentum", "ask_lag5": "momentum", "ask_lag15": "momentum",
    "spread": "spread", "spread_lag1": "spread", "spread_lag5": "spread",
    "spread_chg_5": "spread",
    "ask_intrabar_range": "volatility", "bid_intrabar_range": "volatility",
    "mid_vol_15": "volatility",
    "has_trade": "flow", "vol_sum_5": "flow", "vol_sum_15": "flow",
    "oi_change_1": "flow", "oi_change_5": "flow", "oi_change_15": "flow",
    "vol_zscore": "burst", "vol_ratio": "burst",
    "vol_burst_flag": "burst", "oi_accel": "burst",
    "sum_mid_all": "cross", "rel_mid": "cross", "prob_sum_deviation": "cross",
    "relative_spread": "cross", "avg_spread_all": "cross",
    "spread_dispersion": "cross", "n_tickers_at_bar": "cross",
    "total_vol_cross": "cross", "vol_vs_cross": "cross",
    "bar_hour": "time", "minute_of_hour": "time",
    "tod_sin": "time", "tod_cos": "time",
    "bars_since_metar": "metar", "minutes_to_next_metar": "metar",
    "is_pre_metar": "metar", "is_post_metar": "metar",
    "mins_to_peak": "resolution", "frac_day_elapsed": "resolution",
    "price_polarization": "resolution", "polar_chg_5": "resolution",
}


# ── model factories ────────────────────────────────────────────────────────────

def make_lr_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        # log loss is the default objective for LogisticRegression
        ("model",   LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")),
    ])


def make_gbm_pipeline() -> Pipeline:
    try:
        import lightgbm as lgb
        estimator = lgb.LGBMClassifier(
            objective="binary",        # binary cross-entropy = log loss
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,         # handles ~68/32 class split
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        log.warning("lightgbm not installed — falling back to LogisticRegression")
        estimator = LogisticRegression(max_iter=1000, C=1.0)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   estimator),
    ])


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(name: str, y_true, y_pred, y_prob) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)
    ll   = log_loss(y_true, y_prob)
    # naive log loss: always predict the base rate
    base_rate = float(y_true.mean())
    naive_prob = np.full_like(y_prob, base_rate)
    naive_ll   = log_loss(y_true, naive_prob)
    ll_improvement_pct = (naive_ll - ll) / naive_ll * 100

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    majority  = max(base_rate, 1 - base_rate)

    return {
        "model": name,
        "log_loss": ll, "naive_log_loss": naive_ll,
        "ll_improvement_pct": ll_improvement_pct,
        "accuracy": acc, "majority_baseline": majority,
        "vs_majority_pp": (acc - majority) * 100,
        "auc": auc,
        "precision": precision, "recall": recall,
        "pct_positive": base_rate,
    }


# ── feature importance ────────────────────────────────────────────────────────

def lr_importance(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    coefs = pipeline.named_steps["model"].coef_[0]
    abs_coefs = np.abs(coefs)
    norm = abs_coefs / abs_coefs.sum() if abs_coefs.sum() > 0 else abs_coefs
    return pd.DataFrame({
        "feature":    feature_names,
        "coef":       coefs,
        "importance": norm,
        "category":   [FEATURE_CATEGORIES.get(f, "other") for f in feature_names],
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def gbm_importance(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return lr_importance(pipeline, feature_names)
    importances = model.feature_importances_
    total = importances.sum()
    return pd.DataFrame({
        "feature":        feature_names,
        "importance":     importances,
        "importance_pct": importances / total * 100 if total > 0 else importances,
        "category":       [FEATURE_CATEGORIES.get(f, "other") for f in feature_names],
    }).sort_values("importance", ascending=False).reset_index(drop=True)


# ── report generation ─────────────────────────────────────────────────────────

def ascii_bar(value: float, max_value: float, width: int = 25) -> str:
    filled = int(round(value / max_value * width)) if max_value > 0 else 0
    return "█" * filled + "░" * (width - filled)


def write_report(
    metrics_up: list[dict],
    metrics_down: list[dict],
    lr_imp: pd.DataFrame,
    gbm_imp: pd.DataFrame,
    ticker_acc: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    out_path: pathlib.Path,
) -> None:
    lines = []

    def h(text):    lines.append(f"\n## {text}\n")
    def h3(text):   lines.append(f"\n### {text}\n")
    def p(text=""):  lines.append(text)

    lines.append("# Kalshi Weather Market: Classification Report (Log Loss)")
    lines.append(f"\n_Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_\n")
    lines.append(
        f"**Train**: {df_train['trade_date'].min()} – {df_train['trade_date'].max()} "
        f"({len(df_train):,} bars)  "
        f"**Test**: {df_test['trade_date'].min()} – {df_test['trade_date'].max()} "
        f"({len(df_test):,} bars)"
    )

    h("Methodology")
    p("- **All bars evaluated** — flat-move bars (~35%) score as negative class; "
      "matches live trading where future outcome is unknown")
    p("- **Primary metric: log loss** — measures calibration of predicted probabilities. "
      "Lower is better; 0 = perfect. Naive baseline = always predict the class prior.")
    p("- **Label (up_vs_rest)**: 1 if `target_move_30 > 0`, else 0 (~32% positive)")
    p("- **Label (down_vs_rest)**: 1 if `target_move_30 < 0`, else 0 (~33% positive)")
    p("- **LightGBM**: `objective='binary'` (binary cross-entropy), `is_unbalance=True`")
    p("- **Logistic Regression**: log loss is the default objective")
    p("- **Train**: 2025-06-01 – 2025-08-31 | **Test**: 2025-09-01 – 2025-09-30")

    for direction, metrics in [("UP (move > 0) vs Flat-or-Down", metrics_up),
                                ("DOWN (move < 0) vs Flat-or-Up", metrics_down)]:
        h(f"Results — Predict {direction}")
        naive_ll = metrics[0]["naive_log_loss"]
        majority = metrics[0]["majority_baseline"]
        pct_pos  = metrics[0]["pct_positive"]
        p(f"Positive rate: {pct_pos:.1%}  |  "
          f"Majority-class accuracy baseline: {majority:.1%}  |  "
          f"Naive log loss (predict prior): **{naive_ll:.4f}**\n")
        p("| Model | Log Loss | vs Naive | AUC | Accuracy | vs Maj | Precision | Recall |")
        p("|---|---|---|---|---|---|---|---|")
        for m in metrics:
            sign = "+" if m["ll_improvement_pct"] >= 0 else ""
            acc_sign = "+" if m["vs_majority_pp"] >= 0 else ""
            p(f"| {m['model']} "
              f"| **{m['log_loss']:.4f}** "
              f"| {sign}{m['ll_improvement_pct']:.1f}% "
              f"| {m['auc']:.4f} "
              f"| {m['accuracy']:.1%} "
              f"| {acc_sign}{m['vs_majority_pp']:.1f}pp "
              f"| {m['precision']:.1%} "
              f"| {m['recall']:.1%} |")

    h("Feature Importance: Logistic Regression (model5_lr — all features, up_vs_rest)")
    p("Coefficients normalized to % of total absolute weight after StandardScaler.\n")
    p("| Rank | Feature | Importance % | Coef | Category |")
    p("|---|---|---|---|---|")
    for i, row in lr_imp.head(20).iterrows():
        p(f"| {i+1} | `{row['feature']}` | {row['importance']*100:.2f}% | "
          f"{row['coef']:+.4f} | {row['category']} |")

    h3("Category Summary (Logistic Regression)")
    cat_sum = lr_imp.groupby("category")["importance"].sum().sort_values(ascending=False)
    p("| Category | Total % |")
    p("|---|---|")
    for cat, imp in cat_sum.items():
        p(f"| {cat} | {imp*100:.1f}% |")

    h("Feature Importance: LightGBM (model6_gbm — up_vs_rest)")
    p("Gain importance (total log-loss reduction from splits on this feature).\n")
    max_imp = gbm_imp["importance"].max()
    p("| Rank | Feature | Gain | Share | Category | Bar |")
    p("|---|---|---|---|---|---|")
    for i, row in gbm_imp.head(20).iterrows():
        bar = ascii_bar(row["importance"], max_imp)
        p(f"| {i+1} | `{row['feature']}` | {row['importance']:,.0f} | "
          f"{row['importance_pct']:.1f}% | {row['category']} | `{bar}` |")

    h3("Category Summary (GBM)")
    cat_gbm = gbm_imp.groupby("category")["importance_pct"].sum().sort_values(ascending=False)
    p("| Category | Total Gain % |")
    p("|---|---|")
    for cat, imp in cat_gbm.items():
        p(f"| {cat} | {imp:.1f}% |")

    h("Per-Ticker Log Loss (GBM up_vs_rest, Test Set)")
    p("Sorted by log loss ascending (best first). "
      "High flat% tickers trivially predict the majority class.\n")
    p("| Ticker | Log Loss | Accuracy | N Rows | % Flat |")
    p("|---|---|---|---|---|")
    for _, row in ticker_acc.sort_values("log_loss").head(30).iterrows():
        p(f"| `{row['ticker']}` | {row['log_loss']:.4f} | "
          f"{row['accuracy']:.1%} | {int(row['n_rows']):,} | {row['pct_flat']:.1%} |")

    h("Key Findings")
    gbm_up   = next(m for m in metrics_up   if "gbm" in m["model"])
    gbm_down = next(m for m in metrics_down if "gbm" in m["model"])
    best_lr  = min((m for m in metrics_up if "lr" in m["model"]), key=lambda m: m["log_loss"])
    top3     = gbm_imp.head(3)["feature"].tolist()

    p(f"1. **Naive log loss baseline**: {metrics_up[0]['naive_log_loss']:.4f} "
      f"(always predict {metrics_up[0]['pct_positive']:.1%} probability of up-move)")
    p(f"2. **Best LR model**: `{best_lr['model']}` — log loss {best_lr['log_loss']:.4f} "
      f"({best_lr['ll_improvement_pct']:+.1f}% vs naive), AUC {best_lr['auc']:.4f}")
    p(f"3. **GBM up_vs_rest**: log loss {gbm_up['log_loss']:.4f} "
      f"({gbm_up['ll_improvement_pct']:+.1f}% vs naive), AUC {gbm_up['auc']:.4f}, "
      f"accuracy {gbm_up['accuracy']:.1%} ({gbm_up['vs_majority_pp']:+.1f}pp vs majority)")
    p(f"4. **GBM down_vs_rest**: log loss {gbm_down['log_loss']:.4f} "
      f"({gbm_down['ll_improvement_pct']:+.1f}% vs naive), AUC {gbm_down['auc']:.4f}")
    p(f"5. **Top GBM features**: {', '.join('`'+f+'`' for f in top3)}")

    cat_gbm_sorted = gbm_imp.groupby("category")["importance_pct"].sum().sort_values(ascending=False)
    top_cats = list(cat_gbm_sorted.index[:3])
    p(f"6. **Dominant feature categories** (GBM gain): "
      f"{', '.join(top_cats)} — "
      f"combined {cat_gbm_sorted[top_cats].sum():.1f}% of total gain")

    h("Limitations")
    p("- Flat bars (~35%) are structural noise: any directional prediction is wrong for them, "
      "setting a floor on log loss that no binary model can escape")
    p("- `is_unbalance=True` in LightGBM reweights classes but does not change the label space; "
      "precision/recall tradeoffs still apply")
    p("- Predicted probabilities near 0.5 carry little trading value; "
      "consider filtering to high-confidence predictions (e.g. prob > 0.60) for live use")
    p("- No transaction cost model — bid/ask spread (~1¢–2¢) must be overcome")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    log.info("Report written → %s", out_path)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features",   default="data/features_candles.parquet", type=pathlib.Path)
    ap.add_argument("--models-dir", default="models",                         type=pathlib.Path)
    ap.add_argument("--report",     default="reports/classification_report.md", type=pathlib.Path)
    args = ap.parse_args()

    log.info("Loading features from %s", args.features)
    df = pd.read_parquet(args.features)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date.astype(str)

    train = df[df["trade_date"] <= TRAIN_END].copy()
    test  = df[df["trade_date"] >= TEST_START].copy()
    log.info("Train: %d rows  Test: %d rows", len(train), len(test))

    args.models_dir.mkdir(parents=True, exist_ok=True)

    all_metrics_up:   list[dict] = []
    all_metrics_down: list[dict] = []
    gbm_imp_df    = None
    lr_imp_df     = None
    ticker_acc_df = None

    for direction in ("up", "down"):
        log.info("=== Direction: %s ===", direction)

        y_train = (train[TARGET_COL] > 0 if direction == "up"
                   else train[TARGET_COL] < 0).astype(int).values
        y_test  = (test[TARGET_COL]  > 0 if direction == "up"
                   else test[TARGET_COL]  < 0).astype(int).values

        log.info("  Train positive=%.1f%%  Test positive=%.1f%%",
                 y_train.mean()*100, y_test.mean()*100)

        # LR models — only for "up" direction (symmetric story, avoids doubling output)
        if direction == "up":
            for model_name, features in FEATURE_SETS.items():
                features = [f for f in features if f in df.columns]
                log.info("  Training %s (%d features)...", model_name, len(features))

                pipe = make_lr_pipeline()
                pipe.fit(train[features].values, y_train)
                y_pred = pipe.predict(test[features].values)
                y_prob = pipe.predict_proba(test[features].values)[:, 1]

                m = evaluate(model_name, y_test, y_pred, y_prob)
                all_metrics_up.append(m)
                log.info("    LogLoss=%.4f (%+.1f%% vs naive)  AUC=%.4f  Acc=%.1f%%",
                         m["log_loss"], m["ll_improvement_pct"], m["auc"], m["accuracy"]*100)

                with open(args.models_dir / f"clf_{model_name}.pkl", "wb") as fh:
                    pickle.dump({"pipeline": pipe, "features": features, "metrics": m}, fh)

            # Best LR for importance
            best_lr_name = min(
                ((m["model"], m["log_loss"]) for m in all_metrics_up if "lr" in m["model"]),
                key=lambda x: x[1]
            )[0]
            with open(args.models_dir / f"clf_{best_lr_name}.pkl", "rb") as fh:
                d = pickle.load(fh)
            lr_imp_df = lr_importance(d["pipeline"], d["features"])

        # GBM
        gbm_features = [f for f in ALL_FEATURES if f in df.columns]
        gbm_name = f"model6_gbm{'_down' if direction == 'down' else ''}"
        log.info("  Training %s (%d features)...", gbm_name, len(gbm_features))

        gbm_pipe = make_gbm_pipeline()
        gbm_pipe.fit(train[gbm_features].values, y_train)
        y_pred_gbm = gbm_pipe.predict(test[gbm_features].values)
        y_prob_gbm = gbm_pipe.predict_proba(test[gbm_features].values)[:, 1]

        m_gbm = evaluate(gbm_name, y_test, y_pred_gbm, y_prob_gbm)
        log.info("    LogLoss=%.4f (%+.1f%% vs naive)  AUC=%.4f  Acc=%.1f%%",
                 m_gbm["log_loss"], m_gbm["ll_improvement_pct"],
                 m_gbm["auc"], m_gbm["accuracy"]*100)

        with open(args.models_dir / f"clf_{gbm_name}.pkl", "wb") as fh:
            pickle.dump({"pipeline": gbm_pipe, "features": gbm_features, "metrics": m_gbm}, fh)

        if direction == "up":
            all_metrics_up.append(m_gbm)
            gbm_imp_df = gbm_importance(gbm_pipe, gbm_features)

            # per-ticker log loss
            test2 = test.copy()
            test2["y_true"] = y_test
            test2["y_prob"] = y_prob_gbm
            test2["y_pred"] = y_pred_gbm
            rows = []
            for ticker, grp in test2.groupby("ticker"):
                # log_loss requires at least one positive and one negative sample
                if grp["y_true"].nunique() < 2:
                    ll = float("nan")
                else:
                    ll = log_loss(grp["y_true"], grp["y_prob"])
                rows.append({
                    "ticker":   ticker,
                    "log_loss": ll,
                    "accuracy": accuracy_score(grp["y_true"], grp["y_pred"]),
                    "n_rows":   len(grp),
                    "pct_flat": (test.loc[grp.index, TARGET_COL] == 0).mean(),
                })
            ticker_acc_df = pd.DataFrame(rows)
        else:
            all_metrics_down.append(m_gbm)

    write_report(
        metrics_up=all_metrics_up,
        metrics_down=all_metrics_down,
        lr_imp=lr_imp_df,
        gbm_imp=gbm_imp_df,
        ticker_acc=ticker_acc_df,
        df_train=train,
        df_test=test,
        out_path=args.report,
    )
    log.info("Done.")


if __name__ == "__main__":
    main()
