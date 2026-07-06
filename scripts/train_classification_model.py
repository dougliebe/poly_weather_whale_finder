"""
train_classification_model.py
────────────────────────────────────────────────────────────────────────────
Three-class classification: UP / NEUTRAL / DOWN

Label construction
------------------
  UP      move > +0.05  (ask rose more than 5¢ over 30 min)
  NEUTRAL |move| <= 0.05  (within ±5¢ — economically insignificant)
  DOWN    move < -0.05  (ask fell more than 5¢ over 30 min)

  All bars kept. In live trading we cannot know the future outcome, so the
  model must handle all three cases on every bar.

Loss function
-------------
  Negative log loss (categorical cross-entropy):
    • LogisticRegression: multinomial by default in sklearn
    • LightGBM: objective='multiclass', num_class=3, metric='multi_logloss'

  Naive baseline: always predict the class prior (uniform → log loss = log(3) ≈ 1.099,
  or prior-weighted → log loss based on actual class frequencies).

Models
------
  baseline_lr  — Logistic Regression, yes_ask_close only
  model1_lr    — + momentum
  model2_lr    — + cross-ticker
  model3_lr    — + time-of-day
  model4_lr    — + volume burst signals
  model5_lr    — + METAR timing + time-to-resolution (all features)
  model6_gbm   — LightGBM multiclass, all features

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
    accuracy_score, log_loss,
    confusion_matrix, classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TRAIN_END  = "2025-08-31"
TEST_START = "2025-09-01"
TARGET_COL = "target_move_90"

# Class labels
DOWN, NEUTRAL, UP = 0, 1, 2
CLASS_NAMES = {DOWN: "down", NEUTRAL: "neutral", UP: "up"}

# ── feature groups ────────────────────────────────────────────────────────────
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


# ── label construction ────────────────────────────────────────────────────────

NEUTRAL_THRESHOLD = 0.05  # ±5¢ — moves within this band are labelled NEUTRAL

def make_labels(move: pd.Series) -> np.ndarray:
    labels = np.full(len(move), NEUTRAL, dtype=int)
    labels[move >  NEUTRAL_THRESHOLD] = UP
    labels[move < -NEUTRAL_THRESHOLD] = DOWN
    return labels


# ── model factories ────────────────────────────────────────────────────────────

def make_lr_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=1.0,
        )),
    ])


def make_gbm_pipeline() -> Pipeline:
    try:
        import lightgbm as lgb
        estimator = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            metric="multi_logloss",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",   # handles unequal class frequencies
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        log.warning("lightgbm not installed — falling back to LogisticRegression")
        estimator = LogisticRegression(max_iter=1000)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   estimator),
    ])


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray,
             y_prob: np.ndarray) -> dict:
    ll      = log_loss(y_true, y_prob, labels=[DOWN, NEUTRAL, UP])
    acc     = accuracy_score(y_true, y_pred)

    # naive baseline: predict class priors on every bar
    priors  = np.bincount(y_true, minlength=3) / len(y_true)
    naive_prob = np.tile(priors, (len(y_true), 1))
    naive_ll   = log_loss(y_true, naive_prob, labels=[DOWN, NEUTRAL, UP])
    ll_improv  = (naive_ll - ll) / naive_ll * 100

    # majority-class accuracy baseline
    majority_acc = priors.max()

    # per-class precision / recall from the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[DOWN, NEUTRAL, UP])
    per_class = {}
    for i, cls in enumerate([DOWN, NEUTRAL, UP]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        per_class[CLASS_NAMES[cls]] = {
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "recall":    tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        }

    return {
        "model": name,
        "log_loss": ll, "naive_log_loss": naive_ll,
        "ll_improvement_pct": ll_improv,
        "accuracy": acc, "majority_baseline": majority_acc,
        "vs_majority_pp": (acc - majority_acc) * 100,
        "priors": priors.tolist(),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


# ── feature importance ────────────────────────────────────────────────────────

def lr_importance(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Average absolute coefficient across the 3 OvR planes, normalised."""
    coef = pipeline.named_steps["model"].coef_   # shape (3, n_features)
    avg_abs = np.abs(coef).mean(axis=0)
    norm = avg_abs / avg_abs.sum()
    rows = []
    for i, feat in enumerate(feature_names):
        rows.append({
            "feature":      feat,
            "importance":   norm[i],
            "coef_down":    coef[DOWN, i],
            "coef_neutral": coef[NEUTRAL, i],
            "coef_up":      coef[UP, i],
            "category":     FEATURE_CATEGORIES.get(feat, "other"),
        })
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


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


# ── report ────────────────────────────────────────────────────────────────────

def ascii_bar(value: float, max_value: float, width: int = 25) -> str:
    filled = int(round(value / max_value * width)) if max_value > 0 else 0
    return "█" * filled + "░" * (width - filled)


def write_report(
    metrics: list[dict],
    lr_imp: pd.DataFrame,
    gbm_imp: pd.DataFrame,
    ticker_ll: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    out_path: pathlib.Path,
) -> None:
    lines = []

    def h(t):    lines.append(f"\n## {t}\n")
    def h3(t):   lines.append(f"\n### {t}\n")
    def p(t=""): lines.append(t)

    lines.append("# Kalshi Weather Market: 3-Class Classification (UP / NEUTRAL / DOWN)")
    lines.append(f"\n_Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_\n")
    lines.append(
        f"**Train**: {df_train['trade_date'].min()} – {df_train['trade_date'].max()} "
        f"({len(df_train):,} bars)  "
        f"**Test**: {df_test['trade_date'].min()} – {df_test['trade_date'].max()} "
        f"({len(df_test):,} bars)"
    )

    h("Methodology")
    priors = metrics[0]["priors"]
    p(f"- **Classes**: DOWN (`move < -0.05`, {priors[DOWN]:.1%}), "
      f"NEUTRAL (`|move| ≤ 0.05`, {priors[NEUTRAL]:.1%}), "
      f"UP (`move > +0.05`, {priors[UP]:.1%})")
    p("- **All bars included** — model predicts on every bar as in live trading")
    p("- **Loss**: negative log loss (categorical cross-entropy), lower = better")
    p(f"- **Naive baseline**: predict class priors on every bar → "
      f"log loss = {metrics[0]['naive_log_loss']:.4f}")
    p("- **LightGBM**: `objective='multiclass'`, `num_class=3`, `class_weight='balanced'`")
    p("- **Logistic Regression**: `multi_class='multinomial'`, `solver='lbfgs'`")

    h("Model Comparison")
    p(f"Naive log loss: **{metrics[0]['naive_log_loss']:.4f}**  |  "
      f"Majority-class accuracy baseline: **{metrics[0]['majority_baseline']:.1%}** (always predict NEUTRAL)\n")
    p("| Model | Log Loss | vs Naive | Accuracy | vs Majority | "
      "Prec↑up | Rec↑up | Prec↓down | Rec↓down |")
    p("|---|---|---|---|---|---|---|---|---|")
    for m in metrics:
        sign = "+" if m["ll_improvement_pct"] >= 0 else ""
        acc_sign = "+" if m["vs_majority_pp"] >= 0 else ""
        up   = m["per_class"]["up"]
        down = m["per_class"]["down"]
        p(f"| {m['model']} "
          f"| **{m['log_loss']:.4f}** "
          f"| {sign}{m['ll_improvement_pct']:.1f}% "
          f"| {m['accuracy']:.1%} "
          f"| {acc_sign}{m['vs_majority_pp']:.1f}pp "
          f"| {up['precision']:.1%} | {up['recall']:.1%} "
          f"| {down['precision']:.1%} | {down['recall']:.1%} |")

    # Confusion matrix for best model
    best = min(metrics, key=lambda m: m["log_loss"])
    h3(f"Confusion Matrix — {best['model']}")
    cm = best["confusion_matrix"]
    p("Rows = actual, columns = predicted (DOWN / NEUTRAL / UP)\n")
    p("| Actual \\ Predicted | DOWN | NEUTRAL | UP |")
    p("|---|---|---|---|")
    for i, cls in enumerate(["DOWN", "NEUTRAL", "UP"]):
        row_total = sum(cm[i])
        p(f"| **{cls}** | {cm[i][0]:,} ({cm[i][0]/row_total:.0%}) "
          f"| {cm[i][1]:,} ({cm[i][1]/row_total:.0%}) "
          f"| {cm[i][2]:,} ({cm[i][2]/row_total:.0%}) |")

    h("Feature Importance: Logistic Regression (model5_lr — up_vs_rest)")
    p("Mean absolute coefficient across DOWN / NEUTRAL / UP planes, normalised. "
      "Also shows per-class coefficient direction.\n")
    p("| Rank | Feature | Importance % | Coef↓down | Coef~neutral | Coef↑up | Category |")
    p("|---|---|---|---|---|---|---|")
    for i, row in lr_imp.head(20).iterrows():
        p(f"| {i+1} | `{row['feature']}` | {row['importance']*100:.2f}% "
          f"| {row['coef_down']:+.3f} | {row['coef_neutral']:+.3f} "
          f"| {row['coef_up']:+.3f} | {row['category']} |")

    h3("Category Summary (Logistic Regression)")
    cat_sum = lr_imp.groupby("category")["importance"].sum().sort_values(ascending=False)
    p("| Category | Total % |")
    p("|---|---|")
    for cat, imp in cat_sum.items():
        p(f"| {cat} | {imp*100:.1f}% |")

    h("Feature Importance: LightGBM (model6_gbm)")
    p("Gain importance (total log-loss reduction from splits).\n")
    max_imp = gbm_imp["importance"].max()
    p("| Rank | Feature | Gain | Share | Category | Bar |")
    p("|---|---|---|---|---|---|")
    for i, row in gbm_imp.head(20).iterrows():
        bar = ascii_bar(row["importance"], max_imp)
        p(f"| {i+1} | `{row['feature']}` | {row['importance']:,.0f} "
          f"| {row['importance_pct']:.1f}% | {row['category']} | `{bar}` |")

    h3("Category Summary (GBM)")
    cat_gbm = gbm_imp.groupby("category")["importance_pct"].sum().sort_values(ascending=False)
    p("| Category | Total Gain % |")
    p("|---|---|")
    for cat, imp in cat_gbm.items():
        p(f"| {cat} | {imp:.1f}% |")

    h("Per-Ticker Log Loss (GBM, Test Set)")
    p("Best 30 tickers by log loss. High-flat tickers score trivially well.\n")
    p("| Ticker | Log Loss | Accuracy | N Rows | % Flat |")
    p("|---|---|---|---|---|")
    for _, row in ticker_ll.dropna(subset=["log_loss"]).sort_values("log_loss").head(30).iterrows():
        p(f"| `{row['ticker']}` | {row['log_loss']:.4f} "
          f"| {row['accuracy']:.1%} | {int(row['n_rows']):,} | {row['pct_flat']:.1%} |")

    h("Key Findings")
    gbm_m   = next(m for m in metrics if "gbm" in m["model"])
    best_lr = min((m for m in metrics if "lr" in m["model"]), key=lambda m: m["log_loss"])
    top3    = gbm_imp.head(3)["feature"].tolist()

    p(f"1. **Naive baseline**: log loss {metrics[0]['naive_log_loss']:.4f} "
      f"(predicting class priors {priors[DOWN]:.0%}/{priors[NEUTRAL]:.0%}/{priors[UP]:.0%} always)")
    p(f"2. **Best LR**: `{best_lr['model']}` — log loss {best_lr['log_loss']:.4f} "
      f"({best_lr['ll_improvement_pct']:+.1f}% vs naive), accuracy {best_lr['accuracy']:.1%}")
    p(f"3. **GBM**: log loss {gbm_m['log_loss']:.4f} "
      f"({gbm_m['ll_improvement_pct']:+.1f}% vs naive), accuracy {gbm_m['accuracy']:.1%} "
      f"({gbm_m['vs_majority_pp']:+.1f}pp vs majority)")
    p(f"4. **Top GBM features**: {', '.join('`'+f+'`' for f in top3)}")
    p(f"5. **UP recall**: GBM {gbm_m['per_class']['up']['recall']:.1%} "
      f"vs LR {best_lr['per_class']['up']['recall']:.1%}  |  "
      f"**DOWN recall**: GBM {gbm_m['per_class']['down']['recall']:.1%} "
      f"vs LR {best_lr['per_class']['down']['recall']:.1%}")

    h("Limitations")
    p("- NEUTRAL class (~35%) is structurally unpredictable from prior bars alone; "
      "it sets a log loss floor no model can escape")
    p("- Three-class log loss penalises confident wrong predictions heavily — "
      "a model that says 90% UP when the outcome is DOWN is punished severely")
    p("- Precision on UP/DOWN is what matters for trading; filter on high predicted "
      "probability (e.g. P(UP) > 0.45) to find actionable bars")
    p("- No transaction cost model — bid/ask spread must be overcome for profitability")

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

    y_train = make_labels(train[TARGET_COL])
    y_test  = make_labels(test[TARGET_COL])
    log.info("Train class distribution: down=%.1f%%  neutral=%.1f%%  up=%.1f%%",
             (y_train == DOWN).mean()*100,
             (y_train == NEUTRAL).mean()*100,
             (y_train == UP).mean()*100)
    log.info("Test  class distribution: down=%.1f%%  neutral=%.1f%%  up=%.1f%%",
             (y_test == DOWN).mean()*100,
             (y_test == NEUTRAL).mean()*100,
             (y_test == UP).mean()*100)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict] = []
    gbm_imp_df  = None
    lr_imp_df   = None

    # ── LR models ─────────────────────────────────────────────────────────────
    for model_name, features in FEATURE_SETS.items():
        features = [f for f in features if f in df.columns]
        log.info("Training %s (%d features)...", model_name, len(features))

        pipe = make_lr_pipeline()
        pipe.fit(train[features].values, y_train)
        y_pred = pipe.predict(test[features].values)
        y_prob = pipe.predict_proba(test[features].values)  # (n, 3) — order: 0,1,2 = DOWN,NEUTRAL,UP

        m = evaluate(model_name, y_test, y_pred, y_prob)
        all_metrics.append(m)
        log.info("  LogLoss=%.4f (%+.1f%% vs naive)  Acc=%.1f%%",
                 m["log_loss"], m["ll_improvement_pct"], m["accuracy"]*100)

        with open(args.models_dir / f"clf3_{model_name}.pkl", "wb") as fh:
            pickle.dump({"pipeline": pipe, "features": features, "metrics": m}, fh)

    # Feature importance from best LR
    best_lr_name = min(
        ((m["model"], m["log_loss"]) for m in all_metrics if "lr" in m["model"]),
        key=lambda x: x[1]
    )[0]
    with open(args.models_dir / f"clf3_{best_lr_name}.pkl", "rb") as fh:
        d = pickle.load(fh)
    lr_imp_df = lr_importance(d["pipeline"], d["features"])

    # ── GBM ───────────────────────────────────────────────────────────────────
    gbm_features = [f for f in ALL_FEATURES if f in df.columns]
    log.info("Training model6_gbm (%d features)...", len(gbm_features))

    gbm_pipe = make_gbm_pipeline()
    gbm_pipe.fit(train[gbm_features].values, y_train)
    y_pred_gbm = gbm_pipe.predict(test[gbm_features].values)
    y_prob_gbm = gbm_pipe.predict_proba(test[gbm_features].values)

    m_gbm = evaluate("model6_gbm", y_test, y_pred_gbm, y_prob_gbm)
    all_metrics.append(m_gbm)
    log.info("  LogLoss=%.4f (%+.1f%% vs naive)  Acc=%.1f%%",
             m_gbm["log_loss"], m_gbm["ll_improvement_pct"], m_gbm["accuracy"]*100)

    with open(args.models_dir / "clf3_model6_gbm.pkl", "wb") as fh:
        pickle.dump({"pipeline": gbm_pipe, "features": gbm_features, "metrics": m_gbm}, fh)

    gbm_imp_df = gbm_importance(gbm_pipe, gbm_features)

    # ── per-ticker log loss ────────────────────────────────────────────────────
    test2 = test.copy()
    test2["y_true"] = y_test
    test2["y_pred"] = y_pred_gbm
    for cls in [DOWN, NEUTRAL, UP]:
        test2[f"y_prob_{cls}"] = y_prob_gbm[:, cls]

    ticker_rows = []
    for ticker, grp in test2.groupby("ticker"):
        probs = grp[[f"y_prob_{c}" for c in [DOWN, NEUTRAL, UP]]].values
        # need all 3 classes present for log_loss; skip degenerate tickers
        if grp["y_true"].nunique() < 3:
            ll = float("nan")
        else:
            ll = log_loss(grp["y_true"], probs, labels=[DOWN, NEUTRAL, UP])
        ticker_rows.append({
            "ticker":   ticker,
            "log_loss": ll,
            "accuracy": accuracy_score(grp["y_true"], grp["y_pred"]),
            "n_rows":   len(grp),
            "pct_flat": (df.loc[grp.index, TARGET_COL] == 0).mean(),
        })
    ticker_ll_df = pd.DataFrame(ticker_rows)

    write_report(
        metrics=all_metrics,
        lr_imp=lr_imp_df,
        gbm_imp=gbm_imp_df,
        ticker_ll=ticker_ll_df,
        df_train=train,
        df_test=test,
        out_path=args.report,
    )
    log.info("Done.")


if __name__ == "__main__":
    main()
