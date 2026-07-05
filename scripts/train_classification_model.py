"""
train_classification_model.py
────────────────────────────────────────────────────────────────────────────
Trains binary classifiers to predict whether yes_ask_open will be HIGHER 30
minutes from now vs flat-or-lower (or lower vs flat-or-higher).

Label construction
------------------
  All bars are kept — in live trading we cannot know in advance whether a bar
  will result in a flat move, so we must predict on every bar.

  Three label variants:
    up_vs_rest   — 1 if move > 0,    else 0  (~32% positive, 68% negative)
    down_vs_rest — 1 if move < 0,    else 0  (~33% positive, 67% negative)
    up_vs_down   — 1 if move > thr,  0 if move < -thr, flat bars excluded
                   (used only for training; test includes ALL bars — flat
                   bars are scored as wrong whichever direction we predict)

  Primary reported variant: up_vs_rest (predict price will rise).

Models
------
  baseline_lr     — Logistic Regression on yes_ask_close only
  model1_lr       — + momentum features
  model2_lr       — + cross-ticker features
  model3_lr       — + time-of-day features
  model4_gbm      — LightGBM on all features, up_vs_rest
  model5_gbm_down — LightGBM predicting down_vs_rest

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
    accuracy_score, roc_auc_score,
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

FEATURE_SETS = {
    "baseline_lr": [
        "yes_ask_close",
    ],
    "model1_lr": [
        "yes_ask_close", "mid_close", "spread",
        "mid_ret_1", "mid_ret_5", "mid_ret_15", "mid_ret_30",
        "ask_lag1", "ask_lag5", "ask_lag15",
        "ask_intrabar_range", "bid_intrabar_range",
        "spread_lag1", "spread_lag5",
        "has_trade", "vol_sum_5", "vol_sum_15",
        "oi_change_1", "oi_change_5", "oi_change_15",
        "mid_vol_15", "price_previous",
        "minute_of_hour",
    ],
    "model2_lr": [
        "yes_ask_close", "mid_close", "spread",
        "mid_ret_1", "mid_ret_5", "mid_ret_15", "mid_ret_30",
        "ask_lag1", "ask_lag5", "ask_lag15",
        "ask_intrabar_range", "bid_intrabar_range",
        "spread_lag1", "spread_lag5",
        "has_trade", "vol_sum_5", "vol_sum_15",
        "oi_change_1", "oi_change_5", "oi_change_15",
        "mid_vol_15", "price_previous",
        "minute_of_hour",
        "sum_mid_all", "rel_mid", "prob_sum_deviation",
        "relative_spread", "avg_spread_all", "spread_dispersion",
        "n_tickers_at_bar",
    ],
    "model3_lr": [
        "yes_ask_close", "mid_close", "spread",
        "mid_ret_1", "mid_ret_5", "mid_ret_15", "mid_ret_30",
        "ask_lag1", "ask_lag5", "ask_lag15",
        "ask_intrabar_range", "bid_intrabar_range",
        "spread_lag1", "spread_lag5",
        "has_trade", "vol_sum_5", "vol_sum_15",
        "oi_change_1", "oi_change_5", "oi_change_15",
        "mid_vol_15", "price_previous",
        "sum_mid_all", "rel_mid", "prob_sum_deviation",
        "relative_spread", "avg_spread_all", "spread_dispersion",
        "n_tickers_at_bar",
        "bar_hour", "minute_of_hour", "tod_sin", "tod_cos",
    ],
}

ALL_FEATURES = FEATURE_SETS["model3_lr"]

FEATURE_CATEGORIES = {
    "yes_ask_close": "anchor",
    "mid_close": "anchor",
    "spread": "spread",
    "ask_intrabar_range": "volatility",
    "bid_intrabar_range": "volatility",
    "mid_ret_1": "momentum",
    "mid_ret_5": "momentum",
    "mid_ret_15": "momentum",
    "mid_ret_30": "momentum",
    "ask_lag1": "momentum",
    "ask_lag5": "momentum",
    "ask_lag15": "momentum",
    "spread_lag1": "spread",
    "spread_lag5": "spread",
    "has_trade": "flow",
    "vol_sum_5": "flow",
    "vol_sum_15": "flow",
    "oi_change_1": "flow",
    "oi_change_5": "flow",
    "oi_change_15": "flow",
    "mid_vol_15": "volatility",
    "price_previous": "anchor",
    "sum_mid_all": "cross",
    "rel_mid": "cross",
    "prob_sum_deviation": "cross",
    "relative_spread": "cross",
    "avg_spread_all": "cross",
    "spread_dispersion": "cross",
    "n_tickers_at_bar": "cross",
    "bar_hour": "time",
    "minute_of_hour": "time",
    "tod_sin": "time",
    "tod_cos": "time",
}


# ── model factories ────────────────────────────────────────────────────────────

def make_lr_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")),
    ])


def make_gbm_pipeline() -> Pipeline:
    try:
        import lightgbm as lgb
        estimator = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
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

def evaluate(name: str, y_true, y_pred, y_prob=None) -> dict:
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    majority_acc = max(y_true.mean(), 1 - y_true.mean())
    return {
        "model": name, "accuracy": acc, "auc": auc,
        "precision": precision, "recall": recall,
        "majority_baseline": majority_acc,
        "vs_majority_pp": (acc - majority_acc) * 100,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "pct_positive": float(y_true.mean()),
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

def ascii_bar(value: float, max_value: float, width: int = 30) -> str:
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

    def h(text):   lines.append(f"\n## {text}\n")
    def h3(text):  lines.append(f"\n### {text}\n")
    def p(text=""): lines.append(text)

    lines.append("# Kalshi Weather Market: Price Direction Classification Report")
    lines.append(f"\n_Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_\n")
    lines.append(
        f"**Train**: {df_train['trade_date'].min()} – {df_train['trade_date'].max()} "
        f"({len(df_train):,} bars)  "
        f"**Test**: {df_test['trade_date'].min()} – {df_test['trade_date'].max()} "
        f"({len(df_test):,} bars)"
    )

    h("Methodology")
    p("- **All bars included in evaluation** — flat-move bars (~35%) are scored as incorrect "
      "for any directional prediction, matching real trading conditions where we cannot "
      "pre-filter bars by their future outcome")
    p("- **Label (up_vs_rest)**: 1 if `target_move_30 > 0`, else 0 (~32% positive)")
    p("- **Label (down_vs_rest)**: 1 if `target_move_30 < 0`, else 0 (~33% positive)")
    p("- **Majority-class baseline**: predict 0 always — equals the negative class frequency")
    p("- **Train**: 2025-06-01 – 2025-08-31 (all bars, no filtering)")
    p("- **Test**: 2025-09-01 – 2025-09-30 (all bars, no filtering)")
    p("- **NaN handling**: median imputation fit on training set only")

    h("Model Comparison — Predict UP (move > 0) vs Flat-or-Down")
    pct_pos = metrics_up[0]["pct_positive"]
    majority = metrics_up[0]["majority_baseline"]
    p(f"Positive class frequency: {pct_pos:.1%}  →  Majority-class baseline: **{majority:.1%}**\n")
    p("| Model | Accuracy | vs Baseline | AUC | Precision | Recall |")
    p("|---|---|---|---|---|---|")
    for m in metrics_up:
        sign = "+" if m["vs_majority_pp"] >= 0 else ""
        p(f"| {m['model']} | **{m['accuracy']:.1%}** | {sign}{m['vs_majority_pp']:.1f}pp | "
          f"{m['auc']:.4f} | {m['precision']:.1%} | {m['recall']:.1%} |")

    h("Model Comparison — Predict DOWN (move < 0) vs Flat-or-Up")
    pct_pos_d = metrics_down[0]["pct_positive"]
    majority_d = metrics_down[0]["majority_baseline"]
    p(f"Positive class frequency: {pct_pos_d:.1%}  →  Majority-class baseline: **{majority_d:.1%}**\n")
    p("| Model | Accuracy | vs Baseline | AUC | Precision | Recall |")
    p("|---|---|---|---|---|---|")
    for m in metrics_down:
        sign = "+" if m["vs_majority_pp"] >= 0 else ""
        p(f"| {m['model']} | **{m['accuracy']:.1%}** | {sign}{m['vs_majority_pp']:.1f}pp | "
          f"{m['auc']:.4f} | {m['precision']:.1%} | {m['recall']:.1%} |")

    h("Feature Importance: Logistic Regression (Model 3 — up_vs_rest)")
    p("Coefficients normalized to % of total absolute weight after StandardScaler.\n")
    p("| Rank | Feature | Importance % | Coef | Category |")
    p("|---|---|---|---|---|")
    for i, row in lr_imp.head(20).iterrows():
        p(f"| {i+1} | `{row['feature']}` | {row['importance']*100:.2f}% | "
          f"{row['coef']:+.4f} | {row['category']} |")

    h3("Category Summary (Logistic Regression)")
    cat_sum = lr_imp.groupby("category")["importance"].sum().sort_values(ascending=False)
    p("| Category | Total Importance % |")
    p("|---|---|")
    for cat, imp in cat_sum.items():
        p(f"| {cat} | {imp*100:.1f}% |")

    h("Feature Importance: LightGBM (up_vs_rest)")
    max_imp = gbm_imp["importance"].max()
    p("| Rank | Feature | Gain | Share | Category | Bar |")
    p("|---|---|---|---|---|---|")
    for i, row in gbm_imp.head(20).iterrows():
        bar = ascii_bar(row["importance"], max_imp, 25)
        p(f"| {i+1} | `{row['feature']}` | {row['importance']:,.0f} | "
          f"{row['importance_pct']:.1f}% | {row['category']} | `{bar}` |")

    h3("Category Summary (GBM)")
    cat_gbm = gbm_imp.groupby("category")["importance_pct"].sum().sort_values(ascending=False)
    p("| Category | Total Gain % |")
    p("|---|---|")
    for cat, imp in cat_gbm.items():
        p(f"| {cat} | {imp:.1f}% |")

    h("Per-Ticker Accuracy (GBM up_vs_rest, Test Set)")
    p("Sorted by accuracy descending.\n")
    p("| Ticker | Accuracy | N Rows | Mean |move| | % Flat |")
    p("|---|---|---|---|---|")
    for _, row in ticker_acc.sort_values("accuracy", ascending=False).iterrows():
        p(f"| `{row['ticker']}` | {row['accuracy']:.1%} | {int(row['n_rows']):,} | "
          f"{row['mean_abs_move']:.4f} | {row['pct_flat']:.1%} |")

    h("Key Findings")
    gbm_up   = next(m for m in metrics_up   if "gbm" in m["model"])
    gbm_down = next(m for m in metrics_down if "gbm" in m["model"])
    best_lr  = max((m for m in metrics_up if "lr" in m["model"]), key=lambda m: m["accuracy"])

    p(f"1. **Realistic baseline**: ~{majority:.0%} accuracy for always predicting "
      f"flat-or-down (majority class). Any model must clear this bar.")
    p(f"2. **Best linear model**: `{best_lr['model']}` at {best_lr['accuracy']:.1%} "
      f"(+{best_lr['vs_majority_pp']:.1f}pp vs baseline, AUC {best_lr['auc']:.4f})")
    p(f"3. **GBM (up_vs_rest)**: {gbm_up['accuracy']:.1%} accuracy "
      f"(+{gbm_up['vs_majority_pp']:.1f}pp vs baseline, AUC {gbm_up['auc']:.4f}). "
      f"Precision {gbm_up['precision']:.1%} means {gbm_up['precision']:.0%} of predicted-up "
      f"bars actually rise — {'above' if gbm_up['precision'] > 0.5 else 'below'} coin-flip.")
    p(f"4. **GBM (down_vs_rest)**: {gbm_down['accuracy']:.1%} accuracy "
      f"(+{gbm_down['vs_majority_pp']:.1f}pp vs baseline). "
      f"Up and down signals are {'symmetric' if abs(gbm_up['auc'] - gbm_down['auc']) < 0.01 else 'asymmetric'}.")
    top3 = gbm_imp.head(3)["feature"].tolist()
    p(f"5. **Top GBM features**: {', '.join('`'+f+'`' for f in top3)}")

    h("Limitations")
    p("- Flat bars (~35% of test) mechanically cap accuracy — even a perfect signal "
      "for non-flat bars yields only ~65% overall accuracy if all flat bars are wrong")
    p("- Precision/recall tradeoff: a threshold on predicted probability can trade "
      "recall for higher-precision signals (fewer but more reliable trades)")
    p("- No transaction cost model — bid/ask spread (~1¢–2¢) must be overcome")
    p("- Model trained on summer 2025; seasonal/regime shift may affect Sep performance")

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
    gbm_imp_df  = None
    lr_imp_df   = None
    ticker_acc_df = None

    for direction in ("up", "down"):
        log.info("=== Direction: %s ===", direction)

        if direction == "up":
            y_train = (train[TARGET_COL] > 0).astype(int).values
            y_test  = (test[TARGET_COL]  > 0).astype(int).values
        else:
            y_train = (train[TARGET_COL] < 0).astype(int).values
            y_test  = (test[TARGET_COL]  < 0).astype(int).values

        log.info("  Train: %d rows  positive=%.1f%%", len(y_train), y_train.mean()*100)
        log.info("  Test:  %d rows  positive=%.1f%%", len(y_test),  y_test.mean()*100)

        # Linear models (only for "up" to keep output concise)
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
                log.info("    Acc=%.1f%%  AUC=%.4f  vs_maj=%+.1fpp",
                         m["accuracy"]*100, m["auc"], m["vs_majority_pp"])

                with open(args.models_dir / f"clf_{model_name}.pkl", "wb") as fh:
                    pickle.dump({"pipeline": pipe, "features": features, "metrics": m}, fh)

            # Feature importance for best LR
            best_lr_name = max(
                ((m["model"], m["accuracy"]) for m in all_metrics_up if "lr" in m["model"]),
                key=lambda x: x[1]
            )[0]
            with open(args.models_dir / f"clf_{best_lr_name}.pkl", "rb") as fh:
                best_lr_data = pickle.load(fh)
            lr_imp_df = lr_importance(best_lr_data["pipeline"], best_lr_data["features"])

        # GBM
        gbm_features = [f for f in ALL_FEATURES if f in df.columns]
        model_name   = f"model4_gbm_{direction}"
        log.info("  Training %s (%d features)...", model_name, len(gbm_features))

        gbm_pipe = make_gbm_pipeline()
        gbm_pipe.fit(train[gbm_features].values, y_train)
        y_pred_gbm = gbm_pipe.predict(test[gbm_features].values)
        y_prob_gbm = gbm_pipe.predict_proba(test[gbm_features].values)[:, 1]

        m_gbm = evaluate(model_name, y_test, y_pred_gbm, y_prob_gbm)
        log.info("    Acc=%.1f%%  AUC=%.4f  vs_maj=%+.1fpp",
                 m_gbm["accuracy"]*100, m_gbm["auc"], m_gbm["vs_majority_pp"])

        if direction == "up":
            all_metrics_up.append(m_gbm)
            gbm_imp_df = gbm_importance(gbm_pipe, gbm_features)

            # per-ticker accuracy
            test2 = test.copy()
            test2["y_pred"] = y_pred_gbm
            test2["y_true"] = y_test
            ticker_rows = []
            for ticker, grp in test2.groupby("ticker"):
                ticker_rows.append({
                    "ticker":        ticker,
                    "accuracy":      accuracy_score(grp["y_true"], grp["y_pred"]),
                    "n_rows":        len(grp),
                    "mean_abs_move": grp[TARGET_COL].abs().mean(),
                    "pct_flat":      (grp[TARGET_COL] == 0).mean(),
                })
            ticker_acc_df = pd.DataFrame(ticker_rows)
        else:
            all_metrics_down.append(m_gbm)

        with open(args.models_dir / f"clf_{model_name}.pkl", "wb") as fh:
            pickle.dump({"pipeline": gbm_pipe, "features": gbm_features, "metrics": m_gbm}, fh)

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
