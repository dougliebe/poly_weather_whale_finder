"""
train_classification_model.py
────────────────────────────────────────────────────────────────────────────
Trains binary classifiers to predict whether yes_ask_open will be HIGHER or
LOWER 30 minutes from now.

Label construction
------------------
  Flat bars (target_move_30 == 0, ~35% of data) are excluded — predicting
  "no change" on zero-move bars inflates accuracy without adding signal.
  Remaining rows: up (move > 0) vs down (move < 0), roughly 50/50.

  A second variant uses a 0.5¢ (0.005) minimum-move threshold to further
  filter microstructure noise.

Models
------
  baseline_lr     — Logistic Regression on yes_ask_close only
  model1_lr       — + momentum features (lags, spread, OI, volume, minute_of_hour)
  model2_lr       — + cross-ticker features
  model3_lr       — + time-of-day features
  model4_gbm      — LightGBM on all features
  model5_gbm_thr  — LightGBM with 0.005 threshold filter

Usage
-----
    python scripts/train_classification_model.py
    python scripts/train_classification_model.py --features data/features_candles.parquet
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
    accuracy_score, roc_auc_score, classification_report,
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
    precision_up = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_up    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "model": name, "accuracy": acc, "auc": auc,
        "precision_up": precision_up, "recall_up": recall_up,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
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
        "feature":         feature_names,
        "importance":      importances,
        "importance_pct":  importances / total * 100 if total > 0 else importances,
        "category":        [FEATURE_CATEGORIES.get(f, "other") for f in feature_names],
    }).sort_values("importance", ascending=False).reset_index(drop=True)


# ── report generation ─────────────────────────────────────────────────────────

def ascii_bar(value: float, max_value: float, width: int = 30) -> str:
    filled = int(round(value / max_value * width)) if max_value > 0 else 0
    return "█" * filled + "░" * (width - filled)


def write_report(
    metrics_strict: list[dict],
    metrics_thresh: list[dict],
    lr_imp: pd.DataFrame,
    gbm_imp: pd.DataFrame,
    ticker_acc: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    n_train_strict: int,
    n_test_strict: int,
    n_train_thresh: int,
    n_test_thresh: int,
    out_path: pathlib.Path,
) -> None:
    lines = []

    def h(text):  lines.append(f"\n## {text}\n")
    def h3(text): lines.append(f"\n### {text}\n")
    def p(text=""): lines.append(text)

    lines.append("# Kalshi Weather Market: Price Direction Classification Report")
    lines.append(f"\n_Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_\n")
    lines.append(f"**Train**: {df_train['trade_date'].min()} – {df_train['trade_date'].max()}  "
                 f"**Test**: {df_test['trade_date'].min()} – {df_test['trade_date'].max()}")

    h("Methodology")
    p("- **Target**: binary — did `yes_ask_open` at T+30 go UP (1) or DOWN (0)?")
    p("- **Flat bars excluded**: rows where `target_move_30 == 0` (~35% of data) are dropped; "
      "they carry no directional signal")
    p(f"- **Strict split** (no flat): train {n_train_strict:,} rows, test {n_test_strict:,} rows")
    p(f"- **Threshold split** (|move| ≥ 0.005): train {n_train_thresh:,} rows, test {n_test_thresh:,} rows")
    p("- **Train/Test split**: purely temporal — 2025-06-01–2025-08-31 train, 2025-09-01–2025-09-30 test")
    p("- **Baseline**: predict majority class (50/50 after flat removal → 50% accuracy)")

    h("Model Comparison — Strict (exclude flat bars)")
    p("| Model | Accuracy | AUC | Precision↑ | Recall↑ |")
    p("|---|---|---|---|---|")
    p("| Majority-class baseline | 50.0% | 0.500 | — | — |")
    for m in metrics_strict:
        p(f"| {m['model']} | **{m['accuracy']:.1%}** | {m['auc']:.4f} | "
          f"{m['precision_up']:.1%} | {m['recall_up']:.1%} |")

    h("Model Comparison — Threshold (|move| ≥ 0.005, filters microstructure noise)")
    p("| Model | Accuracy | AUC | Precision↑ | Recall↑ |")
    p("|---|---|---|---|---|")
    p("| Majority-class baseline | 50.0% | 0.500 | — | — |")
    for m in metrics_thresh:
        p(f"| {m['model']} | **{m['accuracy']:.1%}** | {m['auc']:.4f} | "
          f"{m['precision_up']:.1%} | {m['recall_up']:.1%} |")

    h("Feature Importance: Logistic Regression (Model 3 — all linear features)")
    p("Coefficients normalized to % of total absolute weight (StandardScaler applied first).\n")
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

    h("Feature Importance: LightGBM (Model 4)")
    p("Importance by gain.\n")
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

    h("Per-Ticker Accuracy (GBM, Strict, Test Set)")
    p("| Ticker | Accuracy | N Rows | Mean |move| |")
    p("|---|---|---|---|")
    for _, row in ticker_acc.sort_values("accuracy", ascending=False).iterrows():
        p(f"| `{row['ticker']}` | {row['accuracy']:.1%} | {int(row['n_rows']):,} | "
          f"{row['mean_abs_move']:.4f} |")

    h("Key Findings")
    gbm_strict = next(m for m in metrics_strict if "gbm" in m["model"])
    gbm_thresh = next(m for m in metrics_thresh if "gbm" in m["model"])
    best_lr    = max((m for m in metrics_strict if "lr" in m["model"]), key=lambda m: m["accuracy"])

    p(f"1. **Best linear accuracy**: `{best_lr['model']}` at **{best_lr['accuracy']:.1%}** "
      f"(AUC {best_lr['auc']:.4f}) — logistic regression on all features")
    p(f"2. **GBM strict**: **{gbm_strict['accuracy']:.1%}** accuracy (AUC {gbm_strict['auc']:.4f}), "
      f"using all features on bars with any non-zero move")
    p(f"3. **GBM threshold**: **{gbm_thresh['accuracy']:.1%}** accuracy (AUC {gbm_thresh['auc']:.4f}), "
      f"restricting to |move| ≥ 0.005 — {'higher' if gbm_thresh['accuracy'] > gbm_strict['accuracy'] else 'lower'} "
      f"than strict, suggesting {'larger moves are more predictable' if gbm_thresh['accuracy'] > gbm_strict['accuracy'] else 'threshold filter does not help'}")

    top3_gbm = gbm_imp.head(3)["feature"].tolist()
    p(f"4. **GBM top features**: {', '.join('`'+f+'`' for f in top3_gbm)} — "
      f"{'time-of-day signals dominate' if any('tod' in f or 'hour' in f or 'minute' in f for f in top3_gbm[:2]) else 'momentum/flow signals lead'}")

    h("Limitations")
    p("- Flat bars (~35%) excluded — live trading must also handle the no-move case")
    p("- No position sizing: accuracy does not account for magnitude of wrong predictions")
    p("- Bid/ask spread (≈1¢–2¢) means directional accuracy must be sustained to be profitable")
    p("- Same-day only; no cross-day regime features")
    p("- Model may overfit to time-of-day patterns specific to summer 2025")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    log.info("Report written → %s", out_path)


# ── main ───────────────────────────────────────────────────────────────────────

def make_label(move: pd.Series, threshold: float = 0.0) -> pd.Series:
    """Return 1 if move > threshold, 0 if move < -threshold, NaN otherwise."""
    label = pd.Series(np.nan, index=move.index)
    label[move > threshold]  = 1
    label[move < -threshold] = 0
    return label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features",   default="data/features_candles.parquet", type=pathlib.Path)
    ap.add_argument("--models-dir", default="models",                         type=pathlib.Path)
    ap.add_argument("--report",     default="reports/classification_report.md", type=pathlib.Path)
    args = ap.parse_args()

    log.info("Loading features from %s", args.features)
    df = pd.read_parquet(args.features)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date.astype(str)

    train_raw = df[df["trade_date"] <= TRAIN_END].copy()
    test_raw  = df[df["trade_date"] >= TEST_START].copy()

    args.models_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for variant, threshold in [("strict", 0.0), ("thresh", 0.005)]:
        log.info("=== Variant: %s (threshold=%.4f) ===", variant, threshold)

        train_v = train_raw.copy()
        test_v  = test_raw.copy()
        train_v["label"] = make_label(train_v[TARGET_COL], threshold)
        test_v["label"]  = make_label(test_v[TARGET_COL],  threshold)

        train_v = train_v.dropna(subset=["label"])
        test_v  = test_v.dropna(subset=["label"])

        log.info("  Train: %d rows  up=%.1f%%  down=%.1f%%",
                 len(train_v), (train_v["label"] == 1).mean()*100,
                 (train_v["label"] == 0).mean()*100)
        log.info("  Test:  %d rows  up=%.1f%%  down=%.1f%%",
                 len(test_v), (test_v["label"] == 1).mean()*100,
                 (test_v["label"] == 0).mean()*100)

        y_train = train_v["label"].values
        y_test  = test_v["label"].values

        variant_metrics = []

        # ── linear models (strict only for full table; threshold for GBM) ──────
        if variant == "strict":
            for model_name, features in FEATURE_SETS.items():
                features = [f for f in features if f in df.columns]
                log.info("  Training %s (%d features)...", model_name, len(features))
                X_tr = train_v[features].values
                X_te = test_v[features].values

                pipe = make_lr_pipeline()
                pipe.fit(X_tr, y_train)
                y_pred = pipe.predict(X_te)
                y_prob = pipe.predict_proba(X_te)[:, 1]

                m = evaluate(model_name, y_test, y_pred, y_prob)
                variant_metrics.append(m)
                log.info("    Acc=%.1f%%  AUC=%.4f", m["accuracy"]*100, m["auc"])

                with open(args.models_dir / f"clf_{model_name}.pkl", "wb") as fh:
                    pickle.dump({"pipeline": pipe, "features": features, "metrics": m}, fh)

        # ── GBM ───────────────────────────────────────────────────────────────
        gbm_features = [f for f in ALL_FEATURES if f in df.columns]
        model_name   = f"model4_gbm{'_thresh' if variant == 'thresh' else ''}"
        log.info("  Training %s (%d features)...", model_name, len(gbm_features))

        X_tr_gbm = train_v[gbm_features].values
        X_te_gbm = test_v[gbm_features].values

        gbm_pipe = make_gbm_pipeline()
        gbm_pipe.fit(X_tr_gbm, y_train)
        y_pred_gbm = gbm_pipe.predict(X_te_gbm)
        y_prob_gbm = gbm_pipe.predict_proba(X_te_gbm)[:, 1]

        m_gbm = evaluate(model_name, y_test, y_pred_gbm, y_prob_gbm)
        variant_metrics.append(m_gbm)
        log.info("    Acc=%.1f%%  AUC=%.4f", m_gbm["accuracy"]*100, m_gbm["auc"])

        with open(args.models_dir / f"clf_{model_name}.pkl", "wb") as fh:
            pickle.dump({"pipeline": gbm_pipe, "features": gbm_features, "metrics": m_gbm}, fh)

        results[variant] = {
            "metrics": variant_metrics,
            "train_n": len(train_v),
            "test_n":  len(test_v),
        }

        if variant == "strict":
            gbm_imp_df  = gbm_importance(gbm_pipe, gbm_features)
            lr_best_pipe = None
            lr_best_features = None
            # per-ticker accuracy
            test_v2 = test_v.copy()
            test_v2["y_pred"] = y_pred_gbm
            ticker_rows = []
            for ticker, grp in test_v2.groupby("ticker"):
                ticker_rows.append({
                    "ticker":        ticker,
                    "accuracy":      accuracy_score(grp["label"], grp["y_pred"]),
                    "n_rows":        len(grp),
                    "mean_abs_move": grp[TARGET_COL].abs().mean(),
                })
            ticker_acc_df = pd.DataFrame(ticker_rows)

            # best LR pipeline for importance
            best_lr_name = max(
                ((m["model"], m["accuracy"]) for m in variant_metrics if "lr" in m["model"]),
                key=lambda x: x[1]
            )[0]
            with open(args.models_dir / f"clf_{best_lr_name}.pkl", "rb") as fh:
                best_lr_data = pickle.load(fh)
            lr_imp_df = lr_importance(best_lr_data["pipeline"], best_lr_data["features"])

    write_report(
        metrics_strict=results["strict"]["metrics"],
        metrics_thresh=results["thresh"]["metrics"],
        lr_imp=lr_imp_df,
        gbm_imp=gbm_imp_df,
        ticker_acc=ticker_acc_df,
        df_train=train_raw,
        df_test=test_raw,
        n_train_strict=results["strict"]["train_n"],
        n_test_strict=results["strict"]["test_n"],
        n_train_thresh=results["thresh"]["train_n"],
        n_test_thresh=results["thresh"]["test_n"],
        out_path=args.report,
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
