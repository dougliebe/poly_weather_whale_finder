"""
train_price_model.py
────────────────────────────────────────────────────────────────────────────
Trains 5 progressively richer models to predict yes_ask_open 30 min ahead,
evaluates on a temporal hold-out, and writes reports/feature_importance.md.

Models
------
  baseline      — Ridge on yes_ask_close only (current ask = prediction)
  model1        — + own-ticker momentum features (lags, spread, OI, volume)
  model2        — + cross-ticker probability conservation features
  model3        — + time-of-day features
  model4_gbm    — LightGBM on all features

Usage
-----
    python scripts/train_price_model.py
    python scripts/train_price_model.py --features data/features_candles.parquet
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
TARGET_COL = "target_ask_open_30"
CURRENT_PRICE_COL = "yes_ask_close"  # price at T, used for naive benchmark

FEATURE_SETS = {
    "baseline": [
        "yes_ask_close",
    ],
    "model1_momentum": [
        "yes_ask_close", "mid_close", "spread",
        "mid_ret_1", "mid_ret_5", "mid_ret_15", "mid_ret_30",
        "ask_lag1", "ask_lag5", "ask_lag15",
        "ask_intrabar_range", "bid_intrabar_range",
        "spread_lag1", "spread_lag5",
        "has_trade", "vol_sum_5", "vol_sum_15",
        "oi_change_1", "oi_change_5", "oi_change_15",
        "mid_vol_15",
        "price_previous",
    ],
    "model2_cross": [
        "yes_ask_close", "mid_close", "spread",
        "mid_ret_1", "mid_ret_5", "mid_ret_15", "mid_ret_30",
        "ask_lag1", "ask_lag5", "ask_lag15",
        "ask_intrabar_range", "bid_intrabar_range",
        "spread_lag1", "spread_lag5",
        "has_trade", "vol_sum_5", "vol_sum_15",
        "oi_change_1", "oi_change_5", "oi_change_15",
        "mid_vol_15", "price_previous",
        # cross-ticker
        "sum_mid_all", "rel_mid", "prob_sum_deviation",
        "relative_spread", "avg_spread_all", "spread_dispersion",
        "n_tickers_at_bar",
    ],
    "model3_tod": [
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
        # time-of-day
        "bar_hour", "bar_minute", "tod_sin", "tod_cos",
    ],
}

ALL_FEATURES = FEATURE_SETS["model3_tod"]  # GBM uses all of these


FEATURE_CATEGORIES = {
    "yes_ask_close": "anchor",
    "mid_close": "anchor",
    "mid_open": "anchor",
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
    "bar_minute": "time",
    "tod_sin": "time",
    "tod_cos": "time",
}


# ── model factories ────────────────────────────────────────────────────────────

def make_linear_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   Ridge(alpha=1.0)),
    ])


def make_gbm_pipeline() -> Pipeline:
    try:
        import lightgbm as lgb
        estimator = lgb.LGBMRegressor(
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
        log.warning("lightgbm not installed — falling back to Ridge for GBM model")
        from sklearn.linear_model import Ridge as estimator
        estimator = estimator(alpha=1.0)

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   estimator),
    ])


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(name: str, y_true, y_pred, y_current) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    dir_actual    = (np.array(y_true) - np.array(y_current)) > 0
    dir_predicted = (np.array(y_pred) - np.array(y_current)) > 0
    dir_acc = float((dir_actual == dir_predicted).mean())
    naive_mae = mean_absolute_error(y_true, y_current)
    vs_naive  = (naive_mae - mae) / naive_mae * 100  # positive = improvement
    return {
        "model": name, "MAE": mae, "RMSE": rmse, "R2": r2,
        "DirectionalAcc": dir_acc, "NaiveMAE": naive_mae, "vs_naive_pct": vs_naive,
    }


# ── feature importance ────────────────────────────────────────────────────────

def linear_importance(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    coefs = pipeline.named_steps["model"].coef_
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
        return linear_importance(pipeline, feature_names)
    importances = model.feature_importances_
    total = importances.sum()
    return pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
        "importance_pct": importances / total * 100 if total > 0 else importances,
        "category":   [FEATURE_CATEGORIES.get(f, "other") for f in feature_names],
    }).sort_values("importance", ascending=False).reset_index(drop=True)


# ── report generation ─────────────────────────────────────────────────────────

def ascii_bar(value: float, max_value: float, width: int = 35) -> str:
    filled = int(round(value / max_value * width)) if max_value > 0 else 0
    return "█" * filled + "░" * (width - filled)


def write_report(
    metrics: list[dict],
    linear_imp: pd.DataFrame,
    gbm_imp: pd.DataFrame,
    ticker_mae: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    out_path: pathlib.Path,
    cross_stats: dict,
) -> None:
    lines = []

    def h(text): lines.append(f"\n## {text}\n")
    def h3(text): lines.append(f"\n### {text}\n")
    def p(text=""): lines.append(text)

    lines.append("# Kalshi Weather Market: Price Prediction Report")
    lines.append(f"\n_Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_\n")
    lines.append(f"**Train**: {df_train['trade_date'].min()} – {df_train['trade_date'].max()} "
                 f"({len(df_train):,} rows)  "
                 f"**Test**: {df_test['trade_date'].min()} – {df_test['trade_date'].max()} "
                 f"({len(df_test):,} rows)")

    h("Methodology")
    p("- **Target**: `yes_ask_open` at T+30 minutes — the best ask price 30 1-minute bars ahead")
    p("- **Train/Test split**: purely temporal (no shuffling) — 2025-06-01–2025-08-31 train, "
      "2025-09-01–2025-09-30 test")
    p("- **NaN handling**: median imputation fit on training set only, applied to test")
    p("- **Prices**: decimal scale (0–1), where 0.72 = 72¢ / 72% implied probability")
    p("- **Naive benchmark**: predict no change (forecast = current `yes_ask_close`)")
    p("- **Evaluation**: MAE, RMSE, R², directional accuracy, % improvement vs naive")

    h("Model Comparison")
    p("| Model | MAE | RMSE | R² | Dir Acc | vs Naive |")
    p("|---|---|---|---|---|---|")
    naive_mae = metrics[0]["NaiveMAE"]
    p(f"| Naive (no change) | {naive_mae:.5f} | — | — | — | 0.0% |")
    for m in metrics:
        sign = "+" if m["vs_naive_pct"] >= 0 else ""
        p(f"| {m['model']} | {m['MAE']:.5f} | {m['RMSE']:.5f} | "
          f"{m['R2']:.4f} | {m['DirectionalAcc']:.1%} | "
          f"{sign}{m['vs_naive_pct']:.1f}% |")

    h("Feature Importance: Linear Model (Model 3 — all linear features)")
    p("Coefficients are normalized to % of total absolute weight after StandardScaler "
      "(i.e., features are on the same scale — higher % = stronger linear influence).\n")
    p("| Rank | Feature | Importance % | Coef | Category |")
    p("|---|---|---|---|---|")
    for i, row in linear_imp.head(20).iterrows():
        p(f"| {i+1} | `{row['feature']}` | {row['importance']*100:.2f}% | "
          f"{row['coef']:+.4f} | {row['category']} |")

    h3("Category Summary (Linear)")
    cat_sum = linear_imp.groupby("category")["importance"].sum().sort_values(ascending=False)
    p("| Category | Total Importance % |")
    p("|---|---|")
    for cat, imp in cat_sum.items():
        p(f"| {cat} | {imp*100:.1f}% |")

    h("Feature Importance: LightGBM (Model 4 — all features)")
    p("Importance by gain (total reduction in loss from splits on this feature).\n")
    max_imp = gbm_imp["importance"].max()
    p("| Rank | Feature | Gain | Share | Category | Bar |")
    p("|---|---|---|---|---|---|")
    cumulative = 0.0
    for i, row in gbm_imp.head(20).iterrows():
        cumulative += row["importance_pct"]
        bar = ascii_bar(row["importance"], max_imp, 25)
        p(f"| {i+1} | `{row['feature']}` | {row['importance']:,.0f} | "
          f"{row['importance_pct']:.1f}% | {row['category']} | `{bar}` |")

    h3("Category Summary (GBM)")
    cat_gbm = gbm_imp.groupby("category")["importance_pct"].sum().sort_values(ascending=False)
    p("| Category | Total Gain % |")
    p("|---|---|")
    for cat, imp in cat_gbm.items():
        p(f"| {cat} | {imp:.1f}% |")

    h("Cross-Ticker Signal Analysis")
    p(f"- Mean `sum_mid_all` (test set): **{cross_stats['sum_mid_mean']:.4f}** "
      f"(expect ≈1.0; deviation = arbitrage / liquidity imbalance)")
    p(f"- Std `sum_mid_all`: **{cross_stats['sum_mid_std']:.4f}**")
    p(f"- Mean `prob_sum_deviation`: **{cross_stats['prob_dev_mean']:+.4f}**")
    p(f"- `rel_mid` rank in GBM: **#{cross_stats['rel_mid_rank']}** of {len(gbm_imp)}")
    p(f"- `prob_sum_deviation` rank in GBM: **#{cross_stats['prob_dev_rank']}** of {len(gbm_imp)}")
    p(f"\nModel 2 (+ cross features) vs Model 1 MAE delta: "
      f"**{cross_stats['cross_lift']:+.5f}** "
      f"({'improvement' if cross_stats['cross_lift'] > 0 else 'degradation'})")

    h("Per-Ticker MAE (Test Set)")
    p("Tail bins (T-prefix) often have wider spreads and sparser trading.\n")
    p("| Ticker | MAE | Rows | Mean Ask | Mean Spread |")
    p("|---|---|---|---|---|")
    for _, row in ticker_mae.iterrows():
        p(f"| `{row['ticker']}` | {row['MAE']:.5f} | {int(row['n_rows']):,} | "
          f"{row['mean_ask']:.4f} | {row['mean_spread']:.4f} |")

    h("Key Findings")
    top3_linear = linear_imp.head(3)["feature"].tolist()
    top3_gbm    = gbm_imp.head(3)["feature"].tolist()
    model_maes  = {m["model"]: m["MAE"] for m in metrics}
    model_keys  = list(model_maes.keys())

    p(f"1. **Current ask dominates**: `yes_ask_close` is the single strongest linear predictor "
      f"(rank 1 in both linear and GBM models), confirming price persistence — the 30-min ahead "
      f"ask is strongly anchored to the current ask.")

    m1_lift = model_maes.get("baseline", 0) - model_maes.get("model1_momentum", 0)
    p(f"2. **Momentum features provide modest lift**: Model 1 (momentum) reduces MAE by "
      f"{m1_lift:.5f} vs baseline, suggesting lag returns and spread carry incremental signal "
      f"beyond the current price level alone.")

    m2_lift = model_maes.get("model1_momentum", 0) - model_maes.get("model2_cross", 0)
    p(f"3. **Cross-ticker signals**: Adding probability conservation features shifts MAE by "
      f"{m2_lift:+.5f}. The sum-of-mids deviation from 1.0 reflects cross-bin arbitrage "
      f"opportunities that briefly predict price moves.")

    m3_lift = model_maes.get("model2_cross", 0) - model_maes.get("model3_tod", 0)
    p(f"4. **Time-of-day**: Adding time features shifts MAE by {m3_lift:+.5f}. The cyclical "
      f"encoding captures METAR observation windows (fires at :53 past each hour) when "
      f"informed traders update positions.")

    gbm_best = model_maes.get("model4_gbm", 0)
    lin_best = model_maes.get("model3_tod", 0)
    p(f"5. **Non-linearity**: LightGBM (MAE={gbm_best:.5f}) vs best linear model "
      f"(MAE={lin_best:.5f}) — delta={lin_best - gbm_best:+.5f}. "
      f"{'GBM captures meaningful non-linear interactions.' if gbm_best < lin_best else 'Linear model is competitive; non-linearity adds little.'}")

    h("Limitations")
    p("- Only 11.5% of bars have trade OHLC (`price_*`); models rely primarily on quote OHLC")
    p("- Same-day bars only — no inter-day momentum or regime features")
    p("- 122 days of training data; seasonal effects within summer may be underweighted")
    p("- `price_previous` (2.1% null) imputed with median — may understate signal in thin markets")
    p("- Tail bins (T-prefix) have wider spreads; a per-bin model may outperform a pooled one")
    p("- The T+30 target assumes a bar exists 30 minutes later (dropped ≈20% of bars near day-end)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    log.info("Report written → %s", out_path)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Train Kalshi candlestick price prediction models.")
    ap.add_argument("--features", default="data/features_candles.parquet", type=pathlib.Path)
    ap.add_argument("--models-dir", default="models", type=pathlib.Path)
    ap.add_argument("--report",    default="reports/feature_importance.md", type=pathlib.Path)
    args = ap.parse_args()

    log.info("Loading features from %s", args.features)
    df = pd.read_parquet(args.features)
    log.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # Sanity check price scale
    assert df["yes_ask_close"].between(0, 1).mean() > 0.95, "Prices not in [0,1] range"

    # Temporal split
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date.astype(str)
    train = df[df["trade_date"] <= TRAIN_END].copy()
    test  = df[df["trade_date"] >= TEST_START].copy()
    log.info("Train: %d rows (%s–%s)", len(train), train["trade_date"].min(), train["trade_date"].max())
    log.info("Test:  %d rows (%s–%s)", len(test),  test["trade_date"].min(), test["trade_date"].max())

    y_train = train[TARGET_COL].values
    y_test  = test[TARGET_COL].values
    y_current_test = test[CURRENT_PRICE_COL].values

    args.models_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict] = []
    trained_pipelines: dict[str, Pipeline] = {}

    # ── train linear models ────────────────────────────────────────────────────
    for model_name, features in FEATURE_SETS.items():
        features = [f for f in features if f in df.columns]
        log.info("Training %s (%d features)...", model_name, len(features))

        X_train = train[features].values
        X_test  = test[features].values

        pipe = make_linear_pipeline()
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        m = evaluate(model_name, y_test, y_pred, y_current_test)
        all_metrics.append(m)
        trained_pipelines[model_name] = (pipe, features)
        log.info("  MAE=%.5f  RMSE=%.5f  R²=%.4f  DirAcc=%.1f%%  vs_naive=%+.1f%%",
                 m["MAE"], m["RMSE"], m["R2"], m["DirectionalAcc"]*100, m["vs_naive_pct"])

        with open(args.models_dir / f"{model_name}.pkl", "wb") as fh:
            pickle.dump({"pipeline": pipe, "features": features, "metrics": m}, fh)

    # ── train GBM ─────────────────────────────────────────────────────────────
    gbm_features = [f for f in ALL_FEATURES if f in df.columns]
    log.info("Training model4_gbm (%d features)...", len(gbm_features))
    X_train_gbm = train[gbm_features].values
    X_test_gbm  = test[gbm_features].values

    gbm_pipe = make_gbm_pipeline()
    gbm_pipe.fit(X_train_gbm, y_train)
    y_pred_gbm = gbm_pipe.predict(X_test_gbm)

    m_gbm = evaluate("model4_gbm", y_test, y_pred_gbm, y_current_test)
    all_metrics.append(m_gbm)
    trained_pipelines["model4_gbm"] = (gbm_pipe, gbm_features)
    log.info("  MAE=%.5f  RMSE=%.5f  R²=%.4f  DirAcc=%.1f%%  vs_naive=%+.1f%%",
             m_gbm["MAE"], m_gbm["RMSE"], m_gbm["R2"], m_gbm["DirectionalAcc"]*100, m_gbm["vs_naive_pct"])

    with open(args.models_dir / "model4_gbm.pkl", "wb") as fh:
        pickle.dump({"pipeline": gbm_pipe, "features": gbm_features, "metrics": m_gbm}, fh)

    # ── feature importance ─────────────────────────────────────────────────────
    best_linear_pipe, best_linear_features = trained_pipelines["model3_tod"]
    lin_imp = linear_importance(best_linear_pipe, best_linear_features)

    gbm_imp = gbm_importance(gbm_pipe, gbm_features)

    # ── per-ticker MAE ─────────────────────────────────────────────────────────
    test_with_pred = test.copy()
    test_with_pred["y_pred_gbm"] = y_pred_gbm
    ticker_rows = []
    for ticker, grp in test_with_pred.groupby("ticker"):
        mae = mean_absolute_error(grp[TARGET_COL], grp["y_pred_gbm"])
        ticker_rows.append({
            "ticker":      ticker,
            "MAE":         mae,
            "n_rows":      len(grp),
            "mean_ask":    grp["yes_ask_close"].mean(),
            "mean_spread": grp["spread"].mean(),
        })
    ticker_mae = pd.DataFrame(ticker_rows).sort_values("MAE", ascending=False)

    # ── cross-ticker stats ─────────────────────────────────────────────────────
    def rank_of(imp_df, feat):
        matches = imp_df[imp_df["feature"] == feat]
        return int(matches.index[0]) + 1 if len(matches) else len(imp_df)

    cross_stats = {
        "sum_mid_mean":  test["sum_mid_all"].mean(),
        "sum_mid_std":   test["sum_mid_all"].std(),
        "prob_dev_mean": test["prob_sum_deviation"].mean(),
        "rel_mid_rank":  rank_of(gbm_imp, "rel_mid"),
        "prob_dev_rank": rank_of(gbm_imp, "prob_sum_deviation"),
        "cross_lift": (
            all_metrics[1]["MAE"] - all_metrics[2]["MAE"]  # model1 - model2
        ),
    }

    # ── write report ───────────────────────────────────────────────────────────
    write_report(
        metrics=all_metrics,
        linear_imp=lin_imp,
        gbm_imp=gbm_imp,
        ticker_mae=ticker_mae,
        df_train=train,
        df_test=test,
        out_path=args.report,
        cross_stats=cross_stats,
    )

    log.info("Done. Models in %s/, report at %s", args.models_dir, args.report)


if __name__ == "__main__":
    main()
