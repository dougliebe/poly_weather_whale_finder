"""
build_candle_features.py
────────────────────────────────────────────────────────────────────────────
Extracts feature-engineered rows from the Ducklake candles table and writes
them to data/features_candles.parquet for model training.

Target: yes_ask_open at T+30 minutes (ask price 30 bars ahead).

Feature categories
------------------
  own-ticker momentum  — mid-price lag returns, ask lags, spread lags,
                         OI change, rolling volume, intrabar volatility
  cross-ticker         — probability conservation signals across all 6
                         co-existing bins per event timestamp
  time-of-day          — hour/minute, cyclic sin/cos encoding

Usage
-----
    python scripts/build_candle_features.py
    python scripts/build_candle_features.py --catalog data/kalshi.ducklake --out data/features_candles.parquet
"""

import argparse
import logging
import math
from pathlib import Path

import duckdb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FEATURE_SQL = """
WITH base AS (
    SELECT
        ticker,
        series_ticker,
        trade_date,
        end_period_ts,
        end_period_utc,
        interval_min,
        -- mid price and spread (100% non-null since bid/ask are always present)
        (yes_ask_close + yes_bid_close) / 2.0                              AS mid_close,
        (yes_ask_open  + yes_bid_open)  / 2.0                              AS mid_open,
        yes_ask_close - yes_bid_close                                       AS spread,
        yes_ask_high  - yes_ask_low                                         AS ask_intrabar_range,
        yes_bid_high  - yes_bid_low                                         AS bid_intrabar_range,
        yes_ask_close,
        yes_ask_open,
        yes_ask_high,
        yes_ask_low,
        yes_bid_close,
        yes_bid_open,
        volume,
        open_interest,
        price_previous,
        -- time-of-day
        EXTRACT(hour   FROM end_period_utc)::INTEGER                        AS bar_hour,
        EXTRACT(minute FROM end_period_utc)::INTEGER                        AS bar_minute,
        -- cyclic time encoding (fraction of day in minutes)
        SIN(2.0 * {pi} * (
              EXTRACT(hour FROM end_period_utc) * 60
            + EXTRACT(minute FROM end_period_utc)
        ) / 1440.0)                                                         AS tod_sin,
        COS(2.0 * {pi} * (
              EXTRACT(hour FROM end_period_utc) * 60
            + EXTRACT(minute FROM end_period_utc)
        ) / 1440.0)                                                         AS tod_cos,
        CASE WHEN volume > 0 THEN 1 ELSE 0 END                             AS has_trade
    FROM candles
    WHERE interval_min = 1
),

lagged AS (
    SELECT *,
        -- mid-price lag returns (k = 1, 5, 15, 30)
        (mid_close - LAG(mid_close,  1) OVER w) / NULLIF(LAG(mid_close,  1) OVER w, 0) AS mid_ret_1,
        (mid_close - LAG(mid_close,  5) OVER w) / NULLIF(LAG(mid_close,  5) OVER w, 0) AS mid_ret_5,
        (mid_close - LAG(mid_close, 15) OVER w) / NULLIF(LAG(mid_close, 15) OVER w, 0) AS mid_ret_15,
        (mid_close - LAG(mid_close, 30) OVER w) / NULLIF(LAG(mid_close, 30) OVER w, 0) AS mid_ret_30,
        -- ask lags (directly tied to target)
        LAG(yes_ask_close,  1) OVER w  AS ask_lag1,
        LAG(yes_ask_close,  5) OVER w  AS ask_lag5,
        LAG(yes_ask_close, 15) OVER w  AS ask_lag15,
        -- spread lags
        LAG(spread,  1) OVER w         AS spread_lag1,
        LAG(spread,  5) OVER w         AS spread_lag5,
        -- open interest change
        open_interest - LAG(open_interest,  1) OVER w  AS oi_change_1,
        open_interest - LAG(open_interest,  5) OVER w  AS oi_change_5,
        open_interest - LAG(open_interest, 15) OVER w  AS oi_change_15,
        -- rolling volume sums
        SUM(volume) OVER (PARTITION BY ticker, trade_date ORDER BY end_period_ts
                          ROWS BETWEEN  4 PRECEDING AND CURRENT ROW)  AS vol_sum_5,
        SUM(volume) OVER (PARTITION BY ticker, trade_date ORDER BY end_period_ts
                          ROWS BETWEEN 14 PRECEDING AND CURRENT ROW)  AS vol_sum_15,
        -- rolling mid-price volatility
        STDDEV(mid_close) OVER (PARTITION BY ticker, trade_date ORDER BY end_period_ts
                                ROWS BETWEEN 14 PRECEDING AND CURRENT ROW)  AS mid_vol_15
    FROM base
    WINDOW w AS (PARTITION BY ticker, trade_date ORDER BY end_period_ts)
),

cross_ticker AS (
    SELECT
        trade_date,
        end_period_ts,
        SUM(mid_close)    AS sum_mid_all,
        AVG(spread)       AS avg_spread_all,
        STDDEV(spread)    AS spread_dispersion,
        COUNT(*)          AS n_tickers_at_bar
    FROM base
    GROUP BY trade_date, end_period_ts
),

with_cross AS (
    SELECT
        l.*,
        ct.sum_mid_all,
        ct.avg_spread_all,
        ct.spread_dispersion,
        ct.n_tickers_at_bar,
        -- this ticker's probability share
        l.mid_close / NULLIF(ct.sum_mid_all, 0)    AS rel_mid,
        -- deviation from probability conservation (sum should ≈ 1)
        ct.sum_mid_all - 1.0                        AS prob_sum_deviation,
        -- this ticker's spread relative to cross-ticker average
        l.spread / NULLIF(ct.avg_spread_all, 0)     AS relative_spread
    FROM lagged l
    JOIN cross_ticker ct
      ON l.trade_date = ct.trade_date AND l.end_period_ts = ct.end_period_ts
),

with_target AS (
    SELECT
        f.*,
        t30.yes_ask_open                              AS target_ask_open_30,
        (t30.yes_ask_open + t30.yes_bid_open) / 2.0  AS target_mid_30
    FROM with_cross f
    LEFT JOIN base t30
      ON  f.ticker     = t30.ticker
      AND f.trade_date = t30.trade_date
      AND t30.end_period_ts = f.end_period_ts + 1800
)

SELECT * FROM with_target
WHERE target_ask_open_30 IS NOT NULL
ORDER BY ticker, trade_date, end_period_ts
""".format(pi=math.pi)


def build_features(catalog: Path, data_dir: Path, out_path: Path) -> None:
    log.info("Connecting to Ducklake: %s", catalog)
    con = duckdb.connect()
    con.execute("INSTALL ducklake; LOAD ducklake;")
    con.execute(f"ATTACH 'ducklake:{catalog}' AS lake (DATA_PATH '{data_dir}');")
    con.execute("USE lake;")

    log.info("Running feature SQL...")
    df = con.execute(FEATURE_SQL).fetchdf()
    con.close()

    log.info("Rows: %d  Columns: %d", len(df), len(df.columns))

    # Sanity checks
    assert df["yes_ask_close"].between(0, 1).mean() > 0.95, \
        "Prices appear to be in cents, not decimals — check data"
    assert df["target_ask_open_30"].between(0, 1).mean() > 0.95, \
        "Target prices appear to be in cents, not decimals — check data"

    pct_null = df.isnull().mean()
    high_null = pct_null[pct_null > 0.1]
    if not high_null.empty:
        log.info("Columns with >10%% nulls:\n%s", high_null.to_string())

    log.info("Date range: %s to %s", df["trade_date"].min(), df["trade_date"].max())
    log.info("Unique tickers: %d", df["ticker"].nunique())
    log.info("sum_mid_all mean=%.4f  std=%.4f (expect ≈1.0)",
             df["sum_mid_all"].mean(), df["sum_mid_all"].std())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")
    log.info("Wrote %d rows → %s", len(df), out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract candle features to parquet.")
    p.add_argument("--catalog",  default="data/kalshi.ducklake", type=Path)
    p.add_argument("--data-dir", default="data/kalshi_data",     type=Path)
    p.add_argument("--out",      default="data/features_candles.parquet", type=Path)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_features(args.catalog, args.data_dir, args.out)
