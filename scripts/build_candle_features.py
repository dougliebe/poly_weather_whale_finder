"""
build_candle_features.py
────────────────────────────────────────────────────────────────────────────
Extracts feature-engineered rows from the Ducklake candles table and writes
them to data/features_candles.parquet for model training.

Target: target_move_90 = yes_ask_open(T+90) − yes_ask_close(T)
        (signed 30-minute price move in decimal probability units)

Feature categories
------------------
  own-ticker momentum  — mid-price lag returns, ask lags, spread lags,
                         OI change, rolling volume, intrabar volatility
  volume bursts        — z-score vs 15-bar baseline, burst flag,
                         OI acceleration, spread rate of change
  cross-ticker         — probability conservation signals; cross-ticker
                         volume burst (coordinated activity)
  METAR timing         — minutes to/since :53 KLAX observation window,
                         pre/post METAR binary flags
  time-to-resolution   — signed minutes to expected daily high (~3pm PDT),
                         fraction of trading day elapsed, price polarization
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
        -- prices
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
        -- price polarization: distance from 0 or 1 (small = market is certain)
        LEAST(yes_ask_close, 1.0 - yes_bid_close)                          AS price_polarization,

        -- ── time-of-day (UTC epoch arithmetic, no DST dependency) ─────────────
        (end_period_ts % 86400 / 3600)::INTEGER                             AS bar_hour,
        (end_period_ts % 3600  / 60)::INTEGER                               AS minute_of_hour,
        SIN(2.0 * {pi} * (end_period_ts % 86400) / 86400.0)                AS tod_sin,
        COS(2.0 * {pi} * (end_period_ts % 86400) / 86400.0)                AS tod_cos,

        -- ── METAR timing (KLAX obs at :53 UTC each hour) ──────────────────────
        -- bars_since_metar: 0 = observation just fired, 59 = about to fire
        ((end_period_ts % 3600 / 60)::INTEGER - 53 + 60) % 60              AS bars_since_metar,
        -- minutes_to_next_metar: 0 = METAR fires this bar, 59 = just fired
        ((53 - (end_period_ts % 3600 / 60)::INTEGER) + 60) % 60            AS minutes_to_next_metar,
        -- binary window flags
        CASE WHEN ((end_period_ts % 3600 / 60)::INTEGER) BETWEEN 48 AND 52
             THEN 1 ELSE 0 END                                              AS is_pre_metar,
        CASE WHEN ((end_period_ts % 3600 / 60)::INTEGER) BETWEEN 53 AND 58
             THEN 1 ELSE 0 END                                              AS is_post_metar,

        -- ── time-to-resolution ────────────────────────────────────────────────
        -- expected daily high for LAX ≈ 3pm PDT = 22:00 UTC = 1320 min from midnight
        -- positive = still approaching peak, negative = peak likely passed
        (1320 - (end_period_ts % 86400 / 60)::INTEGER)                     AS mins_to_peak,
        -- fraction of trading window elapsed (open ≈ 13:00 UTC, close ≈ 01:00 UTC next day = 720 min)
        GREATEST(0.0, LEAST(1.0,
            ((end_period_ts % 86400 / 60)::INTEGER - 780) / 720.0
        ))                                                                   AS frac_day_elapsed,

        CASE WHEN volume > 0 THEN 1 ELSE 0 END                             AS has_trade
    FROM candles
    WHERE interval_min = 1
      AND trade_date::DATE = strptime(split_part(ticker, '-', 2), '%y%b%d')::DATE
),

lagged AS (
    SELECT *,
        -- mid-price lag returns (k = 1, 5, 15, 30)
        (mid_close - LAG(mid_close,  1) OVER w) / NULLIF(LAG(mid_close,  1) OVER w, 0) AS mid_ret_1,
        (mid_close - LAG(mid_close,  5) OVER w) / NULLIF(LAG(mid_close,  5) OVER w, 0) AS mid_ret_5,
        (mid_close - LAG(mid_close, 15) OVER w) / NULLIF(LAG(mid_close, 15) OVER w, 0) AS mid_ret_15,
        (mid_close - LAG(mid_close, 30) OVER w) / NULLIF(LAG(mid_close, 30) OVER w, 0) AS mid_ret_30,
        -- ask lags
        LAG(yes_ask_close,  1) OVER w  AS ask_lag1,
        LAG(yes_ask_close,  5) OVER w  AS ask_lag5,
        LAG(yes_ask_close, 15) OVER w  AS ask_lag15,
        -- spread lags
        LAG(spread,  1) OVER w         AS spread_lag1,
        LAG(spread,  5) OVER w         AS spread_lag5,
        -- open interest change (levels 1, 5, 15)
        open_interest - LAG(open_interest,  1) OVER w  AS oi_change_1,
        open_interest - LAG(open_interest,  5) OVER w  AS oi_change_5,
        open_interest - LAG(open_interest, 15) OVER w  AS oi_change_15,
        -- OI acceleration: second derivative (is the rate of OI change itself increasing?)
        (open_interest - LAG(open_interest, 1) OVER w)
            - (LAG(open_interest, 1) OVER w - LAG(open_interest, 2) OVER w) AS oi_accel,
        -- rolling volume sums
        SUM(volume) OVER (PARTITION BY ticker, trade_date ORDER BY end_period_ts
                          ROWS BETWEEN  4 PRECEDING AND CURRENT ROW)        AS vol_sum_5,
        SUM(volume) OVER (PARTITION BY ticker, trade_date ORDER BY end_period_ts
                          ROWS BETWEEN 14 PRECEDING AND CURRENT ROW)        AS vol_sum_15,
        -- rolling vol mean and std for z-score
        AVG(volume) OVER (PARTITION BY ticker, trade_date ORDER BY end_period_ts
                          ROWS BETWEEN 14 PRECEDING AND CURRENT ROW)        AS vol_mean_15,
        STDDEV(volume) OVER (PARTITION BY ticker, trade_date ORDER BY end_period_ts
                             ROWS BETWEEN 14 PRECEDING AND CURRENT ROW)     AS vol_std_15,
        -- rolling mid-price volatility
        STDDEV(mid_close) OVER (PARTITION BY ticker, trade_date ORDER BY end_period_ts
                                ROWS BETWEEN 14 PRECEDING AND CURRENT ROW)  AS mid_vol_15,
        -- spread rate of change (compression signals incoming information)
        spread - LAG(spread, 5) OVER w                                      AS spread_chg_5,
        -- polarization trend (is market becoming more certain?)
        price_polarization - LAG(price_polarization, 5) OVER w              AS polar_chg_5
    FROM base
    WINDOW w AS (PARTITION BY ticker, trade_date ORDER BY end_period_ts)
),

-- volume burst features derived from rolling stats
vol_burst AS (
    SELECT *,
        -- z-score: how many std devs above recent mean is current volume?
        (volume - vol_mean_15) / NULLIF(vol_std_15, 0)                      AS vol_zscore,
        -- ratio: current bar vs recent average (2.0 = double the pace)
        volume / NULLIF(vol_mean_15, 0)                                      AS vol_ratio,
        -- binary burst flag (>2x recent average)
        CASE WHEN vol_mean_15 > 0 AND volume > 2.0 * vol_mean_15
             THEN 1 ELSE 0 END                                               AS vol_burst_flag
    FROM lagged
),

cross_ticker AS (
    SELECT
        trade_date,
        end_period_ts,
        SUM(mid_close)    AS sum_mid_all,
        AVG(spread)       AS avg_spread_all,
        STDDEV(spread)    AS spread_dispersion,
        COUNT(*)          AS n_tickers_at_bar,
        -- cross-ticker volume: are multiple bins active simultaneously?
        SUM(volume)       AS total_vol_cross,
        AVG(volume)       AS avg_vol_cross
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
        ct.total_vol_cross,
        -- this ticker's probability share
        l.mid_close / NULLIF(ct.sum_mid_all, 0)         AS rel_mid,
        -- deviation from probability conservation (sum should ≈ 1)
        ct.sum_mid_all - 1.0                             AS prob_sum_deviation,
        -- this ticker's spread vs cross-ticker average
        l.spread / NULLIF(ct.avg_spread_all, 0)          AS relative_spread,
        -- this ticker's volume vs cross-ticker average (coordinated burst signal)
        l.volume / NULLIF(ct.avg_vol_cross, 0)           AS vol_vs_cross
    FROM vol_burst l
    JOIN cross_ticker ct
      ON l.trade_date = ct.trade_date AND l.end_period_ts = ct.end_period_ts
),

with_target AS (
    SELECT
        f.*,
        t30.yes_ask_open                              AS target_ask_open_90,
        (t30.yes_ask_open + t30.yes_bid_open) / 2.0  AS target_mid_90,
        t30.yes_ask_open - f.yes_ask_close            AS target_move_90
    FROM with_cross f
    LEFT JOIN base t30
      ON  f.ticker     = t30.ticker
      AND f.trade_date = t30.trade_date
      AND t30.end_period_ts = f.end_period_ts + 5400
)

SELECT * FROM with_target
WHERE target_move_90 IS NOT NULL
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
    assert df["target_ask_open_90"].between(0, 1).mean() > 0.95, \
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
