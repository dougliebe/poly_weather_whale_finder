# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Two largely independent workstreams live here:

1. **Polymarket whale finder** (`main.py`, `src/`) — original purpose; detects "fast-feed" traders on Polymarket weather markets who front-run price moves. Requires `POLY_API_KEY`, `POLY_API_SECRET`, `POLY_PASSPHRASE` env vars.

2. **Kalshi KXHIGHLAX price prediction** (`scripts/`) — active workstream; ingests 1-minute candlestick data for Kalshi LAX daily high temperature markets, engineers features, trains ML models, and runs calibration / Kelly-criterion analysis to find mispriced cells.

## Key commands

```bash
# Install dependencies
pip install -r requirements.txt
# Additional deps needed for Kalshi scripts (not in requirements.txt yet):
pip install duckdb lightgbm scikit-learn pyarrow scipy

# --- Polymarket whale finder ---
python main.py                          # default: last 30 days, all weather markets
python main.py --days 60 --save         # save CSVs to data/

# --- Kalshi pipeline (run in order) ---
# 1. Ingest candlestick data
python scripts/ingest_candles_ducklake.py --series KXHIGHLAX --start 2025-06-01 --end 2025-09-30
python scripts/ingest_candles_ducklake.py --query   # check what's loaded

# 2. Build feature parquet
python scripts/build_candle_features.py             # writes data/features_candles.parquet

# 3. Train models
python scripts/train_price_model.py                 # regression; writes reports/feature_importance.md
python scripts/train_classification_model.py        # 3-class UP/NEUTRAL/DOWN; writes reports/classification_report.md
```

## Kalshi data architecture

**Storage**: DuckLake (DuckDB community extension) at `data/kalshi.ducklake` with parquet data files in `data/kalshi_data/`. Only one DuckDB connection can hold the write lock at a time.

**Connect pattern** (used in every script):
```python
con = duckdb.connect()
con.execute("INSTALL ducklake; LOAD ducklake;")
con.execute("ATTACH 'ducklake:data/kalshi.ducklake' AS lake (DATA_PATH 'data/kalshi_data/');")
con.execute("USE lake;")
```

**Kalshi API endpoints**:
- Pre-2026 data: `GET /trade-api/v2/historical/markets/{ticker}/candlesticks`
- 2026+ data: `GET /trade-api/v2/series/{series}/markets/{ticker}/candlesticks`
  - These two endpoints have different response schemas: historical uses `"open"/"close"` keys; series uses `"open_dollars"/"close_dollars"`. The ingest script handles both via `_f(d, key)` which tries both variants.
- The cutoff is hardcoded at `2026-01-01` in `fetch_candles()`.

**Ticker format**: `{SERIES}-{YY}{MON}{DD}-{SUFFIX}`
- e.g. `KXHIGHLAX-25JUL02-B70.5` — LAX high temp on Jul 2, 2025, "between 70.5–71.5°F" bin
- Suffixes: `T{n}` = tail (below n-1° or above n+1°), `B{n}.5` = between-bin
- LAX suffixes defined in `_KXHIGHLAX_BIN_SUFFIXES`; Miami in `_KXHIGHMIA_BIN_SUFFIXES`

**Resolution inference** (no direct settlement API): a ticker resolved YES if `max(yes_ask_close after 21:00 UTC) > 0.90`, resolved NO if `< 0.10`. Used in calibration analysis.

## Feature pipeline (`build_candle_features.py`)

All features are computed in SQL CTEs — no lookahead; windows partitioned by `(ticker, trade_date)`. Key target: `target_move_90 = yes_ask_open(T+90min) − yes_ask_close(T)`.

Feature categories: own-ticker momentum (lag returns, ask lags, OI change, volume), volume burst z-scores, cross-ticker probability conservation signals, METAR timing (KLAX observation fires at :53 UTC), time-to-resolution, and bar-level time features derived from epoch arithmetic (not EXTRACT, to avoid DST issues).

## Model files

- `models/clf3_*.pkl` — 3-class classification models, structure: `{"pipeline": Pipeline, "features": list, "metrics": dict}`
- `models/*.pkl` — regression models, same structure
- Both are gitignored.

## Calibration / Kelly analysis

The Bayesian Kelly analysis (run ad-hoc in Python sessions, not a committed script yet) uses:
- **Prior**: `Beta(a·κ, (1-a)·κ)` centered at market price `a`, with `κ=20`
- **YES Kelly**: `(p - a) / (1 - a)` where `a = yes_ask_close`
- **NO Kelly**: `((1-p) - a) / (1 - a)` where `a = 1 - yes_bid_close` (actual no-ask from bid side, not `1 - yes_ask`)
- `P(edge > 0)` drives opacity in heatmap SVGs; cells with low confidence fade out

## Important caveats

**Survivorship bias in calibration**: mid-range prices (35–60¢) resolve YES at much higher rates than face value because YES-resolving contracts pass through those prices on their way to 100¢, while NO-resolving contracts collapse near 0¢ early. The apparent edge at early-morning, mid-price buckets is at least partly this artifact.

**Time zones**: LAX = PDT (UTC-7 summer), Miami = EDT (UTC-4 summer). All raw timestamps in the DB are UTC epoch seconds. Convert with `(end_period_ts % 86400 / 60 + UTC_OFFSET_MIN) % 1440` for local minute-of-day.
