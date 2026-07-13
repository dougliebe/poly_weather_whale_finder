"""
ingest_candles_ducklake.py
────────────────────────────────────────────────────────────────────────────
Fetches Kalshi historical candlesticks for one or more dates and upserts
them into a local Ducklake table via DuckDB.

The Ducklake catalog lives at data/kalshi.ducklake (configurable).
Data files are written alongside it in data/kalshi_data/.

Usage
-----
    # Single date
    python scripts/ingest_candles_ducklake.py \
        --event KXHIGHLAX-25JUL02 --date 2025-07-02 --probe-tickers

    # Date range (pulls every calendar day in [start, end])
    python scripts/ingest_candles_ducklake.py \
        --series KXHIGHLAX \
        --start 2025-06-01 --end 2025-07-31 \
        --probe-tickers

    # Explicit tickers
    python scripts/ingest_candles_ducklake.py \
        --event KXHIGHLAX-25JUL02 --date 2025-07-02 \
        --tickers KXHIGHLAX-25JUL02-B70.5,KXHIGHLAX-25JUL02-B72.5

    # Query the lake after ingestion
    python scripts/ingest_candles_ducklake.py --query

Setup
-----
    pip install duckdb requests
    # Ducklake is a DuckDB community extension — installed automatically on first run.

Storage
-------
    --catalog   Path to the .ducklake catalog file (default: data/kalshi.ducklake)
    --data-dir  Directory for Parquet data files  (default: data/kalshi_data)

    For S3, pass an s3:// path to --catalog and --data-dir (requires
    DuckDB's httpfs extension and AWS credentials in your environment).
"""

import argparse
import logging
import sys
import time
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

import duckdb
import requests

# ── shared discovery / fetch logic (mirrors pull_kalshi_candlesticks.py) ──────

KALSHI_BASE     = "https://external-api.kalshi.com/trade-api/v2"
MARKETS_EP      = f"{KALSHI_BASE}/markets"
EVENTS_EP       = f"{KALSHI_BASE}/events"
HIST_TRADES_EP  = f"{KALSHI_BASE}/historical/trades"
CANDLES_EP_TMPL = f"{KALSHI_BASE}/historical/markets/{{ticker}}/candlesticks"

_KXHIGHLAX_BIN_SUFFIXES = [
    # below-tail (T{x} = "x-1° or below")
    "T60", "T61", "T62", "T63", "T64", "T65", "T66", "T67", "T68", "T69", "T70", "T71", "T72",
    # between-bins (B{x}.5 = "x° to x+1°") — both even and odd bases
    "B61.5", "B62.5", "B63.5", "B64.5", "B65.5", "B66.5", "B67.5", "B68.5", "B69.5", "B70.5",
    "B71.5", "B72.5", "B73.5", "B74.5", "B75.5", "B76.5", "B77.5", "B78.5", "B79.5", "B80.5",
    "B81.5", "B82.5", "B83.5", "B84.5", "B85.5",
    # above-tail (T{x} = "x+1° or above")
    "T73", "T74", "T75", "T76", "T77", "T78", "T79", "T80", "T82", "T84", "T86",
]

# Miami uses coarser 2°F bins; range shifts seasonally ~86–96°F in summer
# Discovery via /markets API works directly — this list is a probe fallback only
_KXHIGHMIA_BIN_SUFFIXES = [
    "T84", "T85", "T86", "T87", "T88", "T89", "T90", "T91", "T92", "T93", "T94", "T95", "T96", "T97",
    "B84.5", "B85.5", "B86.5", "B87.5", "B88.5", "B89.5", "B90.5",
    "B91.5", "B92.5", "B93.5", "B94.5", "B95.5", "B96.5",
]

_SERIES_BIN_SUFFIXES = {
    "KXHIGHLAX": _KXHIGHLAX_BIN_SUFFIXES,
    "KXHIGHMIA": _KXHIGHMIA_BIN_SUFFIXES,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def get_with_retry(url, params, max_retries=6, rate_sleep=0.15):
    backoff = 2.0
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
        except requests.ConnectionError as exc:
            if attempt == max_retries:
                raise
            time.sleep(backoff); backoff = min(backoff * 2, 60); continue

        if resp.status_code == 200:
            time.sleep(rate_sleep)
            return resp.json()

        if resp.status_code == 429 or resp.status_code >= 500:
            wait = max(float(resp.headers.get("Retry-After", backoff)), backoff)
            if attempt == max_retries:
                resp.raise_for_status()
            log.warning("HTTP %d, retry %d/%d in %.1fs", resp.status_code, attempt+1, max_retries, wait)
            time.sleep(wait); backoff = min(backoff * 2, 60); continue

        resp.raise_for_status()
    raise RuntimeError("Exhausted retries")


def day_bounds(d: date):
    s = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
    e = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)
    return int(s.timestamp()), int(e.timestamp())


def discover_tickers(event_ticker, start_ts, end_ts, probe=False, explicit=None, rate_sleep=0.15):
    if explicit:
        return sorted(explicit)

    # Strategy 1: /markets
    body = get_with_retry(MARKETS_EP, {"event_ticker": event_ticker, "limit": 100}, rate_sleep=rate_sleep)
    tickers = sorted(m["ticker"] for m in (body.get("markets") or []) if "ticker" in m)
    if tickers:
        return tickers

    # Strategy 2: /events
    try:
        body = get_with_retry(f"{EVENTS_EP}/{event_ticker}", {}, rate_sleep=rate_sleep)
        tickers = sorted(m["ticker"] for m in (body.get("markets") or []) if "ticker" in m)
        if tickers:
            return tickers
    except requests.HTTPError:
        pass

    # Strategy 3: probe historical trades
    if probe:
        series = event_ticker.split("-")[0]
        candidates = [f"{event_ticker}-{s}" for s in _SERIES_BIN_SUFFIXES.get(series, [])]
        confirmed = []
        for c in candidates:
            try:
                b = get_with_retry(HIST_TRADES_EP, {"ticker": c, "min_ts": start_ts, "max_ts": end_ts, "limit": 1}, rate_sleep=rate_sleep)
                if b.get("trades"):
                    confirmed.append(c)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    continue
                raise
        return sorted(confirmed)

    return []


def fetch_candles(ticker, start_ts, end_ts, interval=1, rate_sleep=0.15) -> list[dict]:
    url  = CANDLES_EP_TMPL.format(ticker=ticker)
    try:
        body = get_with_retry(url, {"start_ts": start_ts, "end_ts": end_ts, "period_interval": interval}, rate_sleep=rate_sleep)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            log.debug("  %s: candlestick 404 (no data), skipping", ticker)
            return []
        raise
    if body.get("error"):
        log.warning("  %s: API error — %s (market not yet in historical archive?)",
                    ticker, body["error"].get("code", "unknown"))
        return []
    rows = []
    for c in (body.get("candlesticks") or []):
        ts  = c.get("end_period_ts")
        p   = c.get("price") or {}
        bid = c.get("yes_bid") or {}
        ask = c.get("yes_ask") or {}
        rows.append({
            "ticker":          ticker,
            "series_ticker":   ticker.split("-")[0],
            "trade_date":      datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat() if ts else None,
            "end_period_ts":   ts,
            "end_period_utc":  datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None,
            "interval_min":    interval,
            "price_open":      float(p["open"])     if p.get("open")     else None,
            "price_high":      float(p["high"])     if p.get("high")     else None,
            "price_low":       float(p["low"])      if p.get("low")      else None,
            "price_close":     float(p["close"])    if p.get("close")    else None,
            "price_mean":      float(p["mean"])     if p.get("mean")     else None,
            "price_previous":  float(p["previous"]) if p.get("previous") else None,
            "yes_bid_open":    float(bid["open"])   if bid.get("open")   else None,
            "yes_bid_high":    float(bid["high"])   if bid.get("high")   else None,
            "yes_bid_low":     float(bid["low"])    if bid.get("low")    else None,
            "yes_bid_close":   float(bid["close"])  if bid.get("close")  else None,
            "yes_ask_open":    float(ask["open"])   if ask.get("open")   else None,
            "yes_ask_high":    float(ask["high"])   if ask.get("high")   else None,
            "yes_ask_low":     float(ask["low"])    if ask.get("low")    else None,
            "yes_ask_close":   float(ask["close"])  if ask.get("close")  else None,
            "volume":          float(c["volume"])        if c.get("volume")        else 0.0,
            "open_interest":   float(c["open_interest"]) if c.get("open_interest") else None,
        })
    return rows


# ── Ducklake setup ────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS candles (
    -- identity
    ticker          VARCHAR      NOT NULL,
    series_ticker   VARCHAR      NOT NULL,   -- e.g. KXHIGHLAX
    trade_date      DATE         NOT NULL,   -- UTC date of the bar
    end_period_ts   BIGINT       NOT NULL,   -- Unix seconds (bar close)
    end_period_utc  TIMESTAMPTZ,
    interval_min    INTEGER      NOT NULL,   -- 1 | 60 | 1440

    -- trade price OHLC + extras
    price_open      DOUBLE,
    price_high      DOUBLE,
    price_low       DOUBLE,
    price_close     DOUBLE,
    price_mean      DOUBLE,   -- VWAP for the bar
    price_previous  DOUBLE,   -- prior bar's close (no lag join needed)

    -- order book bid / ask OHLC
    yes_bid_open    DOUBLE,
    yes_bid_high    DOUBLE,
    yes_bid_low     DOUBLE,
    yes_bid_close   DOUBLE,
    yes_ask_open    DOUBLE,
    yes_ask_high    DOUBLE,
    yes_ask_low     DOUBLE,
    yes_ask_close   DOUBLE,

    -- flow
    volume          DOUBLE,
    open_interest   DOUBLE
    -- dedup enforced via already_loaded() check; Ducklake doesn't support PK/UNIQUE
);
"""


def open_lake(catalog: Path, data_dir: Path) -> duckdb.DuckDBPyConnection:
    """Attach (or create) the Ducklake catalog and return an open connection."""
    data_dir.mkdir(parents=True, exist_ok=True)
    catalog.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("INSTALL ducklake; LOAD ducklake;")
    con.execute(
        f"ATTACH 'ducklake:{catalog}' AS lake "
        f"(DATA_PATH '{data_dir}');"
    )
    con.execute("USE lake;")
    con.execute(DDL)
    return con


def already_loaded(con: duckdb.DuckDBPyConnection, ticker: str, trade_date: date, interval: int) -> bool:
    """Return True if this (ticker, date, interval) partition already has rows."""
    result = con.execute(
        "SELECT COUNT(*) FROM candles WHERE ticker = ? AND trade_date = ? AND interval_min = ?",
        [ticker, trade_date.isoformat(), interval],
    ).fetchone()
    return result[0] > 0


def upsert_rows(con: duckdb.DuckDBPyConnection, rows: list[dict]) -> int:
    """Insert rows into candles table. Caller must ensure no duplicates via already_loaded()."""
    if not rows:
        return 0
    cols = list(rows[0].keys())
    placeholders = ", ".join("?" * len(cols))
    col_list = ", ".join(cols)
    sql = f"INSERT INTO candles ({col_list}) VALUES ({placeholders})"
    tuples = [tuple(r[c] for c in cols) for r in rows]
    con.executemany(sql, tuples)
    return len(tuples)


# ── orchestration ─────────────────────────────────────────────────────────────

def ingest_event_date(
    con: duckdb.DuckDBPyConnection,
    event_ticker: str,
    d: date,
    interval: int,
    probe: bool,
    explicit: list[str] | None,
    rate_sleep: float,
    skip_existing: bool,
) -> int:
    start_ts, end_ts = day_bounds(d)
    tickers = discover_tickers(event_ticker, start_ts, end_ts,
                               probe=probe, explicit=explicit, rate_sleep=rate_sleep)
    if not tickers:
        log.warning("No tickers found for %s on %s", event_ticker, d)
        return 0

    total = 0
    for ticker in tickers:
        if skip_existing and already_loaded(con, ticker, d, interval):
            log.info("  SKIP %s %s (already loaded)", ticker, d)
            continue

        rows = fetch_candles(ticker, start_ts, end_ts, interval, rate_sleep)
        if not rows:
            log.info("  %s %s: 0 candles", ticker, d)
            continue

        n = upsert_rows(con, rows)
        log.info("  %s %s: %d candles inserted (%d fetched)", ticker, d, n, len(rows))
        total += n

    return total


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def event_ticker_for_date(series: str, d: date) -> str:
    """Build event ticker from series + date, e.g. KXHIGHLAX + 2025-07-02 → KXHIGHLAX-25JUL02."""
    month_abbr = d.strftime("%b").upper()           # JUL
    year_2     = d.strftime("%y")                   # 25
    day_2      = d.strftime("%d")                   # 02
    return f"{series}-{year_2}{month_abbr}{day_2}"  # KXHIGHLAX-25JUL02


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Ingest Kalshi candlesticks into a local Ducklake table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Target specification
    g = p.add_mutually_exclusive_group()
    g.add_argument("--event", help="Single event ticker, e.g. KXHIGHLAX-25JUL02")
    g.add_argument("--series", help="Series ticker for date-range mode, e.g. KXHIGHLAX")

    p.add_argument("--date",  help="Single UTC date YYYY-MM-DD (use with --event)")
    p.add_argument("--start", help="Start date YYYY-MM-DD (use with --series)")
    p.add_argument("--end",   help="End date YYYY-MM-DD   (use with --series)")

    p.add_argument("--tickers", help="Comma-separated explicit ticker list")
    p.add_argument("--probe-tickers", action="store_true",
                   help="Probe historical trades API to discover settled-event tickers")
    p.add_argument("--interval", type=int, default=1, choices=[1, 60, 1440],
                   help="Candle width in minutes (default: 1)")
    p.add_argument("--no-skip", action="store_true",
                   help="Re-ingest even if (ticker, date, interval) already loaded")

    # Storage
    p.add_argument("--catalog",  default="data/kalshi.ducklake",
                   help="Path to .ducklake catalog file")
    p.add_argument("--data-dir", default="data/kalshi_data",
                   help="Directory for Parquet data files")

    # Misc
    p.add_argument("--rate",    type=float, default=0.15)
    p.add_argument("--query",   action="store_true",
                   help="Print a summary query and exit (no ingestion)")
    p.add_argument("--debug",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    catalog  = Path(args.catalog)
    data_dir = Path(args.data_dir)

    con = open_lake(catalog, data_dir)

    # ── summary query mode ────────────────────────────────────────────────────
    if args.query:
        print("\n── Ducklake summary ─────────────────────────────────────────")
        print(con.execute("""
            SELECT
                series_ticker,
                interval_min,
                MIN(trade_date)   AS first_date,
                MAX(trade_date)   AS last_date,
                COUNT(DISTINCT trade_date) AS n_dates,
                COUNT(DISTINCT ticker)     AS n_tickers,
                COUNT(*)                   AS n_candles,
                SUM(volume)                AS total_volume
            FROM candles
            GROUP BY series_ticker, interval_min
            ORDER BY series_ticker, interval_min
        """).fetchdf().to_string(index=False))
        print()
        con.close()
        return

    # ── build work list ───────────────────────────────────────────────────────
    explicit = [t.strip() for t in args.tickers.split(",") if t.strip()] if args.tickers else None

    if args.event and args.date:
        try:
            d = date.fromisoformat(args.date)
        except ValueError:
            log.error("--date must be YYYY-MM-DD"); sys.exit(1)
        work = [(args.event.upper(), d)]

    elif args.series and args.start and args.end:
        try:
            start = date.fromisoformat(args.start)
            end   = date.fromisoformat(args.end)
        except ValueError:
            log.error("--start/--end must be YYYY-MM-DD"); sys.exit(1)
        work = [
            (event_ticker_for_date(args.series.upper(), d), d)
            for d in daterange(start, end)
        ]

    else:
        log.error("Provide either (--event + --date) or (--series + --start + --end)")
        sys.exit(1)

    # ── ingest ────────────────────────────────────────────────────────────────
    grand_total = 0
    for event_ticker, d in work:
        log.info("Ingesting %s  %s  interval=%dm", event_ticker, d, args.interval)
        n = ingest_event_date(
            con, event_ticker, d,
            interval=args.interval,
            probe=args.probe_tickers,
            explicit=explicit,
            rate_sleep=args.rate,
            skip_existing=not args.no_skip,
        )
        grand_total += n

    log.info("Total candles inserted this run: %d", grand_total)
    con.close()


if __name__ == "__main__":
    main()
