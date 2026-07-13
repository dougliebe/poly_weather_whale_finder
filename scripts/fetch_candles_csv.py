"""
fetch_candles_csv.py
────────────────────────────────────────────────────────────────────────────
Fetch Kalshi temperature-market candlesticks for all series/dates and write
to CSV files.  No database required — pure CSV output in data/candles_csv/.

Output layout:
    data/candles_csv/{SERIES}/{SERIES}-{YYMONDD}.csv

Usage:
    # All series, last 90 days
    python scripts/fetch_candles_csv.py --start 2026-05-08 --end 2026-07-13

    # One series
    python scripts/fetch_candles_csv.py --series KXHIGHLAX --start 2026-05-08 --end 2026-07-13

    # Combine all CSVs into one parquet (requires pyarrow/pandas)
    python scripts/fetch_candles_csv.py --combine
"""

import argparse
import logging
import sys
import time
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

import requests
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

KALSHI_BASE = "https://external-api.kalshi.com/trade-api/v2"
MARKETS_EP = f"{KALSHI_BASE}/markets"
CANDLES_HIST_TMPL = f"{KALSHI_BASE}/historical/markets/{{ticker}}/candlesticks"
CANDLES_SERIES_TMPL = f"{KALSHI_BASE}/series/{{series}}/markets/{{ticker}}/candlesticks"

# Series endpoint only has data from ~May 8, 2026; historical endpoint covers pre-2026
SERIES_ENDPOINT_CUTOFF = datetime(2026, 1, 1, tzinfo=timezone.utc)

# All known high-temp series
ALL_SERIES = [
    "KXHIGHLAX",    # Los Angeles
    "KXHIGHMIA",    # Miami
    "KXHIGHTDAL",   # Dallas
    "KXHIGHTSATX",  # San Antonio
    "KXHIGHAUS",    # Austin
    "KXHIGHTHOU",   # Houston
    "KXHIGHDEN",    # Denver
    "KXHIGHTPHX",   # Phoenix
]

OUT_DIR = Path("data/candles_csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def get_json(url, params, rate_sleep=0.15, max_retries=6):
    backoff = 2.0
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
        except requests.ConnectionError:
            if attempt == max_retries:
                raise
            time.sleep(backoff); backoff = min(backoff * 2, 60); continue

        if resp.status_code == 200:
            time.sleep(rate_sleep)
            return resp.json()

        if resp.status_code in (429,) or resp.status_code >= 500:
            wait = max(float(resp.headers.get("Retry-After", backoff)), backoff)
            if attempt == max_retries:
                resp.raise_for_status()
            log.warning("HTTP %d – retry %d/%d in %.1fs", resp.status_code, attempt + 1, max_retries, wait)
            time.sleep(wait); backoff = min(backoff * 2, 60); continue

        resp.raise_for_status()
    raise RuntimeError("Exhausted retries")


def _f(d, key):
    """Try key then key_dollars (series endpoint uses _dollars suffix)."""
    v = d.get(key) or d.get(f"{key}_dollars")
    return float(v) if v is not None else None


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_tickers(event_ticker: str) -> list[str]:
    """Return all market tickers for an event via the /markets API."""
    tickers = []
    cursor = None
    while True:
        params = {"event_ticker": event_ticker, "limit": 100}
        if cursor:
            params["cursor"] = cursor
        body = get_json(MARKETS_EP, params)
        markets = body.get("markets") or []
        tickers.extend(m["ticker"] for m in markets if "ticker" in m)
        cursor = body.get("cursor")
        if not cursor or not markets:
            break
    return sorted(tickers)


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_candles(ticker: str, day: date, interval: int = 1) -> list[dict]:
    start_ts = int(datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=timezone.utc).timestamp())
    end_ts = start_ts + 86400 - 1
    params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": interval}
    series = ticker.split("-")[0]

    # Choose endpoints to try (series endpoint only for 2026+)
    if datetime(day.year, day.month, day.day, tzinfo=timezone.utc) >= SERIES_ENDPOINT_CUTOFF:
        urls = [CANDLES_SERIES_TMPL.format(series=series, ticker=ticker)]
    else:
        urls = [
            CANDLES_HIST_TMPL.format(ticker=ticker),
            CANDLES_SERIES_TMPL.format(series=series, ticker=ticker),
        ]

    body = None
    for url in urls:
        try:
            b = get_json(url, params)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                continue
            raise
        if b.get("error"):
            continue
        if b.get("candlesticks") is not None:
            body = b
            break

    if body is None:
        return []

    rows = []
    for c in (body.get("candlesticks") or []):
        ts = c.get("end_period_ts")
        if ts is None:
            continue
        p = c.get("price") or {}
        bid = c.get("yes_bid") or {}
        ask = c.get("yes_ask") or {}
        rows.append({
            "ticker":          ticker,
            "series":          series,
            "trade_date":      day.isoformat(),
            "end_period_ts":   ts,
            "interval_min":    interval,
            "price_open":      _f(p, "open"),
            "price_high":      _f(p, "high"),
            "price_low":       _f(p, "low"),
            "price_close":     _f(p, "close"),
            "price_mean":      _f(p, "mean"),
            "price_previous":  _f(p, "previous"),
            "yes_bid_open":    _f(bid, "open"),
            "yes_bid_high":    _f(bid, "high"),
            "yes_bid_low":     _f(bid, "low"),
            "yes_bid_close":   _f(bid, "close"),
            "yes_ask_open":    _f(ask, "open"),
            "yes_ask_high":    _f(ask, "high"),
            "yes_ask_low":     _f(ask, "low"),
            "yes_ask_close":   _f(ask, "close"),
            "volume":          float(c.get("volume") or c.get("volume_fp") or 0),
            "open_interest":   (float(c["open_interest"]) if c.get("open_interest")
                                else float(c["open_interest_fp"]) if c.get("open_interest_fp")
                                else None),
        })
    return rows


# ── Main ingestion loop ───────────────────────────────────────────────────────

def ingest_series(series: str, start: date, end: date, overwrite: bool = False):
    out_dir = OUT_DIR / series
    out_dir.mkdir(parents=True, exist_ok=True)

    current = start
    total_rows = 0
    while current <= end:
        # Event ticker format: KXHIGHLAX-25JUL02
        ymd = current.strftime("%y%b%d").upper()
        event_ticker = f"{series}-{ymd}"
        csv_path = out_dir / f"{event_ticker}.csv"

        if csv_path.exists() and not overwrite:
            log.info("  skip %s (already exists)", csv_path.name)
            current += timedelta(days=1)
            continue

        # Discover tickers for this day
        tickers = discover_tickers(event_ticker)
        if not tickers:
            log.info("  %s: no markets found", event_ticker)
            current += timedelta(days=1)
            continue

        log.info("  %s: %d tickers", event_ticker, len(tickers))

        # Fetch candles for each ticker
        day_rows = []
        for ticker in tickers:
            rows = fetch_candles(ticker, current)
            day_rows.extend(rows)

        if day_rows:
            pd.DataFrame(day_rows).to_csv(csv_path, index=False)
            total_rows += len(day_rows)
            log.info("    -> %d rows -> %s", len(day_rows), csv_path)
        else:
            log.info("    -> no candle data for %s", event_ticker)

        current += timedelta(days=1)

    return total_rows


def combine_to_parquet(out_path: str = "data/candles_all.parquet"):
    """Merge all CSVs into one parquet file."""
    csvs = sorted(OUT_DIR.rglob("*.csv"))
    if not csvs:
        print("No CSVs found in", OUT_DIR)
        return
    print(f"Combining {len(csvs)} CSV files...")
    dfs = []
    for p in csvs:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f"  skip {p}: {e}")
    df = pd.concat(dfs, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df.to_parquet(out_path, index=False)
    print(f"Written {len(df):,} rows -> {out_path}")
    print(df.groupby("series")[["ticker"]].nunique().rename(columns={"ticker": "n_tickers"}))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Fetch Kalshi temp candles to CSV")
    p.add_argument("--series", nargs="+", default=ALL_SERIES,
                   help="Series to fetch (default: all known high-temp series)")
    p.add_argument("--start", default="2026-05-08", help="Start date YYYY-MM-DD")
    p.add_argument("--end",   default=date.today().isoformat(), help="End date YYYY-MM-DD")
    p.add_argument("--overwrite", action="store_true", help="Re-fetch even if CSV exists")
    p.add_argument("--combine", action="store_true", help="Combine all CSVs to parquet then exit")
    args = p.parse_args()

    if args.combine:
        combine_to_parquet()
        return

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    print(f"Fetching {len(args.series)} series from {start} to {end}")
    print(f"Output: {OUT_DIR.resolve()}/")
    print()

    grand_total = 0
    for series in args.series:
        log.info("=== %s ===", series)
        n = ingest_series(series, start, end, overwrite=args.overwrite)
        log.info("--- %s: %d rows ---", series, n)
        grand_total += n

    print(f"\nDone. {grand_total:,} total rows across {len(args.series)} series.")
    print(f"Run with --combine to merge into data/candles_all.parquet")


if __name__ == "__main__":
    main()
