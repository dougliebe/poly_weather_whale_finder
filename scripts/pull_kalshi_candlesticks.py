"""
pull_kalshi_candlesticks.py
────────────────────────────────────────────────────────────────────────────
Pulls OHLC candlestick data for every market ticker in a Kalshi event on a
given UTC date range, using the public Historical Candlesticks API, and
writes a single CSV.

Usage
-----
    # 1-minute bars, active/recent event
    python scripts/pull_kalshi_candlesticks.py \
        --event KXHIGHLAX-26JUL02 \
        --date  2026-07-02

    # 1-minute bars, settled event (probe to find tickers)
    python scripts/pull_kalshi_candlesticks.py \
        --event KXHIGHLAX-25JUL02 \
        --date  2025-07-02 \
        --probe-tickers

    # hourly bars
    python scripts/pull_kalshi_candlesticks.py \
        --event KXHIGHLAX-25JUL02 --date 2025-07-02 \
        --probe-tickers --interval 60

    # explicit tickers
    python scripts/pull_kalshi_candlesticks.py \
        --event KXHIGHLAX-25JUL02 --date 2025-07-02 \
        --tickers KXHIGHLAX-25JUL02-B70.5,KXHIGHLAX-25JUL02-B72.5

Output CSV columns
------------------
    ticker, end_period_ts, end_period_utc, interval_min,
    price_open, price_high, price_low, price_close, price_mean, price_previous,
    yes_bid_open, yes_bid_high, yes_bid_low, yes_bid_close,
    yes_ask_open, yes_ask_high, yes_ask_low, yes_ask_close,
    volume, open_interest

API reference
-------------
GET /trade-api/v2/historical/markets/{ticker}/candlesticks
  ?start_ts=<unix>&end_ts=<unix>&period_interval=<1|60|1440>
No authentication required.
"""

import argparse
import csv
import sys
import time
import logging
from datetime import datetime, timezone, date
from pathlib import Path

import requests

# ── constants ─────────────────────────────────────────────────────────────────

KALSHI_BASE      = "https://external-api.kalshi.com/trade-api/v2"
MARKETS_EP       = f"{KALSHI_BASE}/markets"
EVENTS_EP        = f"{KALSHI_BASE}/events"
HIST_TRADES_EP   = f"{KALSHI_BASE}/historical/trades"
CANDLES_EP_TMPL  = f"{KALSHI_BASE}/historical/markets/{{ticker}}/candlesticks"

VALID_INTERVALS = {1, 60, 1440}

CSV_FIELDS = [
    "ticker", "end_period_ts", "end_period_utc", "interval_min",
    # trade price OHLC
    "price_open", "price_high", "price_low", "price_close",
    "price_mean", "price_previous",
    # order-book bid/ask OHLC
    "yes_bid_open", "yes_bid_high", "yes_bid_low", "yes_bid_close",
    "yes_ask_open", "yes_ask_high", "yes_ask_low", "yes_ask_close",
    # volume / OI
    "volume", "open_interest",
]

# Same probe list as pull_kalshi_event_trades.py
_KXHIGHLAX_BIN_SUFFIXES = [
    "T60", "T62", "T64", "T66", "T67", "T68", "T69", "T70",
    "B62.5", "B64.5", "B66.5", "B68.5", "B70.5",
    "B72.5", "B74.5", "B76.5", "B78.5", "B80.5",
    "B82.5", "B84.5",
    "T74", "T75", "T76", "T77", "T78", "T80", "T82", "T84", "T86",
]

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── HTTP helper ───────────────────────────────────────────────────────────────

def get_with_retry(url: str, params: dict, max_retries: int, rate_sleep: float) -> dict:
    backoff = 2.0
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
        except requests.ConnectionError as exc:
            if attempt == max_retries:
                raise
            log.warning("Connection error (%s), retry %d/%d in %.1fs",
                        exc, attempt + 1, max_retries, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue

        if resp.status_code == 200:
            time.sleep(rate_sleep)
            return resp.json()

        if resp.status_code == 429 or resp.status_code >= 500:
            wait = max(float(resp.headers.get("Retry-After", backoff)), backoff)
            if attempt == max_retries:
                resp.raise_for_status()
            log.warning("HTTP %d, retry %d/%d in %.1fs",
                        resp.status_code, attempt + 1, max_retries, wait)
            time.sleep(wait)
            backoff = min(backoff * 2, 60)
            continue

        resp.raise_for_status()

    raise RuntimeError("Exhausted retries")


# ── ticker discovery (shared logic with pull_kalshi_event_trades.py) ──────────

def _try_markets_endpoint(event_ticker: str, max_retries: int, rate_sleep: float) -> list[str]:
    tickers, offset, limit = [], 0, 100
    while True:
        body = get_with_retry(MARKETS_EP,
                              {"event_ticker": event_ticker, "limit": limit, "offset": offset},
                              max_retries, rate_sleep)
        markets = body.get("markets") or []
        tickers.extend(m["ticker"] for m in markets if "ticker" in m)
        if len(markets) < limit:
            break
        offset += limit
    return sorted(tickers)


def _try_events_endpoint(event_ticker: str, max_retries: int, rate_sleep: float) -> list[str]:
    try:
        body = get_with_retry(f"{EVENTS_EP}/{event_ticker}", {}, max_retries, rate_sleep)
        return sorted(m["ticker"] for m in (body.get("markets") or []) if "ticker" in m)
    except requests.HTTPError:
        return []


def _probe_via_trades(
    event_ticker: str, min_ts: int, max_ts: int, max_retries: int, rate_sleep: float
) -> list[str]:
    """Probe by pinging the historical trades endpoint (1 result) for each candidate."""
    series = event_ticker.split("-")[0]
    if series != "KXHIGHLAX":
        log.warning("No probe pattern for series %r. Use --tickers.", series)
        return []

    candidates = [f"{event_ticker}-{s}" for s in _KXHIGHLAX_BIN_SUFFIXES]
    log.info("Probing %d candidates via historical trades API...", len(candidates))
    confirmed = []
    for candidate in candidates:
        try:
            body = get_with_retry(
                HIST_TRADES_EP,
                {"ticker": candidate, "min_ts": min_ts, "max_ts": max_ts, "limit": 1},
                max_retries, rate_sleep,
            )
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                log.debug("  — %s (not found)", candidate)
                continue
            raise
        if body.get("trades"):
            log.info("  ✓ %s", candidate)
            confirmed.append(candidate)
        else:
            log.debug("  — %s (no trades)", candidate)
    return sorted(confirmed)


def discover_tickers(
    event_ticker: str,
    min_ts: int,
    max_ts: int,
    explicit: list[str] | None,
    probe: bool,
    max_retries: int,
    rate_sleep: float,
) -> list[str]:
    if explicit:
        log.info("Using %d explicit tickers.", len(explicit))
        return sorted(explicit)

    log.info("Strategy 1: GET /markets?event_ticker=%s", event_ticker)
    t = _try_markets_endpoint(event_ticker, max_retries, rate_sleep)
    if t:
        log.info("Found %d tickers via /markets.", len(t)); return t

    log.info("Strategy 2: GET /events/%s", event_ticker)
    t = _try_events_endpoint(event_ticker, max_retries, rate_sleep)
    if t:
        log.info("Found %d tickers via /events.", len(t)); return t

    if probe:
        log.info("Strategy 3: probing historical trades API...")
        t = _probe_via_trades(event_ticker, min_ts, max_ts, max_retries, rate_sleep)
        if t:
            log.info("Confirmed %d tickers via probe.", len(t)); return t
        log.error("Probe found no tickers.")
    else:
        log.error("No tickers found. For settled events use --probe-tickers or --tickers.")
    return []


# ── candlestick fetch ─────────────────────────────────────────────────────────

def _fp(val) -> str:
    """Return the string value or '' for None/missing."""
    return str(val) if val is not None else ""


def fetch_candlesticks(
    ticker: str,
    start_ts: int,
    end_ts: int,
    interval: int,
    max_retries: int,
    rate_sleep: float,
) -> list[dict]:
    """Fetch all candlesticks for one ticker. API returns all bars in one call (no cursor)."""
    url  = CANDLES_EP_TMPL.format(ticker=ticker)
    body = get_with_retry(
        url,
        {"start_ts": start_ts, "end_ts": end_ts, "period_interval": interval},
        max_retries, rate_sleep,
    )
    candles = body.get("candlesticks") or []
    log.info("  %s: %d candles (%d-min)", ticker, len(candles), interval)

    rows = []
    for c in candles:
        ts  = c.get("end_period_ts")
        utc = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else ""
        p   = c.get("price") or {}
        bid = c.get("yes_bid") or {}
        ask = c.get("yes_ask") or {}
        rows.append({
            "ticker":          ticker,
            "end_period_ts":   ts or "",
            "end_period_utc":  utc,
            "interval_min":    interval,
            # trade price
            "price_open":      _fp(p.get("open")),
            "price_high":      _fp(p.get("high")),
            "price_low":       _fp(p.get("low")),
            "price_close":     _fp(p.get("close")),
            "price_mean":      _fp(p.get("mean")),
            "price_previous":  _fp(p.get("previous")),
            # order book
            "yes_bid_open":    _fp(bid.get("open")),
            "yes_bid_high":    _fp(bid.get("high")),
            "yes_bid_low":     _fp(bid.get("low")),
            "yes_bid_close":   _fp(bid.get("close")),
            "yes_ask_open":    _fp(ask.get("open")),
            "yes_ask_high":    _fp(ask.get("high")),
            "yes_ask_low":     _fp(ask.get("low")),
            "yes_ask_close":   _fp(ask.get("close")),
            # volume / OI
            "volume":          _fp(c.get("volume")),
            "open_interest":   _fp(c.get("open_interest")),
        })
    return rows


# ── main pipeline ─────────────────────────────────────────────────────────────

def day_unix_bounds(d: date) -> tuple[int, int]:
    start = datetime(d.year, d.month, d.day,  0,  0,  0, tzinfo=timezone.utc)
    end   = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)
    return int(start.timestamp()), int(end.timestamp())


def pull_candlesticks_to_csv(
    event_ticker: str,
    trade_date: date,
    out_path: Path,
    interval: int,
    rate_sleep: float,
    max_retries: int,
    explicit_tickers: list[str] | None,
    probe: bool,
    dry_run: bool,
) -> None:
    start_ts, end_ts = day_unix_bounds(trade_date)
    log.info("Date window: %s  start_ts=%d  end_ts=%d  interval=%d min",
             trade_date, start_ts, end_ts, interval)

    tickers = discover_tickers(
        event_ticker, start_ts, end_ts,
        explicit_tickers, probe, max_retries, rate_sleep,
    )
    if not tickers:
        sys.exit(1)

    log.info("Tickers (%d): %s", len(tickers), ", ".join(tickers))

    if dry_run:
        print(f"\nDry-run. {len(tickers)} tickers:")
        for t in tickers:
            print(f"  {t}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for ticker in tickers:
            rows = fetch_candlesticks(ticker, start_ts, end_ts, interval, max_retries, rate_sleep)
            writer.writerows(rows)
            total += len(rows)

    log.info("Done. %d candles → %s", total, out_path)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pull Kalshi historical candlesticks for an event/date to CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--event", required=True, help="Event ticker, e.g. KXHIGHLAX-25JUL02")
    p.add_argument("--date",  required=True, help="UTC date, YYYY-MM-DD")
    p.add_argument("--interval", type=int, default=1, choices=[1, 60, 1440],
                   help="Candle width in minutes: 1, 60, or 1440 (default: 1)")
    p.add_argument("--out", default=None,
                   help="Output CSV path (default: <EVENT>_<date>_candles_<interval>m.csv)")
    p.add_argument("--tickers", default=None,
                   help="Comma-separated explicit ticker list")
    p.add_argument("--probe-tickers", action="store_true",
                   help="Probe historical trades API to discover tickers for settled events")
    p.add_argument("--rate", type=float, default=0.15, metavar="SEC",
                   help="Min seconds between API calls (default: 0.15)")
    p.add_argument("--retries", type=int, default=6,
                   help="Max retries on 429/5xx (default: 6)")
    p.add_argument("--dry-run", action="store_true",
                   help="Discover tickers only; do not fetch candles")
    p.add_argument("--debug", action="store_true", help="Verbose logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        trade_date = date.fromisoformat(args.date)
    except ValueError:
        log.error("--date must be YYYY-MM-DD, got %r", args.date)
        sys.exit(1)

    explicit = (
        [t.strip() for t in args.tickers.split(",") if t.strip()]
        if args.tickers else None
    )

    out_path = Path(args.out) if args.out else Path(
        f"{args.event.upper()}_{args.date}_candles_{args.interval}m.csv"
    )

    pull_candlesticks_to_csv(
        event_ticker=args.event.upper(),
        trade_date=trade_date,
        out_path=out_path,
        interval=args.interval,
        rate_sleep=args.rate,
        max_retries=args.retries,
        explicit_tickers=explicit,
        probe=args.probe_tickers,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
