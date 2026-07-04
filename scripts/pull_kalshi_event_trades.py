"""
pull_kalshi_event_trades.py
────────────────────────────────────────────────────────────────────────────
Pulls all trades for every market ticker in a Kalshi event on a given UTC
date, using the public Historical Trades API, and writes a single CSV.

Usage
-----
    # Auto-discover tickers (works for active/recent events)
    python scripts/pull_kalshi_event_trades.py \
        --event KXHIGHLAX-26JUL02 \
        --date  2026-07-02

    # Settled events: auto-probe common patterns or pass tickers explicitly
    python scripts/pull_kalshi_event_trades.py \
        --event KXHIGHLAX-25JUL02 \
        --date  2025-07-02 \
        --probe-tickers          # probe KXHIGHLAX bin patterns automatically

    python scripts/pull_kalshi_event_trades.py \
        --event KXHIGHLAX-25JUL02 \
        --date  2025-07-02 \
        --tickers KXHIGHLAX-25JUL02-B70.5,KXHIGHLAX-25JUL02-B72.5,KXHIGHLAX-25JUL02-T73

    # Dry-run: discover/probe tickers only, no trade fetch
    python scripts/pull_kalshi_event_trades.py \
        --event KXHIGHLAX-25JUL02 --date 2025-07-02 --probe-tickers --dry-run

Output CSV columns
------------------
    trade_id, ticker, created_time, yes_price, no_price,
    count_fp, taker_outcome_side, taker_book_side, is_block_trade

Ticker discovery strategy (in order)
-------------------------------------
1. GET /markets?event_ticker=<event>          – works for active/recent events
2. GET /events/<event>                        – fallback; also recent only
3. --probe-tickers flag                       – probes candidate bin names via
                                               the historical API itself (a
                                               ticker with no trades returns
                                               an empty array, not an error)
4. --tickers CSV                              – explicit override, always wins

API reference
-------------
GET https://external-api.kalshi.com/trade-api/v2/historical/trades
  ?ticker=<t>&min_ts=<unix>&max_ts=<unix>&limit=<n>&cursor=<cur>
No authentication required for public market data.
"""

import argparse
import csv
import sys
import time
import logging
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Iterator

import requests

# ── constants ─────────────────────────────────────────────────────────────────

KALSHI_BASE    = "https://external-api.kalshi.com/trade-api/v2"
MARKETS_EP     = f"{KALSHI_BASE}/markets"
EVENTS_EP      = f"{KALSHI_BASE}/events"
HIST_TRADES_EP = f"{KALSHI_BASE}/historical/trades"

CSV_FIELDS = [
    "trade_id", "ticker", "created_time",
    "yes_price", "no_price", "count_fp",
    "taker_outcome_side", "taker_book_side", "is_block_trade",
]

# Standard KXHIGHLAX bin pattern: 2°F-wide bins.
# Format: KXHIGHLAX-<YY><MON><DD>-<BIN>
#
# The "below" and "above" tail bins use Tx naming where x is the threshold:
#   "67° or below" → T67   "76° or above" → T76
#   "69° or below" → T69   "78° or above" → T78  etc.
# The between bins use Bx.5 (midpoint of the 2°F range):
#   "68° to 69°" → B68.5   "70° to 71°" → B70.5  etc.
#
# We probe a wide range; tickers not offered on a given event simply 404.
_KXHIGHLAX_BIN_SUFFIXES = [
    # below-threshold tail bins (Kalshi names the floor, e.g. T67 = ≤67°F)
    "T60", "T62", "T64", "T66", "T67", "T68", "T69", "T70",
    # between bins (2°F wide, labelled by midpoint)
    "B62.5", "B64.5", "B66.5", "B68.5", "B70.5",
    "B72.5", "B74.5", "B76.5", "B78.5", "B80.5",
    "B82.5", "B84.5",
    # above-threshold tail bins
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

def get_with_retry(
    url: str,
    params: dict,
    max_retries: int,
    rate_sleep: float,
) -> dict:
    """GET with exponential backoff on 429 and 5xx. Raises on 4xx."""
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
            retry_after = float(resp.headers.get("Retry-After", backoff))
            wait = max(retry_after, backoff)
            if attempt == max_retries:
                resp.raise_for_status()
            log.warning("HTTP %d, retry %d/%d in %.1fs",
                        resp.status_code, attempt + 1, max_retries, wait)
            time.sleep(wait)
            backoff = min(backoff * 2, 60)
            continue

        resp.raise_for_status()

    raise RuntimeError("Exhausted retries")


# ── ticker discovery ──────────────────────────────────────────────────────────

def _try_markets_endpoint(event_ticker: str, max_retries: int, rate_sleep: float) -> list[str]:
    """Strategy 1: GET /markets?event_ticker=... (works for active events)."""
    tickers: list[str] = []
    offset, limit = 0, 100
    while True:
        body = get_with_retry(
            MARKETS_EP,
            {"event_ticker": event_ticker, "limit": limit, "offset": offset},
            max_retries, rate_sleep,
        )
        markets = body.get("markets") or []
        tickers.extend(m["ticker"] for m in markets if "ticker" in m)
        if len(markets) < limit:
            break
        offset += limit
    return sorted(tickers)


def _try_events_endpoint(event_ticker: str, max_retries: int, rate_sleep: float) -> list[str]:
    """Strategy 2: GET /events/<event_ticker> (also mostly active-only)."""
    try:
        body = get_with_retry(
            f"{EVENTS_EP}/{event_ticker}",
            {}, max_retries, rate_sleep,
        )
        markets = body.get("markets") or []
        return sorted(m["ticker"] for m in markets if "ticker" in m)
    except requests.HTTPError:
        return []


def _probe_historical(
    event_ticker: str,
    min_ts: int,
    max_ts: int,
    max_retries: int,
    rate_sleep: float,
) -> list[str]:
    """
    Strategy 3: Probe candidate tickers against the historical API.
    A ticker with no trades on that day returns an empty array (not 404),
    so we keep only tickers that return ≥1 trade.
    Works even for fully settled/archived events.
    """
    # Build candidate list from the known KXHIGHLAX bin pattern
    series = event_ticker.split("-")[0]   # e.g. "KXHIGHLAX"
    date_part = "-".join(event_ticker.split("-")[1:])  # e.g. "25JUL02"

    if series == "KXHIGHLAX":
        candidates = [
            f"{event_ticker}-{suffix}" for suffix in _KXHIGHLAX_BIN_SUFFIXES
        ]
    else:
        log.warning(
            "No probe pattern known for series %r. "
            "Use --tickers to supply tickers explicitly.", series
        )
        return []

    log.info("Probing %d candidate tickers via historical API...", len(candidates))
    confirmed: list[str] = []
    for candidate in candidates:
        try:
            body = get_with_retry(
                HIST_TRADES_EP,
                {"ticker": candidate, "min_ts": min_ts, "max_ts": max_ts, "limit": 1},
                max_retries, rate_sleep,
            )
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                log.debug("  — %s (ticker not found)", candidate)
                continue
            raise
        trades = body.get("trades") or []
        if trades:
            log.info("  ✓ %s", candidate)
            confirmed.append(candidate)
        else:
            log.debug("  — %s (no trades on date)", candidate)

    return sorted(confirmed)


def discover_tickers(
    event_ticker: str,
    min_ts: int,
    max_ts: int,
    explicit_tickers: list[str] | None,
    probe: bool,
    max_retries: int,
    rate_sleep: float,
) -> list[str]:
    """
    Return the ticker list using the best available strategy.
    Explicit --tickers always wins; otherwise tries API endpoints then probe.
    """
    # Explicit override
    if explicit_tickers:
        log.info("Using %d explicitly supplied tickers.", len(explicit_tickers))
        return sorted(explicit_tickers)

    # Strategy 1: /markets endpoint
    log.info("Strategy 1: GET /markets?event_ticker=%s", event_ticker)
    tickers = _try_markets_endpoint(event_ticker, max_retries, rate_sleep)
    if tickers:
        log.info("Found %d tickers via /markets.", len(tickers))
        return tickers

    # Strategy 2: /events endpoint
    log.info("Strategy 2: GET /events/%s", event_ticker)
    tickers = _try_events_endpoint(event_ticker, max_retries, rate_sleep)
    if tickers:
        log.info("Found %d tickers via /events.", len(tickers))
        return tickers

    # Strategy 3: probe historical API
    if probe:
        log.info("Strategy 3: probing historical API for active tickers...")
        tickers = _probe_historical(event_ticker, min_ts, max_ts, max_retries, rate_sleep)
        if tickers:
            log.info("Confirmed %d tickers via historical probe.", len(tickers))
            return tickers
        log.error("Probe found no tickers. Check --event and --date.")
    else:
        log.error(
            "No tickers found via API endpoints. "
            "For settled events, re-run with --probe-tickers or --tickers <list>."
        )

    return []


# ── trade pagination ──────────────────────────────────────────────────────────

def day_unix_bounds(d: date) -> tuple[int, int]:
    start = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
    end   = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)
    return int(start.timestamp()), int(end.timestamp())


def iter_trades(
    ticker: str,
    min_ts: int,
    max_ts: int,
    page_size: int,
    max_retries: int,
    rate_sleep: float,
) -> Iterator[dict]:
    """Yield normalised trade dicts for one ticker, fully paginated."""
    cursor: str | None = None
    page = total = 0

    while True:
        params: dict = {
            "ticker": ticker,
            "min_ts": min_ts,
            "max_ts": max_ts,
            "limit":  page_size,
        }
        if cursor:
            params["cursor"] = cursor

        body   = get_with_retry(HIST_TRADES_EP, params, max_retries, rate_sleep)
        trades = body.get("trades") or []
        page  += 1

        if not trades:
            log.debug("  %s page %d: 0 trades (done)", ticker, page)
            break

        total += len(trades)
        log.debug("  %s page %d: %d trades (running total: %d)",
                  ticker, page, len(trades), total)

        for raw in trades:
            yield {
                "trade_id":           raw.get("trade_id", ""),
                "ticker":             raw.get("ticker", ticker),
                "created_time":       raw.get("created_time", ""),
                "yes_price":          raw.get("yes_price_dollars", ""),
                "no_price":           raw.get("no_price_dollars", ""),
                "count_fp":           raw.get("count_fp", ""),
                "taker_outcome_side": raw.get("taker_outcome_side", ""),
                "taker_book_side":    raw.get("taker_book_side", ""),
                "is_block_trade":     raw.get("is_block_trade", ""),
            }

        next_cursor = body.get("cursor") or ""
        if not next_cursor:
            break
        cursor = next_cursor

    log.info("  %s: %d trades", ticker, total)


# ── main pipeline ─────────────────────────────────────────────────────────────

def pull_event_to_csv(
    event_ticker: str,
    trade_date: date,
    out_path: Path,
    page_size: int,
    rate_sleep: float,
    max_retries: int,
    explicit_tickers: list[str] | None,
    probe: bool,
    dry_run: bool,
) -> None:
    min_ts, max_ts = day_unix_bounds(trade_date)
    log.info("Date window: %s  →  min_ts=%d  max_ts=%d", trade_date, min_ts, max_ts)

    tickers = discover_tickers(
        event_ticker, min_ts, max_ts,
        explicit_tickers, probe, max_retries, rate_sleep,
    )

    if not tickers:
        sys.exit(1)

    log.info("Tickers (%d): %s", len(tickers), ", ".join(tickers))

    if dry_run:
        print(f"\nDry-run complete. {len(tickers)} tickers found:")
        for t in tickers:
            print(f"  {t}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for ticker in tickers:
            log.info("Fetching: %s", ticker)
            for trade in iter_trades(ticker, min_ts, max_ts, page_size, max_retries, rate_sleep):
                writer.writerow(trade)
                rows_written += 1

    log.info("Done. %d trades → %s", rows_written, out_path)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pull all Kalshi historical trades for an event on a given UTC date.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--event", required=True,
                   help="Event ticker, e.g. KXHIGHLAX-25JUL02")
    p.add_argument("--date", required=True,
                   help="UTC date to pull, YYYY-MM-DD")
    p.add_argument("--out", default=None,
                   help="Output CSV path (default: <EVENT>_<date>_trades.csv)")
    p.add_argument("--tickers", default=None,
                   help="Comma-separated explicit ticker list (skips auto-discovery)")
    p.add_argument("--probe-tickers", action="store_true",
                   help="Probe historical API to discover tickers for settled events")
    p.add_argument("--page-size", type=int, default=1000, metavar="N",
                   help="Trades per API page, max 1000 (default: 1000)")
    p.add_argument("--rate", type=float, default=0.15, metavar="SEC",
                   help="Min seconds between API calls (default: 0.15)")
    p.add_argument("--retries", type=int, default=6,
                   help="Max retries on 429/5xx (default: 6)")
    p.add_argument("--dry-run", action="store_true",
                   help="Discover tickers only; do not fetch trades")
    p.add_argument("--debug", action="store_true",
                   help="Verbose per-page logging")
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

    if not (1 <= args.page_size <= 1000):
        log.error("--page-size must be 1–1000")
        sys.exit(1)

    explicit_tickers = (
        [t.strip() for t in args.tickers.split(",") if t.strip()]
        if args.tickers else None
    )

    out_path = Path(args.out) if args.out else Path(
        f"{args.event.upper()}_{args.date}_trades.csv"
    )

    pull_event_to_csv(
        event_ticker=args.event.upper(),
        trade_date=trade_date,
        out_path=out_path,
        page_size=args.page_size,
        rate_sleep=args.rate,
        max_retries=args.retries,
        explicit_tickers=explicit_tickers,
        probe=args.probe_tickers,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
