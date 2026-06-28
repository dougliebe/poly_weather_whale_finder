#!/usr/bin/env python3
"""
poly_weather_whale_finder — main entry point

Fetches weather/temperature markets from Polymarket, pulls full trade history
via the authenticated CLOB API, and reports wallets that appear to front-run
price moves (potential fast-feed traders).

Requires env vars:
    POLY_API_KEY, POLY_API_SECRET, POLY_PASSPHRASE

Usage:
    python main.py [--days 30] [--min-move 0.05] [--lead-minutes 15] [--save]
"""

import argparse
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from src.markets import get_weather_events, get_weather_markets_from_events
from src.trades import get_trades_for_market, normalize_trade
from src.analysis import (
    trades_to_df,
    price_move_events,
    early_movers,
    print_report,
)

DATA_RAW = Path("data/raw")
DATA_PROC = Path("data/processed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find fast-feed weather traders on Polymarket")
    p.add_argument("--days", type=int, default=30, help="Days of trade history to fetch (default: 30)")
    p.add_argument("--min-move", type=float, default=0.05, help="Min hourly price move to flag (default: 0.05)")
    p.add_argument("--lead-minutes", type=int, default=15, help="Lead window before a move in minutes (default: 15)")
    p.add_argument("--save", action="store_true", help="Save raw and processed data to data/")
    p.add_argument("--limit", type=int, default=50, help="Max number of markets to fetch (default: 50)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    after_ts = int((datetime.now(timezone.utc) - timedelta(days=args.days)).timestamp())

    # ── 1. Discover weather markets ────────────────────────────────────────
    print("Fetching weather events from Gamma API…")
    events = get_weather_events(limit=args.limit)
    markets = get_weather_markets_from_events(events)
    print(f"  {len(events)} events → {len(markets)} outcome markets")

    if not markets:
        print("No weather markets found.")
        return

    if args.save:
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        (DATA_RAW / "markets.json").write_text(json.dumps(markets, indent=2, default=str))

    top = sorted(markets, key=lambda m: float(m.get("volume") or 0), reverse=True)[:5]
    for m in top:
        print(f"  • {m['question'][:75]}  vol=${float(m.get('volume') or 0):,.0f}")

    # ── 2. Fetch trades ────────────────────────────────────────────────────
    print(f"\nFetching trades via CLOB API (last {args.days} days)…")
    all_trades_raw = []
    for i, market in enumerate(markets):
        q = market["question"][:60]
        print(f"  [{i+1}/{len(markets)}] {q}…", end=" ", flush=True)
        raw = get_trades_for_market(market, after_ts=after_ts)
        print(f"{len(raw)} trades")
        all_trades_raw.extend(raw)
        time.sleep(0.1)

    print(f"\nTotal raw trades: {len(all_trades_raw):,}")

    if not all_trades_raw:
        print("No trades found. Check API credentials or try a wider date range.")
        return

    if args.save:
        (DATA_RAW / "trades_raw.json").write_text(json.dumps(all_trades_raw, indent=2, default=str))

    # ── 3. Normalize ───────────────────────────────────────────────────────
    normalized = [normalize_trade(t) for t in all_trades_raw]
    df = trades_to_df(normalized)

    if args.save:
        DATA_PROC.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PROC / "trades.csv", index=False)

    # ── 4. Analyze ─────────────────────────────────────────────────────────
    events_df = price_move_events(df, window="1h", min_move=args.min_move)
    movers = early_movers(df, events_df, lead_minutes=args.lead_minutes)

    if args.save and not movers.empty:
        movers.to_csv(DATA_PROC / "early_movers.csv", index=False)

    # ── 5. Report ──────────────────────────────────────────────────────────
    print_report(df, events_df, movers)


if __name__ == "__main__":
    main()
