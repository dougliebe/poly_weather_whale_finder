#!/usr/bin/env python3
"""
poly_weather_whale_finder — main entry point

Pulls weather markets from Polymarket, fetches all trade history,
and prints a report on wallets that appear to front-run price moves.

Usage:
    python main.py [--active-only] [--days 30] [--min-move 0.05] [--save]
"""

import argparse
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from src.markets import get_weather_markets, summarize_market
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
    p.add_argument("--active-only", action="store_true", help="Only fetch active (unclosed) markets")
    p.add_argument("--days", type=int, default=30, help="How many days back to fetch trades (default: 30)")
    p.add_argument("--min-move", type=float, default=0.05, help="Minimum hourly price move to flag (default: 0.05)")
    p.add_argument("--lead-minutes", type=int, default=15, help="Lead window before a move (default: 15)")
    p.add_argument("--save", action="store_true", help="Save raw and processed data to data/")
    p.add_argument("--limit", type=int, default=50, help="Max markets to fetch (default: 50)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    after_ts = int((datetime.now(timezone.utc) - timedelta(days=args.days)).timestamp())

    # ── 1. Discover weather markets ────────────────────────────────────────
    print(f"Fetching weather markets (limit={args.limit}, active_only={args.active_only})…")
    raw_markets = get_weather_markets(limit=args.limit, active_only=args.active_only)
    markets = [summarize_market(m) for m in raw_markets]
    print(f"  Found {len(markets)} markets")

    if not markets:
        print("No weather markets found. Try removing --active-only or check the API.")
        return

    if args.save:
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        (DATA_RAW / "markets.json").write_text(json.dumps(markets, indent=2))
        print(f"  Saved markets → {DATA_RAW / 'markets.json'}")

    for m in markets[:5]:
        print(f"  • {m['question'][:80]}  vol=${m.get('volume') or 0:,.0f}")

    # ── 2. Fetch trades ────────────────────────────────────────────────────
    all_trades_raw = []
    for i, market in enumerate(markets):
        q = market["question"][:60]
        print(f"\n[{i+1}/{len(markets)}] {q}…")
        raw = get_trades_for_market(market, after_ts=after_ts)
        print(f"  → {len(raw)} trades")
        all_trades_raw.extend(raw)
        time.sleep(0.3)

    print(f"\nTotal raw trades fetched: {len(all_trades_raw):,}")

    if not all_trades_raw:
        print("No trades found. The market may have no activity in this period.")
        return

    if args.save:
        (DATA_RAW / "trades_raw.json").write_text(json.dumps(all_trades_raw, indent=2))
        print(f"Saved raw trades → {DATA_RAW / 'trades_raw.json'}")

    # ── 3. Normalize ───────────────────────────────────────────────────────
    normalized = [normalize_trade(t) for t in all_trades_raw]
    df = trades_to_df(normalized)

    if args.save:
        DATA_PROC.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PROC / "trades.csv", index=False)
        print(f"Saved normalized trades → {DATA_PROC / 'trades.csv'}")

    # ── 4. Analyze ─────────────────────────────────────────────────────────
    events = price_move_events(df, window="1h", min_move=args.min_move)
    movers = early_movers(df, events, lead_minutes=args.lead_minutes)

    if args.save and not movers.empty:
        movers.to_csv(DATA_PROC / "early_movers.csv", index=False)
        print(f"Saved early movers → {DATA_PROC / 'early_movers.csv'}")

    # ── 5. Report ──────────────────────────────────────────────────────────
    print_report(df, events, movers)


if __name__ == "__main__":
    main()
