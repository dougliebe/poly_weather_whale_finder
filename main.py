#!/usr/bin/env python3
"""
poly_weather_whale_finder — main entry point

Pulls weather markets from Polymarket, fetches all publicly visible trade history,
and reports on wallets that appear to front-run price moves.

Usage:
    python main.py [--days 3] [--min-move 0.05] [--save] [--pages 50]
"""

import argparse
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from src.markets import get_weather_events, get_weather_markets_from_events
from src.trades import get_weather_trades, normalize_trade
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
    p.add_argument("--days", type=int, default=3, help="How many days back to fetch (default: 3)")
    p.add_argument("--pages", type=int, default=5, help="Max pages per wallet from activity feed (500 trades/page)")
    p.add_argument("--min-move", type=float, default=0.05, help="Min hourly price move to flag (default: 0.05)")
    p.add_argument("--lead-minutes", type=int, default=15, help="Lead window before a move (default: 15)")
    p.add_argument("--save", action="store_true", help="Save raw and processed data to data/")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    min_timestamp = int((datetime.now(timezone.utc) - timedelta(days=args.days)).timestamp())

    # ── 1. Discover weather markets ────────────────────────────────────────
    print("Fetching weather events from Gamma API…")
    events = get_weather_events(limit=100)
    markets = get_weather_markets_from_events(events)

    # Build a set of all known weather token IDs for fast filtering
    token_id_set: set[str] = set()
    token_meta: dict[str, dict] = {}   # token_id → {question, outcome, condition_id}
    for m in markets:
        for t in m["tokens"]:
            tid = t["token_id"]
            token_id_set.add(tid)
            token_meta[tid] = {
                "market_question": m["question"],
                "outcome": t["outcome"],
                "condition_id": m["condition_id"],
            }

    print(f"  {len(events)} events → {len(markets)} markets → {len(token_id_set)} outcome tokens")

    if args.save:
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        (DATA_RAW / "markets.json").write_text(json.dumps(markets, indent=2, default=str))

    # Preview top markets
    top = sorted(markets, key=lambda m: float(m.get("volume") or 0), reverse=True)[:5]
    for m in top:
        print(f"  • {m['question'][:75]}  vol=${float(m.get('volume') or 0):,.0f}")

    # ── 2. Fetch trades ────────────────────────────────────────────────────
    print(f"\nFetching trade history (last {args.days} days, wallet-by-wallet)…")
    raw_trades, wallets = get_weather_trades(
        token_ids=token_id_set,
        min_timestamp=min_timestamp,
        max_wallet_pages=args.pages,
    )

    # Filter by timestamp (belt-and-suspenders)
    raw_trades = [t for t in raw_trades if t.get("timestamp", 0) >= min_timestamp]

    print(f"  {len(raw_trades):,} weather trades from {len(wallets)} wallets")

    if not raw_trades:
        print("No trades in window. Try --days 7 or --pages 200.")
        return

    if args.save:
        (DATA_RAW / "trades_raw.json").write_text(json.dumps(raw_trades, indent=2, default=str))

    # ── 3. Normalize ───────────────────────────────────────────────────────
    normalized = [normalize_trade(t) for t in raw_trades]
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
