#!/usr/bin/env python3
"""
Fetch all trades for a Polymarket weather market day and optionally generate
a self-contained interactive HTML artifact.

Usage:
    python scripts/fetch_market_day.py <slug> [--save] [--artifact] [--no-taker-filter]

Examples:
    python scripts/fetch_market_day.py highest-temperature-in-dallas-on-july-19-2026 --save --artifact
    python scripts/fetch_market_day.py highest-temperature-in-dallas-on-july-15-2026 --artifact
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Allow running from repo root or scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_api import (
    get_event_by_slug,
    build_market_meta,
    day_timestamps,
    fetch_all_trades,
    enrich_trades,
)
from src.artifact import generate_artifact

DATA_RAW  = Path("data/raw")
DATA_ARTS = Path("data/artifacts")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Polymarket weather market day and build explorer")
    p.add_argument("slug", help="Event slug, e.g. highest-temperature-in-dallas-on-july-19-2026")
    p.add_argument("--save",           action="store_true", help="Save raw + enriched JSON to data/raw/")
    p.add_argument("--artifact",       action="store_true", help="Generate HTML artifact to data/artifacts/")
    p.add_argument("--no-taker-filter",action="store_true", help="Fetch all sides, not just taker trades")
    p.add_argument("--no-date-filter", action="store_true", help="Fetch all trades regardless of date (useful for upcoming markets)")
    p.add_argument("--page-size",      type=int, default=500, help="Trades per API page (default 500)")
    return p.parse_args()


def print_summary(enriched: list[dict], event: dict) -> None:
    total_vol  = sum(t["usdc_value"] for t in enriched)
    unique_w   = len(set(t.get("proxyWallet", "") for t in enriched))
    buys       = sum(1 for t in enriched if t.get("side") == "BUY")

    print(f"\n{'='*58}")
    print(f"  {event.get('title', 'Event')}")
    print(f"{'='*58}")
    print(f"  Trades        : {len(enriched):,}")
    print(f"  Volume (USDC) : ${total_vol:,.2f}")
    print(f"  Unique wallets: {unique_w:,}")
    print(f"  BUY / SELL    : {buys:,} / {len(enriched)-buys:,}")

    by_ticker: dict[str, float] = defaultdict(float)
    for t in enriched:
        by_ticker[t["ticker"]] += t["usdc_value"]

    print(f"\n  Volume by ticker (YES+NO combined):")
    for ticker, vol in sorted(by_ticker.items(), key=lambda x: -x[1]):
        bar = "█" * min(30, int(vol / max(by_ticker.values()) * 30))
        print(f"    {ticker:>10}  {bar}  ${vol:,.0f}")
    print()


def main() -> None:
    args = parse_args()
    slug = args.slug

    # ── 1. Event metadata ──────────────────────────────────────────────────────
    print(f"Fetching event: {slug} …", flush=True)
    event = get_event_by_slug(slug)
    event_id = event.get("id")
    if not event_id:
        print("ERROR: No event found for slug.", file=sys.stderr)
        sys.exit(1)
    print(f"  Event ID : {event_id}")
    print(f"  Title    : {event.get('title', '—')}")
    print(f"  Markets  : {len(event.get('markets', []))}")

    # ── 2. Market metadata ─────────────────────────────────────────────────────
    market_meta = build_market_meta(event)
    print(f"  Tokens   : {len(market_meta)} (YES + NO per outcome)")

    # ── 3. Day timestamps ──────────────────────────────────────────────────────
    start_ts, end_ts = day_timestamps(slug)
    from datetime import datetime, timezone, timedelta
    tz = timezone(timedelta(hours=-5))
    print(f"  Window   : {datetime.fromtimestamp(start_ts, tz)} → {datetime.fromtimestamp(end_ts, tz)} (CDT)")

    # ── 4. Fetch trades ────────────────────────────────────────────────────────
    if args.no_date_filter:
        print(f"\nFetching trades (event {event_id}, all dates) …", flush=True)
        raw = fetch_all_trades(
            event_id=int(event_id),
            start_ts=None,
            end_ts=None,
            page_size=args.page_size,
            taker_only=not args.no_taker_filter,
        )
    else:
        print(f"\nFetching trades (event {event_id}, window {start_ts}–{end_ts}) …", flush=True)
        raw = fetch_all_trades(
            event_id=int(event_id),
            start_ts=start_ts,
            end_ts=end_ts,
            page_size=args.page_size,
            taker_only=not args.no_taker_filter,
        )
        # If the event date is in the future or hasn't traded yet, fall back to all trades
        if not raw:
            print("  No trades in date window — retrying without date filter …", flush=True)
            raw = fetch_all_trades(
                event_id=int(event_id),
                start_ts=None,
                end_ts=None,
                page_size=args.page_size,
                taker_only=not args.no_taker_filter,
            )
    print(f"  Raw trades fetched: {len(raw):,}")

    if not raw:
        print("\nNo trades found for this event.")
        sys.exit(0)

    # ── 5. Enrich ──────────────────────────────────────────────────────────────
    enriched = enrich_trades(raw, market_meta)

    # ── 6. Summary ─────────────────────────────────────────────────────────────
    print_summary(enriched, event)

    # ── 7. Save raw JSON ───────────────────────────────────────────────────────
    if args.save:
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        out_path = DATA_RAW / f"{slug}.json"
        out_path.write_text(json.dumps({"event": event, "market_meta": market_meta, "trades": enriched}, indent=2, default=str))
        print(f"  Saved JSON → {out_path}")

    # ── 8. Generate artifact ───────────────────────────────────────────────────
    if args.artifact:
        DATA_ARTS.mkdir(parents=True, exist_ok=True)
        html = generate_artifact(enriched, market_meta, event)
        art_path = DATA_ARTS / f"{slug}.html"
        art_path.write_text(html)
        print(f"  Saved HTML → {art_path}")


if __name__ == "__main__":
    main()
