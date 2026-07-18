"""
Public Polymarket Data API — no authentication required.

Gamma API  : gamma-api.polymarket.com  (market metadata, events by slug)
Data API   : data-api.polymarket.com   (trade history, public)
"""

import json
import time
import requests
from datetime import datetime, timezone, timedelta
from typing import Optional

GAMMA_BASE = "https://gamma-api.polymarket.com"
DATA_BASE  = "https://data-api.polymarket.com"

# Dallas / most US city weather markets use CDT (UTC-5) in July
CITY_TZ_OFFSET = timedelta(hours=-5)


# ── Event metadata ─────────────────────────────────────────────────────────────

def get_event_by_slug(slug: str) -> dict:
    """Fetch full event dict from Gamma API by slug."""
    resp = requests.get(f"{GAMMA_BASE}/events/slug/{slug}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def build_market_meta(event: dict) -> dict:
    """
    Build a token_id → metadata mapping from a Gamma event dict.

    Returns:
        {
            "<token_id>": {
                "ticker":       "90-91°F",
                "outcome":      "Yes" | "No",
                "question":     "Will the highest temperature…",
                "condition_id": "0x…",
            },
            …
        }
    """
    meta = {}
    for m in event.get("markets", []):
        ticker = m.get("groupItemTitle") or m.get("slug", "")
        question = m.get("question", "")
        condition_id = m.get("conditionId", "")

        raw_token_ids = m.get("clobTokenIds", "[]")
        try:
            token_ids = json.loads(raw_token_ids) if isinstance(raw_token_ids, str) else raw_token_ids
        except (json.JSONDecodeError, TypeError):
            token_ids = []

        raw_outcomes = m.get("outcomes", '["Yes","No"]')
        try:
            outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
        except (json.JSONDecodeError, TypeError):
            outcomes = ["Yes", "No"]

        for i, token_id in enumerate(token_ids):
            outcome = outcomes[i] if i < len(outcomes) else f"outcome_{i}"
            meta[str(token_id)] = {
                "ticker":       ticker,
                "outcome":      outcome,
                "question":     question,
                "condition_id": condition_id,
            }
    return meta


def day_timestamps(slug: str, tz_offset: timedelta = CITY_TZ_OFFSET) -> tuple[int, int]:
    """
    Parse the date from a slug like "highest-temperature-in-dallas-on-july-19-2026"
    and return Unix epoch (start, end) for the full local day.

    The slug tail is expected to end with "-<month>-<day>-<year>".
    """
    month_names = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    parts = slug.rstrip("/").split("-")
    # Find year (4-digit number)
    year_idx = next((i for i, p in enumerate(parts) if p.isdigit() and len(p) == 4), None)
    if year_idx is None or year_idx < 2:
        raise ValueError(f"Cannot parse date from slug: {slug!r}")

    year  = int(parts[year_idx])
    day   = int(parts[year_idx - 1])
    month_str = parts[year_idx - 2].lower()
    month = month_names.get(month_str)
    if month is None:
        raise ValueError(f"Unknown month {month_str!r} in slug: {slug!r}")

    local_tz = timezone(tz_offset)
    start = datetime(year, month, day, 0,  0,  0,  tzinfo=local_tz)
    end   = datetime(year, month, day, 23, 59, 59, tzinfo=local_tz)
    return int(start.timestamp()), int(end.timestamp())


# ── Trade data ─────────────────────────────────────────────────────────────────

def _paginate(params_base: dict, page_size: int, sleep_between: float) -> list[dict]:
    trades = []
    offset = 0
    while True:
        params = {**params_base, "offset": offset}
        resp = requests.get(f"{DATA_BASE}/trades", params=params, timeout=30)
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break
        trades.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
        time.sleep(sleep_between)
    return trades


def fetch_all_trades(
    event_id: int,
    start_ts: Optional[int],
    end_ts: Optional[int],
    page_size: int = 500,
    sleep_between: float = 0.15,
    taker_only: bool = True,
) -> list[dict]:
    """
    Fetch all trades for an event within a time window via the public Data API.

    Uses offset-based pagination (limit + offset).

    Returns a list of raw trade dicts with fields:
        proxyWallet, side, asset, conditionId, size, price,
        timestamp, title, slug, outcome, outcomeIndex,
        name, pseudonym, transactionHash
    """
    params_base: dict = {"eventId": event_id, "limit": page_size}
    if start_ts is not None:
        params_base["start"] = start_ts
    if end_ts is not None:
        params_base["end"] = end_ts
    if taker_only:
        params_base["takerOnly"] = "true"
    return _paginate(params_base, page_size, sleep_between)


def fetch_all_trades_with_roles(
    event_id: int,
    start_ts: Optional[int],
    end_ts: Optional[int],
    page_size: int = 500,
    sleep_between: float = 0.15,
) -> list[dict]:
    """
    Fetch all trades with maker/taker role labeling.

    Makes two paginated API calls:
      1. takerOnly=false  → both sides of every trade (2 records per tx)
      2. takerOnly=true   → taker-only records (1 per tx) to identify which side is taker

    Returns all records (both maker and taker) with a "role" field added:
        "TAKER" for the aggressor, "MAKER" for the resting limit order.
    """
    params_base: dict = {"eventId": event_id, "limit": page_size}
    if start_ts is not None:
        params_base["start"] = start_ts
    if end_ts is not None:
        params_base["end"] = end_ts

    all_trades   = _paginate({**params_base, "takerOnly": "false"}, page_size, sleep_between)
    taker_trades = _paginate({**params_base, "takerOnly": "true"}, page_size, sleep_between)

    # Build set of (transactionHash, proxyWallet) for takers
    taker_keys = {(t["transactionHash"], t["proxyWallet"]) for t in taker_trades}

    for t in all_trades:
        key = (t.get("transactionHash", ""), t.get("proxyWallet", ""))
        t["role"] = "TAKER" if key in taker_keys else "MAKER"

    return all_trades


def enrich_trades(raw_trades: list[dict], market_meta: dict) -> list[dict]:
    """
    Add "ticker", "is_yes", and "usdc_value" fields to each trade.

    Trades whose asset is not in market_meta get ticker="unknown", is_yes=None.
    Preserves any "role" field already set (e.g. by fetch_all_trades_with_roles).
    """
    enriched = []
    for t in raw_trades:
        asset = str(t.get("asset", ""))
        m = market_meta.get(asset, {})
        enriched.append({
            **t,
            "ticker": m.get("ticker", "unknown"),
            "is_yes": m.get("outcome") == "Yes" if m else None,
            "usdc_value": round(float(t.get("price", 0)) * float(t.get("size", 0)), 4),
        })
    return enriched
