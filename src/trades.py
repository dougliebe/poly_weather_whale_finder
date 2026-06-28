"""Fetch trade history for a Polymarket market via the CLOB API."""

import time
import requests
from typing import Optional

CLOB_BASE = "https://clob.polymarket.com"
END_CURSOR = "LTE="   # signals last page in Polymarket pagination
DEFAULT_CURSOR = "MA=="


def get_trades_for_token(
    token_id: str,
    maker_address: Optional[str] = None,
    after_ts: Optional[int] = None,
    before_ts: Optional[int] = None,
    max_pages: int = 50,
    sleep_between: float = 0.25,
) -> list[dict]:
    """
    Fetch all trades for a given outcome token (asset_id).

    Parameters
    ----------
    token_id      : The outcome token ID (from market["tokens"][n]["token_id"])
    maker_address : Optional wallet filter — only return trades where this
                    address was the maker.
    after_ts      : Unix timestamp lower bound (inclusive).
    before_ts     : Unix timestamp upper bound (inclusive).
    max_pages     : Safety cap on pagination loops.

    Returns
    -------
    List of trade dicts. Each dict contains:
        trade_id, asset_id, market (condition_id),
        side ("BUY"/"SELL"), price, size,
        maker_address, taker_address,
        timestamp, type, fee_rate_bps
    """
    trades = []
    cursor = DEFAULT_CURSOR

    for _ in range(max_pages):
        params = {
            "asset_id": token_id,
            "next_cursor": cursor,
        }
        if maker_address:
            params["maker_address"] = maker_address
        if after_ts is not None:
            params["after"] = after_ts
        if before_ts is not None:
            params["before"] = before_ts

        resp = requests.get(
            f"{CLOB_BASE}/data/trades",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()

        page_trades = body.get("data", [])
        trades.extend(page_trades)

        cursor = body.get("next_cursor", END_CURSOR)
        if cursor == END_CURSOR or not page_trades:
            break

        time.sleep(sleep_between)

    return trades


def get_trades_for_market(
    market: dict,
    after_ts: Optional[int] = None,
    before_ts: Optional[int] = None,
    max_pages: int = 50,
) -> list[dict]:
    """
    Fetch trades for ALL outcome tokens in a market and tag each trade
    with the outcome label (e.g. "YES >= 75°F", "NO").
    """
    all_trades = []
    for token in market.get("tokens", []):
        token_id = token.get("token_id")
        outcome = token.get("outcome", "UNKNOWN")
        if not token_id:
            continue

        trades = get_trades_for_token(
            token_id,
            after_ts=after_ts,
            before_ts=before_ts,
            max_pages=max_pages,
        )
        for t in trades:
            t["outcome"] = outcome
            t["market_question"] = market.get("question", "")
        all_trades.extend(trades)

    return all_trades


def normalize_trade(raw: dict) -> dict:
    """
    Map raw CLOB trade fields to a clean, consistent schema.

    Raw field reference (from Polymarket CLOB API):
        id, asset_id, market, side, price, size,
        maker_address, taker_address, timestamp,
        type, fee_rate_bps, outcome, market_question
    """
    return {
        "trade_id": raw.get("id"),
        "timestamp": raw.get("timestamp"),       # ISO-8601 string
        "asset_id": raw.get("asset_id"),
        "condition_id": raw.get("market"),
        "outcome": raw.get("outcome"),
        "market_question": raw.get("market_question"),
        "side": raw.get("side"),                 # "BUY" or "SELL"
        "price": float(raw.get("price") or 0),   # 0–1 probability
        "size": float(raw.get("size") or 0),     # shares traded
        "usd_value": float(raw.get("price") or 0) * float(raw.get("size") or 0),
        "maker_address": raw.get("maker_address"),
        "taker_address": raw.get("taker_address"),
        "type": raw.get("type"),                 # "MATCH" etc.
        "fee_rate_bps": raw.get("fee_rate_bps"),
    }
