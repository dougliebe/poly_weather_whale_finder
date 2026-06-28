"""
Fetch weather trade history from Polymarket's authenticated CLOB API.

Requires env vars:
    POLY_API_KEY        - from ClobClient.create_or_derive_api_creds()
    POLY_API_SECRET     - same
    POLY_PASSPHRASE     - same

The CLOB /data/trades endpoint returns full trade history filterable by
asset_id (outcome token), with cursor-based pagination and no depth limit.
"""

import os
import time
import hashlib
import hmac
import base64
import requests
from typing import Optional
from datetime import datetime, timezone

CLOB_BASE = "https://clob.polymarket.com"
END_CURSOR = "LTE="
START_CURSOR = "MA=="
PAGE_SIZE = 500


def _get_creds() -> tuple[str, str, str]:
    key = os.environ.get("POLY_API_KEY", "")
    secret = os.environ.get("POLY_API_SECRET", "")
    passphrase = os.environ.get("POLY_PASSPHRASE", "")
    if not all([key, secret, passphrase]):
        raise EnvironmentError(
            "Missing Polymarket API credentials. "
            "Set POLY_API_KEY, POLY_API_SECRET, POLY_PASSPHRASE."
        )
    return key, secret, passphrase


def _auth_headers(method: str, path: str) -> dict:
    """Generate HMAC-signed headers for the CLOB API."""
    key, secret, passphrase = _get_creds()
    ts = str(int(datetime.now(timezone.utc).timestamp()))
    message = ts + method.upper() + path
    signature = hmac.new(
        base64.b64decode(secret),
        message.encode(),
        hashlib.sha256,
    ).digest()
    sig_b64 = base64.b64encode(signature).decode()
    return {
        "POLY-API-KEY": key,
        "POLY-PASSPHRASE": passphrase,
        "POLY-TIMESTAMP": ts,
        "POLY-SIGNATURE": sig_b64,
    }


def _clob_get(path: str, params: dict) -> dict:
    headers = _auth_headers("GET", path)
    resp = requests.get(
        f"{CLOB_BASE}{path}",
        params=params,
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def get_trades_for_token(
    token_id: str,
    after_ts: Optional[int] = None,
    before_ts: Optional[int] = None,
    max_pages: int = 200,
    sleep_between: float = 0.2,
) -> list[dict]:
    """
    Fetch all trades for a given outcome token (asset_id) via the CLOB API.

    Returns a list of raw trade dicts:
        id, asset_id, market (conditionId), side, price, size,
        maker_address, taker_address, timestamp, type, fee_rate_bps
    """
    trades = []
    cursor = START_CURSOR

    for _ in range(max_pages):
        params: dict = {"asset_id": token_id, "next_cursor": cursor}
        if after_ts is not None:
            params["after"] = after_ts
        if before_ts is not None:
            params["before"] = before_ts

        body = _clob_get("/data/trades", params)
        page = body.get("data", [])
        trades.extend(page)

        cursor = body.get("next_cursor", END_CURSOR)
        if cursor == END_CURSOR or not page:
            break
        time.sleep(sleep_between)

    return trades


def get_trades_for_market(
    market: dict,
    after_ts: Optional[int] = None,
    before_ts: Optional[int] = None,
    max_pages: int = 200,
) -> list[dict]:
    """
    Fetch trades for all outcome tokens in a market, tagging each with the
    outcome label (e.g. "Yes" / "No", or the specific temperature string).
    """
    all_trades = []
    for token in market.get("tokens", []):
        token_id = token.get("token_id")
        outcome = token.get("outcome", "UNKNOWN")
        if not token_id:
            continue
        trades = get_trades_for_token(
            token_id, after_ts=after_ts, before_ts=before_ts, max_pages=max_pages
        )
        for t in trades:
            t["outcome"] = outcome
            t["market_question"] = market.get("question", "")
        all_trades.extend(trades)
    return all_trades


def normalize_trade(raw: dict) -> dict:
    """
    Map raw CLOB trade fields to a clean, consistent schema.

    Raw CLOB fields:
        id, asset_id, market, side, price, size,
        maker_address, taker_address, timestamp,
        type, fee_rate_bps, outcome, market_question
    """
    price = float(raw.get("price") or 0)
    size = float(raw.get("size") or 0)
    return {
        "trade_id": raw.get("id"),
        "timestamp": raw.get("timestamp"),       # ISO-8601 string
        "asset_id": raw.get("asset_id"),
        "condition_id": raw.get("market"),
        "market_question": raw.get("market_question"),
        "outcome": raw.get("outcome"),
        "side": raw.get("side"),                 # "BUY" or "SELL"
        "price": price,                          # 0–1 probability
        "size": size,                            # shares
        "usd_value": price * size,
        "wallet": raw.get("maker_address"),
        "taker_address": raw.get("taker_address"),
        "type": raw.get("type"),
        "fee_rate_bps": raw.get("fee_rate_bps"),
    }
