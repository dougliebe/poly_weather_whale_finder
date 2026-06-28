"""
Fetch weather trade history from Polymarket's public data API.

Strategy:
1. Pull the global recent trade feed to discover active weather market wallets.
2. For each unique wallet, fetch their full activity history via the per-wallet endpoint.
3. Filter all activity for weather market trades only.

This gives us days of history per wallet without requiring CLOB API authentication.
"""

import time
import requests
from typing import Optional

DATA_API = "https://data-api.polymarket.com"
FEED_PAGE_SIZE = 500    # global feed page size
WALLET_PAGE_SIZE = 500  # per-wallet activity page size
FEED_MAX_OFFSET = 3000  # data-api hard limit


def _is_weather(trade: dict) -> bool:
    title = (trade.get("title") or "").lower()
    return any(kw in title for kw in ["temperature", "°c", "°f", "fahrenheit", "celsius"])


def collect_weather_wallets(
    token_ids: Optional[set] = None,
    sleep_between: float = 0.1,
) -> set[str]:
    """
    Scrape the global public trade feed and return all unique wallet addresses
    that traded in weather markets.
    """
    wallets: set[str] = set()
    for offset in range(0, FEED_MAX_OFFSET + 1, FEED_PAGE_SIZE):
        resp = requests.get(
            f"{DATA_API}/trades",
            params={"limit": FEED_PAGE_SIZE, "offset": offset},
            timeout=30,
        )
        if not resp.ok:
            break
        page = resp.json()
        if not page:
            break

        for t in page:
            is_weather = (
                t.get("asset") in token_ids if token_ids
                else _is_weather(t)
            )
            if is_weather and t.get("proxyWallet"):
                wallets.add(t["proxyWallet"])

        time.sleep(sleep_between)

    return wallets


def fetch_wallet_weather_trades(
    wallet: str,
    token_ids: Optional[set] = None,
    min_timestamp: Optional[int] = None,
    max_pages: int = 20,
    sleep_between: float = 0.15,
) -> list[dict]:
    """
    Fetch all weather activity for a single wallet via the per-wallet endpoint.
    Returns filtered list of raw trade dicts.
    """
    trades = []
    for page in range(max_pages):
        offset = page * WALLET_PAGE_SIZE
        resp = requests.get(
            f"{DATA_API}/activity",
            params={"user": wallet, "limit": WALLET_PAGE_SIZE, "offset": offset},
            timeout=30,
        )
        if not resp.ok:
            break
        page_data = resp.json()
        if not isinstance(page_data, list) or not page_data:
            break

        for t in page_data:
            if t.get("type") != "TRADE":
                continue
            is_weather = (
                t.get("asset") in token_ids if token_ids
                else _is_weather(t)
            )
            if is_weather:
                t["proxyWallet"] = wallet
                trades.append(t)

        # Stop if we've gone past our time window
        if min_timestamp is not None:
            oldest = min(t["timestamp"] for t in page_data)
            if oldest < min_timestamp:
                break

        time.sleep(sleep_between)

    return trades


def get_weather_trades(
    token_ids: Optional[set] = None,
    min_timestamp: Optional[int] = None,
    max_wallet_pages: int = 10,
) -> tuple[list[dict], set[str]]:
    """
    Full pipeline: discover wallets from global feed, then fetch each wallet's
    complete weather trade history.

    Returns
    -------
    (trades, wallets)  where trades is a list of raw trade dicts and
                       wallets is the set of discovered wallet addresses.
    """
    print("  Step 1: scanning global feed for weather market wallets…")
    wallets = collect_weather_wallets(token_ids=token_ids)
    print(f"  Found {len(wallets)} unique weather market wallets")

    print("  Step 2: fetching per-wallet trade history…")
    all_trades: list[dict] = []
    for i, wallet in enumerate(wallets):
        trades = fetch_wallet_weather_trades(
            wallet,
            token_ids=token_ids,
            min_timestamp=min_timestamp,
            max_pages=max_wallet_pages,
        )
        all_trades.extend(trades)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(wallets)} wallets processed, {len(all_trades):,} trades so far")

    # Deduplicate by (wallet, tx_hash, asset)
    seen: set[tuple] = set()
    unique: list[dict] = []
    for t in all_trades:
        key = (t.get("proxyWallet"), t.get("transactionHash"), t.get("asset"))
        if key not in seen:
            seen.add(key)
            unique.append(t)

    return unique, wallets


def normalize_trade(raw: dict) -> dict:
    """
    Map raw data-api activity fields to a clean, consistent schema.

    Raw field reference (data-api.polymarket.com/activity):
        proxyWallet, timestamp (unix int), conditionId,
        type, size, usdcSize, transactionHash, price,
        asset (token_id), side, outcomeIndex,
        title, slug, outcome, name, pseudonym
    """
    price = float(raw.get("price") or 0)
    size = float(raw.get("size") or 0)
    return {
        "tx_hash": raw.get("transactionHash"),
        "timestamp": raw.get("timestamp"),           # unix int
        "asset_id": raw.get("asset"),
        "condition_id": raw.get("conditionId"),
        "market_question": raw.get("title"),
        "outcome": raw.get("outcome"),
        "outcome_index": raw.get("outcomeIndex"),
        "side": raw.get("side"),                     # "BUY" or "SELL"
        "price": price,                              # 0–1 probability
        "size": size,                                # shares
        "usd_value": float(raw.get("usdcSize") or price * size),
        "wallet": raw.get("proxyWallet"),
        "name": raw.get("name"),
        "pseudonym": raw.get("pseudonym"),
    }
