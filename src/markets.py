"""Fetch and filter Polymarket weather markets via the Gamma API."""

import requests
import time
from typing import Optional

GAMMA_BASE = "https://gamma-api.polymarket.com"

WEATHER_KEYWORDS = [
    "temperature", "degrees", "fahrenheit", "celsius",
    "weather", "high temp", "low temp",
]


def get_weather_markets(
    limit: int = 100,
    offset: int = 0,
    active_only: bool = False,
) -> list[dict]:
    """
    Search for weather/temperature markets on Polymarket.

    Returns a list of market dicts. Each dict includes:
        - id, question, conditionId, slug
        - tokens: list of {token_id, outcome, price}
        - active, closed, startDate, endDate
        - volume, liquidity
    """
    params = {
        "limit": limit,
        "offset": offset,
        "tag_slug": "weather",  # Gamma API supports tag filtering
        "order": "volume",
        "ascending": "false",
    }
    if active_only:
        params["active"] = "true"
        params["closed"] = "false"

    resp = requests.get(f"{GAMMA_BASE}/markets", params=params, timeout=30)
    resp.raise_for_status()
    markets = resp.json()

    # Also try a keyword search as fallback / supplement
    if not markets:
        markets = _keyword_search_markets(limit)

    return markets


def _keyword_search_markets(limit: int = 100) -> list[dict]:
    """Fallback: search by keyword if tag query returns nothing."""
    all_markets = []
    for kw in ["temperature", "degrees fahrenheit", "degrees celsius"]:
        params = {"limit": limit, "q": kw, "order": "volume", "ascending": "false"}
        try:
            resp = requests.get(f"{GAMMA_BASE}/markets", params=params, timeout=30)
            resp.raise_for_status()
            all_markets.extend(resp.json())
        except requests.HTTPError:
            pass
        time.sleep(0.2)

    # deduplicate by conditionId
    seen = set()
    unique = []
    for m in all_markets:
        cid = m.get("conditionId") or m.get("id")
        if cid not in seen:
            seen.add(cid)
            unique.append(m)
    return unique


def extract_token_ids(market: dict) -> list[dict]:
    """
    Return [{token_id, outcome, price}, ...] for a market's YES/NO tokens.

    Polymarket binary markets have two tokens (outcome shares).
    """
    tokens = market.get("tokens", [])
    return [
        {
            "token_id": t.get("token_id"),
            "outcome": t.get("outcome"),
            "price": t.get("price"),
        }
        for t in tokens
        if t.get("token_id")
    ]


def summarize_market(market: dict) -> dict:
    """Flatten a market dict to the fields we care about."""
    return {
        "id": market.get("id"),
        "condition_id": market.get("conditionId"),
        "question": market.get("question"),
        "slug": market.get("slug"),
        "active": market.get("active"),
        "closed": market.get("closed"),
        "start_date": market.get("startDate"),
        "end_date": market.get("endDate"),
        "volume": market.get("volume"),
        "liquidity": market.get("liquidity"),
        "tokens": extract_token_ids(market),
    }
