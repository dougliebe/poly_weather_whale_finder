"""Fetch and filter Polymarket weather markets via the Gamma API."""

import json
import time
import requests
from typing import Optional

GAMMA_BASE = "https://gamma-api.polymarket.com"


WEATHER_TITLE_KEYWORDS = ["temperature", "°c", "°f", "fahrenheit", "celsius"]


def _is_weather_event(event: dict) -> bool:
    title = (event.get("title") or event.get("slug") or "").lower()
    return any(kw in title for kw in WEATHER_TITLE_KEYWORDS)


def get_weather_events(limit: int = 100) -> list[dict]:
    """
    Return all weather/temperature events from the Gamma API.

    Uses the /events endpoint which groups related temperature-outcome markets
    (e.g., all "28°C / 29°C / 30°C…" markets for a single city/day event).
    """
    all_events: list[dict] = []
    for kw in ["highest temperature", "lowest temperature"]:
        resp = requests.get(
            f"{GAMMA_BASE}/events",
            params={"limit": limit, "q": kw, "order": "startDate", "ascending": "false"},
            timeout=30,
        )
        resp.raise_for_status()
        all_events.extend(resp.json())
        time.sleep(0.2)

    # Deduplicate by event id and keep only genuine weather events
    seen: set = set()
    unique: list[dict] = []
    for e in all_events:
        if e["id"] not in seen and _is_weather_event(e):
            seen.add(e["id"])
            unique.append(e)
    return unique


def get_weather_markets_from_events(events: list[dict]) -> list[dict]:
    """
    Flatten events into individual market dicts, each with token_ids parsed.
    """
    markets = []
    for event in events:
        for m in event.get("markets", []):
            markets.append(parse_market(m, event))
    return markets


def parse_market(m: dict, event: Optional[dict] = None) -> dict:
    """Extract the fields we care about from a Gamma market dict."""
    raw_tokens = m.get("clobTokenIds", "[]")
    try:
        token_ids: list[str] = json.loads(raw_tokens) if isinstance(raw_tokens, str) else raw_tokens
    except (json.JSONDecodeError, TypeError):
        token_ids = []

    raw_outcomes = m.get("outcomes", '["Yes","No"]')
    try:
        outcomes: list[str] = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
    except (json.JSONDecodeError, TypeError):
        outcomes = ["Yes", "No"]

    raw_prices = m.get("outcomePrices", "[null,null]")
    try:
        prices: list = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
    except (json.JSONDecodeError, TypeError):
        prices = []

    tokens = [
        {
            "token_id": tid,
            "outcome": outcomes[i] if i < len(outcomes) else f"outcome_{i}",
            "price": float(prices[i]) if i < len(prices) and prices[i] is not None else None,
        }
        for i, tid in enumerate(token_ids)
    ]

    return {
        "id": m.get("id"),
        "condition_id": m.get("conditionId"),
        "question": m.get("question"),
        "slug": m.get("slug"),
        "active": m.get("active"),
        "closed": m.get("closed"),
        "start_date": m.get("startDate"),
        "end_date": m.get("endDate"),
        "volume": m.get("volumeNum") or m.get("volume"),
        "liquidity": m.get("liquidityNum") or m.get("liquidity"),
        "neg_risk": m.get("negRisk", False),
        "neg_risk_market_id": m.get("negRiskMarketID"),
        "resolution_source": m.get("resolutionSource"),
        "tokens": tokens,
        "event_title": event.get("title") if event else None,
        "event_slug": event.get("slug") if event else None,
    }
