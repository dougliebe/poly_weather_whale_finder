"""
Analyze trade data to surface wallets that consistently get ahead of
large price moves in weather markets — potential fast-feed traders.
"""

import pandas as pd
from typing import Optional


def trades_to_df(normalized_trades: list[dict]) -> pd.DataFrame:
    """Convert a list of normalized trade dicts to a DataFrame."""
    df = pd.DataFrame(normalized_trades)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def wallet_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-wallet aggregate stats across all markets.

    Columns:
        wallet, trade_count, total_usd, markets_traded,
        avg_price, buy_count, sell_count
    """
    if df.empty:
        return pd.DataFrame()

    rows = []
    for wallet, grp in df.groupby("wallet"):
        name = grp["name"].iloc[0] if "name" in grp.columns else ""
        rows.append({
            "wallet": wallet,
            "name": name,
            "trade_count": len(grp),
            "total_usd": grp["usd_value"].sum(),
            "markets_traded": grp["condition_id"].nunique(),
            "avg_price": grp["price"].mean(),
            "buy_count": (grp["side"] == "BUY").sum(),
            "sell_count": (grp["side"] == "SELL").sum(),
        })
    return pd.DataFrame(rows).sort_values("total_usd", ascending=False)


def price_move_events(
    df: pd.DataFrame,
    window: str = "1h",
    min_move: float = 0.05,
) -> pd.DataFrame:
    """
    Identify time windows where the market price moved sharply.

    Returns a DataFrame of events with:
        window_start, window_end, asset_id, outcome,
        price_start, price_end, price_delta, direction
    """
    if df.empty:
        return pd.DataFrame()

    events = []
    for (asset_id, outcome), grp in df.groupby(["asset_id", "outcome"]):
        grp = grp.set_index("timestamp").sort_index()
        # resample to get OHLC-style price per window
        resampled = grp["price"].resample(window).ohlc().dropna()
        for start, row in resampled.iterrows():
            delta = row["close"] - row["open"]
            if abs(delta) >= min_move:
                events.append({
                    "window_start": start,
                    "window_end": start + pd.Timedelta(window),
                    "asset_id": asset_id,
                    "outcome": outcome,
                    "price_open": row["open"],
                    "price_close": row["close"],
                    "price_delta": delta,
                    "direction": "UP" if delta > 0 else "DOWN",
                })
    result = pd.DataFrame(events)
    return result.sort_values("window_start") if not result.empty else result


def early_movers(
    df: pd.DataFrame,
    events: pd.DataFrame,
    lead_minutes: int = 15,
) -> pd.DataFrame:
    """
    For each price-move event, find wallets that traded in the correct
    direction within `lead_minutes` minutes BEFORE the event window.

    Returns a DataFrame with columns:
        wallet, event_window_start, asset_id, outcome, direction,
        trades_before_move, usd_before_move, avg_lead_minutes
    """
    if df.empty or events.empty:
        return pd.DataFrame()

    lead = pd.Timedelta(minutes=lead_minutes)
    rows = []

    for _, evt in events.iterrows():
        window_start = evt["window_start"]
        asset_id = evt["asset_id"]
        direction = evt["direction"]

        # trades in the lead window for this token
        mask = (
            (df["asset_id"] == asset_id)
            & (df["timestamp"] >= window_start - lead)
            & (df["timestamp"] < window_start)
        )
        pre_trades = df[mask].copy()
        if pre_trades.empty:
            continue

        # correct-direction trades:
        #   if price went UP → BUY trades are prescient
        #   if price went DOWN → SELL trades are prescient
        correct_side = "BUY" if direction == "UP" else "SELL"
        correct = pre_trades[pre_trades["side"] == correct_side]
        if correct.empty:
            continue

        for wallet, wgrp in correct.groupby("wallet"):
            avg_lead = (window_start - wgrp["timestamp"]).dt.total_seconds().mean() / 60
            rows.append({
                "wallet": wallet,
                "event_window_start": window_start,
                "asset_id": asset_id,
                "outcome": evt["outcome"],
                "direction": direction,
                "trades_before_move": len(wgrp),
                "usd_before_move": wgrp["usd_value"].sum(),
                "avg_lead_minutes": round(avg_lead, 2),
            })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # summarize per wallet: how often do they front-run moves?
    summary = (
        result.groupby("wallet")
        .agg(
            events_front_run=("event_window_start", "count"),
            total_usd_front_run=("usd_before_move", "sum"),
            avg_lead_minutes=("avg_lead_minutes", "mean"),
            markets=("asset_id", "nunique"),
        )
        .reset_index()
        .sort_values("events_front_run", ascending=False)
    )
    return summary


def print_report(df: pd.DataFrame, events: pd.DataFrame, movers: pd.DataFrame) -> None:
    """Print a quick console summary of findings."""
    print(f"\n{'='*60}")
    print(f"TRADES LOADED : {len(df):,}")
    print(f"UNIQUE WALLETS: {df['wallet'].nunique():,}")
    print(f"PRICE EVENTS  : {len(events):,}  (≥5¢ move per hour)")
    print(f"{'='*60}\n")

    print("TOP 10 WALLETS BY VOLUME:")
    ws = wallet_summary(df).head(10)
    print(ws[["wallet", "trade_count", "total_usd", "buy_count", "sell_count"]].to_string(index=False))

    if not movers.empty:
        print("\nTOP 10 POTENTIAL FAST-FEED WALLETS (front-ran most moves):")
        print(movers.head(10).to_string(index=False))
    else:
        print("\nNo early-mover signals found (need more trade history).")
