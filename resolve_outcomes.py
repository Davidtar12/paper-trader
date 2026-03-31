#!/usr/bin/env python3
"""
resolve_outcomes.py -- Auto-resolve WIN/LOSS for open paper trades

For each OPEN row in paper_trades.csv:
  - Fetches historical bars from Twelve Data after the entry time
  - Checks bar by bar whether TP or SL was hit first
  - Updates status=CLOSED, outcome=WIN or LOSS

Run daily (after NY close) via GitHub Actions.
"""

import csv
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

API_KEY  = os.environ["TWELVE_DATA_API_KEY"]
CSV_PATH = Path(__file__).parent / "paper_trades.csv"
CSV_HEADERS = [
    "date", "time_utc", "strategy", "instrument", "direction",
    "entry_price", "sl_price", "tp_price", "status", "outcome", "notes",
]

INSTRUMENT_INTERVAL = {
    "XAU/USD": "15min",
    "SPY":     "30min",
}


def fetch_ohlcv(symbol: str, interval: str, outputsize: int = 500) -> pd.DataFrame:
    resp = requests.get(
        "https://api.twelvedata.com/time_series",
        params=dict(symbol=symbol, interval=interval, outputsize=outputsize,
                    timezone="UTC", apikey=API_KEY),
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") == "error":
        raise ValueError(f"Twelve Data [{symbol}]: {data.get('message')}")
    values = data.get("values", [])
    if not values:
        raise ValueError(f"No bars for {symbol}")
    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    return df


def resolve_trade(row: dict, df: pd.DataFrame) -> str | None:
    """Return 'WIN', 'LOSS', or None if not yet resolved."""
    entry_dt = datetime.fromisoformat(f"{row['date']} {row['time_utc']}")
    tp = float(row["tp_price"])
    sl = float(row["sl_price"])
    direction = row["direction"]

    bars_after = df[df.index > entry_dt]
    if bars_after.empty:
        return None

    for _, bar in bars_after.iterrows():
        if direction == "BUY":
            if bar["high"] >= tp:
                return "WIN"
            if bar["low"] <= sl:
                return "LOSS"
        else:  # SELL
            if bar["low"] <= tp:
                return "WIN"
            if bar["high"] >= sl:
                return "LOSS"

    return None  # still open


def main() -> int:
    if not CSV_PATH.exists():
        log.info("No paper_trades.csv yet")
        return 0

    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    open_rows = [r for r in rows if r["status"] == "OPEN"]
    if not open_rows:
        log.info("No open trades to resolve")
        return 0

    # Fetch data once per instrument
    cache: dict[str, pd.DataFrame] = {}
    resolved = 0

    for row in open_rows:
        instrument = row["instrument"]
        if instrument not in cache:
            interval = INSTRUMENT_INTERVAL.get(instrument, "15min")
            try:
                cache[instrument] = fetch_ohlcv(instrument, interval, 5000)
            except Exception as exc:
                log.error(f"Fetch failed for {instrument}: {exc}")
                continue

        outcome = resolve_trade(row, cache[instrument])
        if outcome:
            row["status"]  = "CLOSED"
            row["outcome"] = outcome
            log.info(f"Resolved {row['date']} {row['strategy']} {row['direction']} -> {outcome}")
            resolved += 1

    if resolved:
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"{resolved} trade(s) resolved and saved.")
    else:
        log.info("No trades resolved (still open or insufficient data).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
