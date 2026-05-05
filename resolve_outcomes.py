#!/usr/bin/env python3
"""
resolve_outcomes.py -- Auto-resolve outcomes for OPEN paper trades.

For each OPEN row in paper_trades.csv:
  - Fetches historical bars from Twelve Data after the entry time.
  - Walks bar by bar to determine which exit fires first:
      * SL hit       -> outcome=LOSS, exit_reason=SL
      * TP hit       -> outcome=WIN,  exit_reason=TP
      * Same bar     -> conservative: assume SL fills first
      * Time stop    -> outcome=WIN/LOSS by sign of PnL, exit_reason=TIME
  - Writes exit_price, exit_time_utc, exit_reason, r_realized.

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
    "entry_price", "sl_price", "tp_price",
    "entry_atr", "entry_adx", "entry_slope",
    "status", "outcome",
    "exit_price", "exit_time_utc", "exit_reason", "r_realized",
    "notes",
]

INSTRUMENT_INTERVAL = {
    "XAU/USD": "15min",
    "SPY":     "30min",
}

# Maximum holding period (in bars) before time stop fires.
TIME_STOP_BARS = {
    "NYOpen_US500":       12,   # 12 * 30min = 6h
    "XAUUSD_EmaPullback": 32,   # 32 * 15min = 8h
}


def _base_strategy(strategy: str) -> str:
    for suffix in ("_Filtered", "_Raw"):
        if strategy.endswith(suffix):
            return strategy[: -len(suffix)]
    return strategy


def fetch_ohlcv(symbol: str, interval: str, outputsize: int = 5000) -> pd.DataFrame:
    resp = requests.get(
        "https://api.twelvedata.com/time_series",
        params=dict(symbol=symbol, interval=interval, outputsize=outputsize,
                    timezone="UTC", apikey=API_KEY),
        timeout=30,
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


def _normalize_row(row: dict) -> dict:
    return {k: row.get(k, "") for k in CSV_HEADERS}


def _r_realized(direction: str, entry: float, exit_px: float, sl: float) -> float:
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0
    if direction == "BUY":
        return round((exit_px - entry) / risk, 4)
    return round((entry - exit_px) / risk, 4)


def resolve_trade(row: dict, df: pd.DataFrame) -> dict | None:
    """Return an exit dict {outcome, exit_price, exit_time_utc, exit_reason, r_realized}
    or None if the trade is not yet resolvable (no data after entry / still open and within time stop)."""
    entry_dt = datetime.fromisoformat(f"{row['date']} {row['time_utc']}")
    entry  = float(row["entry_price"])
    sl     = float(row["sl_price"])
    tp     = float(row["tp_price"])
    direction = row["direction"]
    strategy  = row["strategy"]
    max_bars  = TIME_STOP_BARS.get(_base_strategy(strategy), 32)

    bars_after = df[df.index > entry_dt]
    if bars_after.empty:
        return None

    bars_to_walk = bars_after.iloc[:max_bars]

    for ts, bar in bars_to_walk.iterrows():
        sl_hit = bar["low"] <= sl if direction == "BUY" else bar["high"] >= sl
        tp_hit = bar["high"] >= tp if direction == "BUY" else bar["low"] <= tp

        if sl_hit and tp_hit:
            # Conservative: assume SL fills first within the same bar.
            return dict(
                outcome="LOSS",
                exit_price=round(sl, 4),
                exit_time_utc=ts.strftime("%Y-%m-%d %H:%M"),
                exit_reason="SL",
                r_realized=_r_realized(direction, entry, sl, sl),
            )
        if sl_hit:
            return dict(
                outcome="LOSS",
                exit_price=round(sl, 4),
                exit_time_utc=ts.strftime("%Y-%m-%d %H:%M"),
                exit_reason="SL",
                r_realized=_r_realized(direction, entry, sl, sl),
            )
        if tp_hit:
            return dict(
                outcome="WIN",
                exit_price=round(tp, 4),
                exit_time_utc=ts.strftime("%Y-%m-%d %H:%M"),
                exit_reason="TP",
                r_realized=_r_realized(direction, entry, tp, sl),
            )

    # Time stop: only if we actually walked the full max_bars window.
    if len(bars_after) >= max_bars:
        last = bars_to_walk.iloc[-1]
        ts = bars_to_walk.index[-1]
        exit_px = float(last["close"])
        r = _r_realized(direction, entry, exit_px, sl)
        return dict(
            outcome="WIN" if r > 0 else "LOSS",
            exit_price=round(exit_px, 4),
            exit_time_utc=ts.strftime("%Y-%m-%d %H:%M"),
            exit_reason="TIME",
            r_realized=r,
        )

    return None  # not enough bars yet to time-stop


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

        result = resolve_trade(row, cache[instrument])
        if result:
            row["status"]        = "CLOSED"
            row["outcome"]       = result["outcome"]
            row["exit_price"]    = result["exit_price"]
            row["exit_time_utc"] = result["exit_time_utc"]
            row["exit_reason"]   = result["exit_reason"]
            row["r_realized"]    = result["r_realized"]
            log.info(
                f"Resolved {row['date']} {row['strategy']} {row['direction']} -> "
                f"{result['outcome']} via {result['exit_reason']} (R={result['r_realized']})"
            )
            resolved += 1

    # Always rewrite the CSV so the header is migrated to the latest schema.
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_normalize_row(r) for r in rows)

    if resolved:
        log.info(f"{resolved} trade(s) resolved and saved.")
    else:
        log.info("No trades resolved (still open or insufficient data).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
