#!/usr/bin/env python3
"""
signal_logger.py -- Paper Trading Signal Logger

Fetches live OHLCV data from Twelve Data and checks two strategies for entry signals.
Logs new signals to paper_trades.csv (one entry per strategy per day max).

Strategies:
  NYOpen_US500      -- SPY M30, NY Opening Range Breakout (14:00-14:30 UTC range)
  XAUUSD_EmaPullback -- XAU/USD M15, EMA(3/14/24) crossover + pullback breakout
"""

import csv
import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

API_KEY = os.environ["TWELVE_DATA_API_KEY"]
CSV_PATH = Path(__file__).parent / "paper_trades.csv"
CSV_HEADERS = [
    "date", "time_utc", "strategy", "instrument", "direction",
    "entry_price", "sl_price", "tp_price", "status", "outcome", "notes",
]


# ---------------------------------------------------------------------------
# Twelve Data helpers
# ---------------------------------------------------------------------------

def fetch_ohlcv(symbol: str, interval: str, outputsize: int = 100) -> pd.DataFrame:
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
        raise ValueError(f"No bars returned for {symbol}")
    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    log.info(f"Fetched {len(df)} bars for {symbol} ({interval}), latest: {df.index[-1]}")
    return df


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_csv() -> list:
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def already_today(rows: list, strategy: str) -> bool:
    today = date.today().isoformat()
    return any(r["date"] == today and r["strategy"] == strategy for r in rows)


def append_and_save(rows: list, row: dict):
    rows.append(row)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"SIGNAL: {row['strategy']} {row['direction']} {row['instrument']} "
             f"@ {row['entry_price']}  SL={row['sl_price']}  TP={row['tp_price']}")


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# NYOpen_US500  (SPY M30)
# Range 14:00-14:30 UTC | Trade 14:30-16:00 UTC
# SL=3.0 SPY pts (=30 SPX pts), TP=6.0 SPY pts (=60 SPX pts)
# ---------------------------------------------------------------------------

def check_nyopen(df: pd.DataFrame) -> dict | None:
    now = datetime.now(timezone.utc)
    hm = now.hour * 60 + now.minute
    if not (14 * 60 + 30 <= hm < 16 * 60):
        log.info("NYOpen: outside trade window 14:30-16:00 UTC")
        return None

    today = now.date()
    range_start = datetime(today.year, today.month, today.day, 14, 0, tzinfo=timezone.utc)
    range_end   = datetime(today.year, today.month, today.day, 14, 30, tzinfo=timezone.utc)
    range_bars  = df[(df.index >= range_start) & (df.index < range_end)]

    if len(range_bars) < 1:
        log.info("NYOpen: no range bars for today yet")
        return None

    hi = range_bars["high"].max()
    lo = range_bars["low"].min()
    if (hi - lo) < 1.0:   # 1 SPY pt ~ 10 SPX pts
        log.info(f"NYOpen: range {hi - lo:.2f} pts too small")
        return None

    price = df["close"].iloc[-1]
    sl, tp = 3.0, 6.0

    if price > hi:
        return dict(direction="BUY",  entry=round(price, 2),
                    sl=round(price - sl, 2), tp=round(price + tp, 2))
    if price < lo:
        return dict(direction="SELL", entry=round(price, 2),
                    sl=round(price + sl, 2), tp=round(price - tp, 2))

    log.info(f"NYOpen: price {price:.2f} inside range [{lo:.2f}, {hi:.2f}]")
    return None


# ---------------------------------------------------------------------------
# XAUUSD_EmaPullback  (XAU/USD M15)
# EMA(3/14/24) crossover + 1-3 bar pullback + breakout
# SL = 2.5 x ATR(14),  TP = 12.0 x ATR(14)
# Trade window: 07:00-18:00 UTC
# ---------------------------------------------------------------------------

def check_xauusd(df: pd.DataFrame) -> dict | None:
    now = datetime.now(timezone.utc)
    hm = now.hour * 60 + now.minute
    if not (7 * 60 <= hm < 18 * 60):
        log.info("XAUUSD: outside trade window 07:00-18:00 UTC")
        return None

    if len(df) < 30:
        log.info("XAUUSD: not enough bars")
        return None

    ema_f = df["close"].ewm(span=3,  adjust=False).mean()
    ema_m = df["close"].ewm(span=14, adjust=False).mean()
    ema_s = df["close"].ewm(span=24, adjust=False).mean()
    atr   = calc_atr(df, 14)
    pb_bars = 3

    for i in range(len(df) - 1, max(len(df) - pb_bars - 5, 25), -1):
        ef, em, es = ema_f.iloc[i], ema_m.iloc[i], ema_s.iloc[i]

        # LONG: fast > mid > slow
        if ef > em > es:
            cross_idx = None
            for j in range(1, pb_bars + 3):
                if i - j >= 0 and ema_f.iloc[i - j] <= ema_m.iloc[i - j]:
                    cross_idx = i - j
                    break
            if cross_idx is None:
                continue

            pb_cnt, pb_low = 0, None
            for j in range(cross_idx + 1, i):
                if df["close"].iloc[j] < df["close"].iloc[j - 1]:
                    pb_cnt += 1
                    lo = df["low"].iloc[j]
                    pb_low = lo if pb_low is None else min(pb_low, lo)
                else:
                    break
            if not (1 <= pb_cnt <= pb_bars) or pb_low is None:
                continue

            pb_hi = df["high"].iloc[cross_idx + 1: i].max() if cross_idx + 1 < i else None
            if pb_hi is None:
                continue
            if df["close"].iloc[i] > pb_hi:
                p, a = df["close"].iloc[i], atr.iloc[i]
                return dict(direction="BUY", entry=round(p, 2),
                            sl=round(p - 2.5 * a, 2), tp=round(p + 12.0 * a, 2))

        # SHORT: fast < mid < slow
        elif ef < em < es:
            cross_idx = None
            for j in range(1, pb_bars + 3):
                if i - j >= 0 and ema_f.iloc[i - j] >= ema_m.iloc[i - j]:
                    cross_idx = i - j
                    break
            if cross_idx is None:
                continue

            pb_cnt, pb_hi = 0, None
            for j in range(cross_idx + 1, i):
                if df["close"].iloc[j] > df["close"].iloc[j - 1]:
                    pb_cnt += 1
                    hi = df["high"].iloc[j]
                    pb_hi = hi if pb_hi is None else max(pb_hi, hi)
                else:
                    break
            if not (1 <= pb_cnt <= pb_bars) or pb_hi is None:
                continue

            pb_lo = df["low"].iloc[cross_idx + 1: i].min() if cross_idx + 1 < i else None
            if pb_lo is None:
                continue
            if df["close"].iloc[i] < pb_lo:
                p, a = df["close"].iloc[i], atr.iloc[i]
                return dict(direction="SELL", entry=round(p, 2),
                            sl=round(p + 2.5 * a, 2), tp=round(p - 12.0 * a, 2))

    log.info("XAUUSD: no setup found in lookback window")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    now = datetime.now(timezone.utc)
    log.info(f"Signal check at {now.strftime('%Y-%m-%d %H:%M')} UTC")

    existing = load_csv()
    new_sigs = 0

    # -- XAUUSD --
    if not already_today(existing, "XAUUSD_EmaPullback"):
        try:
            df_xau = fetch_ohlcv("XAU/USD", "15min", 100)
            sig = check_xauusd(df_xau)
            if sig:
                append_and_save(existing, dict(
                    date=now.date().isoformat(),
                    time_utc=now.strftime("%H:%M"),
                    strategy="XAUUSD_EmaPullback",
                    instrument="XAU/USD",
                    **sig,
                    status="OPEN", outcome="", notes="",
                ))
                new_sigs += 1
        except Exception as exc:
            log.error(f"XAUUSD error: {exc}")
            raise
    else:
        log.info("XAUUSD: already signaled today")

    # -- NYOpen_US500 --
    if not already_today(existing, "NYOpen_US500"):
        try:
            df_spy = fetch_ohlcv("SPY", "30min", 50)
            sig = check_nyopen(df_spy)
            if sig:
                append_and_save(existing, dict(
                    date=now.date().isoformat(),
                    time_utc=now.strftime("%H:%M"),
                    strategy="NYOpen_US500",
                    instrument="SPY",
                    **sig,
                    status="OPEN", outcome="", notes="",
                ))
                new_sigs += 1
        except Exception as exc:
            log.error(f"NYOpen error: {exc}")
            raise
    else:
        log.info("NYOpen: already signaled today")

    log.info(f"Done -- {new_sigs} new signal(s) logged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
