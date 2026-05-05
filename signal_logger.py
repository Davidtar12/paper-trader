#!/usr/bin/env python3
"""
signal_logger.py -- Paper Trading Signal Logger

Fetches live OHLCV data from Twelve Data and checks two strategies for entry signals.
Logs new signals to paper_trades.csv (one entry per strategy per day max).

Strategy variants:
    NYOpen_US500_Filtered       -- SPY M30 opening range breakout with regime gate
    NYOpen_US500_Raw            -- same NYOpen logic without the regime gate
    XAUUSD_EmaPullback_Filtered -- XAU/USD M15 EMA pullback with regime gate
    XAUUSD_EmaPullback_Raw      -- same XAUUSD logic without the regime gate

The filtered variants require:
    - ADX(14) > ADX_THRESHOLD AND ADX rising over the last 3 bars
    - SMA50 slope over 10 bars > +SLOPE_THRESHOLD (BUY) or < -SLOPE_THRESHOLD (SELL)
"""

import csv
import logging
import os
import sys
from datetime import datetime, timezone
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
    "entry_price", "sl_price", "tp_price",
    "entry_atr", "entry_adx", "entry_slope",
    "status", "outcome",
    "exit_price", "exit_time_utc", "exit_reason", "r_realized",
    "notes",
]

XAUUSD_VARIANTS = (
    ("XAUUSD_EmaPullback_Filtered", True),
    ("XAUUSD_EmaPullback_Raw", False),
)
NYOPEN_VARIANTS = (
    ("NYOpen_US500_Filtered", True),
    ("NYOpen_US500_Raw", False),
)

# ---- Regime filter parameters --------------------------------------------
ADX_THRESHOLD    = 25.0     # min ADX(14) to call it a trend
ADX_LOOKBACK     = 3        # bars used to confirm "rising"
SLOPE_THRESHOLD  = 0.0015   # min |SMA50 slope over 10 bars| (0.15%)
SLOPE_LOOKBACK   = 10
SMA_PERIOD       = 50

# ---- NYOpen parameters ---------------------------------------------------
NYOPEN_MIN_RANGE_ATR = 0.5  # opening range must be at least 0.5 * ATR(14)


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
    today = datetime.now(timezone.utc).date().isoformat()
    return any(r["date"] == today and r["strategy"] == strategy for r in rows)


def _normalize_row(row: dict) -> dict:
    """Ensure every header key exists; missing -> empty string."""
    return {k: row.get(k, "") for k in CSV_HEADERS}


def append_and_save(rows: list, row: dict):
    rows.append(row)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_normalize_row(r) for r in rows)
    log.info(f"SIGNAL: {row['strategy']} {row['direction']} {row['instrument']} "
             f"@ {row['entry_price']}  SL={row['sl_price']}  TP={row['tp_price']}")


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    up = h - h.shift(1)
    dn = l.shift(1) - l
    plus_dm  = ((up > dn) & (up > 0)).astype(float) * up
    minus_dm = ((dn > up) & (dn > 0)).astype(float) * dn
    atr_w = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w
    denom = (plus_di + minus_di).replace(0, pd.NA)
    dx = 100 * (plus_di - minus_di).abs() / denom
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def regime_context(df: pd.DataFrame) -> dict:
    """Return ADX, ADX a few bars ago, and SMA50 slope at the latest bar."""
    if len(df) < SMA_PERIOD + SLOPE_LOOKBACK + 5:
        return {"adx": None, "adx_prev": None, "slope": None}
    adx = calc_adx(df, 14)
    sma = df["close"].rolling(SMA_PERIOD).mean()
    slope = (sma - sma.shift(SLOPE_LOOKBACK)) / sma.shift(SLOPE_LOOKBACK)
    a      = adx.iloc[-1]
    a_prev = adx.iloc[-1 - ADX_LOOKBACK]
    s      = slope.iloc[-1]
    return {
        "adx":      None if pd.isna(a)      else float(a),
        "adx_prev": None if pd.isna(a_prev) else float(a_prev),
        "slope":    None if pd.isna(s)      else float(s),
    }


def regime_ok(ctx: dict, direction: str) -> bool:
    a, a_prev, s = ctx["adx"], ctx["adx_prev"], ctx["slope"]
    if a is None or a_prev is None or s is None:
        return False
    if not (a > ADX_THRESHOLD and a > a_prev):
        return False
    if direction == "BUY":
        return s > SLOPE_THRESHOLD
    return s < -SLOPE_THRESHOLD


def _ctx_to_row_fields(ctx: dict) -> dict:
    return {
        "entry_adx":   "" if ctx.get("adx")   is None else round(ctx["adx"], 2),
        "entry_slope": "" if ctx.get("slope") is None else round(ctx["slope"], 5),
    }


def _build_row(
    now: datetime,
    strategy: str,
    instrument: str,
    sig: dict,
    ctx: dict,
    require_regime_filter: bool,
) -> dict:
    regime_passed = regime_ok(ctx, sig["direction"])
    row = dict(
        date=now.date().isoformat(),
        time_utc=now.strftime("%H:%M"),
        strategy=strategy,
        instrument=instrument,
        direction=sig["direction"],
        entry_price=sig["entry_price"],
        sl_price=sig["sl_price"],
        tp_price=sig["tp_price"],
        entry_atr=sig["entry_atr"],
        status="OPEN",
        outcome="",
        notes=(
            f"regime_filter={'on' if require_regime_filter else 'off'};"
            f"regime_passed={str(regime_passed).lower()}"
        ),
    )
    row.update(_ctx_to_row_fields(ctx))
    return row


# ---------------------------------------------------------------------------
# NYOpen_US500  (SPY M30)
# Range 14:00-14:30 UTC | Trade 14:30-16:00 UTC
# Stop = 1.5 * ATR(14)   |   Target = 3.0 * ATR(14)   (~1:2 R)
# ---------------------------------------------------------------------------

def check_nyopen(df: pd.DataFrame, require_regime_filter: bool) -> tuple[dict | None, dict]:
    ctx = regime_context(df)
    now = datetime.now(timezone.utc)
    hm = now.hour * 60 + now.minute
    if not (14 * 60 + 30 <= hm < 16 * 60):
        log.info("NYOpen: outside trade window 14:30-16:00 UTC")
        return None, ctx

    today = now.date()
    range_start = datetime(today.year, today.month, today.day, 14, 0)
    range_end   = datetime(today.year, today.month, today.day, 14, 30)
    range_bars  = df[(df.index >= range_start) & (df.index < range_end)]
    if len(range_bars) < 1:
        log.info("NYOpen: no range bars for today yet")
        return None, ctx

    atr = calc_atr(df, 14).iloc[-1]
    if pd.isna(atr) or atr <= 0:
        log.info("NYOpen: ATR not ready")
        return None, ctx

    hi = range_bars["high"].max()
    lo = range_bars["low"].min()
    rng = hi - lo
    if rng < NYOPEN_MIN_RANGE_ATR * atr:
        log.info(f"NYOpen: range {rng:.2f} < {NYOPEN_MIN_RANGE_ATR}*ATR ({atr:.2f})")
        return None, ctx

    price = df["close"].iloc[-1]
    sl_dist = 1.5 * atr
    tp_dist = 3.0 * atr

    if price > hi:
        if require_regime_filter and not regime_ok(ctx, "BUY"):
            log.info(
                f"NYOpen filtered: BUY breakout blocked by regime. "
                f"adx={ctx['adx']} slope={ctx['slope']}"
            )
            return None, ctx
        return dict(direction="BUY",  entry_price=round(price, 2),
                    sl_price=round(price - sl_dist, 2),
                    tp_price=round(price + tp_dist, 2),
                    entry_atr=round(float(atr), 4)), ctx
    if price < lo:
        if require_regime_filter and not regime_ok(ctx, "SELL"):
            log.info(
                f"NYOpen filtered: SELL breakout blocked by regime. "
                f"adx={ctx['adx']} slope={ctx['slope']}"
            )
            return None, ctx
        return dict(direction="SELL", entry_price=round(price, 2),
                    sl_price=round(price + sl_dist, 2),
                    tp_price=round(price - tp_dist, 2),
                    entry_atr=round(float(atr), 4)), ctx

    log.info(
        f"NYOpen: no setup. regime_filter={require_regime_filter} "
        f"price={price:.2f} range=[{lo:.2f},{hi:.2f}] "
        f"adx={ctx['adx']} slope={ctx['slope']}"
    )
    return None, ctx


# ---------------------------------------------------------------------------
# XAUUSD_EmaPullback  (XAU/USD M15)
# EMA(3/14/24) crossover + 1-3 bar pullback + breakout
# SL = 2.5 x ATR(14),  TP = 6.0 x ATR(14)   (~1:2.4 R)
# Trade window: 07:00-18:00 UTC
# ---------------------------------------------------------------------------

def check_xauusd(df: pd.DataFrame, require_regime_filter: bool) -> tuple[dict | None, dict]:
    ctx = regime_context(df)
    now = datetime.now(timezone.utc)
    hm = now.hour * 60 + now.minute
    if not (7 * 60 <= hm < 18 * 60):
        log.info("XAUUSD: outside trade window 07:00-18:00 UTC")
        return None, ctx
    if len(df) < 60:
        log.info("XAUUSD: not enough bars")
        return None, ctx

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
                if require_regime_filter and not regime_ok(ctx, "BUY"):
                    continue
                p, a = df["close"].iloc[i], atr.iloc[i]
                return dict(direction="BUY",
                            entry_price=round(p, 2),
                            sl_price=round(p - 2.5 * a, 2),
                            tp_price=round(p + 6.0 * a, 2),
                            entry_atr=round(float(a), 4)), ctx

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
                if require_regime_filter and not regime_ok(ctx, "SELL"):
                    continue
                p, a = df["close"].iloc[i], atr.iloc[i]
                return dict(direction="SELL",
                            entry_price=round(p, 2),
                            sl_price=round(p + 2.5 * a, 2),
                            tp_price=round(p - 6.0 * a, 2),
                            entry_atr=round(float(a), 4)), ctx

    log.info(
        f"XAUUSD: no setup. regime_filter={require_regime_filter} "
        f"adx={ctx['adx']} slope={ctx['slope']}"
    )
    return None, ctx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    now = datetime.now(timezone.utc)
    log.info(f"Signal check at {now.strftime('%Y-%m-%d %H:%M')} UTC")

    existing = load_csv()
    new_sigs = 0

    # -- XAUUSD variants --
    if any(not already_today(existing, strategy) for strategy, _ in XAUUSD_VARIANTS):
        try:
            df_xau = fetch_ohlcv("XAU/USD", "15min", 200)
            for strategy, require_filter in XAUUSD_VARIANTS:
                if already_today(existing, strategy):
                    log.info(f"{strategy}: already signaled today")
                    continue
                sig, ctx = check_xauusd(df_xau, require_filter)
                if sig:
                    row = _build_row(now, strategy, "XAU/USD", sig, ctx, require_filter)
                    append_and_save(existing, row)
                    new_sigs += 1
        except Exception as exc:
            log.error(f"XAUUSD error: {exc}")
            raise
    else:
        log.info("XAUUSD variants: already signaled today")

    # -- NYOpen_US500 variants --
    if any(not already_today(existing, strategy) for strategy, _ in NYOPEN_VARIANTS):
        try:
            df_spy = fetch_ohlcv("SPY", "30min", 200)
            for strategy, require_filter in NYOPEN_VARIANTS:
                if already_today(existing, strategy):
                    log.info(f"{strategy}: already signaled today")
                    continue
                sig, ctx = check_nyopen(df_spy, require_filter)
                if sig:
                    row = _build_row(now, strategy, "SPY", sig, ctx, require_filter)
                    append_and_save(existing, row)
                    new_sigs += 1
        except Exception as exc:
            log.error(f"NYOpen error: {exc}")
            raise
    else:
        log.info("NYOpen variants: already signaled today")

    log.info(f"Done -- {new_sigs} new signal(s) logged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
