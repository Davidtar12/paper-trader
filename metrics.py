#!/usr/bin/env python3
"""
metrics.py -- Compute per-strategy stats from paper_trades.csv.

Reports, per strategy:
  trades, wins, losses, win_rate, avg_R, profit_factor, sharpe_R, max_drawdown_R,
  best_R, worst_R, exit_reason breakdown, and a regime-bucket breakdown
  (trades where ADX > 25 vs not, slope above/below threshold).

R is read from the `r_realized` column when available; otherwise it is reconstructed
from entry/sl/tp + outcome (WIN -> reward/risk, LOSS -> -1).
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

CSV_PATH = Path(__file__).parent / "paper_trades.csv"

ADX_THRESHOLD   = 25.0
SLOPE_THRESHOLD = 0.0015


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _row_R(row: dict) -> float | None:
    r = _to_float(row.get("r_realized"))
    if r is not None:
        return r
    if row.get("status") != "CLOSED":
        return None
    entry = _to_float(row.get("entry_price"))
    sl    = _to_float(row.get("sl_price"))
    tp    = _to_float(row.get("tp_price"))
    if entry is None or sl is None or tp is None:
        return None
    risk = abs(entry - sl)
    if risk <= 0:
        return None
    if row.get("outcome") == "WIN":
        return abs(tp - entry) / risk
    if row.get("outcome") == "LOSS":
        return -1.0
    return None


def _max_drawdown(returns: list[float]) -> float:
    equity = 0.0
    peak   = 0.0
    max_dd = 0.0
    for r in returns:
        equity += r
        peak    = max(peak, equity)
        max_dd  = min(max_dd, equity - peak)
    return max_dd  # negative number


def _summarize(strategy: str, rows: list[dict]) -> dict:
    closed = [r for r in rows if r.get("status") == "CLOSED"]
    Rs = [r for r in (_row_R(row) for row in closed) if r is not None]
    wins   = sum(1 for r in Rs if r > 0)
    losses = sum(1 for r in Rs if r <= 0)

    win_rate = (wins / len(Rs)) if Rs else float("nan")
    avg_R    = statistics.mean(Rs) if Rs else float("nan")
    sharpe_R = (
        avg_R / statistics.stdev(Rs)
        if len(Rs) >= 2 and statistics.stdev(Rs) > 0
        else float("nan")
    )
    gross_w  = sum(r for r in Rs if r > 0)
    gross_l  = -sum(r for r in Rs if r < 0)
    pf       = (gross_w / gross_l) if gross_l > 0 else float("inf") if gross_w > 0 else float("nan")

    exit_reasons = Counter(r.get("exit_reason", "") or "n/a" for r in closed)

    # Regime buckets
    buckets = defaultdict(list)
    for row in closed:
        adx   = _to_float(row.get("entry_adx"))
        slope = _to_float(row.get("entry_slope"))
        R     = _row_R(row)
        if R is None:
            continue
        adx_b   = "adx>25" if adx is not None and adx > ADX_THRESHOLD else "adx<=25"
        slope_b = (
            "slope_up"   if slope is not None and slope >  SLOPE_THRESHOLD else
            "slope_down" if slope is not None and slope < -SLOPE_THRESHOLD else
            "slope_flat"
        )
        buckets[(adx_b, slope_b)].append(R)

    bucket_summary = {
        f"{k[0]}|{k[1]}": {
            "n": len(v),
            "win_rate": round(sum(1 for r in v if r > 0) / len(v), 3),
            "avg_R":    round(statistics.mean(v), 3),
        }
        for k, v in sorted(buckets.items())
    }

    return {
        "strategy":        strategy,
        "trades":          len(Rs),
        "wins":            wins,
        "losses":          losses,
        "win_rate":        round(win_rate, 4) if not math.isnan(win_rate) else None,
        "avg_R":           round(avg_R, 4) if not math.isnan(avg_R) else None,
        "sharpe_R":        None if math.isnan(sharpe_R) else round(sharpe_R, 4),
        "profit_factor":   None if math.isnan(pf) else round(pf, 3),
        "max_drawdown_R":  round(_max_drawdown(Rs), 3),
        "best_R":          round(max(Rs), 3) if Rs else None,
        "worst_R":         round(min(Rs), 3) if Rs else None,
        "exit_reasons":    dict(exit_reasons),
        "regime_buckets":  bucket_summary,
        "live_deploy_ok":  bool(Rs and len(Rs) >= 50 and (sharpe_R or 0) >= 0.6),
    }


def main() -> int:
    if not CSV_PATH.exists():
        print("No paper_trades.csv yet")
        return 0
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    strategies = sorted({r["strategy"] for r in rows if r.get("strategy")})
    if not strategies:
        print("No strategies in CSV")
        return 0

    print("=" * 78)
    print(f"Paper Trader metrics  (rows={len(rows)})")
    print("=" * 78)

    for strategy in strategies:
        s_rows = [r for r in rows if r["strategy"] == strategy]
        summary = _summarize(strategy, s_rows)
        print()
        print(f"[{strategy}]")
        for key in ("trades", "wins", "losses", "win_rate", "avg_R",
                    "sharpe_R", "profit_factor", "max_drawdown_R",
                    "best_R", "worst_R", "live_deploy_ok"):
            print(f"  {key:<16} {summary[key]}")
        print(f"  exit_reasons     {summary['exit_reasons']}")
        if summary["regime_buckets"]:
            print(f"  regime_buckets:")
            for bk, bv in summary["regime_buckets"].items():
                print(f"    {bk:<22} n={bv['n']:>3}  win_rate={bv['win_rate']:.3f}  avg_R={bv['avg_R']:.3f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
