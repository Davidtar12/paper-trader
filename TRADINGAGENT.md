# Trading Agent

## What This Repo Is

Scheduled paper-trading bot that runs through GitHub Actions, fetches market data from Twelve Data, logs paper entries, and resolves outcomes into a CSV ledger.

## Key Files

- `.github/workflows/signal_logger.yml`: scheduler, retries, metrics, and bot commit step
- `signal_logger.py`: fetches market data and logs new paper-trade signals
- `resolve_outcomes.py`: resolves open trades against later bars and writes exits
- `metrics.py`: prints current per-strategy metrics from the CSV ledger
- `paper_trades.csv`: source-of-truth paper-trade ledger used by metrics and workflow commits

## Current Strategies

- `NYOpen_US500_Filtered`: SPY opening-range breakout with regime filter
- `NYOpen_US500_Raw`: SPY opening-range breakout without regime filter
- `XAUUSD_EmaPullback_Filtered`: XAU/USD EMA pullback with regime filter
- `XAUUSD_EmaPullback_Raw`: XAU/USD EMA pullback without regime filter
- `XAUUSD_EmaPullback_Loose`: XAU/USD shadow variant with looser pullback settings
- `OptionExpirationWeek_US500`: SPY calendar strategy for OPEX week

## Workflow Schedule

- XAU checks: every 30 minutes during 07:00-17:30 UTC, Monday-Friday
- NY open checks: 14:35 UTC and 15:30 UTC, Monday-Friday
- Outcome resolution: 21:00 UTC, Monday-Friday

## Recent Important Fixes

- `a3e61ba`: replaced `nick-fields/retry@v3` with explicit shell retry loops in the workflow
- `043cab8`: tightened retry semantics and split transient vendor failures from unexpected internal errors
- `34abbc9`: added the initial repo handoff documentation file

## Operational Rules

- Treat Twelve Data outages as vendor issues first, not as immediate strategy bugs
- Do not change strategy logic, trade windows, stop/target logic, or risk assumptions without asking
- All schedule references are UTC
- Prefer minimal fixes over strategy rewrites
- Do not commit or push without explicit user approval

## How To Read Workflow Failures

- A green run does not guarantee a new signal; many slots correctly log no setup
- A transient Twelve Data failure can now log a warning and skip that slot without turning the run red
- A red run should be treated as a real code/config/runtime problem first, not as normal vendor noise
- If investigating a red run, check whether the failure came from `signal_logger.py`, `resolve_outcomes.py`, dependency setup, or git push/rebase logic

## Quick Commands

```powershell
cd 01_Finance\paper_trader
python -m pip install -r requirements.txt
python signal_logger.py
python resolve_outcomes.py
python metrics.py
```

## Session Summary Format

End each work session with a short summary containing:

- issue investigated or goal completed
- root cause or key decision
- files changed
- whether anything was committed and pushed
- remaining risk or next verification step

Update this file only when durable repo context changes or a meaningful fix should be carried into future sessions. Do not append full session transcripts.

## Important Note

This file is documentation only. The GitHub Actions workflow and Python scripts do not read `TRADINGAGENT.md`.
The repo-level Copilot instructions file should point agents here at session start.

If you want runtime behavior to depend on this file, that must be implemented explicitly.

## Current Status As Of 2026-05-10

- Workflow retry behavior is hardened against the Node.js 20 deprecation path
- Transient market-data failures should skip a slot cleanly instead of producing false-red workflow runs
- Runtime behavior was changed only in the workflow and error-boundary handling, not in strategy logic
*** Add File: c:\dscodingpython\01_Finance\paper_trader\.github\copilot-instructions.md
# paper_trader — Coding Agent Instructions

## FIRST ACTION (every session)
Read `TRADINGAGENT.md` in the repo root before editing code:

```text
c:\dscodingpython\01_Finance\paper_trader\TRADINGAGENT.md
```

Use that file as the source of current repo context, recent important fixes, workflow failure interpretation, and the expected end-of-session summary format.

## Behavior

- Read the relevant code before editing. Never guess.
- Do not change strategy logic, trade windows, stop/target rules, or risk assumptions without asking.
- Treat Twelve Data outages as vendor problems first, not immediate strategy bugs.
- All schedule references are UTC.
- Do not commit or push without explicit user approval.

## End Of Session

Give a concise summary that includes:

- issue investigated or goal completed
- root cause or key decision
- files changed
- whether anything was committed and pushed
- remaining risk or next verification step

Update `TRADINGAGENT.md` only when durable repo context or a meaningful fix needs to be carried into future sessions. Do not append full session transcripts.