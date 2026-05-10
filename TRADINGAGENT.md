# Trading Agent

## Purpose

This repository runs a scheduled paper-trading bot for strategy signal logging and outcome resolution.

## Runtime Components

- `.github/workflows/signal_logger.yml`: scheduled GitHub Actions workflow
- `signal_logger.py`: fetches market data from Twelve Data, checks strategy entries, appends signals to `paper_trades.csv`
- `resolve_outcomes.py`: resolves open trades against later bars and writes exits
- `metrics.py`: prints current performance summary from `paper_trades.csv`
- `paper_trades.csv`: paper-trade ledger

## Active Strategy Set

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

## Current Operational Status

- Branch: `master`
- Latest verified fix commit: `043cab8`
- `nick-fields/retry@v3` removed from workflow to avoid the Node.js 20 deprecation path
- Workflow now opts remaining JavaScript actions into Node.js 24 early with `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true`
- Transient Twelve Data failures are handled as warnings for the missed slot instead of being treated as code defects
- Unexpected exceptions still fail the workflow after retries

## Error Handling Policy

- Transient market-data failures:
  - request/network failures from Twelve Data
  - invalid or empty vendor responses
  - behavior: log warning, skip that slot, continue workflow
- Unexpected internal failures:
  - parsing bugs
  - logic errors
  - file/schema issues outside the transient market-data path
  - behavior: raise error, retry, fail workflow red if persistent

## Important Note

This file is documentation only. The GitHub Actions workflow and Python scripts do not read `TRADINGAGENT.md`.
If you want runtime behavior to depend on this file, that must be implemented explicitly.