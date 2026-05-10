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