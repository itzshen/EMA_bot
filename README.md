# XM US100 — M5 EMA210 cross bot (H4 trend + Discord)

Python bot that connects to **MetaTrader 5** (XM account, symbol such as `US100Cash`), watches **M5** closes vs **EMA210**, and filters alerts by **H4 EMA210** direction over the last **N** closed H4 bars (default **6** ≈ 24h). Qualified signals are sent to a **Discord** incoming webhook.

## Requirements

- Windows (or the same OS where your XM MT5 terminal runs).
- **MetaTrader 5** installed, logged into XM, with the symbol visible in Market Watch (name must match `SYMBOL` exactly).
- Python 3.10+ recommended.

## Setup

1. Copy `.env.example` to `.env` and set `DISCORD_WEBHOOK_URL` (omit if you use `DRY_RUN=1` only).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python bot.py
```

On first start the bot records the current last closed M5 bar and **does not** send alerts until the **next** M5 close (avoids spamming history). Progress is stored in `state.json` so restarts do not re-alert the same bar.

Optional environment variables: `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER` if you are not already logged in via the terminal; `MT5_PATH` if `MetaTrader5` cannot find `terminal64.exe`; `H4_TREND_BARS` (default `6`); `DRY_RUN=1` to log only.

## Notes

- If EMA values differ slightly from XM’s chart, compare EMA definition (pandas `ewm(span=210, adjust=False)`) with the platform.
- Keep MT5 connected while the bot runs.
