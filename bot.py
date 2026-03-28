"""
XM US100 — M5 EMA210 cross bot with H4 trend filter and Discord alerts.
Requires MetaTrader 5 (XM) running locally with the symbol available.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import MetaTrader5 as mt5
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

EMA_PERIOD = 210
M5_COUNT = 2000
H4_COUNT = 500
STATE_FILE = Path(__file__).resolve().parent / "state.json"
POLL_SECONDS = 2.0

Trend = Literal["up", "down", "flat"]


@dataclass
class Config:
    discord_webhook_url: str
    symbol: str
    h4_trend_bars: int
    dry_run: bool
    mt5_login: int | None
    mt5_password: str | None
    mt5_server: str | None
    mt5_path: str | None


def load_config() -> Config:
    dry = os.environ.get("DRY_RUN", "0").strip().lower() in ("1", "true", "yes")
    webhook = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook and not dry:
        logger.error("DISCORD_WEBHOOK_URL is required unless DRY_RUN=1")
        sys.exit(1)

    h4_bars = int(os.environ.get("H4_TREND_BARS", "6"))
    if h4_bars < 1:
        logger.error("H4_TREND_BARS must be >= 1")
        sys.exit(1)

    login_raw = os.environ.get("MT5_LOGIN", "").strip()
    password = os.environ.get("MT5_PASSWORD", "").strip() or None
    server = os.environ.get("MT5_SERVER", "").strip() or None
    mt5_path = os.environ.get("MT5_PATH", "").strip() or None

    login: int | None = None
    if login_raw:
        try:
            login = int(login_raw)
        except ValueError:
            logger.error("MT5_LOGIN must be an integer")
            sys.exit(1)

    return Config(
        discord_webhook_url=webhook,
        symbol=os.environ.get("SYMBOL", "US100Cash").strip(),
        h4_trend_bars=h4_bars,
        dry_run=dry,
        mt5_login=login,
        mt5_password=password,
        mt5_server=server,
        mt5_path=mt5_path,
    )


def init_mt5(cfg: Config) -> None:
    kwargs: dict[str, Any] = {}
    if cfg.mt5_path:
        kwargs["path"] = cfg.mt5_path

    if not mt5.initialize(**kwargs):
        logger.error("mt5.initialize() failed: %s", mt5.last_error())
        sys.exit(1)

    if cfg.mt5_login is not None and cfg.mt5_password and cfg.mt5_server:
        authorized = mt5.login(
            cfg.mt5_login,
            password=cfg.mt5_password,
            server=cfg.mt5_server,
        )
        if not authorized:
            logger.error("mt5.login failed: %s", mt5.last_error())
            mt5.shutdown()
            sys.exit(1)

    logger.info("MT5 initialized")


def ensure_symbol(symbol: str) -> None:
    if not mt5.symbol_select(symbol, True):
        logger.error("symbol_select(%r) failed: %s", symbol, mt5.last_error())
        sys.exit(1)


def rates_to_df(rates: Any) -> pd.DataFrame:
    df = pd.DataFrame(rates)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def fetch_rates(symbol: str, timeframe: int, count: int) -> pd.DataFrame | None:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        logger.warning("No rates for %s tf=%s: %s", symbol, timeframe, mt5.last_error())
        return None
    return rates_to_df(rates)


def closed_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the last row (current forming bar)."""
    if len(df) < 2:
        return df.iloc[0:0].copy()
    return df.iloc[:-1].copy()


def ema(series: pd.Series, period: int = EMA_PERIOD) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def h4_trend(ema_h4: pd.Series, n_bars: int) -> Trend | None:
    """
    Compare EMA on last closed H4 bar vs EMA n_bars earlier (plan: N=6 -> index -1 vs -(N+1)).
    """
    if len(ema_h4) < n_bars + 1:
        return None
    ema_last = float(ema_h4.iloc[-1])
    ema_n = float(ema_h4.iloc[-(n_bars + 1)])
    scale = max(abs(ema_last), abs(ema_n), 1.0)
    eps = 1e-9 * scale
    if abs(ema_last - ema_n) < eps:
        return "flat"
    return "up" if ema_last > ema_n else "down"


def detect_m5_cross(
    m5_closed: pd.DataFrame, ema_m5: pd.Series
) -> tuple[bool, bool] | None:
    """Returns (bullish_cross, bearish_cross) for the last completed M5 bar."""
    need = EMA_PERIOD + 2
    if len(m5_closed) < need or len(ema_m5) < need:
        return None
    prev_close = float(m5_closed["close"].iloc[-2])
    curr_close = float(m5_closed["close"].iloc[-1])
    prev_ema = float(ema_m5.iloc[-2])
    curr_ema = float(ema_m5.iloc[-1])
    bullish = prev_close <= prev_ema and curr_close > curr_ema
    bearish = prev_close >= prev_ema and curr_close < curr_ema
    return bullish, bearish


def load_state() -> dict[str, Any]:
    if not STATE_FILE.is_file():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def bar_time_key(ts: pd.Timestamp) -> int:
    return int(ts.value // 10**9)


def send_discord(cfg: Config, text: str) -> None:
    if cfg.dry_run:
        logger.info("[DRY_RUN] Discord: %s", text)
        return
    if not cfg.discord_webhook_url:
        logger.warning("No DISCORD_WEBHOOK_URL; skipping POST")
        return
    try:
        r = requests.post(
            cfg.discord_webhook_url,
            json={"content": text[:2000]},
            timeout=15,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        logger.error("Discord webhook failed: %s", e)


def run(cfg: Config) -> None:
    init_mt5(cfg)
    ensure_symbol(cfg.symbol)

    state = load_state()
    last_processed: int | None = state.get("last_processed_m5_time")
    bootstrapped = last_processed is not None

    logger.info(
        "Watching %s M5 EMA%d | H4 trend: EMA%d over %d closed H4 bars | dry_run=%s",
        cfg.symbol,
        EMA_PERIOD,
        EMA_PERIOD,
        cfg.h4_trend_bars,
        cfg.dry_run,
    )

    try:
        while True:
            m5_raw = fetch_rates(cfg.symbol, mt5.TIMEFRAME_M5, M5_COUNT)
            h4_raw = fetch_rates(cfg.symbol, mt5.TIMEFRAME_H4, H4_COUNT)
            if m5_raw is None or h4_raw is None:
                time.sleep(POLL_SECONDS)
                continue

            m5_closed = closed_only(m5_raw)
            h4_closed = closed_only(h4_raw)

            min_h4 = EMA_PERIOD + cfg.h4_trend_bars + 1
            if len(h4_closed) < min_h4:
                logger.warning(
                    "Not enough H4 bars (have %s, need %s)",
                    len(h4_closed),
                    min_h4,
                )
                time.sleep(POLL_SECONDS)
                continue

            if len(m5_closed) < EMA_PERIOD + 2:
                logger.warning("Not enough M5 bars for EMA")
                time.sleep(POLL_SECONDS)
                continue

            ema_h4 = ema(h4_closed["close"])
            trend = h4_trend(ema_h4, cfg.h4_trend_bars)
            if trend is None:
                time.sleep(POLL_SECONDS)
                continue

            last_bar_ts = m5_closed["time"].iloc[-1]
            last_key = bar_time_key(last_bar_ts)

            if not bootstrapped:
                last_processed = last_key
                bootstrapped = True
                save_state({"last_processed_m5_time": last_processed})
                logger.info("Bootstrap: skip alerts until next M5 close (bar time=%s)", last_key)
                time.sleep(POLL_SECONDS)
                continue

            if last_processed is not None and last_key <= last_processed:
                time.sleep(POLL_SECONDS)
                continue

            ema_m5 = ema(m5_closed["close"])
            crosses = detect_m5_cross(m5_closed, ema_m5)
            if crosses is None:
                last_processed = last_key
                save_state({"last_processed_m5_time": last_processed})
                time.sleep(POLL_SECONDS)
                continue

            bullish, bearish = crosses
            fire = False
            label = ""
            if trend == "up" and bullish:
                fire = True
                label = "BULLISH M5 cross (H4 uptrend)"
            elif trend == "down" and bearish:
                fire = True
                label = "BEARISH M5 cross (H4 downtrend)"
            elif trend == "flat":
                logger.debug("H4 trend flat; no M5 alert")

            if fire:
                prev_c = float(m5_closed["close"].iloc[-2])
                curr_c = float(m5_closed["close"].iloc[-1])
                prev_e = float(ema_m5.iloc[-2])
                curr_e = float(ema_m5.iloc[-1])
                msg = (
                    f"**{cfg.symbol}** — {label}\n"
                    f"M5 close: {curr_c:.2f} (prev {prev_c:.2f})\n"
                    f"M5 EMA{EMA_PERIOD}: {curr_e:.2f} (prev {prev_e:.2f})\n"
                    f"H4 trend: {trend} (EMA{EMA_PERIOD} vs {cfg.h4_trend_bars} bars back)\n"
                    f"Bar UTC: {m5_closed['time'].iloc[-1]}"
                )
                logger.info("Alert: %s", label)
                send_discord(cfg, msg)

            last_processed = last_key
            save_state({"last_processed_m5_time": last_processed})

            time.sleep(POLL_SECONDS)
    finally:
        mt5.shutdown()
        logger.info("MT5 shutdown")


def main() -> None:
    cfg = load_config()
    run(cfg)


if __name__ == "__main__":
    main()
