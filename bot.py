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
    discord_role_id: str | None
    symbol: str
    h4_trend_bars: int
    m5_prep_minutes: int
    m5_prep_window_seconds: int
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

    role_id_raw = os.environ.get("DISCORD_ROLE_ID", "").strip()
    role_id = role_id_raw or None

    h4_bars = int(os.environ.get("H4_TREND_BARS", "6"))
    if h4_bars < 1:
        logger.error("H4_TREND_BARS must be >= 1")
        sys.exit(1)

    # Parsing prep minutes as an integer
    try:
        prep_minutes = int(os.environ.get("M5_PREP_MINUTES", "3"))
    except ValueError:
        logger.error("M5_PREP_MINUTES must be an integer (e.g. 3)")
        sys.exit(1)

    if prep_minutes < 0 or prep_minutes >= 5:
        logger.error("M5_PREP_MINUTES must be >= 0 and < 5")
        sys.exit(1)

    # Parsing prep window seconds
    try:
        prep_window_s = int(os.environ.get("M5_PREP_WINDOW_SECONDS", "90"))
    except ValueError:
        logger.error("M5_PREP_WINDOW_SECONDS must be an integer")
        sys.exit(1)

    if prep_window_s < 1 or prep_window_s > 300:
        logger.error("M5_PREP_WINDOW_SECONDS must be between 1 and 300")
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
        discord_role_id=role_id,
        symbol=os.environ.get("SYMBOL", "US100Cash").strip(),
        h4_trend_bars=h4_bars,
        m5_prep_minutes=prep_minutes,
        m5_prep_window_seconds=prep_window_s,
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
        
    # We now use the OPEN and CLOSE of the completed candle, matching the prep logic
    curr_open = float(m5_closed["open"].iloc[-1])
    curr_close = float(m5_closed["close"].iloc[-1])
    curr_ema = float(ema_m5.iloc[-1])
    
    # Bullish: Opened below/on the EMA, closed strictly above it
    bullish = (curr_open <= curr_ema) and (curr_close > curr_ema)
    # Bearish: Opened above/on the EMA, closed strictly below it
    bearish = (curr_open >= curr_ema) and (curr_close < curr_ema)
    
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
        mention = f"<@&{cfg.discord_role_id}> " if cfg.discord_role_id else ""
        logger.info("[DRY_RUN] Discord: %s", f"{mention}{text}")
        return
    if not cfg.discord_webhook_url:
        logger.warning("No DISCORD_WEBHOOK_URL; skipping POST")
        return
    try:
        content = text
        if cfg.discord_role_id:
            prefix = f"<@&{cfg.discord_role_id}> "
            content = prefix + text[: max(0, 2000 - len(prefix))]

        r = requests.post(
            cfg.discord_webhook_url,
            json={"content": content[:2000]},
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
    last_prep: int | None = state.get("last_prep_m5_time")
    bootstrapped = last_processed is not None

    logger.info(
        "Watching %s M5 EMA%d | H4 trend: EMA%d over %d closed H4 bars | dry_run=%s",
        cfg.symbol,
        EMA_PERIOD,
        EMA_PERIOD,
        cfg.h4_trend_bars,
        cfg.dry_run,
    )

    startup_msg = (
        f"🟢 **BOT STARTED** 🟢\n"
        f"Watching: **{cfg.symbol}**\n"
        f"Strategy: M5 EMA{EMA_PERIOD} Cross (H4 Trend Filter)\n"
        f"Prep Ping: {cfg.m5_prep_minutes}m into candle"
    )
    send_discord(cfg, startup_msg)
    
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

            # --- PREPARATION PING LOGIC ---
            if cfg.m5_prep_minutes >= 0 and len(m5_raw) >= EMA_PERIOD + 2:
                forming_open_ts = m5_raw["time"].iloc[-1]
                forming_key = bar_time_key(forming_open_ts)

                if bootstrapped:
                    # 1. Get accurate server time from MT5
                    tick = mt5.symbol_info_tick(cfg.symbol)
                    if tick is not None:
                        current_server_time = tick.time
                        forming_open_time_int = int(forming_open_ts.timestamp())
                        elapsed_s = float(current_server_time - forming_open_time_int)

                        # 2. Check if we are in the time window (e.g. at 3 mins)
                        start_s = cfg.m5_prep_minutes * 60
                        end_s = start_s + float(cfg.m5_prep_window_seconds)

                        if start_s <= elapsed_s < end_s and (last_prep is None or forming_key != last_prep):
                            
                            # 3. Check for an ACTUAL forming crossover using Open and Close
                            ema_m5_live = ema(m5_raw["close"])
                            forming_open = float(m5_raw["open"].iloc[-1])
                            forming_close = float(m5_raw["close"].iloc[-1])
                            forming_ema = float(ema_m5_live.iloc[-1])

                            bullish_forming = (forming_open <= forming_ema) and (forming_close > forming_ema)
                            bearish_forming = (forming_open >= forming_ema) and (forming_close < forming_ema)

                            # 4. Verify the forming M5 cross aligns with H4 trend
                            prep_ok = False
                            label_prep = ""
                            if trend == "up" and bullish_forming:
                                prep_ok = True
                                label_prep = "BULLISH"
                            elif trend == "down" and bearish_forming:
                                prep_ok = True
                                label_prep = "BEARISH"

                            # 5. Fire the Discord alert
                            if prep_ok:
                                seconds_left = int(300 - elapsed_s)
                                msg = (
                                    f"⏳ **PREP ALERT: {cfg.symbol}** ⏳\n"
                                    f"An M5 {label_prep} cross is actively piercing the EMA with ~{seconds_left}s left to close.\n"
                                    f"H4 Trend is {trend.upper()}.\n"
                                    f"Get ready!"
                                )
                                logger.info("Prep Alert Fired: %s cross forming.", label_prep)
                                send_discord(cfg, msg)
                                
                                # Update state so we don't spam the channel
                                last_prep = forming_key
                                save_state({
                                    "last_processed_m5_time": last_processed,
                                    "last_prep_m5_time": last_prep
                                })
            # --- END PREPARATION PING LOGIC ---

            last_bar_ts = m5_closed["time"].iloc[-1]
            last_key = bar_time_key(last_bar_ts)

            if not bootstrapped:
                last_processed = last_key
                # Also set last_prep to current unclosed bar to prevent instant firing on boot
                last_prep = bar_time_key(m5_raw["time"].iloc[-1]) 
                bootstrapped = True
                save_state({
                    "last_processed_m5_time": last_processed,
                    "last_prep_m5_time": last_prep
                })
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
                save_state({
                    "last_processed_m5_time": last_processed,
                    "last_prep_m5_time": last_prep
                })
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
                    f"🚨 **{cfg.symbol}** — {label} 🚨\n"
                    f"M5 close: {curr_c:.2f} (prev {prev_c:.2f})\n"
                    f"M5 EMA{EMA_PERIOD}: {curr_e:.2f} (prev {prev_e:.2f})\n"
                    f"H4 trend: {trend} (EMA{EMA_PERIOD} vs {cfg.h4_trend_bars} bars back)\n"
                    f"Bar UTC: {m5_closed['time'].iloc[-1]}"
                )
                logger.info("Alert: %s", label)
                send_discord(cfg, msg)

            last_processed = last_key
            save_state({
                "last_processed_m5_time": last_processed,
                "last_prep_m5_time": last_prep
            })

            time.sleep(POLL_SECONDS)
    finally:
        mt5.shutdown()
        logger.info("MT5 shutdown")

def main() -> None:
    cfg = load_config()
    run(cfg)

if __name__ == "__main__":
    main()