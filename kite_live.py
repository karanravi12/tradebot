"""Zerodha Kite API — fallback data source when Groww is unavailable.

Auth flow (daily):
  1. User visits /zerodha/login  → redirected to Kite login page
  2. Kite redirects back to /zerodha/callback?request_token=xxx
  3. We exchange request_token for access_token and store in memory + .env

Once authenticated, provides the same interface as groww_live:
  - fetch_intraday_candles(symbol, interval_minutes)
  - fetch_multi_day_candles(symbol, days, interval_minutes)
  - fetch_historical_candles(symbol, from_dt, to_dt, interval_minutes)
  - fetch_batch_ltp(symbols)
"""

import logging
import os
import threading
import time
from datetime import datetime, timedelta

import pandas as pd
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# ── Kite client singleton ──────────────────────────────────────────────────
_kite       = None          # KiteConnect instance
_kite_lock  = threading.Lock()

# ── Instrument token cache ─────────────────────────────────────────────────
# Maps NSE trading symbol → Zerodha instrument_token (int)
_instrument_map: dict[str, int] = {}
_instruments_loaded = False
_instruments_lock   = threading.Lock()

# ── Interval mapping ───────────────────────────────────────────────────────
_INTERVAL_MAP = {
    1:  "minute",
    3:  "3minute",
    5:  "5minute",
    10: "10minute",
    15: "15minute",
    30: "30minute",
    60: "60minute",
}


def is_available() -> bool:
    """True if API key is configured AND an access token is set for today."""
    return bool(
        os.environ.get("KITE_API_KEY")
        and os.environ.get("KITE_ACCESS_TOKEN")
    )


def get_login_url() -> str:
    """Return the Kite login URL. User must visit this daily."""
    from kiteconnect import KiteConnect
    api_key = os.environ.get("KITE_API_KEY", "")
    if not api_key:
        raise RuntimeError("KITE_API_KEY not set in .env")
    kite = KiteConnect(api_key=api_key)
    return kite.login_url()


def set_access_token(request_token: str) -> str:
    """Exchange request_token for access_token. Call from /zerodha/callback."""
    global _kite
    from kiteconnect import KiteConnect
    api_key    = os.environ.get("KITE_API_KEY", "")
    api_secret = os.environ.get("KITE_API_SECRET", "")
    if not api_key or not api_secret:
        raise RuntimeError("KITE_API_KEY or KITE_API_SECRET not set in .env")

    kite = KiteConnect(api_key=api_key)
    data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = data["access_token"]

    kite.set_access_token(access_token)
    with _kite_lock:
        _kite = kite

    # Persist in env + .env file for restarts
    os.environ["KITE_ACCESS_TOKEN"] = access_token
    _write_env_token(access_token)

    logger.info("Kite access token set — Zerodha data source active")
    return access_token


def _write_env_token(token: str):
    """Update KITE_ACCESS_TOKEN in .env file."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    try:
        with open(env_path) as f:
            lines = f.readlines()
        with open(env_path, "w") as f:
            for line in lines:
                if line.startswith("KITE_ACCESS_TOKEN="):
                    f.write(f"KITE_ACCESS_TOKEN={token}\n")
                else:
                    f.write(line)
    except Exception as e:
        logger.warning(f"Could not update .env with Kite token: {e}")


def _get_kite():
    """Return authenticated KiteConnect instance, or raise if not logged in."""
    global _kite
    with _kite_lock:
        if _kite is not None:
            return _kite

        api_key      = os.environ.get("KITE_API_KEY", "")
        access_token = os.environ.get("KITE_ACCESS_TOKEN", "")
        if not api_key or not access_token:
            raise RuntimeError(
                "Kite not authenticated. Visit /zerodha/login to log in."
            )

        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        _kite = kite
        return _kite


def _load_instruments():
    """Download NSE instrument list and build symbol → token map."""
    global _instrument_map, _instruments_loaded
    with _instruments_lock:
        if _instruments_loaded:
            return
        try:
            kite        = _get_kite()
            instruments = kite.instruments("NSE")
            _instrument_map = {
                i["tradingsymbol"]: i["instrument_token"]
                for i in instruments
                if i["exchange"] == "NSE"
            }
            _instruments_loaded = True
            logger.info(f"Kite: loaded {len(_instrument_map)} NSE instruments")
        except Exception as e:
            logger.error(f"Kite: failed to load instruments: {e}")


def _token(symbol: str) -> int:
    """Return Zerodha instrument_token for an NSE symbol."""
    if not _instruments_loaded:
        _load_instruments()
    tok = _instrument_map.get(symbol)
    if tok is None:
        raise KeyError(f"Symbol not found in Kite instruments: {symbol}")
    return tok


def _to_df(records: list, symbol: str) -> pd.DataFrame | None:
    """Convert Kite historical_data records to OHLCV DataFrame (IST-indexed)."""
    if not records:
        return None
    df = pd.DataFrame(records)
    df.rename(columns={
        "date":   "Date",
        "open":   "Open",
        "high":   "High",
        "low":    "Low",
        "close":  "Close",
        "volume": "Volume",
    }, inplace=True)
    df.set_index("Date", inplace=True)
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize(IST)
    else:
        df.index = df.index.tz_convert(IST)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def fetch_intraday_candles(symbol: str, interval_minutes: int = 5) -> pd.DataFrame | None:
    """Fetch today's intraday candles for a symbol."""
    try:
        kite     = _get_kite()
        tok      = _token(symbol)
        interval = _INTERVAL_MAP.get(interval_minutes, "5minute")
        today    = datetime.now(IST).date()
        from_dt  = datetime(today.year, today.month, today.day, 9, 15, tzinfo=IST)
        to_dt    = datetime(today.year, today.month, today.day, 15, 30, tzinfo=IST)
        records  = kite.historical_data(tok, from_dt, to_dt, interval)
        return _to_df(records, symbol)
    except Exception as e:
        logger.debug(f"[{symbol}] Kite intraday error: {e}")
        return None


def fetch_multi_day_candles(symbol: str, days: int = 5, interval_minutes: int = 5) -> pd.DataFrame | None:
    """Fetch candles for the last N trading days."""
    try:
        kite     = _get_kite()
        tok      = _token(symbol)
        interval = _INTERVAL_MAP.get(interval_minutes, "5minute")
        to_dt    = datetime.now(IST)
        from_dt  = to_dt - timedelta(days=days + 2)  # buffer for weekends
        records  = kite.historical_data(tok, from_dt, to_dt, interval)
        return _to_df(records, symbol)
    except Exception as e:
        logger.debug(f"[{symbol}] Kite multi-day error: {e}")
        return None


def fetch_historical_candles(
    symbol: str,
    from_dt: datetime,
    to_dt: datetime,
    interval_minutes: int = 5,
) -> pd.DataFrame | None:
    """Fetch candles for an explicit date range (used by backtest)."""
    try:
        kite     = _get_kite()
        tok      = _token(symbol)
        interval = _INTERVAL_MAP.get(interval_minutes, "5minute")
        records  = kite.historical_data(tok, from_dt, to_dt, interval)
        return _to_df(records, symbol)
    except Exception as e:
        logger.debug(f"[{symbol}] Kite historical error: {e}")
        return None


def fetch_batch_ltp(symbols: list[str]) -> dict[str, float]:
    """Fetch last traded prices for multiple symbols in one call.

    Returns dict of {symbol: ltp}. Missing symbols are omitted.
    Kite quote() accepts up to 500 instruments per call.
    """
    if not symbols:
        return {}
    try:
        kite        = _get_kite()
        if not _instruments_loaded:
            _load_instruments()
        instruments = []
        sym_map     = {}  # "NSE:SYMBOL" → symbol
        for sym in symbols:
            tok = _instrument_map.get(sym)
            if tok:
                key          = f"NSE:{sym}"
                instruments.append(key)
                sym_map[key] = sym

        if not instruments:
            return {}

        quotes = kite.ltp(instruments)
        result = {}
        for key, data in quotes.items():
            sym = sym_map.get(key)
            if sym and data.get("last_price"):
                result[sym] = data["last_price"]
        return result
    except Exception as e:
        logger.debug(f"Kite batch LTP error: {e}")
        return {}
