"""Zerodha Kite API — fallback data source when Groww is unavailable.

Connection flow:
  1. On app start, is_available() checks for KITE_API_KEY + KITE_ACCESS_TOKEN in env.
  2. If not authenticated, user visits /zerodha/login → redirected to Kite login page.
  3. Kite redirects back to /zerodha/callback?request_token=xxx
  4. set_access_token() exchanges request_token for access_token, stores in memory + .env.
  5. Access tokens expire daily — user must re-login each morning before market open.

Data flow (all methods):
  1. _get_kite() returns authenticated KiteConnect instance (or raises if not logged in).
  2. _token(symbol) resolves NSE symbol → instrument_token via cached instrument list.
  3. Kite historical_data() API returns OHLCV records.
  4. _to_df() converts records to IST-indexed pandas DataFrame.

Provides the same interface as groww_live:
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
# Guarded by _kite_lock. Initialised lazily on first API call after login.
_kite       = None          # KiteConnect instance
_kite_lock  = threading.Lock()

# ── Instrument token cache ─────────────────────────────────────────────────
# Maps NSE trading symbol → Zerodha instrument_token (int).
# Refreshed once per day (instruments change rarely, but new listings appear).
_instrument_map: dict[str, int] = {}
_instruments_loaded = False
_instruments_lock   = threading.Lock()
_instruments_loaded_date: str = ""  # "YYYY-MM-DD" when last loaded — refresh daily

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
    """Update KITE_ACCESS_TOKEN in .env file (local dev only — no-op on Railway)."""
    if os.environ.get("RAILWAY_ENVIRONMENT"):
        return  # Railway env vars are set in dashboard, not writable at runtime
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
    """Download NSE instrument list and build symbol → token map.

    Called lazily on first symbol lookup. Refreshes daily — the instrument
    list is ~2000 entries and changes slowly, but new listings or symbol
    renames do happen. The date check ensures we pick up changes each morning.
    """
    global _instrument_map, _instruments_loaded, _instruments_loaded_date
    today_str = datetime.now(IST).strftime("%Y-%m-%d")
    with _instruments_lock:
        # Skip if already loaded today
        if _instruments_loaded and _instruments_loaded_date == today_str:
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
            _instruments_loaded_date = today_str
            logger.info(f"Kite: loaded {len(_instrument_map)} NSE instruments (date={today_str})")
        except Exception as e:
            logger.warning(f"Kite: failed to load instruments: {e}")


def _token(symbol: str) -> int:
    """Return Zerodha instrument_token for an NSE symbol.

    Triggers instrument list download on first call of the day.
    Raises KeyError if symbol isn't in the NSE instrument list.
    """
    today_str = datetime.now(IST).strftime("%Y-%m-%d")
    if not _instruments_loaded or _instruments_loaded_date != today_str:
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
    """Fetch today's intraday candles for a symbol.

    Returns IST-indexed DataFrame with OHLCV columns, or None on any error.
    """
    try:
        kite     = _get_kite()
        tok      = _token(symbol)
        interval = _INTERVAL_MAP.get(interval_minutes, "5minute")
        today    = datetime.now(IST).date()
        from_dt  = datetime(today.year, today.month, today.day, 9, 15, tzinfo=IST)
        to_dt    = datetime(today.year, today.month, today.day, 15, 30, tzinfo=IST)
        records  = kite.historical_data(tok, from_dt, to_dt, interval)
        return _to_df(records, symbol)
    except KeyError:
        # Symbol not in instrument list — not actionable, skip quietly
        logger.debug(f"[{symbol}] Kite: symbol not in instruments")
        return None
    except Exception as e:
        logger.warning(f"[{symbol}] Kite intraday error: {e}")
        return None


def fetch_multi_day_candles(symbol: str, days: int = 5, interval_minutes: int = 5) -> pd.DataFrame | None:
    """Fetch candles for the last N trading days.

    Adds 2-day buffer to account for weekends when calculating the start date.
    """
    try:
        kite     = _get_kite()
        tok      = _token(symbol)
        interval = _INTERVAL_MAP.get(interval_minutes, "5minute")
        to_dt    = datetime.now(IST)
        from_dt  = to_dt - timedelta(days=days + 2)  # buffer for weekends
        records  = kite.historical_data(tok, from_dt, to_dt, interval)
        return _to_df(records, symbol)
    except KeyError:
        logger.debug(f"[{symbol}] Kite: symbol not in instruments")
        return None
    except Exception as e:
        logger.warning(f"[{symbol}] Kite multi-day error: {e}")
        return None


def fetch_historical_candles(
    symbol: str,
    from_dt: datetime,
    to_dt: datetime,
    interval_minutes: int = 5,
) -> pd.DataFrame | None:
    """Fetch candles for an explicit date range (used by backtest).

    Kite historical API has a per-request limit of ~60 days for minute-level
    data. For longer ranges, the caller should chunk the requests.
    """
    try:
        kite     = _get_kite()
        tok      = _token(symbol)
        interval = _INTERVAL_MAP.get(interval_minutes, "5minute")
        records  = kite.historical_data(tok, from_dt, to_dt, interval)
        return _to_df(records, symbol)
    except KeyError:
        logger.debug(f"[{symbol}] Kite: symbol not in instruments")
        return None
    except Exception as e:
        logger.warning(f"[{symbol}] Kite historical error: {e}")
        return None


def fetch_batch_ltp(symbols: list[str]) -> dict[str, float]:
    """Fetch last traded prices for multiple symbols in one call.

    Returns dict of {symbol: ltp}. Missing symbols are silently omitted.
    Kite quote() accepts up to 500 instruments per call, so no batching needed
    for our ~223 symbol universe.

    On any error (auth expired, network timeout, etc.), returns an empty dict.
    The caller (trader.py batch LTP pre-screen) falls back to scanning all symbols.
    """
    if not symbols:
        return {}
    try:
        kite = _get_kite()
        today_str = datetime.now(IST).strftime("%Y-%m-%d")
        if not _instruments_loaded or _instruments_loaded_date != today_str:
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
            logger.warning(f"Kite batch LTP: no symbols resolved to instrument tokens "
                          f"(requested {len(symbols)})")
            return {}

        quotes = kite.ltp(instruments)
        result = {}
        for key, data in quotes.items():
            sym = sym_map.get(key)
            if sym and data.get("last_price"):
                result[sym] = data["last_price"]

        if len(result) < len(symbols) * 0.5:
            logger.warning(f"Kite batch LTP: only {len(result)}/{len(symbols)} symbols returned prices")

        return result
    except RuntimeError as e:
        # Auth not configured — expected during normal Groww-only operation
        logger.debug(f"Kite batch LTP: not authenticated ({e})")
        return {}
    except Exception as e:
        logger.warning(f"Kite batch LTP error ({len(symbols)} symbols): {e}")
        return {}
