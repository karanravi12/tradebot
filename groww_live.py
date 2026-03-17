"""Groww Partner API client — single source of truth for all market data.

Uses the official growwapi Python SDK.
Docs: https://groww.in/trade-api/docs/python-sdk

Authentication:
  GROWW_API_TOKEN = long-lived auth key (auth-totp role)
  GROWW_API_SECRET = secret — used to exchange for a ~12h session token
  Session tokens are cached and auto-refreshed before expiry.

Provides:
  - Live LTP / quote           → fetch_ltp(), fetch_quote()
  - Today's intraday candles   → fetch_intraday_candles()
  - Historical candles (date range) → fetch_historical_candles()
  - Multi-day candles          → fetch_multi_day_candles()

Strictly READ-ONLY. No order placement. No account access.
"""

import base64
import json
import logging
import os
import threading
import time
import warnings
from datetime import datetime, timedelta

import pandas as pd
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

_client = None          # GrowwAPI instance
_session_expiry = 0.0   # Unix timestamp when the session token expires

# ── Rate limiter ──────────────────────────────────────────────────────────────
# Groww API enforces 10 req/sec and 300 req/min. We enforce a minimum gap of 0.2s
# (~300 req/min) to stay at the limit. The lock makes this thread-safe so
# concurrent callers (scheduler + UI) don't race.
_RATE_LIMIT_DELAY = 0.20   # seconds between API calls (~300 req/min, matching Groww limit)
_last_api_call_ts = 0.0
_rate_limit_lock = threading.Lock()
_RATE_LIMIT_BACKOFF = 30.0  # seconds to freeze the pipeline after a 429 response


def _rate_limit():
    """Sleep if needed to respect the Groww API rate limit (thread-safe)."""
    global _last_api_call_ts
    with _rate_limit_lock:
        elapsed = time.time() - _last_api_call_ts
        if elapsed < _RATE_LIMIT_DELAY:
            time.sleep(_RATE_LIMIT_DELAY - elapsed)
        _last_api_call_ts = time.time()


def _apply_backoff():
    """After a 429 response, freeze the global rate limiter for BACKOFF seconds.

    This prevents the cascade where every subsequent symbol in the scan
    immediately hits the rate limit too. All callers share this delay since
    _last_api_call_ts is module-level.
    """
    global _last_api_call_ts
    with _rate_limit_lock:
        # Push next-allowed-call time forward. _rate_limit() will then sleep
        # (DELAY - elapsed) where elapsed is negative → sleeps for ~BACKOFF.
        _last_api_call_ts = time.time() + _RATE_LIMIT_BACKOFF
    logger.warning(f"Rate limit hit — pausing all API calls for {_RATE_LIMIT_BACKOFF}s")


def _next_6am_ist() -> float:
    """Return Unix timestamp of the next 6:00 AM IST (when Groww tokens expire)."""
    now = datetime.now(IST)
    candidate = now.replace(hour=6, minute=0, second=0, microsecond=0)
    if now >= candidate:
        candidate = candidate + timedelta(days=1)
    return candidate.timestamp()


def _jwt_expiry(token: str) -> float:
    """Extract expiry (Unix timestamp) from a JWT payload without a library.

    Groww tokens carry a JWT exp field set years in the future, but the docs
    state the token is invalidated at 6:00 AM IST every day.  We take the
    minimum of the two so we always re-auth before the server rejects us.
    """
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        jwt_exp = float(payload.get("exp", 0))
    except Exception:
        jwt_exp = 0.0
    next_6am = _next_6am_ist()
    # Use whichever is sooner; fall back to next 6 AM if JWT is missing/zero
    return min(jwt_exp, next_6am) if jwt_exp > 0 else next_6am


def _get_client():
    """Return a GrowwAPI client with a valid session token, refreshing if needed."""
    global _client, _session_expiry

    # Fast path — no lock needed if client is valid
    if _client is not None and time.time() <= _session_expiry - 300:
        return _client

    # Slow path — acquire lock so only one thread calls get_access_token
    with _rate_limit_lock:
        # Re-check after acquiring lock (another thread may have just refreshed)
        if _client is not None and time.time() <= _session_expiry - 300:
            return _client

        api_key = os.environ.get("GROWW_API_TOKEN", "")
        secret  = os.environ.get("GROWW_API_SECRET", "")
        if not api_key:
            raise RuntimeError("GROWW_API_TOKEN not set — check .env file")

        from growwapi import GrowwAPI

        if secret:
            session_token = GrowwAPI.get_access_token(api_key=api_key, secret=secret)
            _session_expiry = _jwt_expiry(session_token)
            logger.debug(f"Groww session token refreshed, expires at {datetime.fromtimestamp(_session_expiry, tz=IST).strftime('%Y-%m-%d %H:%M IST')}")
        else:
            session_token = api_key
            _session_expiry = _jwt_expiry(session_token)

        _client = GrowwAPI(session_token)

    return _client


def is_available() -> bool:
    return bool(os.environ.get("GROWW_API_TOKEN", ""))


# ── Candle parser ─────────────────────────────────────────────────────────────

def _parse_candles(raw: list) -> pd.DataFrame | None:
    """Parse Groww SDK candle list into an IST-indexed DataFrame.

    SDK format: [unix_seconds, open, high, low, close, volume]
    """
    if not raw:
        return None
    rows = []
    for c in raw:
        try:
            ts = pd.Timestamp(c[0], unit="s", tz="UTC").tz_convert(IST)
            rows.append({
                "timestamp": ts,
                "Open":   float(c[1]),
                "High":   float(c[2]),
                "Low":    float(c[3]),
                "Close":  float(c[4]),
                "Volume": int(c[5]) if c[5] is not None else 0,
            })
        except (IndexError, TypeError, ValueError) as e:
            logger.debug(f"Candle parse skip: {e} | raw={c}")
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("timestamp")
    df.index.name = None
    return df.sort_index()


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_ltp(symbol: str) -> float | None:
    """Last traded price for an NSE equity symbol."""
    try:
        _rate_limit()
        g = _get_client()
        q = g.get_quote(exchange=g.EXCHANGE_NSE, segment=g.SEGMENT_CASH,
                        trading_symbol=symbol)
        ltp = q.get("last_price") or (q.get("ohlc") or {}).get("close")
        return float(ltp) if ltp is not None else None
    except RuntimeError:
        raise
    except Exception as e:
        if "rate limit" in str(e).lower():
            _apply_backoff()
        logger.warning(f"Groww LTP error for {symbol}: {e}")
        return None


def fetch_quote(symbol: str) -> dict | None:
    """Full live quote (OHLCV + LTP) for an NSE equity symbol."""
    try:
        _rate_limit()
        g = _get_client()
        q = g.get_quote(exchange=g.EXCHANGE_NSE, segment=g.SEGMENT_CASH,
                        trading_symbol=symbol)
        ohlc = q.get("ohlc") or {}
        return {
            "open":   float(ohlc.get("open",  0)),
            "high":   float(ohlc.get("high",  0)),
            "low":    float(ohlc.get("low",   0)),
            "close":  float(ohlc.get("close", 0)),
            "ltp":    float(q.get("last_price") or ohlc.get("close") or 0),
            "volume": int(q.get("volume", 0)),
        }
    except RuntimeError:
        raise
    except Exception as e:
        if "rate limit" in str(e).lower():
            _apply_backoff()
        logger.warning(f"Groww quote error for {symbol}: {e}")
        return None


_BATCH_SIZE = 50   # Groww Live Data API: max 50 instruments per call


def fetch_batch_ltp(symbols: list[str]) -> dict[str, float]:
    """Batch-fetch LTP for up to N symbols using a single API call per 50 instruments.

    Groww's get_ltp() accepts a tuple of "NSE_<SYMBOL>" strings (max 50 per call).
    Returns {symbol: ltp} for every symbol that returned a valid price.
    Failed symbols are silently skipped (caller should handle missing keys).
    """
    if not symbols:
        return {}

    result: dict[str, float] = {}
    try:
        g = _get_client()
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning(f"fetch_batch_ltp: client init failed: {e}")
        return result

    for i in range(0, len(symbols), _BATCH_SIZE):
        chunk = symbols[i : i + _BATCH_SIZE]
        exchange_symbols = tuple(f"NSE_{s}" for s in chunk)
        for attempt in range(3):
            try:
                _rate_limit()
                data = g.get_ltp(
                    exchange_trading_symbols=exchange_symbols,
                    segment=g.SEGMENT_CASH,
                )
                # Response: {"NSE_RELIANCE": {"ltp": 1234.5}, ...} or similar
                for key, val in (data or {}).items():
                    sym = key.replace("NSE_", "", 1)
                    ltp = None
                    if isinstance(val, dict):
                        ltp = val.get("ltp") or val.get("last_price")
                    elif isinstance(val, (int, float)):
                        ltp = val
                    if ltp is not None:
                        result[sym] = float(ltp)
                break
            except Exception as e:
                err_str = str(e)
                if "rate limit" in err_str.lower():
                    _apply_backoff()
                    if attempt < 2:
                        continue
                logger.warning(f"fetch_batch_ltp chunk {i//50+1} error: {e}")
                break

    logger.debug(f"fetch_batch_ltp: {len(result)}/{len(symbols)} symbols fetched")
    return result


def fetch_historical_candles(
    symbol: str,
    from_dt: datetime,
    to_dt: datetime,
    interval_minutes: int = 5,
) -> pd.DataFrame | None:
    """Historical OHLCV candles for a date range (IST-indexed).

    Args:
        symbol:           NSE trading symbol e.g. "RELIANCE"
        from_dt:          start datetime (naive, treated as IST)
        to_dt:            end   datetime (naive, treated as IST)
        interval_minutes: 1, 2, 3, 5, 10, 15, 30, 60
    """
    try:
        g = _get_client()
        start_str = from_dt.strftime("%Y-%m-%d %H:%M:%S")
        end_str   = to_dt.strftime("%Y-%m-%d %H:%M:%S")

        for attempt in range(3):
            _rate_limit()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    data = g.get_historical_candle_data(
                        trading_symbol=symbol,
                        exchange=g.EXCHANGE_NSE,
                        segment=g.SEGMENT_CASH,
                        start_time=start_str,
                        end_time=end_str,
                        interval_in_minutes=interval_minutes,
                    )
                candles = data.get("candles") or []
                df = _parse_candles(candles)
                if df is not None:
                    logger.debug(f"[{symbol}] Groww historical: {len(df)} candles "
                                 f"({start_str[:10]}→{end_str[:10]})")
                return df
            except Exception as e:
                err_str = str(e)
                if "rate limit" in err_str.lower() or "Rate limit" in err_str:
                    logger.warning(f"Groww historical error ({symbol}): {e}")
                    _apply_backoff()   # freeze the global pipeline, then retry
                    continue
                logger.warning(f"Groww historical error ({symbol}): {e}")
                return None
        return None
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning(f"Groww historical error ({symbol}): {e}")
        return None


def fetch_intraday_candles(symbol: str, interval_minutes: int = 5) -> pd.DataFrame | None:
    """Today's intraday OHLCV candles (IST-indexed)."""
    today = datetime.now(ZoneInfo("Asia/Kolkata"))
    from_dt = datetime(today.year, today.month, today.day, 9, 15, 0)
    to_dt   = datetime(today.year, today.month, today.day, 15, 30, 0)
    return fetch_historical_candles(symbol, from_dt, to_dt, interval_minutes)


def fetch_multi_day_candles(
    symbol: str,
    days: int = 6,
    interval_minutes: int = 5,
) -> pd.DataFrame | None:
    """Fetch the last N trading days of intraday candles (IST-indexed)."""
    now     = datetime.now(ZoneInfo("Asia/Kolkata"))
    to_dt   = now.replace(tzinfo=None)
    from_dt = (now - timedelta(days=days + 3)).replace(tzinfo=None)
    from_dt = from_dt.replace(hour=9, minute=15, second=0, microsecond=0)

    df = fetch_historical_candles(symbol, from_dt, to_dt, interval_minutes)
    if df is None:
        return None
    # Trim to last `days` unique trading dates
    unique_dates = sorted(set(ts.date() for ts in df.index))
    if len(unique_dates) > days:
        cutoff = unique_dates[-days]
        df = df[pd.Series(df.index).apply(lambda t: t.date() >= cutoff).values]
    return df if len(df) >= 10 else None


def test_connection() -> dict:
    """Test the Groww API — returns status dict."""
    result = {"ok": False, "ltp": None, "candles": None, "error": None}
    if not is_available():
        result["error"] = "GROWW_API_TOKEN not set in .env"
        return result
    try:
        ltp = fetch_ltp("RELIANCE")
        if ltp and ltp > 0:
            result["ok"]  = True
            result["ltp"] = ltp
            logger.info(f"Groww API ✓ — RELIANCE LTP: ₹{ltp:.2f}")
        else:
            result["error"] = "Could not fetch LTP — check token / API access"
            return result

        df = fetch_intraday_candles("RELIANCE")
        if df is not None:
            result["candles"] = len(df)
    except RuntimeError as e:
        result["error"] = str(e)
    except Exception as e:
        result["error"] = f"Unexpected: {e}"
    return result
