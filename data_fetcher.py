"""Market data fetcher — Groww primary, Zerodha Kite fallback.

Connection strategy:
  1. Try Groww first (primary data source, configured via GROWW_API_TOKEN).
  2. If Groww fails or returns no data, try Kite (fallback, requires daily login).
  3. If both fail, return None — the caller handles missing data gracefully.

Circuit breaker:
  Tracks consecutive failures across calls. If error rate exceeds threshold,
  logs a clear warning so the operator knows the data pipeline is degraded.
  The breaker is advisory — we still attempt fetches (they might recover).

Two main entry points:
  - fetch_data(symbol)     → live bot (IST-indexed, today's candles)
  - fetch_intraday(symbol) → backtest (UTC-indexed, date-range candles)
"""

import logging
import time

import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED = ["Open", "High", "Low", "Close", "Volume"]

# ── Circuit breaker ──────────────────────────────────────────────────────────
# Tracks consecutive data-fetch failures. When the pipeline is broken (API down,
# token expired, rate limited), this prevents flooding logs with per-symbol errors
# and gives one clear "pipeline degraded" warning.
_consecutive_failures = 0
_CIRCUIT_BREAKER_THRESHOLD = 10   # warn after this many consecutive failures
_last_circuit_warning = 0.0       # Unix timestamp of last warning (rate-limit logs)


def _record_success():
    """Reset failure counter on any successful data fetch."""
    global _consecutive_failures
    _consecutive_failures = 0


def _record_failure(symbol: str):
    """Increment failure counter and warn if pipeline looks broken."""
    global _consecutive_failures, _last_circuit_warning
    _consecutive_failures += 1
    now = time.time()
    if (_consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD
            and now - _last_circuit_warning > 60):
        logger.error(
            f"DATA PIPELINE DEGRADED: {_consecutive_failures} consecutive fetch failures. "
            f"Last failed symbol: {symbol}. Check Groww token and Kite login status."
        )
        _last_circuit_warning = now


def _validate(df: pd.DataFrame, symbol: str, min_rows: int = 10) -> pd.DataFrame | None:
    """Validate DataFrame has required OHLCV columns and enough rows.

    Returns cleaned DataFrame or None if validation fails.
    """
    if df is None or df.empty:
        return None
    for col in _REQUIRED:
        if col not in df.columns:
            logger.warning(f"[{symbol}] Missing column: {col}")
            return None
    df = df[_REQUIRED].copy().dropna()
    if len(df) < min_rows:
        logger.warning(f"[{symbol}] Only {len(df)} rows (need {min_rows})")
        return None
    return df


def _groww():
    """Return the groww_live module, or None if unavailable.

    Checks is_available() which verifies GROWW_API_TOKEN is set in env.
    Does NOT check if the token is actually valid — that happens on first API call.
    """
    try:
        import groww_live
        if not groww_live.is_available():
            return None
        return groww_live
    except ImportError:
        return None


def _kite():
    """Return the kite_live module, or None if unavailable/not logged in.

    Checks is_available() which verifies both KITE_API_KEY and KITE_ACCESS_TOKEN
    are set. Access tokens expire daily — returns None until user re-authenticates.
    """
    try:
        import kite_live
        if not kite_live.is_available():
            return None
        return kite_live
    except ImportError:
        return None


def fetch_intraday(
    symbol: str,
    start_date=None,
    end_date=None,
) -> pd.DataFrame | None:
    """Fetch multi-day 5-min candles for backtest. Groww first, Kite fallback.

    Returns UTC-indexed DataFrame (backtest engine works in UTC internally).
    If start_date/end_date are provided, fetches that exact range.
    Otherwise fetches last 6 trading days.
    """
    # ── Groww ──────────────────────────────────────────────────────────────
    g = _groww()
    if g:
        try:
            if start_date is not None and end_date is not None:
                from datetime import datetime as dt
                from_dt = dt(start_date.year, start_date.month, start_date.day, 9, 15, 0)
                to_dt   = dt(end_date.year, end_date.month, end_date.day, 15, 30, 0)
                df = g.fetch_historical_candles(symbol, from_dt, to_dt, interval_minutes=5)
            else:
                df = g.fetch_multi_day_candles(symbol, days=6, interval_minutes=5)
            df = _validate(df, symbol, min_rows=10)
            if df is not None:
                # Normalize to UTC for backtest engine
                if df.index.tzinfo is not None:
                    df.index = df.index.tz_convert("UTC")
                else:
                    df.index = df.index.tz_localize("Asia/Kolkata").tz_convert("UTC")
                logger.debug(f"[{symbol}] Groww: {len(df)} candles")
                _record_success()
                return df
        except Exception as e:
            logger.warning(f"[{symbol}] Groww fetch_intraday error: {e}")

    # ── Kite fallback ──────────────────────────────────────────────────────
    k = _kite()
    if k:
        try:
            if start_date is not None and end_date is not None:
                from datetime import datetime as dt
                from zoneinfo import ZoneInfo
                ist = ZoneInfo("Asia/Kolkata")
                from_dt = dt(start_date.year, start_date.month, start_date.day, 9, 15, 0, tzinfo=ist)
                to_dt   = dt(end_date.year, end_date.month, end_date.day, 15, 30, 0, tzinfo=ist)
                df = k.fetch_historical_candles(symbol, from_dt, to_dt, interval_minutes=5)
            else:
                df = k.fetch_multi_day_candles(symbol, days=6, interval_minutes=5)
            df = _validate(df, symbol, min_rows=10)
            if df is not None:
                # Normalize to UTC for backtest engine
                if df.index.tzinfo is not None:
                    df.index = df.index.tz_convert("UTC")
                else:
                    df.index = df.index.tz_localize("Asia/Kolkata").tz_convert("UTC")
                logger.info(f"[{symbol}] Kite fallback: {len(df)} candles")
                _record_success()
                return df
        except Exception as e:
            logger.warning(f"[{symbol}] Kite fetch_intraday error: {e}")

    _record_failure(symbol)
    logger.warning(f"[{symbol}] No data from Groww or Kite")
    return None


def fetch_data(symbol: str) -> pd.DataFrame | None:
    """Fetch live intraday candles. Groww first, Kite fallback.

    Returns IST-indexed DataFrame (live bot works in IST).

    Attempts Groww intraday first, then Groww multi-day, then Kite intraday,
    then Kite multi-day. Returns None only if all four attempts fail.
    """
    # ── Groww ──────────────────────────────────────────────────────────────
    g = _groww()
    if g:
        try:
            df = g.fetch_intraday_candles(symbol, interval_minutes=5)
            df = _validate(df, symbol, min_rows=5)
            if df is not None:
                logger.debug(f"[{symbol}] Groww live: {len(df)} candles")
                _record_success()
                return df

            # Today's candles may be empty before market open — try multi-day
            df = g.fetch_multi_day_candles(symbol, days=5, interval_minutes=5)
            df = _validate(df, symbol, min_rows=10)
            if df is not None:
                logger.debug(f"[{symbol}] Groww historical: {len(df)} candles")
                _record_success()
                return df
        except Exception as e:
            logger.warning(f"[{symbol}] Groww fetch_data error: {e}")

    # ── Kite fallback ──────────────────────────────────────────────────────
    k = _kite()
    if k:
        try:
            df = k.fetch_intraday_candles(symbol, interval_minutes=5)
            df = _validate(df, symbol, min_rows=5)
            if df is not None:
                logger.info(f"[{symbol}] Kite live: {len(df)} candles")
                _record_success()
                return df

            df = k.fetch_multi_day_candles(symbol, days=5, interval_minutes=5)
            df = _validate(df, symbol, min_rows=10)
            if df is not None:
                logger.info(f"[{symbol}] Kite historical: {len(df)} candles")
                _record_success()
                return df
        except Exception as e:
            logger.warning(f"[{symbol}] Kite fetch_data error: {e}")

    _record_failure(symbol)
    logger.warning(f"[{symbol}] No data from Groww or Kite")
    return None
