"""Market data fetcher — Groww primary, Zerodha Kite fallback.

Live bot (fetch_data):     Groww first → Kite if Groww fails/rate-limited.
Backtest (fetch_intraday): Groww first → Kite fallback for date range fetches.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED = ["Open", "High", "Low", "Close", "Volume"]


def _validate(df: pd.DataFrame, symbol: str, min_rows: int = 10) -> pd.DataFrame | None:
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
    """Return the groww_live module, or None if unavailable."""
    try:
        import groww_live
        if not groww_live.is_available():
            return None
        return groww_live
    except ImportError:
        return None


def _kite():
    """Return the kite_live module, or None if unavailable/not logged in."""
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

    Returns UTC-indexed DataFrame.
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
                if df.index.tzinfo is not None:
                    df.index = df.index.tz_convert("UTC")
                else:
                    df.index = df.index.tz_localize("Asia/Kolkata").tz_convert("UTC")
                logger.debug(f"[{symbol}] Groww: {len(df)} candles")
                return df
        except Exception as e:
            logger.debug(f"[{symbol}] Groww fetch_intraday error: {e}")

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
                if df.index.tzinfo is not None:
                    df.index = df.index.tz_convert("UTC")
                else:
                    df.index = df.index.tz_localize("Asia/Kolkata").tz_convert("UTC")
                logger.debug(f"[{symbol}] Kite fallback: {len(df)} candles")
                return df
        except Exception as e:
            logger.debug(f"[{symbol}] Kite fetch_intraday error: {e}")

    logger.warning(f"[{symbol}] No data from Groww or Kite")
    return None


def fetch_data(symbol: str) -> pd.DataFrame | None:
    """Fetch live intraday candles. Groww first, Kite fallback.

    Returns IST-indexed DataFrame, or None if both sources fail.
    """
    # ── Groww ──────────────────────────────────────────────────────────────
    g = _groww()
    if g:
        try:
            df = g.fetch_intraday_candles(symbol, interval_minutes=5)
            df = _validate(df, symbol, min_rows=5)
            if df is not None:
                logger.debug(f"[{symbol}] Groww live: {len(df)} candles")
                return df

            df = g.fetch_multi_day_candles(symbol, days=5, interval_minutes=5)
            df = _validate(df, symbol, min_rows=10)
            if df is not None:
                logger.debug(f"[{symbol}] Groww historical: {len(df)} candles")
                return df
        except Exception as e:
            logger.debug(f"[{symbol}] Groww fetch_data error: {e}")

    # ── Kite fallback ──────────────────────────────────────────────────────
    k = _kite()
    if k:
        try:
            df = k.fetch_intraday_candles(symbol, interval_minutes=5)
            df = _validate(df, symbol, min_rows=5)
            if df is not None:
                logger.debug(f"[{symbol}] Kite live: {len(df)} candles")
                return df

            df = k.fetch_multi_day_candles(symbol, days=5, interval_minutes=5)
            df = _validate(df, symbol, min_rows=10)
            if df is not None:
                logger.debug(f"[{symbol}] Kite historical: {len(df)} candles")
                return df
        except Exception as e:
            logger.debug(f"[{symbol}] Kite fetch_data error: {e}")

    logger.warning(f"[{symbol}] No data from Groww or Kite")
    return None
