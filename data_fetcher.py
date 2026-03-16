"""Market data fetcher — Groww primary for both live and backtest.

Live bot (fetch_data):   Groww only — real-time IST candles.
Backtest (fetch_intraday): Groww only — uses fetch_historical_candles for the
                           requested date range. Yahoo was removed per user preference.
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
    """Return the groww_live module, or raise if unavailable."""
    try:
        import groww_live
        if not groww_live.is_available():
            raise RuntimeError(
                "GROWW_API_TOKEN not set. Add it to .env and restart."
            )
        return groww_live
    except ImportError:
        raise RuntimeError("groww_live.py module not found.")


def fetch_intraday(
    symbol: str,
    start_date=None,
    end_date=None,
) -> pd.DataFrame | None:
    """Fetch multi-day 5-min candles for backtest. Uses Groww only.

    Args:
        symbol:     NSE trading symbol
        start_date: optional date — when provided with end_date, fetches that range from Groww
        end_date:   optional date — required if start_date is provided

    Returns UTC-indexed DataFrame.
    """
    try:
        g = _groww()
        if start_date is not None and end_date is not None:
            from datetime import datetime as dt
            from zoneinfo import ZoneInfo
            ist = ZoneInfo("Asia/Kolkata")
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
        logger.debug(f"[{symbol}] Groww fetch error: {e}")
    logger.warning(f"[{symbol}] No data from Groww")
    return None


def fetch_data(symbol: str) -> pd.DataFrame | None:
    """Fetch live intraday candles for the trading bot (IST-indexed).

    Tries today's live session first, then falls back to multi-day history.
    Returns None if Groww is unreachable or the symbol has no data.
    """
    try:
        g = _groww()

        # Today's live session
        df = g.fetch_intraday_candles(symbol, interval_minutes=5)
        df = _validate(df, symbol, min_rows=5)
        if df is not None:
            logger.debug(f"[{symbol}] Groww live: {len(df)} candles")
            return df

        # Fall back to multi-day historical
        df = g.fetch_multi_day_candles(symbol, days=5, interval_minutes=5)
        df = _validate(df, symbol, min_rows=10)
        if df is not None:
            logger.debug(f"[{symbol}] Groww historical: {len(df)} candles")
            return df

        logger.warning(f"[{symbol}] No data from Groww")
        return None
    except RuntimeError as e:
        logger.error(f"[{symbol}] Groww unavailable: {e}")
        return None
    except Exception as e:
        logger.warning(f"[{symbol}] fetch_data error: {e}")
        return None
