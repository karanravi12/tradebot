"""Market data fetcher — Yahoo Finance primary for backtest, Groww for live.

Live bot (fetch_data):   Groww only — real-time IST candles.
Backtest (fetch_intraday): Yahoo Finance first (up to 60 days of 5-min history
                           for NSE stocks); falls back to Groww (~6 days) if
                           Yahoo has no data for the symbol.
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


def _yfinance_intraday(symbol: str) -> pd.DataFrame | None:
    """Fetch up to 60 days of 5-min candles from Yahoo Finance (.NS suffix).

    Returns UTC-indexed OHLCV DataFrame, or None on failure.
    Only used as a backtest fallback — never called by the live bot.
    """
    try:
        import yfinance as yf

        # Some tickers need special mapping for Yahoo Finance
        _YF_OVERRIDES = {
            "M&M":        "M&M.NS",
            "M&MFIN":     "M&MFIN.NS",
            "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
            "MCDOWELL-N": "MCDOWELL-N.NS",
        }
        yf_ticker = _YF_OVERRIDES.get(symbol, f"{symbol}.NS")

        df = yf.download(
            yf_ticker,
            period="60d",
            interval="5m",
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty:
            return None

        # yfinance returns (Price, Ticker) MultiIndex — drop the ticker level
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.rename(columns={
            "Open": "Open", "High": "High", "Low": "Low",
            "Close": "Close", "Volume": "Volume",
        })
        df = _validate(df, symbol, min_rows=10)
        if df is None:
            return None

        # Ensure UTC index
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        # Keep only NSE market hours (9:15–15:30 IST = 3:45–10:00 UTC)
        ist_idx = df.index.tz_convert("Asia/Kolkata")
        mask = (
            (ist_idx.hour > 9) | ((ist_idx.hour == 9) & (ist_idx.minute >= 15))
        ) & (
            (ist_idx.hour < 15) | ((ist_idx.hour == 15) & (ist_idx.minute <= 30))
        )
        df = df[mask]

        logger.debug(f"[{symbol}] Yahoo Finance: {len(df)} candles")
        return df if len(df) >= 10 else None
    except Exception as e:
        logger.debug(f"[{symbol}] Yahoo Finance error: {e}")
        return None


def fetch_intraday(symbol: str) -> pd.DataFrame | None:
    """Fetch multi-day 5-min candles for backtest.

    Tries Yahoo Finance first (up to 60 days of 5-min history for NSE stocks).
    Falls back to Groww (recent ~6 days) if Yahoo has no data for the symbol.
    Returns UTC-indexed DataFrame.
    """
    # ── 1. Try Yahoo Finance (60-day history) ────────────────────────────
    df = _yfinance_intraday(symbol)
    if df is not None:
        return df

    # ── 2. Fall back to Groww (recent ~6 days) ───────────────────────────
    try:
        g = _groww()
        df = g.fetch_multi_day_candles(symbol, days=6, interval_minutes=5)
        df = _validate(df, symbol, min_rows=10)
        if df is not None:
            if df.index.tzinfo is not None:
                df.index = df.index.tz_convert("UTC")
            else:
                df.index = df.index.tz_localize("Asia/Kolkata").tz_convert("UTC")
            logger.debug(f"[{symbol}] Groww: {len(df)} candles")
            return df
    except Exception:
        pass

    logger.warning(f"[{symbol}] No data from Yahoo Finance or Groww")
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
