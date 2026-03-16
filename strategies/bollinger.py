"""Bollinger Bands squeeze and breakout detection with Keltner Channel overlay."""

import numpy as np
import pandas as pd
import ta
import config


def _keltner_channels(df: pd.DataFrame, period: int, multiplier: float) -> dict:
    """Calculate Keltner Channel upper and lower bands."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # EMA of close
    mid = close.ewm(span=period, adjust=False).mean()

    # ATR
    tr = pd.DataFrame({
        "hl": high - low,
        "hc": (high - close.shift(1)).abs(),
        "lc": (low - close.shift(1)).abs(),
    }).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()

    return {
        "upper": mid + multiplier * atr,
        "lower": mid - multiplier * atr,
        "mid": mid,
    }


def analyze(df: pd.DataFrame) -> dict:
    """Analyze using Bollinger Bands squeeze + breakout.

    Returns:
        dict with keys: score (-3 to +3), confidence (0-1), details (dict)
    """
    score = 0
    details = {}
    close = df["Close"]
    current_price = close.iloc[-1]

    # ── Bollinger Bands ───────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(
        close, window=config.BB_PERIOD, window_dev=config.BB_STD
    )
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_mid = bb.bollinger_mavg()
    bb_width = bb.bollinger_wband()

    details["bb_upper"] = round(bb_upper.iloc[-1], 2)
    details["bb_lower"] = round(bb_lower.iloc[-1], 2)
    details["bb_mid"] = round(bb_mid.iloc[-1], 2)

    # ── Keltner Channels ──────────────────────────────────────────────────
    kc = _keltner_channels(df, config.KC_PERIOD, config.KC_MULTIPLIER)

    # ── Squeeze Detection ─────────────────────────────────────────────────
    # Squeeze is ON when BB is inside KC
    squeeze_on = (
        bb_lower.iloc[-1] > kc["lower"].iloc[-1]
        and bb_upper.iloc[-1] < kc["upper"].iloc[-1]
    )
    # Was squeeze on previously?
    prev_squeeze = False
    if len(bb_lower) >= 2:
        prev_squeeze = (
            bb_lower.iloc[-2] > kc["lower"].iloc[-2]
            and bb_upper.iloc[-2] < kc["upper"].iloc[-2]
        )

    details["squeeze"] = squeeze_on

    # Squeeze release — volatility breakout imminent
    if prev_squeeze and not squeeze_on:
        details["squeeze_release"] = True
        # Direction of breakout
        if current_price > bb_upper.iloc[-1]:
            score += 2
            details["breakout"] = "bullish"
        elif current_price < bb_lower.iloc[-1]:
            score -= 2
            details["breakout"] = "bearish"
        else:
            # Squeeze released but no clear breakout yet
            details["breakout"] = "pending"
    else:
        details["squeeze_release"] = False

    # ── Mean Reversion Signals ────────────────────────────────────────────
    rsi = ta.momentum.RSIIndicator(close, window=config.RSI_PERIOD).rsi()
    current_rsi = rsi.iloc[-1]

    if current_price <= bb_lower.iloc[-1] and current_rsi < 35:
        score += 1
        details["mean_reversion"] = "oversold_at_lower_band"
    elif current_price >= bb_upper.iloc[-1] and current_rsi > 65:
        score -= 1
        details["mean_reversion"] = "overbought_at_upper_band"
    else:
        details["mean_reversion"] = "neutral"

    # ── Band Width (volatility measure) ───────────────────────────────────
    if len(bb_width) >= 20:
        avg_width = bb_width.iloc[-20:].mean()
        current_width = bb_width.iloc[-1]
        details["bb_width_vs_avg"] = round(current_width / avg_width, 2) if avg_width > 0 else 1.0

    score = max(-3, min(3, score))
    confidence = min(abs(score) / 3.0, 1.0)

    return {"score": score, "confidence": confidence, "details": details}
