"""Classic technical indicators: RSI, MACD, EMA crossover."""

import pandas as pd
import ta
import config


def analyze(df: pd.DataFrame) -> dict:
    """Run RSI, MACD, and EMA crossover on OHLCV data.

    Returns:
        dict with keys: score (-3 to +3), confidence (0-1), details (dict)
    """
    score = 0
    details = {}

    close = df["Close"]

    # ── RSI ───────────────────────────────────────────────────────────────
    rsi = ta.momentum.RSIIndicator(close, window=config.RSI_PERIOD).rsi()
    current_rsi = rsi.iloc[-1]
    details["rsi"] = round(current_rsi, 2)

    if current_rsi < config.RSI_OVERSOLD:
        score += 1
        details["rsi_signal"] = "oversold"
    elif current_rsi > config.RSI_OVERBOUGHT:
        score -= 1
        details["rsi_signal"] = "overbought"
    else:
        details["rsi_signal"] = "neutral"

    # ── MACD ──────────────────────────────────────────────────────────────
    macd_ind = ta.trend.MACD(
        close,
        window_slow=config.MACD_SLOW,
        window_fast=config.MACD_FAST,
        window_sign=config.MACD_SIGNAL,
    )
    macd_line = macd_ind.macd()
    signal_line = macd_ind.macd_signal()

    if len(macd_line) >= 2 and len(signal_line) >= 2:
        prev_macd = macd_line.iloc[-2]
        curr_macd = macd_line.iloc[-1]
        prev_signal = signal_line.iloc[-2]
        curr_signal = signal_line.iloc[-1]

        # Bullish crossover
        if prev_macd <= prev_signal and curr_macd > curr_signal:
            score += 1
            details["macd_signal"] = "bullish_crossover"
        # Bearish crossover
        elif prev_macd >= prev_signal and curr_macd < curr_signal:
            score -= 1
            details["macd_signal"] = "bearish_crossover"
        else:
            details["macd_signal"] = "neutral"

        details["macd"] = round(curr_macd, 4)
        details["macd_signal_line"] = round(curr_signal, 4)

    # ── EMA Crossover ─────────────────────────────────────────────────────
    ema_short = close.ewm(span=config.EMA_SHORT, adjust=False).mean()
    ema_long = close.ewm(span=config.EMA_LONG, adjust=False).mean()

    details["ema_short"] = round(ema_short.iloc[-1], 2)
    details["ema_long"] = round(ema_long.iloc[-1], 2)

    if len(ema_short) >= 2 and len(ema_long) >= 2:
        prev_short = ema_short.iloc[-2]
        curr_short = ema_short.iloc[-1]
        prev_long = ema_long.iloc[-2]
        curr_long = ema_long.iloc[-1]

        # Bullish crossover
        if prev_short <= prev_long and curr_short > curr_long:
            score += 1
            details["ema_signal"] = "bullish_crossover"
        elif prev_short >= prev_long and curr_short < curr_long:
            score -= 1
            details["ema_signal"] = "bearish_crossover"
        elif curr_short > curr_long:
            details["ema_signal"] = "bullish"
        else:
            details["ema_signal"] = "bearish"

    # Clamp score
    score = max(-3, min(3, score))

    # Confidence based on how many indicators agree
    abs_score = abs(score)
    confidence = abs_score / 3.0

    return {"score": score, "confidence": confidence, "details": details}
