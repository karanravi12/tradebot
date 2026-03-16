"""Supertrend indicator — ATR-based trend following, popular in Indian markets."""

import numpy as np
import pandas as pd
import config


def _calculate_supertrend(
    df: pd.DataFrame, period: int, multiplier: float
) -> pd.DataFrame:
    """Calculate Supertrend values and direction.

    Returns DataFrame with columns: supertrend, direction (1=up, -1=down)
    """
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    n = len(df)

    # ATR calculation
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    tr[0] = high[0] - low[0]

    atr = np.zeros(n)
    atr[:period] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    # Basic upper and lower bands
    hl2 = (high + low) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    # Final bands with carry-forward logic
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish (up), -1 = bearish (down)

    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    supertrend[0] = basic_upper[0]
    direction[0] = 1

    for i in range(1, n):
        # Final upper band
        if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Final lower band
        if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction and supertrend value
        if direction[i - 1] == 1:  # was bullish
            if close[i] < final_lower[i]:
                direction[i] = -1
                supertrend[i] = final_upper[i]
            else:
                direction[i] = 1
                supertrend[i] = final_lower[i]
        else:  # was bearish
            if close[i] > final_upper[i]:
                direction[i] = 1
                supertrend[i] = final_lower[i]
            else:
                direction[i] = -1
                supertrend[i] = final_upper[i]

    return pd.DataFrame(
        {"supertrend": supertrend, "direction": direction},
        index=df.index,
    )


def analyze(df: pd.DataFrame) -> dict:
    """Analyze using Supertrend indicator.

    Returns:
        dict with keys: score (-3 to +3), confidence (0-1), details (dict)
    """
    score = 0
    details = {}

    st = _calculate_supertrend(df, config.SUPERTREND_PERIOD, config.SUPERTREND_MULTIPLIER)

    current_dir = st["direction"].iloc[-1]
    prev_dir = st["direction"].iloc[-2] if len(st) >= 2 else current_dir
    current_st = st["supertrend"].iloc[-1]
    current_price = df["Close"].iloc[-1]

    details["supertrend_value"] = round(current_st, 2)
    details["direction"] = "bullish" if current_dir == 1 else "bearish"

    # Direction-based signal
    if current_dir == 1:
        score += 1
    else:
        score -= 1

    # Supertrend flip (direction change) — stronger signal
    if prev_dir == -1 and current_dir == 1:
        score += 1  # flip to bullish (+2 total)
        details["flip"] = "bearish_to_bullish"
    elif prev_dir == 1 and current_dir == -1:
        score -= 1  # flip to bearish (-2 total)
        details["flip"] = "bullish_to_bearish"
    else:
        details["flip"] = None

    # Price distance from supertrend (momentum strength)
    if current_st > 0:
        distance_pct = (current_price - current_st) / current_st
        details["distance_pct"] = round(distance_pct * 100, 2)

    score = max(-3, min(3, score))
    confidence = min(abs(score) / 3.0, 1.0)

    return {"score": score, "confidence": confidence, "details": details}
