"""Auto Fibonacci retracement with swing detection."""

import numpy as np
import pandas as pd
import config


def _find_swings(df: pd.DataFrame, lookback: int) -> tuple[list, list]:
    """Find swing highs and swing lows for Fibonacci calculation."""
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)

    swing_highs = []  # (index, price)
    swing_lows = []

    for i in range(lookback, n - lookback):
        if highs[i] == max(highs[i - lookback : i + lookback + 1]):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(lows[i - lookback : i + lookback + 1]):
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


def _calculate_fib_levels(
    swing_high: float, swing_low: float, direction: str
) -> dict:
    """Calculate Fibonacci retracement levels.

    direction='up': retracement of an upswing (high to low retrace)
    direction='down': retracement of a downswing (low to high retrace)
    """
    diff = swing_high - swing_low
    levels = {}

    if direction == "up":
        # Retracement levels from the high
        for fib in config.FIB_LEVELS:
            levels[fib] = swing_high - diff * fib
    else:
        # Retracement levels from the low
        for fib in config.FIB_LEVELS:
            levels[fib] = swing_low + diff * fib

    return levels


def analyze(df: pd.DataFrame) -> dict:
    """Analyze using auto Fibonacci retracements.

    Returns:
        dict with keys: score (-3 to +3), confidence (0-1), details (dict)
    """
    score = 0
    details = {}

    lookback = config.SWING_LOOKBACK
    swing_highs, swing_lows = _find_swings(df, lookback)

    current_price = df["Close"].iloc[-1]
    prev_price = df["Close"].iloc[-2] if len(df) >= 2 else current_price

    if not swing_highs or not swing_lows:
        details["status"] = "insufficient_swings"
        return {"score": 0, "confidence": 0, "details": details}

    # Find the most recent significant swing high and low
    last_sh = swing_highs[-1]
    last_sl = swing_lows[-1]

    # Determine trend direction: which came last?
    if last_sh[0] > last_sl[0]:
        direction = "up"  # uptrend — retracing from high
        swing_high = last_sh[1]
        swing_low = last_sl[1]
    else:
        direction = "down"  # downtrend — retracing from low
        swing_high = last_sh[1]
        swing_low = last_sl[1]

    details["trend"] = direction
    details["swing_high"] = round(swing_high, 2)
    details["swing_low"] = round(swing_low, 2)

    # Calculate Fibonacci levels
    fib_levels = _calculate_fib_levels(swing_high, swing_low, direction)
    details["fib_levels"] = {k: round(v, 2) for k, v in fib_levels.items()}

    # ── Check price at key Fibonacci levels ───────────────────────────────
    tolerance = 0.005  # 0.5% tolerance around fib level

    for fib, level in fib_levels.items():
        proximity = abs(current_price - level) / level if level > 0 else 1

        if proximity <= tolerance:
            details[f"at_fib_{fib}"] = True

            if fib in (0.5, 0.618):
                # Key retracement levels
                if direction == "up":
                    # Price bouncing at 50%/61.8% in uptrend → buy
                    if current_price > prev_price:
                        score += 1
                        details["fib_signal"] = f"bounce_at_{fib}_uptrend"
                else:
                    # Price rejecting at 50%/61.8% in downtrend → sell
                    if current_price < prev_price:
                        score -= 1
                        details["fib_signal"] = f"rejection_at_{fib}_downtrend"

    # ── Golden Zone (0.618 – 0.786) ───────────────────────────────────────
    golden_top = fib_levels.get(0.618, 0)
    golden_bottom = fib_levels.get(0.786, 0)

    if golden_top and golden_bottom:
        # Normalize zone boundaries
        zone_high = max(golden_top, golden_bottom)
        zone_low = min(golden_top, golden_bottom)

        if zone_low <= current_price <= zone_high:
            details["in_golden_zone"] = True

            if direction == "up" and current_price > prev_price:
                score += 2
                details["golden_zone_signal"] = "bullish_reversal"
            elif direction == "down" and current_price < prev_price:
                score -= 2
                details["golden_zone_signal"] = "bearish_reversal"
        else:
            details["in_golden_zone"] = False

    # ── Extension check (price beyond swing high/low) ─────────────────────
    if direction == "up" and current_price > swing_high:
        details["fib_extension"] = True
        details["extension_signal"] = "new_high"
    elif direction == "down" and current_price < swing_low:
        details["fib_extension"] = True
        details["extension_signal"] = "new_low"

    score = max(-3, min(3, score))
    confidence = min(abs(score) / 3.0, 1.0)

    return {"score": score, "confidence": confidence, "details": details}
