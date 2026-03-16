"""Smart Money Concepts (LuxAlgo-style): Order Blocks, BOS, CHoCH, FVG, Liquidity Sweeps."""

import numpy as np
import pandas as pd
import config


def _find_swing_points(df: pd.DataFrame, lookback: int) -> tuple[list, list]:
    """Detect swing highs and swing lows using a rolling window."""
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)

    swing_highs = []  # (index, price)
    swing_lows = []   # (index, price)

    for i in range(lookback, n - lookback):
        # Swing high: highest point in the window
        if highs[i] == max(highs[i - lookback : i + lookback + 1]):
            swing_highs.append((i, highs[i]))
        # Swing low: lowest point in the window
        if lows[i] == min(lows[i - lookback : i + lookback + 1]):
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


def _detect_order_blocks(df: pd.DataFrame) -> dict:
    """Detect bullish and bearish order blocks.

    Bullish OB: last bearish candle before a strong bullish move.
    Bearish OB: last bullish candle before a strong bearish move.
    """
    opens = df["Open"].values
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)
    threshold = config.SMC_IMPULSE_THRESHOLD

    bullish_obs = []
    bearish_obs = []

    for i in range(1, n - 1):
        body_pct = abs(closes[i + 0] - opens[i + 0]) / opens[i + 0] if opens[i + 0] > 0 else 0

        # Check for impulsive move after candle i
        if i + 1 < n:
            move_pct = (closes[i + 1] - closes[i]) / closes[i] if closes[i] > 0 else 0

            # Bullish OB: bearish candle (close < open) followed by strong bullish move
            if closes[i] < opens[i] and move_pct > threshold:
                bullish_obs.append({
                    "index": i,
                    "top": opens[i],
                    "bottom": closes[i],
                    "high": highs[i],
                    "low": lows[i],
                })

            # Bearish OB: bullish candle (close > open) followed by strong bearish move
            if closes[i] > opens[i] and move_pct < -threshold:
                bearish_obs.append({
                    "index": i,
                    "top": closes[i],
                    "bottom": opens[i],
                    "high": highs[i],
                    "low": lows[i],
                })

    return {"bullish": bullish_obs, "bearish": bearish_obs}


def _detect_bos_choch(swing_highs: list, swing_lows: list, closes: np.ndarray) -> dict:
    """Detect Break of Structure (BOS) and Change of Character (CHoCH).

    BOS: trend continuation — price breaks past a swing point in the same direction.
    CHoCH: trend reversal — price breaks a swing point in the opposite direction.
    """
    result = {"bos": None, "choch": None}

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return result

    current_price = closes[-1]
    prev_price = closes[-2] if len(closes) >= 2 else current_price

    # Determine recent trend by comparing last two swing highs and lows
    last_sh = swing_highs[-1]
    prev_sh = swing_highs[-2] if len(swing_highs) >= 2 else last_sh
    last_sl = swing_lows[-1]
    prev_sl = swing_lows[-2] if len(swing_lows) >= 2 else last_sl

    # Uptrend: higher highs and higher lows
    uptrend = last_sh[1] > prev_sh[1] and last_sl[1] > prev_sl[1]
    # Downtrend: lower highs and lower lows
    downtrend = last_sh[1] < prev_sh[1] and last_sl[1] < prev_sl[1]

    # BOS: break in the direction of the trend
    if uptrend and current_price > last_sh[1] and prev_price <= last_sh[1]:
        result["bos"] = "bullish"
    elif downtrend and current_price < last_sl[1] and prev_price >= last_sl[1]:
        result["bos"] = "bearish"

    # CHoCH: break against the trend (reversal)
    if uptrend and current_price < last_sl[1] and prev_price >= last_sl[1]:
        result["choch"] = "bearish"  # was uptrend, now breaking down
    elif downtrend and current_price > last_sh[1] and prev_price <= last_sh[1]:
        result["choch"] = "bullish"  # was downtrend, now breaking up

    return result


def _detect_fvg(df: pd.DataFrame) -> dict:
    """Detect Fair Value Gaps (3-candle imbalance pattern)."""
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)
    min_gap = config.FVG_MIN_GAP_PCT

    bullish_fvgs = []
    bearish_fvgs = []

    for i in range(2, n):
        # Bullish FVG: candle 1 high < candle 3 low (gap up)
        gap = (lows[i] - highs[i - 2])
        mid_price = (highs[i - 2] + lows[i]) / 2 if (highs[i - 2] + lows[i]) > 0 else 1
        if gap > 0 and (gap / mid_price) > min_gap:
            bullish_fvgs.append({
                "index": i,
                "top": lows[i],
                "bottom": highs[i - 2],
            })

        # Bearish FVG: candle 1 low > candle 3 high (gap down)
        gap = (lows[i - 2] - highs[i])
        mid_price = (lows[i - 2] + highs[i]) / 2 if (lows[i - 2] + highs[i]) > 0 else 1
        if gap > 0 and (gap / mid_price) > min_gap:
            bearish_fvgs.append({
                "index": i,
                "top": lows[i - 2],
                "bottom": highs[i],
            })

    return {"bullish": bullish_fvgs, "bearish": bearish_fvgs}


def _detect_liquidity_sweep(
    swing_highs: list, swing_lows: list, df: pd.DataFrame
) -> str | None:
    """Detect liquidity sweeps (stop hunts).

    Price spikes beyond a swing high/low then reverses sharply in the same candle
    or the next candle.
    """
    if len(swing_highs) < 1 or len(swing_lows) < 1:
        return None

    n = len(df)
    if n < 3:
        return None

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    opens = df["Open"].values

    last_sh_price = swing_highs[-1][1]
    last_sl_price = swing_lows[-1][1]

    # Check last 2 candles for sweep
    for i in range(max(0, n - 2), n):
        # Bearish sweep: wick above swing high but closes below it
        if highs[i] > last_sh_price and closes[i] < last_sh_price:
            return "bearish_sweep"

        # Bullish sweep: wick below swing low but closes above it
        if lows[i] < last_sl_price and closes[i] > last_sl_price:
            return "bullish_sweep"

    return None


def analyze(df: pd.DataFrame) -> dict:
    """Run full Smart Money Concepts analysis.

    Returns:
        dict with keys: score (-3 to +3), confidence (0-1), details (dict)
    """
    score = 0
    details = {}
    lookback = config.SMC_SWING_LOOKBACK

    swing_highs, swing_lows = _find_swing_points(df, lookback)
    closes = df["Close"].values
    current_price = closes[-1]

    # ── Order Blocks ──────────────────────────────────────────────────────
    obs = _detect_order_blocks(df)
    details["order_blocks"] = {"bullish_count": len(obs["bullish"]), "bearish_count": len(obs["bearish"])}

    # Check if price is retesting a recent order block
    if obs["bullish"]:
        last_bull_ob = obs["bullish"][-1]
        if last_bull_ob["bottom"] <= current_price <= last_bull_ob["top"]:
            score += 1
            details["ob_signal"] = "price_at_demand_zone"
    if obs["bearish"]:
        last_bear_ob = obs["bearish"][-1]
        if last_bear_ob["bottom"] <= current_price <= last_bear_ob["top"]:
            score -= 1
            details["ob_signal"] = details.get("ob_signal", "") + "price_at_supply_zone"

    # ── BOS / CHoCH ───────────────────────────────────────────────────────
    structure = _detect_bos_choch(swing_highs, swing_lows, closes)
    details["bos"] = structure["bos"]
    details["choch"] = structure["choch"]

    if structure["bos"] == "bullish":
        score += 1
    elif structure["bos"] == "bearish":
        score -= 1

    if structure["choch"] == "bullish":
        score += 2  # CHoCH is a strong reversal signal
    elif structure["choch"] == "bearish":
        score -= 2

    # ── Fair Value Gaps ───────────────────────────────────────────────────
    fvgs = _detect_fvg(df)
    details["fvg"] = {"bullish_count": len(fvgs["bullish"]), "bearish_count": len(fvgs["bearish"])}

    # Check if price is filling a recent FVG
    if fvgs["bullish"]:
        last_fvg = fvgs["bullish"][-1]
        if last_fvg["bottom"] <= current_price <= last_fvg["top"]:
            score += 1
            details["fvg_signal"] = "filling_bullish_fvg"
    if fvgs["bearish"]:
        last_fvg = fvgs["bearish"][-1]
        if last_fvg["bottom"] <= current_price <= last_fvg["top"]:
            score -= 1
            details["fvg_signal"] = details.get("fvg_signal", "") + "filling_bearish_fvg"

    # ── Liquidity Sweep ───────────────────────────────────────────────────
    sweep = _detect_liquidity_sweep(swing_highs, swing_lows, df)
    details["liquidity_sweep"] = sweep

    if sweep == "bullish_sweep":
        score += 2
    elif sweep == "bearish_sweep":
        score -= 2

    # Clamp score
    score = max(-3, min(3, score))
    confidence = min(abs(score) / 3.0, 1.0)

    return {"score": score, "confidence": confidence, "details": details}
