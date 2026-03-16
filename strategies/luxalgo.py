"""LuxAlgo Smart Money Concepts (SMC) — institutional price action analysis.

Implements the core building blocks of Smart Money Concepts as popularised
by LuxAlgo's indicators:

  1. Swing Highs / Lows — pivot points used as structural reference
  2. Break of Structure (BOS) — price breaks a prior swing high/low (trend continuation)
  3. Change of Character (CHoCH) — trend reversal signal
  4. Fair Value Gaps (FVG) — candle imbalances; price tends to fill them
  5. Order Blocks (OB) — last opposing candle before an impulsive move; often retested
  6. Premium / Discount / Equilibrium — price relative to current swing range

Score  Meaning
─────  ──────────────────────────────────────────────────────────────────
 +4    Bullish BOS + price at bullish OB + discount zone + FVG support below
 +3    Bullish structure + discount zone + bullish OB confluence
 +2    Bullish structure (HH+HL) + one of: OB touched / FVG below / discount
 +1    Bullish structure only (equilibrium zone, no extra confluence)
  0    Neutral — insufficient data or no clear structure
 -1    Bearish structure only (equilibrium zone, no extra confluence)
 -2    Bearish structure + one of: OB touched / FVG above / premium
 -3    Bearish structure + premium zone + bearish OB confluence
 -4    Bearish BOS + price at bearish OB + premium zone + FVG resistance above
"""

import pandas as pd

# ── Parameters ────────────────────────────────────────────────────────────────
SWING_N     = 5      # pivot lookback (N bars on each side)
IMPULSE_PCT = 0.015  # 1.5% minimum move to classify as impulsive
FVG_MIN_PCT = 0.003  # 0.3% minimum gap to call an FVG
OB_LOOKBACK = 40     # bars to look back for order blocks
MIN_BARS    = 30     # minimum candles needed for any analysis


# ── 1. Swing High / Low Detection ─────────────────────────────────────────────

def _find_swings(df: pd.DataFrame, n: int = SWING_N) -> tuple[list, list]:
    """Return (swing_highs, swing_lows) as lists of (bar_index, price) tuples.

    A bar is a swing high if its High is strictly >= all surrounding N bars
    on both sides.  Lows are the mirror image.
    """
    highs, lows = [], []
    for i in range(n, len(df) - n):
        h = float(df["High"].iloc[i])
        lo = float(df["Low"].iloc[i])

        neighbors_h = [float(df["High"].iloc[i + j]) for j in range(-n, n + 1) if j != 0]
        neighbors_l = [float(df["Low"].iloc[i + j])  for j in range(-n, n + 1) if j != 0]

        if h >= max(neighbors_h):
            highs.append((i, h))
        if lo <= min(neighbors_l):
            lows.append((i, lo))

    return highs, lows


# ── 2. Market Structure (BOS / CHoCH) ─────────────────────────────────────────

def _detect_structure(swing_highs: list, swing_lows: list) -> dict:
    """Detect Break of Structure and Change of Character from swing pivots.

    Logic (simplified but faithful to real SMC):
      - Bullish BOS:  latest swing high > prior swing high (HH) AND
                      latest swing low  > prior swing low  (HL)  → strong bull
      - Bearish BOS:  latest swing high < prior swing high (LH) AND
                      latest swing low  < prior swing low  (LL)  → strong bear
      - Partial bull: HH only or HL only  → structure_score = +1
      - Partial bear: LH only or LL only  → structure_score = -1
      - CHoCH:        after 2+ consecutive HH moves, a LL appears → reversal
    """
    out = {
        "trend":           "neutral",
        "last_bos":        None,
        "last_choch":      None,
        "structure_score": 0,
    }
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return out

    sh = sorted(swing_highs, key=lambda x: x[0])
    sl = sorted(swing_lows,  key=lambda x: x[0])

    hh = sh[-1][1] > sh[-2][1]   # Higher High
    lh = sh[-1][1] < sh[-2][1]   # Lower High
    hl = sl[-1][1] > sl[-2][1]   # Higher Low
    ll = sl[-1][1] < sl[-2][1]   # Lower Low

    if hh and hl:
        out["trend"] = "bullish"
        out["structure_score"] = 2
        out["last_bos"] = "bullish"
    elif hh:
        out["trend"] = "bullish"
        out["structure_score"] = 1
        out["last_bos"] = "bullish"
    elif hl:
        out["trend"] = "bullish"
        out["structure_score"] = 1
    elif lh and ll:
        out["trend"] = "bearish"
        out["structure_score"] = -2
        out["last_bos"] = "bearish"
    elif lh:
        out["trend"] = "bearish"
        out["structure_score"] = -1
        out["last_bos"] = "bearish"
    elif ll:
        out["trend"] = "bearish"
        out["structure_score"] = -1

    # CHoCH: was it trending one way then suddenly flipped?
    if len(sh) >= 3 and len(sl) >= 3:
        was_bull = sh[-2][1] > sh[-3][1]  # prior two highs were HH
        was_bear = sh[-2][1] < sh[-3][1]  # prior two highs were LH

        if was_bull and ll:
            out["last_choch"] = "bearish"
            # CHoCH overrides BOS — it's a reversal signal
            out["structure_score"] = min(out["structure_score"] - 1, -1)
        elif was_bear and hh:
            out["last_choch"] = "bullish"
            out["structure_score"] = max(out["structure_score"] + 1, 1)

    return out


# ── 3. Fair Value Gaps ─────────────────────────────────────────────────────────

def _find_fvgs(df: pd.DataFrame, min_gap_pct: float = FVG_MIN_PCT) -> dict:
    """Detect Fair Value Gaps (price imbalances between three consecutive candles).

    Bullish FVG:  candle[i-1].High < candle[i+1].Low  → gap price must fill on retest
    Bearish FVG:  candle[i-1].Low  > candle[i+1].High → resistance gap above
    """
    bull_fvgs = []
    bear_fvgs = []

    for i in range(1, len(df) - 1):
        high_prev = float(df["High"].iloc[i - 1])
        low_prev  = float(df["Low"].iloc[i - 1])
        low_next  = float(df["Low"].iloc[i + 1])
        high_next = float(df["High"].iloc[i + 1])
        mid_close = float(df["Close"].iloc[i])

        if mid_close <= 0:
            continue

        if low_next > high_prev:
            gap_pct = (low_next - high_prev) / mid_close
            if gap_pct >= min_gap_pct:
                bull_fvgs.append({
                    "top":     round(low_next, 2),
                    "bottom":  round(high_prev, 2),
                    "bar_idx": i,
                    "gap_pct": round(gap_pct * 100, 3),
                })

        elif high_next < low_prev:
            gap_pct = (low_prev - high_next) / mid_close
            if gap_pct >= min_gap_pct:
                bear_fvgs.append({
                    "top":     round(low_prev, 2),
                    "bottom":  round(high_next, 2),
                    "bar_idx": i,
                    "gap_pct": round(gap_pct * 100, 3),
                })

    return {
        "bullish_fvgs": bull_fvgs[-3:],
        "bearish_fvgs": bear_fvgs[-3:],
    }


# ── 4. Order Blocks ────────────────────────────────────────────────────────────

def _find_order_blocks(df: pd.DataFrame, lookback: int = OB_LOOKBACK) -> dict:
    """Identify Order Blocks — the last opposing candle before an impulsive leg.

    Bullish OB:  last bearish candle before a 1.5%+ bullish impulse
    Bearish OB:  last bullish candle before a 1.5%+ bearish impulse

    Price returning to an OB is an institutional re-entry opportunity.
    """
    bull_obs = []
    bear_obs = []

    start = max(0, len(df) - lookback - 2)
    for i in range(start, len(df) - 1):
        o = float(df["Open"].iloc[i])
        c = float(df["Close"].iloc[i])
        h = float(df["High"].iloc[i])
        lo = float(df["Low"].iloc[i])
        nc = float(df["Close"].iloc[i + 1])

        if c <= 0:
            continue

        # Bullish OB: bearish candle followed by bullish impulse
        if c < o:
            impulse_up = (nc - c) / c
            if impulse_up >= IMPULSE_PCT:
                bull_obs.append({
                    "top":      round(h, 2),
                    "bottom":   round(lo, 2),
                    "bar_idx":  i,
                    "strength": round(impulse_up * 100, 2),
                })

        # Bearish OB: bullish candle followed by bearish impulse
        elif c > o:
            impulse_dn = (c - nc) / c
            if impulse_dn >= IMPULSE_PCT:
                bear_obs.append({
                    "top":      round(h, 2),
                    "bottom":   round(lo, 2),
                    "bar_idx":  i,
                    "strength": round(impulse_dn * 100, 2),
                })

    return {
        "bullish_obs": bull_obs[-3:],
        "bearish_obs": bear_obs[-3:],
    }


# ── 5. Premium / Discount Zones ────────────────────────────────────────────────

def _premium_discount(price: float, swing_high: float, swing_low: float) -> str:
    """Classify price relative to the current swing range using Fibonacci levels.

    Discount  < 38.2% of range → buy zone (cheap relative to range)
    Equilibrium 38.2%–61.8%   → neutral
    Premium   > 61.8% of range → sell zone (expensive relative to range)
    """
    rng = swing_high - swing_low
    if rng <= 0:
        return "equilibrium"
    pos = (price - swing_low) / rng
    if pos > 0.618:
        return "premium"
    if pos < 0.382:
        return "discount"
    return "equilibrium"


# ── Public API ─────────────────────────────────────────────────────────────────

def analyze(df: pd.DataFrame) -> dict:
    """Full LuxAlgo Smart Money Concepts analysis on a candle DataFrame.

    Score  Meaning
    ─────  ──────────────────────────────────────────────────────────────────
     +4    Bullish BOS + bullish OB + discount zone + FVG support below
     +3    Bullish BOS + two of three confluence factors
     +2    Bullish structure + one confluence factor
     +1    Bullish structure only
      0    Neutral / insufficient data
     -1    Bearish structure only
     -2    Bearish structure + one confluence factor
     -3    Bearish BOS + two of three confluence factors
     -4    Bearish BOS + bearish OB + premium zone + FVG resistance above

    Args:
        df: OHLCV DataFrame, newest last.  Must have ≥ MIN_BARS rows.

    Returns:
        {
            "score":      int (-4 to +4),
            "confidence": float (0–1),
            "smc_bias":   "bullish" | "bearish" | "neutral",
            "details":    dict (structure, fvgs, obs, zones, swing levels)
        }
    """
    if len(df) < MIN_BARS:
        return {
            "score": 0, "confidence": 0.0, "smc_bias": "neutral",
            "details": {"signal": "insufficient_data"},
        }

    current_price = float(df["Close"].iloc[-1])

    swing_highs, swing_lows = _find_swings(df)
    structure = _detect_structure(swing_highs, swing_lows)
    fvg_data  = _find_fvgs(df)
    ob_data   = _find_order_blocks(df)

    # Reference swing levels for zone classification
    last_sh = swing_highs[-1][1] if swing_highs else current_price * 1.02
    last_sl = swing_lows[-1][1]  if swing_lows  else current_price * 0.98
    pd_zone = _premium_discount(current_price, last_sh, last_sl)

    # OB touches
    at_bull_ob = any(
        ob["bottom"] <= current_price <= ob["top"]
        for ob in ob_data["bullish_obs"]
    )
    at_bear_ob = any(
        ob["bottom"] <= current_price <= ob["top"]
        for ob in ob_data["bearish_obs"]
    )

    # FVG proximity
    fvg_below = any(f["top"] < current_price for f in fvg_data["bullish_fvgs"])
    fvg_above = any(f["bottom"] > current_price for f in fvg_data["bearish_fvgs"])

    # ── Score: structure base (-2..+2) + confluence bonuses ───────────────────
    score = structure["structure_score"]

    if score > 0:
        if pd_zone == "discount":
            score += 1
        if at_bull_ob:
            score += 1
        if fvg_below:
            score += 1
    elif score < 0:
        if pd_zone == "premium":
            score -= 1
        if at_bear_ob:
            score -= 1
        if fvg_above:
            score -= 1

    score = max(-4, min(4, score))
    confidence = round(min(abs(score) / 4.0, 1.0), 3)
    smc_bias = "bullish" if score > 0 else ("bearish" if score < 0 else "neutral")

    details = {
        "trend":           structure["trend"],
        "last_bos":        structure["last_bos"],
        "last_choch":      structure["last_choch"],
        "structure_score": structure["structure_score"],
        "pd_zone":         pd_zone,
        "at_bullish_ob":   at_bull_ob,
        "at_bearish_ob":   at_bear_ob,
        "fvg_below":       fvg_below,
        "fvg_above":       fvg_above,
        "swing_high":      round(last_sh, 2),
        "swing_low":       round(last_sl, 2),
        "bullish_fvgs":    fvg_data["bullish_fvgs"],
        "bearish_fvgs":    fvg_data["bearish_fvgs"],
        "bullish_obs":     ob_data["bullish_obs"][-1:],
        "bearish_obs":     ob_data["bearish_obs"][-1:],
    }

    return {
        "score":      score,
        "confidence": confidence,
        "smc_bias":   smc_bias,
        "details":    details,
    }
