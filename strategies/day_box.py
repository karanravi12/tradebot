"""Darvas Box Theory — strict confirmation-based implementation with ATR and fee awareness.

Classical Darvas rules (from Nicolas Darvas's original method):
  1. Candidate TOP = a High that is a local new high (> prior 5 bars)
  2. Box TOP confirmed when the next CONFIRM_BARS (3) bars all have High < top
  3. Box BOTTOM = lowest Low during the CONFIRM_BARS consolidation window
  4. BUY  : Close breaks above box_top + BREAKOUT_BUF (0.2%) → real breakout
  5. SELL : Close breaks below box_bottom by more than ATR_BREAKDOWN_FACTOR × ATR
             (ATR buffer prevents selling on minor wicks/noise)
  6. Support bounce (bottom 20%) / resistance rejection (top 20%) = milder signals
  7. Volume > 1.5x average adds one grade of conviction

Modern improvements in this version:
  - ATR-buffered breakdown: price must close meaningfully below box_bottom
    (not just a wick dip). Eliminates whipsaw sells on minor noise.
  - New zone "below_box" (score -1): price is below box_bottom but hasn't cleared
    the ATR buffer yet — tentative, don't sell yet, wait for confirmation.
  - Whipsaw risk flag: low-volume breakdown = higher chance of false signal.
  - Fee viability: box_width_pct vs round-trip fee cost — is this box tradeable?
  - close_below_count: how many of the last 2 candles closed below box_bottom.
  - Staleness guard: boxes older than MAX_BOX_AGE bars are discarded (~3.5 h).
  - Tightness = box width < 2% of mid-price.
  - Local-high filter: candidate must beat the prior 5 bars to count.
"""

import numpy as np
import pandas as pd

# ── Parameters ────────────────────────────────────────────────────────────────
CONFIRM_BARS = 3      # consecutive bars below top needed to confirm it
MAX_BOX_AGE  = 40     # ignore boxes confirmed more than N candles ago (~3.3 h)
MIN_BARS     = 20     # minimum prior bars required to attempt box detection
LOCAL_HI_N   = 5      # candidate top must beat the prior N bars' highs

BREAKOUT_BUF        = 0.002   # 0.2% above box_top = confirmed breakout (not a fake tick)
ATR_BREAKDOWN_FACTOR = 0.30   # price must be box_bottom - 0.3×ATR below for real breakdown
ATR_PERIOD          = 14      # period for Average True Range calculation
ZONE_EDGE           = 0.20    # top/bottom 20% of box = trade zones
TIGHT_PCT           = 2.0     # box width < 2% of price = high-quality setup
VOL_THRESH          = 1.5     # current vol / avg vol threshold for "volume confirmed"

# ── Exported for use by trader.py and backtest.py ─────────────────────────────
# ATR_BREAKDOWN_FACTOR — how far below box_bottom counts as a real breakdown
# VOL_THRESH           — minimum vol_ratio for high-volume confirmation (1 close)
# (imported as: from strategies.day_box import ATR_BREAKDOWN_FACTOR, VOL_THRESH)

# Groww intraday fee approximations (used for fee-viability info only)
_SELL_FEE_RATE  = 0.00147    # ~0.147%: brokerage + STT(0.025%) + exchange + SEBI + GST
_BUY_FEE_RATE   = 0.00125    # ~0.125%: brokerage + stamp + exchange + SEBI + GST
ROUND_TRIP_FEE_PCT = (_SELL_FEE_RATE + _BUY_FEE_RATE) * 100   # ~0.272%


# ── ATR calculation ───────────────────────────────────────────────────────────

def _calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    """Average True Range over the last `period` candles."""
    if len(df) < period + 1:
        # Fallback: 0.5% of current close
        return float(df["Close"].iloc[-1]) * 0.005

    highs  = df["High"].values
    lows   = df["Low"].values
    closes = df["Close"].values

    tr_list = []
    for i in range(1, len(df)):
        hl  = highs[i] - lows[i]
        hpc = abs(highs[i] - closes[i - 1])
        lpc = abs(lows[i]  - closes[i - 1])
        tr_list.append(max(hl, hpc, lpc))

    return float(np.mean(tr_list[-period:]))


# ── Core box detection ────────────────────────────────────────────────────────

def _find_confirmed_box(prior: pd.DataFrame) -> dict | None:
    """Locate the most recently confirmed Darvas box in the prior candles.

    Scans backwards from the most recent candle, looking for a candidate
    HIGH that was followed by CONFIRM_BARS consecutive bars all below it.

    Returns a dict or None if no valid box was found.
    """
    n = len(prior)
    if n < MIN_BARS:
        return None

    search_start = n - CONFIRM_BARS - 1
    search_limit = max(0, n - MAX_BOX_AGE)

    for i in range(search_start, search_limit - 1, -1):
        candidate_top = float(prior["High"].iloc[i])

        # ── 1. Must be a local new high vs prior LOCAL_HI_N bars ──────────
        look_back_start = max(0, i - LOCAL_HI_N)
        prior_highs = prior["High"].iloc[look_back_start:i]
        if len(prior_highs) > 0 and candidate_top <= float(prior_highs.max()):
            continue

        # ── 2. Next CONFIRM_BARS bars must all have High < candidate_top ──
        window = prior.iloc[i + 1 : i + 1 + CONFIRM_BARS]
        if len(window) < CONFIRM_BARS:
            continue
        if not all(float(h) < candidate_top for h in window["High"].values):
            continue

        # ── 3. Box bottom = lowest Low in the confirmation window ─────────
        box_bottom = float(window["Low"].min())
        if box_bottom >= candidate_top:
            continue   # degenerate box

        box_age = n - 1 - (i + CONFIRM_BARS)  # bars since box was confirmed

        # Volume context: average vol over entire prior window
        avg_vol = float(prior["Volume"].mean())

        return {
            "box_top":    round(candidate_top, 2),
            "box_bottom": round(box_bottom, 2),
            "box_age":    box_age,
            "avg_vol":    avg_vol,
        }

    return None


def _zone(price: float, box: dict, atr: float) -> str:
    """Classify current price into a Darvas zone.

    New: "below_box" zone catches price that has dipped below box_bottom but
    hasn't yet cleared the ATR buffer. This prevents triggering breakdown sells
    on minor wicks or one-candle noise dips.

    Zones:
      breakout_up   — price cleared box_top + buffer (buy signal)
      buy_zone      — price in bottom 20% of box (mild buy)
      middle        — price in middle of box (neutral)
      sell_zone     — price in top 20% of box (mild caution)
      below_box     — price < box_bottom but > breakdown_threshold (tentative)
      breakout_down — price < box_bottom - ATR×factor (confirmed breakdown, sell)
    """
    top    = box["box_top"]
    bottom = box["box_bottom"]

    # True breakout requires price to clear box_top by BREAKOUT_BUF
    if price > top * (1 + BREAKOUT_BUF):
        return "breakout_up"

    # ATR-buffered breakdown: must be meaningfully below box_bottom
    breakdown_threshold = bottom - (atr * ATR_BREAKDOWN_FACTOR)
    if price < breakdown_threshold:
        return "breakout_down"

    # Below box_bottom but not past ATR buffer → tentative, risky to sell yet
    if price < bottom:
        return "below_box"

    width = top - bottom
    pos = (price - bottom) / width if width > 0 else 0.5
    if pos <= ZONE_EDGE:
        return "buy_zone"
    if pos >= (1.0 - ZONE_EDGE):
        return "sell_zone"
    return "middle"


# ── Public API ────────────────────────────────────────────────────────────────

def analyze(df: pd.DataFrame) -> dict:
    """Darvas Box analysis on the provided candle DataFrame.

    Score  Meaning
    ─────  ──────────────────────────────────────────────────────────────────
     +3    Tight-box (< 2% wide) breakout UP   + volume confirmation
     +2    Standard breakout above confirmed box top
     +1    Price at box support (bottom 20%), bouncing
      0    Dead zone / no confirmed box / insufficient data
     -1    Below box_bottom but not past ATR buffer (tentative — do NOT sell)
     -2    ATR-confirmed breakdown below box bottom (genuine sell signal)
     -3    Tight-box ATR-confirmed breakdown DOWN + volume (strongest sell)

    Args:
        df: DataFrame with columns Open/High/Low/Close/Volume, newest last.
            Must be IST-indexed 5-min candles.

    Returns:
        {"score": int, "confidence": float 0-1, "details": dict}
    """
    score   = 0
    details = {}

    if len(df) < MIN_BARS + CONFIRM_BARS + 1:
        return {"score": 0, "confidence": 0.0,
                "details": {"signal": "insufficient_data", "zone": "no_box"}}

    current_price = float(df["Close"].iloc[-1])
    prev_price    = float(df["Close"].iloc[-2]) if len(df) >= 2 else current_price
    current_vol   = float(df["Volume"].iloc[-1])

    # Use all prior candles (exclude the current one) for box detection + ATR
    prior = df.iloc[:-1]
    atr   = _calculate_atr(prior)
    box   = _find_confirmed_box(prior)

    if box is None:
        return {"score": 0, "confidence": 0.0,
                "details": {"signal": "no_confirmed_box", "zone": "no_box"}}

    top    = box["box_top"]
    bottom = box["box_bottom"]
    width  = top - bottom
    mid    = (top + bottom) / 2.0

    vol_ratio  = (current_vol / box["avg_vol"]) if box["avg_vol"] > 0 else 1.0
    has_volume = vol_ratio >= VOL_THRESH
    is_tight   = (width / mid * 100) < TIGHT_PCT if mid > 0 else False

    # ── Whipsaw indicators ─────────────────────────────────────────────────
    # Count how many of last 2 prior candles closed below box_bottom
    last_2_closes = prior["Close"].iloc[-2:].values if len(prior) >= 2 else []
    close_below_count = int(sum(1 for c in last_2_closes if float(c) < bottom))

    # Low-volume breakdown = higher whipsaw probability
    low_vol_on_breakdown = not has_volume

    # ── Fee viability (informational for AI) ───────────────────────────────
    box_width_pct = width / mid * 100 if mid > 0 else 0
    # Box must be wide enough to cover round-trip fees with a margin
    fee_viable = box_width_pct >= (ROUND_TRIP_FEE_PCT * 2)

    # ATR breakdown threshold for reference
    breakdown_threshold = round(bottom - atr * ATR_BREAKDOWN_FACTOR, 4)

    zone = _zone(current_price, box, atr)

    details.update({
        "box_top":              round(top, 2),
        "box_bottom":           round(bottom, 2),
        "box_width_pct":        round(box_width_pct, 2),
        "box_age_bars":         box["box_age"],
        "is_tight_box":         is_tight,
        "vol_ratio":            round(vol_ratio, 2),
        "zone":                 zone,
        "atr":                  round(atr, 4),
        "breakdown_threshold":  breakdown_threshold,
        "close_below_count":    close_below_count,
        "whipsaw_risk":         (zone in ("breakout_down", "below_box")) and low_vol_on_breakdown,
        "fee_viable":           fee_viable,
        "round_trip_fee_pct":   round(ROUND_TRIP_FEE_PCT, 3),
        "sell_fee_rate_pct":    round(_SELL_FEE_RATE * 100, 3),
    })

    if width > 0:
        details["position_in_box_pct"] = round(
            (current_price - bottom) / width * 100, 1
        )
    else:
        details["position_in_box_pct"] = 50.0

    # ── Score assignment ────────────────────────────────────────────────────
    if zone == "breakout_up":
        score = 3 if (is_tight and has_volume) else 2
        details["signal"] = ("tight_box_breakout_vol" if score == 3
                             else "breakout_above_box")

    elif zone == "breakout_down":
        # ATR-confirmed breakdown (genuine structural failure)
        score = -3 if (is_tight and has_volume) else -2
        details["signal"] = ("tight_box_breakdown_vol" if score == -3
                             else "breakdown_below_box")

    elif zone == "below_box":
        # Below box_bottom but NOT past the ATR buffer — tentative dip
        # Score -1: caution, but don't sell yet (might be a wick or brief dip)
        # Whipsaw risk is high here, especially on low volume
        score = -1
        details["signal"] = "below_box_unconfirmed"

    elif zone == "buy_zone":
        # At box support — only score if price is bouncing upward
        if current_price >= prev_price:
            score = 1
            details["signal"] = "bouncing_off_box_support"
        else:
            score = 0
            details["signal"] = "falling_through_support"

    elif zone == "sell_zone":
        # At box resistance — only score if price is rolling over
        if current_price <= prev_price:
            score = -1
            details["signal"] = "rejecting_at_box_resistance"
        else:
            score = 0
            details["signal"] = "pushing_through_resistance"

    else:  # middle
        score = 0
        details["signal"] = "in_dead_zone"

    score      = max(-3, min(3, score))
    confidence = round(min(abs(score) / 3.0, 1.0), 3)

    return {"score": score, "confidence": confidence, "details": details}
