"""VWAP (Volume Weighted Average Price) and session volume profile."""

import numpy as np
import pandas as pd
import config


def _calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate cumulative VWAP from the start of available data."""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vol = df["Volume"].cumsum()
    cum_tp_vol = (typical_price * df["Volume"]).cumsum()

    vwap = cum_tp_vol / cum_vol
    vwap = vwap.replace([np.inf, -np.inf], np.nan).ffill()
    return vwap


def _volume_profile(df: pd.DataFrame, bins: int = 20) -> dict:
    """Build a simple volume-at-price profile and find Point of Control (POC)."""
    price_min = df["Low"].min()
    price_max = df["High"].max()

    if price_min == price_max or price_max == 0:
        return {"poc": df["Close"].iloc[-1], "levels": []}

    bin_edges = np.linspace(price_min, price_max, bins + 1)
    vol_at_price = np.zeros(bins)

    closes = df["Close"].values
    volumes = df["Volume"].values

    for i in range(len(df)):
        bin_idx = int((closes[i] - price_min) / (price_max - price_min) * (bins - 1))
        bin_idx = max(0, min(bins - 1, bin_idx))
        vol_at_price[bin_idx] += volumes[i]

    poc_idx = np.argmax(vol_at_price)
    poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2

    return {"poc": poc_price, "vol_at_price": vol_at_price, "bin_edges": bin_edges}


def _detect_volume_spike(df: pd.DataFrame) -> bool:
    """Check if current volume is a spike (> 2x the 20-period average)."""
    if len(df) < config.VOLUME_AVG_PERIOD + 1:
        return False

    avg_vol = df["Volume"].iloc[-(config.VOLUME_AVG_PERIOD + 1) : -1].mean()
    current_vol = df["Volume"].iloc[-1]

    if avg_vol > 0:
        return current_vol > config.VOLUME_SPIKE_MULTIPLIER * avg_vol
    return False


def analyze(df: pd.DataFrame) -> dict:
    """Analyze using VWAP, volume profile, and volume spikes.

    Returns:
        dict with keys: score (-3 to +3), confidence (0-1), details (dict)
    """
    score = 0
    details = {}

    current_price = df["Close"].iloc[-1]
    prev_price = df["Close"].iloc[-2] if len(df) >= 2 else current_price

    # ── VWAP ──────────────────────────────────────────────────────────────
    vwap = _calculate_vwap(df)
    current_vwap = vwap.iloc[-1]
    prev_vwap = vwap.iloc[-2] if len(vwap) >= 2 else current_vwap

    details["vwap"] = round(current_vwap, 2)
    details["price_vs_vwap"] = "above" if current_price > current_vwap else "below"

    # Price crosses VWAP
    if prev_price <= prev_vwap and current_price > current_vwap:
        score += 1
        details["vwap_signal"] = "bullish_crossover"
    elif prev_price >= prev_vwap and current_price < current_vwap:
        score -= 1
        details["vwap_signal"] = "bearish_crossover"
    else:
        details["vwap_signal"] = "neutral"

    # ── Volume Profile / POC ──────────────────────────────────────────────
    vp = _volume_profile(df)
    poc = vp["poc"]
    details["poc"] = round(poc, 2)

    poc_proximity = abs(current_price - poc) / poc if poc > 0 else 1
    details["poc_proximity_pct"] = round(poc_proximity * 100, 2)

    # Price near POC (within 0.5%) — acts as support/resistance
    if poc_proximity < 0.005:
        if current_price > poc:
            score += 1
            details["poc_signal"] = "bouncing_off_support"
        else:
            score -= 1
            details["poc_signal"] = "rejecting_at_resistance"

    # ── Volume Spike ──────────────────────────────────────────────────────
    spike = _detect_volume_spike(df)
    details["volume_spike"] = spike

    if spike:
        # Volume spike confirms the current direction
        if current_price > prev_price:
            confidence_boost = 0.2
            details["volume_confirms"] = "bullish_breakout"
        else:
            confidence_boost = 0.2
            details["volume_confirms"] = "bearish_breakdown"
    else:
        confidence_boost = 0

    score = max(-3, min(3, score))
    confidence = min(abs(score) / 3.0 + confidence_boost, 1.0)

    return {"score": score, "confidence": confidence, "details": details}
