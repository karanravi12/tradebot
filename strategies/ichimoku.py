"""Ichimoku Cloud system — full 5-component analysis."""

import pandas as pd
import config


def _calculate_ichimoku(df: pd.DataFrame) -> dict:
    """Calculate all 5 Ichimoku components."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # Tenkan-sen (Conversion Line) = (highest high + lowest low) / 2 over 9 periods
    tenkan = (
        high.rolling(window=config.ICHIMOKU_TENKAN).max()
        + low.rolling(window=config.ICHIMOKU_TENKAN).min()
    ) / 2

    # Kijun-sen (Base Line) = (highest high + lowest low) / 2 over 26 periods
    kijun = (
        high.rolling(window=config.ICHIMOKU_KIJUN).max()
        + low.rolling(window=config.ICHIMOKU_KIJUN).min()
    ) / 2

    # Senkou Span A (Leading Span A) = (Tenkan + Kijun) / 2, shifted 26 periods forward
    senkou_a = ((tenkan + kijun) / 2).shift(config.ICHIMOKU_KIJUN)

    # Senkou Span B (Leading Span B) = (highest high + lowest low) / 2 over 52, shifted 26
    senkou_b = (
        (
            high.rolling(window=config.ICHIMOKU_SENKOU_B).max()
            + low.rolling(window=config.ICHIMOKU_SENKOU_B).min()
        )
        / 2
    ).shift(config.ICHIMOKU_KIJUN)

    # Chikou Span (Lagging Span) = Close shifted 26 periods back
    chikou = close.shift(-config.ICHIMOKU_KIJUN)

    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
        "chikou": chikou,
    }


def analyze(df: pd.DataFrame) -> dict:
    """Analyze using Ichimoku Cloud system.

    Returns:
        dict with keys: score (-3 to +3), confidence (0-1), details (dict)
    """
    score = 0
    details = {}
    signals_aligned = 0

    ichi = _calculate_ichimoku(df)
    current_price = df["Close"].iloc[-1]

    tenkan_val = ichi["tenkan"].iloc[-1]
    kijun_val = ichi["kijun"].iloc[-1]
    senkou_a_val = ichi["senkou_a"].iloc[-1] if not pd.isna(ichi["senkou_a"].iloc[-1]) else None
    senkou_b_val = ichi["senkou_b"].iloc[-1] if not pd.isna(ichi["senkou_b"].iloc[-1]) else None

    details["tenkan"] = round(tenkan_val, 2) if not pd.isna(tenkan_val) else None
    details["kijun"] = round(kijun_val, 2) if not pd.isna(kijun_val) else None

    # ── Price vs Cloud ────────────────────────────────────────────────────
    if senkou_a_val is not None and senkou_b_val is not None:
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)

        details["cloud_top"] = round(cloud_top, 2)
        details["cloud_bottom"] = round(cloud_bottom, 2)

        if current_price > cloud_top:
            score += 1
            signals_aligned += 1
            details["price_vs_cloud"] = "above"
        elif current_price < cloud_bottom:
            score -= 1
            signals_aligned += 1
            details["price_vs_cloud"] = "below"
        else:
            details["price_vs_cloud"] = "inside"

    # ── Tenkan / Kijun Cross ──────────────────────────────────────────────
    if not pd.isna(tenkan_val) and not pd.isna(kijun_val):
        if len(ichi["tenkan"]) >= 2 and len(ichi["kijun"]) >= 2:
            prev_tenkan = ichi["tenkan"].iloc[-2]
            prev_kijun = ichi["kijun"].iloc[-2]

            if not pd.isna(prev_tenkan) and not pd.isna(prev_kijun):
                if prev_tenkan <= prev_kijun and tenkan_val > kijun_val:
                    details["tk_cross"] = "bullish"
                    signals_aligned += 1
                elif prev_tenkan >= prev_kijun and tenkan_val < kijun_val:
                    details["tk_cross"] = "bearish"
                    signals_aligned += 1
                elif tenkan_val > kijun_val:
                    details["tk_cross"] = "bullish_aligned"
                    signals_aligned += 1
                else:
                    details["tk_cross"] = "bearish_aligned"
                    signals_aligned += 1

    # ── Chikou vs Price ───────────────────────────────────────────────────
    # Chikou span is shifted back 26 periods, so we check its past position
    chikou_check_idx = -config.ICHIMOKU_KIJUN - 1
    if len(df) > abs(chikou_check_idx):
        chikou_val = df["Close"].iloc[-1]  # current close IS the chikou
        price_26_ago = df["Close"].iloc[chikou_check_idx]
        if chikou_val > price_26_ago:
            signals_aligned += 1
            details["chikou_signal"] = "bullish"
        else:
            signals_aligned += 1
            details["chikou_signal"] = "bearish"

    # ── Cloud Twist (Senkou A crosses Senkou B) ───────────────────────────
    if senkou_a_val is not None and senkou_b_val is not None:
        if len(ichi["senkou_a"]) >= 2 and len(ichi["senkou_b"]) >= 2:
            prev_a = ichi["senkou_a"].iloc[-2]
            prev_b = ichi["senkou_b"].iloc[-2]
            if not pd.isna(prev_a) and not pd.isna(prev_b):
                if prev_a <= prev_b and senkou_a_val > senkou_b_val:
                    details["cloud_twist"] = "bullish"
                elif prev_a >= prev_b and senkou_a_val < senkou_b_val:
                    details["cloud_twist"] = "bearish"

    # ── Full Confluence Check ─────────────────────────────────────────────
    # If all components agree, strong signal
    bullish_count = sum(1 for k, v in details.items() if isinstance(v, str) and "bullish" in v)
    bearish_count = sum(1 for k, v in details.items() if isinstance(v, str) and "bearish" in v)

    if bullish_count >= 3:
        score = 2
        details["confluence"] = "strong_bullish"
    elif bearish_count >= 3:
        score = -2
        details["confluence"] = "strong_bearish"

    score = max(-3, min(3, score))
    confidence = min(abs(score) / 3.0, 1.0)

    return {"score": score, "confidence": confidence, "details": details}
