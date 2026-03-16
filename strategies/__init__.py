"""Strategy data collector — runs all 8 strategies and collects raw indicator data for the AI brain."""

import logging
import pandas as pd
from strategies import technical, smart_money, supertrend, vwap_session, ichimoku, bollinger, fibonacci, day_box

logger = logging.getLogger(__name__)

STRATEGIES = {
    "technical":   technical,
    "smart_money": smart_money,
    "supertrend":  supertrend,
    "vwap":        vwap_session,
    "ichimoku":    ichimoku,
    "bollinger":   bollinger,
    "fibonacci":   fibonacci,
    "day_box":     day_box,
}


def collect_indicators(df: pd.DataFrame, symbol: str) -> dict:
    """Run all 8 strategies and collect raw indicator data.

    Unlike the old analyze_all(), this does NOT compute a composite score
    or make BUY/SELL decisions. It just gathers the data for the AI brain.

    Args:
        df: OHLCV DataFrame for the stock
        symbol: stock ticker (for logging)

    Returns:
        dict with keys:
            indicators: {strategy_name: {score, confidence, details}} — raw data per strategy
            has_signal: bool — True if any strategy has a non-zero score (pre-filter for AI)
            active_strategies: list of strategy names with non-zero scores
    """
    indicators = {}
    has_signal = False
    active_strategies = []

    for name, strategy_module in STRATEGIES.items():
        try:
            result = strategy_module.analyze(df)
            indicators[name] = result
            if result["score"] != 0:
                has_signal = True
                active_strategies.append(f"{name}={result['score']:+d}")
        except Exception as e:
            logger.warning(f"Strategy '{name}' failed for {symbol}: {e}")
            indicators[name] = {"score": 0, "confidence": 0, "details": {"error": str(e)}}

    if active_strategies:
        logger.debug(f"[{symbol}] Active signals: {', '.join(active_strategies)}")

    return {
        "indicators": indicators,
        "has_signal": has_signal,
        "active_strategies": active_strategies,
    }
