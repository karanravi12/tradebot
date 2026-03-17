"""Claude AI Trading Brain — replaces algorithmic scoring with AI-powered decisions."""

import json
import logging
import os
import time
from datetime import datetime

import anthropic
import pytz

import config
import api_costs
import fees as groww_fees

logger = logging.getLogger(__name__)

IST = pytz.timezone(config.TIMEZONE)

SYSTEM_PROMPT = """You are an elite NSE intraday trading AI combining two institutional-grade methodologies:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METHODOLOGY 1: DARVAS BOX THEORY (Classical Technical Analysis)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Confirmed box top = candidate High followed by 3 consecutive bars all below it
• True breakout = Close > box_top × 1.002 (0.2% buffer — avoids fake tick-overs)
• Tight boxes (width < 2% of price) + high volume = highest-quality setups (score +3)
• Score +2: standard breakout above confirmed top
• Score +1: price bouncing off box support (bottom 20% of box range)
• Score -2: breakdown below box bottom (structural failure — EXIT)
• Score -3: tight-box breakdown + volume confirmation — strong short signal
• Box staleness: boxes older than 40 bars (~3.3h) are stale — discount them
• vol_ratio ≥ 1.5 = volume confirmed; is_tight_box = True = pristine setup

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METHODOLOGY 2: LUXALGO SMART MONEY CONCEPTS (Institutional PA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• ORDER BLOCKS (OB): The last opposing candle before an impulsive move (≥1.5%).
  When price returns to an OB — that's where institutions re-enter.
  at_bullish_ob=True + bullish structure = STRONG BUY confluence.

• FAIR VALUE GAPS (FVG): Imbalance between candle[i-1].High and candle[i+1].Low.
  Price "fills" FVGs — bullish FVG below current price = support magnet.
  Bearish FVG above = resistance that can cap upside.
  fvg_below=True = floor support from institutional imbalance.

• BREAK OF STRUCTURE (BOS): Higher High above prior swing high = bullish continuation.
  Lower Low below prior swing low = bearish continuation.
  last_bos="bullish" means smart money is accumulating.

• CHANGE OF CHARACTER (CHoCH): Trend REVERSAL signal.
  After HH sequence → sudden LL = institutions distributing.
  last_choch="bearish" after uptrend = exit/reduce longs immediately.

• PREMIUM / DISCOUNT ZONES (based on Fibonacci of swing range):
  discount  (price < 38.2% of swing range) → cheap — institutions BUY here
  equilibrium (38.2%–61.8%)              → neutral — wait for confluence
  premium   (price > 61.8% of swing range) → expensive — institutions SELL here
  Best entries: in discount zone + bullish OB + FVG below = maximum confluence

• SMC Score (-4 to +4):
  +4 = bullish BOS + discount + OB + FVG below (institutional accumulation)
  +3 = bullish structure + 2 confluence factors
  -3 = bearish structure + 2 confluence factors
  -4 = bearish BOS + premium + OB + FVG above (institutional distribution)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METHODOLOGY 3: COMMODITY HEDGE ROTATION (Risk Management)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHEN NIFTY IS BEARISH (market regime = BEARISH):
  → Rotate capital into GOLDBEES (gold ETF) or SILVERBEES (silver ETF)
  → Gold has historically INVERSE correlation to equities during sell-offs
  → This keeps money working (no cash drag) while protecting from equity downside
  → Even neutral gold/silver signals justify rotation during bearish Nifty

WHEN NIFTY IS BULLISH:
  → Focus on equity breakouts with Darvas + SMC confluence
  → GOLDBEES/SILVERBEES only if they show independent breakout signals

WHEN NIFTY IS NEUTRAL:
  → Be highly selective — only the very best setups (confidence ≥ 0.80)
  → Consider 1–2 defensive commodity positions as partial hedge

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RISK RULES (NON-NEGOTIABLE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Max 20% of portfolio per single position
• Always specify stop-loss (2–5% depending on volatility; tighter for ETFs)
• Minimum risk:reward = 1:2 (if you risk 3%, target at least 6%)
• HOLD is valid — sitting out during uncertainty preserves capital
• Max 5 simultaneous positions (enforced by the system)
• No new entries in first 15 minutes (09:15–09:30) — opening volatility
• Be cautious after 15:15 — late entries have poor risk:reward
• TRADING FEES (CRITICAL): Every buy+sell round-trip costs ~0.3% of trade value in Groww
  brokerage + STT + exchange charges + GST. The NET P&L after fees is what matters.
  → Never sell just because unrealized P&L is slightly positive — fees will make it a loss.
  → Minimum NET gain to actually profit: ≥0.35% above entry (after fees on both legs).
  → If NET P&L AFTER FEES is negative and no structural sell signal exists → HOLD.

CONFLUENCE HIERARCHY (use this framework for every decision):
  BOTH Darvas breakout (score ≥ +2) AND SMC bullish (score ≥ +2) = STRONG BUY
  Darvas breakout ONLY = moderate buy (watch for SMC resistance)
  SMC bullish ONLY (no box breakout) = wait or reduced size
  Any bearish CHoCH or BOS bearish = AVOID buying; consider defensive rotation

You must ALWAYS respond with valid JSON and nothing else. No markdown, no explanation outside the JSON."""

SINGLE_ANALYSIS_TEMPLATE = """Analyze this HELD position and decide: HOLD or SELL?

STOCK: {symbol}
CURRENT PRICE: ₹{price}
ENTRY PRICE: ₹{entry_price}
QUANTITY HELD: {qty}
UNREALIZED P&L: ₹{unrealized_pnl} ({pnl_pct}%)
EST. SELL FEE: ₹{est_sell_fee} | NET P&L AFTER FEES: ₹{net_pnl_after_fees} | BREAK-EVEN PRICE: ₹{breakeven_price}
⚠️  Do NOT sell unless NET P&L AFTER FEES is positive OR a clear structural sell signal exists.
CURRENT TIME: {current_time}

━━━ MARKET REGIME (NIFTYBEES) ━━━
{nifty_context_block}

━━━ DARVAS TRAILING STOP STATE ━━━
Entry Box Bottom (anchor): ₹{entry_box_bottom}
Current Trailing Stop: ₹{trailing_box_bottom}
Breakdown Close Count: {breakdown_count} (sell triggers after required closes below trailing stop)

━━━ DARVAS BOX ANALYSIS ━━━
{box_analysis}

━━━ SMART MONEY CONCEPTS (LuxAlgo) ━━━
{smc_analysis}

━━━ RECENT PRICE ACTION (last 10 candles) ━━━
{recent_candles}

━━━ PORTFOLIO STATE ━━━
Cash: ₹{cash} | Positions: {num_positions}/{max_positions} | Portfolio Value: ₹{portfolio_value} | Realized P&L: ₹{realized_pnl}

SELL SIGNALS TO WATCH FOR:
• Darvas box breakdown (score ≤ -2) — structural failure
• SMC bearish CHoCH (last_choch=bearish) — trend reversal
• Price entering premium zone with bearish OB above — distribution
• P&L deteriorating with no recovery signal from either system

HOLD SIGNALS:
• Box still intact (zone=middle or buy_zone, not breakdown)
• SMC structure still bullish (trend=bullish, no CHoCH)
• Price at bullish OB or FVG support below — institutions holding
• Unrealized profit with upward momentum in both systems

Respond with JSON:
{{
  "action": "HOLD" or "SELL",
  "confidence": 0.0 to 1.0,
  "reasoning": "Reference specific Box Theory + SMC signals in your explanation",
  "stop_loss_price": number (updated stop-loss based on box/swing levels),
  "target_price": number (next resistance / box top / swing high)
}}"""

BATCH_ANALYSIS_TEMPLATE = """You are the trading AI. Review all candidates and select the BEST buy opportunities (0 to 3 stocks max).

CURRENT TIME: {current_time}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MARKET REGIME (NIFTYBEES — Nifty 50 proxy):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{nifty_context_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PORTFOLIO STATE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cash Available: ₹{cash}
Current Positions: {num_positions}/{max_positions} (holding: {held_symbols})
Total Portfolio Value: ₹{portfolio_value}
Realized P&L today: ₹{realized_pnl}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CANDIDATES — FULL DUAL-STRATEGY ANALYSIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each candidate includes:
  • darvas_box: Darvas Box Theory score (-3 to +3), zone, box levels, volume
  • smart_money: LuxAlgo SMC score (-4 to +4), structure, OBs, FVGs, premium/discount zone
  • NOTE: GOLDBEES and SILVERBEES are ALWAYS included as commodity hedge alternatives

{candidates_json}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION FRAMEWORK — APPLY IN THIS ORDER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — MARKET REGIME CHECK:
  • If BEARISH Nifty → strongly prefer GOLDBEES/SILVERBEES for capital protection
  • If BULLISH Nifty → focus on equity breakouts (equities will outperform gold)
  • If NEUTRAL Nifty → mixed approach; higher bar for equity conviction

STEP 2 — COMMODITY CANDIDATES (GOLDBEES, SILVERBEES):
  • During BEARISH market: buy even with modest signals (SMC neutral is fine)
  • Gold/Silver provide inverse correlation to falling equities — they ARE the hedge
  • Stop-loss tighter for ETFs (1.5–2% vs 3% for stocks)

STEP 3 — EQUITY CANDIDATES (all other stocks):
  STRONG BUY: Darvas score ≥ +2 AND SMC score ≥ +2 (dual confirmation)
  MODERATE:   Darvas score ≥ +2 OR SMC score ≥ +3 (single strong signal)
  SKIP:       Any bearish CHoCH or price in premium zone without OB support

STEP 4 — CONFLUENCE SCORING for equities:
  at_bullish_ob + fvg_below + discount zone = 3/3 → maximum conviction
  at_bullish_ob + fvg_below                = 2/3 → high conviction
  is_tight_box + vol_ratio ≥ 1.5           = tight breakout → add size
  bearish CHoCH or premium zone             = AVOID regardless of box signal

STEP 5 — POSITION SIZING & RISK:
  • High confidence (≥ 0.85): 15–20% of portfolio
  • Medium confidence (0.70–0.84): 10–15%
  • Gold/silver in defensive mode: 15–20% (higher allocation, safer asset)
  • Stop-loss = below box_bottom OR below nearest bullish OB / swing low (whichever tighter)
  • Target = next swing high OR box measured move

STEP 6 — FEE REALITY CHECK:
  • Every round-trip (buy + sell) costs ~0.3% of trade value in Groww fees (brokerage + STT + GST).
  • Minimum move needed to profit after fees: ≥0.35% from entry price.
  • Your stop-loss and target MUST account for fees — a 0.2% gain is actually a loss.
  • For a ₹10,000 position: ~₹30 in fees → need ₹35+ gain just to break even.
  • If the expected move is small (<0.5%), do NOT enter — fees destroy the trade.

FINAL RULES:
  • 0 picks is valid — do not force trades when setup quality is low
  • Max 3 picks; consider sector diversification (not 3 banking stocks)
  • With ₹{cash} cash, ensure you can afford the position
  • confidence ≥ 0.70 required to include any pick

Respond with a JSON object (NOT an array):
{{
  "picks": [
    {{"symbol": "TICKER", "action": "BUY", "confidence": 0.0-1.0, "reasoning": "...", "position_size_pct": 0.05-0.20, "stop_loss_pct": 0.015-0.05, "target_pct": 0.03-0.12, "trade_type": "equity_breakout" or "commodity_hedge" or "equity_smc"}}
  ],
  "reasoning": "REQUIRED — Always explain your decision in 1-2 sentences. If no buys: say why (e.g. 'No high-conviction setups — equities in premium zone, no tight box breakouts' or 'Fees would eat the edge on these moves'). If buying: briefly cite the best setup."
}}

picks can be empty []. reasoning is ALWAYS required."""


def _get_client() -> anthropic.Anthropic:
    """Create Anthropic client with API key from env or config."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", getattr(config, "ANTHROPIC_API_KEY", ""))
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Export it: export ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic(api_key=api_key)


def _check_budget() -> bool:
    """Check if we're within daily API budget. Returns True if OK to proceed."""
    allowed, spent, limit = api_costs.check_daily_limit()
    if not allowed:
        logger.warning(f"[AI] Daily API limit reached: ${spent:.4f} / ${limit:.4f}")
    return allowed


def _call_api(system: str, prompt: str, purpose: str, temperature: float | None = None) -> str:
    """Make a streaming API call to Claude, track costs, return response text.

    Uses streaming to avoid the 10-minute timeout issue in the Anthropic SDK.
    Automatically retries on 429 rate-limit errors with exponential backoff.
    """
    if not _check_budget():
        raise RuntimeError(
            f"Daily API spending limit (${getattr(config, 'API_DAILY_LIMIT_USD', 0)}) reached. "
            "Increase the limit in Settings or wait until tomorrow."
        )

    client = _get_client()
    temp = temperature if temperature is not None else config.CLAUDE_TEMPERATURE
    model = config.CLAUDE_MODEL

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response_text = ""
            input_tokens = 0
            output_tokens = 0

            with client.messages.stream(
                model=model,
                max_tokens=config.CLAUDE_MAX_TOKENS,
                temperature=temp,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    response_text += text

                final = stream.get_final_message()
                input_tokens = final.usage.input_tokens
                output_tokens = final.usage.output_tokens

            api_costs.record_usage(model, input_tokens, output_tokens, purpose)
            return response_text

        except anthropic.RateLimitError as e:
            wait = 15 * (2 ** attempt)  # 15s, 30s, 60s
            if attempt < max_retries - 1:
                logger.warning(f"[AI] Rate limit hit ({purpose}), retrying in {wait}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Rate limit reached after {max_retries} retries. "
                    "The API allows 30,000 tokens/min. Please wait a minute and try again."
                ) from e


def _format_candles(df, n: int = 10) -> str:
    """Format last N candles as a compact table for the prompt."""
    recent = df.tail(n)
    lines = ["Time | Open | High | Low | Close | Volume"]
    for idx, row in recent.iterrows():
        ts = idx.strftime("%H:%M") if hasattr(idx, "strftime") else str(idx)
        lines.append(
            f"{ts} | {row['Open']:.2f} | {row['High']:.2f} | "
            f"{row['Low']:.2f} | {row['Close']:.2f} | {int(row['Volume'])}"
        )
    return "\n".join(lines)


def _parse_json_response(text: str) -> dict | list:
    """Parse JSON from Claude's response, handling edge cases."""
    text = text.strip()
    # Remove markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()

    return json.loads(text)


def _fmt_box_analysis(indicators: dict) -> str:
    """Format Darvas Box data as a concise readable block for the prompt."""
    box = indicators.get("day_box", {})
    if not box:
        return "No box data available."
    det = box.get("details", {})
    lines = [
        f"Score: {box.get('score', 0)}  |  Zone: {det.get('zone', 'unknown')}",
        f"Box Top: ₹{det.get('box_top', 'N/A')}  |  Box Bottom: ₹{det.get('box_bottom', 'N/A')}",
        f"Box Width: {det.get('box_width_pct', 0):.2f}%  |  Tight Box: {det.get('is_tight_box', False)}",
        f"Volume Ratio: {det.get('vol_ratio', 1.0):.2f}x  |  Box Age: {det.get('box_age_bars', 'N/A')} bars",
        f"Signal: {det.get('signal', 'N/A')}  |  Position in Box: {det.get('position_in_box_pct', 50):.1f}%",
    ]
    return "\n".join(lines)


def _fmt_smc_analysis(indicators: dict) -> str:
    """Format LuxAlgo SMC data as a concise readable block for the prompt."""
    smc = indicators.get("smc", {})
    if not smc:
        return "SMC data not available."
    det = smc.get("details", {})
    lines = [
        f"Score: {smc.get('score', 0)}  |  Bias: {smc.get('smc_bias', 'neutral').upper()}",
        f"Trend: {det.get('trend', 'unknown')}  |  Last BOS: {det.get('last_bos', 'none')}  |  Last CHoCH: {det.get('last_choch', 'none')}",
        f"P/D Zone: {det.get('pd_zone', 'equilibrium').upper()}  |  Swing High: ₹{det.get('swing_high', 'N/A')}  |  Swing Low: ₹{det.get('swing_low', 'N/A')}",
        f"At Bullish OB: {det.get('at_bullish_ob', False)}  |  At Bearish OB: {det.get('at_bearish_ob', False)}",
        f"FVG Support Below: {det.get('fvg_below', False)}  |  FVG Resistance Above: {det.get('fvg_above', False)}",
    ]
    if det.get("bullish_obs"):
        ob = det["bullish_obs"][-1]
        lines.append(f"Nearest Bullish OB Zone: ₹{ob['bottom']}–₹{ob['top']} (strength {ob['strength']}%)")
    if det.get("bearish_obs"):
        ob = det["bearish_obs"][-1]
        lines.append(f"Nearest Bearish OB Zone: ₹{ob['bottom']}–₹{ob['top']} (strength {ob['strength']}%)")
    return "\n".join(lines)


def analyze_single(
    symbol: str,
    df,
    indicators: dict,
    portfolio_state: dict,
    current_time_str: str | None = None,
    nifty_context: dict | None = None,
) -> dict:
    """Ask Claude AI whether to HOLD or SELL a currently held position."""
    if df is None or df.empty:
        return {"action": "HOLD", "reason": "No price data available", "confidence": 0}
    val = df["Close"].iloc[-1]
    current_price = float(val.iloc[0]) if hasattr(val, "iloc") else float(val)
    pos = portfolio_state.get("position", {})
    entry_price = pos.get("avg_price", current_price)
    qty = pos.get("qty", 0)
    unrealized_pnl = (current_price - entry_price) * qty
    pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

    trade_value = current_price * qty
    sell_fee_info = groww_fees.calculate_sell_fees(current_price, qty)
    est_sell_fee = round(sell_fee_info["total_fees"], 2)
    net_pnl_after_fees = round(unrealized_pnl - est_sell_fee, 2)
    # Breakeven: buy fees already paid; just need sell-side fees + buy-side fees to be covered
    buy_fee_info = groww_fees.calculate_buy_fees(entry_price, qty)
    total_round_trip_fees = buy_fee_info["total_fees"] + sell_fee_info["total_fees"]
    breakeven_price = round(entry_price + (total_round_trip_fees / qty if qty > 0 else 0), 2)

    entry_box_bottom = pos.get("entry_box_bottom", "N/A")
    trailing_box_bottom = pos.get("trailing_box_bottom", "N/A")
    breakdown_count = pos.get("breakdown_count", 0)

    prompt = SINGLE_ANALYSIS_TEMPLATE.format(
        symbol=symbol,
        price=f"{current_price:.2f}",
        entry_price=f"{entry_price:.2f}",
        qty=qty,
        unrealized_pnl=f"{unrealized_pnl:.2f}",
        pnl_pct=f"{pnl_pct:.1f}",
        est_sell_fee=f"{est_sell_fee:.2f}",
        net_pnl_after_fees=f"{net_pnl_after_fees:.2f}",
        breakeven_price=f"{breakeven_price:.2f}",
        current_time=current_time_str or datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
        nifty_context_block=_fmt_nifty_block(nifty_context),
        entry_box_bottom=entry_box_bottom,
        trailing_box_bottom=trailing_box_bottom,
        breakdown_count=breakdown_count,
        box_analysis=_fmt_box_analysis(indicators),
        smc_analysis=_fmt_smc_analysis(indicators),
        recent_candles=_format_candles(df),
        cash=f"{portfolio_state['cash']:.2f}",
        num_positions=portfolio_state["num_positions"],
        max_positions=config.MAX_POSITIONS,
        portfolio_value=f"{portfolio_state['portfolio_value']:.2f}",
        realized_pnl=f"{portfolio_state['realized_pnl']:.2f}",
    )

    try:
        response_text = _call_api(SYSTEM_PROMPT, prompt, f"analyze_single:{symbol}")
        result = _parse_json_response(response_text)
        logger.info(
            f"[AI] {symbol}: {result.get('action', 'HOLD')} "
            f"(confidence={result.get('confidence', 0):.0%}) — "
            f"{result.get('reasoning', '')[:120]}"
        )
        return result

    except json.JSONDecodeError as e:
        logger.error(f"[AI] Failed to parse response for {symbol}: {e}")
        return {"action": "HOLD", "confidence": 0, "reasoning": f"JSON parse error: {e}"}
    except Exception as e:
        logger.error(f"[AI] API error for {symbol}: {e}")
        return {"action": "HOLD", "confidence": 0, "reasoning": f"API error: {e}"}


def _fmt_nifty_block(nifty_context: dict | None) -> str:
    """Format Nifty market regime context as a clear directive block."""
    if not nifty_context:
        return "Market regime data unavailable — proceed with standard analysis (neutral assumption)."

    regime = nifty_context.get("market_regime", "NEUTRAL")
    emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(regime, "🟡")

    lines = [
        f"{emoji} MARKET REGIME: {regime}",
        f"Nifty Box Score: {nifty_context.get('box_score', 0)}  |  Zone: {nifty_context.get('box_zone', 'unknown')}",
        f"Nifty SMC Score: {nifty_context.get('smc_score', 0)}  |  Bias: {nifty_context.get('smc_bias', 'neutral').upper()}",
        f"Nifty Trend: {nifty_context.get('nifty_trend', 'unknown')}  |  P/D Zone: {nifty_context.get('nifty_pd_zone', 'equilibrium').upper()}",
        f"⟹  {nifty_context.get('regime_note', '')}",
    ]
    return "\n".join(lines)


def _fmt_candidate(c: dict) -> dict:
    """Build a rich candidate dict with structured Box + SMC sections."""
    ind = c.get("indicators", {})
    box = ind.get("day_box", {})
    smc = ind.get("smc", {})

    box_det = box.get("details", {}) if box else {}
    smc_det = smc.get("details", {}) if smc else {}

    symbol = c["symbol"]
    is_commodity = symbol in {"GOLDBEES", "SILVERBEES", "AXISGOLD", "SILVERETF",
                               "HDFCGOLD", "ICICIGOLD", "KOTAKGOLD", "QGOLDHALF"}
    sector = "Commodity ETF / Safe Haven" if is_commodity else c.get("sector", "Equity")

    return {
        "symbol":  symbol,
        "price":   round(c["price"], 2),
        "sector":  sector,
        "is_commodity_hedge": is_commodity,
        "darvas_box": {
            "score":             box.get("score", 0),
            "zone":              box_det.get("zone", "unknown"),
            "signal":            box_det.get("signal", "N/A"),
            "box_top":           box_det.get("box_top"),
            "box_bottom":        box_det.get("box_bottom"),
            "box_width_pct":     box_det.get("box_width_pct"),
            "is_tight_box":      box_det.get("is_tight_box", False),
            "vol_ratio":         box_det.get("vol_ratio"),
            "position_in_box_%": box_det.get("position_in_box_pct"),
            "box_age_bars":      box_det.get("box_age_bars"),
        },
        "smart_money_concepts": {
            "score":          smc.get("score", 0),
            "smc_bias":       smc.get("smc_bias", "neutral"),
            "trend":          smc_det.get("trend", "unknown"),
            "last_bos":       smc_det.get("last_bos"),
            "last_choch":     smc_det.get("last_choch"),
            "pd_zone":        smc_det.get("pd_zone", "equilibrium"),
            "at_bullish_ob":  smc_det.get("at_bullish_ob", False),
            "at_bearish_ob":  smc_det.get("at_bearish_ob", False),
            "fvg_below":      smc_det.get("fvg_below", False),
            "fvg_above":      smc_det.get("fvg_above", False),
            "swing_high":     smc_det.get("swing_high"),
            "swing_low":      smc_det.get("swing_low"),
            "bullish_ob":     smc_det.get("bullish_obs", [{}])[-1] if smc_det.get("bullish_obs") else None,
            "bearish_ob":     smc_det.get("bearish_obs", [{}])[-1] if smc_det.get("bearish_obs") else None,
            "bullish_fvgs":   smc_det.get("bullish_fvgs", [])[-2:],
            "bearish_fvgs":   smc_det.get("bearish_fvgs", [])[-2:],
        },
        "recent_candles": c.get("recent_candles", ""),
    }


def analyze_batch(
    candidates: list[dict],
    portfolio_state: dict,
    current_time_str: str | None = None,
    nifty_context: dict | None = None,
) -> list[dict]:
    """Ask Claude AI to pick the best buy candidates from a batch.

    Args:
        candidates:      List of candidate dicts (symbol, price, indicators, recent_candles).
                         indicators dict should contain "day_box" AND "smc" keys.
        portfolio_state: Current portfolio snapshot.
        current_time_str: Formatted timestamp string.
        nifty_context:   Market regime dict from NIFTYBEES box+SMC analysis (optional).
    """
    if not candidates:
        return {"picks": [], "reasoning": "No candidates provided"}

    candidates_formatted = [_fmt_candidate(c) for c in candidates]
    held = list(portfolio_state.get("held_symbols", []))

    prompt = BATCH_ANALYSIS_TEMPLATE.format(
        current_time=current_time_str or datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
        cash=f"{portfolio_state['cash']:.2f}",
        num_positions=portfolio_state["num_positions"],
        max_positions=config.MAX_POSITIONS,
        held_symbols=", ".join(held) if held else "none",
        portfolio_value=f"{portfolio_state['portfolio_value']:.2f}",
        realized_pnl=f"{portfolio_state['realized_pnl']:.2f}",
        nifty_context_block=_fmt_nifty_block(nifty_context),
        candidates_json=json.dumps(candidates_formatted, indent=2, default=str),
    )

    try:
        response_text = _call_api(SYSTEM_PROMPT, prompt, "analyze_batch")
        parsed = _parse_json_response(response_text)

        # Support both new format {picks, reasoning} and legacy array
        if isinstance(parsed, dict) and "picks" in parsed:
            picks = parsed.get("picks", [])
            reasoning = parsed.get("reasoning", "")
        elif isinstance(parsed, list):
            picks = parsed
            reasoning = ""
        else:
            picks = [parsed] if isinstance(parsed, dict) else []
            reasoning = ""

        for r in picks:
            logger.info(
                f"[AI] BUY recommendation: {r.get('symbol', '?')} "
                f"(confidence={r.get('confidence', 0):.0%}, "
                f"size={r.get('position_size_pct', 0):.0%}) — "
                f"{r.get('reasoning', '')[:120]}"
            )

        return {"picks": picks, "reasoning": reasoning}

    except json.JSONDecodeError as e:
        logger.error(f"[AI] Failed to parse batch response: {e}")
        return {"picks": [], "reasoning": f"Parse error: {e}"}
    except Exception as e:
        logger.error(f"[AI] Batch API error: {e}")
        return {"picks": [], "reasoning": str(e)}


CHAT_SYSTEM_PROMPT = """You are an expert Indian stock market (NSE) trading assistant. You have real-time access to the user's paper trading portfolio and can discuss:
- Current positions, P&L, and portfolio performance
- Market analysis using 8 technical strategies (RSI, MACD, EMA, Smart Money Concepts, Supertrend, VWAP, Ichimoku, Bollinger, Fibonacci, Darvas Box)
- Trading strategy recommendations
- Risk management advice
- NSE market conditions and sectoral trends

Be concise, use INR currency, and reference actual portfolio data when available. You are helpful and conversational."""


def _slim_trade(t: dict) -> dict:
    """Strip heavy fields from a trade record to reduce token count."""
    keys = ("timestamp", "action", "symbol", "qty", "price", "pnl", "pnl_pct",
            "fees", "reason", "confidence")
    return {k: t[k] for k in keys if k in t}


def chat(message: str, portfolio_state: dict, recent_trades: list) -> str:
    """Chat with the AI trader about portfolio, market, or strategy questions."""
    # Keep positions compact — only the key fields
    positions = [
        {k: p[k] for k in ("symbol", "qty", "avg_price", "current_price", "pnl", "pnl_pct") if k in p}
        for p in portfolio_state.get("positions", [])
    ]

    # Trim trades: last 5 only, no fees_detail/ai_reasoning blobs
    slim_trades = [_slim_trade(t) for t in recent_trades[-5:]]

    context = f"""PORTFOLIO:
Cash=₹{portfolio_state.get('cash', 0):,.0f} | Value=₹{portfolio_state.get('total_value', 0):,.0f} | P&L=₹{portfolio_state.get('overall_pnl', 0):+,.0f} ({portfolio_state.get('overall_pnl_pct', 0):+.1f}%) | Realized=₹{portfolio_state.get('realized_pnl', 0):+,.0f} | Positions={portfolio_state.get('num_positions', 0)}/{portfolio_state.get('max_positions', 5)} | WinRate={portfolio_state.get('win_rate', 0):.0f}% | Trades={portfolio_state.get('total_trades', 0)}

POSITIONS: {json.dumps(positions, default=str)}

RECENT TRADES (last 5): {json.dumps(slim_trades, default=str)}

TIME: {datetime.now(IST).strftime('%Y-%m-%d %H:%M IST')}

USER: {message}"""

    try:
        return _call_api(CHAT_SYSTEM_PROMPT, context, "chat", temperature=0.5)
    except Exception as e:
        logger.error(f"[AI] Chat error: {e}")
        return f"Error communicating with AI: {e}"
