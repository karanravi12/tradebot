"""AI-powered trading engine — scans stocks, feeds data to Claude AI, executes paper trades."""

import csv
import logging
import os
import threading
from datetime import datetime, timedelta

import pytz

import config
import data_fetcher
import ai_brain
import fees as groww_fees
from portfolio import Portfolio
from strategies.day_box import (
    analyze as box_analyze,
    ATR_BREAKDOWN_FACTOR,   # how far below box_bottom counts as confirmed breakdown
    VOL_THRESH,             # vol_ratio ≥ 1.5 → high-volume confirmation (1 close to sell)
)
from strategies.luxalgo import analyze as smc_analyze

import pandas as pd

logger = logging.getLogger(__name__)
IST = pytz.timezone(config.TIMEZONE)

# SocketIO instance — set by app.py for real-time updates, None in CLI mode
_socketio = None

# Prevent concurrent scans (scheduler + manual API call overlap)
_scan_lock = threading.Lock()

# ── Interval-based AI eval (every 1 hour from app start, during market hours) ──
_AI_EVAL_INTERVAL_SECONDS = 3600        # 1 hour
_last_ai_eval_ts: datetime | None = None   # None = hasn't fired yet this session


def _ai_eval_should_fire(now: datetime) -> bool:
    """True if AI has never run this session OR 1 hour has elapsed since last run."""
    if _last_ai_eval_ts is None:
        return True
    return (now - _last_ai_eval_ts).total_seconds() >= _AI_EVAL_INTERVAL_SECONDS


def _mark_ai_eval_fired(now: datetime) -> None:
    global _last_ai_eval_ts
    _last_ai_eval_ts = now


def get_ai_eval_info() -> dict:
    """Returns AI eval timing info for the web UI (last run, next run, countdown)."""
    now = datetime.now(IST)
    if _last_ai_eval_ts is None:
        next_ts = now                    # fires immediately on next scan
        seconds_until = 0
    else:
        next_ts = _last_ai_eval_ts + timedelta(seconds=_AI_EVAL_INTERVAL_SECONDS)
        seconds_until = max(0, (next_ts - now).total_seconds())
    return {
        "last_eval": _last_ai_eval_ts.isoformat() if _last_ai_eval_ts else None,
        "next_eval": next_ts.isoformat(),
        "seconds_until_next": int(seconds_until),
        "interval_seconds": _AI_EVAL_INTERVAL_SECONDS,
    }


# ── Hybrid strategy constants ──────────────────────────────────────────────
COMMODITY_ETFS     = {"GOLDBEES", "SILVERBEES", "AXISGOLD",
                      "HDFCGOLD", "QGOLDHALF"}
NIFTY_PROXY        = "NIFTYBEES"   # used for market regime — not a buy candidate
BOX_SELL_THRESHOLD = -2            # box score ≤ this → instant sell (no AI needed)
MIN_HOLD_MINUTES   = 15            # hold ≥ 15 min before any box-triggered sell


def _safe_price(df: pd.DataFrame, col: str = "Close", idx: int = -1) -> float:
    """Safely extract a scalar price from a DataFrame column."""
    val = df[col].iloc[idx]
    if isinstance(val, pd.Series):
        return float(val.iloc[0])
    return float(val)


def set_socketio(sio):
    """Set the SocketIO instance for real-time web UI updates."""
    global _socketio
    _socketio = sio


def _emit(event: str, data: dict):
    """Emit a WebSocket event if SocketIO is available."""
    if _socketio:
        _socketio.emit(event, data)


# LTP cache from the previous scan — used to detect movers without candle data
_prev_ltp: dict[str, float] = {}


TRADES_CSV = os.path.join(os.path.dirname(__file__), "logs", "trades.csv")
CSV_HEADERS = [
    "timestamp", "action", "symbol", "qty", "price", "cost_or_revenue",
    "pnl", "pnl_pct", "reason", "signal", "confidence",
    "cash_remaining", "portfolio_value", "ai_reasoning",
]


def _ensure_csv():
    """Create trades.csv with headers if it doesn't exist."""
    os.makedirs(os.path.dirname(TRADES_CSV), exist_ok=True)
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)


def _log_trade_csv(trade: dict, portfolio_value: float):
    """Append a trade record to the CSV log."""
    _ensure_csv()
    row = [
        trade.get("timestamp", ""),
        trade.get("action", ""),
        trade.get("symbol", ""),
        trade.get("qty", 0),
        trade.get("price", 0),
        trade.get("cost", trade.get("revenue", 0)),
        trade.get("pnl", ""),
        trade.get("pnl_pct", ""),
        trade.get("reason", ""),
        trade.get("signal", ""),
        trade.get("confidence", 0),
        trade.get("cash_remaining", 0),
        round(portfolio_value, 2),
        trade.get("ai_reasoning", ""),
    ]
    with open(TRADES_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _build_portfolio_state(portfolio: Portfolio, current_prices: dict) -> dict:
    """Build portfolio state dict for the AI brain."""
    return {
        "cash": portfolio.cash,
        "num_positions": len(portfolio.positions),
        "portfolio_value": portfolio.total_value(current_prices),
        "realized_pnl": portfolio.realized_pnl,
        "held_symbols": list(portfolio.positions.keys()),
    }


def scan_and_trade(portfolio: Portfolio, force_ai: bool = False):
    """Main scan cycle: fetch data → collect indicators → AI decides → execute.

    force_ai=True resets the 1-hour interval so AI fires immediately (used by manual scan).
    """
    if not _scan_lock.acquire(blocking=False):
        logger.info("Scan already in progress — skipping this cycle")
        return
    if force_ai:
        global _last_ai_eval_ts
        _last_ai_eval_ts = None
    try:
        _scan_and_trade_impl(portfolio)
    finally:
        _scan_lock.release()


def _scan_and_trade_impl(portfolio: Portfolio):
    logger.info("=" * 60)
    logger.info(f"AI SCAN STARTED at {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")
    logger.info(f"Cash: ₹{portfolio.cash:.2f} | Positions: {len(portfolio.positions)}/{config.MAX_POSITIONS}")
    logger.info("=" * 60)

    current_prices = {}
    scan_count = 0
    error_count = 0
    _niftybees_df = None   # populated in Phase 2; pre-declared so Phase 1 can reference it

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: Manage existing positions
    #   Order mirrors backtest: stop-loss → AI (eval windows only) → box auto-sell
    #   SMC enriches AI context only — not a standalone sell trigger
    # ══════════════════════════════════════════════════════════════════════
    now = datetime.now(IST)
    current_time_str = now.strftime("%Y-%m-%d %H:%M IST")
    ai_eval_fires = _ai_eval_should_fire(now)

    # Compute Nifty regime once per cycle — used for both sells and buys
    _nifty_context_cache = None
    if ai_eval_fires:
        nifty_df = _niftybees_df if _niftybees_df is not None else data_fetcher.fetch_data(NIFTY_PROXY)
        if nifty_df is not None and len(nifty_df) >= 25:
            n_box = box_analyze(nifty_df)
            n_smc = smc_analyze(nifty_df)
            n_box_score = int(n_box.get("score", 0))
            n_smc_score = int(n_smc.get("score", 0))
            combined = n_box_score + n_smc_score
            if combined >= 2:
                regime, regime_note = "BULLISH", "Nifty trending up — equity breakouts favoured"
            elif combined <= -2:
                regime, regime_note = "BEARISH", "Nifty in downtrend — DEFENSIVE MODE: prefer GOLDBEES/SILVERBEES"
            else:
                regime, regime_note = "NEUTRAL", "Nifty sideways — be selective; high-conviction setups only"
            _nifty_context_cache = {
                "proxy": NIFTY_PROXY,
                "market_regime": regime,
                "regime_note": regime_note,
                "box_score": n_box_score,
                "box_zone": n_box.get("details", {}).get("zone", "unknown"),
                "smc_score": n_smc_score,
                "smc_bias": n_smc.get("smc_bias", "neutral"),
                "nifty_trend": n_smc.get("details", {}).get("trend", "unknown"),
                "nifty_pd_zone": n_smc.get("details", {}).get("pd_zone", "equilibrium"),
            }
            logger.info(f"Nifty regime: {regime} (box={n_box_score}, smc={n_smc_score})")

    for symbol in list(portfolio.positions.keys()):
        df = data_fetcher.fetch_data(symbol)
        if df is None:
            error_count += 1
            continue

        current_price = _safe_price(df)
        current_prices[symbol] = current_price
        portfolio.update_highest_price(symbol, current_price)

        # 1. Hard stop-loss (no AI, every scan) — mirrors backtest
        if portfolio.check_stop_loss(symbol, current_price):
            signal_info = {"signal": "STOP_LOSS", "composite_score": 0}
            trade = portfolio.sell(symbol, current_price, "stop_loss", signal_info)
            if trade:
                trade["ai_reasoning"] = "Hard stop-loss triggered (3% loss from entry)"
                trade["confidence"] = 1.0
                _log_trade_csv(trade, portfolio.total_value(current_prices))
                _emit("trade_executed", trade)
            continue

        # 2. Compute box + SMC (SMC enriches AI context; box drives auto-sell)
        box_result = box_analyze(df)
        box_score  = int(box_result.get("score", 0))
        smc_result = smc_analyze(df)

        # 3. AI evaluates hold/sell — only at 4 windows (9:30, 11:30, 13:30, 15:00)
        if ai_eval_fires:
            portfolio_state = _build_portfolio_state(portfolio, current_prices)
            portfolio_state["position"] = portfolio.positions[symbol]

            ai_decision = ai_brain.analyze_single(
                symbol=symbol,
                df=df,
                indicators={"day_box": box_result, "smc": smc_result},
                portfolio_state=portfolio_state,
                current_time_str=current_time_str,
                nifty_context=_nifty_context_cache,
            )

            if ai_decision.get("action") == "SELL" and ai_decision.get("confidence", 0) >= 0.65:
                signal_info = {"signal": "AI_SELL", "composite_score": ai_decision.get("confidence", 0)}
                trade = portfolio.sell(symbol, current_price, "ai_signal", signal_info)
                if trade:
                    trade["ai_reasoning"] = ai_decision.get("reasoning", "")
                    trade["confidence"] = ai_decision.get("confidence", 0)
                    _log_trade_csv(trade, portfolio.total_value(current_prices))
                    _emit("trade_executed", trade)
                continue

        # 4. Darvas Trailing Stop — the correct Darvas exit methodology.
        #
        #    ROOT CAUSE FIX: Previously we re-detected a new box every 5 minutes.
        #    After a breakout, price consolidates briefly forming a micro-box above
        #    the entry, and any small dip triggers a -2/-3 sell at -0.1%.  This
        #    killed winners before they could run.
        #
        #    FIX: Store the entry box's floor at buy time ("entry_box_bottom").
        #    Use THAT as the trailing stop anchor — not whatever new micro-box just
        #    formed.  Trail the stop UP only when a NEW fully-confirmed Darvas box
        #    forms with a bottom HIGHER than the current trailing stop AND its top
        #    is above the avg entry price (i.e. price has genuinely moved up).
        #
        #    Volume-weighted close confirmation prevents whipsaw exits:
        #      • vol_ratio ≥ 1.5  (high volume)  → 1 close below threshold to sell
        #      • vol_ratio ≥ 0.8  (normal volume) → 2 consecutive closes required
        #      • vol_ratio <  0.8  (low volume)   → 3 consecutive closes required
        #        (low-volume dips below the box are the classic false breakdown)
        entry_ts = portfolio.positions[symbol].get("entry_ts")
        if entry_ts and isinstance(entry_ts, str):
            entry_ts = datetime.fromisoformat(entry_ts)
        # entry_ts is always naive (stored without tzinfo); compare with naive now
        held_minutes = (datetime.now() - entry_ts).total_seconds() / 60 if entry_ts else 999

        if held_minutes >= MIN_HOLD_MINUTES:
            pos                  = portfolio.positions[symbol]
            trailing_box_bottom  = pos.get("trailing_box_bottom")

            if trailing_box_bottom is not None:
                box_details    = box_result.get("details", {})
                new_box_bottom = box_details.get("box_bottom")
                new_box_top    = box_details.get("box_top")
                atr            = box_details.get("atr", current_price * 0.005)
                vol_ratio      = box_details.get("vol_ratio", 1.0)

                # ── Trail stop UP ─────────────────────────────────────────────
                # Only raise the trailing stop when a new, fully-confirmed box
                # has formed whose BOTTOM is above the breakeven price (entry +
                # ~0.4% round-trip fees). This ensures the raised stop only ever
                # locks in real profit — never raises above entry on a micro-box.
                # We NEVER lower the trailing stop.
                breakeven = pos["avg_price"] * 1.004
                if (new_box_bottom and new_box_top
                        and new_box_bottom > trailing_box_bottom
                        and new_box_bottom > breakeven):
                    logger.info(
                        f"Trail stop UP {symbol}: ₹{trailing_box_bottom:.2f}"
                        f" → ₹{new_box_bottom:.2f}  (new box top=₹{new_box_top:.2f})"
                    )
                    pos["trailing_box_bottom"] = new_box_bottom
                    trailing_box_bottom = new_box_bottom

                # ── Breakdown check ───────────────────────────────────────────
                # breakdown_threshold = trailing_box_bottom minus ATR buffer.
                # Price must close MEANINGFULLY below the floor — not just a wick.
                breakdown_threshold = trailing_box_bottom - (atr * ATR_BREAKDOWN_FACTOR)

                if current_price < breakdown_threshold:
                    pos["breakdown_count"] = pos.get("breakdown_count", 0) + 1
                    count = pos["breakdown_count"]

                    # Volume-weighted confirmation threshold:
                    # High volume breakdown is more likely real → sell faster.
                    # Low volume dip is likely a whipsaw → require more patience.
                    if vol_ratio >= VOL_THRESH:
                        required_closes = 1   # strong institutional selling → act now
                    elif vol_ratio >= 0.8:
                        required_closes = 2   # normal volume → wait for 2nd confirmation
                    else:
                        required_closes = 3   # low-volume dip → high whipsaw risk

                    if count >= required_closes:
                        sell_fee_info = groww_fees.calculate_sell_fees(current_price, pos["qty"])
                        fee_note = (
                            f"trailing_stop=₹{trailing_box_bottom:.2f} "
                            f"threshold=₹{breakdown_threshold:.2f} "
                            f"closes_below={count}/{required_closes} "
                            f"vol_ratio={vol_ratio:.2f} "
                            f"atr={atr:.2f} "
                            f"sell_fee=₹{sell_fee_info['total_fees']:.2f}"
                        )
                        signal_info_sell = {"signal": "BOX_TRAIL_STOP", "composite_score": -2}
                        trade = portfolio.sell(symbol, current_price, "box_trail_stop", signal_info_sell)
                        if trade:
                            trade["ai_reasoning"] = f"Darvas trailing stop | {fee_note}"
                            trade["confidence"]   = 0.9
                            _log_trade_csv(trade, portfolio.total_value(current_prices))
                            _emit("trade_executed", trade)
                    else:
                        logger.info(
                            f"Trail stop WARNING {symbol}: {count}/{required_closes} closes "
                            f"below ₹{breakdown_threshold:.2f} "
                            f"(vol_ratio={vol_ratio:.2f}) — waiting for confirmation"
                        )
                else:
                    # Price recovered above breakdown threshold → reset counter.
                    # This is the key anti-whipsaw mechanism: a brief dip that
                    # doesn't persist gets forgiven and we stay in the trade.
                    if pos.get("breakdown_count", 0) > 0:
                        logger.info(
                            f"Reset breakdown_count {symbol}: price ₹{current_price:.2f} "
                            f"recovered above ₹{breakdown_threshold:.2f}"
                        )
                    pos["breakdown_count"] = 0

            else:
                # No entry_box_bottom stored (position opened before this update).
                # Fall back to the original micro-box score logic for safety.
                if box_score <= BOX_SELL_THRESHOLD:
                    qty_pos = pos["qty"]
                    sell_fee_info = groww_fees.calculate_sell_fees(current_price, qty_pos)
                    signal_info_sell = {"signal": "BOX_BREAKDOWN", "composite_score": box_score}
                    trade = portfolio.sell(symbol, current_price, "box_breakdown", signal_info_sell)
                    if trade:
                        trade["ai_reasoning"] = (
                            f"Legacy box breakdown (no entry_box_bottom stored) "
                            f"score={box_score} sell_fee=₹{sell_fee_info['total_fees']:.2f}"
                        )
                        trade["confidence"] = 0.9
                        _log_trade_csv(trade, portfolio.total_value(current_prices))
                        _emit("trade_executed", trade)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: Scan all stocks — box + SMC indicators, pre-filter for AI
    #   Equities: need box score ≥ 2 to pass
    #   Commodity ETFs: always pass (AI decides based on Nifty regime)
    #   NIFTYBEES: captured for regime check, not a buy candidate
    # ══════════════════════════════════════════════════════════════════════
    ai_candidates = []

    # ── Batch LTP pre-screen ──────────────────────────────────────────────────
    # Groww's get_ltp() accepts 50 symbols per call, so 223 symbols = 5 API
    # calls instead of 223.  We use this to filter out stocks that haven't
    # moved meaningfully since the last scan, then only fetch full candle
    # history (one call each) for the filtered shortlist.
    # Always include: held positions, commodity ETFs, NIFTY proxy.
    # For equities: only fetch candles if price moved ≥ LTP_MOVE_THRESHOLD %
    # since last scan OR we have no previous price (first scan of session).
    LTP_MOVE_THRESHOLD = 0.003   # 0.3% move since last scan → fetch candles

    scan_symbols = [s for s in config.STOCK_SYMBOLS if s not in portfolio.positions]
    always_scan  = COMMODITY_ETFS | {NIFTY_PROXY}

    try:
        import groww_live
        batch_ltp = groww_live.fetch_batch_ltp(scan_symbols)
    except Exception as e:
        logger.warning(f"Batch LTP pre-screen failed, scanning all symbols: {e}")
        batch_ltp = {}

    # Determine which symbols to fetch full candles for
    if batch_ltp:
        to_fetch = []
        for sym in scan_symbols:
            if sym in always_scan:
                to_fetch.append(sym)
                continue
            ltp = batch_ltp.get(sym)
            if ltp is None:
                to_fetch.append(sym)   # unknown price → include to be safe
                continue
            prev = _prev_ltp.get(sym)
            if prev is None or abs(ltp - prev) / prev >= LTP_MOVE_THRESHOLD:
                to_fetch.append(sym)
        # Update cache for next scan
        _prev_ltp.update(batch_ltp)
        logger.info(f"Batch LTP: {len(batch_ltp)}/{len(scan_symbols)} fetched, "
                    f"{len(to_fetch)} symbols selected for candle fetch "
                    f"(skipped {len(scan_symbols)-len(to_fetch)} unchanged)")
    else:
        to_fetch = scan_symbols   # fallback: scan everything

    for symbol in to_fetch:
        scan_count += 1
        df = data_fetcher.fetch_data(symbol)
        if df is None:
            error_count += 1
            continue

        current_price = _safe_price(df)
        current_prices[symbol] = current_price

        # Capture NIFTYBEES for regime check — not a buy candidate
        if symbol == NIFTY_PROXY:
            _niftybees_df = df
            continue

        box_result   = box_analyze(df)
        is_commodity = symbol in COMMODITY_ETFS

        # Pre-filter: equities need a box breakout; commodity ETFs always pass
        if config.AI_PRE_FILTER and box_result.get("score", 0) < 2 and not is_commodity:
            continue

        smc_result = smc_analyze(df)
        ai_candidates.append({
            "symbol": symbol,
            "price": current_price,
            "indicators": {"day_box": box_result, "smc": smc_result},
            "recent_candles": _format_recent_candles(df),
        })

    eq_count  = sum(1 for c in ai_candidates if c["symbol"] not in COMMODITY_ETFS)
    etf_count = sum(1 for c in ai_candidates if c["symbol"] in COMMODITY_ETFS)
    logger.info(f"Pre-filtered {len(ai_candidates)} candidates ({eq_count} equity + {etf_count} ETF) from {scan_count} scanned")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 3: Compute Nifty regime + ask AI to pick best buys
    #   Buys only fire at the 4 AI eval windows — mirrors backtest exactly.
    #   Box auto-sells still happen every scan; AI rebuy happens at next window.
    # ══════════════════════════════════════════════════════════════════════
    if ai_eval_fires and ai_candidates and portfolio.can_buy(1):  # at least some cash
        portfolio_state = _build_portfolio_state(portfolio, current_prices)

        # Reuse Nifty context already computed for the sell phase above
        nifty_context = _nifty_context_cache

        # Send to AI in batch (1 API call for all candidates)
        ai_result = ai_brain.analyze_batch(
            candidates=[{
                "symbol": c["symbol"],
                "price": c["price"],
                "indicators": c["indicators"],
                "recent_candles": c["recent_candles"],
            } for c in ai_candidates],
            portfolio_state=portfolio_state,
            current_time_str=current_time_str,
            nifty_context=nifty_context,
        )
        ai_picks = ai_result.get("picks", ai_result) if isinstance(ai_result, dict) else ai_result

        # Execute AI's buy recommendations
        for pick in ai_picks:
            symbol = pick.get("symbol", "")
            confidence = pick.get("confidence", 0)
            position_size_pct = pick.get("position_size_pct", config.MAX_POSITION_PCT)

            if confidence < 0.7:
                logger.info(f"Skipping {symbol}: AI confidence too low ({confidence:.0%})")
                continue

            if symbol not in current_prices:
                continue

            price = current_prices[symbol]
            if not portfolio.can_buy(price):
                logger.info(f"Cannot buy {symbol}: max positions or insufficient cash")
                break

            # Use AI's suggested position size
            portfolio_value = portfolio.total_value(current_prices)
            allocation = portfolio_value * min(position_size_pct, config.MAX_POSITION_PCT)
            qty = int(min(allocation, portfolio.cash) // price)
            if qty <= 0:
                continue

            # Extract the entry box bottom from the candidate's Darvas analysis.
            # This is the box floor at the moment AI selected this breakout.
            # Stored on the position and used as the trailing stop anchor.
            cand = next((c for c in ai_candidates if c["symbol"] == symbol), None)
            entry_box_bottom = (
                cand["indicators"].get("day_box", {}).get("details", {}).get("box_bottom")
                if cand else None
            )

            signal_info = {
                "signal": "AI_BUY",
                "composite_score": confidence,
            }
            trade = portfolio.buy(symbol, price, qty, signal_info, entry_box_bottom=entry_box_bottom)
            if trade:
                trade["ai_reasoning"] = pick.get("reasoning", "")
                trade["confidence"] = confidence
                _log_trade_csv(trade, portfolio.total_value(current_prices))
                _emit("trade_executed", trade)

    # Mark AI eval as fired so next one waits 1 hour
    if ai_eval_fires:
        _mark_ai_eval_fired(now)
        _emit("ai_eval_fired", get_ai_eval_info())

    # ══════════════════════════════════════════════════════════════════════
    # Phase 4: Save state and log summary
    # ══════════════════════════════════════════════════════════════════════
    portfolio.save()

    summary = portfolio.summary(current_prices)
    logger.info(f"\n{summary}")
    scan_summary = {
        "stocks_scanned": scan_count,
        "errors": error_count,
        "ai_candidates": len(ai_candidates),
        "timestamp": datetime.now(IST).isoformat(),
    }
    logger.info(
        f"Scan complete: {scan_count} stocks scanned, {error_count} errors, "
        f"{len(ai_candidates)} sent to AI"
    )
    _emit("scan_update", scan_summary)
    _emit("portfolio_update", portfolio.to_dict(current_prices))


def _format_recent_candles(df, n: int = 10) -> str:
    """Format last N candles as compact text."""
    recent = df.tail(n)
    lines = []
    for idx, row in recent.iterrows():
        ts = idx.strftime("%H:%M") if hasattr(idx, "strftime") else str(idx)
        lines.append(
            f"{ts}|O:{row['Open']:.1f}|H:{row['High']:.1f}|"
            f"L:{row['Low']:.1f}|C:{row['Close']:.1f}|V:{int(row['Volume'])}"
        )
    return "; ".join(lines)
