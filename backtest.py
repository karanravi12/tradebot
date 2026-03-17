"""Hybrid backtest — Week of Mon 23 Feb to Fri 27 Feb 2026  |  Capital: ₹1,00,000.

Strategy: Box Theory does the fast work; AI makes the big calls.

AI decision points (3 per day):
  09:30 — Market open: AI batch-picks buys from box breakout candidates
  11:30 — Mid-session: AI evaluates each held position (hold or sell?)
  13:30 — Afternoon:   AI evaluates each held position (hold or sell?)
  15:00 — Pre-close:   AI evaluates each held position (hold overnight?)

Box Theory handles everything in between (every 5-min candle):
  - Hard stop-loss (3%)       → instant sell, no AI
  - Box breakdown (score ≤-2) → instant sell, no AI (fast algorithmic)
  - Overnight gap check       → instant sell/skip, no AI

No new buys outside AI decision windows.
Positions carry overnight (no forced EOD close except Friday).
Winners run freely — no take-profit cap; AI decides when to exit.

Run:  python backtest.py
"""

import logging
import os
from datetime import date, timedelta

import pandas as pd
import pytz

# ── Load .env ─────────────────────────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

import config
import data_fetcher
import fees as groww_fees
import ai_brain
from strategies.day_box import (
    analyze as box_analyze,
    ATR_BREAKDOWN_FACTOR,   # how far below box_bottom counts as a real breakdown
    VOL_THRESH,             # vol_ratio ≥ 1.5 → high-volume confirmation (1 close)
)
from strategies.luxalgo import analyze as smc_analyze

config.API_DAILY_LIMIT_USD = 0   # no cap for backtest

# ── Mode toggle ───────────────────────────────────────────────────────────────
# True  → pure Darvas Box only, no AI calls, buys at every 5-min candle
# False → hybrid (AI windows for buys + AI eval for exits)
BOX_ONLY = False
NO_OVERNIGHT = False  # Force-close all positions at EOD (no carrying overnight)

logging.basicConfig(level=logging.WARNING)

IST = pytz.timezone("Asia/Kolkata")

BACKTEST_DATES = [
    date(2026, 3, 16),   # Monday (today)
]
LAST_DATE = BACKTEST_DATES[-1]

# ── Thresholds ────────────────────────────────────────────────────────────────
BUY_SCORE_THRESHOLD  = 2       # box breakout_up (score 2–3)
SELL_SCORE_THRESHOLD = -2      # box breakdown (score ≤ -2) — box auto-sell
MIN_HOLD_CANDLES     = 3       # hold ≥ 15 min before any box-triggered sell
OVERNIGHT_STOP_MULT  = 1.5     # overnight gap stop = STOP_LOSS_PCT × 1.5

# ── AI decision schedule (hour, minute) ──────────────────────────────────────
# AI makes buy decisions at open and evaluates positions at each checkpoint.
AI_BUY_WINDOWS  = {(9, 30), (11, 30), (13, 30)}          # batch buy calls
AI_EVAL_WINDOWS = {(9, 30), (11, 30), (13, 30), (15, 0)} # hold/sell evaluations

# Tolerance: snap the nearest 5-min candle within ±2 candles of each window
AI_WINDOW_TOLERANCE_MIN = 10  # ±10 min (covers 09:20–09:40 for 09:30 window)

# ── Portfolio state (persists across days) ────────────────────────────────────
cash          = float(config.INITIAL_CAPITAL)
positions     = {}      # symbol → {qty, avg_price, last_price, entry_time, entry_ts}
trade_history = []
total_fees    = 0.0
day_summaries = []

# Track which AI windows have already fired today
_ai_buy_done  = set()   # (date, hour, minute) tuples
_ai_eval_done = set()   # (date, hour, minute) tuples

# ── Commodity hedge & Nifty proxy ─────────────────────────────────────────────
# These ETFs are ALWAYS included as buy candidates so the AI can rotate into
# them when the Nifty regime turns bearish (Option B hedge).
COMMODITY_ETFS = {"GOLDBEES", "SILVERBEES", "AXISGOLD",
                  "HDFCGOLD", "QGOLDHALF"}
NIFTY_PROXY = "NIFTYBEES"   # Nifty 50 proxy for market regime

# ── Box drawing helpers ───────────────────────────────────────────────────────
W = 72
_silent = False

def _nout(*a, **k): pass
def box_top():    (_nout if _silent else print)("╔" + "═" * (W - 2) + "╗")
def box_mid():    (_nout if _silent else print)("╠" + "═" * (W - 2) + "╣")
def box_bot():    (_nout if _silent else print)("╚" + "═" * (W - 2) + "╝")
def box_row(txt): (_nout if _silent else print)(f"║  {txt:<{W-4}}║")
def box_sep():    (_nout if _silent else print)("╟" + "─" * (W - 2) + "╢")

_DAY = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]


def _safe_float(val):
    if isinstance(val, pd.Series):
        return float(val.iloc[0])
    return float(val)


def _composite(indicators: dict) -> float:
    return float(indicators.get("day_box", {}).get("score", 0))


def _buy(symbol, price, scan_time, indicators):
    global cash, total_fees
    if len(positions) >= config.MAX_POSITIONS or symbol in positions:
        return None

    portfolio_value = cash + sum(p["qty"] * p["last_price"] for p in positions.values())
    allocation  = min(portfolio_value * config.MAX_POSITION_PCT, cash * 0.98)
    fee_est     = max(20, allocation * 0.001) + allocation * 0.0001
    qty         = int((allocation - fee_est) // price)
    if qty <= 0:
        return None

    fee_info      = groww_fees.calculate_buy_fees(price, qty)
    total_outflow = price * qty + fee_info["total_fees"]
    if total_outflow > cash:
        qty = int((cash * 0.998) // price)
        if qty <= 0:
            return None
        fee_info      = groww_fees.calculate_buy_fees(price, qty)
        total_outflow = price * qty + fee_info["total_fees"]

    cash       -= total_outflow
    total_fees += fee_info["total_fees"]
    composite   = _composite(indicators)
    time_str    = scan_time.strftime("%H:%M")

    # Store entry box bottom for Darvas trailing stop
    entry_box_bottom = indicators.get("day_box", {}).get("details", {}).get("box_bottom")

    positions[symbol] = {
        "qty": qty, "avg_price": price, "last_price": price,
        "entry_time": time_str, "entry_ts": scan_time,
        "composite_at_entry": round(composite, 2),
        # Darvas trailing stop fields
        "entry_box_bottom":    entry_box_bottom,  # anchor: box floor at breakout
        "trailing_box_bottom": entry_box_bottom,  # trails UP as new boxes form above entry
        "breakdown_count":     0,                 # consecutive closes below breakdown_threshold
    }
    trade = {
        "date": scan_time.strftime("%a %d/%m"), "time": time_str,
        "action": "BUY", "symbol": symbol,
        "qty": qty, "price": round(price, 2),
        "cost": round(price * qty, 2),
        "fees": round(fee_info["total_fees"], 2),
        "composite": round(composite, 2),
        "cash_after": round(cash, 2),
    }
    trade_history.append(trade)
    return trade


def _sell(symbol, price, scan_time, reason, indicators):
    global cash, total_fees
    if symbol not in positions:
        return None

    pos         = positions.pop(symbol)
    qty         = pos["qty"]
    avg_price   = pos["avg_price"]
    fee_info    = groww_fees.calculate_sell_fees(price, qty)
    net_revenue = price * qty - fee_info["total_fees"]
    pnl         = net_revenue - avg_price * qty
    pnl_pct     = (pnl / (avg_price * qty) * 100) if avg_price > 0 else 0
    time_str    = scan_time.strftime("%H:%M")

    cash       += net_revenue
    total_fees += fee_info["total_fees"]

    trade = {
        "date": scan_time.strftime("%a %d/%m"), "time": time_str,
        "action": "SELL", "symbol": symbol,
        "qty": qty, "price": round(price, 2),
        "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
        "fees": round(fee_info["total_fees"], 2),
        "reason": reason, "composite": round(_composite(indicators), 2),
        "cash_after": round(cash, 2),
        "hold_from": pos["entry_time"],
    }
    trade_history.append(trade)
    return trade


def _hold_candles(symbol, scan_time) -> int:
    entry_ts = positions[symbol].get("entry_ts")
    if entry_ts is None:
        return 99
    return int((scan_time - entry_ts).total_seconds() / 300)


def _fmt_candles(df_slice, n: int = 10) -> str:
    recent = df_slice.tail(n)
    lines  = ["Time|Open|High|Low|Close|Volume"]
    for idx, row in recent.iterrows():
        ts = idx.strftime("%H:%M") if hasattr(idx, "strftime") else str(idx)
        lines.append(
            f"{ts}|{row['Open']:.1f}|{row['High']:.1f}|"
            f"{row['Low']:.1f}|{row['Close']:.1f}|{int(row['Volume'])}"
        )
    return "; ".join(lines)


def _ai_portfolio_state(symbol=None) -> dict:
    state = {
        "cash":            cash,
        "num_positions":   len(positions),
        "portfolio_value": cash + sum(p["qty"] * p["last_price"] for p in positions.values()),
        "realized_pnl":    sum(t["pnl"] for t in trade_history if t["action"] == "SELL"),
        "held_symbols":    list(positions.keys()),
    }
    if symbol and symbol in positions:
        state["position"] = positions[symbol]
    return state


def _nearest_window(scan_time, window_set):
    """Return the matching (h, m) window if scan_time is within tolerance, else None.
    Converts to IST first — critical since data_fetcher returns UTC.
    """
    ts = scan_time
    if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
        try:
            ts = ts.tz_convert(IST) if hasattr(ts, "tz_convert") else ts.astimezone(IST)
        except Exception as e:
            logger.warning(f"_nearest_window: timezone conversion failed for {ts}: {e}")
    sh, sm = int(ts.hour), int(ts.minute)
    total_mins = sh * 60 + sm
    for wh, wm in sorted(window_set):
        w_mins = wh * 60 + wm
        if abs(total_mins - w_mins) <= AI_WINDOW_TOLERANCE_MIN:
            return (wh, wm)
    return None


def run_backtest(
    start_date: date,
    end_date: date,
    silent: bool = True,
    emit_fn=None,
) -> dict:
    """Run backtest for the given date range. Returns dict with trades, summary, equity_curve.

    When emit_fn(msg) is provided, each print/line is streamed to it in real time.
    """
    global BACKTEST_DATES, LAST_DATE, cash, positions, trade_history, total_fees, day_summaries
    global _ai_buy_done, _ai_eval_done, _silent

    dates = []
    d = start_date
    while d <= end_date:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    if not dates:
        return {"error": "No weekdays in date range", "trades": [], "summary": {}, "equity_curve": []}

    BACKTEST_DATES = dates
    LAST_DATE = dates[-1]
    cash = float(config.INITIAL_CAPITAL)
    positions.clear()
    trade_history.clear()
    total_fees = 0.0
    day_summaries.clear()
    _ai_buy_done.clear()
    _ai_eval_done.clear()
    _silent = silent and not emit_fn

    import sys
    if emit_fn:
        class _EmitWriter:
            def __init__(self, fn):
                self.fn = fn
                self.buf = ""
            def write(self, s):
                self.buf += s
                while "\n" in self.buf or "\r" in self.buf:
                    i = min(
                        self.buf.find("\n") if "\n" in self.buf else 999,
                        self.buf.find("\r") if "\r" in self.buf else 999,
                    )
                    line = self.buf[:i].strip()
                    self.buf = self.buf[i + 1:].lstrip("\r\n")
                    if line:
                        self.fn(line)
            def flush(self):
                pass
        _old_stdout = sys.stdout
        sys.stdout = _EmitWriter(emit_fn)
        try:
            result = _execute_backtest()
        finally:
            sys.stdout = _old_stdout
        return result

    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return _execute_backtest()


def _execute_backtest() -> dict:
    """Execute the backtest using global BACKTEST_DATES. Returns JSON-serializable results."""
    global cash, positions, trade_history, total_fees, day_summaries, _ai_buy_done, _ai_eval_done

# ── Load all stock data once (covers full week) ───────────────────────────────
    box_top()
    _mode_label = "BOX THEORY ONLY" if BOX_ONLY else "HYBRID (AI windows + Box auto)"
    _dates_label = f"{BACKTEST_DATES[0]} → {BACKTEST_DATES[-1]}" if len(BACKTEST_DATES) > 1 else str(BACKTEST_DATES[0])
    box_row(f"BACKTEST  —  {_dates_label}  —  {_mode_label}")
    box_row("Strategy: Strict Darvas Box  " + ("(no AI)" if BOX_ONLY else "+ Claude Sonnet 4.6 (AI windows only)"))
    box_mid()
    box_row(f"Starting Capital: ₹{config.INITIAL_CAPITAL:,.0f}  |  Positions carried overnight")
    if not BOX_ONLY:
        box_row(f"AI Windows: 09:30 (buy+eval)  11:30 (eval)  13:30 (eval)  15:00 (pre-close eval)")
    box_row(f"Exit: Darvas trailing stop (entry box floor + ATR buffer, N-close vol-confirmed)  |  hard stop-loss {config.STOP_LOSS_PCT*100:.0f}%  |  winners run")
    box_row(f"Strict Darvas: 3-bar confirmed box top + 0.2% breakout buffer")
    box_bot()
    print()

    print(f"Loading {len(config.STOCK_SYMBOLS)} stocks...", end="", flush=True)

    stock_data = {}
    week_start = BACKTEST_DATES[0]
    week_end   = BACKTEST_DATES[-1]
    for sym in config.STOCK_SYMBOLS:
        df = data_fetcher.fetch_intraday(sym, start_date=week_start, end_date=week_end)
        if df is None or df.empty:
            continue
        df_ist = df.copy()
        if df_ist.index.tzinfo is None:
            df_ist.index = df_ist.index.tz_localize("UTC").tz_convert(IST)
        else:
            df_ist.index = df_ist.index.tz_convert(IST)
        df_week = df_ist[
            (df_ist.index.date >= week_start) &
            (df_ist.index.date <= week_end)
        ]
        if len(df_week) >= 20:
            stock_data[sym] = df_week

    print(f" {len(stock_data)} stocks with week data.\n")

    # ── Day-by-day loop ───────────────────────────────────────────────────────────
    for day_idx, backtest_date in enumerate(BACKTEST_DATES):
        day_name    = _DAY[backtest_date.weekday()]
        is_last_day = (backtest_date == LAST_DATE)

        _ai_buy_done.clear()
        _ai_eval_done.clear()

        day_times = sorted(set(
            ts for df in stock_data.values() for ts in df.index
            if ts.date() == backtest_date
            and ts.time() >= pd.Timestamp("09:30").time()
            and ts.time() <= pd.Timestamp("15:25").time()
        ))

        if not day_times:
            print(f"  No data for {day_name} {backtest_date} — skipping\n")
            continue

        cash_at_open    = cash
        trades_today    = len(trade_history)
        portfolio_open  = cash + sum(p["qty"] * p["last_price"] for p in positions.values())

        print(f"╔{'═'*70}╗")
        held_str = f"{len(positions)} position(s) carried in" if positions else "flat"
        print(f"║  {day_name:9}  {backtest_date}  │  Cash: ₹{cash:>10,.0f}  │  {held_str:<26}║")
        print(f"╚{'═'*70}╝")
        print(f"  {'Time':>5}  {'Who':<4}  {'Act':<4}  {'Symbol':<12}  {'Price':>8}  {'Qty':>5}  {'P&L':>10}  Note")
        print("  " + "─" * 67)

        # ── Overnight gap check ────────────────────────────────────────────────
        if positions and day_times:
            first_scan    = day_times[0]
            overnight_stop = config.STOP_LOSS_PCT * OVERNIGHT_STOP_MULT
            for symbol in list(positions.keys()):
                if symbol not in stock_data:
                    continue
                df_slice   = stock_data[symbol].loc[:first_scan]
                if df_slice.empty:
                    continue
                open_price = _safe_float(df_slice["Open"].iloc[-1])
                avg_price  = positions[symbol]["avg_price"]
                positions[symbol]["last_price"] = open_price

                gap_loss = (avg_price - open_price) / avg_price if avg_price > 0 else 0

                ind = {"day_box": box_analyze(df_slice)}
                if gap_loss >= overnight_stop:
                    t = _sell(symbol, open_price, first_scan, "overnight_gap_down", ind)
                    if t:
                        print(f"  {first_scan.strftime('%H:%M')}  BOX   SELL {symbol:<12} ₹{t['price']:>8.2f}  "
                              f"{t['qty']:>5}  ₹{t['pnl']:>+8.2f}  🌙 GAP DOWN ({t['pnl_pct']:+.1f}%)")

        # ── Main scan loop ─────────────────────────────────────────────────────
        for candle_idx, scan_time in enumerate(day_times):
            time_str      = scan_time.strftime("%H:%M")
            is_near_close = scan_time.time() >= pd.Timestamp("15:15").time()

            # Check if this candle falls in an AI window
            eval_window = _nearest_window(scan_time, AI_EVAL_WINDOWS)
            buy_window  = _nearest_window(scan_time, AI_BUY_WINDOWS)

            eval_key = (backtest_date, eval_window) if eval_window else None
            buy_key  = (backtest_date, buy_window)  if buy_window  else None

            ai_eval_fires = eval_key and eval_key not in _ai_eval_done
            ai_buy_fires  = buy_key  and buy_key  not in _ai_buy_done and not is_near_close
            # Fallback: if _nearest_window returned None (tz mismatch?), fire at first 3 candles
            if not BOX_ONLY and not ai_buy_fires and not is_near_close:
                if candle_idx < 3 and (backtest_date, (9, 30)) not in _ai_buy_done:
                    buy_key = (backtest_date, (9, 30))
                    ai_buy_fires = True

            # ══════════════════════════════════════════════════════════════════
            # Phase 1: Manage existing positions
            # ══════════════════════════════════════════════════════════════════
            for symbol in list(positions.keys()):
                if symbol not in stock_data:
                    continue
                df_slice = stock_data[symbol].loc[:scan_time]
                if df_slice.empty:
                    continue

                current_price = _safe_float(df_slice["Close"].iloc[-1])
                positions[symbol]["last_price"] = current_price
                avg_price = positions[symbol]["avg_price"]
                ind       = {"day_box": box_analyze(df_slice)}
                if not BOX_ONLY and ai_eval_fires:
                    ind["smc"] = smc_analyze(df_slice)

                # ── Hard stop-loss (BOX, no AI) ────────────────────────────
                loss_pct = (avg_price - current_price) / avg_price if avg_price > 0 else 0
                if loss_pct >= config.STOP_LOSS_PCT:
                    t = _sell(symbol, current_price, scan_time, "stop_loss", ind)
                    if t:
                        print(f"  {time_str}  BOX   SELL {symbol:<12} ₹{t['price']:>8.2f}  "
                              f"{t['qty']:>5}  ₹{t['pnl']:>+8.2f}  ⛔ STOP-LOSS ({t['pnl_pct']:+.1f}%)")
                    continue

                # ── Friday week-end force-close (BOX, no AI) ───────────────
                if is_last_day and is_near_close:
                    t = _sell(symbol, current_price, scan_time, "week_end_close", ind)
                    if t:
                        print(f"  {time_str}  BOX   SELL {symbol:<12} ₹{t['price']:>8.2f}  "
                              f"{t['qty']:>5}  ₹{t['pnl']:>+8.2f}  🏁 WEEK-END ({t['pnl_pct']:+.1f}%)")
                    continue

                # ── AI evaluation window — ask Claude (hold or sell?) ───────
                if not BOX_ONLY and ai_eval_fires and symbol in positions:
                    ct_str = scan_time.strftime("%Y-%m-%d %H:%M IST")
                    try:
                        ai_dec = ai_brain.analyze_single(
                            symbol=symbol, df=df_slice, indicators=ind,
                            portfolio_state=_ai_portfolio_state(symbol),
                            current_time_str=ct_str,
                        )
                    except Exception as ex:
                        print(f"  {time_str}  🤖AI  HOLD {symbol:<12}  [AI ERROR: {ex}]")
                        ai_dec = {"action": "HOLD", "confidence": 0, "reasoning": str(ex)}
                    action = ai_dec.get("action", "HOLD")
                    conf   = ai_dec.get("confidence", 0)
                    if action == "SELL" and conf >= 0.65:
                        t = _sell(symbol, current_price, scan_time, "ai_eval", ind)
                        if t:
                            print(f"  {time_str}  🤖AI  SELL {symbol:<12} ₹{t['price']:>8.2f}  "
                                  f"{t['qty']:>5}  ₹{t['pnl']:>+8.2f}  AI conf={conf:.0%} ({t['pnl_pct']:+.1f}%)")
                    else:
                        reason = ai_dec.get("reasoning", "")[:60]
                        print(f"  {time_str}  🤖AI  HOLD {symbol:<12}  conf={conf:.0%} — {reason}")
                    continue

                # ── Darvas Trailing Stop (every candle, after MIN_HOLD_CANDLES) ──
                if _hold_candles(symbol, scan_time) < MIN_HOLD_CANDLES:
                    continue

                pos                 = positions[symbol]
                trailing_box_bottom = pos.get("trailing_box_bottom")

                if trailing_box_bottom is None:
                    composite = _composite(ind)
                    if composite <= SELL_SCORE_THRESHOLD:
                        t = _sell(symbol, current_price, scan_time, "box_signal", ind)
                        if t:
                            print(f"  {time_str}  BOX   SELL {symbol:<12} ₹{t['price']:>8.2f}  "
                                  f"{t['qty']:>5}  ₹{t['pnl']:>+8.2f}  "
                                  f"📉 legacy_box={composite:.0f} ({t['pnl_pct']:+.1f}%)")
                    continue

                box_details    = ind["day_box"].get("details", {})
                new_box_bottom = box_details.get("box_bottom")
                new_box_top    = box_details.get("box_top")
                atr            = box_details.get("atr", current_price * 0.005)
                vol_ratio      = box_details.get("vol_ratio", 1.0)

                breakeven = pos["avg_price"] * 1.004
                if (new_box_bottom and new_box_top
                        and new_box_bottom > trailing_box_bottom
                        and new_box_bottom > breakeven):
                    pos["trailing_box_bottom"] = new_box_bottom
                    trailing_box_bottom = new_box_bottom

                breakdown_threshold = trailing_box_bottom - (atr * ATR_BREAKDOWN_FACTOR)

                if current_price < breakdown_threshold:
                    pos["breakdown_count"] = pos.get("breakdown_count", 0) + 1
                    count = pos["breakdown_count"]
                    if vol_ratio >= VOL_THRESH:
                        required_closes = 1
                    elif vol_ratio >= 0.8:
                        required_closes = 2
                    else:
                        required_closes = 3
                    if count >= required_closes:
                        t = _sell(symbol, current_price, scan_time, "box_trail_stop", ind)
                        if t:
                            print(f"  {time_str}  BOX   SELL {symbol:<12} ₹{t['price']:>8.2f}  "
                                  f"{t['qty']:>5}  ₹{t['pnl']:>+8.2f}  "
                                  f"📉 trail=₹{trailing_box_bottom:.0f} "
                                  f"({count}cl vol={vol_ratio:.1f}x) ({t['pnl_pct']:+.1f}%)")
                else:
                    pos["breakdown_count"] = 0

            # Mark eval window as done after processing all positions
        if not BOX_ONLY and ai_eval_fires and eval_key:
            _ai_eval_done.add(eval_key)

        # ══════════════════════════════════════════════════════════════════
        # Phase 2: Buy entries
        #   BOX_ONLY → buy any candle on box breakout signal
        #   HYBRID   → AI buy windows only; Claude picks from box hits
        # ══════════════════════════════════════════════════════════════════
        if not BOX_ONLY and (not ai_buy_fires or len(positions) >= config.MAX_POSITIONS):
            continue
        if BOX_ONLY and (is_near_close or len(positions) >= config.MAX_POSITIONS):
            continue

        # ── Build candidate list ────────────────────────────────────────────
        # Equities: include only if Darvas box breakout (score ≥ BUY_SCORE_THRESHOLD)
        # Commodity ETFs: ALWAYS include so AI can choose hedge rotation
        box_hits = []
        for symbol, full_df in stock_data.items():
            if symbol in positions:
                continue
            df_slice = full_df.loc[:scan_time]
            if len(df_slice) < 25:
                continue

            box_result = box_analyze(df_slice)
            composite  = float(box_result.get("score", 0))
            is_commodity = symbol in COMMODITY_ETFS

            if composite < BUY_SCORE_THRESHOLD and not is_commodity:
                continue  # equities need a box breakout; commodities always pass

            price      = _safe_float(df_slice["Close"].iloc[-1])
            ind        = {"day_box": box_result, "smc": smc_analyze(df_slice)}
            box_hits.append({
                "symbol":    symbol,
                "price":     price,
                "composite": composite,
                "ind":       ind,
                "df_slice":  df_slice,
            })

        # ── Compute Nifty market regime from NIFTYBEES ──────────────────────
        nifty_context = None
        if NIFTY_PROXY in stock_data:
            nifty_df = stock_data[NIFTY_PROXY].loc[:scan_time]
            if len(nifty_df) >= 25:
                n_box = box_analyze(nifty_df)
                n_smc = smc_analyze(nifty_df)
                n_box_score = int(n_box.get("score", 0))
                n_smc_score = int(n_smc.get("score", 0))
                combined    = n_box_score + n_smc_score
                if combined >= 2:
                    regime      = "BULLISH"
                    regime_note = "Nifty trending up — equity breakouts favoured"
                elif combined <= -2:
                    regime      = "BEARISH"
                    regime_note = "Nifty in downtrend — DEFENSIVE MODE: prefer GOLDBEES/SILVERBEES"
                else:
                    regime      = "NEUTRAL"
                    regime_note = "Nifty sideways — be selective; high-conviction setups only"
                nifty_context = {
                    "proxy":          NIFTY_PROXY,
                    "market_regime":  regime,
                    "regime_note":    regime_note,
                    "box_score":      n_box_score,
                    "box_zone":       n_box.get("details", {}).get("zone", "unknown"),
                    "smc_score":      n_smc_score,
                    "smc_bias":       n_smc.get("smc_bias", "neutral"),
                    "nifty_trend":    n_smc.get("details", {}).get("trend", "unknown"),
                    "nifty_pd_zone":  n_smc.get("details", {}).get("pd_zone", "equilibrium"),
                }

        if BOX_ONLY:
            # Pure box: buy all equity hits immediately (no AI, no commodities)
            equity_hits = [c for c in box_hits if c["symbol"] not in COMMODITY_ETFS]
            for cand in sorted(equity_hits, key=lambda x: x["composite"], reverse=True):
                if len(positions) >= config.MAX_POSITIONS:
                    break
                sym = cand["symbol"]
                t = _buy(sym, cand["price"], scan_time, cand["ind"])
                if t:
                    print(f"  {time_str}  BOX   BUY  {sym:<12} ₹{t['price']:>8.2f}  "
                          f"{t['qty']:>5}  {'':>10}  box={cand['composite']:.0f}  fees=₹{t['fees']:.2f}")
        else:
            # Hybrid: AI picks from box hits + commodity ETFs + Nifty regime context
            if box_hits:
                ct_str   = scan_time.strftime("%Y-%m-%d %H:%M IST")
                regime_str = nifty_context["market_regime"] if nifty_context else "?"
                print(f"  {time_str}  🤖AI  ---  {len(box_hits)} candidates "
                      f"({sum(1 for c in box_hits if c['symbol'] not in COMMODITY_ETFS)} eq + "
                      f"{sum(1 for c in box_hits if c['symbol'] in COMMODITY_ETFS)} ETF)  "
                      f"Nifty={regime_str}")
                try:
                    ai_result = ai_brain.analyze_batch(
                        candidates=[{
                            "symbol":        c["symbol"],
                            "price":         c["price"],
                            "indicators":    c["ind"],
                            "recent_candles": _fmt_candles(c["df_slice"]),
                        } for c in box_hits],
                        portfolio_state=_ai_portfolio_state(),
                        current_time_str=ct_str,
                        nifty_context=nifty_context,
                    )
                    ai_picks = ai_result.get("picks", ai_result) if isinstance(ai_result, dict) else ai_result
                    ai_reasoning = ai_result.get("reasoning", "") if isinstance(ai_result, dict) else ""
                except Exception as ex:
                    print(f"  {time_str}  🤖AI  ---  [AI BATCH ERROR: {ex}]")
                    ai_picks = []
                    ai_reasoning = str(ex)
                buys_done = 0
                for pick in sorted(ai_picks, key=lambda x: x.get("confidence", 0), reverse=True):
                    sym      = pick.get("symbol", "")
                    conf     = pick.get("confidence", 0)
                    t_type   = pick.get("trade_type", "equity")
                    if conf < 0.70 or len(positions) >= config.MAX_POSITIONS:
                        continue
                    cand = next((c for c in box_hits if c["symbol"] == sym), None)
                    if not cand:
                        continue
                    t = _buy(sym, cand["price"], scan_time, cand["ind"])
                    if t:
                        buys_done += 1
                        hedge_tag = " 🛡️HEDGE" if sym in COMMODITY_ETFS else ""
                        print(f"  {time_str}  🤖AI  BUY  {sym:<12} ₹{t['price']:>8.2f}  "
                              f"{t['qty']:>5}  {'':>10}  conf={conf:.0%} box={cand['composite']:.0f}  "
                              f"type={t_type}{hedge_tag}  fees=₹{t['fees']:.2f}")
                if buys_done == 0 and ai_reasoning:
                    print(f"  {time_str}  🤖AI  ---  No buy: {ai_reasoning[:120]}")
            else:
                print(f"  {time_str}  🤖AI  ---  No box breakouts + no commodities to buy")

            if buy_key:
                _ai_buy_done.add(buy_key)

        # ── Force-close all positions at EOD (no overnight carry) ──────────────
        if NO_OVERNIGHT and positions:
            eod_time = day_times[-1] if day_times else None
            for sym in list(positions.keys()):
                eod_price = positions[sym]["last_price"]
                t = _sell(sym, eod_price, eod_time or scan_time, "eod_close", {})
                if t:
                    print(f"  {t['time']}  EOD   SELL {sym:<12} ₹{t['price']:>8.2f}  "
                          f"{t['qty']:>5}  ₹{t['pnl']:>+10,.2f}  forced close")

        # ── End of day summary ─────────────────────────────────────────────────
        today_trades    = trade_history[trades_today:]
        today_sells     = [t for t in today_trades if t["action"] == "SELL"]
        day_pnl         = sum(t["pnl"] for t in today_sells)
        portfolio_close = cash + sum(p["qty"] * p["last_price"] for p in positions.values())
        overnight_held  = list(positions.keys()) if positions else []

        print()
        print(f"  ── {day_name} EOD ──────────────────────────────────────────────────────────")
        print(f"  Cash: ₹{cash:>10,.2f}  │  Day P&L: ₹{day_pnl:>+8,.2f}  │  "
              f"Portfolio: ₹{portfolio_close:>10,.2f}")
        if overnight_held:
            print(f"  Carrying overnight: {', '.join(overnight_held)}")
        print()

        day_summaries.append({
            "date":      backtest_date,
            "day":       day_name,
            "trades":    len(today_trades),
            "pnl":       round(day_pnl, 2),
            "portfolio": round(portfolio_close, 2),
            "overnight": overnight_held[:],
        })

    # ── Final report ──────────────────────────────────────────────────────────────
    _dates_label = f"{BACKTEST_DATES[0]} → {BACKTEST_DATES[-1]}" if len(BACKTEST_DATES) > 1 else str(BACKTEST_DATES[0])
    buys    = [t for t in trade_history if t["action"] == "BUY"]
    sells   = [t for t in trade_history if t["action"] == "SELL"]
    winners = [t for t in sells if t["pnl"] > 0]
    losers  = [t for t in sells if t["pnl"] <= 0]
    sl_sells   = [t for t in sells if t.get("reason") == "stop_loss"]
    ai_sells   = [t for t in sells if t.get("reason") == "ai_eval"]
    box_sells  = [t for t in sells if t.get("reason") in ("box_trail_stop", "box_signal")]
    total_pnl  = sum(t["pnl"] for t in sells)
    pnl_pct    = total_pnl / config.INITIAL_CAPITAL * 100
    win_rate  = (len(winners) / len(sells) * 100) if sells else 0

    box_top()
    box_row(f"RESULTS  —  {_dates_label}")
    box_mid()
    box_row(f"Starting Capital  :  ₹{config.INITIAL_CAPITAL:>12,.2f}")
    box_row(f"Final Cash        :  ₹{cash:>12,.2f}")
    box_row(f"Total Realized P&L:  ₹{total_pnl:>+12,.2f}   ({pnl_pct:>+.2f}%)")
    box_row(f"Groww Fees Paid   :  ₹{total_fees:>12,.2f}")
    box_sep()
    box_row(f"Total Trades      :  {len(trade_history):>6d}   ({len(buys)} buy / {len(sells)} sell)")
    box_row(f"Winning Trades    :  {len(winners):>6d}")
    box_row(f"Losing Trades     :  {len(losers):>6d}")
    box_row(f"Win Rate          :  {win_rate:>10.1f}%")
    box_sep()
    box_row(f"Exits by source   :  🤖 AI={len(ai_sells)}  📉 trail_stop={len(box_sells)}  ⛔ SL={len(sl_sells)}  other={len(sells)-len(ai_sells)-len(box_sells)-len(sl_sells)}")
    if winners:
        best = max(winners, key=lambda x: x["pnl"])
        box_row(f"Best Trade        :  {best['symbol']}  {best['date']} {best['time']}  +₹{best['pnl']:.2f} ({best['pnl_pct']:+.1f}%)")
    if losers:
        worst = min(losers, key=lambda x: x["pnl"])
        box_row(f"Worst Trade       :  {worst['symbol']}  {worst['date']} {worst['time']}  ₹{worst['pnl']:.2f} ({worst['pnl_pct']:+.1f}%)")
    box_sep()
    box_row("DAY-BY-DAY BREAKDOWN:")
    for ds in day_summaries:
        on = f"  overnight: {', '.join(ds['overnight'])}" if ds['overnight'] else ""
        box_row(f"  {ds['day']:<9}  {ds['date']}   P&L ₹{ds['pnl']:>+9,.2f}   "
                f"Portfolio ₹{ds['portfolio']:>10,.2f}{on}")
    box_bot()

    # ── Strategy Comparison ────────────────────────────────────────────────────────
    print()
    box_top()
    _dates_str = f"{BACKTEST_DATES[0]} → {BACKTEST_DATES[-1]}" if len(BACKTEST_DATES) > 1 else str(BACKTEST_DATES[0])
    box_row(f"STRATEGY COMPARISON  —  {_dates_str}")
    box_mid()
    box_row(f"  {'Strategy':<34}  {'P&L':>12}  {'Return':>8}  {'Fees':>8}  {'Trades':>7}  {'Win Rate':>9}")
    box_sep()
    _active  = "◀ ACTIVE"
    if BOX_ONLY:
        box_row(f"  {'Box Theory Only (no AI)':<34}  ₹{total_pnl:>+11,.2f}  {pnl_pct:>+7.2f}%  ₹{total_fees:>6,.0f}  {len(trade_history):>7d}  {win_rate:>8.1f}%  {_active}")
        box_row(f"  {'Hybrid (AI windows + Box auto)':<34}  {'(run BOX_ONLY=False to measure)':>42}")
    else:
        box_row(f"  {'Box Theory Only (no AI)':<34}  {'(run BOX_ONLY=True to measure)':>42}")
        box_row(f"  {'Hybrid (AI windows + Box auto)':<34}  ₹{total_pnl:>+11,.2f}  {pnl_pct:>+7.2f}%  ₹{total_fees:>6,.0f}  {len(trade_history):>7d}  {win_rate:>8.1f}%  {_active}")
    box_bot()

    # ── Full trade log ─────────────────────────────────────────────────────────────
    if trade_history:
        print("\n  TRADE LOG:")
        print(f"  {'Date':<9}  {'Time':>5}  {'Who':<4}  {'Act':<4}  {'Symbol':<12}  {'Price':>8}  {'Qty':>5}  {'P&L':>10}  Reason")
        print("  " + "─" * 78)
        for t in trade_history:
            pnl_str = f"₹{t['pnl']:>+8.2f}" if "pnl" in t else f"cost ₹{t['cost']:,.0f}"
            reason  = t.get("reason", "signal")
            who     = "🤖AI" if reason in ("ai_eval",) else "BOX "
            print(f"  {t['date']:<9}  {t['time']:>5}  {who}  {t['action']:<4}  {t['symbol']:<12}  "
                  f"₹{t['price']:>8.2f}  {t['qty']:>5}  {pnl_str:>10}  {reason}")

    # Build equity curve for UI chart
    capital = float(config.INITIAL_CAPITAL)
    equity_curve = [{"trade": 0, "value": capital}]
    for i, t in enumerate(trade_history):
        if t.get("action") == "SELL" and "pnl" in t:
            capital += t["pnl"]
            equity_curve.append({"trade": i + 1, "value": round(capital, 2)})

    trades = []
    for t in trade_history:
        ts = {"action": t["action"], "symbol": t["symbol"], "qty": t["qty"], "price": t["price"],
              "fees": t.get("fees", 0), "reason": t.get("reason", "signal")}
        if "pnl" in t:
            ts["pnl"] = t["pnl"]
            ts["pnl_pct"] = t.get("pnl_pct", 0)
        if "cost" in t:
            ts["cost"] = t["cost"]
        ts["timestamp"] = f"{t.get('date','')} {t.get('time','')}"
        trades.append(ts)

    return {
        "trades": trades,
        "summary": {
            "total_pnl": total_pnl,
            "pnl_pct": round(pnl_pct, 2),
            "win_rate": round(win_rate, 1),
            "wins": len(winners),
            "losses": len(losers),
            "total_trades": len(trade_history),
            "total_fees": round(total_fees, 2),
            "final_cash": round(cash, 2),
            "start_date": str(BACKTEST_DATES[0]),
            "end_date": str(BACKTEST_DATES[-1]),
        },
        "day_summaries": day_summaries,
        "equity_curve": equity_curve,
    }


if __name__ == "__main__":
    _execute_backtest()
