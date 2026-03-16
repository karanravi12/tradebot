"""Comprehensive test suite for the AI Trading Bot.

Tests:
1. Portfolio — 100 buys + 100 sells stress test, P&L accuracy, edge cases
2. Data fetcher — real stock data via Groww API
3. Strategy indicators — all 8 strategies on real data
4. AI Brain — analyze_single, analyze_batch, chat (live API calls)
5. Full scan_and_trade cycle
6. API endpoints via Flask test client
"""

import json
import logging
import os
import random
import sys
import time
from datetime import datetime

# Suppress noisy logs during tests
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# Ensure imports work
sys.path.insert(0, os.path.dirname(__file__))
import config
import data_fetcher
from portfolio import Portfolio
from strategies import collect_indicators

PASS = 0
FAIL = 0
ERRORS = []


def test(name):
    """Decorator to run and report test results."""
    def decorator(func):
        def wrapper():
            global PASS, FAIL
            try:
                func()
                PASS += 1
                print(f"  PASS  {name}")
            except AssertionError as e:
                FAIL += 1
                ERRORS.append(f"{name}: {e}")
                print(f"  FAIL  {name}: {e}")
            except Exception as e:
                FAIL += 1
                ERRORS.append(f"{name}: EXCEPTION {e}")
                print(f"  FAIL  {name}: EXCEPTION {type(e).__name__}: {e}")
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════
# TEST SUITE 1: PORTFOLIO — 100 BUYS + 100 SELLS + EDGE CASES
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TEST SUITE 1: PORTFOLIO — STRESS TEST (100 BUY + 100 SELL)")
print("=" * 70)


@test("Fresh portfolio starts with correct capital")
def test_initial():
    p = Portfolio()
    assert p.cash == config.INITIAL_CAPITAL, f"Expected {config.INITIAL_CAPITAL}, got {p.cash}"
    assert len(p.positions) == 0
    assert p.realized_pnl == 0.0
    assert p.total_trades == 0
test_initial()


@test("100 BUY orders — diverse stocks at deterministic prices")
def test_100_buys():
    p = Portfolio()
    p.cash = 10_000_000  # give plenty of cash for 100 buys

    old_max = config.MAX_POSITIONS
    config.MAX_POSITIONS = 200

    signal = {"signal": "TEST", "composite_score": 0.8}
    successful_buys = 0

    for i in range(100):
        symbol = f"TEST{i:03d}"
        price = round(50 + i * 5, 2)  # 50, 55, 60, ... 545
        qty = 5

        trade = p.buy(symbol, price, qty, signal)

        if trade:
            successful_buys += 1
            assert trade["action"] == "BUY", f"Expected BUY, got {trade['action']}"
            assert trade["symbol"] == symbol
            assert trade["qty"] == 5
            assert trade["price"] == price
            assert trade["cost"] == round(price * 5, 2)

    config.MAX_POSITIONS = old_max

    assert successful_buys == 100, f"Expected 100 buys, got {successful_buys}"
    assert len(p.positions) == 100
    assert p.total_trades == 100
    assert p.cash < 10_000_000  # some cash was spent
    assert p.cash >= 0  # not negative
test_100_buys()


@test("100 SELL orders — sell all positions, verify P&L")
def test_100_sells():
    p = Portfolio()
    p.cash = 1_000_000
    signal = {"signal": "TEST", "composite_score": 0.8}

    old_max = config.MAX_POSITIONS
    config.MAX_POSITIONS = 200

    # Buy 100 stocks at known prices
    buy_prices = {}
    for i in range(100):
        symbol = f"TEST{i:03d}"
        price = round(100 + i * 10, 2)  # 100, 110, 120, ...
        qty = 5
        p.buy(symbol, price, qty, signal)
        buy_prices[symbol] = price

    config.MAX_POSITIONS = old_max

    cash_after_buys = p.cash
    total_cost = 1_000_000 - cash_after_buys

    # Sell all 100 at +5% profit
    total_expected_pnl = 0
    successful_sells = 0
    for i in range(100):
        symbol = f"TEST{i:03d}"
        sell_price = round(buy_prices[symbol] * 1.05, 2)  # 5% profit
        expected_pnl = round((sell_price - buy_prices[symbol]) * 5, 2)
        total_expected_pnl += expected_pnl

        trade = p.sell(symbol, sell_price, "test", signal)
        if trade:
            successful_sells += 1
            assert trade["action"] == "SELL"
            assert trade["qty"] == 5
            assert trade["pnl"] > 0, f"Expected profit but got pnl={trade['pnl']}"

    assert successful_sells == 100, f"Expected 100 sells, got {successful_sells}"
    assert len(p.positions) == 0, "All positions should be closed"
    assert p.winning_trades == 100
    assert p.losing_trades == 0
    assert p.total_trades == 200  # 100 buys + 100 sells
    assert p.realized_pnl > 0, f"Expected positive P&L, got {p.realized_pnl}"
    assert p.cash > cash_after_buys, "Cash should increase after profitable sells"
test_100_sells()


@test("Mixed profit/loss trades — P&L calculation accuracy (with Groww fees)")
def test_pnl_accuracy():
    import fees as groww_fees

    p = Portfolio()
    p.cash = 200_000  # give enough cash so all 100 buys succeed
    signal = {"signal": "TEST", "composite_score": 0.5}

    old_max = config.MAX_POSITIONS
    config.MAX_POSITIONS = 200

    # Buy 100 stocks at ₹100, qty=10 each
    for i in range(50):
        p.buy(f"WIN{i}", 100.0, 10, signal)
    for i in range(50):
        p.buy(f"LOSE{i}", 100.0, 10, signal)

    config.MAX_POSITIONS = old_max

    assert p.cash >= 0, f"Cash should not go negative, got {p.cash}"
    assert p.total_fees_paid > 0, "Should have deducted buy-side Groww fees"
    assert len(p.positions) == 100, f"Should have 100 positions, got {len(p.positions)}"

    # Sell winners at 110 (+10% gross per trade)
    for i in range(50):
        p.sell(f"WIN{i}", 110.0, "profit", signal)

    # Sell losers at 90 (-10% gross per trade)
    for i in range(50):
        p.sell(f"LOSE{i}", 90.0, "loss", signal)

    assert len(p.positions) == 0, "All positions should be closed"
    assert p.winning_trades == 50
    assert p.losing_trades == 50

    # Gross P&L (ignoring fees): 50 * (110-100)*10 - 50 * (100-90)*10 = 5000 - 5000 = 0
    # Net P&L = gross - sell_fees (buy fees already out of cash, not in realized_pnl)
    # Both sell fees are ~₹6.2 per order → net ≈ -(50 * sell_fee_110 + 50 * sell_fee_90)
    sell_fee_win = groww_fees.calculate_sell_fees(110.0, 10)["total_fees"]
    sell_fee_lose = groww_fees.calculate_sell_fees(90.0, 10)["total_fees"]
    expected_realized = -(50 * sell_fee_win + 50 * sell_fee_lose)
    assert abs(p.realized_pnl - expected_realized) < 2.0, (
        f"realized_pnl {p.realized_pnl:.2f} should ≈ {expected_realized:.2f}"
    )
    # Total fees includes both buy and sell fees
    assert p.total_fees_paid > abs(expected_realized), "Total fees > sell-only fees"
test_pnl_accuracy()


@test("Max positions limit enforced")
def test_max_positions():
    p = Portfolio()
    p.cash = 1_000_000
    signal = {"signal": "TEST", "composite_score": 0.5}

    for i in range(config.MAX_POSITIONS):
        trade = p.buy(f"POS{i}", 100.0, 1, signal)
        assert trade is not None

    # Next buy should fail (can_buy returns False)
    assert not p.can_buy(100.0), "Should not allow more than MAX_POSITIONS"
test_max_positions()


@test("Insufficient cash handling")
def test_insufficient_cash():
    p = Portfolio()
    p.cash = 50  # Only ₹50
    signal = {"signal": "TEST", "composite_score": 0.5}

    # Try to buy stock at ₹100 — should fail
    assert not p.can_buy(100.0), "Should not buy when price > cash"

    # Try to buy with qty that exceeds cash — should auto-reduce qty
    trade = p.buy("CHEAP", 30.0, 5, signal)  # 5*30=150 > 50
    assert trade is not None, "Should buy with reduced qty"
    assert trade["qty"] == 1, f"Should buy 1 share (50//30=1), got {trade['qty']}"
    assert p.cash >= 0, "Cash should never go negative"
test_insufficient_cash()


@test("Stop-loss detection at exactly 3%")
def test_stop_loss():
    p = Portfolio()
    p.cash = 100_000
    signal = {"signal": "TEST", "composite_score": 0.5}
    p.buy("SL_TEST", 100.0, 10, signal)

    # At 97.01 — NOT stop loss (2.99% loss)
    assert not p.check_stop_loss("SL_TEST", 97.01), "97.01 should not trigger stop-loss"

    # At 97.00 — exactly 3% loss — SHOULD trigger
    assert p.check_stop_loss("SL_TEST", 97.0), "97.0 should trigger stop-loss"

    # At 96.00 — 4% loss — SHOULD trigger
    assert p.check_stop_loss("SL_TEST", 96.0), "96.0 should trigger stop-loss"
test_stop_loss()


@test("Sell non-existent position returns None")
def test_sell_nonexistent():
    p = Portfolio()
    signal = {"signal": "TEST", "composite_score": 0.5}
    trade = p.sell("NOEXIST", 100.0, "test", signal)
    assert trade is None, "Should return None for non-existent position"
test_sell_nonexistent()


@test("Buy with zero/negative qty or price returns None")
def test_buy_invalid():
    p = Portfolio()
    signal = {"signal": "TEST", "composite_score": 0.5}
    assert p.buy("X", 100.0, 0, signal) is None, "qty=0 should fail"
    assert p.buy("X", 100.0, -1, signal) is None, "qty=-1 should fail"
    assert p.buy("X", 0, 5, signal) is None, "price=0 should fail"
    assert p.buy("X", -100, 5, signal) is None, "price=-100 should fail"
test_buy_invalid()


@test("Averaging into existing position")
def test_averaging():
    p = Portfolio()
    p.cash = 100_000
    signal = {"signal": "TEST", "composite_score": 0.5}

    p.buy("AVG", 100.0, 10, signal)  # 10 @ 100
    p.buy("AVG", 200.0, 10, signal)  # 10 @ 200
    # Avg should be (100*10 + 200*10) / 20 = 150
    pos = p.positions["AVG"]
    assert pos["qty"] == 20
    assert abs(pos["avg_price"] - 150.0) < 0.01, f"Avg price should be 150, got {pos['avg_price']}"
test_averaging()


@test("Deposit and withdraw")
def test_wallet():
    p = Portfolio()
    initial = p.cash

    result = p.deposit(5000)
    assert "error" not in result
    assert p.cash == initial + 5000

    result = p.withdraw(3000)
    assert "error" not in result
    assert p.cash == initial + 2000

    # Withdraw more than available
    result = p.withdraw(999_999)
    assert "error" in result
    assert p.cash == initial + 2000  # unchanged

    # Negative amounts
    result = p.deposit(-100)
    assert "error" in result
    result = p.withdraw(-100)
    assert "error" in result
test_wallet()


@test("to_dict serialization with positions")
def test_to_dict():
    p = Portfolio()
    p.cash = 50_000
    signal = {"signal": "TEST", "composite_score": 0.5}
    p.buy("INFY", 1500.0, 5, signal)
    p.buy("TCS", 3500.0, 2, signal)

    prices = {"INFY": 1550.0, "TCS": 3400.0}
    d = p.to_dict(prices)

    assert "cash" in d
    assert "total_value" in d
    assert "positions" in d
    assert len(d["positions"]) == 2
    assert d["num_positions"] == 2

    # Verify JSON serializable
    json_str = json.dumps(d)
    assert len(json_str) > 0
test_to_dict()


@test("Save and load round-trip")
def test_save_load():
    p = Portfolio()
    p.cash = 12345.67
    signal = {"signal": "TEST", "composite_score": 0.5}
    p.buy("SAVE_TEST", 500.0, 3, signal)
    p.realized_pnl = 999.99
    p.save()

    p2 = Portfolio.load()
    assert abs(p2.cash - p.cash) < 0.01
    assert "SAVE_TEST" in p2.positions
    assert p2.positions["SAVE_TEST"]["qty"] == 3
    assert abs(p2.realized_pnl - 999.99) < 0.01

    # Clean up
    p2.reset()
test_save_load()


@test("Reset returns to initial state")
def test_reset():
    p = Portfolio()
    p.cash = 999
    signal = {"signal": "TEST", "composite_score": 0.5}
    p.buy("RESET_TEST", 100, 5, signal)
    p.reset()

    assert p.cash == config.INITIAL_CAPITAL
    assert len(p.positions) == 0
    assert p.realized_pnl == 0
    assert p.total_trades == 0
test_reset()


@test("100 rapid buy-sell cycles — fees correctly deducted each round trip")
def test_rapid_cycles():
    import fees as groww_fees

    p = Portfolio()
    p.cash = 20_000
    signal = {"signal": "TEST", "composite_score": 0.5}
    initial_cash = p.cash

    price = 100.0
    qty = 10  # ₹1000 per order

    # Pre-calculate fees per cycle
    buy_fee = groww_fees.calculate_buy_fees(price, qty)["total_fees"]
    sell_fee = groww_fees.calculate_sell_fees(price, qty)["total_fees"]
    fee_per_cycle = buy_fee + sell_fee  # total cash consumed per round trip

    for i in range(100):
        p.buy("CYCLE", price, qty, signal)
        p.sell("CYCLE", price, "cycle_test", signal)

    # Cash = initial - (buy_fees * 100) - (sell_fees * 100)
    # buy_fees come out of cash at buy time; sell_fees reduce net_revenue at sell time
    expected_cash = initial_cash - fee_per_cycle * 100
    assert abs(p.cash - expected_cash) < 1.0, (
        f"Cash should be ~₹{expected_cash:.2f}, got ₹{p.cash:.2f}"
    )
    # realized_pnl = sum of (net_revenue - buy_cost) per sell
    # net_revenue = sell_price*qty - sell_fee; buy_cost = avg_price*qty = sell_price*qty (same price)
    # so pnl per sell = -sell_fee
    expected_realized = -sell_fee * 100
    assert abs(p.realized_pnl - expected_realized) < 1.0, (
        f"realized_pnl {p.realized_pnl:.2f} should ≈ {expected_realized:.2f} (sell fees only)"
    )
    assert p.total_trades == 200
    assert p.total_fees_paid > 0, "Should have paid fees"
    assert abs(p.total_fees_paid - fee_per_cycle * 100) < 1.0, "total_fees_paid should = all round-trip fees"
    assert p.losing_trades == 100  # pnl after fees always negative at same price
test_rapid_cycles()


# ═══════════════════════════════════════════════════════════════════════
# TEST SUITE 2: DATA FETCHER — REAL STOCK DATA
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TEST SUITE 2: DATA FETCHER — REAL STOCK DATA")
print("=" * 70)

TEST_STOCKS = ["HDFCBANK", "INFY", "RELIANCE", "TCS", "SBIN"]


@test("Fetch intraday data for 5 major stocks")
def test_fetch_intraday():
    for sym in TEST_STOCKS:
        df = data_fetcher.fetch_intraday(sym)
        if df is not None:
            assert len(df) >= 30, f"{sym}: only {len(df)} rows"
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                assert col in df.columns, f"{sym}: missing {col}"
            price = float(df["Close"].iloc[-1])
            assert price > 0, f"{sym}: price should be positive, got {price}"
test_fetch_intraday()


@test("fetch_data returns valid DataFrame for all test stocks")
def test_fetch_data():
    for sym in TEST_STOCKS:
        df = data_fetcher.fetch_data(sym)
        assert df is not None, f"Failed to fetch data for {sym}"
        assert len(df) >= 30
test_fetch_data()


@test("Non-existent symbol returns None gracefully")
def test_nonexistent_symbol():
    df = data_fetcher.fetch_data("ZZZZZZZ_FAKE")
    assert df is None, "Non-existent symbol should return None"
test_nonexistent_symbol()


@test("TATAMOTORS (known delisted) handled gracefully")
def test_delisted():
    df = data_fetcher.fetch_data("TATAMOTORS")
    # Should return None without crashing
    # (it's delisted on Yahoo Finance)
test_delisted()


@test("_safe_price extracts scalar from DataFrame correctly")
def test_safe_price():
    import pandas as pd
    from trader import _safe_price

    df = pd.DataFrame({"Close": [100.0, 200.0, 300.0]})
    assert _safe_price(df) == 300.0

    df2 = pd.DataFrame({"Close": [150.0]})
    assert _safe_price(df2) == 150.0
test_safe_price()


# ═══════════════════════════════════════════════════════════════════════
# TEST SUITE 3: STRATEGY INDICATORS — ALL 8 STRATEGIES
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TEST SUITE 3: STRATEGY INDICATORS — ALL 8 ON REAL DATA")
print("=" * 70)


@test("All 8 strategies run without error on HDFCBANK")
def test_all_strategies():
    df = data_fetcher.fetch_data("HDFCBANK")
    assert df is not None, "Failed to fetch HDFCBANK data"

    result = collect_indicators(df, "HDFCBANK")
    assert "indicators" in result
    assert "has_signal" in result

    expected_strategies = ["technical", "smart_money", "supertrend", "vwap", "ichimoku", "bollinger", "fibonacci", "day_box"]
    for name in expected_strategies:
        assert name in result["indicators"], f"Missing strategy: {name}"
        strat = result["indicators"][name]
        assert "score" in strat, f"{name}: missing 'score'"
        assert "confidence" in strat, f"{name}: missing 'confidence'"
        assert "details" in strat, f"{name}: missing 'details'"
        assert -3 <= strat["score"] <= 3, f"{name}: score {strat['score']} out of range"
        assert 0 <= strat["confidence"] <= 1, f"{name}: confidence {strat['confidence']} out of range"
        # Verify no numpy bools (JSON serialization issue)
        json.dumps(strat, default=str)  # Should not raise
test_all_strategies()


@test("Strategies work on all 5 test stocks")
def test_strategies_multiple_stocks():
    for sym in TEST_STOCKS:
        df = data_fetcher.fetch_data(sym)
        if df is None:
            continue
        result = collect_indicators(df, sym)
        assert len(result["indicators"]) == 8, f"{sym}: expected 8 strategies, got {len(result['indicators'])}"
        # Verify JSON serializable
        json.dumps(result, default=str)
test_strategies_multiple_stocks()


@test("Indicator data is JSON serializable (no numpy types)")
def test_json_serializable():
    df = data_fetcher.fetch_data("RELIANCE")
    assert df is not None
    result = collect_indicators(df, "RELIANCE")
    # This will raise if there are numpy bools/ints/floats
    json_str = json.dumps(result, default=str)
    assert len(json_str) > 100
test_json_serializable()


# ═══════════════════════════════════════════════════════════════════════
# TEST SUITE 4: AI BRAIN — LIVE API CALLS
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TEST SUITE 4: AI BRAIN — LIVE API CALLS")
print("=" * 70)

# Load .env for API key
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
if not has_api_key:
    print("  SKIP  No API key — skipping AI tests")
else:
    import ai_brain
    import api_costs

    @test("ai_brain.chat — returns non-empty response")
    def test_chat():
        portfolio_state = {
            "cash": 20000, "total_value": 20000, "overall_pnl": 0,
            "overall_pnl_pct": 0, "realized_pnl": 0, "unrealized_pnl": 0,
            "num_positions": 0, "max_positions": 5, "win_rate": 0,
            "total_trades": 0, "positions": [],
        }
        response = ai_brain.chat("What stocks should I look at today?", portfolio_state, [])
        assert len(response) > 20, f"Response too short: {response[:50]}"
        assert "Error" not in response, f"Got error: {response[:100]}"
    test_chat()
    time.sleep(5)  # avoid rate limit between AI calls


    @test("ai_brain.analyze_single — returns valid HOLD/SELL decision")
    def test_analyze_single():
        df = data_fetcher.fetch_data("HDFCBANK")
        assert df is not None

        indicators = collect_indicators(df, "HDFCBANK")["indicators"]
        price = float(df["Close"].iloc[-1]) if not hasattr(df["Close"].iloc[-1], "iloc") else float(df["Close"].iloc[-1].iloc[0])
        portfolio_state = {
            "cash": 15000, "num_positions": 1,
            "portfolio_value": 20000, "realized_pnl": 0,
            "position": {"qty": 3, "avg_price": price * 0.98},
        }

        result = ai_brain.analyze_single("HDFCBANK", df, indicators, portfolio_state)
        assert "action" in result, f"Missing 'action' in response: {result}"
        assert result["action"] in ("HOLD", "SELL"), f"Invalid action: {result['action']}"
        assert "confidence" in result
        assert "reasoning" in result
        assert len(result["reasoning"]) > 10
    test_analyze_single()
    time.sleep(5)  # avoid rate limit between AI calls


    @test("ai_brain.analyze_batch — returns valid BUY recommendations")
    def test_analyze_batch():
        candidates = []
        for sym in ["INFY", "TCS", "SBIN"]:
            df = data_fetcher.fetch_data(sym)
            if df is None:
                continue
            indicators = collect_indicators(df, sym)["indicators"]
            price = float(df["Close"].iloc[-1]) if not hasattr(df["Close"].iloc[-1], "iloc") else float(df["Close"].iloc[-1].iloc[0])
            candidates.append({
                "symbol": sym,
                "price": price,
                "indicators": indicators,
                "recent_candles": "",
            })

        portfolio_state = {
            "cash": 20000, "num_positions": 0,
            "portfolio_value": 20000, "realized_pnl": 0,
            "held_symbols": [],
        }

        results = ai_brain.analyze_batch(candidates, portfolio_state)
        assert isinstance(results, list), f"Expected list, got {type(results)}"
        for r in results:
            assert "symbol" in r
            assert "confidence" in r
            assert "reasoning" in r
            assert r["symbol"] in [c["symbol"] for c in candidates]
    test_analyze_batch()


    @test("API cost tracking recorded calls")
    def test_cost_tracking():
        summary = api_costs.get_cost_summary()
        assert summary["today"]["calls"] >= 3, f"Expected >=3 calls today, got {summary['today']['calls']}"
        assert summary["today"]["cost_usd"] > 0, "Expected non-zero cost"
        assert summary["total_calls"] >= 3
        assert len(summary["recent_calls"]) >= 3
    test_cost_tracking()


    @test("Daily limit enforcement works")
    def test_daily_limit():
        old_limit = config.API_DAILY_LIMIT_USD
        config.API_DAILY_LIMIT_USD = 0.0001  # impossibly low

        allowed, spent, limit = api_costs.check_daily_limit()
        assert not allowed, "Should be blocked with tiny limit"

        config.API_DAILY_LIMIT_USD = old_limit  # restore
    test_daily_limit()


# ═══════════════════════════════════════════════════════════════════════
# TEST SUITE 5: FLASK API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TEST SUITE 5: FLASK API ENDPOINTS")
print("=" * 70)

import app as flask_app
flask_app.portfolio.reset()  # Start fresh
client = flask_app.app.test_client()


@test("GET / returns 200")
def test_homepage():
    resp = client.get("/")
    assert resp.status_code == 200
test_homepage()


@test("GET /api/portfolio returns valid JSON")
def test_api_portfolio():
    resp = client.get("/api/portfolio")
    assert resp.status_code == 200
    d = resp.get_json()
    assert "cash" in d
    assert "total_value" in d
    assert "positions" in d
    assert d["cash"] == config.INITIAL_CAPITAL
test_api_portfolio()


@test("GET /api/config returns all fields")
def test_api_config():
    resp = client.get("/api/config")
    assert resp.status_code == 200
    d = resp.get_json()
    required = ["claude_model", "claude_max_tokens", "claude_temperature", "max_positions",
                "max_position_pct", "stop_loss_pct", "scan_interval", "ai_pre_filter",
                "api_daily_limit", "available_models", "has_api_key"]
    for key in required:
        assert key in d, f"Missing config field: {key}"
test_api_config()


@test("POST /api/config updates settings")
def test_api_update_config():
    resp = client.post("/api/config", json={"max_positions": 7})
    assert resp.status_code == 200
    d = resp.get_json()
    assert "max_positions" in d["updated"]
    assert config.MAX_POSITIONS == 7

    # Restore
    config.MAX_POSITIONS = 5
test_api_update_config()


@test("POST /api/wallet/deposit adds cash")
def test_api_deposit():
    resp = client.post("/api/wallet/deposit", json={"amount": 10000})
    assert resp.status_code == 200
    d = resp.get_json()
    assert "error" not in d
    assert d["cash_after"] == config.INITIAL_CAPITAL + 10000
test_api_deposit()


@test("POST /api/wallet/withdraw removes cash")
def test_api_withdraw():
    resp = client.post("/api/wallet/withdraw", json={"amount": 5000})
    assert resp.status_code == 200
    d = resp.get_json()
    assert "error" not in d
test_api_withdraw()


@test("POST /api/wallet/withdraw fails with insufficient cash")
def test_api_withdraw_fail():
    resp = client.post("/api/wallet/withdraw", json={"amount": 999_999})
    d = resp.get_json()
    assert "error" in d
test_api_withdraw_fail()


@test("GET /api/bot/status returns market info")
def test_api_bot_status():
    resp = client.get("/api/bot/status")
    assert resp.status_code == 200
    d = resp.get_json()
    assert "bot_running" in d
    assert "market_open" in d
    assert "current_time" in d
    assert "market_hours" in d
    assert "is_holiday" in d
test_api_bot_status()


@test("GET /api/trades returns list")
def test_api_trades():
    resp = client.get("/api/trades")
    assert resp.status_code == 200
    d = resp.get_json()
    assert isinstance(d, list)
test_api_trades()


@test("GET /api/costs returns cost data")
def test_api_costs():
    resp = client.get("/api/costs")
    assert resp.status_code == 200
    d = resp.get_json()
    assert "today" in d
    assert "total_usd" in d
    assert "daily_limit_usd" in d
    assert "recent_calls" in d
test_api_costs()


@test("POST /api/portfolio/reset works")
def test_api_reset():
    resp = client.post("/api/portfolio/reset", json={})
    assert resp.status_code == 200
    d = resp.get_json()
    assert d["status"] == "reset"

    # Verify actually reset
    resp2 = client.get("/api/portfolio")
    d2 = resp2.get_json()
    assert d2["cash"] == config.INITIAL_CAPITAL
    assert d2["num_positions"] == 0
test_api_reset()


@test("POST /api/config with daily limit updates correctly")
def test_api_daily_limit():
    resp = client.post("/api/config", json={"api_daily_limit": 10.0})
    assert resp.status_code == 200
    assert config.API_DAILY_LIMIT_USD == 10.0

    # Restore
    config.API_DAILY_LIMIT_USD = 5.0
test_api_daily_limit()


if has_api_key:
    time.sleep(10)  # let rate-limit token bucket recover before chat tests

    @test("POST /api/chat returns AI response")
    def test_api_chat():
        resp = client.post("/api/chat", json={"message": "Hi, what is my balance?"})
        assert resp.status_code == 200
        d = resp.get_json()
        assert "response" in d
        assert len(d["response"]) > 10
        assert "Error" not in d["response"]
    test_api_chat()


    @test("POST /api/chat rejects empty message")
    def test_api_chat_empty():
        resp = client.post("/api/chat", json={"message": "  "})
        d = resp.get_json()
        assert "error" in d
    test_api_chat_empty()


# ═══════════════════════════════════════════════════════════════════════
# TEST SUITE 6: FULL SCAN_AND_TRADE CYCLE (WITH AI)
# ═══════════════════════════════════════════════════════════════════════

if has_api_key:
    print("\n" + "=" * 70)
    print("TEST SUITE 6: FULL scan_and_trade CYCLE")
    print("=" * 70)

    import trader

    @test("Full scan_and_trade executes without errors")
    def test_full_scan():
        p = Portfolio()
        p.cash = 20_000
        # Run the full scan — fetches data, runs all strategies, calls AI
        trader.scan_and_trade(p)
        # Should not crash, should update state
        assert p.total_trades >= 0  # may or may not trade depending on AI
        assert p.cash >= 0
        assert p.cash <= 20_000 + 1  # shouldn't magically gain cash
    test_full_scan()


# ═══════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(f"RESULTS: {PASS} passed, {FAIL} failed")
print("=" * 70)

if ERRORS:
    print("\nFAILURES:")
    for err in ERRORS:
        print(f"  - {err}")
    print()

# Clean up test state
Portfolio().reset()

sys.exit(1 if FAIL > 0 else 0)
