"""Portfolio manager — tracks positions, cash, P&L, and persists state.

Thread safety:
  All mutations (buy, sell, deposit, withdraw, reset) are guarded by an RLock.
  File writes use atomic rename (write to .tmp, then os.replace) to prevent
  corruption on crash or concurrent access.

  The lock is reentrant (RLock) so methods that call save() internally
  don't deadlock.
"""

import json
import logging
import os
import tempfile
import threading
from datetime import datetime
import config
import fees as groww_fees

logger = logging.getLogger(__name__)

STATE_FILE = os.environ.get(
    "PORTFOLIO_PATH",
    os.path.join(os.path.dirname(__file__), "portfolio.json")
)

# Module-level lock — all Portfolio mutations must hold this.
# RLock so nested calls (e.g. reset() → save()) don't deadlock.
_portfolio_lock = threading.RLock()

# Maximum trade history entries kept in memory and on disk.
# Older entries are trimmed on save(). At ~50 trades/day this is ~20 days.
_MAX_TRADE_HISTORY = 1000


class Portfolio:
    def __init__(self):
        self.cash: float = config.INITIAL_CAPITAL
        self.initial_capital: float = config.INITIAL_CAPITAL
        self.positions: dict = {}  # symbol -> {qty, avg_price, highest_price, ...}
        self.realized_pnl: float = 0.0
        self.trade_history: list = []
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.total_fees_paid: float = 0.0

    # ── Serialization ─────────────────────────────────────────────────────

    def save(self):
        """Persist portfolio state to disk using atomic write.

        Writes to a temp file first, then atomically replaces the target.
        This prevents corruption if the process crashes mid-write or disk fills up.
        """
        state = {
            "cash": self.cash,
            "initial_capital": self.initial_capital,
            "positions": self.positions,
            "realized_pnl": self.realized_pnl,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_fees_paid": self.total_fees_paid,
            "trade_history": self.trade_history[-_MAX_TRADE_HISTORY:],
        }
        try:
            state_dir = os.path.dirname(STATE_FILE) or "."
            # Write to temp file in the same directory (same filesystem for atomic rename)
            fd, tmp_path = tempfile.mkstemp(dir=state_dir, suffix=".tmp", prefix="portfolio_")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(state, f, indent=2, default=str)
                os.replace(tmp_path, STATE_FILE)  # atomic on POSIX and Windows
            except Exception:
                # Clean up temp file if rename failed
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            logger.debug("Portfolio state saved")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to save portfolio state: {e}")

    @classmethod
    def load(cls) -> "Portfolio":
        """Load portfolio from disk. Returns fresh portfolio if file missing or corrupt."""
        p = cls()
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as f:
                    state = json.load(f)
                p.cash = state.get("cash", config.INITIAL_CAPITAL)
                p.initial_capital = state.get("initial_capital", p.cash)
                p.positions = state.get("positions", {})
                p.realized_pnl = state.get("realized_pnl", 0.0)
                p.total_trades = state.get("total_trades", 0)
                p.winning_trades = state.get("winning_trades", 0)
                p.losing_trades = state.get("losing_trades", 0)
                p.total_fees_paid = state.get("total_fees_paid", 0.0)
                p.trade_history = state.get("trade_history", [])
                logger.info(f"Portfolio loaded: cash=₹{p.cash:.2f}, positions={len(p.positions)}")
            except Exception as e:
                logger.error(f"Failed to load portfolio: {e}. Starting fresh.")
        else:
            logger.info(f"No saved portfolio. Starting with ₹{config.INITIAL_CAPITAL}")
        return p

    # ── Position Sizing ───────────────────────────────────────────────────

    def max_position_value(self) -> float:
        """Max amount to allocate to a single position."""
        total_value = self.total_value({})
        return total_value * config.MAX_POSITION_PCT

    def can_buy(self, price: float) -> bool:
        """Check if we can open a new position."""
        if len(self.positions) >= config.MAX_POSITIONS:
            return False
        if price <= 0:
            return False
        return self.cash >= price

    def calculate_qty(self, price: float) -> int:
        """Calculate how many shares to buy based on position sizing."""
        if price <= 0:
            return 0
        max_value = self.max_position_value()
        affordable = self.cash
        allocation = min(max_value, affordable)
        qty = int(allocation // price)
        return max(qty, 0)

    # ── Trading ───────────────────────────────────────────────────────────

    def buy(self, symbol: str, price: float, qty: int, signal_info: dict, entry_box_bottom: float = None) -> dict | None:
        """Execute a paper buy. Returns trade record or None.

        Thread-safe: acquires _portfolio_lock. Saves state immediately after
        the trade so a crash won't lose the position.

        entry_box_bottom: the Darvas box bottom at breakout time — used as the
        anchor for the trailing stop.
        """
        with _portfolio_lock:
            if qty <= 0 or price <= 0:
                return None

            cost = price * qty
            fee_info = groww_fees.calculate_buy_fees(price, qty)
            total_outflow = cost + fee_info["total_fees"]

            if total_outflow > self.cash:
                budget = self.cash * 0.999
                qty = int(budget // price)
                if qty <= 0:
                    return None
                cost = price * qty
                fee_info = groww_fees.calculate_buy_fees(price, qty)
                total_outflow = cost + fee_info["total_fees"]
                if total_outflow > self.cash:
                    return None

            if symbol in self.positions:
                pos = self.positions[symbol]
                total_qty = pos["qty"] + qty
                avg_price = ((pos["avg_price"] * pos["qty"]) + cost) / total_qty
                self.positions[symbol] = {
                    "qty": total_qty,
                    "avg_price": round(avg_price, 2),
                    "highest_price": max(pos.get("highest_price", price), price),
                    "entry_ts":            pos.get("entry_ts"),
                    "entry_box_bottom":    pos.get("entry_box_bottom"),
                    "trailing_box_bottom": pos.get("trailing_box_bottom"),
                    "breakdown_count":     pos.get("breakdown_count", 0),
                }
            else:
                self.positions[symbol] = {
                    "qty": qty,
                    "avg_price": round(price, 2),
                    "highest_price": price,
                    "entry_ts": datetime.now(),
                    "entry_box_bottom":    entry_box_bottom,
                    "trailing_box_bottom": entry_box_bottom,
                    "breakdown_count":     0,
                }

            self.cash -= total_outflow
            self.total_fees_paid += fee_info["total_fees"]

            trade = {
                "timestamp": datetime.now().isoformat(),
                "action": "BUY",
                "symbol": symbol,
                "qty": qty,
                "price": round(price, 2),
                "cost": round(cost, 2),
                "fees": round(fee_info["total_fees"], 2),
                "fees_detail": fee_info,
                "total_outflow": round(total_outflow, 2),
                "signal": signal_info.get("signal", ""),
                "composite_score": signal_info.get("composite_score", 0),
                "cash_remaining": round(self.cash, 2),
            }
            self.trade_history.append(trade)
            self.total_trades += 1

            # Save immediately — don't risk losing a position on crash
            self.save()

            logger.info(
                f"BUY {symbol}: {qty} shares @ ₹{price:.2f} = ₹{cost:.2f} "
                f"| Fees: ₹{fee_info['total_fees']:.2f} | Cash: ₹{self.cash:.2f}"
            )
            return trade

    def sell(self, symbol: str, price: float, reason: str, signal_info: dict) -> dict | None:
        """Execute a paper sell. Returns trade record or None.

        Thread-safe: acquires _portfolio_lock. Saves state immediately.
        """
        with _portfolio_lock:
            if symbol not in self.positions:
                return None

            pos = self.positions[symbol]
            qty = pos["qty"]
            avg_price = pos["avg_price"]
            revenue = price * qty
            fee_info = groww_fees.calculate_sell_fees(price, qty)
            net_revenue = revenue - fee_info["total_fees"]

            buy_cost = avg_price * qty
            pnl = net_revenue - buy_cost
            pnl_pct = (pnl / buy_cost * 100) if buy_cost > 0 else 0

            self.cash += net_revenue
            self.realized_pnl += pnl
            self.total_fees_paid += fee_info["total_fees"]
            del self.positions[symbol]

            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1

            trade = {
                "timestamp": datetime.now().isoformat(),
                "action": "SELL",
                "symbol": symbol,
                "qty": qty,
                "price": round(price, 2),
                "revenue": round(revenue, 2),
                "fees": round(fee_info["total_fees"], 2),
                "fees_detail": fee_info,
                "net_revenue": round(net_revenue, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "reason": reason,
                "signal": signal_info.get("signal", ""),
                "composite_score": signal_info.get("composite_score", 0),
                "cash_remaining": round(self.cash, 2),
            }
            self.trade_history.append(trade)
            self.total_trades += 1

            # Save immediately — don't risk lost sell on crash
            self.save()

            logger.info(
                f"SELL {symbol}: {qty} shares @ ₹{price:.2f} = ₹{revenue:.2f} "
                f"| Fees: ₹{fee_info['total_fees']:.2f} | Net P&L: ₹{pnl:+.2f} ({pnl_pct:+.1f}%) | Reason: {reason}"
            )
            return trade

    # ── Stop Loss Check ───────────────────────────────────────────────────

    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """Check if a position has hit the stop-loss threshold."""
        if symbol not in self.positions:
            return False

        pos = self.positions[symbol]
        avg_price = pos["avg_price"]
        loss_pct = (avg_price - current_price) / avg_price if avg_price > 0 else 0

        return loss_pct >= config.STOP_LOSS_PCT

    def update_highest_price(self, symbol: str, current_price: float):
        """Update trailing highest price for a position."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            if current_price > pos.get("highest_price", 0):
                pos["highest_price"] = current_price

    # ── Portfolio Value ───────────────────────────────────────────────────

    def total_value(self, current_prices: dict) -> float:
        """Calculate total portfolio value (cash + positions)."""
        position_value = sum(
            pos["qty"] * current_prices.get(sym, pos["avg_price"])
            for sym, pos in self.positions.items()
        )
        return self.cash + position_value

    def unrealized_pnl(self, current_prices: dict) -> float:
        """Calculate total unrealized P&L across all positions."""
        pnl = 0
        for sym, pos in self.positions.items():
            current = current_prices.get(sym, pos["avg_price"])
            pnl += (current - pos["avg_price"]) * pos["qty"]
        return pnl

    # ── Wallet Management ───────────────────────────────────────────────

    def deposit(self, amount: float) -> dict:
        """Add money to the virtual wallet. Thread-safe."""
        with _portfolio_lock:
            if amount <= 0:
                return {"error": "Amount must be positive"}
            self.cash += amount
            record = {
                "timestamp": datetime.now().isoformat(),
                "action": "DEPOSIT",
                "amount": round(amount, 2),
                "cash_after": round(self.cash, 2),
            }
            self.trade_history.append(record)
            self.save()
            logger.info(f"DEPOSIT: +₹{amount:.2f} → Cash: ₹{self.cash:.2f}")
            return record

    def withdraw(self, amount: float) -> dict:
        """Remove money from the virtual wallet. Thread-safe."""
        with _portfolio_lock:
            if amount <= 0:
                return {"error": "Amount must be positive"}
            if amount > self.cash:
                return {"error": f"Insufficient cash. Available: ₹{self.cash:.2f}"}
            self.cash -= amount
            record = {
                "timestamp": datetime.now().isoformat(),
                "action": "WITHDRAW",
                "amount": round(amount, 2),
                "cash_after": round(self.cash, 2),
            }
            self.trade_history.append(record)
            self.save()
            logger.info(f"WITHDRAW: -₹{amount:.2f} → Cash: ₹{self.cash:.2f}")
            return record

    def reset(self):
        """Reset portfolio to initial state. Thread-safe."""
        with _portfolio_lock:
            self.cash = config.INITIAL_CAPITAL
            self.positions = {}
            self.realized_pnl = 0.0
            self.trade_history = []
            self.total_trades = 0
            self.winning_trades = 0
            self.losing_trades = 0
            self.total_fees_paid = 0.0
            self.save()
            logger.info(f"Portfolio RESET to ₹{config.INITIAL_CAPITAL}")

    def to_dict(self, current_prices: dict) -> dict:
        """Return portfolio state as a JSON-serializable dict for the web UI."""
        total = self.total_value(current_prices)
        unrealized = self.unrealized_pnl(current_prices)
        initial = self.initial_capital
        overall_pnl = total - initial
        overall_pct = (overall_pnl / initial * 100) if initial > 0 else 0
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        positions_list = []
        for sym, pos in self.positions.items():
            curr = current_prices.get(sym, pos["avg_price"])
            pos_pnl = (curr - pos["avg_price"]) * pos["qty"]
            pos_pct = (curr - pos["avg_price"]) / pos["avg_price"] * 100 if pos["avg_price"] > 0 else 0
            positions_list.append({
                "symbol": sym,
                "qty": pos["qty"],
                "avg_price": pos["avg_price"],
                "current_price": round(curr, 2),
                "pnl": round(pos_pnl, 2),
                "pnl_pct": round(pos_pct, 2),
            })

        return {
            "cash": round(self.cash, 2),
            "total_value": round(total, 2),
            "initial_capital": round(initial, 2),
            "overall_pnl": round(overall_pnl, 2),
            "overall_pnl_pct": round(overall_pct, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(unrealized, 2),
            "total_fees_paid": round(self.total_fees_paid, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(win_rate, 1),
            "num_positions": len(self.positions),
            "max_positions": config.MAX_POSITIONS,
            "positions": positions_list,
        }

    # ── Display ───────────────────────────────────────────────────────────

    def summary(self, current_prices: dict) -> str:
        """Generate a human-readable portfolio summary."""
        total = self.total_value(current_prices)
        unrealized = self.unrealized_pnl(current_prices)
        initial = self.initial_capital
        overall_pnl = total - initial
        overall_pct = (overall_pnl / initial * 100) if initial > 0 else 0
        win_rate = (
            (self.winning_trades / self.total_trades * 100)
            if self.total_trades > 0
            else 0
        )

        lines = [
            "=" * 60,
            "PORTFOLIO SUMMARY",
            "=" * 60,
            f"Cash:           ₹{self.cash:>12,.2f}",
            f"Positions:      {len(self.positions):>12d} / {config.MAX_POSITIONS}",
            f"Total Value:    ₹{total:>12,.2f}",
            f"Initial:        ₹{initial:>12,.2f}",
            f"Overall P&L:    ₹{overall_pnl:>+12,.2f} ({overall_pct:+.1f}%)",
            f"Realized P&L:   ₹{self.realized_pnl:>+12,.2f}",
            f"Unrealized P&L: ₹{unrealized:>+12,.2f}",
            f"Groww Fees:     ₹{self.total_fees_paid:>12,.2f}",
            f"Total Trades:   {self.total_trades:>12d}",
            f"Win Rate:       {win_rate:>11.1f}%",
            "-" * 60,
        ]

        if self.positions:
            lines.append("OPEN POSITIONS:")
            for sym, pos in self.positions.items():
                curr = current_prices.get(sym, pos["avg_price"])
                pos_pnl = (curr - pos["avg_price"]) * pos["qty"]
                pos_pct = (curr - pos["avg_price"]) / pos["avg_price"] * 100 if pos["avg_price"] > 0 else 0
                lines.append(
                    f"  {sym:<15s} {pos['qty']:>4d} @ ₹{pos['avg_price']:<10.2f} "
                    f"CMP: ₹{curr:<10.2f} P&L: ₹{pos_pnl:>+8.2f} ({pos_pct:+.1f}%)"
                )
            lines.append("-" * 60)

        return "\n".join(lines)
