"""Groww intraday equity trading fee calculator.

Based on official Groww charges (NSE):
  - Brokerage:             ₹20 or 0.1% per order (whichever is lower, min ₹5)
  - STT:                   0.025% of sell order value (sell side only)
  - Exchange Tx Charge:    0.00297% of order value (both sides, NSE)
  - Stamp Duty:            0.003% of buy order value (buy side only)
  - SEBI Turnover Charge:  0.0001% of order value (both sides)
  - IPFT Charge:           0.0001% of order value (both sides, NSE)
  - GST:                   18% on (brokerage + exchange charges + SEBI + IPFT)
  - DP Charges:            ₹0 for intraday trades

Reference: https://groww.in/pricing
"""


def _brokerage(order_value: float) -> float:
    """Calculate brokerage for one order (buy or sell).

    Groww rule: ₹20 or 0.1% whichever is LOWER, minimum ₹5.
    """
    pct_charge = order_value * 0.001  # 0.1%
    if pct_charge < 5.0:
        return 5.0  # minimum brokerage is always ₹5
    return min(pct_charge, 20.0)  # cap at ₹20


def calculate_buy_fees(price: float, qty: int) -> dict:
    """Calculate all Groww fees for a BUY order.

    Returns a dict with individual components and total_fees.
    """
    order_value = price * qty

    brokerage = _brokerage(order_value)
    exchange_charge = order_value * 0.0000297   # 0.00297%
    stamp_duty = order_value * 0.00003          # 0.003%
    sebi_charge = order_value * 0.000001        # 0.0001%
    ipft_charge = order_value * 0.000001        # 0.0001%

    # GST on brokerage + exchange charges + SEBI + IPFT (not on STT/stamp)
    gst_base = brokerage + exchange_charge + sebi_charge + ipft_charge
    gst = gst_base * 0.18

    stt = 0.0          # no STT on buy for intraday
    dp_charge = 0.0    # no DP charges for intraday

    total = brokerage + exchange_charge + stamp_duty + sebi_charge + ipft_charge + gst + stt + dp_charge

    return {
        "order_value": round(order_value, 2),
        "brokerage": round(brokerage, 4),
        "stt": round(stt, 4),
        "exchange_charge": round(exchange_charge, 4),
        "stamp_duty": round(stamp_duty, 4),
        "sebi_charge": round(sebi_charge, 4),
        "ipft_charge": round(ipft_charge, 4),
        "gst": round(gst, 4),
        "dp_charge": round(dp_charge, 4),
        "total_fees": round(total, 4),
    }


def calculate_sell_fees(price: float, qty: int) -> dict:
    """Calculate all Groww fees for a SELL order.

    Returns a dict with individual components and total_fees.
    """
    order_value = price * qty

    brokerage = _brokerage(order_value)
    stt = order_value * 0.00025             # 0.025% on sell
    exchange_charge = order_value * 0.0000297
    stamp_duty = 0.0                        # no stamp duty on sell
    sebi_charge = order_value * 0.000001
    ipft_charge = order_value * 0.000001

    gst_base = brokerage + exchange_charge + sebi_charge + ipft_charge
    gst = gst_base * 0.18

    dp_charge = 0.0    # no DP charges for intraday

    total = brokerage + stt + exchange_charge + stamp_duty + sebi_charge + ipft_charge + gst + dp_charge

    return {
        "order_value": round(order_value, 2),
        "brokerage": round(brokerage, 4),
        "stt": round(stt, 4),
        "exchange_charge": round(exchange_charge, 4),
        "stamp_duty": round(stamp_duty, 4),
        "sebi_charge": round(sebi_charge, 4),
        "ipft_charge": round(ipft_charge, 4),
        "gst": round(gst, 4),
        "dp_charge": round(dp_charge, 4),
        "total_fees": round(total, 4),
    }


def calculate_round_trip_fees(buy_price: float, sell_price: float, qty: int) -> dict:
    """Calculate total fees for a complete BUY + SELL round trip.

    Args:
        buy_price:  Price at which shares were bought
        sell_price: Price at which shares were sold
        qty:        Number of shares

    Returns:
        dict with buy_fees, sell_fees, total_fees, effective_breakeven_pct
    """
    buy = calculate_buy_fees(buy_price, qty)
    sell = calculate_sell_fees(sell_price, qty)
    total = buy["total_fees"] + sell["total_fees"]

    buy_value = buy_price * qty
    # Breakeven: how much % does the stock need to move up just to cover fees
    breakeven_pct = (total / buy_value * 100) if buy_value > 0 else 0

    return {
        "buy_fees": buy,
        "sell_fees": sell,
        "total_fees": round(total, 4),
        "total_fees_inr": round(total, 2),
        "breakeven_move_pct": round(breakeven_pct, 4),
    }


def format_fee_summary(fee_dict: dict) -> str:
    """Format fee dict into a human-readable string for logs/UI."""
    f = fee_dict
    return (
        f"Brokerage: ₹{f['brokerage']:.2f} | STT: ₹{f['stt']:.2f} | "
        f"Exchg: ₹{f['exchange_charge']:.2f} | Stamp: ₹{f['stamp_duty']:.2f} | "
        f"SEBI: ₹{f['sebi_charge']:.2f} | GST: ₹{f['gst']:.2f} | "
        f"TOTAL: ₹{f['total_fees']:.2f}"
    )
