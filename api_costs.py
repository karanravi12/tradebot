"""API cost tracker — tracks Claude API usage and enforces daily spending limits."""

import json
import logging
import os
import threading
from datetime import date, datetime

import config

logger = logging.getLogger(__name__)

COSTS_FILE = os.path.join(os.path.dirname(__file__), "logs", "api_costs.json")

# Pricing per million tokens (USD)
MODEL_PRICING = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
}

_lock = threading.Lock()


def _load_costs() -> dict:
    """Load cost data from file."""
    if os.path.exists(COSTS_FILE):
        try:
            with open(COSTS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"daily": {}, "total_usd": 0.0, "total_calls": 0, "history": []}


def _save_costs(data: dict):
    """Save cost data to file."""
    os.makedirs(os.path.dirname(COSTS_FILE), exist_ok=True)
    with open(COSTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def record_usage(model: str, input_tokens: int, output_tokens: int, purpose: str = ""):
    """Record an API call's token usage and cost.

    Args:
        model: model ID used (e.g. "claude-sonnet-4-6")
        input_tokens: number of input tokens
        output_tokens: number of output tokens
        purpose: what the call was for (e.g. "chat", "analyze_single", "analyze_batch")
    """
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["claude-sonnet-4-6"])
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    today = date.today().isoformat()

    with _lock:
        data = _load_costs()

        # Update daily totals
        if today not in data["daily"]:
            data["daily"][today] = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }

        day = data["daily"][today]
        day["calls"] += 1
        day["input_tokens"] += input_tokens
        day["output_tokens"] += output_tokens
        day["cost_usd"] = round(day["cost_usd"] + total_cost, 6)

        # Update lifetime totals
        data["total_usd"] = round(data["total_usd"] + total_cost, 6)
        data["total_calls"] = data.get("total_calls", 0) + 1

        # Append to history (keep last 200 entries)
        data["history"].append({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "purpose": purpose,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(total_cost, 6),
        })
        data["history"] = data["history"][-200:]

        # Prune daily data older than 30 days
        all_days = sorted(data["daily"].keys())
        if len(all_days) > 30:
            for old_day in all_days[:-30]:
                del data["daily"][old_day]

        _save_costs(data)

    logger.info(
        f"[COST] {purpose}: {input_tokens} in + {output_tokens} out = "
        f"${total_cost:.4f} ({model}) | Today: ${day['cost_usd']:.4f}"
    )


def check_daily_limit() -> tuple[bool, float, float]:
    """Check if today's spending is within the daily limit.

    Returns:
        (allowed, spent_today, daily_limit) — allowed is True if under limit
    """
    daily_limit = getattr(config, "API_DAILY_LIMIT_USD", 0)
    if daily_limit <= 0:
        return True, 0.0, 0.0  # No limit set

    today = date.today().isoformat()
    data = _load_costs()
    spent = data.get("daily", {}).get(today, {}).get("cost_usd", 0.0)

    return spent < daily_limit, spent, daily_limit


def get_cost_summary() -> dict:
    """Get cost summary for the UI."""
    data = _load_costs()
    today = date.today().isoformat()
    today_data = data.get("daily", {}).get(today, {
        "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
    })

    daily_limit = getattr(config, "API_DAILY_LIMIT_USD", 0)

    # Last 7 days breakdown
    last_7_days = {}
    all_days = sorted(data.get("daily", {}).keys(), reverse=True)[:7]
    for d in all_days:
        last_7_days[d] = data["daily"][d]

    return {
        "today": {
            "calls": today_data.get("calls", 0),
            "input_tokens": today_data.get("input_tokens", 0),
            "output_tokens": today_data.get("output_tokens", 0),
            "cost_usd": round(today_data.get("cost_usd", 0), 4),
        },
        "total_usd": round(data.get("total_usd", 0), 4),
        "total_calls": data.get("total_calls", 0),
        "daily_limit_usd": daily_limit,
        "limit_remaining_usd": round(max(0, daily_limit - today_data.get("cost_usd", 0)), 4) if daily_limit > 0 else None,
        "last_7_days": last_7_days,
        "recent_calls": data.get("history", [])[-20:],
    }
