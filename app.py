"""Flask web server for the AI Trading Bot — REST API + WebSocket + UI."""

import json
import logging
import os
import sys
import threading
import queue
from datetime import datetime, date

import pytz
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO

# Load .env file for API key
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

import config
import ai_brain
import api_costs
import trader
from portfolio import Portfolio
from scheduler import start_scheduler, _is_market_holiday

# ── Logging ───────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "bot.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("trade-bot")

# ── App Setup ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24).hex()
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Wire SocketIO into trader for real-time updates
trader.set_socketio(socketio)

# Global state
portfolio = Portfolio.load()
scheduler_instance = None
IST = pytz.timezone(config.TIMEZONE)
_bot_lock = threading.Lock()   # prevents race condition on concurrent start/stop requests

# ── Auto-start bot on launch (cloud-friendly) ──────────────────────────────
# Guard: only start in the actual server process, not the werkzeug reloader watcher.
# WERKZEUG_RUN_MAIN is 'true' in the child (server) and unset in direct runs.
# It is 'false' only in the reloader parent — we skip there to avoid double schedulers.
if config.AUTO_START_BOT and os.environ.get("WERKZEUG_RUN_MAIN") != "false":
    scheduler_instance = start_scheduler(portfolio)
    logger.info("AUTO_START_BOT=true — scheduler started automatically")


# ── Pages ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Zerodha Kite auth ──────────────────────────────────────────────────────

@app.route("/zerodha/login")
def zerodha_login():
    """Redirect to Kite login page. Visit this once per day."""
    try:
        import kite_live
        url = kite_live.get_login_url()
        from flask import redirect
        return redirect(url)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/zerodha/callback")
def zerodha_callback():
    """Kite redirects here with ?request_token=xxx after login."""
    request_token = request.args.get("request_token")
    status        = request.args.get("status", "")
    if not request_token or status != "success":
        return jsonify({"error": "Login failed or cancelled", "status": status}), 400
    try:
        import kite_live
        access_token = kite_live.set_access_token(request_token)
        return jsonify({
            "message": "Zerodha login successful. Kite data source is now active.",
            "access_token_preview": access_token[:8] + "...",
        })
    except Exception as e:
        logger.error(f"Zerodha callback error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/zerodha/status")
def zerodha_status():
    """Check if Kite is authenticated."""
    try:
        import kite_live
        active   = kite_live.is_available()
        domain   = os.environ.get("RAILWAY_PUBLIC_DOMAIN")
        base_url = f"https://{domain}" if domain else "http://localhost:5001"
        return jsonify({
            "authenticated": active,
            "login_url": f"{base_url}/zerodha/login" if not active else None,
        })
    except Exception as e:
        return jsonify({"authenticated": False, "error": str(e)})


# ── Portfolio API ─────────────────────────────────────────────────────────

@app.route("/api/portfolio")
def api_portfolio():
    return jsonify(portfolio.to_dict({}))


@app.route("/api/trades")
def api_trades():
    trades = [t for t in portfolio.trade_history if t.get("action") in ("BUY", "SELL")]
    return jsonify(trades[-50:])


# ── Config API ────────────────────────────────────────────────────────────

@app.route("/api/config", methods=["GET"])
def api_get_config():
    return jsonify({
        "claude_model": config.CLAUDE_MODEL,
        "claude_max_tokens": config.CLAUDE_MAX_TOKENS,
        "claude_temperature": config.CLAUDE_TEMPERATURE,
        "max_positions": config.MAX_POSITIONS,
        "max_position_pct": config.MAX_POSITION_PCT,
        "stop_loss_pct": config.STOP_LOSS_PCT,
        "scan_interval": config.SCAN_INTERVAL_MINUTES,
        "ai_pre_filter": config.AI_PRE_FILTER,
        "api_daily_limit": config.API_DAILY_LIMIT_USD,
        "available_models": config.AVAILABLE_MODELS,
        "has_api_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
    })


@app.route("/api/config", methods=["POST"])
def api_update_config():
    data = request.json
    updated = []

    if "api_key" in data and data["api_key"]:
        os.environ["ANTHROPIC_API_KEY"] = data["api_key"]
        # Persist to .env file
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        with open(env_path, "w") as f:
            f.write(f"ANTHROPIC_API_KEY={data['api_key']}\n")
        updated.append("api_key")

    field_map = {
        "claude_model": ("CLAUDE_MODEL", str),
        "claude_max_tokens": ("CLAUDE_MAX_TOKENS", int),
        "claude_temperature": ("CLAUDE_TEMPERATURE", float),
        "max_positions": ("MAX_POSITIONS", int),
        "max_position_pct": ("MAX_POSITION_PCT", float),
        "stop_loss_pct": ("STOP_LOSS_PCT", float),
        "scan_interval": ("SCAN_INTERVAL_MINUTES", int),
        "ai_pre_filter": ("AI_PRE_FILTER", bool),
        "api_daily_limit": ("API_DAILY_LIMIT_USD", float),
    }

    for key, (config_key, cast) in field_map.items():
        if key in data:
            config.update_config(config_key, cast(data[key]))
            updated.append(key)

    logger.info(f"Config updated: {updated}")
    return jsonify({"updated": updated})


# ── Wallet API ────────────────────────────────────────────────────────────

@app.route("/api/wallet/deposit", methods=["POST"])
def api_deposit():
    amount = request.json.get("amount", 0)
    result = portfolio.deposit(float(amount))
    if "error" not in result:
        socketio.emit("portfolio_update", portfolio.to_dict({}))
    return jsonify(result)


@app.route("/api/wallet/withdraw", methods=["POST"])
def api_withdraw():
    amount = request.json.get("amount", 0)
    result = portfolio.withdraw(float(amount))
    if "error" not in result:
        socketio.emit("portfolio_update", portfolio.to_dict({}))
    return jsonify(result)


# ── Bot Control API ──────────────────────────────────────────────────────

def _safe_get_ai_eval() -> dict | None:
    try:
        return trader.get_ai_eval_info()
    except Exception as e:
        logger.warning(f"get_ai_eval_info failed: {e}")
        return None


@app.route("/api/bot/status")
def api_bot_status():
    now = datetime.now(IST)
    market_open = now.replace(hour=config.MARKET_OPEN_HOUR, minute=config.MARKET_OPEN_MINUTE, second=0)
    market_close = now.replace(hour=config.MARKET_CLOSE_HOUR, minute=config.MARKET_CLOSE_MINUTE, second=0)
    is_market_open = (
        now.weekday() < 5
        and market_open <= now <= market_close
        and not _is_market_holiday()
    )

    return jsonify({
        "bot_running": scheduler_instance is not None and scheduler_instance.running,
        "market_open": is_market_open,
        "current_time": now.strftime("%Y-%m-%d %H:%M:%S IST"),
        "market_hours": f"{config.MARKET_OPEN_HOUR}:{config.MARKET_OPEN_MINUTE:02d} - {config.MARKET_CLOSE_HOUR}:{config.MARKET_CLOSE_MINUTE:02d}",
        "is_holiday": _is_market_holiday(),
        "ai_eval": _safe_get_ai_eval(),
    })


@app.route("/api/bot/start", methods=["POST"])
def api_bot_start():
    global scheduler_instance
    with _bot_lock:
        if scheduler_instance and scheduler_instance.running:
            return jsonify({"status": "already_running"})
        # Shut down any lingering (non-running) scheduler before creating a fresh one.
        if scheduler_instance:
            scheduler_instance.shutdown(wait=False)
        scheduler_instance = start_scheduler(portfolio)
    socketio.emit("bot_status", {"running": True})
    return jsonify({"status": "started"})


@app.route("/api/bot/stop", methods=["POST"])
def api_bot_stop():
    global scheduler_instance
    with _bot_lock:
        if scheduler_instance and scheduler_instance.running:
            scheduler_instance.shutdown(wait=False)
            scheduler_instance = None
            socketio.emit("bot_status", {"running": False})
            return jsonify({"status": "stopped"})
    return jsonify({"status": "not_running"})


@app.route("/api/ai/test")
def api_ai_test():
    """Test if AI (Claude) is callable — returns status and sample response."""
    try:
        import ai_brain
        response = ai_brain.chat("Reply with exactly: AI is working.", {}, [])
        return jsonify({
            "ok": True,
            "message": "AI is callable",
            "response_preview": (response or "")[:200],
        })
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e), "hint": "Set ANTHROPIC_API_KEY in .env"}), 400
    except Exception as e:
        logger.exception("AI test failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/groww/test")
def api_groww_test():
    """Test the Groww live data connection."""
    try:
        import groww_live
        result = groww_live.test_connection()
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/scan", methods=["POST"])
def api_manual_scan():
    def run_scan():
        trader.scan_and_trade(portfolio)
    threading.Thread(target=run_scan, daemon=True).start()
    return jsonify({"status": "scan_started"})


@app.route("/api/portfolio/reset", methods=["POST"])
def api_reset_portfolio():
    portfolio.reset()
    socketio.emit("portfolio_update", portfolio.to_dict({}))
    return jsonify({"status": "reset"})


# ── API Costs ────────────────────────────────────────────────────────

@app.route("/api/costs")
def api_costs_summary():
    return jsonify(api_costs.get_cost_summary())


# ── Backtest API ───────────────────────────────────────────────────────────

_backtest_running = False
_backtest_queue = None
_BACKTEST_SENTINEL = object()


def _backtest_stream_generator():
    """SSE generator: reads from queue until complete."""
    global _backtest_queue
    if _backtest_queue is None:
        yield f"data: {json.dumps({'type': 'error', 'msg': 'No backtest running'})}\n\n"
        return
    try:
        while True:
            try:
                item = _backtest_queue.get(timeout=0.5)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if item is _BACKTEST_SENTINEL:
                break
            if isinstance(item, dict):
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("type") == "complete":
                    break
    except GeneratorExit:
        pass


@app.route("/api/backtest/stream")
def api_backtest_stream():
    """SSE endpoint for backtest live output. Connect after POST /api/backtest."""
    def gen():
        for chunk in _backtest_stream_generator():
            yield chunk
    return Response(
        gen(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    """Run backtest for given date range. Stream via GET /api/backtest/stream (SSE)."""
    global _backtest_running, _backtest_queue
    if _backtest_running:
        return jsonify({"error": "Backtest already running"}), 409

    data = request.json or {}
    start_str = data.get("start_date", "")
    end_str   = data.get("end_date", "")
    stream   = data.get("stream", True)
    if not start_str or not end_str:
        return jsonify({"error": "start_date and end_date required (YYYY-MM-DD)"}), 400
    try:
        start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
        end_date   = datetime.strptime(end_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
    if start_date > end_date:
        return jsonify({"error": "start_date must be before end_date"}), 400

    if not stream:
        try:
            from backtest import run_backtest
            result = run_backtest(start_date, end_date, silent=True)
            if "error" in result:
                return jsonify(result), 400
            return jsonify(result)
        except Exception as e:
            logger.exception("Backtest failed")
            return jsonify({"error": str(e), "trades": [], "summary": {}, "equity_curve": []}), 500

    _backtest_queue = queue.Queue()

    def _run():
        global _backtest_running, _backtest_queue
        _backtest_running = True
        try:
            def emit(msg):
                if _backtest_queue:
                    _backtest_queue.put({"type": "event", "msg": msg, "ts": datetime.now().isoformat()})
            _backtest_queue.put({"type": "started", "start_date": start_str, "end_date": end_str})
            from backtest import run_backtest
            result = run_backtest(start_date, end_date, emit_fn=emit)
            _backtest_queue.put({"type": "complete", "data": result})
        except Exception as e:
            logger.exception("Backtest failed")
            _backtest_queue.put({"type": "complete", "data": {"error": str(e), "trades": [], "summary": {}, "equity_curve": []}})
        finally:
            _backtest_queue.put(_BACKTEST_SENTINEL)
            _backtest_running = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started", "message": "Backtest running — connect to stream", "stream_url": "/api/backtest/stream"})


# ── Chat API ──────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def api_chat():
    message = request.json.get("message", "")
    if not message.strip():
        return jsonify({"error": "Empty message"})

    portfolio_state = portfolio.to_dict({})
    recent_trades = [t for t in portfolio.trade_history if t.get("action") in ("BUY", "SELL")]

    response = ai_brain.chat(message, portfolio_state, recent_trades[-10:])
    return jsonify({"response": response})


# ── Start ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  AI TRADING BOT — WEB UI")
    logger.info(f"  http://localhost:5001")
    logger.info("=" * 60)
    socketio.run(app, host="0.0.0.0", port=5001, debug=False, allow_unsafe_werkzeug=True)
