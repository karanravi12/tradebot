# Gunicorn configuration for AI Trading Bot
# Flask-SocketIO with async_mode="threading" — use 1 worker + multiple threads

import os
bind        = f"0.0.0.0:{os.environ.get('PORT', '5001')}"
workers     = 1          # MUST be 1 — scheduler & socket state live in-process
threads     = 8          # Handle concurrent HTTP + WebSocket connections
worker_class = "gthread"
timeout     = 120        # Long-running scan endpoints need more time
keepalive   = 5

# Logging — stdout for cloud platforms (Railway/Render), file for VPS
import os
if os.path.isdir("/var/log/tradebot"):
    accesslog      = "/var/log/tradebot/access.log"
    errorlog       = "/var/log/tradebot/error.log"
else:
    accesslog      = "-"   # stdout
    errorlog       = "-"   # stderr
loglevel           = "info"
capture_output     = True

# Process naming
proc_name   = "tradebot"

# Graceful restart
graceful_timeout = 30
max_requests     = 1000
max_requests_jitter = 50
