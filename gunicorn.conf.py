# Gunicorn configuration for AI Trading Bot
# Flask-SocketIO with async_mode="threading" — use 1 worker + multiple threads

bind        = "127.0.0.1:5001"
workers     = 1          # MUST be 1 — scheduler & socket state live in-process
threads     = 8          # Handle concurrent HTTP + WebSocket connections
worker_class = "gthread"
timeout     = 120        # Long-running scan endpoints need more time
keepalive   = 5

# Logging
accesslog   = "/var/log/tradebot/access.log"
errorlog    = "/var/log/tradebot/error.log"
loglevel    = "info"
capture_output = True

# Process naming
proc_name   = "tradebot"

# Graceful restart
graceful_timeout = 30
max_requests     = 1000
max_requests_jitter = 50
