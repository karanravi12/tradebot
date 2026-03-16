"""Entry point for the Indian Stock Market Paper Trading Bot."""

import logging
import os
import signal
import sys
import time
from datetime import datetime

import pytz

import config
from portfolio import Portfolio
from scheduler import start_scheduler

# ── Logging Setup ─────────────────────────────────────────────────────────
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


def main():
    IST = pytz.timezone(config.TIMEZONE)
    now = datetime.now(IST)

    logger.info("=" * 60)
    logger.info("  INDIAN STOCK MARKET PAPER TRADING BOT")
    logger.info("=" * 60)
    logger.info(f"  Started at:    {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"  Initial capital: ₹{config.INITIAL_CAPITAL:,.2f}")
    logger.info(f"  Stock universe:  {len(config.STOCK_SYMBOLS)} stocks")
    logger.info(f"  Max positions:   {config.MAX_POSITIONS}")
    logger.info(f"  Stop-loss:       {config.STOP_LOSS_PCT * 100:.0f}%")
    logger.info(f"  Scan interval:   {config.SCAN_INTERVAL_MINUTES} minutes")
    logger.info(f"  Market hours:    {config.MARKET_OPEN_HOUR}:{config.MARKET_OPEN_MINUTE:02d} - "
                f"{config.MARKET_CLOSE_HOUR}:{config.MARKET_CLOSE_MINUTE:02d} IST")
    logger.info("=" * 60)

    # Load or create portfolio
    portfolio = Portfolio.load()
    logger.info(f"\n{portfolio.summary({})}")

    # Start scheduler
    sched = start_scheduler(portfolio)

    # Graceful shutdown handler
    def shutdown(signum, frame):
        logger.info("\nShutting down...")
        sched.shutdown(wait=False)
        portfolio.save()
        logger.info("Portfolio saved. Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("Bot is running. Press Ctrl+C to stop.\n")

    # Keep main thread alive
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        shutdown(None, None)


if __name__ == "__main__":
    main()
