"""Scheduler — runs the trading bot at fixed intervals during NSE market hours.

Uses a simple threading loop instead of APScheduler so it survives macOS sleep/wake.
Every 30 seconds it checks datetime.now() — when the Mac wakes from sleep, the thread
resumes and immediately sees the correct time, firing within ≤30s of wake.
"""

import logging
import threading
from datetime import datetime, date

import pytz

import config
from portfolio import Portfolio
import trader

logger = logging.getLogger(__name__)
IST = pytz.timezone(config.TIMEZONE)


def _is_market_holiday() -> bool:
    """Check if today is an NSE holiday."""
    today = date.today().isoformat()
    return today in config.NSE_HOLIDAYS


def _should_scan_now() -> bool:
    """Return True if current time is within trading hours (Mon-Fri, post-open buffer)."""
    if _is_market_holiday():
        return False
    now = datetime.now(IST)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    market_open = now.replace(
        hour=config.MARKET_OPEN_HOUR,
        minute=config.MARKET_OPEN_MINUTE + config.SKIP_FIRST_MINUTES,
        second=0,
        microsecond=0,
    )
    market_close = now.replace(
        hour=config.MARKET_CLOSE_HOUR,
        minute=config.MARKET_CLOSE_MINUTE,
        second=0,
        microsecond=0,
    )
    return market_open <= now <= market_close


class SimpleScheduler:
    """Lightweight scheduler that survives macOS sleep/wake cycles.

    Wakes every 30 seconds, checks the real wall-clock time, and fires
    scan_and_trade() whenever the current minute aligns with the scan interval.
    Uses _last_scan_minute to prevent double-firing within the same minute.
    """

    def __init__(self):
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.running = False
        self._last_scan_minute = -1  # total minutes from midnight of last fired scan

    def _loop(self, portfolio: Portfolio):
        logger.info("Scheduler thread started (sleep/wake resilient)")
        while not self._stop.is_set():
            try:
                now = datetime.now(IST)
                total_minute = now.hour * 60 + now.minute
                interval = config.SCAN_INTERVAL_MINUTES

                if (
                    _should_scan_now()
                    and total_minute % interval == 0
                    and total_minute != self._last_scan_minute
                ):
                    self._last_scan_minute = total_minute
                    logger.info(f"Scheduler firing scan at {now.strftime('%H:%M IST')}")
                    try:
                        trader.scan_and_trade(portfolio)
                    except Exception as exc:
                        logger.error(f"Scan error: {exc}", exc_info=True)

            except Exception as exc:
                logger.error(f"Scheduler loop error: {exc}", exc_info=True)

            # Sleep in short bursts — resumes quickly after macOS wake
            self._stop.wait(timeout=30)

        self.running = False
        logger.info("Scheduler thread stopped")

    def start(self, portfolio: Portfolio):
        self._stop.clear()
        self.running = True
        self._thread = threading.Thread(
            target=self._loop, args=[portfolio], daemon=True, name="TradingScheduler"
        )
        self._thread.start()

    def shutdown(self, wait: bool = False):
        self._stop.set()
        self.running = False
        if wait and self._thread:
            self._thread.join(timeout=5)


def start_scheduler(portfolio: Portfolio) -> SimpleScheduler:
    """Start the scheduler and return the instance."""
    scheduler = SimpleScheduler()
    scheduler.start(portfolio)
    logger.info(
        f"Scheduler started: every {config.SCAN_INTERVAL_MINUTES} min, "
        f"Mon-Fri {config.MARKET_OPEN_HOUR}:{config.MARKET_OPEN_MINUTE:02d}"
        f"–{config.MARKET_CLOSE_HOUR}:{config.MARKET_CLOSE_MINUTE:02d} IST"
    )
    return scheduler
