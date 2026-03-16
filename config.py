"""Trading bot configuration — stock universe, strategy parameters, risk controls."""

# ── Capital ──────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 500_000  # INR

# ── Market Hours (IST) ──────────────────────────────────────────────────────
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30
SKIP_FIRST_MINUTES = 15  # avoid opening volatility (trade from 9:30)
SCAN_INTERVAL_MINUTES = 5
TIMEZONE = "Asia/Kolkata"

# ── Risk Management ─────────────────────────────────────────────────────────
MAX_POSITIONS = 7
MAX_POSITION_PCT = 0.20       # max 20% of portfolio per stock
STOP_LOSS_PCT = 0.03          # 3% stop-loss
TRAILING_STOP_PCT = 0.02      # 2% trailing stop (optional)

# ── Claude AI Configuration ──────────────────────────────────────────────────
# API key: set via environment variable ANTHROPIC_API_KEY or via UI
CLAUDE_MODEL = "claude-sonnet-4-6"   # claude-sonnet-4-6 or claude-opus-4-6
CLAUDE_MAX_TOKENS = 1024
CLAUDE_TEMPERATURE = 0.3             # Low temperature for consistent decisions
AI_PRE_FILTER = True                 # Only send stocks with active indicator signals to AI
API_DAILY_LIMIT_USD = 5.00           # Max API spend per day in USD (0 = no limit)

AVAILABLE_MODELS = [
    {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "cost": "$3/$15 per MTok"},
    {"id": "claude-opus-4-6", "name": "Claude Opus 4.6", "cost": "$15/$75 per MTok"},
]


def update_config(key: str, value):
    """Update a config value at runtime (called from web UI)."""
    import config as _self
    if hasattr(_self, key):
        setattr(_self, key, value)

# ── Strategy Parameters ─────────────────────────────────────────────────────
# Technical
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
EMA_SHORT = 20
EMA_LONG = 50

# Supertrend
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0

# Bollinger Bands
BB_PERIOD = 20
BB_STD = 2.0
KC_PERIOD = 20
KC_MULTIPLIER = 1.5

# Ichimoku
ICHIMOKU_TENKAN = 9
ICHIMOKU_KIJUN = 26
ICHIMOKU_SENKOU_B = 52

# Fibonacci
SWING_LOOKBACK = 5
FIB_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]

# VWAP
VOLUME_SPIKE_MULTIPLIER = 2.0
VOLUME_AVG_PERIOD = 20

# Smart Money Concepts
SMC_SWING_LOOKBACK = 5
SMC_IMPULSE_THRESHOLD = 0.015  # 1.5% move to qualify as impulsive
FVG_MIN_GAP_PCT = 0.003        # 0.3% minimum gap for FVG

# ── Stock Universe (NSE) ────────────────────────────────────────────────────
# All data fetched via Groww Partner API (groww_live.py)
STOCK_SYMBOLS = [
    # ── Large-cap Banking ─────────────────────────────────────────────────
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
    "CANBK", "PNB", "BANKBARODA", "MAHABANK",
    # ── Mid-cap Banking / Small Finance ──────────────────────────────────
    "FEDERALBNK", "INDUSINDBK", "IDFCFIRSTB", "BANDHANBNK", "RBLBANK",
    "YESBANK", "AUBANK", "EQUITASBNK",
    # ── NBFC / Lending ────────────────────────────────────────────────────
    "BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "SHRIRAMFIN", "JIOFIN",
    "BAJAJHFL", "MUTHOOTFIN", "MANAPPURAM", "LICHSGFIN", "SUNDARMFIN",
    "IIFL", "M&MFIN",
    # ── IT (Large) ────────────────────────────────────────────────────────
    "INFY", "TCS", "HCLTECH", "WIPRO", "TECHM",
    # ── IT (Mid) ──────────────────────────────────────────────────────────
    "PERSISTENT", "COFORGE", "MPHASIS", "OFSS", "KPITTECH", "LTTS",
    "MASTEK",
    # ── Oil & Gas ─────────────────────────────────────────────────────────
    "IOC", "ONGC", "GAIL", "BPCL", "PETRONET", "MGL", "IGL", "GUJGASLTD",
    # ── Power ─────────────────────────────────────────────────────────────
    "POWERGRID", "NTPC", "TATAPOWER", "ADANIPOWER", "JSWENERGY",
    "ADANIGREEN", "ADANIENSOL", "NHPC", "TORNTPOWER", "CESC",
    "SJVN", "IRCON",
    # ── Power Finance ─────────────────────────────────────────────────────
    "IRFC", "RECLTD", "PFC",
    # ── Metals & Mining ───────────────────────────────────────────────────
    "TATASTEEL", "VEDL", "HINDALCO", "JSWSTEEL", "JINDALSTEL",
    "HINDZINC", "HINDCOPPER", "NMDC", "MOIL", "SAIL", "NATIONALUM",
    "WELCORP", "APLLTD",
    # ── Auto (Large) ──────────────────────────────────────────────────────
    "MOTHERSON", "M&M", "TVSMOTOR", "MARUTI", "EICHERMOT",
    "BAJAJ-AUTO", "HYUNDAI", "HEROMOTOCO",
    # ── Auto (Mid / Ancillary) ────────────────────────────────────────────
    "ASHOKLEY", "TMPV", "APOLLOTYRE", "CEATLTD", "MRF",
    "EXIDEIND", "ARE&M", "SUNDRMFAST", "BOSCHLTD",
    # ── Defence ───────────────────────────────────────────────────────────
    "BEL", "HAL", "MAZDOCK", "SOLARINDS", "BHEL", "BEML",
    "COCHINSHIP", "GRSE",
    # ── Consumer / FMCG ──────────────────────────────────────────────────
    "ITC", "HINDUNILVR", "BRITANNIA", "NESTLEIND", "TATACONSUM",
    "GODREJCP", "VBL", "UNITDSPR", "MARICO", "DABUR", "COLPAL",
    "EMAMILTD", "RADICO",
    # ── Diversified / Conglomerate ────────────────────────────────────────
    "RELIANCE", "ADANIENT", "GRASIM",
    # ── Telecom ───────────────────────────────────────────────────────────
    "BHARTIARTL", "INDUSTOWER",
    # ── Pharma / Healthcare ───────────────────────────────────────────────
    "SUNPHARMA", "DRREDDY", "CIPLA", "ZYDUSLIFE", "DIVISLAB",
    "TORNTPHARM", "MAXHEALTH", "APOLLOHOSP", "LUPIN", "AUROPHARMA",
    "BIOCON", "IPCALAB", "ALKEM", "LAURUSLABS", "GRANULES",
    # ── Insurance ─────────────────────────────────────────────────────────
    "HDFCLIFE", "SBILIFE", "ICICIGI", "LICI", "ICICIPRULI", "NIACL",
    # ── Infra / Engineering ───────────────────────────────────────────────
    "LT", "CGPOWER", "HAVELLS", "SIEMENS", "ABB", "ENGINERSIN",
    "KEC", "KPIL", "RVNL", "TITAGARH",
    # ── Real Estate ───────────────────────────────────────────────────────
    "DLF", "LODHA", "PRESTIGE", "BRIGADE", "SOBHA", "GODREJPROP",
    # ── Retail / Consumer Discretionary ──────────────────────────────────
    "TRENT", "DMART", "VMM", "PAGEIND", "RAYMOND",
    # ── Tech / Internet ───────────────────────────────────────────────────
    "ETERNAL", "NAUKRI", "TEJASNET", "REDINGTON", "NETWEB",
    "CYIENT", "DIXON", "BSE", "ANGELONE", "MOTILALOFS",
    # ── Consumer / Paint / Adhesive ──────────────────────────────────────
    "ASIANPAINT", "PIDILITIND", "TITAN", "BERGEPAINT", "KANSAINER",
    # ── Cement ────────────────────────────────────────────────────────────
    "AMBUJACEM", "ULTRACEMCO", "SHREECEM", "JKCEMENT", "RAMCOCEM",
    # ── Chemicals ─────────────────────────────────────────────────────────
    "SRF", "NAVINFLUOR", "FINEORG", "TATACHEM",
    "DEEPAKNTR", "ATUL", "CLEAN",
    # ── Infrastructure ────────────────────────────────────────────────────
    "ADANIPORTS", "CONCOR",
    # ── Aviation ──────────────────────────────────────────────────────────
    "INDIGO",
    # ── Hospitality ───────────────────────────────────────────────────────
    "INDHOTEL", "LEMONTREE", "EIHOTEL",
    # ── Media / Entertainment ─────────────────────────────────────────────
    "PVRINOX", "SUNTV", "ZEEL",
    # ── Textile ───────────────────────────────────────────────────────────
    "VTL", "WELSPUNLIV",
    # ── Holding / Misc ────────────────────────────────────────────────────
    "BAJAJHLDNG", "ENRIN",
    # ── Agri / Sugar / Others ─────────────────────────────────────────────
    "COALINDIA", "BALRAMCHIN", "POLYPLEX", "ANURAS", "AVANTIFEED",
    "KRBL",

    # ══ Commodity ETFs (NSE-listed) ════════════════════════════════════════
    # Prices reflect spot Gold/Silver; tradeable like stocks on NSE
    "GOLDBEES",    # Nippon India Gold ETF  — tracks domestic gold price
    "SILVERBEES",  # Mirae Asset Silver ETF — tracks domestic silver price
    "AXISGOLD",    # Axis Gold ETF
    "HDFCGOLD",    # HDFC Gold ETF
    "QGOLDHALF",   # Quantum Gold Fund (half-unit)
    "NIFTYBEES",   # Nippon India Nifty 50 ETF  (tracks index)
    "BANKBEES",    # Nippon India Bank ETF       (tracks Bank Nifty)
    "JUNIORBEES",  # Nippon India Junior BeES    (Nifty Next 50)
]

# NSE holidays 2026 — sourced from NSE circular (weekday holidays only;
# weekend holidays like Mahashivratri Feb-15, Id-Ul-Fitr Mar-21,
# Independence Day Aug-15, Diwali Laxmi Pujan Nov-8 fall on Sat/Sun and
# are already skipped by the Mon-Fri scheduler)
NSE_HOLIDAYS = [
    "2026-01-15",  # Municipal Corporation Election, Maharashtra
    "2026-01-26",  # Republic Day
    "2026-03-03",  # Holi
    "2026-03-26",  # Shri Ram Navami
    "2026-03-31",  # Shri Mahavir Jayanti
    "2026-04-03",  # Good Friday
    "2026-04-14",  # Dr. Baba Saheb Ambedkar Jayanti
    "2026-05-01",  # Maharashtra Day
    "2026-05-28",  # Bakri Id
    "2026-06-26",  # Muharram
    "2026-09-14",  # Ganesh Chaturthi
    "2026-10-02",  # Mahatma Gandhi Jayanti
    "2026-10-20",  # Dussehra
    "2026-11-10",  # Diwali Balipratipada
    "2026-11-24",  # Prakash Gurpurb Sri Guru Nanak Dev
    "2026-12-25",  # Christmas
]
