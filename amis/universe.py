"""
Global explanatory universe and target catalogue.
=================================================================

The Market Valuation Engine prices an asset *relative to the world*, so the
explanatory panel has to span the risk factors that actually move global
capital: equity beta and its style/sector/regional decomposition, the whole
term structure, credit across the quality spectrum, inflation compensation,
commodities by complex, the dollar and the EM currency block, volatility, and
real assets.

This catalogue merges the original AMIS panel with the full predictor
universe of the Tattva terminal (its ``GLOBAL_MACRO_MAP`` and
``MACRO_SYMBOLS_YF``), deduplicated by ticker and re-cut into finer blocks.
The taxonomy is deliberately more granular than "equities / bonds /
commodities" because the block layer of the valuation engine reports a
coefficient per block: "Rates — Government" and "Credit — High Yield" moving
in opposite directions is a legible statement about a valuation, whereas
"Fixed Income" netting them to zero is not.

The list is deliberately redundant.  Redundancy is removed statistically --
random-matrix eigenvalue clipping in :mod:`amis.factors` -- rather than by
hand-picking, so the cost of an extra correlated line is a fraction of a
noise eigenvalue, not a distortion.

Single-name equities are targets, never predictors.  Putting them in the
explanatory panel would let a target be "explained" by its own sector-mates
at 0.9 correlation, which is valuation by tautology.

References
----------
Ross, S. A. (1976). "The arbitrage theory of capital asset pricing."
    *Journal of Economic Theory* 13(3).
Ilmanen, A. (2011). *Expected Returns*. Wiley -- the cross-asset factor
    taxonomy this grouping follows.
"""

from __future__ import annotations

# ===========================================================================
# EXPLANATORY UNIVERSE — ticker : label, grouped by economic block
# ===========================================================================

EQUITY_US_BROAD = {
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow 30", "IWM": "Russell 2000",
    "MDY": "S&P Midcap 400", "RSP": "S&P 500 Equal Weight", "VTI": "US Total Market",
    "IJR": "S&P Smallcap 600", "IJH": "S&P Midcap", "OEF": "S&P 100",
    "ACWI": "Global Equity (ACWI)",
}

EQUITY_US_SECTOR = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care",
    "XLI": "Industrials", "XLY": "Cons. Discretionary", "XLP": "Cons. Staples",
    "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real Estate",
    "XLC": "Communication Svcs",
}

EQUITY_US_INDUSTRY = {
    "SMH": "Semiconductors", "SOXX": "Semiconductors (alt)", "IGV": "Software",
    "XBI": "Biotech (EW)", "IBB": "Biotech", "ITB": "Homebuilders",
    "XHB": "Homebuilding", "XRT": "Retail", "XME": "Metals & Mining",
    "XOP": "Oil & Gas E&P", "OIH": "Oil Services", "KRE": "Regional Banks",
    "KBE": "Banks", "IYT": "Transports", "JETS": "Airlines", "ITA": "Aerospace/Defense",
    "HACK": "Cybersecurity", "SKYY": "Cloud", "FDN": "Internet",
    "PAVE": "Infrastructure", "MOO": "Agribusiness", "PBW": "Clean Energy",
    "TAN": "Solar", "URA": "Uranium", "LIT": "Lithium & Battery",
    "COPX": "Copper Miners", "PICK": "Global Miners", "SLX": "Steel",
    "REMX": "Rare Earth / Strategic Metals", "IGF": "Global Infrastructure",
    "WOOD": "Timber & Forestry",
}

EQUITY_US_STYLE = {
    "MTUM": "Momentum", "QUAL": "Quality", "VLUE": "Value", "USMV": "Min Volatility",
    "SPHB": "High Beta", "SPLV": "Low Volatility", "IWF": "Large Growth",
    "IWD": "Large Value", "SPYG": "S&P Growth", "SPYV": "S&P Value",
    "VIG": "Dividend Growth", "SDY": "Dividend Aristocrats", "SIZE": "Size Factor",
    "VTV": "Value (VTV)", "VUG": "Growth (VUG)", "VYM": "High Dividend",
}

EQUITY_INTERNATIONAL = {
    "EFA": "EAFE", "IEFA": "Core EAFE", "EEM": "Emerging Markets", "IEMG": "Core EM",
    "VGK": "Europe", "VPL": "Pacific", "AAXJ": "Asia ex-Japan", "SCZ": "Intl Small Cap",
    "EWJ": "Japan", "EWG": "Germany", "EWU": "United Kingdom", "EWQ": "France",
    "EWC": "Canada", "EWA": "Australia", "EWH": "Hong Kong", "EWY": "South Korea",
    "EWT": "Taiwan", "EWZ": "Brazil", "EWW": "Mexico", "EWS": "Singapore",
    "EWI": "Italy", "EWP": "Spain", "EWD": "Sweden", "EWL": "Switzerland",
    "EWN": "Netherlands", "EZU": "Eurozone", "FXI": "China Large Cap",
    "MCHI": "China Broad", "KWEB": "China Internet", "EZA": "South Africa",
    "ILF": "Latin America", "TUR": "Turkey", "VNM": "Vietnam",
    "EPHE": "Philippines", "EIDO": "Indonesia", "UAE": "United Arab Emirates",
}

EQUITY_INDIA = {
    "INDA": "India (MSCI)", "EPI": "India Earnings",
    "LIQUIDBEES.NS": "India Overnight Liquid",
    "NIFTYIETF.NS": "Nifty 50 ETF", "MONIFTY500.NS": "Nifty 500 ETF",
    "MIDCAPIETF.NS": "India Midcap ETF", "MOSMALL250.NS": "India Smallcap ETF",
    "ITIETF.NS": "India IT ETF", "PSUBNKIETF.NS": "India PSU Bank ETF",
    "PVTBANIETF.NS": "India Pvt Bank ETF", "FINIETF.NS": "India Financials ETF",
    "AUTOIETF.NS": "India Auto ETF", "FMCGIETF.NS": "India FMCG ETF",
    "HEALTHIETF.NS": "India Healthcare ETF", "METALIETF.NS": "India Metals ETF",
    "OILIETF.NS": "India Energy ETF", "INFRAIETF.NS": "India Infra ETF",
    "CONSUMIETF.NS": "India Consumption ETF", "CPSEETF.NS": "India CPSE ETF",
    "COMMOIETF.NS": "India Commodities ETF", "MOREALTY.NS": "India Realty ETF",
    "CHEMICAL.NS": "India Chemicals ETF", "MODEFENCE.NS": "India Defence ETF",
    "MAKEINDIA.NS": "India Manufacturing ETF", "MNC.NS": "India MNC ETF",
}

GLOBAL_INDEX = {
    "^GSPC": "S&P 500 Index", "^NDX": "Nasdaq 100 Index", "^DJI": "Dow Jones",
    "^RUT": "Russell 2000 Index", "^NYA": "NYSE Composite", "^FTSE": "FTSE 100",
    "^GDAXI": "DAX", "^FCHI": "CAC 40", "^STOXX50E": "Euro Stoxx 50",
    "^N225": "Nikkei 225", "^HSI": "Hang Seng", "^AXJO": "ASX 200",
    "^BSESN": "BSE Sensex", "^NSEI": "Nifty 50", "^KS11": "KOSPI",
    "^KQ11": "KOSDAQ", "^TWII": "Taiwan Weighted", "^BVSP": "Bovespa",
    "^MXX": "IPC Mexico", "^GSPTSE": "TSX Composite", "^SSMI": "SMI",
    "^AEX": "AEX", "^IBEX": "IBEX 35", "^OMX": "OMX Stockholm",
    "^JKSE": "Jakarta Composite", "^STI": "Straits Times",
    "000001.SS": "Shanghai Composite", "399001.SZ": "Shenzhen Component",
}

RATES_GOVERNMENT = {
    "BIL": "US T-Bills 1-3M", "SGOV": "US T-Bills 0-3M", "SHV": "US Ultra-Short 0-1Y",
    "SHY": "US Treasury 1-3Y", "VGSH": "US Treasury Short (Vanguard)",
    "IEI": "US Treasury 3-7Y", "IEF": "US Treasury 7-10Y",
    "VGIT": "US Treasury Intermediate (Vanguard)", "TLH": "US Treasury 10-20Y",
    "TLT": "US Treasury 20Y+", "VGLT": "US Treasury Long (Vanguard)",
    "GOVT": "US Treasury Broad", "EDV": "Extended Duration", "ZROZ": "Zero Coupon 25Y+",
    "BSV": "Short-Term Broad Bond", "BLV": "Long-Term Broad Bond",
    "AGG": "US Core Aggregate", "BND": "US Total Bond",
    "BNDW": "Global Aggregate (Hedged)", "BNDX": "Intl Bond ex-US",
    "IGOV": "Intl Treasury ex-US", "BWX": "Intl Treasury (SPDR)",
    "IEGA.L": "Eurozone Government", "IBGL.L": "Germany Bunds (Long)",
    "SDEU.L": "Germany Schatz (Short)", "IGLT.L": "UK Gilts",
    "INXG.L": "UK Gilts Inflation-Linked", "VGB.AX": "Australia Government",
    "XBB.TO": "Canada Aggregate", "IIND.L": "India Govt (LSE proxy)",
    "LTGILTBEES.NS": "India 8-13Y G-Sec", "GILT5YBEES.NS": "India 5Y G-Sec",
    "CBON": "China Government", "CNYB.L": "China CNY Local",
}

RATES_YIELDS = {
    "^IRX": "US 13-Week Bill Yield", "^FVX": "US 5Y Yield",
    "^TNX": "US 10Y Yield", "^TYX": "US 30Y Yield",
}

CREDIT = {
    "LQD": "US IG Corporate", "VCSH": "US IG Short 1-5Y", "VCIT": "US IG Intermediate",
    "VCLT": "US IG Long", "IBND": "International Corporate",
    "IEAC.L": "Eurozone IG Corporate", "SLXX.L": "UK Corporate",
    "HYG": "US High Yield", "JNK": "US High Yield (SPDR)", "SJNK": "Short High Yield",
    "GHYG": "Global High Yield", "FALN": "Fallen Angels", "BGRN": "Global Green Bond",
    "PFF": "Preferred Stock", "CWB": "Convertible Bonds", "BKLN": "Senior Loans",
    "MBB": "Agency MBS", "VMBS": "Agency MBS (Vanguard)", "FLOT": "Floating Rate",
    "MUB": "US Municipal National", "VTEB": "US Municipal Tax-Exempt",
    "EMB": "EM Sovereign USD", "PCY": "EM Sovereign USD (Invesco)",
    "EMLC": "EM Sovereign Local", "EMHY": "EM High Yield Corporate",
    "EBBETF0430.NS": "India AAA PSU Bond",
}

INFLATION_LINKED = {
    "TIP": "US TIPS Broad", "VTIP": "US TIPS Short", "STIP": "US TIPS 0-5Y",
    "SCHP": "US TIPS (Schwab)", "WIP": "International Inflation-Linked",
    "RINF": "Inflation Expectations",
}

COMMODITY_BROAD = {
    "DBC": "Broad Commodity (DBC)", "GSG": "GSCI Commodity",
    "PDBC": "Optimum Yield Commodity", "DBB": "Base Metals",
}

COMMODITY_METALS = {
    "GLD": "Gold (GLD)", "IAU": "Gold (IAU)", "SLV": "Silver", "PPLT": "Platinum",
    "PALL": "Palladium", "GLTR": "Precious Metals Basket", "GDX": "Gold Miners",
    "GDXJ": "Junior Gold Miners", "SIL": "Silver Miners",
    "GC=F": "Gold Futures", "SI=F": "Silver Futures", "HG=F": "Copper Futures",
    "PL=F": "Platinum Futures", "CPER": "Copper (CPER)",
    "GOLDIETF.NS": "India Gold ETF", "SILVERIETF.NS": "India Silver ETF",
}

COMMODITY_ENERGY = {
    "USO": "WTI Crude (USO)", "BNO": "Brent Crude (BNO)", "DBO": "Oil (DBO)",
    "UNG": "Natural Gas", "CL=F": "WTI Futures", "BZ=F": "Brent Futures",
    "NG=F": "Nat Gas Futures", "RB=F": "RBOB Gasoline", "HO=F": "Heating Oil",
}

COMMODITY_AGRICULTURE = {
    "DBA": "Agriculture (DBA)", "CORN": "Corn (ETF)", "WEAT": "Wheat (ETF)",
    "SOYB": "Soybeans (ETF)", "ZC=F": "Corn Futures", "ZW=F": "Wheat Futures",
    "ZS=F": "Soybean Futures", "ZL=F": "Soybean Oil", "CT=F": "Cotton",
    "KC=F": "Coffee", "SB=F": "Sugar", "CC=F": "Cocoa", "LE=F": "Live Cattle",
}

CURRENCY_MAJOR = {
    "DX-Y.NYB": "US Dollar Index", "UUP": "USD Bullish (UUP)",
    "UDN": "USD Bearish (UDN)", "USDU": "USD Bullish Broad",
    "FXE": "Euro", "FXY": "Japanese Yen", "FXB": "British Pound",
    "FXF": "Swiss Franc", "FXA": "Australian Dollar", "FXC": "Canadian Dollar",
    "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "AUDUSD=X": "AUD/USD",
    "JPY=X": "USD/JPY", "USDCAD=X": "USD/CAD", "USDCHF=X": "USD/CHF",
    "USDSEK=X": "USD/SEK", "USDNOK=X": "USD/NOK",
}

CURRENCY_EM = {
    "CEW": "EM Currencies (CEW)", "INR=X": "USD/INR", "CNY=X": "USD/CNY",
    "CNH=X": "USD/CNH", "KRW=X": "USD/KRW", "USDMXN=X": "USD/MXN",
    "BRL=X": "USD/BRL", "ZAR=X": "USD/ZAR", "THB=X": "USD/THB",
    "TWD=X": "USD/TWD", "MYR=X": "USD/MYR", "IDR=X": "USD/IDR",
    "PHP=X": "USD/PHP", "SGD=X": "USD/SGD", "USDTRY=X": "USD/TRY",
    "USDVND=X": "USD/VND", "EURINR=X": "EUR/INR", "GBPINR=X": "GBP/INR",
    "JPYINR=X": "JPY/INR",
}

VOLATILITY = {
    "^VIX": "VIX", "^VIX3M": "3-Month VIX", "^VXN": "Nasdaq VIX",
    "^RVX": "Russell VIX", "^OVX": "Crude Oil VIX", "^GVZ": "Gold VIX",
    "^MOVE": "MOVE (Bond Volatility)", "VIXM": "Mid-Term VIX Futures",
}

REAL_ASSETS = {
    "VNQ": "US REITs", "IYR": "US Real Estate", "VNQI": "International REITs",
    "REET": "Global REITs",
}

EQUITY_FUTURES = {
    "ES=F": "S&P 500 Futures", "NQ=F": "Nasdaq Futures", "YM=F": "Dow Futures",
    "RTY=F": "Russell Futures", "ZB=F": "30Y Bond Futures",
    "ZN=F": "10Y Note Futures", "ZF=F": "5Y Note Futures",
    "6E=F": "Euro FX Futures", "6J=F": "Yen Futures",
}

DIGITAL = {
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum",
}

#: Economic block -> {ticker: label}.  Order is the sidebar order.
ASSET_CLASSES: dict[str, dict[str, str]] = {
    "US Equity — Broad": EQUITY_US_BROAD,
    "US Equity — Sector": EQUITY_US_SECTOR,
    "US Equity — Industry & Theme": EQUITY_US_INDUSTRY,
    "US Equity — Style Factor": EQUITY_US_STYLE,
    "International Equity": EQUITY_INTERNATIONAL,
    "India Equity & ETFs": EQUITY_INDIA,
    "Global Index": GLOBAL_INDEX,
    "Rates — Government": RATES_GOVERNMENT,
    "Rates — Yield Indices": RATES_YIELDS,
    "Credit": CREDIT,
    "Inflation-Linked": INFLATION_LINKED,
    "Commodity — Broad": COMMODITY_BROAD,
    "Commodity — Metals": COMMODITY_METALS,
    "Commodity — Energy": COMMODITY_ENERGY,
    "Commodity — Agriculture": COMMODITY_AGRICULTURE,
    "Currency — Major": CURRENCY_MAJOR,
    "Currency — Emerging": CURRENCY_EM,
    "Volatility": VOLATILITY,
    "Real Assets & REITs": REAL_ASSETS,
    "Futures": EQUITY_FUTURES,
    "Digital Asset": DIGITAL,
}

UNIVERSE: dict[str, str] = {t: n for grp in ASSET_CLASSES.values() for t, n in grp.items()}
TICKER_CLASS: dict[str, str] = {t: cls for cls, grp in ASSET_CLASSES.items() for t in grp}
UNIVERSE_TICKERS: list[str] = sorted(UNIVERSE)


# ===========================================================================
# TARGETS — what the user can point the engines at
# ===========================================================================
# Every explanatory instrument is also a valid target (self-excluded from its
# own panel).  On top of that: single-name equities, and the free-form symbol
# classes for US and India stocks.

SINGLE_NAMES_US: dict[str, str] = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon",
    "GOOGL": "Alphabet", "META": "Meta Platforms", "TSLA": "Tesla",
    "BRK-B": "Berkshire Hathaway", "JPM": "JPMorgan Chase", "V": "Visa",
    "MA": "Mastercard", "UNH": "UnitedHealth", "JNJ": "Johnson & Johnson",
    "XOM": "Exxon Mobil", "CVX": "Chevron", "WMT": "Walmart", "PG": "Procter & Gamble",
    "HD": "Home Depot", "KO": "Coca-Cola", "PEP": "PepsiCo", "MRK": "Merck",
    "ABBV": "AbbVie", "LLY": "Eli Lilly", "PFE": "Pfizer", "AVGO": "Broadcom",
    "AMD": "AMD", "INTC": "Intel", "MU": "Micron", "QCOM": "Qualcomm",
    "TXN": "Texas Instruments", "CRM": "Salesforce", "ORCL": "Oracle",
    "ADBE": "Adobe", "NFLX": "Netflix", "DIS": "Disney", "BA": "Boeing",
    "CAT": "Caterpillar", "DE": "Deere", "GE": "GE Aerospace", "HON": "Honeywell",
    "MMM": "3M", "UPS": "UPS", "FDX": "FedEx", "GS": "Goldman Sachs",
    "MS": "Morgan Stanley", "BAC": "Bank of America", "C": "Citigroup",
    "WFC": "Wells Fargo", "BLK": "BlackRock", "SCHW": "Charles Schwab",
    "COST": "Costco", "MCD": "McDonald's", "NKE": "Nike", "SBUX": "Starbucks",
    "T": "AT&T", "VZ": "Verizon", "CMCSA": "Comcast", "NEE": "NextEra Energy",
    "DUK": "Duke Energy", "SO": "Southern Company", "LIN": "Linde",
    "FCX": "Freeport-McMoRan", "NEM": "Newmont", "COP": "ConocoPhillips",
    "SLB": "SLB", "OXY": "Occidental", "PLTR": "Palantir", "UBER": "Uber",
    "ABNB": "Airbnb", "COIN": "Coinbase", "SHOP": "Shopify", "SQ": "Block",
    "PYPL": "PayPal", "SNOW": "Snowflake", "NOW": "ServiceNow", "PANW": "Palo Alto",
    "SMCI": "Super Micro", "ARM": "Arm Holdings", "TSM": "TSMC", "ASML": "ASML",
    "BABA": "Alibaba", "TM": "Toyota", "SAP": "SAP", "SHEL": "Shell",
    "NVO": "Novo Nordisk", "AZN": "AstraZeneca", "HSBC": "HSBC", "RIO": "Rio Tinto",
    "BHP": "BHP Group",
}

SINGLE_NAMES_INDIA: dict[str, str] = {
    "RELIANCE.NS": "Reliance Industries", "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank", "ICICIBANK.NS": "ICICI Bank", "INFY.NS": "Infosys",
    "BHARTIARTL.NS": "Bharti Airtel", "SBIN.NS": "State Bank of India",
    "LT.NS": "Larsen & Toubro", "ITC.NS": "ITC", "HINDUNILVR.NS": "Hindustan Unilever",
    "AXISBANK.NS": "Axis Bank", "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "BAJFINANCE.NS": "Bajaj Finance", "MARUTI.NS": "Maruti Suzuki",
    "M&M.NS": "Mahindra & Mahindra", "TATAMOTORS.NS": "Tata Motors",
    "SUNPHARMA.NS": "Sun Pharmaceutical", "TITAN.NS": "Titan Company",
    "ULTRACEMCO.NS": "UltraTech Cement", "ASIANPAINT.NS": "Asian Paints",
    "NTPC.NS": "NTPC", "POWERGRID.NS": "Power Grid", "ONGC.NS": "ONGC",
    "COALINDIA.NS": "Coal India", "TATASTEEL.NS": "Tata Steel",
    "JSWSTEEL.NS": "JSW Steel", "HINDALCO.NS": "Hindalco", "WIPRO.NS": "Wipro",
    "HCLTECH.NS": "HCL Technologies", "TECHM.NS": "Tech Mahindra",
    "ADANIENT.NS": "Adani Enterprises", "ADANIPORTS.NS": "Adani Ports",
    "NESTLEIND.NS": "Nestlé India", "BAJAJFINSV.NS": "Bajaj Finserv",
    "INDUSINDBK.NS": "IndusInd Bank", "GRASIM.NS": "Grasim Industries",
    "DRREDDY.NS": "Dr. Reddy's", "CIPLA.NS": "Cipla", "DIVISLAB.NS": "Divi's Labs",
    "EICHERMOT.NS": "Eicher Motors", "HEROMOTOCO.NS": "Hero MotoCorp",
    "BRITANNIA.NS": "Britannia", "TATACONSUM.NS": "Tata Consumer",
    "BPCL.NS": "BPCL", "IOC.NS": "Indian Oil", "VEDL.NS": "Vedanta",
    "DMART.NS": "Avenue Supermarts", "PIDILITIND.NS": "Pidilite",
    "SBILIFE.NS": "SBI Life", "HDFCLIFE.NS": "HDFC Life",
}

#: Asset-class label -> market key.  These render a free-form symbol box
#: instead of a dropdown, exactly as in Tattva: the class supplies the
#: suffix policy that :func:`resolve_stock_symbol` applies.
FREEFORM_STOCK_CLASSES: dict[str, str] = {
    "US Stocks — any symbol": "us",
    "India Stocks — any symbol": "india",
}

#: Curated single names offered as a dropdown beside the free-form box.
CURATED_TARGET_CLASSES: dict[str, dict[str, str]] = {
    "US Single Names": SINGLE_NAMES_US,
    "India Single Names": SINGLE_NAMES_INDIA,
}

ALL_LABELS: dict[str, str] = {
    **UNIVERSE, **SINGLE_NAMES_US, **SINGLE_NAMES_INDIA,
}


def target_classes() -> dict[str, dict[str, str]]:
    """Ordered {asset class -> {ticker: label}} for the target selector.

    Every explanatory block is offered as a target class as well: an
    instrument that helps price others is itself worth pricing, and the
    engine excludes a target from its own panel automatically.
    """
    out: dict[str, dict[str, str]] = {}
    out.update(CURATED_TARGET_CLASSES)
    out.update(ASSET_CLASSES)
    return out


def label_for(ticker: str) -> str:
    return ALL_LABELS.get(ticker, ticker)


def class_for(ticker: str) -> str:
    if ticker in TICKER_CLASS:
        return TICKER_CLASS[ticker]
    if ticker in SINGLE_NAMES_US:
        return "US Single Names"
    if ticker in SINGLE_NAMES_INDIA:
        return "India Single Names"
    if ticker.endswith((".NS", ".BO")):
        return "India Single Names"
    return "Single Name"


def explanatory_universe(target: str) -> list[str]:
    """Explanatory panel for a target, with the target itself removed.

    Self-exclusion is a correctness requirement, not a nicety: a fair value
    estimated from a panel containing the target is not an estimate.
    """
    return [t for t in UNIVERSE_TICKERS if t != target]


# ===========================================================================
# Free-form symbol resolution  (adapted from Tattva's data/universe.py)
# ===========================================================================
_symbol_fail_memo: dict[tuple[str, str], str] = {}


def resolve_stock_symbol(raw: str, market: str) -> tuple[str | None, str]:
    """Resolve a user-typed symbol to a Yahoo ticker, with listing auto-detect.

    ``market='india'``: an explicit ``.NS``/``.BO`` suffix is respected as
    typed; otherwise ``SYMBOL.NS`` is probed first, then ``SYMBOL.BO`` (NSE
    takes precedence for dual-listed names).  ``market='us'``: the bare
    symbol, uppercased, with ``.`` translated to ``-`` (Yahoo's convention,
    so ``BRK.B`` becomes ``BRK-B``).

    A candidate "hits" when it returns enough history for the engines to burn
    in, which is the same bar the pipeline applies -- so a successful
    resolution means the subsequent run is a store hit, not a re-probe.

    Returns ``(ticker, exchange_label)`` or ``(None, error_message)``.  Only
    successes are memoised to disk: a transient outage must not brand a
    symbol invalid for a week, so failures are remembered for this session
    only.
    """
    from .data import probe_symbol
    from .resilience import symbol_cache

    cleaned = (raw or "").strip().upper()
    if not cleaned or " " in cleaned or len(cleaned) > 24:
        return None, f"{raw!r} is not a valid ticker symbol."

    memo_key = (cleaned, market)
    cached = symbol_cache.get(cleaned, market)
    if cached is not None:
        return tuple(cached)                                    # type: ignore[return-value]
    if memo_key in _symbol_fail_memo:
        return None, _symbol_fail_memo[memo_key]

    if market == "india":
        if cleaned.endswith((".NS", ".BO")):
            candidates = [(cleaned, "NSE" if cleaned.endswith(".NS") else "BSE")]
        else:
            candidates = [(f"{cleaned}.NS", "NSE"), (f"{cleaned}.BO", "BSE")]
    else:
        candidates = [(cleaned.replace(".", "-"), "US")]

    for ticker, exch in candidates:
        if probe_symbol(ticker):
            symbol_cache.put(cleaned, market, value=(ticker, exch))
            return ticker, exch

    tried = " or ".join(f"{t} ({e})" for t, e in candidates)
    if market == "india":
        msg = f"{raw!r} not found on NSE (.NS) or BSE (.BO) — tried {tried}."
    else:
        msg = f"{raw!r} not found on Yahoo Finance — tried {tried}."
    _symbol_fail_memo[memo_key] = msg
    return None, msg


def universe_summary() -> dict:
    return {
        "n_instruments": len(UNIVERSE_TICKERS),
        "n_classes": len(ASSET_CLASSES),
        "per_class": {k: len(v) for k, v in ASSET_CLASSES.items()},
    }
