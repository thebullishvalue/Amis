"""
Global explanatory universe.
=================================================================

The Market Valuation Engine prices an asset *relative to the world*, so the
explanatory panel has to span the risk factors that actually move global
capital: equity beta and its style/sector decomposition, the term structure,
credit, commodities, FX, and volatility.  The list below is deliberately
redundant -- redundancy is removed statistically (random-matrix eigenvalue
clipping in :mod:`amis.factors`), not by hand-picking.

Instruments are Yahoo Finance symbols.  Anything that fails to download, or
that lacks enough history at a given point in time, is dropped automatically
by the data layer; the universe is therefore a superset, not a requirement.
"""

from __future__ import annotations

# name -> (ticker, human label)
EQUITY_BROAD = {
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow 30", "IWM": "Russell 2000",
    "MDY": "S&P Midcap 400", "RSP": "S&P 500 Equal Weight", "VTI": "US Total Market",
    "IJR": "S&P Smallcap 600", "IJH": "S&P Midcap", "OEF": "S&P 100",
}

EQUITY_SECTOR = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care",
    "XLI": "Industrials", "XLY": "Cons. Discretionary", "XLP": "Cons. Staples",
    "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real Estate",
    "XLC": "Communication Svcs",
}

EQUITY_INDUSTRY = {
    "SMH": "Semiconductors", "SOXX": "Semiconductors (alt)", "IGV": "Software",
    "XBI": "Biotech (EW)", "IBB": "Biotech", "ITB": "Homebuilders",
    "XHB": "Homebuilding", "XRT": "Retail", "XME": "Metals & Mining",
    "XOP": "Oil & Gas E&P", "OIH": "Oil Services", "KRE": "Regional Banks",
    "KBE": "Banks", "IYT": "Transports", "JETS": "Airlines", "ITA": "Aerospace/Defense",
    "HACK": "Cybersecurity", "SKYY": "Cloud", "FDN": "Internet", "VNQ": "US REITs",
    "IYR": "Real Estate", "PAVE": "Infrastructure", "MOO": "Agribusiness",
    "PBW": "Clean Energy", "TAN": "Solar", "URA": "Uranium", "LIT": "Lithium",
    "COPX": "Copper Miners", "WOOD": "Timber", "PICK": "Metals Producers",
}

EQUITY_STYLE = {
    "MTUM": "Momentum", "QUAL": "Quality", "VLUE": "Value", "USMV": "Min Volatility",
    "SPHB": "High Beta", "SPLV": "Low Volatility", "IWF": "Large Growth",
    "IWD": "Large Value", "SPYG": "S&P Growth", "SPYV": "S&P Value",
    "VIG": "Dividend Growth", "SDY": "Dividend Aristocrats", "SIZE": "Size Factor",
}

EQUITY_INTERNATIONAL = {
    "EFA": "EAFE", "IEFA": "Core EAFE", "EEM": "Emerging Markets", "IEMG": "Core EM",
    "VGK": "Europe", "VPL": "Pacific", "AAXJ": "Asia ex-Japan", "SCZ": "Intl Small Cap",
    "EWJ": "Japan", "EWG": "Germany", "EWU": "United Kingdom", "EWQ": "France",
    "EWC": "Canada", "EWA": "Australia", "EWH": "Hong Kong", "EWY": "South Korea",
    "EWT": "Taiwan", "EWZ": "Brazil", "EWW": "Mexico", "EWS": "Singapore",
    "EWI": "Italy", "EWP": "Spain", "EWD": "Sweden", "EWL": "Switzerland",
    "EWN": "Netherlands", "INDA": "India", "FXI": "China Large Cap",
    "MCHI": "China", "KWEB": "China Internet", "EPI": "India Earnings",
    "EZA": "South Africa", "ILF": "Latin America", "TUR": "Turkey",
}

FIXED_INCOME = {
    "TLT": "20+ Yr Treasury", "TLH": "10-20 Yr Treasury", "IEF": "7-10 Yr Treasury",
    "IEI": "3-7 Yr Treasury", "SHY": "1-3 Yr Treasury", "SHV": "Short Treasury",
    "BIL": "T-Bills", "GOVT": "US Treasury Broad", "EDV": "Extended Duration",
    "ZROZ": "Zero Coupon 25+", "AGG": "US Aggregate", "BND": "Total Bond",
    "LQD": "IG Corporate", "VCIT": "IG Intermediate", "VCSH": "IG Short",
    "HYG": "High Yield", "JNK": "High Yield (alt)", "SJNK": "Short High Yield",
    "TIP": "TIPS", "STIP": "Short TIPS", "EMB": "EM USD Debt",
    "BWX": "Intl Treasury", "IGOV": "Intl Govt", "MBB": "Agency MBS",
    "FLOT": "Floating Rate", "BKLN": "Senior Loans", "PFF": "Preferred Stock",
    "CWB": "Convertibles",
}

COMMODITY = {
    "GLD": "Gold", "IAU": "Gold (alt)", "SLV": "Silver", "PPLT": "Platinum",
    "PALL": "Palladium", "GLTR": "Precious Metals Basket", "GDX": "Gold Miners",
    "GDXJ": "Junior Gold Miners", "SIL": "Silver Miners", "USO": "WTI Crude",
    "BNO": "Brent Crude", "UNG": "Natural Gas", "DBA": "Agriculture",
    "DBC": "Broad Commodity", "DBB": "Base Metals", "DBO": "Oil",
    "CORN": "Corn", "WEAT": "Wheat", "SOYB": "Soybeans", "CPER": "Copper",
    "GSG": "GSCI Commodity", "PDBC": "Optimum Yield Commodity",
}

CURRENCY = {
    "UUP": "US Dollar Bull", "UDN": "US Dollar Bear", "FXE": "Euro", "FXY": "Yen",
    "FXB": "Sterling", "FXF": "Swiss Franc", "FXA": "Aussie Dollar",
    "FXC": "Canadian Dollar", "EURUSD=X": "EUR/USD", "JPY=X": "USD/JPY",
    "GBPUSD=X": "GBP/USD", "AUDUSD=X": "AUD/USD", "USDCAD=X": "USD/CAD",
    "USDCHF=X": "USD/CHF", "USDCNY=X": "USD/CNY", "USDMXN=X": "USD/MXN",
    "USDKRW=X": "USD/KRW", "USDINR=X": "USD/INR", "USDBRL=X": "USD/BRL",
    "DX-Y.NYB": "US Dollar Index",
}

RATES_AND_VOL = {
    "^TNX": "US 10Y Yield", "^TYX": "US 30Y Yield", "^FVX": "US 5Y Yield",
    "^IRX": "US 13W Bill", "^VIX": "VIX", "^VXN": "Nasdaq VIX",
    "^RVX": "Russell VIX", "^OVX": "Crude Oil VIX", "^GVZ": "Gold VIX",
    "^VIX3M": "3-Month VIX",
}

GLOBAL_INDEX = {
    "^GSPC": "S&P 500 Index", "^NDX": "Nasdaq 100 Index", "^DJI": "Dow Jones",
    "^RUT": "Russell 2000 Index", "^NYA": "NYSE Composite", "^FTSE": "FTSE 100",
    "^GDAXI": "DAX", "^FCHI": "CAC 40", "^STOXX50E": "Euro Stoxx 50",
    "^N225": "Nikkei 225", "^HSI": "Hang Seng", "^AXJO": "ASX 200",
    "^BSESN": "BSE Sensex", "^KS11": "KOSPI", "^TWII": "Taiwan Weighted",
    "^BVSP": "Bovespa", "^MXX": "IPC Mexico", "^GSPTSE": "TSX Composite",
    "^SSMI": "SMI", "^AEX": "AEX", "^IBEX": "IBEX 35", "^OMX": "OMX Stockholm",
    "^JKSE": "Jakarta Composite", "^STI": "Straits Times",
}

FUTURES = {
    "ES=F": "S&P 500 Futures", "NQ=F": "Nasdaq Futures", "YM=F": "Dow Futures",
    "RTY=F": "Russell Futures", "CL=F": "Crude Oil Futures", "GC=F": "Gold Futures",
    "SI=F": "Silver Futures", "HG=F": "Copper Futures", "NG=F": "Nat Gas Futures",
    "ZC=F": "Corn Futures", "ZS=F": "Soybean Futures", "ZW=F": "Wheat Futures",
    "ZB=F": "30Y Bond Futures", "ZN=F": "10Y Note Futures", "ZF=F": "5Y Note Futures",
    "6E=F": "Euro FX Futures", "6J=F": "Yen Futures", "KC=F": "Coffee Futures",
    "SB=F": "Sugar Futures", "CT=F": "Cotton Futures", "LE=F": "Live Cattle",
    "HO=F": "Heating Oil", "RB=F": "RBOB Gasoline", "PL=F": "Platinum Futures",
}

DIGITAL = {
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum",
}

ASSET_CLASSES: dict[str, dict[str, str]] = {
    "US Equity - Broad": EQUITY_BROAD,
    "US Equity - Sector": EQUITY_SECTOR,
    "US Equity - Industry": EQUITY_INDUSTRY,
    "US Equity - Style": EQUITY_STYLE,
    "International Equity": EQUITY_INTERNATIONAL,
    "Fixed Income": FIXED_INCOME,
    "Commodity": COMMODITY,
    "Currency": CURRENCY,
    "Rates & Volatility": RATES_AND_VOL,
    "Global Index": GLOBAL_INDEX,
    "Futures": FUTURES,
    "Digital Asset": DIGITAL,
}

#: Flat ticker -> label map for the whole explanatory universe.
UNIVERSE: dict[str, str] = {t: n for grp in ASSET_CLASSES.values() for t, n in grp.items()}

#: Ticker -> asset class.
TICKER_CLASS: dict[str, str] = {
    t: cls for cls, grp in ASSET_CLASSES.items() for t in grp
}

UNIVERSE_TICKERS: list[str] = sorted(UNIVERSE)

# ---------------------------------------------------------------------------
# Single-name equities offered as analysis targets.  These are *not* part of
# the explanatory panel -- including individual names there would let a
# target be explained by its own sector-mates at 0.9 correlation, which is
# valuation by tautology rather than by global pricing.
# ---------------------------------------------------------------------------
SINGLE_NAMES: dict[str, str] = {
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


def selectable_assets() -> dict[str, list[tuple[str, str]]]:
    """Grouped (ticker, label) pairs offered in the asset selector."""
    groups: dict[str, list[tuple[str, str]]] = {}
    for cls, grp in ASSET_CLASSES.items():
        groups[cls] = sorted(grp.items())
    groups["US Single Names"] = sorted(SINGLE_NAMES.items())
    return groups


def label_for(ticker: str) -> str:
    return UNIVERSE.get(ticker) or SINGLE_NAMES.get(ticker) or ticker


def class_for(ticker: str) -> str:
    return TICKER_CLASS.get(ticker, "Single Name" if ticker in SINGLE_NAMES else "Other")


def explanatory_universe(target: str) -> list[str]:
    """Explanatory panel for a target, with the target itself removed.

    Self-exclusion is a correctness requirement, not a nicety: a fair value
    estimated from a panel containing the target is not an estimate.
    """
    return [t for t in UNIVERSE_TICKERS if t != target]
