"""
Market data layer: append-only price store and causal panel construction.
=================================================================

Two properties matter here and nothing else does.

**Immutability.**  Total-return adjustment at the vendor rewrites history
every time a dividend is paid: yesterday's "adjusted close for 2015-03-04"
is not today's.  A model can be perfectly non-repainting and still produce
different history if its *inputs* are rewritten underneath it.  The cache is
therefore append-only -- once a (ticker, date) close has been written it is
never overwritten, only extended.  Reproducibility is then a property of the
system as a whole rather than of the model in isolation.

**Causal alignment.**  Instruments trade on different calendars.  Gaps are
filled forward only.  A missing observation is carried from the past, never
interpolated from the future, and an instrument contributes nothing before
its own first print.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

CACHE_DIR = Path(__file__).resolve().parent.parent / ".amis_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_START = "2005-01-01"

#: Minimum history before an instrument may enter the explanatory panel.
#: Set by estimability of a second moment, not by preference.
MIN_HISTORY = 250

#: An instrument is dropped if it has not printed for this many sessions --
#: a delisted or halted line would otherwise contribute a frozen constant.
MAX_STALE = 30


#: Symbols the vendor rejected are not re-requested for this many days.  The
#: universe is a deliberate superset and a few of its lines are delisted at
#: any moment; retrying them on every run costs a round trip and a warning
#: each time, and the TTL means a symbol that comes back is picked up.
FAILURE_TTL_DAYS = 7
_FAILED_PATH = CACHE_DIR / "_unavailable.json"


def _safe_name(ticker: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in ticker)


def _load_failures() -> dict:
    try:
        import json
        with open(_FAILED_PATH) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_failures(d: dict) -> None:
    try:
        import json
        with open(_FAILED_PATH, "w") as fh:
            json.dump(d, fh)
    except Exception:
        pass


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{_safe_name(ticker)}.parquet"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def _download_batch(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    """Adjusted closes for a batch of tickers; missing names simply absent."""
    import yfinance as yf

    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            threads=True,
            group_by="column",
        )
    except Exception:
        return pd.DataFrame()
    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]

    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close.astype(float)


def _read_cached(ticker: str) -> pd.Series | None:
    p = _cache_path(ticker)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None
    if df.empty or "close" not in df.columns:
        return None
    s = df["close"]
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _append_cache(ticker: str, series: pd.Series) -> pd.Series:
    """Merge new observations into the store *without* touching existing ones."""
    series = series.dropna()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    old = _read_cached(ticker)
    if old is not None and len(old):
        new_only = series[~series.index.isin(old.index)]
        merged = pd.concat([old, new_only]).sort_index()
    else:
        merged = series
    merged = merged[~merged.index.duplicated(keep="first")]
    if len(merged):
        try:
            pd.DataFrame({"close": merged}).to_parquet(_cache_path(ticker))
        except Exception:
            pass
    return merged


def load_prices(
    tickers: list[str],
    start: str = DEFAULT_START,
    end: str | None = None,
    refresh: bool = True,
    batch_size: int = 40,
    progress_cb=None,
) -> pd.DataFrame:
    """Close prices for `tickers`, served from the append-only store.

    Only dates strictly after each ticker's cached maximum are requested, so
    a warm store makes an incremental run cheap and, more importantly, keeps
    already-published history byte-identical.
    """
    tickers = list(dict.fromkeys(tickers))
    out: dict[str, pd.Series] = {}
    to_fetch: dict[str, list[str]] = {}

    for t in tickers:
        cached = _read_cached(t)
        if cached is not None and len(cached):
            out[t] = cached
            # Backfill is as necessary as forward extension: a store warmed
            # by an earlier short request must not silently truncate a later
            # long one.  Both directions only ever *add* rows.
            if pd.Timestamp(start) < cached.index.min() - pd.Timedelta(days=5):
                to_fetch.setdefault(start, []).append(t)
            if refresh:
                last = cached.index.max()
                nxt = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                today = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
                if pd.Timestamp(nxt) <= today:
                    to_fetch.setdefault(nxt, []).append(t)
        else:
            to_fetch.setdefault(start, []).append(t)

    failures = _load_failures()
    now = pd.Timestamp.today().normalize()
    fresh = {t: d for t, d in failures.items()
             if (now - pd.Timestamp(d)).days < FAILURE_TTL_DAYS}
    groups = {s: [t for t in g if not (t in fresh and t not in out)]
              for s, g in to_fetch.items()}

    total = sum(len(v) for v in groups.values())
    done = 0
    for s, group in groups.items():
        for i in range(0, len(group), batch_size):
            chunk = group[i: i + batch_size]
            df = _download_batch(chunk, s, end)
            for t in chunk:
                got = False
                if t in df.columns:
                    ser = df[t].dropna()
                    if len(ser):
                        out[t] = _append_cache(t, ser)
                        got = True
                        fresh.pop(t, None)
                if not got and t not in out:
                    fresh[t] = now.strftime("%Y-%m-%d")
            done += len(chunk)
            if progress_cb is not None:
                progress_cb(min(done / max(total, 1), 1.0))
    _save_failures(fresh)

    if not out:
        return pd.DataFrame()

    panel = pd.DataFrame(out).sort_index()
    panel = panel[~panel.index.duplicated(keep="last")]
    if end is not None:
        panel = panel.loc[: pd.Timestamp(end)]
    panel = panel.loc[pd.Timestamp(start):]
    # Always return a column per requested instrument, empty if the vendor
    # had nothing.  The cross-sectional dimension then depends on the
    # universe definition rather than on whether one symbol happened to be
    # reachable today -- otherwise two machines running the same version
    # against the same dates could disagree about the factor structure.
    missing = [t for t in tickers if t not in panel.columns]
    for t in missing:
        panel[t] = np.nan
    return panel[sorted(panel.columns)]


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------
def build_panel(
    prices: pd.DataFrame,
    target: str,
    min_history: int = MIN_HISTORY,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, dict]:
    """Align a target series and its explanatory panel on a causal calendar.

    Returns ``(target_price, explanatory_prices, printed_mask, diagnostics)``.

    The trading calendar is the target's own -- valuation is published when
    the target prints, not when some other market does.  Explanatory series
    are forward-filled onto that calendar, which is the only fill direction
    that respects the arrow of time.

    **No instrument is admitted or rejected using full-sample information.**
    This is not fastidiousness.  Screening the panel on total history --
    "keep instruments with at least 250 observations" -- reaches backwards:
    an ETF launched in 2021 fails the screen in a run made in 2021 and
    passes it in a run made in 2026, so the 2026 run silently rewrites the
    factor structure of 2015.  That is repainting introduced by the data
    layer, and it would quietly invalidate the revision-invariance claim
    while every line of the inference code remained blameless.  Admission is
    therefore delegated to the engines, which apply it per timestamp against
    the history accumulated so far.  What this function returns is the raw
    material: prices carried forward, and a mask saying where a genuine
    print occurred.
    """
    diag: dict = {}
    if target not in prices.columns:
        raise KeyError(f"target {target!r} missing from price panel")

    tgt = prices[target].dropna()
    if len(tgt) < min_history:
        raise ValueError(
            f"{target}: only {len(tgt)} observations, need at least {min_history}"
        )

    cal = tgt.index
    raw = prices.drop(columns=[target], errors="ignore")
    # a print is a genuine, strictly positive observation on this calendar
    printed = raw.reindex(cal).notna() & (raw.reindex(cal) > 0).fillna(False)
    expl = raw.reindex(cal).ffill()
    expl = expl.where(expl > 0)

    # An instrument contributes nothing before its own first print: carrying
    # a value forward from the past is a stale-quote assumption, carrying one
    # backward would be fabrication.
    #
    # Note what is *not* done here: columns are never removed, not even ones
    # that never print at all.  Removing them would make the width of the
    # panel a function of the record's endpoint -- truncate to 2017 and every
    # instrument launched later disappears, changing the cross-sectional
    # dimension, the Marchenko-Pastur edge, and therefore the factor
    # structure of 2012.  An empty column costs nothing (it is never
    # admitted) and keeps the panel's shape a property of the universe rather
    # than of when the analysis happened to be run.
    never = []
    for c in expl.columns:
        if not bool(printed[c].any()):
            expl[c] = np.nan
            never.append(c)
            continue
        fv = printed[c].idxmax()
        expl.loc[expl.index < fv, c] = np.nan

    diag["n_explanatory"] = int(expl.shape[1])
    diag["n_never_printed"] = len(never)
    diag["dropped"] = {c: "no print on this calendar" for c in never}
    diag["calendar_start"] = cal[0]
    diag["calendar_end"] = cal[-1]
    diag["n_observations"] = len(cal)
    diag["min_history"] = min_history
    return tgt, expl, printed, diag


def dataset_fingerprint(target_px: pd.Series, expl_px: pd.DataFrame) -> str:
    """SHA-256 over the exact numeric inputs.

    Two runs that report the same fingerprint are required to produce
    bit-identical output; this is what makes the determinism claim testable
    rather than aspirational.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(target_px.values, dtype=np.float64).tobytes())
    h.update(str(target_px.name).encode())
    for c in sorted(expl_px.columns):
        h.update(c.encode())
        v = np.ascontiguousarray(expl_px[c].values, dtype=np.float64)
        h.update(np.nan_to_num(v, nan=-9.87654321e300).tobytes())
    h.update(np.ascontiguousarray(
        target_px.index.view("int64"), dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def cache_status() -> dict:
    files = list(CACHE_DIR.glob("*.parquet"))
    size = sum(f.stat().st_size for f in files) / 1e6
    return {"n_series": len(files), "size_mb": round(size, 2), "path": str(CACHE_DIR)}
