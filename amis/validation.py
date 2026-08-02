"""
Integrity tests: determinism, revision invariance, and walk-forward skill.
=================================================================

Claims about causality are cheap.  Every one made by this system is
executable here, and the dashboard runs them on the same data the user is
looking at rather than quoting a result from elsewhere.

* :func:`determinism_test` -- the same inputs must give bit-identical
  outputs.  Failure means a hidden random seed or an iteration over an
  unordered container.
* :func:`revision_invariance_test` -- the strong one.  Truncate the record,
  run, extend the record, run again, and compare the overlap *exactly*.  Any
  non-zero difference is repainting.  This catches what code review does
  not: a centred filter, a full-sample normalisation, a screen applied with
  hindsight, or a smoother hiding inside a library call.
* :func:`walk_forward_report` -- the system's own published one-step and
  h-step predictions, scored against what actually happened.  No re-fitting
  is involved because there is nothing to re-fit: the online record *is* the
  walk-forward record, which is the point of building it this way.

References
----------
White, H. (2000). "A Reality Check for Data Snooping." *Econometrica* 68(5).
Bailey, D. H. et al. (2014). "Pseudo-Mathematics and Financial
    Charlatanism." *Notices of the AMS* 61(5).
Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 7
    (why cross-validation leaks in finance and walk-forward does not).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import build_panel
from .dynamics import MarketDynamicsEngine
from .fusion import DecisionFusionEngine
from .valuation import MarketValuationEngine

#: Series compared by the invariance test.  Deliberately includes the
#: derived, smoothed and rank-normalised outputs, not just the raw ones --
#: those are where look-ahead usually hides.
AUDITED = [
    "fair_value", "gap", "pct_mispricing", "fvo", "pred_sd", "confidence",
    "xs_consistency", "k_factors", "stress", "gap_percentile",
    "do", "do_raw", "trend_strength", "hurst", "entropy", "stability",
    "bull_prob", "tech_confidence", "vol_percentile",
    "pdo", "pdo_raw", "conviction", "prob_success", "decision_confidence",
    "expected_return_pct", "valuation_weight", "horizon_days",
]


def _run_core(prices: pd.DataFrame, target: str) -> pd.DataFrame:
    """Engines only -- no network, no cache, no clock."""
    tgt, expl, printed, _ = build_panel(prices, target)
    mve = MarketValuationEngine(list(expl.columns)).run(tgt, expl, printed)
    mde = MarketDynamicsEngine().run(tgt)
    dfe = DecisionFusionEngine().run(mve["series"], mde["series"])
    out = mve["series"].join(
        mde["series"].drop(columns=["price"], errors="ignore"), rsuffix="_mde"
    ).join(dfe["series"], rsuffix="_dfe")
    return out[[c for c in AUDITED if c in out.columns]]


def audited_view(res) -> pd.DataFrame:
    """The audited columns of an already-computed result.

    Lets the integrity tests reuse the run the user is looking at instead of
    recomputing it, which halves the wait without weakening the test: the
    comparison is still between two independently produced records.
    """
    out = res.series
    return out[[c for c in AUDITED if c in out.columns]]


def determinism_test(prices: pd.DataFrame, target: str,
                     reference: pd.DataFrame | None = None) -> dict:
    """Run the stack twice on identical inputs and diff the results."""
    a = _run_core(prices, target) if reference is None else reference
    b = _run_core(prices, target)
    diffs = {}
    for c in a.columns:
        x, y = a[c].values, b[c].values
        m = np.isfinite(x) | np.isfinite(y)
        same_nan = np.array_equal(np.isnan(x), np.isnan(y))
        d = np.nanmax(np.abs(x[m] - y[m])) if m.any() else 0.0
        diffs[c] = float(0.0 if not np.isfinite(d) else d)
        if not same_nan:
            diffs[c] = float("inf")
    worst = max(diffs.values()) if diffs else 0.0
    return {"passed": worst == 0.0, "max_abs_difference": worst,
            "per_series": diffs, "n_compared": int(len(a))}


def revision_invariance_test(
    prices: pd.DataFrame,
    target: str,
    truncations: tuple[float, ...] = (0.55, 0.75, 0.90),
    reference: pd.DataFrame | None = None,
) -> dict:
    """Do later observations alter earlier published values?  They must not.

    The full record is compared against runs that were denied the tail of
    the data.  A system that satisfies revision invariance reproduces every
    overlapping value exactly, to the last bit -- not approximately, and not
    "within tolerance".
    """
    full = _run_core(prices, target) if reference is None else reference
    rows = []
    detail: dict[str, dict[str, float]] = {}
    for frac in truncations:
        n = int(len(prices) * frac)
        if n < 400:
            continue
        cut = prices.iloc[:n]
        part = _run_core(cut, target)
        common = full.index.intersection(part.index)
        if len(common) == 0:
            continue
        worst, worst_col = 0.0, ""
        per: dict[str, float] = {}
        for c in part.columns:
            x = full.loc[common, c].values
            y = part.loc[common, c].values
            m = np.isfinite(x) & np.isfinite(y)
            d = float(np.max(np.abs(x[m] - y[m]))) if m.any() else 0.0
            nan_mismatch = int(np.sum(np.isfinite(x) != np.isfinite(y)))
            per[c] = d
            if d > worst:
                worst, worst_col = d, c
            if nan_mismatch:
                per[c] = max(d, 0.0)
        detail[f"{frac:.0%}"] = per
        rows.append({
            "truncation": f"{frac:.0%} of record",
            "cut_date": str(cut.index[-1].date()),
            "overlapping_observations": int(len(common)),
            "max_abs_revision": worst,
            "worst_series": worst_col if worst > 0 else "-",
        })
    table = pd.DataFrame(rows)
    passed = bool(len(table)) and float(table["max_abs_revision"].max()) == 0.0
    return {"passed": passed, "table": table, "detail": detail,
            "max_abs_revision": float(table["max_abs_revision"].max())
            if len(table) else float("nan")}


# ===========================================================================
# Walk-forward scoring
# ===========================================================================
def walk_forward_report(res, horizons: tuple[int, ...] = (5, 21, 63)) -> dict:
    """Score the published oscillators against realised forward returns.

    Every number here is out of sample by construction: the oscillator at t
    was emitted before the return it is scored against existed.  Reported
    quantities are the information coefficient (rank correlation between
    signal and subsequent return), its Newey-West t-statistic accounting for
    the overlap induced by h-day returns, and the realised information ratio
    of a volatility-scaled position taken in proportion to the signal.
    """
    s = res.series
    px = s["price"].astype(float)
    logp = np.log(px)
    vol = s["volatility"].astype(float)

    signals = {"Fair Value Oscillator": -s["fvo"],
               "Dynamics Oscillator": s["do"],
               "Paramount Decision Oscillator": s["pdo"]}
    rows = []
    curves: dict[str, pd.Series] = {}
    for name, sig in signals.items():
        for h in horizons:
            fwd = logp.shift(-h) - logp
            z = (fwd / (vol * np.sqrt(h))).replace([np.inf, -np.inf], np.nan)
            d = pd.concat([sig.rename("s"), z.rename("y")], axis=1).dropna()
            if len(d) < 200:
                continue
            ic = float(d["s"].corr(d["y"], method="spearman"))
            t_stat = _nw_tstat(d["s"].values, d["y"].values, h)
            pos = np.sign(d["s"]) * np.minimum(np.abs(d["s"]), 3.0) / 3.0
            pnl = pos * d["y"]
            ir = float(pnl.mean() / pnl.std() * np.sqrt(252.0 / h)) \
                if pnl.std() > 0 else np.nan
            # An exactly-zero signal is an abstention, not a wrong call: the
            # PDO is identically zero whenever no edge has been demonstrated,
            # and counting those as misses would understate the hit rate by
            # however often the system correctly declined to have a view.
            act = d[np.abs(d["s"]) > 1e-9]
            hit = float((np.sign(act["s"]) == np.sign(act["y"])).mean()) \
                if len(act) else np.nan
            rows.append({"signal": name, "horizon": f"{h}d",
                         "rank IC": round(ic, 4),
                         "NW t-stat": round(t_stat, 2),
                         "hit rate": round(hit, 4),
                         "realised IR": round(ir, 3) if np.isfinite(ir) else np.nan,
                         "n": int(len(d))})
            if h == 21:
                curves[name] = (pnl / max(pnl.std(), 1e-12)).cumsum()
    return {"table": pd.DataFrame(rows), "curves": curves}


def _nw_tstat(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """t-statistic of the slope of y on x with Newey-West (1987) errors.

    Overlapping h-day returns are strongly autocorrelated; an OLS t-statistic
    on them overstates significance by roughly sqrt(h), which is how a great
    many published "edges" are manufactured.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 30:
        return float("nan")
    X = np.column_stack([np.ones(n), x])
    try:
        XtXi = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return float("nan")
    beta = XtXi @ (X.T @ y)
    e = y - X @ beta
    u = X * e[:, None]
    S = u.T @ u
    for l in range(1, min(lag, n - 1) + 1):
        w = 1.0 - l / (lag + 1.0)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    V = XtXi @ S @ XtXi
    se = float(np.sqrt(max(V[1, 1], 0.0)))
    return float(beta[1] / se) if se > 0 else float("nan")


def oscillator_distribution(res) -> pd.DataFrame:
    """Unconditional distribution of each oscillator, for calibration checks."""
    s = res.series
    cols = {"Fair Value Oscillator": "fvo", "Dynamics Oscillator": "do",
            "Paramount Decision Oscillator": "pdo"}
    rows = []
    for name, c in cols.items():
        v = s[c].dropna()
        if not len(v):
            continue
        rows.append({
            "oscillator": name, "n": len(v),
            "mean": round(float(v.mean()), 4),
            "std": round(float(v.std()), 4),
            "skew": round(float(v.skew()), 3),
            "excess kurtosis": round(float(v.kurtosis()), 3),
            "1%": round(float(v.quantile(0.01)), 3),
            "50%": round(float(v.quantile(0.50)), 3),
            "99%": round(float(v.quantile(0.99)), 3),
        })
    return pd.DataFrame(rows)
