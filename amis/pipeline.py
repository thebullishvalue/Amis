"""
Orchestration: data -> three engines -> a single reproducible result object.
=================================================================

The pipeline is a pure function of (target, price panel, model version).  It
holds no state between calls and consults no clock, so two invocations that
report the same dataset fingerprint are required to be bit-identical.  That
requirement is not asserted here -- it is *tested*, in :mod:`amis.validation`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import MODEL_VERSION
from .data import build_panel, dataset_fingerprint, load_prices
from .dynamics import MarketDynamicsEngine
from .fusion import DecisionFusionEngine
from .universe import UNIVERSE_TICKERS, explanatory_universe, label_for
from .valuation import MarketValuationEngine


@dataclass
class AMISResult:
    target: str
    label: str
    fingerprint: str
    model_version: str
    mve: dict
    mde: dict
    dfe: dict
    diagnostics: dict = field(default_factory=dict)

    # -- convenience views --------------------------------------------------
    @property
    def series(self) -> pd.DataFrame:
        """All three engines' published series on one index."""
        v = self.mve["series"]
        d = self.mde["series"].drop(columns=["price"], errors="ignore")
        f = self.dfe["series"]
        return v.join(d, rsuffix="_mde").join(f, rsuffix="_dfe")

    @property
    def latest(self) -> pd.Series:
        s = self.series
        live = s.dropna(subset=["pdo"])
        return live.iloc[-1] if len(live) else s.iloc[-1]

    def as_of(self, when) -> pd.Series:
        s = self.series.loc[:pd.Timestamp(when)]
        return s.iloc[-1]


def run_amis(
    target: str,
    start: str = "2005-01-01",
    end: str | None = None,
    refresh: bool = True,
    progress_cb=None,
) -> AMISResult:
    """Run the full stack for one asset.

    `progress_cb(fraction, message)` is called with monotonically increasing
    fractions; it exists purely for the UI and has no effect on results.
    """
    def stage(lo: float, hi: float, msg: str):
        def cb(frac: float) -> None:
            if progress_cb is not None:
                progress_cb(lo + (hi - lo) * float(np.clip(frac, 0.0, 1.0)), msg)
        return cb

    t0 = time.time()
    tickers = sorted(set(UNIVERSE_TICKERS) | {target})
    if progress_cb:
        progress_cb(0.0, f"Loading {len(tickers)} instruments")
    prices = load_prices(tickers, start=start, end=end, refresh=refresh,
                         progress_cb=stage(0.0, 0.30, "Loading market data"))
    t_data = time.time() - t0

    if target not in prices.columns:
        raise ValueError(
            f"No price history available for {target!r}. Check the symbol.")

    keep = [c for c in prices.columns
            if c == target or c in explanatory_universe(target)]
    tgt_px, expl_px, printed, diag = build_panel(prices[keep], target)

    fp = dataset_fingerprint(tgt_px, expl_px)

    t1 = time.time()
    mve = MarketValuationEngine(list(expl_px.columns)).run(
        tgt_px, expl_px, printed,
        progress_cb=stage(0.30, 0.70, "Market Valuation Engine"))
    t_mve = time.time() - t1

    t2 = time.time()
    mde = MarketDynamicsEngine().run(
        tgt_px, progress_cb=stage(0.70, 0.90, "Market Dynamics Engine"))
    t_mde = time.time() - t2

    t3 = time.time()
    dfe = DecisionFusionEngine().run(
        mve["series"], mde["series"],
        progress_cb=stage(0.90, 1.0, "Decision Fusion Engine"))
    t_dfe = time.time() - t3

    diag.update({
        "seconds_data": round(t_data, 2),
        "seconds_mve": round(t_mve, 2),
        "seconds_mde": round(t_mde, 2),
        "seconds_dfe": round(t_dfe, 2),
        "seconds_total": round(time.time() - t0, 2),
        "n_universe_requested": len(tickers),
        "n_universe_available": int(prices.shape[1]),
        "first_publication": _first_valid(dfe["series"], "pdo"),
    })

    if progress_cb:
        progress_cb(1.0, "Complete")

    return AMISResult(
        target=target,
        label=label_for(target),
        fingerprint=fp,
        model_version=MODEL_VERSION,
        mve=mve,
        mde=mde,
        dfe=dfe,
        diagnostics=diag,
    )


def _first_valid(df: pd.DataFrame, col: str):
    return df[col].first_valid_index()


def summarise(res: AMISResult) -> dict:
    """Flat dictionary of the current reading, for headline display."""
    r = res.latest
    def g(k, default=np.nan):
        v = r.get(k, default)
        return float(v) if isinstance(v, (int, float, np.floating)) else v
    return {
        "as_of": r.name,
        "price": g("price"),
        "fair_value": g("fair_value"),
        "mispricing_pct": 100.0 * g("pct_mispricing"),
        "fvo": g("fvo"),
        "do": g("do"),
        "pdo": g("pdo"),
        "conviction": g("conviction"),
        "prob_success": g("prob_success"),
        "expected_return_pct": 100.0 * g("expected_return_pct"),
        "downside_pct": 100.0 * g("downside_pct"),
        "decision_confidence": g("decision_confidence"),
        "regime": r.get("regime_label", ""),
        "stance": r.get("stance", ""),
        "horizon_days": g("horizon_days"),
        "valuation_weight": g("valuation_weight"),
    }
