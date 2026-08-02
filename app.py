"""
Autonomous Market Intelligence System -- dashboard.
=================================================================

The only input this application accepts is *which asset to analyse*.  There
is no lookback slider, no smoothing control, no factor count, no threshold,
and no weighting scheme, because every one of those is inferred from the
data by the engines themselves.  A control that lets the analyst tune a
signal until it looks right is a control that manufactures the result; its
absence is the point of the system, not an omission from the interface.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from amis import MODEL_VERSION
from amis import viz
from amis.data import cache_status
from amis.pipeline import run_amis, summarise
from amis.universe import UNIVERSE, UNIVERSE_TICKERS, selectable_assets
from amis.validation import (audited_view, determinism_test,
                             oscillator_distribution, revision_invariance_test,
                             walk_forward_report)

st.set_page_config(page_title="AMIS", page_icon="◈", layout="wide",
                   initial_sidebar_state="expanded")


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
def _is_dark() -> bool:
    try:
        t = getattr(st.context, "theme", None)
        if t is not None and getattr(t, "type", None):
            return str(t.type).lower() == "dark"
    except Exception:
        pass
    base = st.get_option("theme.base")
    return str(base).lower() == "dark" if base else False


DARK = _is_dark()
TH = viz.theme(DARK)

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1500px; }}
  .amis-tile {{
      background: {TH['surface']};
      border: 1px solid {'rgba(255,255,255,0.10)' if DARK else 'rgba(11,11,11,0.10)'};
      border-radius: 10px; padding: 12px 14px; height: 100%;
  }}
  .amis-label {{ font-size: 11px; letter-spacing: .05em; text-transform: uppercase;
      color: {TH['muted']}; margin-bottom: 4px; }}
  .amis-value {{ font-size: 26px; font-weight: 600; color: {TH['ink']};
      line-height: 1.15; }}
  .amis-sub {{ font-size: 12px; color: {TH['ink2']}; margin-top: 2px; }}
  .amis-bar {{ height: 6px; border-radius: 3px; background: {TH['grid']};
      margin-top: 9px; overflow: hidden; }}
  .amis-bar > div {{ height: 100%; border-radius: 3px; }}
  .amis-note {{ font-size: 12px; color: {TH['ink2']}; line-height: 1.55; }}
  .amis-cite {{ font-size: 11.5px; color: {TH['muted']}; line-height: 1.6; }}
  div[data-testid="stMetricValue"] {{ font-size: 22px; }}
</style>
""", unsafe_allow_html=True)


def tile(label: str, value: str, sub: str = "", frac: float | None = None,
         colour: str | None = None) -> str:
    bar = ""
    if frac is not None and np.isfinite(frac):
        w = float(np.clip(frac, 0.0, 1.0)) * 100.0
        bar = (f"<div class='amis-bar'><div style='width:{w:.1f}%;"
               f"background:{colour or TH['series'][0]}'></div></div>")
    return (f"<div class='amis-tile'><div class='amis-label'>{label}</div>"
            f"<div class='amis-value' style='color:{colour or TH['ink']}'>{value}</div>"
            f"<div class='amis-sub'>{sub}</div>{bar}</div>")


def fig(f, key: str) -> None:
    st.plotly_chart(f, width='stretch', key=key,
                    config={"displayModeBar": False, "scrollZoom": False})


# ---------------------------------------------------------------------------
# Cached computation
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600, max_entries=8)
def _run(target: str, start: str, day: str):
    """`day` participates in the cache key so a new session refreshes data."""
    box = st.empty()
    bar = st.progress(0.0)

    def cb(frac, msg):
        bar.progress(float(np.clip(frac, 0, 1)))
        box.caption(f"{msg} — {frac:.0%}")

    try:
        res = run_amis(target, start=start, progress_cb=cb)
    finally:
        bar.empty()
        box.empty()
    return res


@st.cache_data(show_spinner=False, ttl=3600, max_entries=8)
def _prices_for(target: str, start: str, day: str):
    from amis.data import load_prices
    from amis.universe import explanatory_universe
    tk = sorted(set(UNIVERSE_TICKERS) | {target})
    px = load_prices(tk, start=start, refresh=False)
    keep = [c for c in px.columns if c == target or c in explanatory_universe(target)]
    return px[keep]


TODAY = pd.Timestamp.today().strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Sidebar: the single input
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ◈ AMIS")
    st.caption(f"Autonomous Market Intelligence System · {MODEL_VERSION}")

    groups = selectable_assets()
    options: list[str] = []
    labels: dict[str, str] = {}
    for grp, items in groups.items():
        for tk, name in items:
            options.append(tk)
            labels[tk] = f"{name} ({tk}) · {grp}"

    default = options.index("SPY") if "SPY" in options else 0
    choice = st.selectbox("Asset", options, index=default,
                          format_func=lambda t: labels.get(t, t))
    custom = st.text_input("…or any Yahoo Finance symbol", value="",
                           placeholder="e.g. NVDA, BTC-USD, ^N225").strip().upper()
    target = custom or choice

    st.caption("The asset is the only input. Every other quantity — memory, "
               "factor count, weighting, thresholds, horizon — is inferred.")
    go_btn = st.button("Analyse", type="primary", width='stretch')

    st.divider()
    cs = cache_status()
    st.caption(f"Price store: {cs['n_series']} series · {cs['size_mb']} MB · "
               "append-only")

if "target" not in st.session_state:
    st.session_state["target"] = target
if go_btn:
    st.session_state["target"] = target
target = st.session_state["target"]

try:
    res = _run(target, "2005-01-01", TODAY)
except Exception as exc:                       # pragma: no cover - UI path
    st.error(f"Could not analyse **{target}** — {exc}")
    st.stop()

S = res.series
summary = summarise(res)
last = res.latest

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(f"## {res.label} · `{res.target}`")
st.caption(
    f"As of {summary['as_of']:%d %b %Y} · {res.diagnostics['n_explanatory']} "
    f"explanatory instruments · {res.diagnostics['n_observations']:,} sessions · "
    f"first publication {res.diagnostics['first_publication']:%b %Y} · "
    f"dataset fingerprint `{res.fingerprint}` · {res.model_version}")

pdo = summary["pdo"]
pdo_colour = TH["pos"] if pdo > 0 else (TH["neg"] if pdo < 0 else TH["muted"])
misp = summary["mispricing_pct"]
misp_colour = TH["neg"] if misp > 0 else TH["pos"]

c = st.columns(5)
c[0].markdown(tile(
    "Paramount Decision Oscillator", f"{pdo:+.2f}", summary["stance"],
    frac=min(abs(pdo) / 3.0, 1.0), colour=pdo_colour), unsafe_allow_html=True)
c[1].markdown(tile(
    "Mispricing", f"{misp:+.1f}%",
    f"fair value {summary['fair_value']:,.2f} vs price {summary['price']:,.2f}",
    frac=min(abs(misp) / 25.0, 1.0), colour=misp_colour), unsafe_allow_html=True)
c[2].markdown(tile(
    "Decision confidence", f"{summary['decision_confidence']:.0%}",
    "demonstrated, not assumed", frac=summary["decision_confidence"],
    colour=TH["series"][0]), unsafe_allow_html=True)
c[3].markdown(tile(
    "Market regime", str(summary["regime"]).split(" / ")[0],
    str(summary["regime"]), frac=float(last.get("stress", np.nan)),
    colour=TH["series"][2]), unsafe_allow_html=True)
c[4].markdown(tile(
    "Inferred horizon", f"{summary['horizon_days']:.0f}d",
    f"valuation weight {summary['valuation_weight']:.0%}",
    frac=summary["valuation_weight"], colour=TH["series"][1]),
    unsafe_allow_html=True)

st.write("")

TABS = st.tabs([
    "Overview", "Valuation Engine", "Dynamics Engine", "Decision Engine",
    "Explainability", "Uncertainty", "Integrity", "Methodology",
])

# ===========================================================================
# Overview
# ===========================================================================
with TABS[0]:
    fig(viz.price_vs_fair_value(S, TH, res.target), "ov_pfv")
    st.caption("Fair value is the market-implied level from a dynamic "
               "cointegrating regression on the integrated global factors; the "
               "band is the 95% one-step predictive interval, which widens "
               "automatically when the cross-section stops explaining the asset.")

    fig(viz.regime_ribbon(S, TH), "ov_regime")

    a, b = st.columns(2)
    with a:
        fig(viz.oscillator(S["fvo"], TH, "Overvalued", "Undervalued",
                           title="Fair Value Oscillator (sigma)",
                           invert_colour=True, ytitle="sigma"), "ov_fvo")
    with b:
        fig(viz.oscillator(S["do"], TH, "Bullish pressure", "Bearish pressure",
                           title="Dynamics Oscillator (sigma)",
                           ytitle="sigma"), "ov_do")
    fig(viz.oscillator(S["pdo"], TH, "Constructive", "Adverse", height=300,
                       title="Paramount Decision Oscillator "
                             "(confidence-shrunk expected annualised information ratio)",
                       ytitle="expected IR", bands=(1.0, 2.0)), "ov_pdo")
    st.caption("The PDO is exactly zero whenever the fusion engine has no "
               "demonstrated association between its own past predictions and "
               "their realised outcomes. Flat stretches are abstentions, not "
               "neutral opinions.")

# ===========================================================================
# Valuation
# ===========================================================================
with TABS[1]:
    m = st.columns(5)
    m[0].metric("Fair value", f"{last.get('fair_value', np.nan):,.2f}")
    m[1].metric("Gap", f"{100*last.get('pct_mispricing', np.nan):+.2f}%")
    m[2].metric("FVO", f"{last.get('fvo', np.nan):+.2f}σ")
    m[3].metric("Latent factors", f"{last.get('k_factors', np.nan):.0f}")
    m[4].metric("Valuation confidence", f"{last.get('confidence', np.nan):.0%}")

    fig(viz.oscillator(100 * S["pct_mispricing"], TH, "Overvalued",
                       "Undervalued", height=280, invert_colour=True,
                       title="Historical valuation gap",
                       ytitle="% above / below fair value",
                       bands=()), "va_gap")

    a, b = st.columns(2)
    with a:
        fig(viz.multi_line(S[["confidence", "xs_consistency", "mr_prob"]]
                           .rename(columns={"confidence": "valuation confidence",
                                            "xs_consistency": "cross-sectional consistency",
                                            "mr_prob": "P(gap mean-reverts)"}),
                           TH, title="Is the valuation trustworthy today?",
                           ytitle="probability", fmt=":.0%"), "va_conf")
        st.caption("Cross-sectional consistency is the sign agreement of twelve "
                   "leave-one-asset-class-out refits. When independent slices of "
                   "the world disagree about the mispricing, the decision layer "
                   "sees it.")
    with b:
        fig(viz.multi_line(S[["k_factors", "gap_halflife", "adapt_memory"]]
                           .rename(columns={"k_factors": "latent factors (count)",
                                            "gap_halflife": "gap half-life (days)",
                                            "adapt_memory": "coefficient memory (days)"}),
                           TH, title="What the engine inferred about itself",
                           ytitle="days / count"), "va_infer")
        st.caption("Factor count comes from the Marchenko-Pastur edge; "
                   "coefficient memory from dynamic model averaging over "
                   "discount factors. Neither is set by hand.")

    st.markdown("**Named economic drivers of fair value**")
    fig(viz.contribution_heatmap(res.mve["block_contrib"], TH,
                                 title="Asset-class contribution to the fair value level"),
        "va_heat")

    a, b = st.columns(2)
    with a:
        bc = res.mve["block_contrib"].dropna(how="all")
        if len(bc):
            fig(viz.signed_bar(bc.iloc[-1], TH,
                               title="Today's contribution by asset class",
                               xtitle="log-price contribution"), "va_bar")
    with b:
        bi = res.mve["block_importance"].dropna(how="all")
        if len(bi):
            fig(viz.magnitude_bar(bi.iloc[-1], TH,
                                  title="Ablation importance (leave-one-class-out)",
                                  xtitle="share of total gap displacement"),
                "va_imp")
    with st.expander("Contribution table (today)"):
        if len(bc):
            st.dataframe(pd.DataFrame({
                "contribution to log fair value": bc.iloc[-1].round(4),
                "coefficient": res.mve["block_beta"].iloc[-1]
                .reindex(bc.columns).round(4),
                "ablation importance": bi.iloc[-1].round(4) if len(bi) else np.nan,
            }).sort_values("contribution to log fair value",
                           key=abs, ascending=False),
                width='stretch')

# ===========================================================================
# Dynamics
# ===========================================================================
with TABS[2]:
    m = st.columns(6)
    m[0].metric("DO", f"{last.get('do', np.nan):+.2f}σ")
    m[1].metric("Trend strength", f"{last.get('trend_strength', np.nan):+.2f}")
    m[2].metric("Hurst", f"{last.get('hurst', np.nan):.3f}",
                help="Above 0.5 means persistent (trending); below means "
                     "anti-persistent (mean-reverting).")
    m[3].metric("P(trending)", f"{last.get('persistence_prob', np.nan):.0%}")
    m[4].metric("Volatility", f"{100*last.get('vol_annualised', np.nan):.1f}%")
    m[5].metric("Structural stability", f"{last.get('stability', np.nan):.0%}")

    fig(viz.oscillator(S["do"], TH, "Bullish pressure", "Bearish pressure",
                       height=300, title="Dynamics Oscillator", ytitle="sigma"),
        "dy_do")

    a, b = st.columns(2)
    with a:
        fig(viz.multi_line(
            S[["trend_component", "reversion_component"]].rename(columns={
                "trend_component": "trend (weighted by P(trending))",
                "reversion_component": "reversion (weighted by 1 − P)"}),
            TH, title="How the oscillator was composed",
            ytitle="contribution"), "dy_comp")
        st.caption("The Hurst exponent decides which reading of the same price "
                   "path is operative. Because it carries a standard error, the "
                   "weight is graded rather than a switch.")
        fig(viz.multi_line(
            S[["hurst", "persistence_prob", "efficiency"]].rename(columns={
                "hurst": "Hurst exponent", "persistence_prob": "P(H > 0.5)",
                "efficiency": "entropy percentile"}), TH,
            title="Memory and efficiency", ytitle="value", fmt=":.3f"),
            "dy_mem")
    with b:
        fig(viz.multi_line(
            S[["vol_annualised", "vol_percentile"]].rename(columns={
                "vol_annualised": "annualised volatility",
                "vol_percentile": "its own percentile"}), TH,
            title="Volatility state", ytitle="", fmt=":.3f"), "dy_vol")
        st.caption("Volatility memory is selected per session by dynamic model "
                   "averaging over exponential half-lives — no ATR(14).")
        fig(viz.multi_line(
            S[["stability", "variance_stability", "entropy", "complexity"]]
            .rename(columns={"stability": "level stability (CUSUM)",
                             "variance_stability": "variance stability",
                             "entropy": "permutation entropy",
                             "complexity": "Lempel-Ziv complexity"}), TH,
            title="Structure and complexity", ytitle="", fmt=":.3f"), "dy_str")

    a, b = st.columns(2)
    with a:
        fig(viz.multi_line(S[["dominant_cycle", "mean_rev_halflife",
                              "trend_memory", "vol_memory", "window"]]
                           .rename(columns={
                               "dominant_cycle": "dominant cycle (Burg)",
                               "mean_rev_halflife": "reversion half-life",
                               "trend_memory": "trend memory",
                               "vol_memory": "volatility half-life",
                               "window": "inferred analysis window"}),
                           TH, title="Inferred time scales (sessions)",
                           ytitle="sessions"), "dy_scales")
    with b:
        fig(viz.multi_line(S[["bull_prob", "tech_confidence"]].rename(columns={
            "bull_prob": "calibrated P(up next session)",
            "tech_confidence": "technical confidence"}), TH,
            title="Calibration and demonstrated skill", ytitle="probability",
            fmt=":.1%"), "dy_conf")
        st.caption("Technical confidence is P(rho > 0) under the Fisher-z law "
                   "for the correlation between yesterday's oscillator and "
                   "today's return. A flat 50% means no evidence either way.")

# ===========================================================================
# Decision
# ===========================================================================
with TABS[3]:
    m = st.columns(6)
    m[0].metric("PDO", f"{summary['pdo']:+.2f}")
    m[1].metric("Conviction", f"{summary['conviction']:.0%}")
    m[2].metric("P(success)", f"{summary['prob_success']:.0%}")
    m[3].metric("Expected return", f"{summary['expected_return_pct']:+.2f}%",
                help="Over the inferred horizon, in price terms.")
    m[4].metric("Expected downside", f"{summary['downside_pct']:+.2f}%")
    m[5].metric("Decision confidence", f"{summary['decision_confidence']:.0%}")

    fig(viz.oscillator(S["pdo"], TH, "Constructive", "Adverse", height=320,
                       title="Paramount Decision Oscillator", ytitle="expected IR"),
        "de_pdo")

    a, b = st.columns(2)
    with a:
        fig(viz.multi_line(S[["decision_confidence", "conviction",
                              "prob_success"]].rename(columns={
            "decision_confidence": "decision confidence (kappa)",
            "conviction": "conviction percentile",
            "prob_success": "P(favourable outcome)"}), TH,
            title="Quality of the current opportunity", ytitle="probability",
            fmt=":.1%"), "de_conf")
        fig(viz.stacked_share(res.dfe["horizon_weights"], TH,
                              title="Inferred decision horizon (posterior weight)",
                              ytitle="weight"), "de_hz")
    with b:
        fig(viz.multi_line(S[["valuation_weight", "dynamics_weight"]].rename(
            columns={"valuation_weight": "valuation drives the decision",
                     "dynamics_weight": "dynamics drives the decision"}), TH,
            title="Which engine is in control", ytitle="share of sensitivity",
            fmt=":.0%"), "de_wt")
        st.caption("Read off the partial derivatives of expected return with "
                   "respect to each oscillator, including their interactions "
                   "with stress, confidence and each other. Nothing here is a "
                   "chosen weight.")
        fig(viz.multi_line(S[["expected_return_pct", "downside_pct"]].rename(
            columns={"expected_return_pct": "expected return",
                     "downside_pct": "expected downside"}), TH,
            title="Reward and downside over the inferred horizon",
            ytitle="fraction of price", fmt=":.2%"), "de_rr")

    st.markdown("**Historical decision annotations**")
    fig(viz.decision_annotations(S, TH), "de_ann")
    st.caption("The 40 sessions at which the PDO reached its largest absolute "
               "readings. Each marker is what the system published that day, "
               "using only what it knew that day.")

    st.markdown("**Walk-forward evaluation**")
    wf = walk_forward_report(res)
    st.dataframe(wf["table"], width='stretch', hide_index=True)
    st.caption("Rank IC is Spearman correlation between the published signal "
               "and the subsequent standardised return. The t-statistic is "
               "Newey-West corrected for the overlap that h-day returns "
               "induce — an uncorrected t-statistic on overlapping data "
               "overstates significance by roughly sqrt(h), which is how many "
               "published 'edges' are manufactured.")
    if wf["curves"]:
        fig(viz.walk_forward_curves(wf["curves"], TH), "de_wf")

# ===========================================================================
# Explainability
# ===========================================================================
with TABS[4]:
    st.markdown("**Latent factor dashboard**")
    a, b = st.columns([3, 2])
    with a:
        fig(viz.multi_line(res.mve["factor_levels"], TH, height=320,
                           title="Integrated latent factor levels",
                           ytitle="cumulative standardised factor"), "ex_fl")
        fig(viz.contribution_heatmap(res.mve["factor_contrib"], TH,
                                     title="Latent factor contribution to fair value",
                                     height=300, max_rows=10), "ex_fc")
    with b:
        L = res.mve["loadings"]
        if L.shape[1] > 0:
            top = L.iloc[:, 0].abs().sort_values(ascending=False).index[:18]
            fig(viz.signed_bar(L.loc[top, L.columns[0]], TH, height=430,
                               title="Loadings on the dominant factor",
                               xtitle="eigenvector weight"), "ex_load")
            st.caption("The dominant factor's composition names it: if the "
                       "heavy weights are equity beta the factor is risk "
                       "appetite; if they are duration it is the term "
                       "structure.")

    st.markdown("**Instrument-level attribution**")
    ia = res.mve["instrument_attribution"]
    if len(ia):
        cur = ia.iloc[-1]
        top = cur.abs().sort_values(ascending=False).index[:20]
        a, b = st.columns([2, 3])
        with a:
            fig(viz.signed_bar(cur[top].rename(
                index=lambda t: f"{UNIVERSE.get(t, t)}"), TH, height=520,
                title="Which instruments make up today's fair value",
                xtitle="contribution to log fair value"), "ex_inst")
        with b:
            fig(viz.contribution_heatmap(ia[list(top)].rename(
                columns=lambda t: UNIVERSE.get(t, t)), TH, height=520,
                title="…and how that has changed", max_rows=20), "ex_ih")
        st.caption("This is an exact decomposition, not an approximation: fair "
                   "value is linear in the standardised returns of the "
                   "instruments, so the contributions sum to it identically. No "
                   "post-hoc explainer is involved.")

    st.markdown("**Decision coefficients**")
    fig(viz.multi_line(res.dfe["coefficients"].drop(columns=["const"]), TH,
                       height=320, title="Time-varying fusion coefficients",
                       ytitle="coefficient", max_series=6, fmt=":.3f"),
        "ex_coef")
    with st.expander("Today's fusion state vector"):
        st.dataframe(pd.DataFrame({
            "feature value": res.dfe["features"].iloc[-1].round(4),
            "coefficient": res.dfe["coefficients"].iloc[-1].round(4),
            "contribution": (res.dfe["features"].iloc[-1]
                             * res.dfe["coefficients"].iloc[-1]).round(4),
        }), width='stretch')

# ===========================================================================
# Uncertainty
# ===========================================================================
with TABS[5]:
    a, b = st.columns(2)
    with a:
        fig(viz.multi_line(S[["pred_sd", "resid_rmse"]].rename(columns={
            "pred_sd": "predictive standard deviation",
            "resid_rmse": "realised residual RMSE"}), TH,
            title="Valuation uncertainty (log-price units)",
            ytitle="log price", fmt=":.4f"), "un_val")
        st.caption("If the predictive standard deviation systematically "
                   "understated the realised residual the model would be "
                   "overconfident; the two tracking each other is the "
                   "calibration check.")
        fig(viz.multi_line(S[["dyn_uncertainty"]].rename(columns={
            "dyn_uncertainty": "dynamics uncertainty"}), TH,
            title="Dynamics uncertainty (propagated posterior)",
            ytitle="oscillator units", fmt=":.3f"), "un_dyn")
    with b:
        fig(viz.multi_line(S[["decision_uncertainty"]].rename(columns={
            "decision_uncertainty": "decision uncertainty"}), TH,
            title="Decision uncertainty", ytitle="standardised return",
            fmt=":.3f"), "un_dec")
        fig(viz.multi_line(S[["stress", "xs_dispersion_pct", "switch_prob",
                              "regime_entropy"]].rename(columns={
            "stress": "market stress percentile",
            "xs_dispersion_pct": "cross-sectional dispersion percentile",
            "switch_prob": "P(regime switch)",
            "regime_entropy": "regime ambiguity"}), TH,
            title="Sources of uncertainty in the environment",
            ytitle="probability", fmt=":.0%"), "un_env")
        st.caption("Cross-sectional dispersion is a liquidity proxy that needs "
                   "no volume data: when liquidity provision withdraws, the "
                   "cross-section fans out even after every name has been "
                   "normalised by its own volatility.")

    st.markdown("**Oscillator distribution analysis**")
    dist = oscillator_distribution(res)
    st.dataframe(dist, width='stretch', hide_index=True)
    cols = st.columns(3)
    for i, (name, col) in enumerate([("Fair Value Oscillator", "fvo"),
                                     ("Dynamics Oscillator", "do"),
                                     ("Paramount Decision Oscillator", "pdo")]):
        with cols[i]:
            fig(viz.distribution(S[col], TH, float(last.get(col, np.nan)),
                                 title=name, xtitle="value"), f"un_d{i}")

# ===========================================================================
# Integrity
# ===========================================================================
with TABS[6]:
    st.markdown("#### Causality is a testable claim, so it is tested here")
    st.markdown(
        "<div class='amis-note'>These run the real engines on the data you are "
        "looking at. The revision-invariance test is the demanding one: it "
        "truncates the record, runs the whole stack on the shortened history, "
        "then compares every overlapping published value against the full run. "
        "Any non-zero difference is repainting.</div>",
        unsafe_allow_html=True)
    st.write("")

    d = res.diagnostics
    m = st.columns(4)
    m[0].metric("Dataset fingerprint", res.fingerprint)
    m[1].metric("Model version", res.model_version)
    m[2].metric("Compute", f"{d['seconds_mve'] + d['seconds_mde'] + d['seconds_dfe']:.0f}s")
    m[3].metric("Explanatory instruments", d["n_explanatory"])

    run_tests = st.button("Run integrity tests", type="secondary")
    if run_tests:
        px = _prices_for(res.target, "2005-01-01", TODAY)
        ref = audited_view(res)
        with st.spinner("Re-running the stack on identical inputs…"):
            det = determinism_test(px, res.target, reference=ref)
        st.session_state["det"] = det
        with st.spinner("Re-running the stack on a truncated record…"):
            rev = revision_invariance_test(px, res.target,
                                           truncations=(0.6, 0.85),
                                           reference=ref)
        st.session_state["rev"] = rev

    det = st.session_state.get("det")
    rev = st.session_state.get("rev")
    if det:
        ok = det["passed"]
        st.markdown(
            f"**Determinism** — {'PASS' if ok else 'FAIL'} · maximum absolute "
            f"difference between two independent runs on identical inputs: "
            f"`{det['max_abs_difference']:.3g}` across {det['n_compared']:,} "
            f"sessions and {len(det['per_series'])} published series.")
    if rev:
        ok = rev["passed"]
        st.markdown(
            f"**Revision invariance** — {'PASS' if ok else 'FAIL'} · maximum "
            f"absolute revision to previously published values: "
            f"`{rev['max_abs_revision']:.3g}`")
        st.dataframe(rev["table"], width='stretch', hide_index=True)
    if not det and not rev:
        st.info("Tests re-run the full stack two more times "
                "(roughly one to two minutes).")

    st.divider()
    st.markdown("#### What guarantees this by construction")
    st.markdown("""
<div class='amis-note'>

**Every estimator is a forward recursion.** Kalman/DLM filters, exponentially
weighted moments, recursive least squares, expanding empirical CDFs, online EM.
State at *t* is a function of observations 1..*t*. There is no smoother, no
centred window, no bidirectional filter, and no full-sample normalisation
anywhere in the codebase.

**The hidden Markov model is filtered, never smoothed.** Baum-Welch runs a
backward pass, so the regime it assigns to a date is computed from data that
came after it. Only the forward filter is used here, which is why the regime
history looks less decisive than the ones usually published — it is the honest
one.

**Instrument admission is causal.** An instrument joins the cross-section on
the day its own accumulated print count first reaches the estimability floor.
Screening the panel on total history would let a fund launched in 2021 rewrite
the factor structure of 2015.

**Labels lag by their horizon.** The fusion model that speaks at *t* was
trained on outcomes observable through *t − h*, and it is scored against the
prediction it actually emitted at the time, not a re-prediction from today's
parameters.

**The price store is append-only.** Total-return adjustment at the vendor
rewrites history on every dividend. A model can be perfectly non-repainting and
still produce different history if its inputs are rewritten underneath it, so
once a (ticker, date) close is written here it is never overwritten — only
extended, in both directions.

**Nothing is random.** No seeds, no stochastic initialisation, no iteration
over unordered containers. Eigenvector signs follow a fixed convention and are
matched to the previous basis, so a factor cannot silently change identity.

</div>
""", unsafe_allow_html=True)

# ===========================================================================
# Methodology
# ===========================================================================
with TABS[7]:
    st.markdown("#### Three inference problems, not one indicator")
    st.markdown("""
<div class='amis-note'>

**Engine 1 — Market Valuation.** Log price is regressed on the integrated
common factors of ~200 global instruments with time-varying coefficients: a
dynamic cointegrating regression, so the residual is a *level* and fair value
is a price rather than a forecast. Factors come from an exponentially weighted
correlation matrix whose eigenvalues are clipped at the Marchenko-Pastur edge,
which answers "how many factors" from the spectrum instead of from a
convention. Correlation memory is itself inferred: three memories run in
parallel and are scored by their own causal predictive likelihood. Two
valuation views — latent factors and named asset-class blocks — are averaged by
out-of-sample evidence, and twelve leave-one-class-out refits supply both
driver importance and a cross-sectional consistency score.

**Engine 2 — Market Dynamics.** No RSI, MACD, ATR, ADX, Stochastic, CCI or
Williams %R. Each of those is a fixed-window proxy for a latent quantity, so
the latent quantity is estimated directly: level and slope from a local linear
trend model, conditional volatility by dynamic model averaging over
exponential memories, roughness from the multiscale variance ratio, efficiency
from permutation entropy, algorithmic structure from Lempel-Ziv complexity,
the dominant cycle from a Burg maximum-entropy spectrum with AIC order
selection, and structural stability from a CUSUM of recursive residuals. The
oscillator is composed by the market's own inferred memory: the Hurst exponent,
with its standard error, decides how much weight trend persistence and mean
reversion each carry.

**Engine 3 — Decision Fusion.** A Bayesian decision problem, not an average.
Expected return, expected upside and expected downside are estimated
separately — the asymmetry is the decision-relevant part — with the horizon
itself model-averaged. The design matrix carries both oscillators, their
interactions with stress and with their own confidences, and their product, so
reinforcement and conflict are represented without being coded by hand. The
reported "valuation weight" is the partial derivative of expected return with
respect to the Fair Value Oscillator as a share of total sensitivity.

**The PDO** is the confidence-shrunk expected annualised information ratio of
acting on the current state. The shrinkage factor is the demonstrated
association between the model's own past predictions and their outcomes,
expressed as P(rho > 0) under the Fisher-z law. When nothing has been
demonstrated, the oscillator is zero — the system's way of saying it does not
know.

</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Two negative results worth stating")
    st.markdown("""
<div class='amis-note'>

**Supervised prediction of the next session's return, from the full latent
technical state, produced a forgetting-weighted R² of about −0.01 on every
asset examined.** That is the correct empirical answer for daily data, and it
is why the Dynamics Oscillator is a state description whose association with
forward returns is *measured* rather than a fitted forecast.

**Scoring discount factors by one-step predictive likelihood is degenerate for
a level regression.** The model that tracks price most closely always wins, and
its limit is the useless statement "fair value = price". Admitting a five-month
coefficient memory collapsed the mispricing to 4% standard deviation with a
seven-day half-life. The valuation family is therefore restricted to memories
of four years and longer, within which the data still selects; shifting the
grid a notch slower moves the reported mispricing by well under a percentage
point.

</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Foundations")
    st.markdown("""
<div class='amis-cite'>

Bai, J. & Ng, S. (2002). Determining the Number of Factors in Approximate Factor Models. *Econometrica* 70(1).<br>
Bandt, C. & Pompe, B. (2002). Permutation Entropy: A Natural Complexity Measure for Time Series. *Physical Review Letters* 88(17).<br>
Berger, J. O. (1985). *Statistical Decision Theory and Bayesian Analysis*. Springer.<br>
Bierens, H. J. & Martins, L. F. (2010). Time-Varying Cointegration. *Econometric Theory* 26(5).<br>
Brown, R. L., Durbin, J. & Evans, J. M. (1975). Techniques for Testing the Constancy of Regression Relationships over Time. *JRSS-B* 37(2).<br>
Burg, J. P. (1975). *Maximum Entropy Spectral Analysis*. Stanford University.<br>
Cappé, O. (2011). Online EM Algorithm for Hidden Markov Models. *JCGS* 20(3).<br>
Cesa-Bianchi, N. & Lugosi, G. (2006). *Prediction, Learning, and Games*. Cambridge.<br>
Engle, R. F. & Granger, C. W. J. (1987). Co-integration and Error Correction. *Econometrica* 55(2).<br>
Hamilton, J. D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle. *Econometrica* 57(2).<br>
Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge.<br>
Kelly, J. L. (1956). A New Interpretation of Information Rate. *Bell System Technical Journal* 35(4).<br>
Koop, G. & Korobilis, D. (2012). Forecasting Inflation Using Dynamic Model Averaging. *International Economic Review* 53(3).<br>
Laloux, L., Cizeau, P., Bouchaud, J.-P. & Potters, M. (1999). Noise Dressing of Financial Correlation Matrices. *Physical Review Letters* 83.<br>
Lempel, A. & Ziv, J. (1976). On the Complexity of Finite Sequences. *IEEE Transactions on Information Theory* 22(1).<br>
Lo, A. W. & MacKinlay, A. C. (1988). Stock Market Prices Do Not Follow Random Walks. *Review of Financial Studies* 1(1).<br>
López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.<br>
Marchenko, V. A. & Pastur, L. A. (1967). Distribution of Eigenvalues for Some Sets of Random Matrices. *Sbornik: Mathematics* 1(4).<br>
Newey, W. K. & West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica* 55(3).<br>
Opper, M. (1998). A Bayesian Approach to Online Learning. In *Online Learning in Neural Networks*. Cambridge.<br>
Plerou, V. et al. (2002). Random Matrix Approach to Cross Correlations in Financial Data. *Physical Review E* 65.<br>
Raftery, A. E., Kárný, M. & Ettler, P. (2010). Online Prediction Under Model Uncertainty via Dynamic Model Averaging. *Technometrics* 52(1).<br>
Ross, S. A. (1976). The Arbitrage Theory of Capital Asset Pricing. *Journal of Economic Theory* 13(3).<br>
Timmermann, A. (2006). Forecast Combinations. *Handbook of Economic Forecasting* 1.<br>
West, M. & Harrison, J. (1997). *Bayesian Forecasting and Dynamic Models*, 2nd ed. Springer.<br>
White, H. (2000). A Reality Check for Data Snooping. *Econometrica* 68(5).<br>
Zunino, L. et al. (2009). Forbidden Patterns, Permutation Entropy and Stock Market Inefficiency. *Physica A* 388.

</div>
""", unsafe_allow_html=True)

st.divider()
st.caption(
    "AMIS is a research instrument. It estimates the statistical quality of an "
    "opportunity; it does not issue investment advice, and position sizing "
    "depends on mandate, risk budget and capital constraints that it has no "
    "knowledge of.")
