"""
Engine 2 -- Market Dynamics Engine (MDE).
=================================================================

Question answered: *how is this asset behaving right now?*  Independently of
what it is worth.

The engine does not implement RSI, MACD, ATR, ADX, Stochastic, CCI or
Williams %R.  Each of those is a fixed-window, fixed-threshold proxy for a
latent statistical quantity, and the proxy is what makes them fragile:
RSI(14) is a crude estimate of short-horizon mean-reversion pressure, ADX(14)
of trend persistence, ATR(14) of conditional volatility.  Here the latent
quantities themselves are estimated, with their own uncertainty, and with the
memory inferred rather than declared:

===========================  =====================================================
Latent quantity              Estimator
===========================  =====================================================
Level, slope                 Local linear trend DLM (Harvey 1989), discount bank
Trend strength               Drift accumulated over the filter's own inferred
                             memory, in units of one standard deviation of that
                             accumulation -- comparable across assets, unlike
                             the posterior t-ratio, which grows without bound
                             as the filter gains confidence and routinely
                             exceeds 100
Momentum persistence         Recursive AR(1) on the filtered slope
Conditional volatility       Per-asset dynamic model averaging over EW memories
Mean-reversion potential     Online Ornstein-Uhlenbeck fit to trend deviation
Roughness / memory           Multiscale variance ratio -> generalised Hurst
Market efficiency            Permutation entropy (Bandt & Pompe 2002)
Complexity                   Lempel-Ziv complexity of the binarised path
Dominant cycle               Burg maximum-entropy spectrum, order by AIC
Structural stability         Exponentially weighted CUSUM of recursive residuals
Regime                       Forward-filtered online HMM
===========================  =====================================================

Everything is one-sided.  No centred moving average, no bidirectional
filter, no smoother appears anywhere in this file -- those are the standard
sources of look-ahead in technical work, and they are precisely what makes
a backtest of an "indicator" unfalsifiable.

References
----------
Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the
    Kalman Filter*.
Lo, A. W. & MacKinlay, A. C. (1988). "Stock Market Prices Do Not Follow
    Random Walks: Evidence from a Simple Specification Test." *RFS* 1(1).
Bandt, C. & Pompe, B. (2002). "Permutation Entropy: A Natural Complexity
    Measure for Time Series." *Phys. Rev. Lett.* 88(17).
Lempel, A. & Ziv, J. (1976). "On the Complexity of Finite Sequences."
    *IEEE Trans. Information Theory* 22(1).
Burg, J. P. (1975). *Maximum Entropy Spectral Analysis*. PhD thesis,
    Stanford University.
Brown, R. L., Durbin, J. & Evans, J. M. (1975). "Techniques for Testing the
    Constancy of Regression Relationships over Time." *JRSS-B* 37(2).
Uhlenbeck, G. E. & Ornstein, L. S. (1930). "On the Theory of the Brownian
    Motion." *Physical Review* 36.
Zunino, L. et al. (2009). "Forbidden patterns, permutation entropy and
    stock market inefficiency." *Physica A* 388.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .causal import (EPS, DynamicModelAverage, EWMA, ExpandingRank, OnlineAR1,
                     OnlineCorr, OnlineLogistic, OnlineSkill, norm_cdf_scalar,
                     probit)
from .factors import AdaptiveVolPanel
from .regime import RegimeFilter

BURN_IN = 252

#: Discount grid for the local linear trend.  Spans a fast-adapting trend
#: (~20 sessions of memory) to a near-constant drift; the data selects.
TREND_DELTAS = (0.95, 0.97, 0.98, 0.99, 0.995, 0.999)

#: Dyadic horizons for the variance-ratio / Hurst scaling profile.  Powers of
#: two are the standard scaling ladder in multiscale analysis, not a tuned
#: set; the slope across them is what is used, never a single horizon.
VR_SCALES = (2, 4, 8, 16, 32, 64)

#: Ordinal pattern length for permutation entropy.  d = 4 gives 24 patterns,
#: the largest alphabet whose frequencies are estimable from a few hundred
#: observations (Bandt & Pompe recommend 3 <= d <= 7 with n >> d!).
PE_ORDER = 4

#: Window estimators are refreshed every 5 sessions and held constant in
#: between.  This is a compute budget, and it is one-sided: a held value is
#: always a *past* estimate, never a future one, so it cannot repaint.
WINDOW_STRIDE = 5


# ===========================================================================
# Window estimators (causal: they see a trailing window and nothing else)
# ===========================================================================
def permutation_entropy(x: np.ndarray, order: int = PE_ORDER) -> float:
    """Normalised Bandt-Pompe permutation entropy of a trailing window.

    Measures how close the ordinal structure of the path is to that of an
    i.i.d. sequence.  1.0 means every ordering is equally likely -- the
    signature of an efficient, unpredictable market; values below 1 quantify
    exploitable temporal structure without assuming what form it takes.
    """
    n = len(x)
    if n < order * 10:
        return np.nan
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        w = sliding_window_view(x, order)
    except Exception:
        return np.nan
    perms = np.argsort(w, axis=1, kind="stable")
    # encode each ordinal pattern as a base-`order` integer
    codes = np.zeros(len(perms), dtype=np.int64)
    for j in range(order):
        codes = codes * order + perms[:, j]
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    h = -float(np.sum(p * np.log(p)))
    return h / math.log(math.factorial(order))


def _lz76(s: np.ndarray) -> int:
    """LZ76 production count of a binary sequence."""
    n = len(s)
    i, k, l = 0, 1, 1
    c, kmax = 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > kmax:
                kmax = k
            i += 1
            if i == l:
                c += 1
                l += kmax
                if l + 1 > n:
                    break
                i, k, kmax = 0, 1, 1
            else:
                k = 1
    return c


try:                                    # pragma: no cover - optional accelerator
    from numba import njit as _njit

    _lz76_fast = _njit(cache=True, nogil=True)(_lz76)
    _lz76_fast(np.zeros(8, dtype=np.int8))          # warm the cache, deterministic
except Exception:                       # pragma: no cover
    _lz76_fast = _lz76


def lempel_ziv(x: np.ndarray) -> float:
    """Normalised LZ76 complexity of the sign-binarised path.

    Complements permutation entropy: entropy is a distributional statement,
    LZ complexity an algorithmic one (how compressible the realised path is).
    A market can look high-entropy and still be algorithmically simple.
    Normalisation is by n / log2(n), the asymptotic complexity of an i.i.d.
    binary source, so 1.0 is the incompressible reference.
    """
    n = len(x)
    if n < 64:
        return np.nan
    s = (x > np.median(x)).astype(np.int8)
    return float(_lz76_fast(s) * math.log2(n) / n)


def burg_dominant_cycle(x: np.ndarray) -> tuple[float, float]:
    """Dominant period and its spectral prominence, by Burg's method.

    Burg's maximum-entropy estimator is used rather than a periodogram
    because the window is short (hundreds of points) and the periodogram's
    resolution is then too coarse to separate a genuine cycle from the
    1/f-like background.  Model order is selected by AIC, so the cycle count
    is inferred rather than imposed.
    """
    n = len(x)
    if n < 64:
        return np.nan, np.nan
    x = x - x.mean()
    sd = x.std()
    if sd <= EPS:
        return np.nan, np.nan
    x = x / sd

    pmax = int(min(30, n // 8))
    f = x.copy()
    b = x.copy()
    a = np.array([1.0])
    e = float(np.dot(x, x)) / n
    best = (np.inf, None, None)
    for p in range(1, pmax + 1):
        num = -2.0 * float(np.dot(f[p:], b[p - 1:-1]))
        den = float(np.dot(f[p:], f[p:]) + np.dot(b[p - 1:-1], b[p - 1:-1]))
        if abs(den) < EPS:
            break
        k = num / den
        a = np.concatenate([a, [0.0]]) + k * np.concatenate([[0.0], a[::-1]])
        e *= (1.0 - k * k)
        if e <= EPS:
            break
        fn = f[p:] + k * b[p - 1:-1]
        bn = b[p - 1:-1] + k * f[p:]
        f = np.concatenate([f[:p], fn])
        b = np.concatenate([b[:p], bn])
        aic = n * math.log(max(e, 1e-12)) + 2.0 * p
        if aic < best[0]:
            best = (aic, a.copy(), e)
    if best[1] is None:
        return np.nan, np.nan

    coef = best[1]
    # Evaluate the AR spectrum from the Nyquist limit out to ~6 months.
    # Below 4 sessions the estimate is aliased; beyond roughly 128 sessions a
    # spectral peak is indistinguishable from the 1/f background of a drifting
    # price and would be reported as a "cycle" on every asset.
    periods = np.arange(4.0, min(128.0, max(8.0, n / 4.0)))
    w = 2.0 * math.pi / periods
    j = np.arange(len(coef))
    denom = np.abs(np.exp(-1j * np.outer(w, j)) @ coef) ** 2
    spec = best[2] / np.maximum(denom, 1e-12)
    i = int(np.argmax(spec))
    prom = float(spec[i] / max(np.median(spec), 1e-12))
    return float(periods[i]), prom


# ===========================================================================
# Engine
# ===========================================================================
class MarketDynamicsEngine:
    """Recursive inference of the asset's latent technical state."""

    def run(self, target_px: pd.Series, progress_cb=None) -> dict:
        idx = target_px.index
        T = len(idx)
        P = np.log(np.asarray(target_px.values, dtype=float))
        r = np.concatenate([[np.nan], np.diff(P)])

        G = np.array([[1.0, 1.0], [0.0, 1.0]])          # local linear trend
        llt: DynamicModelAverage | None = None

        vol = AdaptiveVolPanel(1)
        regime = RegimeFilter(d=2)

        slope_ar = OnlineAR1(halflife=252.0)
        ret_ar = OnlineAR1(halflife=252.0)
        dev_ar = OnlineAR1(halflife=252.0)

        # multiscale variance ratio: EW second moments of q-period returns
        vr_ew = [EWMA(halflife=504.0) for _ in VR_SCALES]
        r1_ew = EWMA(halflife=504.0)

        logvol_ew = EWMA(halflife=252.0)
        vol_rank = ExpandingRank()
        cusum = EWMA(halflife=126.0)
        cusum2 = EWMA(halflife=126.0)

        do_rank = ExpandingRank()
        ent_rank = ExpandingRank()
        # Platt-style online calibration of the oscillator into a probability.
        # Two free parameters (slope and intercept) fitted causally -- the
        # oscillator's *construction* is theoretical, only its mapping onto a
        # probability scale is learned.
        do_logit = OnlineLogistic(2, delta=0.999, prior_scale=1.0)
        do_corr = OnlineCorr(halflife=504.0)
        skill = OnlineSkill(halflife=504.0)

        keys = ("level", "slope", "trend_strength", "trend_tstat",
                "trend_persistence", "momentum_persistence", "mean_rev_prob",
                "mean_rev_halflife", "trend_deviation", "volatility",
                "vol_annualised", "vol_percentile", "vol_of_vol", "noise_ratio",
                "signal_to_noise", "entropy", "efficiency", "complexity",
                "hurst", "hurst_se", "persistence_prob", "variance_ratio",
                "dominant_cycle", "cycle_prominence", "stability",
                "variance_stability", "do", "do_raw", "trend_component",
                "reversion_component", "bull_prob", "bear_prob",
                "tech_confidence", "dyn_uncertainty", "regime_stress",
                "switch_prob", "regime_entropy", "vol_memory", "trend_memory",
                "hit_rate", "pred_r2", "window")
        out = {k: np.full(T, np.nan) for k in keys}
        out["regime_label"] = np.array(["initialising"] * T, dtype=object)
        feat_names = ["trend_snr", "mom_persist", "vr_dev", "hurst_dev",
                      "reversion", "vol_state", "entropy_dev", "stability",
                      "ret_autocorr"]
        feats = np.full((T, len(feat_names)), np.nan)

        prev_feat = None
        cache = {"entropy": np.nan, "complexity": np.nan, "cycle": np.nan,
                 "prom": np.nan, "win": np.nan}

        for t in range(T):
            p = P[t]
            if not np.isfinite(p):
                continue

            # ---- volatility (pre-update: knows only t-1) -------------------
            sig = float(vol.sigma()[0])
            vol_hl = float(vol.effective_halflife()[0])
            rt = r[t]
            vol.update(np.array([rt if np.isfinite(rt) else np.nan]))

            lv = math.log(max(sig, 1e-8))
            vpct = vol_rank.cdf(lv) if logvol_ew.n > 60 else np.nan
            vol_rank.update(lv)
            vov = logvol_ew.std
            logvol_ew.update(lv)

            # ---- regime ----------------------------------------------------
            rz = rt / sig if (np.isfinite(rt) and sig > EPS) else 0.0
            lvz = (lv - logvol_ew.mean) / max(logvol_ew.std, 1e-6) if logvol_ew.n > 20 else 0.0
            reg = regime.update(np.array([np.clip(rz, -8, 8), np.clip(lvz, -8, 8)]))

            # ---- local linear trend ---------------------------------------
            if llt is None:
                if t < 30:
                    continue
                hist = P[max(0, t - 30):t]
                v0 = max(float(np.var(np.diff(hist))), 1e-8)
                llt = DynamicModelAverage(
                    2, grid=TREND_DELTAS, prior_scale=v0 * 100.0,
                    prior_var=v0, prior_mean=np.array([float(np.mean(hist)), 0.0]),
                    G=G)
            F = np.array([1.0, 0.0])
            f_pred, q_pred = llt.forecast(F)
            llt.update(F, p)
            coef = llt.coef
            cvar = llt.coef_var
            level, slope = float(coef[0]), float(coef[1])
            slope_sd = math.sqrt(max(cvar[1], 1e-16))
            trend_t = slope / slope_sd if slope_sd > EPS else 0.0

            innov = (p - f_pred) / math.sqrt(max(q_pred, 1e-16))
            cusum.update(innov)
            cusum2.update(innov * innov - 1.0)
            n_eff = 1.0 / (1.0 - cusum.lam)
            c_stat = cusum.mean * math.sqrt(n_eff)
            c2_stat = cusum2.mean * math.sqrt(n_eff / 2.0)
            stability = math.exp(-0.5 * min(c_stat * c_stat, 60.0))
            var_stability = math.exp(-0.5 * min(c2_stat * c2_stat, 60.0))

            # ---- persistence / reversion ----------------------------------
            slope_ar.update(slope)
            phi_slope, _ = slope_ar.solve()
            if np.isfinite(rt):
                ret_ar.update(rt / max(sig, 1e-8))
            phi_ret, se_ret = ret_ar.solve()

            dev = (p - level) / max(sig * math.sqrt(max(vol_hl, 1.0)), 1e-8)
            dev_ar.update(dev)
            phi_dev, se_dev = dev_ar.solve()
            if np.isfinite(phi_dev) and np.isfinite(se_dev) and se_dev > EPS:
                mr_prob = norm_cdf_scalar((1.0 - phi_dev) / se_dev)
                mr_hl = (math.log(2.0) / -math.log(min(max(phi_dev, 1e-6), 0.999999))
                         if 0 < phi_dev < 1 else np.nan)
            else:
                mr_prob, mr_hl = np.nan, np.nan

            # ---- multiscale variance ratio and Hurst ----------------------
            if np.isfinite(rt):
                r1_ew.update(rt * rt)
                for si, q in enumerate(VR_SCALES):
                    if t >= q:
                        rq = P[t] - P[t - q]
                        if np.isfinite(rq):
                            vr_ew[si].update(rq * rq)
            v1 = r1_ew.mean
            vr_vals = np.array([m.mean for m in vr_ew])
            hurst, hurst_se, vr_main = np.nan, np.nan, np.nan
            if v1 > EPS and np.all(vr_vals > EPS):
                # log Var(q-period return) = 2H log q + c.  Weighted least
                # squares with Var(log v_q) ~ 2q/n_eff, the Gaussian sampling
                # variance of a log variance with n_eff/q independent blocks:
                # this is what makes the standard error of H meaningful, and
                # the standard error is what turns the exponent into a
                # *probability* of trending rather than a point guess.
                n_eff = min(float(t), 1.0 / (1.0 - vr_ew[0].lam))
                lq = np.log(np.array(VR_SCALES, dtype=float))
                lv_ = np.log(vr_vals / v1)
                wq = n_eff / (2.0 * np.array(VR_SCALES, dtype=float))
                A = np.vstack([np.ones_like(lq), lq]).T
                Aw = A * wq[:, None]
                try:
                    XtX = A.T @ Aw
                    sol = np.linalg.solve(XtX, Aw.T @ lv_)
                    hurst = float(sol[1] / 2.0)
                    hurst_se = float(0.5 * math.sqrt(max(np.linalg.inv(XtX)[1, 1], 0.0)))
                except np.linalg.LinAlgError:
                    pass
                vr_main = float(vr_vals[2] / (VR_SCALES[2] * v1))     # q = 8

            # ---- window estimators (refreshed on a stride) -----------------
            win = int(np.clip(round(10.0 * vol_hl), 250, 750))
            if t >= win and (t % WINDOW_STRIDE == 0 or np.isnan(cache["entropy"])):
                seg = r[t - win + 1: t + 1]
                seg = seg[np.isfinite(seg)]
                if len(seg) > 100:
                    cache["entropy"] = permutation_entropy(seg)
                    cache["complexity"] = lempel_ziv(seg)
                    dseg = P[t - win + 1: t + 1]
                    dseg = dseg[np.isfinite(dseg)]
                    if len(dseg) > 64:
                        # detrend one-sidedly: cumulative sum of demeaned
                        # returns leaves the oscillatory component
                        cyc, prom = burg_dominant_cycle(np.diff(dseg))
                        cache["cycle"], cache["prom"] = cyc, prom
                    cache["win"] = win

            snr = 0.0
            if llt is not None:
                sv = float(llt.obs_var)
                snr = (slope * slope) / max(sv, 1e-12)

            if t < BURN_IN or llt is None:
                continue

            # ================================================================
            # Dynamics Oscillator
            # ----------------------------------------------------------------
            # The oscillator is *constructed*, not fitted, and the thing that
            # decides how to construct it is the market's own inferred memory
            # structure.  Trend persistence and mean reversion are the two
            # directional readings of an identical price path; which one is
            # operative is exactly the question the Hurst exponent answers.
            # pi = P(H > 1/2) therefore does the weighting -- and because it
            # comes with a standard error it is a graded weight, not a switch.
            #
            # Deliberate negative result: supervised prediction of the next
            # session's return from these same latent states was tried and
            # produced a forgetting-weighted R^2 of about -0.01 across every
            # asset examined.  That is the correct empirical answer for daily
            # data and it is why no such forecast is published here.  What is
            # published instead is a state description whose association with
            # forward returns is *measured* online and reported as confidence.
            # ================================================================
            mem = max(float(llt.effective_memory), 1.0)
            trend_snr = slope * math.sqrt(mem) / max(sig, 1e-8)
            if np.isfinite(hurst) and np.isfinite(hurst_se) and hurst_se > EPS:
                zH = (hurst - 0.5) / hurst_se
                pi = norm_cdf_scalar(zH)
                sd_pi = math.exp(-0.5 * zH * zH) / math.sqrt(2.0 * math.pi)
            else:
                pi, sd_pi = 0.5, 0.5

            c_trend = math.tanh(trend_snr)
            c_rev = math.tanh(-dev)
            do_raw = stability * (pi * c_trend + (1.0 - pi) * c_rev)

            # first-order propagation of the posterior uncertainty in the
            # level, the slope and the memory weight
            u_tr = (math.sqrt(max(cvar[1], 0.0)) * math.sqrt(mem) / max(sig, 1e-8)
                    ) * (1.0 - c_trend * c_trend)
            u_rv = (math.sqrt(max(cvar[0], 0.0)) / max(sig * math.sqrt(mem), 1e-8)
                    ) * (1.0 - c_rev * c_rev)
            unc = float(math.sqrt((pi * u_tr) ** 2 + ((1.0 - pi) * u_rv) ** 2
                                  + ((c_trend - c_rev) * sd_pi) ** 2))

            u = do_rank.cdf(do_raw)
            do_rank.update(do_raw)
            # probit of the causal empirical rank: an oscillator whose scale
            # is fixed by the asset's own realised history, not by a constant
            do = float(np.clip(probit(u), -3.5, 3.5))

            ent = cache["entropy"]
            eff_pct = ent_rank.cdf(ent) if np.isfinite(ent) else np.nan
            if np.isfinite(ent):
                ent_rank.update(ent)

            # calibrate the oscillator onto a probability scale, and measure
            # its realised association with the return it did not see
            if prev_feat is not None and np.isfinite(rt):
                do_prev, sig_prev = prev_feat
                y = np.clip(rt / max(sig_prev, 1e-8), -8.0, 8.0)
                do_logit.update(np.array([1.0, do_prev]), 1.0 if rt > 0 else 0.0)
                do_corr.update(do_prev, y)
                skill.update(do_prev, y)          # sign agreement only
            pbull, plat = do_logit.predict(np.array([1.0, do]))
            prev_feat = (do, sig)

            conf = float(do_corr.evidence)
            r2 = do_corr.rho ** 2 * np.sign(do_corr.rho)

            x = np.array([
                math.tanh(trend_snr),
                np.tanh(phi_slope * 2.0) if np.isfinite(phi_slope) else 0.0,
                np.tanh((vr_main - 1.0) * 2.0) if np.isfinite(vr_main) else 0.0,
                (2.0 * pi - 1.0),
                -np.tanh(dev),
                (vpct - 0.5) * 2.0 if np.isfinite(vpct) else 0.0,
                (eff_pct - 0.5) * 2.0 if np.isfinite(eff_pct) else 0.0,
                stability * 2.0 - 1.0,
                np.tanh(phi_ret * 3.0) if np.isfinite(phi_ret) else 0.0,
            ])
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            out["level"][t] = math.exp(level)
            out["slope"][t] = slope
            # trend strength is reported as accumulated drift over the
            # filter's own memory, in units of one standard deviation of that
            # accumulation -- a comparable quantity across assets, unlike the
            # raw posterior t-ratio, which grows without bound as the filter
            # gains confidence and routinely exceeds 100
            out["trend_strength"][t] = trend_snr
            out["trend_tstat"][t] = trend_t
            out["trend_persistence"][t] = phi_slope
            out["momentum_persistence"][t] = phi_ret
            out["mean_rev_prob"][t] = mr_prob
            out["mean_rev_halflife"][t] = mr_hl
            out["trend_deviation"][t] = dev
            out["volatility"][t] = sig
            out["vol_annualised"][t] = sig * math.sqrt(252.0)
            out["vol_percentile"][t] = vpct
            out["vol_of_vol"][t] = vov
            out["noise_ratio"][t] = 1.0 / (1.0 + snr)
            out["signal_to_noise"][t] = snr
            out["entropy"][t] = ent
            out["efficiency"][t] = eff_pct
            out["complexity"][t] = cache["complexity"]
            out["hurst"][t] = hurst
            out["hurst_se"][t] = hurst_se
            out["persistence_prob"][t] = pi
            out["variance_ratio"][t] = vr_main
            out["dominant_cycle"][t] = cache["cycle"]
            out["cycle_prominence"][t] = cache["prom"]
            out["stability"][t] = stability
            out["variance_stability"][t] = var_stability
            out["do"][t] = do
            out["do_raw"][t] = do_raw
            out["trend_component"][t] = pi * c_trend
            out["reversion_component"][t] = (1.0 - pi) * c_rev
            out["bull_prob"][t] = pbull
            out["bear_prob"][t] = 1.0 - pbull
            out["tech_confidence"][t] = conf
            out["dyn_uncertainty"][t] = unc
            out["regime_stress"][t] = reg["stress"]
            out["switch_prob"][t] = reg["switch_prob"]
            out["regime_entropy"][t] = reg["entropy"]
            out["regime_label"][t] = reg["label"]
            out["vol_memory"][t] = vol_hl
            out["trend_memory"][t] = mem
            out["hit_rate"][t] = skill.hit_rate
            out["pred_r2"][t] = r2
            out["window"][t] = cache["win"]
            feats[t] = x

            if progress_cb is not None and (t % 250 == 0 or t == T - 1):
                progress_cb((t + 1) / T)

        df = pd.DataFrame(out, index=idx)
        df["price"] = np.exp(P)
        return {
            "series": df,
            "features": pd.DataFrame(feats, index=idx, columns=feat_names),
            "feature_names": feat_names,
            "burn_in": BURN_IN,
        }
