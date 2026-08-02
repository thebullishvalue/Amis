"""
Engine 3 -- Decision Fusion Engine (DFE).
=================================================================

Question answered: *given the valuation state, the technical state, and how
much either can be trusted right now, what is the statistically optimal
decision?*

Not an average
--------------
Averaging two oscillators asserts in advance that they are equally
informative in every regime, which is the one thing the data reliably
refutes.  The fusion here is a Bayesian decision problem in the sense of
Berger (1985): posit a loss, estimate the predictive distribution of the
outcome conditional on the state, and report the action that minimises
expected loss together with the *quality* of that action.

Three quantities are estimated online, each by a bank of time-varying
regressions averaged over adaptation speeds:

    E[y | state],   E[max(y,0) | state],   E[min(y,0) | state]

where ``y`` is the forward return standardised by the volatility knowable at
decision time.  Upside and downside are estimated separately rather than
inferred from a symmetric predictive distribution, because the asymmetry is
the decision-relevant part: two states with identical expected return and
identical variance are not equally attractive if one of them gets there
through a fat left tail.

Where the adaptive weighting comes from
---------------------------------------
The design matrix contains the two oscillators, their interactions with
market stress and with their own confidences, and their product.  The
product term is what lets the model represent *reinforcement* and *conflict*
without either being coded by hand: the partial derivative of expected
return with respect to the Fair Value Oscillator,

    d mu / d FVO = b1 + b3*stress + b5*val_conf + b9*DO ,

is itself a function of the regime and of the other engine's reading.  The
reported "valuation weight" is this derivative's share of total sensitivity.
Nothing about it is chosen; it is read off coefficients that were fitted
forward in time.

Horizon
-------
The decision horizon is not specified either.  A bank of horizons is run in
parallel and averaged by predictive likelihood.  Because every target is
standardised to unit unconditional variance, densities are directly
comparable across horizons: each measures how far the conditional
distribution improves on the unconditional one.

Label timing
------------
An h-day outcome is only observable h days after the state that produced it.
The model that speaks at time t was therefore trained on labels through
t-h -- never on the label it is being asked about.  This lag is visible in
the code (:class:`_HorizonModel.pending`) rather than assumed away, and it
is the single most common place where systems of this kind leak.

The Paramount Decision Oscillator
---------------------------------
    PDO = kappa * (mu / s) * sqrt(252 / h)

the confidence-shrunk expected *annualised information ratio* of acting on
the current state.  ``kappa`` is not a prior belief: it is the demonstrated
association between this model's past predictions and their realised
outcomes, expressed as P(rho > 0) under the Fisher-z law and mapped to zero
when there is no evidence.  A PDO of +1.8 asserts that acting on today's
state has, on the evidence accumulated so far and after shrinking for
demonstrated reliability, the expected quality of a 1.8 information-ratio
opportunity.  When the engines conflict, or when neither has shown skill in
this regime, kappa collapses and the PDO goes to zero on its own.

References
----------
Berger, J. O. (1985). *Statistical Decision Theory and Bayesian Analysis*.
Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance* 7(1).
Kelly, J. L. (1956). "A New Interpretation of Information Rate." *Bell
    System Technical Journal* 35(4).
Cesa-Bianchi, N. & Lugosi, G. (2006). *Prediction, Learning, and Games*.
Raftery, A. E., Karny, M. & Ettler, P. (2010). "Online Prediction Under
    Model Uncertainty via Dynamic Model Averaging." *Technometrics* 52(1).
Timmermann, A. (2006). "Forecast Combinations." *Handbook of Economic
    Forecasting*, vol. 1.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import pandas as pd

from .causal import (EPS, DynamicModelAverage, ExpandingRank, OnlineCorr,
                     OnlineLogistic, norm_cdf_scalar)
from .dynamics import _probit

#: Decision horizons in sessions (one week to one quarter).  The bank is the
#: support of a prior over horizon; the weights are data-driven.
HORIZONS = (5, 21, 63)

#: Adaptation speeds for the fusion regressions -- faster than the valuation
#: family, because the *relationship* between state and outcome genuinely
#: does change with the regime, whereas a long-run pricing relation should
#: not.
FUSION_DELTAS = (0.99, 0.995, 0.999, 1.0)

FEATURES = [
    "const", "fvo", "do", "fvo_x_stress", "do_x_stress", "fvo_x_conf",
    "do_x_conf", "vol_state", "stability", "fvo_x_do",
]
NF = len(FEATURES)

#: Indices whose coefficients enter the valuation sensitivity and the
#: dynamics sensitivity respectively.
_VAL_TERMS = (1, 3, 5, 9)
_DYN_TERMS = (2, 4, 6, 9)


class _HorizonModel:
    """One decision horizon: mean, upside, downside, and success probability."""

    def __init__(self, h: int) -> None:
        self.h = h
        self.mean = DynamicModelAverage(NF, grid=FUSION_DELTAS,
                                        prior_scale=0.02, prior_var=1.0)
        self.up = DynamicModelAverage(NF, grid=FUSION_DELTAS,
                                      prior_scale=0.02, prior_var=1.0)
        self.down = DynamicModelAverage(NF, grid=FUSION_DELTAS,
                                        prior_scale=0.02, prior_var=1.0)
        self.logit = OnlineLogistic(NF, delta=0.999, prior_scale=0.02)
        self.corr = OnlineCorr(halflife=756.0)
        self.logw = 0.0
        self.last_ll = 0.0
        #: (state, log price, volatility) awaiting their outcome
        self.pending: deque = deque()

    def absorb(self, x: np.ndarray, p_then: float, sig_then: float,
               mu_then: float, p_now: float) -> None:
        """Train on a state whose h-day outcome has just become observable.

        ``mu_then`` is the prediction that was genuinely emitted at the time
        the state was recorded, not a re-prediction from the model as it
        stands today.  Scoring against a re-prediction would flatter the
        engine: the model has absorbed h days of data since.
        """
        denom = sig_then * math.sqrt(self.h)
        if denom <= EPS or not np.isfinite(p_now) or not np.isfinite(p_then):
            return
        y = float(np.clip((p_now - p_then) / denom, -8.0, 8.0))
        w_pre = self.mean.w.copy()
        self.mean.update(x, y)
        self.up.update(x, max(y, 0.0))
        self.down.update(x, min(y, 0.0))
        self.logit.update(x, 1.0 if y > 0 else 0.0)
        self.corr.update(mu_then, y)
        ll = np.where(np.isfinite(self.mean.log_pred_lik),
                      self.mean.log_pred_lik, -1e6)
        mx = float(ll.max())
        self.last_ll = float(mx + math.log(
            max(float(np.sum(w_pre * np.exp(ll - mx))), 1e-300)))

    def predict(self, x: np.ndarray) -> dict:
        mu, var = self.mean.forecast(x)
        up, _ = self.up.forecast(x)
        dn, _ = self.down.forecast(x)
        p, _ = self.logit.predict(x)
        s = math.sqrt(max(var, 1e-12))
        return {"mu": mu, "s": s, "up": max(up, 0.0), "down": min(dn, 0.0),
                "p_success": p, "coef": self.mean.coef}


class DecisionFusionEngine:
    """Probabilistic synthesis of valuation, dynamics, regime and uncertainty."""

    def run(self, mve: pd.DataFrame, mde: pd.DataFrame,
            progress_cb=None) -> dict:
        idx = mve.index
        T = len(idx)
        P = np.log(np.asarray(mve["price"].values, dtype=float))

        col = {c: np.asarray(mve[c].values, dtype=float) for c in
               ("fvo", "confidence", "stress", "xs_consistency", "switch_prob",
                "xs_dispersion_pct", "pred_sd", "mr_prob")}
        dcol = {c: np.asarray(mde[c].values, dtype=float) for c in
                ("do", "tech_confidence", "vol_percentile", "stability",
                 "dyn_uncertainty", "volatility", "persistence_prob",
                 "efficiency")}

        models = [_HorizonModel(h) for h in HORIZONS]
        logw = np.zeros(len(HORIZONS))
        pdo_rank = ExpandingRank()
        conv_rank = ExpandingRank()

        keys = ("pdo", "pdo_raw", "conviction", "expected_quality",
                "expected_reward", "expected_downside", "expected_shortfall",
                "prob_success", "decision_confidence", "decision_uncertainty",
                "expected_return_pct", "downside_pct", "reward_risk",
                "valuation_weight", "dynamics_weight", "agreement",
                "horizon_days", "kelly_fraction", "signal_state")
        out = {k: np.full(T, np.nan) for k in keys}
        out["stance"] = np.array([""] * T, dtype=object)
        coef_hist = np.full((T, NF), np.nan)
        hw_hist = np.full((T, len(HORIZONS)), np.nan)

        feats = np.full((T, NF), np.nan)

        for t in range(T):
            fvo = col["fvo"][t]
            do = dcol["do"][t]
            if not (np.isfinite(fvo) and np.isfinite(do)):
                continue

            vc = col["confidence"][t]
            tc = dcol["tech_confidence"][t]
            st = col["stress"][t]
            vp = dcol["vol_percentile"][t]
            sb = dcol["stability"][t]
            vc = 0.5 if not np.isfinite(vc) else vc
            tc = 0.5 if not np.isfinite(tc) else tc
            st = 0.5 if not np.isfinite(st) else st
            vp = 0.5 if not np.isfinite(vp) else vp
            sb = 0.5 if not np.isfinite(sb) else sb

            f = np.clip(fvo / 3.0, -1.5, 1.5)
            d = np.clip(do / 3.0, -1.5, 1.5)
            x = np.array([
                1.0, f, d,
                f * (2.0 * st - 1.0), d * (2.0 * st - 1.0),
                f * (2.0 * vc - 1.0), d * (2.0 * tc - 1.0),
                (2.0 * vp - 1.0), (2.0 * sb - 1.0),
                f * d,
            ])
            feats[t] = x

            # ---- absorb every state whose outcome has now matured ---------
            # Order is load-bearing: labels first, then predict, then queue
            # today's state.  Reversing the first two would let today's own
            # outcome inform today's prediction.
            for m in models:
                while len(m.pending) >= m.h:
                    x_old, p_old, s_old, mu_old = m.pending.popleft()
                    if np.isfinite(s_old):
                        m.absorb(x_old, p_old, s_old, mu_old, P[t])
            lls = np.array([m.last_ll for m in models])
            if np.all(np.isfinite(lls)) and np.any(lls != 0.0):
                logw = 0.99 * logw + lls
                logw -= logw.max()
            w = np.exp(logw - logw.max())
            w = w / max(w.sum(), EPS)

            preds = [m.predict(x) for m in models]
            for m, pr in zip(models, preds):
                m.pending.append((x.copy(), P[t], dcol["volatility"][t], pr["mu"]))
            hw_hist[t] = w

            mu = float(sum(wi * p["mu"] for wi, p in zip(w, preds)))
            s = float(math.sqrt(max(sum(
                wi * (p["s"] ** 2 + (p["mu"] - mu) ** 2)
                for wi, p in zip(w, preds)), 1e-12)))
            up = float(sum(wi * p["up"] for wi, p in zip(w, preds)))
            dn = float(sum(wi * p["down"] for wi, p in zip(w, preds)))
            psucc = float(sum(wi * p["p_success"] for wi, p in zip(w, preds)))
            coef = np.sum([wi * p["coef"] for wi, p in zip(w, preds)], axis=0)
            h_eff = float(sum(wi * m.h for wi, m in zip(w, models)))

            # demonstrated, not assumed: kappa is zero unless this model's own
            # past predictions have actually tracked their outcomes
            ev = float(sum(wi * m.corr.evidence for wi, m in zip(w, models)))
            kappa = max(0.0, 2.0 * (ev - 0.5))

            ir = mu / max(s, 1e-9)
            pdo_raw = ir * math.sqrt(252.0 / max(h_eff, 1.0))
            pdo = kappa * pdo_raw

            # ---- sensitivities: which engine is driving the decision -------
            s_val = (coef[1] + coef[3] * (2.0 * st - 1.0)
                     + coef[5] * (2.0 * vc - 1.0) + coef[9] * d)
            s_dyn = (coef[2] + coef[4] * (2.0 * st - 1.0)
                     + coef[6] * (2.0 * tc - 1.0) + coef[9] * f)
            tot = abs(s_val) + abs(s_dyn)
            w_val = abs(s_val) / tot if tot > EPS else np.nan
            # sign agreement of the two engines' *contributions* to the
            # decision, which is not the same as agreement of the oscillators
            cv, cd = s_val * f, s_dyn * d
            agree = (math.copysign(1.0, cv) * math.copysign(1.0, cd)
                     if abs(cv) > EPS and abs(cd) > EPS else 0.0)

            sig_t = dcol["volatility"][t]
            scale = (sig_t * math.sqrt(h_eff)) if np.isfinite(sig_t) else np.nan
            unc = s * math.sqrt(1.0 + max(dcol["dyn_uncertainty"][t], 0.0) ** 2) \
                if np.isfinite(dcol["dyn_uncertainty"][t]) else s
            # Gaussian expected shortfall of the conditional predictive at 5%
            z05 = 1.6448536269514722
            phi05 = math.exp(-0.5 * z05 * z05) / math.sqrt(2.0 * math.pi)
            es = mu - s * phi05 / 0.05

            quality = kappa * 0.5 * ir * ir * (252.0 / max(h_eff, 1.0))
            kelly = mu / max(s * s, 1e-9) * kappa

            u = pdo_rank.cdf(pdo)
            pdo_rank.update(pdo)
            conv = conv_rank.cdf(abs(pdo))
            conv_rank.update(abs(pdo))

            out["pdo"][t] = pdo
            out["pdo_raw"][t] = pdo_raw
            out["conviction"][t] = conv
            out["expected_quality"][t] = quality
            out["expected_reward"][t] = up
            out["expected_downside"][t] = dn
            out["expected_shortfall"][t] = es
            out["prob_success"][t] = psucc
            out["decision_confidence"][t] = kappa
            out["decision_uncertainty"][t] = unc
            out["expected_return_pct"][t] = mu * scale
            out["downside_pct"][t] = dn * scale
            out["reward_risk"][t] = up / abs(dn) if abs(dn) > EPS else np.nan
            out["valuation_weight"][t] = w_val
            out["dynamics_weight"][t] = 1.0 - w_val if np.isfinite(w_val) else np.nan
            out["agreement"][t] = agree
            out["horizon_days"][t] = h_eff
            out["kelly_fraction"][t] = kelly
            out["signal_state"][t] = _probit(u)
            out["stance"][t] = _stance(pdo, kappa)
            coef_hist[t] = coef

            if progress_cb is not None and (t % 250 == 0 or t == T - 1):
                progress_cb((t + 1) / T)

        df = pd.DataFrame(out, index=idx)
        return {
            "series": df,
            "coefficients": pd.DataFrame(coef_hist, index=idx, columns=FEATURES),
            "horizon_weights": pd.DataFrame(
                hw_hist, index=idx, columns=[f"{h}d" for h in HORIZONS]),
            "features": pd.DataFrame(feats, index=idx, columns=FEATURES),
        }


def _stance(pdo: float, kappa: float) -> str:
    """Qualitative reading of the oscillator.

    Deliberately *not* a buy/sell instruction: the PDO measures opportunity
    quality, and position sizing is the caller's mandate, risk budget and
    capital constraint -- none of which this system knows about.
    """
    if not np.isfinite(pdo):
        return ""
    if kappa < 0.05:
        return "No demonstrated edge"
    a = abs(pdo)
    if a < 0.25:
        return "Negligible opportunity"
    tier = "Marginal" if a < 0.5 else ("Moderate" if a < 1.0 else
                                       ("Strong" if a < 2.0 else "Exceptional"))
    return f"{tier} {'constructive' if pdo > 0 else 'adverse'}"
