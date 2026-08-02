# AMIS — Autonomous Market Intelligence System

Three latent inference problems about any traded asset, answered from the
global cross-section, with one input: **which asset**.

```
                        Global market data (227 instruments, 12 asset classes)
                                       │
              ┌────────────────────────┴────────────────────────┐
              ▼                                                 ▼
   Market Valuation Engine                          Market Dynamics Engine
   "where should it trade?"                         "how is it behaving?"
              │                                                 │
   Fair Value Oscillator                            Dynamics Oscillator
              └────────────────────────┬────────────────────────┘
                                       ▼
                            Decision Fusion Engine
                       "what is the optimal decision?"
                                       │
                        Paramount Decision Oscillator
```

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads 227 instruments (about a minute) into an append-only
parquet store under `.amis_cache/`; later runs are incremental. A full
analysis is roughly 45 seconds of compute.

---

## What the three engines do

### Engine 1 — Market Valuation

Log price is regressed on the **integrated common factors of the global
cross-section** with time-varying coefficients — a dynamic cointegrating
regression (Bierens & Martins 2010). Because the regression is in levels, the
residual is a level too, so *fair value is a price rather than a forecast*,
and if the relation is genuinely cointegrating the residual is stationary and
its mean reversion is testable online. That test is the gate the decision
layer uses to decide whether valuation is informative today.

- **Factor count** is read off the spectrum at the Marchenko–Pastur edge
  (Laloux et al. 1999), not chosen. Eigenvalues below the edge are noise;
  clipping them also yields the positive-definite cleaned correlation matrix
  the likelihood needs.
- **Correlation memory** is inferred. Three exponentially weighted estimators
  with different half-lives run in parallel and are scored by their own causal
  predictive likelihood.
- **Coefficient adaptation speed** is inferred by dynamic model averaging over
  discount factors (Raftery, Kárný & Ettler 2010).
- **Two valuation views** — latent factors and named asset-class blocks — are
  averaged by out-of-sample evidence. Twelve leave-one-class-out refits give
  ablation-based driver importance and a cross-sectional consistency score.
- **Attribution is exact.** Fair value is linear in the standardised returns of
  the instruments, so per-instrument contributions sum to it identically. No
  post-hoc explainer is involved.

### Engine 2 — Market Dynamics

No RSI, MACD, ATR, ADX, Stochastic, CCI or Williams %R. Each is a
fixed-window proxy for a latent quantity; the latent quantity is estimated
directly instead, with its own uncertainty and with the memory inferred.

| Latent quantity | Estimator |
|---|---|
| Level, slope | Local linear trend DLM (Harvey 1989), discount bank |
| Conditional volatility | Dynamic model averaging over exponential memories |
| Roughness / memory | Multiscale variance ratio → generalised Hurst, with standard error |
| Mean-reversion potential | Online Ornstein–Uhlenbeck fit to trend deviation |
| Market efficiency | Permutation entropy (Bandt & Pompe 2002) |
| Complexity | Lempel–Ziv complexity of the binarised path |
| Dominant cycle | Burg maximum-entropy spectrum, AIC order selection |
| Structural stability | Exponentially weighted CUSUM of recursive residuals |
| Regime | Forward-filtered online HMM, model-averaged over 2–4 states |

The oscillator is composed by the market's own inferred memory structure: the
Hurst exponent decides how much weight trend persistence and mean reversion
each carry, and because it carries a standard error the weight is graded
rather than a switch.

### Engine 3 — Decision Fusion

A Bayesian decision problem, not an average of the two oscillators. Expected
return, expected **upside** and expected **downside** are estimated separately
— the asymmetry is the decision-relevant part — with the decision horizon
itself model-averaged over 5, 21 and 63 sessions.

The design matrix carries both oscillators, their interactions with market
stress and with their own confidences, and their product, so *reinforcement*
and *conflict* are represented without being coded by hand. The reported
valuation weight is

```
∂E[return]/∂FVO = b₁ + b₃·stress + b₅·val_conf + b₉·DO
```

as a share of total sensitivity — read off coefficients fitted forward in
time, never chosen.

**The PDO** is the confidence-shrunk expected annualised information ratio of
acting on the current state:

```
PDO = κ · (μ̂ / ŝ) · √(252 / h)
```

`κ` is not a prior belief. It is the demonstrated association between this
model's own past predictions and their realised outcomes, expressed as
P(ρ > 0) under the Fisher-z law and mapped to zero when there is no evidence.
**When nothing has been demonstrated the oscillator is exactly zero** — flat
stretches in the PDO are abstentions, not neutral opinions.

---

## Causality, non-repainting, revision invariance

These are the load-bearing claims, so the dashboard **executes** them rather
than asserting them (Integrity tab), and they hold to the last bit:

```
Determinism         PASS   max |difference| = 0.0
Revision invariance PASS   max |revision|   = 0.0   (records truncated to 40%, 65%, 90%)
```

What makes that true by construction:

- **Every estimator is a forward recursion.** DLM/Kalman filters,
  exponentially weighted moments, recursive least squares, expanding
  empirical CDFs, online EM. No smoother, no centred window, no bidirectional
  filter, no full-sample normalisation anywhere.
- **The HMM is filtered, never smoothed.** Baum–Welch runs a backward pass, so
  the regime it assigns to a date is computed from data that came after it.
  Only the forward filter is used here — which is why this regime history looks
  less decisive than most published ones, and why it is the honest one.
- **Instrument admission is causal, and the panel's width is not a function of
  when you ran it.** An instrument joins the cross-section on the day its own
  accumulated print count first reaches the estimability floor. Columns are
  never dropped — not even ones that never print — because dropping them would
  make the cross-sectional dimension depend on the record's endpoint, and the
  Marchenko–Pastur edge is (1 + √(N/T))², so N moving would change the factor
  structure of 2012 when you extend the record to 2026. *This was a real bug,
  caught by the revision test on live data and not by reading the code*: the
  factor count screen was already causal, but truncating the record to 2017
  still removed every instrument launched after it. The fix was to decompose
  only the admitted sub-cross-section and let dead columns cost nothing.
- **Labels lag by their horizon.** The fusion model speaking at *t* was trained
  on outcomes observable through *t − h*, and is scored against the prediction
  it actually emitted, not a re-prediction from today's parameters.
- **The price store is append-only.** Total-return adjustment at the vendor
  rewrites history on every dividend; a model can be perfectly non-repainting
  and still produce different history if its inputs are rewritten underneath
  it. Once a (ticker, date) close is written it is never overwritten, only
  extended — in both directions.
- **Nothing is random.** No seeds, no stochastic initialisation, no iteration
  over unordered containers. Eigenvector signs follow a fixed convention and
  are matched to the previous basis so a factor cannot silently change
  identity.

Because the record is built by a single forward pass, **the online record *is*
the walk-forward record**. There is no separate backtest to run and nothing to
re-fit.

---

## Two negative results

Reported because they shaped the design.

**Supervised prediction of the next session's return, from the full latent
technical state, produced a forgetting-weighted R² of about −0.01 on every
asset examined.** That is the correct empirical answer for daily data. It is
why the Dynamics Oscillator is a state description whose association with
forward returns is *measured* online, rather than a fitted forecast.

**Scoring discount factors by one-step predictive likelihood is degenerate for
a level regression.** The model that tracks price most closely always wins,
and its limit is the useless statement "fair value = price". Admitting a
five-month coefficient memory collapsed SPY's mispricing to 0.8% standard
deviation with a 3-day half-life — a residual, not a valuation. The valuation
family is therefore restricted to coefficient memories of four years and
longer, within which the data still selects; shifting the grid a notch slower
moves the reported mispricing by well under a percentage point.

---

## Layout

```
app.py                 Streamlit dashboard — the asset is the only input
amis/
  causal.py            Forward-recursion primitives: discounted DLM bank,
                       dynamic model averaging, online logistic, expanding CDF
  data.py              Append-only price store, causal panel construction
  universe.py          227 explanatory instruments across 12 asset classes
  factors.py           Online PCA, Marchenko–Pastur clipping, adaptive volatility
  regime.py            Forward-filtered online HMM, averaged over state counts
  valuation.py         Engine 1 — Market Valuation
  dynamics.py          Engine 2 — Market Dynamics
  fusion.py            Engine 3 — Decision Fusion
  pipeline.py          Orchestration; a pure function of (asset, prices, version)
  validation.py        Determinism, revision invariance, walk-forward scoring
  viz.py               Plotly layer (validated colourblind-safe palette)
tests/test_amis.py     23 checks on synthetic data
```

```bash
python -m pytest tests -q          # offline, ~4 minutes
```

The suite covers the two claims above plus a stronger one — **corrupting the
tail of the record must leave every value in the head bit-identical** — and
pins the primitives: the DLM recovers known coefficients, the
Marchenko–Pastur clip finds no factor in pure noise and does find a planted
one, permutation entropy is ~1 on i.i.d. data and ~0 on a monotone path, Burg
recovers a planted 40-session cycle, and the factor attributions reconstruct
the factor levels to 1e-8.

Every internal constant is either a machine-precision guard, a statistical
necessity (a second moment needs ~250 observations), the support of a prior
that the data then selects within (discount grids, memory grids, horizon
banks), or a compute budget applied one-sidedly so it cannot repaint. None is
tuned to an asset, and none is exposed as a control — a knob that lets the
analyst tune a signal until it looks right is a knob that manufactures the
result.

---

## Reading the output

| Reading | Meaning |
|---|---|
| **FVO** ≈ 0 | trading at the level the global cross-section implies |
| **FVO** > 0 | expensive relative to that level, in units of the predictive standard deviation |
| **DO** > 0 | net bullish pressure from the latent technical state |
| **PDO** = 0 | *no demonstrated edge* — the system declines to have a view |
| **PDO** = +1.8 | acting on this state has the expected quality of a 1.8 information-ratio opportunity, after shrinking for demonstrated reliability |

AMIS estimates the statistical quality of an opportunity. It does not issue
investment advice, and position sizing depends on mandate, risk budget and
capital constraints that it has no knowledge of.

---

## References

Bai & Ng (2002) *Econometrica* · Bandt & Pompe (2002) *PRL* · Berger (1985)
*Statistical Decision Theory and Bayesian Analysis* · Bierens & Martins (2010)
*Econometric Theory* · Brown, Durbin & Evans (1975) *JRSS-B* · Burg (1975)
*Maximum Entropy Spectral Analysis* · Cappé (2011) *JCGS* · Cesa-Bianchi &
Lugosi (2006) *Prediction, Learning, and Games* · Engle & Granger (1987)
*Econometrica* · Hamilton (1989) *Econometrica* · Harvey (1989) *Forecasting,
Structural Time Series Models and the Kalman Filter* · Kelly (1956) *BSTJ* ·
Koop & Korobilis (2012) *IER* · Laloux et al. (1999) *PRL* · Lempel & Ziv
(1976) *IEEE TIT* · Lo & MacKinlay (1988) *RFS* · López de Prado (2018)
*Advances in Financial Machine Learning* · Marchenko & Pastur (1967)
*Sbornik* · Newey & West (1987) *Econometrica* · Opper (1998) *Online Learning
in Neural Networks* · Plerou et al. (2002) *PRE* · Raftery, Kárný & Ettler
(2010) *Technometrics* · Ross (1976) *JET* · Timmermann (2006) *Handbook of
Economic Forecasting* · West & Harrison (1997) *Bayesian Forecasting and
Dynamic Models* · White (2000) *Econometrica* · Zunino et al. (2009)
*Physica A*
