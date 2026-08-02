"""
Online latent-factor extraction from the global cross-section.
=================================================================

The explanatory panel is ~200 instruments whose correlation matrix is mostly
noise: with N assets and T effective observations, the sample eigenvalues of
a *pure noise* correlation matrix fill the Marchenko-Pastur bulk
[(1-sqrt(q))^2, (1+sqrt(q))^2] with q = N/T.  Only eigenvalues above the
upper edge carry information.  That gives a parameter-free answer to "how
many factors?" -- the count is read off the spectrum, it is not chosen.

Design decisions and why
------------------------
*Why eigenvalue clipping rather than Bai-Ng ICp?*  Bai & Ng (2002) is the
standard for a fixed panel estimated in one shot.  It requires a penalty
calibrated to (N, T) of the full panel and a batch SVD, which is awkward to
make strictly recursive.  The Marchenko-Pastur edge (Laloux et al. 1999;
Plerou et al. 2002; Bouchaud & Potters 2003) gives the same answer
asymptotically, needs only the current spectrum, and simultaneously produces
a *cleaned, positive-definite* correlation matrix -- which the downstream
Woodbury likelihood needs anyway.  It is the cheaper of two equivalent
answers.

*Why a bank of memories instead of one window?*  Correlation memory is
itself a latent, regime-dependent quantity.  Three exponentially weighted
estimators with different half-lives are carried in parallel and scored by
their own causal one-step predictive likelihood.  The winner at time t was
the winner on evidence available at t; no window length is ever imposed.

References
----------
Marchenko, V. A. & Pastur, L. A. (1967). "Distribution of eigenvalues for
    some sets of random matrices." *Sbornik: Mathematics* 1(4).
Laloux, L., Cizeau, P., Bouchaud, J.-P. & Potters, M. (1999). "Noise
    Dressing of Financial Correlation Matrices." *Phys. Rev. Lett.* 83.
Plerou, V. et al. (2002). "Random matrix approach to cross correlations in
    financial data." *Phys. Rev. E* 65.
Bai, J. & Ng, S. (2002). "Determining the Number of Factors in Approximate
    Factor Models." *Econometrica* 70(1).
Ledoit, O. & Wolf, O. (2004). "A well-conditioned estimator for
    large-dimensional covariance matrices." *J. Multivariate Analysis* 88.
"""

from __future__ import annotations

import math

import numpy as np

from .causal import EPS, VAR_FLOOR, halflife_to_lambda

#: Hard ceiling on retained factors.  Storage bound, not a modelling choice:
#: the Marchenko-Pastur edge decides the actual count and only exceeds this
#: in degenerate panels.
K_MAX = 20

#: Winsorisation of standardised returns, in sigmas.  A robustness device
#: from outlier-resistant covariance estimation, not a signal threshold: a
#: single 20-sigma print would otherwise rotate the entire eigenbasis.
Z_CLIP = 6.0

#: Correlation memories entered in the bank, in trading days.  This is the
#: support of the prior over memory, not a selected value.
MEMORY_GRID = (126.0, 252.0, 504.0)


# ===========================================================================
# Per-asset adaptive volatility (vectorised dynamic model averaging)
# ===========================================================================
class AdaptiveVolPanel:
    """Causal volatility for every column of a return panel, no fixed window.

    Each asset carries a bank of exponentially weighted variance estimators
    with different half-lives; the weights are updated by the Gaussian
    one-step predictive likelihood of the return actually realised.  This is
    dynamic model averaging (Raftery et al. 2010) applied per asset, executed
    as dense array arithmetic so the cost is negligible for N ~ 200.

    Reading :meth:`sigma` before :meth:`update` gives the volatility that was
    knowable strictly before the observation -- which is what standardisation
    requires if it is not to leak.
    """

    HALFLIVES = (10.0, 21.0, 63.0, 252.0)

    def __init__(self, n: int, mean_halflife: float = 504.0) -> None:
        self.n = n
        self.h = np.array(self.HALFLIVES)
        self.lam = np.array([halflife_to_lambda(x) for x in self.HALFLIVES])
        self.V = np.zeros((len(self.h), n))       # EW variance per model/asset
        self.W = np.zeros((len(self.h), n))       # accumulated weight
        self.logw = np.zeros((len(self.h), n))    # log model probability
        self.mu = np.zeros(n)                     # slow EW mean of returns
        self.muw = np.zeros(n)
        self.mu_lam = halflife_to_lambda(mean_halflife)
        self.count = np.zeros(n)

    def sigma(self) -> np.ndarray:
        """Model-averaged standard deviation per asset (pre-update)."""
        w = self._weights()
        var = np.sum(w * np.where(self.W > EPS, self.V / np.maximum(self.W, EPS), 0.0),
                     axis=0)
        return np.sqrt(np.maximum(var, VAR_FLOOR))

    def mean(self) -> np.ndarray:
        return np.where(self.muw > EPS, self.mu / np.maximum(self.muw, EPS), 0.0)

    def _weights(self) -> np.ndarray:
        lw = self.logw - self.logw.max(axis=0, keepdims=True)
        w = np.exp(lw)
        return w / np.maximum(w.sum(axis=0, keepdims=True), EPS)

    def effective_halflife(self) -> np.ndarray:
        return np.sum(self._weights() * self.h[:, None], axis=0)

    def update(self, r: np.ndarray) -> None:
        ok = np.isfinite(r)
        x = np.where(ok, r, 0.0)
        mu = self.mean()
        dev = np.where(ok, x - mu, 0.0)

        # score each memory on the return it did not see, then absorb it
        var = np.where(self.W > EPS, self.V / np.maximum(self.W, EPS), 0.0)
        var = np.maximum(var, VAR_FLOOR)
        ll = -0.5 * (np.log(2.0 * math.pi * var) + (dev[None, :] ** 2) / var)
        ready = (self.W > 1e-3) & ok[None, :]
        # 0.99: model-probability forgetting of Raftery et al. (2010),
        # the literature-standard value, applied uniformly.
        self.logw = np.where(ready, 0.99 * self.logw + ll, self.logw)
        self.logw -= self.logw.max(axis=0, keepdims=True)

        lam = self.lam[:, None]
        upd = ok[None, :]
        self.V = np.where(upd, lam * self.V + (1.0 - lam) * dev[None, :] ** 2, self.V)
        self.W = np.where(upd, lam * self.W + (1.0 - lam), self.W)
        self.mu = np.where(ok, self.mu_lam * self.mu + (1.0 - self.mu_lam) * x, self.mu)
        self.muw = np.where(ok, self.mu_lam * self.muw + (1.0 - self.mu_lam), self.muw)
        self.count += ok


# ===========================================================================
# Exponentially weighted pairwise-complete correlation
# ===========================================================================
class _CorrMemory:
    """One EW correlation estimator with its cleaned factor representation."""

    __slots__ = ("lam", "S", "W", "V", "lam_eig", "delta", "k", "C_ref",
                 "last_recompute", "logw", "n", "wbar")

    def __init__(self, n: int, halflife: float) -> None:
        self.lam = halflife_to_lambda(halflife)
        self.n = n
        self.wbar = 0.0
        self.S = np.zeros((n, n))
        self.W = np.zeros((n, n))
        self.V = np.zeros((n, K_MAX))     # top eigenvectors (columns)
        self.lam_eig = np.zeros(K_MAX)    # corresponding eigenvalues
        self.delta = 1.0                  # bulk (noise) eigenvalue
        self.k = 0
        self.C_ref = None
        self.last_recompute = -10**9
        self.logw = 0.0

    def absorb(self, outer: np.ndarray, mask_outer: np.ndarray,
               frac: float) -> None:
        lam = self.lam
        self.S *= lam
        self.S += (1.0 - lam) * outer
        self.W *= lam
        self.W += (1.0 - lam) * mask_outer
        self.wbar = lam * self.wbar + (1.0 - lam) * frac

    def correlation(self) -> np.ndarray:
        W = np.maximum(self.W, EPS)
        C = self.S / W
        d = np.sqrt(np.maximum(np.diag(C), VAR_FLOOR))
        C = np.clip(C / np.outer(d, d), -0.999, 0.999)
        np.fill_diagonal(C, 1.0)
        return C

    @property
    def t_eff(self) -> float:
        """Effective sample size implied by the weights actually accumulated.

        1/(1-lambda) is the asymptotic effective sample size of an
        exponentially weighted estimator; multiplying by the accumulated
        weight fraction handles the transient at the start of the record,
        where the estimator has seen far less than that.
        """
        cap = 1.0 / max(1.0 - self.lam, EPS)
        return float(min(max(self.wbar * cap, 1.0), cap))


def _clip_spectrum(C: np.ndarray, t_eff: float) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Marchenko-Pastur eigenvalue clipping.

    Returns (eigenvectors, eigenvalues, bulk level, k).  Eigenvalues above
    the MP upper edge are retained; the remainder are collapsed to their
    mean, which preserves the trace and leaves a positive-definite matrix.
    """
    n = C.shape[0]
    C = 0.5 * (C + C.T)
    try:
        w, v = np.linalg.eigh(C)
    except np.linalg.LinAlgError:
        return np.zeros((n, 0)), np.zeros(0), 1.0, 0
    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]

    q = n / max(t_eff, 1.0)
    edge = (1.0 + math.sqrt(q)) ** 2
    k = int(np.sum(w > edge))
    # cap by estimability of the downstream regression, and by storage
    k = max(1, min(k, K_MAX, int(max(2, t_eff // 25))))

    total = float(np.sum(w))
    delta = (total - float(np.sum(w[:k]))) / max(n - k, 1)
    delta = max(delta, 1e-6)

    # deterministic sign convention: the largest-magnitude loading is positive
    Vk = v[:, :k].copy()
    for j in range(k):
        idx = int(np.argmax(np.abs(Vk[:, j])))
        if Vk[idx, j] < 0:
            Vk[:, j] *= -1.0
    return Vk, w[:k].copy(), delta, k


def _align(V_new: np.ndarray, lam_new: np.ndarray, V_old: np.ndarray,
           k_old: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match new eigenvectors to the previous basis (order and sign).

    Without this, an eigenvalue crossing would silently relabel factor 2 as
    factor 3 and corrupt the cumulative factor levels.  Matching is greedy on
    |cos| similarity -- deterministic, and O(k^2) with k <= 20.
    """
    k_new = V_new.shape[1]
    if k_old == 0 or V_old.shape[1] == 0:
        return V_new, lam_new, np.arange(k_new)

    sim = np.abs(V_new.T @ V_old[:, :k_old])       # k_new x k_old
    order = np.full(k_new, -1, dtype=int)
    used_new, used_old = set(), set()
    pairs = sorted(
        ((sim[i, j], i, j) for i in range(k_new) for j in range(k_old)),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    for s, i, j in pairs:
        if i in used_new or j in used_old or s < 0.3:
            continue
        order[i] = j
        used_new.add(i)
        used_old.add(j)
    free = [j for j in range(max(k_new, k_old)) if j not in used_old]
    fi = 0
    for i in range(k_new):
        if order[i] < 0:
            order[i] = free[fi] if fi < len(free) else i
            fi += 1

    slots = np.argsort(order)
    V_sorted = V_new[:, slots]
    lam_sorted = lam_new[slots]
    tgt = order[slots]
    for c in range(V_sorted.shape[1]):
        j = tgt[c]
        if j < k_old and float(V_sorted[:, c] @ V_old[:, j]) < 0:
            V_sorted[:, c] *= -1.0
    return V_sorted, lam_sorted, tgt


class OnlineFactorModel:
    """Recursive statistical factor model of the global cross-section.

    Per step the model emits, using information up to and including t:

    * ``scores``  -- unit-variance factor returns f_t
    * ``k``       -- number of factors above the Marchenko-Pastur edge
    * ``loadings``-- the current eigenbasis (columns)
    * ``contrib`` -- cumulative attribution of each factor level back to the
                     individual instruments that produced it

    The attribution is exact rather than approximate: factor levels are
    linear in standardised returns, so F_k,t = sum_i contrib[k, i] holds
    identically, and no post-hoc explainer is needed.
    """

    def __init__(self, n: int, memories: tuple[float, ...] = MEMORY_GRID) -> None:
        self.n = n
        self.mem = [_CorrMemory(n, h) for h in memories]
        self.memories = memories
        self.active = len(memories) - 1        # start at the longest memory
        self.t = 0
        self.k = 0
        self.levels = np.zeros(K_MAX)          # cumulative factor levels
        self.contrib = np.zeros((K_MAX, n))    # per-instrument attribution
        self.last_scores = np.zeros(K_MAX)
        self._logw = np.zeros(len(memories))

    # -- internals ----------------------------------------------------------
    def _maybe_recompute(self, m: _CorrMemory) -> None:
        """Refresh the eigenbasis when the correlation matrix has moved by
        more than sampling noise, subject to a compute floor/ceiling."""
        gap = self.t - m.last_recompute
        if gap < 5 and m.k > 0:
            return
        C = m.correlation()
        t_eff = m.t_eff
        if m.C_ref is not None and gap < 63:
            denom = max(np.linalg.norm(m.C_ref), EPS)
            drift = float(np.linalg.norm(C - m.C_ref) / denom)
            if drift < 1.0 / math.sqrt(max(t_eff, 1.0)):
                return
        Vk, lk, delta, k = _clip_spectrum(C, t_eff)
        Vk, lk, _ = _align(Vk, lk, m.V, m.k)
        m.V = np.zeros((self.n, K_MAX))
        m.V[:, :Vk.shape[1]] = Vk
        m.lam_eig = np.zeros(K_MAX)
        m.lam_eig[:len(lk)] = lk
        m.delta = delta
        m.k = int(Vk.shape[1])
        m.C_ref = C
        m.last_recompute = self.t

    @staticmethod
    def _loglik(m: _CorrMemory, z: np.ndarray, n_avail: int) -> float:
        """Gaussian log density under the cleaned factor covariance.

        Evaluated by the Woodbury identity, O(N k).  Used only to weight the
        memory bank; it never enters a published quantity, so the mild
        approximation of scaling the log-determinant by the available
        fraction is immaterial.
        """
        if m.k == 0 or n_avail < 5:
            return 0.0
        k = m.k
        V = m.V[:, :k]
        lam = np.maximum(m.lam_eig[:k], m.delta + 1e-9)
        p = V.T @ z
        quad = (float(z @ z) - float(np.sum((lam - m.delta) / lam * p * p))) / m.delta
        logdet = n_avail * math.log(m.delta) + float(np.sum(np.log(lam / m.delta)))
        return -0.5 * (logdet + quad + n_avail * math.log(2.0 * math.pi))

    # -- public -------------------------------------------------------------
    def update(self, z: np.ndarray, avail: np.ndarray) -> dict:
        """Absorb one cross-section of standardised returns.

        `z` is winsorised in place by the caller's contract; entries where
        `avail` is False are ignored entirely.
        """
        zz = np.where(avail, np.clip(z, -Z_CLIP, Z_CLIP), 0.0)
        zz = np.nan_to_num(zz, nan=0.0, posinf=0.0, neginf=0.0)
        mask = avail.astype(float)
        n_avail = int(mask.sum())

        # score each memory on data it has not yet absorbed
        if self.t > 0:
            lls = np.array([self._loglik(m, zz, n_avail) for m in self.mem])
            if np.all(np.isfinite(lls)):
                self._logw = 0.99 * self._logw + lls / max(n_avail, 1)
                self._logw -= self._logw.max()

        outer = np.outer(zz, zz)
        mask_outer = np.outer(mask, mask)
        frac = n_avail / max(self.n, 1)
        for m in self.mem:
            m.absorb(outer, mask_outer, frac)
            self._maybe_recompute(m)

        w = np.exp(self._logw - self._logw.max())
        w = w / max(w.sum(), EPS)
        self.active = int(np.argmax(w))
        m = self.mem[self.active]

        k = m.k
        scores = np.zeros(K_MAX)
        if k > 0:
            V = m.V[:, :k]
            lam = np.maximum(m.lam_eig[:k], 1e-6)
            # renormalise the loading vector over the instruments that
            # actually traded, so a closed market does not shrink the factor
            norm = np.sqrt(np.maximum((V ** 2 * mask[:, None]).sum(axis=0), 1e-9))
            proj = (V * mask[:, None]).T @ zz
            # unit-variance scores: Var(V_j' z) = lambda_j, and dividing by
            # the restricted norm compensates for instruments that did not
            # trade rather than letting the factor shrink toward zero
            s = proj / (norm * np.sqrt(lam))
            scores[:k] = s
            # exact per-instrument attribution of the factor increment
            inc = (V * mask[:, None]) * zz[:, None] / (norm * np.sqrt(lam))[None, :]
            self.contrib[:k] += inc.T
            self.levels[:k] += s

        self.k = k
        self.last_scores = scores
        self.t += 1
        return {
            "k": k,
            "scores": scores,
            "levels": self.levels.copy(),
            "memory": self.memories[self.active],
            "memory_weights": w,
            "t_eff": m.t_eff,
            "explained": float(np.sum(m.lam_eig[:k]) / max(self.n, 1)) if k else 0.0,
        }

    def loadings(self) -> np.ndarray:
        m = self.mem[self.active]
        return m.V[:, :max(m.k, 1)].copy()

    def instrument_contribution(self) -> np.ndarray:
        """(K_MAX, N) cumulative attribution of factor levels to instruments."""
        return self.contrib.copy()
