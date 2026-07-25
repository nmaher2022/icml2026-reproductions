# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
"""Core library for the numerical audit of arXiv:2606.08681 (ICML 2026 #16514).

Everything is an independent re-implementation from the paper's formulas:
  - g(u): the Gaussian mechanism's optimal delta at privacy eps (Eq. 6)
  - u0(delta): minimal Gaussian variance achieving (eps, delta)-DP (Eq. 7)
  - GGamma(alpha, beta, p) radial law (Def 4.1): density, moments, sampler
  - SGG(alpha, beta, p) in R^T (Def 4.2): X = R*U
  - delta_sgg: optimal delta for the worst-case shift ||mu||=s of an SGG
    mechanism, via the privacy-loss representation over (R, W):
       Z(r, w) = ((alpha+1-T)/2) * ln(r^2 / (r^2 - 2 s r w + s^2))
                 + beta * ((r^2 - 2 s r w + s^2)^(p/2) - r^p),
    with R ~ GGamma, W = cos(angle) ~ 2*Beta((T-1)/2,(T-1)/2)-1,
    delta(eps) = E[(1 - exp(eps - Z))_+]      (hockey-stick divergence)
  - beta calibration to a target (eps, delta) at fixed (alpha, p)  [Alg 4 spirit]
  - PLD/PRV discretization + FFT k-fold composition               [Alg 7 spirit]

Cross-checks (asserted in the claim scripts, not here):
  Gaussian special case (p, alpha, beta) = (2, T-1, 1/(2 sigma^2)) must satisfy
  delta_sgg == g(sigma^2) for every T; composed Gaussian must match the
  analytic k-fold closed form.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, special, stats

# ---------------------------------------------------------------------------
# Gaussian mechanism: optimal delta g(u) (Eq. 6) and its inverse u0(delta)
# ---------------------------------------------------------------------------

def gauss_g(u, eps, s=1.0):
    """Optimal delta of the Gaussian mechanism with per-coordinate variance u."""
    u = np.asarray(u, dtype=float)
    sq = np.sqrt(u)
    a = -eps * sq / s + s / (2 * sq)
    b = -eps * sq / s - s / (2 * sq)
    return stats.norm.cdf(a) - np.exp(eps) * stats.norm.cdf(b)


def gauss_u0(delta, eps, s=1.0):
    """Minimal variance u with g(u) <= delta (g is decreasing for relevant u)."""
    f = lambda lu: gauss_g(np.exp(lu), eps, s) - delta
    lo, hi = -60.0, 60.0
    return float(np.exp(optimize.brentq(f, lo, hi, xtol=1e-14)))


# ---------------------------------------------------------------------------
# GGamma radial law (Def 4.1):  f_R(r) = p b^((a+1)/p)/Gamma((a+1)/p) r^a e^{-b r^p}
# ---------------------------------------------------------------------------

def ggamma_logpdf(r, alpha, beta, p):
    k = (alpha + 1) / p
    return (np.log(p) + k * np.log(beta) - special.gammaln(k)
            + alpha * np.log(r) - beta * r ** p)


def ggamma_moment(alpha, beta, p, m):
    """E[R^m] in closed form."""
    k = (alpha + 1) / p
    return float(np.exp(special.gammaln((alpha + 1 + m) / p) - special.gammaln(k))
                 * beta ** (-m / p))


def ggamma_sample(alpha, beta, p, size, rng):
    """R = (Y / beta)^(1/p) with Y ~ Gamma((alpha+1)/p, 1)."""
    y = rng.gamma((alpha + 1) / p, 1.0, size)
    return (y / beta) ** (1.0 / p)


def ggamma_ppf(q, alpha, beta, p):
    y = special.gammaincinv((alpha + 1) / p, q)
    return (y / beta) ** (1.0 / p)


def sgg_mse(alpha, beta, p):
    """Mechanism MSE = E[||X||^2] = E[R^2]."""
    return ggamma_moment(alpha, beta, p, 2)


# ---------------------------------------------------------------------------
# Direction cosine W in dimension T:  (W+1)/2 ~ Beta((T-1)/2, (T-1)/2)
# ---------------------------------------------------------------------------

def w_nodes(T, n):
    """Gauss-Jacobi-style nodes/weights for E_W[h(W)] with the exact W law."""
    # integrate with the Beta((T-1)/2,(T-1)/2) transform: W = 2B-1
    xs, ws = np.polynomial.legendre.leggauss(n)
    b = (xs + 1) / 2                      # (0,1)
    dens = stats.beta.pdf(b, (T - 1) / 2, (T - 1) / 2) / 2  # W-density at 2b-1
    return 2 * b - 1, ws * dens           # sum(w_i h(W_i)) approximates E[h(W)]


# ---------------------------------------------------------------------------
# SGG privacy: delta(eps) for the worst-case pair (X, X + mu), ||mu|| = s
# ---------------------------------------------------------------------------

def privacy_loss_Z(r, w, alpha, beta, p, T, s):
    """log f_X(x) - log f_X(x - mu) at a point with ||x||=r, cos(angle(x, mu))=w."""
    r2 = r * r
    q2 = r2 - 2 * s * r * w + s * s          # ||x - mu||^2
    q2 = np.maximum(q2, 1e-300)
    return ((alpha + 1 - T) / 2) * (np.log(r2) - np.log(q2)) \
        + beta * (q2 ** (p / 2) - r ** p)


def delta_sgg(eps, alpha, beta, p, T, s=1.0, n_r=400, n_w=64, q_lo=1e-12, q_hi=1 - 1e-13):
    """delta(eps) = E_{R,W}[(1 - exp(eps - Z(R,W)))_+] by kink-split quadrature.

    R-integral by quantile transform (equal-probability nodes).  For each r the
    integrand vanishes below the kink w*(r) where Z(r, w*) = eps (Z is
    increasing in w), so the W-integral runs over [w*(r), 1] with Legendre
    nodes mapped per-r against the exact Beta density of W — restoring
    smooth-integrand accuracy despite the (.)_+ hinge.
    """
    qs = (np.arange(n_r) + 0.5) / n_r * (q_hi - q_lo) + q_lo
    rs = ggamma_ppf(qs, alpha, beta, p)          # equal-probability R nodes
    # Z(r, w) is DECREASING in w for alpha <= T-1 (the distinguishing region is
    # anti-aligned with the shift), so the integrand is supported on [-1, w*(r)]
    # with Z(r, w*) = eps.  Vectorized bisection for w*.
    lo = np.full_like(rs, -1.0)
    hi = np.full_like(rs, 1.0)
    z_lo = privacy_loss_Z(rs, lo, alpha, beta, p, T, s)
    none_active = z_lo <= eps                     # integrand identically 0
    for _ in range(60):
        mid = (lo + hi) / 2
        zm = privacy_loss_Z(rs, mid, alpha, beta, p, T, s)
        take_lo = zm > eps                        # still above eps -> move lo up
        lo = np.where(take_lo, mid, lo)
        hi = np.where(take_lo, hi, mid)
    wstar = np.where(none_active, -1.0, (lo + hi) / 2)
    # Gauss-Jacobi nodes on [-1, wstar] per r: absorb the (1+w)^((T-3)/2)
    # endpoint factor of the W-density exactly (it is singular for T=2).
    aj = (T - 3) / 2
    xs, xw = special.roots_jacobi(n_w, 0.0, aj)   # weight (1+x)^aj on [-1,1]
    half = (wstar + 1.0) / 2
    wgrid = -1.0 + half[:, None] * (xs[None, :] + 1)             # (n_r, n_w)
    # W-density = C_T (1-w)^aj (1+w)^aj with C_T = 1/(B((T-1)/2,(T-1)/2) 2^(T-2))
    logC = -(special.betaln((T - 1) / 2, (T - 1) / 2) + (T - 2) * np.log(2))
    smooth = np.exp(logC) * np.maximum(1.0 - wgrid, 0.0) ** aj    # non-singular part
    Z = privacy_loss_Z(rs[:, None], wgrid, alpha, beta, p, T, s)
    integrand = np.clip(1.0 - np.exp(np.minimum(eps - Z, 50.0)), 0.0, None)
    # (1+w) = half*(1+x) -> the Jacobi weight contributes half^aj scale
    per_r = (integrand * smooth * xw[None, :]).sum(1) * half ** (aj + 1)
    return float(per_r.mean() * (q_hi - q_lo))    # E_R over the uniform quantiles


def delta_sgg_mc(eps, alpha, beta, p, T, s=1.0, n=2_000_000, seed=0):
    """Monte-Carlo estimate of the same quantity (ground-truth arbiter)."""
    rng = np.random.default_rng(seed)
    r = ggamma_sample(alpha, beta, p, n, rng)
    w = 2 * rng.beta((T - 1) / 2, (T - 1) / 2, n) - 1
    Z = privacy_loss_Z(r, w, alpha, beta, p, T, s)
    x = np.clip(1.0 - np.exp(np.minimum(eps - Z, 50.0)), 0.0, None)
    return float(x.mean()), float(x.std() / np.sqrt(n))


def calibrate_beta(eps, delta, alpha, p, T, s=1.0, **quad_kw):
    """Smallest noise (largest beta ... careful: delta decreases as noise grows,
    i.e. as beta decreases scale grows). Find beta with delta_sgg == delta."""
    f = lambda lb: delta_sgg(eps, alpha, np.exp(lb), p, T, s, **quad_kw) - delta
    lo, hi = -30.0, 30.0
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        raise ValueError(f"calibration bracket failed: f({lo})={flo:.3g}, f({hi})={fhi:.3g}")
    lb = optimize.brentq(f, lo, hi, xtol=1e-12, rtol=1e-12)
    return float(np.exp(lb))


# ---------------------------------------------------------------------------
# Generic spherical mechanism with arbitrary radial quantile function
# (for Claim 1: any additive spherical noise with a given MSE budget)
# ---------------------------------------------------------------------------

def delta_spherical(eps, r_of_q, logshape_of_r, T, s=1.0, n_r=400, n_w=200):
    """delta(eps) for spherical noise with radial quantile function r_of_q and
    log density-shape function  logshape(r) = log g(r^2)  (up to a constant).
    Z(r,w) = logshape(r) - logshape(sqrt(r^2-2srw+s^2))."""
    qs = (np.arange(n_r) + 0.5) / n_r
    rs = r_of_q(qs)
    wn, ww = w_nodes(T, n_w)
    q = np.sqrt(np.maximum(rs[:, None] ** 2 - 2 * s * rs[:, None] * wn[None, :] + s * s, 1e-300))
    Z = logshape_of_r(rs[:, None]) - logshape_of_r(q)
    integrand = np.clip(1.0 - np.exp(np.minimum(eps - Z, 50.0)), 0.0, None)
    return float((integrand @ ww).mean())


# ---------------------------------------------------------------------------
# PRV / PLD accounting (Prop 4.2 + Alg 7 spirit)
# ---------------------------------------------------------------------------

def prv_grid(alpha, beta, p, T, s=1.0, n_r=1500, n_w=800):
    """Weighted sample of the one-step PRV Z (values + probabilities)."""
    qs = (np.arange(n_r) + 0.5) / n_r
    rs = ggamma_ppf(qs, alpha, beta, p)
    wn, ww = w_nodes(T, n_w)
    Z = privacy_loss_Z(rs[:, None], wn[None, :], alpha, beta, p, T, s).ravel()
    W = np.tile(ww / n_r, (n_r, 1)).ravel()
    return Z, W / W.sum()


def compose_fft(Z, P, k, n_bins=200_000, tail=60.0):
    """k-fold i.i.d. composition of the PRV by histogram + FFT convolution.
    Returns (grid, pmf) of the composed PRV sum."""
    lo, hi = -tail, tail
    edges = np.linspace(lo, hi, n_bins + 1)
    h, _ = np.histogram(np.clip(Z, lo, hi - 1e-9), bins=edges, weights=P)
    # full linear k-fold convolution support is (n_bins-1)*k + 1 bins; the FFT
    # length must cover it or the tail wraps around (aliasing zeroed k>=8 in a
    # first draft of this audit)
    m = int(2 ** np.ceil(np.log2((n_bins - 1) * k + 1)))
    F = np.fft.rfft(h, m)
    comp = np.fft.irfft(F ** k, m)[: (n_bins - 1) * k + 1]
    comp = np.clip(comp, 0, None)
    centers = (edges[:-1] + edges[1:]) / 2
    grid = centers[0] * k + (centers[1] - centers[0]) * np.arange(len(comp))
    return grid, comp / comp.sum()


def delta_from_pld(grid, pmf, eps):
    x = np.clip(1.0 - np.exp(np.minimum(eps - grid, 50.0)), 0.0, None)
    return float((x * pmf).sum())
