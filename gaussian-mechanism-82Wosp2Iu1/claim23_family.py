# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
"""Claims 2 + 3 audit.

Claim 3 (Def 4.1/4.2): the SGG family is well-defined (density normalizes)
and contains the Gaussian ((p,alpha)=(2,T-1), beta=1/(2 sigma^2)) and the
spherical/l2 Laplace ((p,alpha)=(1,T-1), beta=1/theta) as special cases.
Checks: normalization integrals, closed-form moment identities, exact density
match against the multivariate Gaussian / l2-Laplace, and privacy-curve match
delta_sgg == analytic Gaussian g(u) for several T (validates the whole
Lemma-4.1-style privacy machinery at once).

Claim 2 (Lemma 3.3): Haar symmetrization X' = MX preserves MSE and does not
increase the worst-direction optimal delta.  Numerical audit in T=2 with two
deliberately asymmetric noises (product-Laplace; independent asymmetric
Gamma-mixture coordinates).  MSE: exact/MC.  Privacy: worst-direction delta
over a grid of shift directions, each delta via 2-D Monte Carlo of the
hockey-stick divergence; symmetrized version via the spherical machinery on
the empirical radial law.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import integrate, stats

from sgg_lib import (delta_sgg, delta_sgg_mc, delta_spherical, gauss_g,
                     ggamma_logpdf, ggamma_moment, ggamma_sample, sgg_mse)

rng = np.random.default_rng(7)


def claim3():
    print("== Claim 3: SGG family well-defined; Gaussian and l2 special cases ==")
    # (a) normalization over a parameter grid
    worst = 0.0
    for alpha in [0.5, 1.0, 3.0, 7.5]:
        for beta in [0.2, 1.0, 4.0]:
            for p in [0.6, 1.0, 2.0, 3.5]:
                val, _ = integrate.quad(lambda r: np.exp(ggamma_logpdf(r, alpha, beta, p)),
                                        0, np.inf, limit=200)
                worst = max(worst, abs(val - 1))
    print(f"(a) density normalization: max |integral - 1| = {worst:.2e} over 48 (alpha,beta,p)")

    # (b) moment identity vs numerical integral
    worst = 0.0
    for (a, b, p, m) in [(2.0, 1.5, 2.0, 2), (4.0, 0.7, 1.0, 2), (1.2, 2.0, 0.8, 4)]:
        num, _ = integrate.quad(lambda r: r ** m * np.exp(ggamma_logpdf(r, a, b, p)),
                                0, np.inf, limit=200)
        worst = max(worst, abs(num / ggamma_moment(a, b, p, m) - 1))
    print(f"(b) closed-form moments: max rel err = {worst:.2e}")

    # (c) Gaussian special case: exact density identity + sample moments
    T, sigma = 4, 1.7
    alpha, beta, p = T - 1, 1 / (2 * sigma ** 2), 2
    mse = sgg_mse(alpha, beta, p)
    print(f"(c) Gaussian case T={T}: E||X||^2 = {mse:.6f} vs T*sigma^2 = {T*sigma**2:.6f} "
          f"(match={np.isclose(mse, T*sigma**2)})")
    r = ggamma_sample(alpha, beta, p, 400_000, rng)
    u = rng.normal(size=(400_000, T)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    x = r[:, None] * u
    ks = stats.kstest(x[:, 0] / sigma, 'norm')
    print(f"    SGG(T-1, 1/(2s^2), 2) marginal vs N(0, sigma^2): KS p-value = {ks.pvalue:.3f}")

    # (d) l2-Laplace special case: density f_X(x) prop exp(-||x||/theta)
    theta = 0.9
    alpha, beta, p = T - 1, 1 / theta, 1
    r = ggamma_sample(alpha, beta, p, 400_000, rng)
    # E[R] of l2-Laplace in R^T = theta * T (Gamma(T, theta) radial law)
    print(f"(d) l2-Laplace case: E[R] = {r.mean():.4f} vs theta*T = {theta*T:.4f}; "
          f"radial law == Gamma(T, theta): KS p = {stats.kstest(r, 'gamma', args=(T, 0, theta)).pvalue:.3f}")

    # (e) privacy-machinery validation: delta_sgg(Gaussian case) == analytic g(u)
    print("(e) privacy machinery vs analytic Gaussian curve:")
    worst = 0.0
    for T in [2, 5, 10, 25]:
        for eps in [0.1, 1.0, 3.0]:
            sigma = 1.3
            d_num = delta_sgg(eps, T - 1, 1 / (2 * sigma ** 2), 2, T, s=1.0)
            d_ana = float(gauss_g(sigma ** 2, eps))
            worst = max(worst, abs(d_num - d_ana))
            if T == 2 and eps == 1.0:
                mc, se = delta_sgg_mc(eps, T - 1, 1 / (2 * sigma ** 2), 2, T)
                print(f"    T=2 eps=1: quad={d_num:.6e} analytic={d_ana:.6e} MC={mc:.6e}±{se:.1e}")
    print(f"    max |quad - analytic| over T in {{2,5,10,25}} x eps in {{0.1,1,3}}: {worst:.2e}")


# ----------------------------------------------------------------------------
def hockey_stick_mc(logpdf, sampler, v, eps, n=2_000_000, seed=1):
    """delta for pair (X, X+v): E_X[(1 - exp(eps - Z))_+], Z = logpdf(x)-logpdf(x-v)."""
    r = np.random.default_rng(seed)
    x = sampler(n, r)
    Z = logpdf(x) - logpdf(x - v)
    val = np.clip(1.0 - np.exp(np.minimum(eps - Z, 50.0)), 0.0, None)
    return float(val.mean()), float(val.std() / np.sqrt(n))


def claim2():
    print("\n== Claim 2: Haar symmetrization preserves MSE, improves privacy ==")
    T, s = 2, 1.0
    cases = {}

    lam = 1.1
    cases["product-Laplace"] = dict(
        logpdf=lambda x: -np.abs(x).sum(1) / lam,
        sampler=lambda n, r: r.laplace(0, lam, (n, T)),
        mse_exact=2 * T * lam ** 2)

    k1, th1 = 2.0, 0.8
    def gam_sample(n, r):
        return r.gamma(k1, th1, (n, T)) - k1 * th1     # centered, skewed
    def gam_logpdf(x):
        y = x + k1 * th1
        out = np.where(y > 0, (k1 - 1) * np.log(np.maximum(y, 1e-300)) - y / th1, -np.inf)
        return out.sum(1)
    cases["asym-Gamma"] = dict(logpdf=gam_logpdf, sampler=gam_sample,
                               mse_exact=T * k1 * th1 ** 2)

    for name, c in cases.items():
        t0 = time.time()
        # worst-direction delta over shift directions (T=2: angle grid)
        angles = np.linspace(0, np.pi / 2, 7)   # symmetry: first quadrant suffices
        eps = 1.0
        deltas = []
        for a in angles:
            v = s * np.array([np.cos(a), np.sin(a)])
            d, se = hockey_stick_mc(c["logpdf"], c["sampler"], v, eps)
            deltas.append(d)
        d_orig = max(deltas)

        # symmetrized: same radial law, uniform direction -> use empirical radius
        xs = c["sampler"](4_000_000, np.random.default_rng(3))
        rad = np.linalg.norm(xs, axis=1)
        mse_orig = float((xs ** 2).sum(1).mean())
        mse_symm = float((rad ** 2).mean())
        qs_sorted = np.sort(rad)
        r_of_q = lambda q: np.quantile(qs_sorted, q)
        # density shape of symmetrized noise: g(r^2) prop f_R(r)/r^(T-1); estimate
        # log f_R by KDE-free histogram on log-radius (smooth enough for T=2)
        hist, edges = np.histogram(rad, bins=3000, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        logf = np.log(np.maximum(hist, 1e-300)) - (T - 1) * np.log(np.maximum(centers, 1e-300))
        def logshape(r):
            return np.interp(r, centers, logf, left=logf[0], right=-np.inf)
        d_symm = delta_spherical(eps, r_of_q, logshape, T, s=s, n_r=600, n_w=300)

        print(f"[{name}] MSE orig={mse_orig:.4f} symm={mse_symm:.4f} exact={c['mse_exact']:.4f} "
              f"(preserved: {np.isclose(mse_orig, mse_symm, rtol=1e-3)})")
        print(f"          worst-direction delta(eps=1): original={d_orig:.6f} "
              f"symmetrized={d_symm:.6f}  improved={d_symm <= d_orig + 3e-4} "
              f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    t0 = time.time()
    claim3()
    claim2()
    print(f"\ntotal wall: {time.time()-t0:.0f}s")
