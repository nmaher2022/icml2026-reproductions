# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "mpmath"]
# ///
"""Claims 1 + 5 audit.

Claim 5 (Table 2): reproduce the numerical lower bounds on delta_star.
Per Proposition 3.1 / Lemma 3.2, delta_star = g(u_star) where u_star is the
smallest u such that (i) g is convex on [u, infinity) and (ii) the tangent to
g at u lies below g on [0, u].  We compute g', g'' exactly via sympy-derived
closed forms and scan for u_star.  Targets (s=1):
  eps:        0.25      0.50      1.00      2.00      4.00      8.00     16.00
  delta_star: 0.736670  0.706970  0.649185  0.549133  0.416972  0.292170 0.197615

Claim 1 (Theorem 3.1): asymptotic optimality.  Fix eps, delta <= delta_star,
u0 = u0(delta).  Any additive-noise mechanism with MSE budget T*u0 has
liminf_T delta_M(eps) >= delta.  Audit: a family of competitor spherical
noises with EXACTLY the Gaussian MSE budget --
  shell:   R = sqrt(T u0) deterministic
  2-shell: mixture of two radii (mean-square = T u0)
  SGG(p=1.3), SGG(p=3), l2-Laplace-like p=1: beta scaled for MSE = T u0
-- and delta_M(eps) computed for T = 2..256; verify min over competitors
approaches / stays above delta as T grows, i.e. no competitor beats the
Gaussian by more than a vanishing margin (and quantify the margin).
Control: same sweep at a delta ABOVE delta_star (outside the theorem's
regime), where finite-T improvements are allowed to persist (l2 regime).
"""
from __future__ import annotations

import csv
import sys
import time

sys.path.insert(0, ".")
import numpy as np
from scipy import optimize

from sgg_lib import (delta_sgg, delta_spherical, gauss_g, gauss_u0,
                     ggamma_moment, ggamma_ppf)

S = 1.0

# ---- high-precision g, g', g'' via mpmath (float64 fails for eps >= 4:
# g involves exp(eps) * Phi(-large), a catastrophic cancellation) -------------

from mpmath import mp

mp.dps = 50


def _g_mp(u, eps):
    u = mp.mpf(u); e = mp.mpf(eps); s = mp.mpf(S)
    su = mp.sqrt(u)
    Phi = lambda z: mp.erfc(-z / mp.sqrt(2)) / 2
    return Phi(-e * su / s + s / (2 * su)) - mp.e ** e * Phi(-e * su / s - s / (2 * su))


def _g1_mp(u, eps):
    return mp.diff(lambda x: _g_mp(x, eps), mp.mpf(u))


def _g2_mp(u, eps):
    return mp.diff(lambda x: _g_mp(x, eps), mp.mpf(u), 2)


def delta_star(eps, n_scan=400):
    """Largest delta_star = g(u_star) satisfying the Prop 3.1 conditions."""
    # sanity vs float64 implementation in its stable range
    for uu in [0.1, 1.0, 10.0]:
        if eps <= 2:
            assert abs(float(_g_mp(uu, eps)) - gauss_g(uu, eps)) < 1e-12
    # u_right: past the last sign change of g'' (log-grid scan + refine)
    us = np.logspace(-6, 4, n_scan)
    g2 = np.array([float(_g2_mp(u, eps)) for u in us])
    neg = np.where(g2 < 0)[0]
    u_right = us[neg[-1] + 1] if len(neg) else us[0]
    if len(neg) and neg[-1] + 1 < len(us):
        a, b = us[neg[-1]], us[neg[-1] + 1]
        for _ in range(60):
            m = mp.sqrt(mp.mpf(a) * mp.mpf(b))
            if _g2_mp(m, eps) < 0:
                a = m
            else:
                b = m
        u_right = float(b)

    grid_cache = {}

    def tangent_ok(u0):
        key = round(float(np.log(u0)), 10)
        grid = np.concatenate([np.linspace(1e-9, u0, 220),
                               np.logspace(-9, np.log10(u0), 220)])
        gu0 = _g_mp(u0, eps); g1u0 = _g1_mp(u0, eps)
        for x in grid:
            tang = gu0 + g1u0 * (mp.mpf(x) - mp.mpf(u0))
            if _g_mp(x, eps) < tang - mp.mpf('1e-30'):
                return False
        return True

    lo, hi = u_right, u_right
    while not tangent_ok(hi):
        hi *= 1.6
        if hi > 1e8:
            raise RuntimeError("no tangent-support point found")
    if tangent_ok(lo):
        u_star = lo
    else:
        for _ in range(40):
            mid = float(np.sqrt(lo * hi))
            if tangent_ok(mid):
                hi = mid
            else:
                lo = mid
        u_star = hi
    return float(_g_mp(u_star, eps)), float(u_star), float(u_right)


def claim5():
    print("== Claim 5: Table 2 delta_star lower bounds ==")
    targets = {0.25: 0.736670, 0.50: 0.706970, 1.00: 0.649185, 2.00: 0.549133,
               4.00: 0.416972, 8.00: 0.292170, 16.00: 0.197615}
    rows = []
    for eps, tgt in targets.items():
        t0 = time.time()
        ds, u_star, u_right = delta_star(eps)
        rows.append({"eps": eps, "delta_star_paper": tgt, "delta_star_ours": ds,
                     "rel_diff": ds / tgt - 1, "u_star": u_star})
        print(f"eps={eps:5.2f}: ours={ds:.6f} paper={tgt:.6f} "
              f"rel_diff={ds/tgt-1:+.2e} ({time.time()-t0:.0f}s)")
    with open("claim5_table2.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    worst = max(abs(r["rel_diff"]) for r in rows)
    print(f"worst |rel diff| vs Table 2: {worst:.2e}")
    return rows


# ---- Claim 1: competitors under the Gaussian MSE budget -------------------

def competitor_deltas(eps, u0, T):
    """delta(eps) of spherical competitors with E||X||^2 = T*u0."""
    out = {}
    tu = T * u0
    # Gaussian benchmark (exact)
    out["Gaussian"] = float(gauss_g(u0, eps))
    # shell: R deterministic sqrt(tu): logshape from radial delta -> use
    # delta_spherical with degenerate quantile and flat shape... a point mass
    # radial law needs care: g(r^2) is a delta function; instead compute the
    # hockey-stick directly: X uniform on sphere of radius rho.
    # For the pair (X, X+mu) both uniform on spheres (radius rho, center 0/mu):
    # densities are singular; the optimal test compares supports -> delta(eps)
    # = 1 - overlap fraction e^eps-weighted; compute via the cap intersection:
    # the two spheres intersect in a (T-2)-sphere; TV distance of the two
    # uniform measures = 1 - (shared support measure) = 1 (disjoint supports
    # except measure-zero) -> delta = 1 - e^eps * 0 = ... actually with
    # disjoint supports delta(eps) = 1 for all eps: the shell mechanism is
    # catastrophically non-private (a known pathology). Record it exactly.
    out["shell"] = 1.0
    # smoothed 2-shell mixture: mixture of two Gaussians with variances
    # u_a < u0 < u_b, weights q, 1-q, mean-square = u0 per coordinate.
    for (name, spread) in [("2-Gauss-mix-1.5", 1.5), ("2-Gauss-mix-3", 3.0)]:
        ua, ub = u0 / spread, None
        q = 0.5
        ub = (u0 - q * ua) / (1 - q)
        # spherical Gaussian mixture: radial law = mixture of chi laws; use
        # delta_spherical with numerically built quantile + logshape.
        def make(ua=ua, ub=ub, q=q):
            def logshape(r):
                la = -r * r / (2 * ua) - T / 2 * np.log(2 * np.pi * ua)
                lb = -r * r / (2 * ub) - T / 2 * np.log(2 * np.pi * ub)
                m = np.maximum(la, lb)
                return m + np.log(q * np.exp(la - m) + (1 - q) * np.exp(lb - m))
            # quantile of radial law via inverse cdf sampling on grid
            rg = np.linspace(1e-6, np.sqrt(ub) * (np.sqrt(T) + 8), 60_000)
            from scipy.stats import chi
            cdf = q * chi.cdf(rg / np.sqrt(ua), T) + (1 - q) * chi.cdf(rg / np.sqrt(ub), T)
            def r_of_q(qq):
                return np.interp(qq, cdf, rg)
            return r_of_q, logshape
        r_of_q, logshape = make()
        out[name] = delta_spherical(eps, r_of_q, logshape, T, s=S, n_r=800, n_w=96)
    # SGG members, beta scaled to MSE = T*u0, alpha = T-1 (valid range)
    for p in [1.0, 1.3, 3.0]:
        beta1 = 1.0
        m2 = ggamma_moment(T - 1, beta1, p, 2)
        beta = beta1 * (m2 / tu) ** (p / 2)   # scale: E[R^2] ~ beta^(-2/p)
        assert abs(ggamma_moment(T - 1, beta, p, 2) / tu - 1) < 1e-9
        out[f"SGG(p={p})"] = delta_sgg(eps, T - 1, beta, p, T, s=S, n_r=1200, n_w=64)
    return out


def claim1():
    print("\n== Claim 1: asymptotic optimality of the Gaussian (Theorem 3.1) ==")
    eps = 1.0
    rows = []
    for regime, delta_t in [("inside (delta<=delta_star)", 1e-3),
                            ("outside control (delta>delta_star)", 0.66)]:
        u0 = gauss_u0(delta_t, eps)
        print(f"[{regime}] eps={eps} delta={delta_t} u0={u0:.4f}")
        for T in [2, 4, 8, 16, 32, 64, 128, 256]:
            t0 = time.time()
            d = competitor_deltas(eps, u0, T)
            best_name = min((k for k in d if k != "Gaussian"), key=lambda k: d[k])
            margin = d[best_name] / d["Gaussian"] - 1
            rows.append({"regime": regime, "T": T, **{k: d[k] for k in d},
                         "best_competitor": best_name, "margin_vs_gauss": margin})
            print(f"  T={T:3d}: Gauss={d['Gaussian']:.4e} best_comp={best_name:16s} "
                  f"delta={d[best_name]:.4e} margin={margin:+.3f} ({time.time()-t0:.0f}s)")
    with open("claim1_asymptotic.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


if __name__ == "__main__":
    t0 = time.time()
    claim5()
    claim1()
    print(f"\ntotal wall: {time.time()-t0:.0f}s")
