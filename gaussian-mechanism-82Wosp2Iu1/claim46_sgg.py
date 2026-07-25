# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
"""Claims 4 + 6 audit.

Claim 4 (Figure 2): at eps = 0.1, s = 1, there exist (T, delta) settings where
an SGG mechanism has MSE up to ~15% below BOTH the Gaussian and the l2
mechanism, and the advantage shrinks with T.  Paper's own operating points:
    T=2 : delta=2.25e-3, SGG p*=1.37, MSE 364 vs Gauss/l2 431   (-15.5%)
    T=5 : delta=1.46e-3, SGG p*=1.32, -6.2%
    T=10: delta=1.18e-3, SGG p*=1.92, -2.5%/-2.8%
Audit: for each (T, delta), calibrate Gaussian (closed form), l2 (p=1,
alpha=T-1, bisect beta), and grid-search SGG over (alpha, p) in the valid
range alpha in (-1, T-1], p in (0.5, 4], calibrating beta at each point and
minimizing MSE.  Compare improvements to the paper's.

Claim 6 (Prop 4.2 + Alg 7): FFT-based composition accounting is tight.
Checks:
 (i)  k=1: FFT PLD delta == direct quadrature delta (self-consistency);
 (ii) Gaussian member: k-fold FFT == closed-form k-fold Gaussian
      (PRV of Gaussian is N(m, 2m), sum = N(km, 2km) == single Gaussian with
      m'=km, i.e. analytic g with s -> s*sqrt(k));
 (iii) l2 mechanism (p=1): k-fold FFT vs brute-force Monte-Carlo of the
      summed PRVs -- the 'tight composition of the l2 mechanism' that answers
      the open question of Joseph et al. (2025).
"""
from __future__ import annotations

import csv
import sys
import time

sys.path.insert(0, ".")
import numpy as np
from scipy import optimize

from sgg_lib import (calibrate_beta, compose_fft, delta_from_pld, delta_sgg,
                     gauss_g, gauss_u0, ggamma_sample, privacy_loss_Z,
                     prv_grid, sgg_mse)

S = 1.0
EPS = 0.1


def calibrated_mse(eps, delta, alpha, p, T):
    beta = calibrate_beta(eps, delta, alpha, p, T, s=S, n_r=1200, n_w=64)
    return sgg_mse(alpha, beta, p), beta


def claim4():
    print("== Claim 4: low-dimensional SGG improvements (Figure 2) ==")
    settings = [(2, 2.25e-3, 15.5, 1.37), (5, 1.46e-3, 6.2, 1.32), (10, 1.18e-3, 2.5, 1.92)]
    rows = []
    for T, delta_t, paper_gain, paper_p in settings:
        t0 = time.time()
        u0 = gauss_u0(delta_t, EPS)
        mse_gauss = T * u0
        mse_l2, _ = calibrated_mse(EPS, delta_t, T - 1, 1.0, T)
        # (alpha, p) grid + local refine
        best = (np.inf, None, None, None)
        for al in np.linspace(max(-0.5, T - 1 - 6), T - 1, 8):
            for p in np.linspace(0.8, 2.6, 10):
                try:
                    m, b = calibrated_mse(EPS, delta_t, al, p, T)
                except Exception:
                    continue
                if m < best[0]:
                    best = (m, al, p, b)
        # Nelder-Mead refine from the best grid point
        def obj(x):
            al, p = x
            if not (-0.99 < al <= T - 1 and 0.5 < p <= 4):
                return 1e18
            try:
                return calibrated_mse(EPS, delta_t, al, p, T)[0]
            except Exception:
                return 1e18
        res = optimize.minimize(obj, [best[1], best[2]], method="Nelder-Mead",
                                options=dict(maxiter=60, xatol=1e-3, fatol=1e-3))
        mse_sgg = min(best[0], res.fun)
        al_s, p_s = (res.x if res.fun <= best[0] else (best[1], best[2]))
        base = min(mse_gauss, mse_l2)
        gain = (1 - mse_sgg / base) * 100
        rows.append({"T": T, "delta": delta_t, "mse_gauss": mse_gauss,
                     "mse_l2": mse_l2, "mse_sgg": mse_sgg,
                     "sgg_alpha": al_s, "sgg_p": p_s,
                     "gain_pct_ours": gain, "gain_pct_paper": paper_gain,
                     "p_star_paper": paper_p})
        print(f"T={T:2d} delta={delta_t:.2e}: MSE gauss={mse_gauss:.1f} l2={mse_l2:.1f} "
              f"SGG={mse_sgg:.1f} (alpha={al_s:.2f}, p={p_s:.2f}) "
              f"gain={gain:.1f}% (paper {paper_gain}%, p*={paper_p}) ({time.time()-t0:.0f}s)")
    with open("claim4_fig2.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    gains = [r["gain_pct_ours"] for r in rows]
    print(f"monotone shrinking with T: {all(gains[i] > gains[i+1] for i in range(len(gains)-1))}")
    return rows


def claim6():
    print("\n== Claim 6: tight FFT composition (Prop 4.2 / Alg 7) ==")
    T = 5
    # (i) k=1 self-consistency, Gaussian member
    sigma2 = 2.0
    al, be, p = T - 1, 1 / (2 * sigma2), 2
    Z, P = prv_grid(al, be, p, T, s=S)
    grid, pmf = compose_fft(Z, P, 1)
    for eps in [0.5, 1.0, 2.0]:
        d_fft = delta_from_pld(grid, pmf, eps)
        d_dir = delta_sgg(eps, al, be, p, T, s=S, n_r=2000, n_w=96)
        print(f"(i) k=1 eps={eps}: FFT={d_fft:.6e} direct={d_dir:.6e} "
              f"relerr={abs(d_fft/d_dir-1):.2e}")
    # (ii) Gaussian k-fold vs closed form
    print("(ii) Gaussian member, k-fold vs analytic (s -> s*sqrt(k)):")
    rows = []
    for k in [2, 4, 8, 16]:
        grid, pmf = compose_fft(Z, P, k)
        for eps in [1.0, 3.0]:
            d_fft = delta_from_pld(grid, pmf, eps)
            d_ana = float(gauss_g(sigma2 / k, eps))  # k-fold == s*sqrt(k) == u/k
            rows.append({"k": k, "eps": eps, "d_fft": d_fft, "d_analytic": d_ana,
                         "rel_err": d_fft / d_ana - 1})
            print(f"    k={k:2d} eps={eps}: FFT={d_fft:.6e} analytic={d_ana:.6e} "
                  f"relerr={abs(d_fft/d_ana-1):.2e}")
    # (iii) l2 mechanism k-fold vs Monte-Carlo of summed PRVs
    print("(iii) l2 mechanism (p=1), k-fold FFT vs direct MC of summed PRVs:")
    theta = 1.0 / 0.7
    al, be, p = T - 1, 1 / theta, 1
    Z, P = prv_grid(al, be, p, T, s=S)
    rng = np.random.default_rng(0)
    n_mc = 4_000_000
    def prv_samples(n):
        r = ggamma_sample(al, be, p, n, rng)
        w = 2 * rng.beta((T - 1) / 2, (T - 1) / 2, n) - 1
        return privacy_loss_Z(r, w, al, be, p, T, S)
    l2rows = []
    for k in [2, 4]:
        grid, pmf = compose_fft(Z, P, k)
        zsum = np.zeros(n_mc)
        for _ in range(k):
            zsum += prv_samples(n_mc)
        for eps_tot in [0.5, 1.0, 2.0]:
            d_fft = delta_from_pld(grid, pmf, eps_tot)
            x = np.clip(1 - np.exp(np.minimum(eps_tot - zsum, 50.0)), 0, None)
            d_mc, se = float(x.mean()), float(x.std() / np.sqrt(n_mc))
            z = abs(d_fft - d_mc) / max(se, 1e-12)
            l2rows.append({"k": k, "eps_tot": eps_tot, "d_fft": d_fft,
                           "d_mc": d_mc, "mc_se": se, "z_score": z})
            print(f"    k={k} eps_tot={eps_tot}: FFT={d_fft:.6e} MC={d_mc:.6e}±{se:.1e} z={z:.2f}")
    with open("claim6_composition.csv", "w", newline="") as fh:
        allrows = rows + l2rows
        keys = sorted({k for r in allrows for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(allrows)
    # bonus: the open-question answer in action -- tight l2 accounting vs the
    # naive basic-composition bound (k*eps at same per-step delta)
    grid, pmf = compose_fft(Z, P, 4)
    d1 = delta_sgg(0.5, al, be, p, T, s=S, n_r=2000, n_w=96)
    d_tight = delta_from_pld(grid, pmf, 2.0)
    print(f"    tightness dividend: 4-step l2, per-step delta(0.5)={d1:.3e}; "
          f"basic composition would claim delta(2.0)<={4*d1:.3e}, "
          f"tight FFT accountant gives {d_tight:.3e}")


if __name__ == "__main__":
    t0 = time.time()
    claim4()
    claim6()
    print(f"\ntotal wall: {time.time()-t0:.0f}s")
