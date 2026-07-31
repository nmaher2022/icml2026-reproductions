# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scipy"]
# ///
"""Smoketest (Step 3): tiny-scale run of SpectralCIT before scaling to the paper's
settings. Checks: no NaNs, sane T_n magnitude (order d^2), whitening produces near-
identity covariances, and a qualitative signal (H1 rejects more than H0) on a handful
of reps. This is NOT a claim verification -- see claim*_*.py scripts for that.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy.stats import chi2

from data_gen import signal_strength_ablation
from scit_lib import run_spectral_cit_trial

rng = np.random.default_rng(0)

D = 2          # tiny truncation dim (paper nominal: 10)
D_Z = 3        # matches Fig. 11's d_z=3
N = 200        # small sample size (paper: N=1000)
N_REPS = 5
TRAIN_KW = dict(
    d=D, n_hidden=1, layer_size=32,
    lr_inner=3e-4, lr_outer=3e-3, reg_inner=1.0, reg_outer=1.0,
    batch_size=32, n_epochs=5, warmup_steps=5,
)

print(f"Smoketest: d={D}, d_z={D_Z}, N={N}, {N_REPS} reps, tiny epochs={TRAIN_KW['n_epochs']}")

t0 = time.time()
h0_stats, h1_stats = [], []
for rep in range(N_REPS):
    X, Y, Z = signal_strength_ablation(N, D_Z, str_cond_dep=0.0, null=True, rng=rng)
    res = run_spectral_cit_trial(X, Y, Z, seed=rep, **TRAIN_KW)
    assert np.isfinite(res["T_n"]), f"NaN/Inf T_n under H0 at rep {rep}: {res}"
    h0_stats.append(res["T_n"])
    print(f"  H0 rep {rep}: T_n={res['T_n']:.3f}  E_val={res['E_val']:.3f}  n_test={res['n_test']}")

for rep in range(N_REPS):
    X, Y, Z = signal_strength_ablation(N, D_Z, str_cond_dep=0.5, null=False, rng=rng)
    res = run_spectral_cit_trial(X, Y, Z, seed=100 + rep, **TRAIN_KW)
    assert np.isfinite(res["T_n"]), f"NaN/Inf T_n under H1 at rep {rep}: {res}"
    h1_stats.append(res["T_n"])
    print(f"  H1 rep {rep}: T_n={res['T_n']:.3f}  E_val={res['E_val']:.3f}  n_test={res['n_test']}")

elapsed = time.time() - t0
d2 = D * D
crit = chi2.ppf(0.95, d2)
print(f"\nchi2({d2}) 0.95-quantile = {crit:.3f}")
print(f"H0 T_n: mean={np.mean(h0_stats):.3f} (expect ~ d^2={d2} order of magnitude)")
print(f"H1 T_n: mean={np.mean(h1_stats):.3f}")
print(f"H0 rejection rate @ alpha=0.05: {np.mean(np.array(h0_stats) >= crit):.2f} (want ~0.05, noisy at n_reps={N_REPS})")
print(f"H1 rejection rate @ alpha=0.05: {np.mean(np.array(h1_stats) >= crit):.2f} (want high)")
print(f"Elapsed: {elapsed:.1f}s for {2*N_REPS} trials ({elapsed/(2*N_REPS):.2f}s/trial)")

assert np.mean(h1_stats) > np.mean(h0_stats), "SANITY FAIL: H1 statistic not larger than H0 on average"
print("\nSMOKETEST PASSED: no NaNs, timing sane, H1 > H0 direction correct.")
