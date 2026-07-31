# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scipy"]
# ///
"""Claim 5 (Assumption 4.1): validity requires the learned representations to be
K-sub-Gaussian, operationalized by the paper via bounded (Tanh) activations.

Ablation: retrain the SAME H0 signal-strength-ablation setting (Claims 1/2's cheapest
benchmark) with Tanh replaced by an unbounded activation (Identity -- i.e. plain linear
layers, the most direct way to break the sub-Gaussian/bounded-output property while
changing nothing else about the architecture or training procedure) and compare Type I
error control against the Tanh baseline. If Assumption 4.1 is actually load-bearing,
removing the bound should degrade (inflate) Type I error relative to the Tanh runs
already collected in claim1_2_summary.csv's H0 row.
"""
from __future__ import annotations

import csv
import sys
import time

import numpy as np
import torch
from scipy.stats import chi2, kstest
from torch import nn

from data_gen import signal_strength_ablation
from scit_lib import run_spectral_cit_trial

D = 10
D_Z = 3
N = 1000
ALPHA = 0.05
PERC_DIM_PRUNE = 0.9
N_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30

TRAIN_KW = dict(
    d=D, n_hidden=2, layer_size=128,
    lr_inner=3e-5, lr_outer=2.1e-3, reg_inner=3.3, reg_outer=1.9,
    batch_size=128, n_epochs=400, warmup_steps=100,
)

d2 = D * D
crit_raw = chi2.ppf(1 - ALPHA, d2)
print(f"Claim 5 ablation: Tanh (bounded, satisfies Assumption 4.1) vs Identity "
      f"(unbounded, violates it) under H0. d={D}, d_z={D_Z}, N={N}, {N_REPS} reps each.")

rng = np.random.default_rng(7)
raw_rows = []
summary_rows = []

for act_name, act_cls in [("Tanh", nn.Tanh), ("Identity_unbounded", nn.Identity)]:
    Ts, Ts_pruned, k_last = [], [], None
    t0 = time.time()
    for rep in range(N_REPS):
        X, Y, Z = signal_strength_ablation(N, D_Z, str_cond_dep=0.0, null=True, rng=rng)
        res = run_spectral_cit_trial(X, Y, Z, seed=hash((act_name, rep)) % (2**31),
                                      perc_dim_prune=PERC_DIM_PRUNE, activation=act_cls,
                                      **TRAIN_KW)
        Ts.append(res["T_n"])
        Ts_pruned.append(res["T_n_pruned"])
        k_last = res["pruned_dim"]
        raw_rows.append([act_name, rep, res["T_n"], res["T_n_pruned"], res["pruned_dim"], res["E_val"]])
        print(f"[{time.time()-t0:6.1f}s] {act_name} rep {rep+1}/{N_REPS}: "
              f"T_n={res['T_n']:.3f} T_n_pruned={res['T_n_pruned']:.3f} E_val={res['E_val']:.3f}")

    Ts = np.array(Ts)
    Ts_pruned = np.array(Ts_pruned)
    crit_pruned = chi2.ppf(1 - ALPHA, k_last * k_last)
    reject_raw = float(np.mean(Ts >= crit_raw))
    reject_pruned = float(np.mean(Ts_pruned >= crit_pruned))
    ks_stat, ks_p = kstest(Ts, "chi2", args=(d2,))
    print(f"  -> {act_name}: Type I error raw={reject_raw:.3f} pruned={reject_pruned:.3f} "
          f"(nominal {ALPHA}), mean T_n={Ts.mean():.2f} (chi2({d2}) mean={d2}), KS p={ks_p:.3f}")
    summary_rows.append(dict(activation=act_name, n_reps=N_REPS, mean_Tn=float(Ts.mean()),
                              reject_rate_raw=reject_raw, reject_rate_pruned=reject_pruned,
                              ks_pvalue=float(ks_p)))

with open("claim5_raw.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["activation", "rep", "T_n", "T_n_pruned", "pruned_dim", "E_val"])
    w.writerows(raw_rows)

with open("claim5_summary.csv", "w", newline="") as f:
    fieldnames = ["activation", "n_reps", "mean_Tn", "reject_rate_raw", "reject_rate_pruned", "ks_pvalue"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in summary_rows:
        w.writerow(row)

print("\nWrote claim5_raw.csv, claim5_summary.csv")
