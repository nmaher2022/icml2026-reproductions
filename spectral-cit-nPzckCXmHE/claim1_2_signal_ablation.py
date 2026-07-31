# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scipy"]
# ///
"""Claims 1 (Thm 4.1, validity) and 2 (Thm 4.2, power).

Reproduces the paper's Fig. 11 signal-strength-ablation benchmark (Appendix C, the
cheapest/smallest synthetic experiment): fixed d_Z=3, str_z=0.1, noise_str=0.25, and
str_cond_dep in {0.05, 0.15, 0.5}. Paper uses N unspecified explicitly for this one
figure (Appendix C text doesn't restate N for Fig 11) -- we use N=1000 to match every
other synthetic benchmark in the paper. Paper repeats 500 times per setting; here we use
a reduced rep count (still gives a directionally trustworthy Monte Carlo estimate; see
REPRO_LOG.md for the time-budget reasoning).

Reports BOTH the raw statistic (T_n vs chi2(d^2)) and the Appendix-C "dimension-pruned"
statistic (T_n_pruned vs chi2(k^2), k = floor(perc_dim_prune * d)) -- see BUGFIX_LOG.md
entry 3 for why pruning turned out to matter in this from-scratch reimplementation.

Writes raw per-trial values to claim1_2_raw.csv (so the numbers are checkable, not just
the summary this script prints) and a summary to claim1_2_summary.csv.
"""
from __future__ import annotations

import csv
import sys
import time

import numpy as np
from scipy.stats import chi2, kstest

from data_gen import signal_strength_ablation
from scit_lib import run_spectral_cit_trial

D = 10  # nominal truncation dim (Table 2 reference value)
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
print(f"SpectralCIT signal-strength ablation: d={D} (raw d^2={d2} df), d_z={D_Z}, N={N}, "
      f"{N_REPS} reps/condition, alpha={ALPHA}, raw chi2 crit={crit_raw:.2f}, "
      f"perc_dim_prune={PERC_DIM_PRUNE}")

raw_rows = []  # condition, rep, T_n, T_n_pruned, pruned_dim, E_val
summary_rows = []

rng = np.random.default_rng(42)

conditions = [("H0", 0.0, True)] + [
    (f"H1_str{s}", s, False) for s in (0.05, 0.15, 0.5)
]

t_start = time.time()
for cond_name, strength, is_null in conditions:
    Ts, Ts_pruned, k_last = [], [], None
    for rep in range(N_REPS):
        X, Y, Z = signal_strength_ablation(N, D_Z, str_cond_dep=strength, null=is_null, rng=rng)
        res = run_spectral_cit_trial(X, Y, Z, seed=hash((cond_name, rep)) % (2**31),
                                      perc_dim_prune=PERC_DIM_PRUNE, **TRAIN_KW)
        Ts.append(res["T_n"])
        Ts_pruned.append(res["T_n_pruned"])
        k_last = res["pruned_dim"]
        raw_rows.append([cond_name, rep, res["T_n"], res["T_n_pruned"], res["pruned_dim"], res["E_val"]])
        elapsed = time.time() - t_start
        print(f"[{elapsed:7.1f}s] {cond_name} rep {rep+1}/{N_REPS}: "
              f"T_n={res['T_n']:.3f} T_n_pruned={res['T_n_pruned']:.3f} (k={res['pruned_dim']}) "
              f"E_val={res['E_val']:.3f}")

    Ts = np.array(Ts)
    Ts_pruned = np.array(Ts_pruned)
    crit_pruned = chi2.ppf(1 - ALPHA, k_last * k_last)
    reject_raw = float(np.mean(Ts >= crit_raw))
    reject_pruned = float(np.mean(Ts_pruned >= crit_pruned))
    row = dict(condition=cond_name, strength=strength, is_null=is_null, n_reps=N_REPS,
               mean_Tn=float(Ts.mean()), std_Tn=float(Ts.std()), reject_rate_raw=reject_raw,
               pruned_dim=k_last, mean_Tn_pruned=float(Ts_pruned.mean()),
               std_Tn_pruned=float(Ts_pruned.std()), reject_rate_pruned=reject_pruned)
    if is_null:
        ks_stat, ks_p = kstest(Ts, "chi2", args=(d2,))
        ks_stat_p, ks_p_p = kstest(Ts_pruned, "chi2", args=(k_last * k_last,))
        row.update(ks_stat=float(ks_stat), ks_pvalue=float(ks_p),
                   ks_stat_pruned=float(ks_stat_p), ks_pvalue_pruned=float(ks_p_p))
        print(f"  -> H0 RAW:    mean T_n={Ts.mean():.2f} (chi2({d2}) mean={d2}), "
              f"Type I error={reject_raw:.3f} (nominal {ALPHA}), KS p={ks_p:.3f}")
        print(f"  -> H0 PRUNED: mean T_n={Ts_pruned.mean():.2f} (chi2({k_last*k_last}) "
              f"mean={k_last*k_last}), Type I error={reject_pruned:.3f} (nominal {ALPHA}), "
              f"KS p={ks_p_p:.3f}")
    else:
        print(f"  -> {cond_name} RAW power={reject_raw:.3f}  PRUNED power={reject_pruned:.3f}")
    summary_rows.append(row)

with open("claim1_2_raw.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "rep", "T_n", "T_n_pruned", "pruned_dim", "E_val"])
    w.writerows(raw_rows)

with open("claim1_2_summary.csv", "w", newline="") as f:
    fieldnames = ["condition", "strength", "is_null", "n_reps", "mean_Tn", "std_Tn",
                  "reject_rate_raw", "pruned_dim", "mean_Tn_pruned", "std_Tn_pruned",
                  "reject_rate_pruned", "ks_stat", "ks_pvalue", "ks_stat_pruned", "ks_pvalue_pruned"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in summary_rows:
        w.writerow(row)

print(f"\nTotal elapsed: {time.time()-t_start:.1f}s. Wrote claim1_2_raw.csv, claim1_2_summary.csv")
