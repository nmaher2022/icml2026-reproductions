# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
Claim 3 (Algorithm 1, Sec 4.2.2): D&L combines a multi-expert framework
(UCB, EXP3, FTRL mixture) with Lagrangian dual coordination across
overlapping subproblems. This audits the *structural* claim by ablation,
mirroring the paper's own ablation 11.3 ("the multi-experts framework
outperforms any single expert") and the coordination mechanism from
Theorem 4.5 -- i.e. does removing either piece from Algorithm 1 hurt
solution quality, as the paper's design would predict?
"""
from pathlib import Path
import time
import numpy as np
import csv

from dl_synthetic import SyntheticEnv
from dl_core import run_dl

HERE = Path(__file__).resolve().parent


def final_quality(env, n, num_actions, K, T, seed, **kwargs):
    out = run_dl(n=n, num_actions=num_actions, reward_fn=env, K=K, T=T,
                  seed=seed, **kwargs)
    return env.expected(out["final_x"])


def main():
    results = []
    n, num_actions, K, T = 48, 5, 6, 500
    reps = 6

    print("=== Claim 3a: multi-expert mixture vs each single expert alone ===")
    configs = [
        ("all (UCB+EXP3+FTRL)", dict(experts="all", use_ts=False)),
        ("all-TS (TS+EXP3+FTRL)", dict(experts="all", use_ts=True)),
        ("UCB-only", dict(experts="ucb")),
        ("EXP3-only", dict(experts="exp3")),
        ("FTRL-only", dict(experts="ftrl")),
        ("TS-only", dict(experts="ts")),
    ]
    for label, kw in configs:
        vals = []
        for rep in range(reps):
            env = SyntheticEnv(n=n, num_actions=num_actions, coupling=0.2,
                                 seed=600 + rep)
            q = final_quality(env, n, num_actions, K, T, seed=rep, **kw)
            vals.append(q / env.optimal_expected())  # fraction of optimum
        m, s = np.mean(vals), np.std(vals)
        print(f"  {label:24s}  final_quality/optimum = {m:.4f} +/- {s:.4f}")
        results.append(dict(part="3a", config=label, n=n, K=K, T=T,
                             frac_optimum_mean=m, frac_optimum_std=s))

    print("\n=== Claim 3b: with vs without Lagrangian coordination "
          "(overlapping subproblems only -- use fixed nonzero overlap) ===")
    for coord in [True, False]:
        vals = []
        for rep in range(reps):
            env = SyntheticEnv(n=n, num_actions=num_actions, coupling=0.25,
                                 seed=700 + rep)
            out = run_dl(n=n, num_actions=num_actions, reward_fn=env, K=K, T=T,
                          o0=3, alpha=0.0, seed=rep, use_lagrangian=coord)
            vals.append(env.expected(out["final_x"]) / env.optimal_expected())
        m, s = np.mean(vals), np.std(vals)
        print(f"  Lagrangian coordination={coord!s:5s}  "
              f"final_quality/optimum = {m:.4f} +/- {s:.4f}")
        results.append(dict(part="3b", config=f"coordination={coord}", n=n,
                             K=K, T=T, frac_optimum_mean=m, frac_optimum_std=s))

    print("\n=== Claim 3c: with vs without coordination, HEAVIER overlap "
          "(o0=6 vs 3) and stronger coupling (0.4 vs 0.25), more reps -- "
          "does the coordination benefit grow with overlap severity, as "
          "Theorem 4.5's K^2*Q*R_max*sqrt(T) term predicts it should? ===")
    for coord in [True, False]:
        vals = []
        for rep in range(12):
            env = SyntheticEnv(n=n, num_actions=num_actions, coupling=0.4,
                                 seed=700 + rep)
            out = run_dl(n=n, num_actions=num_actions, reward_fn=env, K=K, T=T,
                          o0=6, alpha=0.0, seed=rep, use_lagrangian=coord)
            vals.append(env.expected(out["final_x"]) / env.optimal_expected())
        m, s = np.mean(vals), np.std(vals)
        print(f"  Lagrangian coordination={coord!s:5s}  "
              f"final_quality/optimum = {m:.4f} +/- {s:.4f}  (12 reps)")
        results.append(dict(part="3c", config=f"coordination={coord}", n=n,
                             K=K, T=T, frac_optimum_mean=m, frac_optimum_std=s))

    with open(HERE / "claim3_ablation.csv",
              "w", newline="") as f:
        keys = sorted({k for r in results for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print("\nSaved -> results/claim3_ablation.csv")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal wall-clock: {time.time()-t0:.1f}s")
