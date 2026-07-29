# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
Claim 1 (Corollary 5.3): D&L's regret bound O(d*sqrt(T log T)) depends on
subproblem dimensionality d=n/K, not full problem size n.

Claim 2 (Theorem 4.5): coupling error across overlapping subproblems is
bounded by O(K^2 * Q * R_max * sqrt(T)), and Lagrangian coordination is what
keeps it there (vs. an uncoordinated control).

Independent numerical audit on the synthetic decomposable-with-bounded-
coupling environment (dl_synthetic.py) since no official code/benchmarks
exist for the theory sections. Controls included: (a) K=1 "monolithic"
baseline (d=n, no decomposition) to show regret DOES grow with n when not
decomposed; (b) Lagrangian coordination OFF to show coupling error is not
controlled without it.
"""
from pathlib import Path
import time, itertools
import numpy as np
import csv

from dl_synthetic import SyntheticEnv
from dl_core import run_dl

HERE = Path(__file__).resolve().parent


def brute_force_optimum(env, n, num_actions):
    best = -1e9
    for combo in itertools.product(range(num_actions), repeat=n):
        r = env.expected(np.array(combo))
        if r > best:
            best = r
    return best


def proxy_optimum(env):
    return env.optimal_expected()


def regret_curve(env, n, num_actions, K, T, seed, use_lagrangian=True):
    out = run_dl(n=n, num_actions=num_actions, reward_fn=env, K=K, T=T,
                 o0=2, alpha=0.0, seed=seed, use_lagrangian=use_lagrangian)
    r_star = proxy_optimum(env)
    inst_regret = r_star - out["exp_rewards"]
    cum_regret = np.cumsum(inst_regret)
    avg_regret_final = cum_regret[-1] / T
    return avg_regret_final, cum_regret, out["coupling_err"]


def main():
    rng_seed_base = 0
    results = []

    # --- sanity check: proxy optimum vs brute force at small n ---
    env_small = SyntheticEnv(n=10, num_actions=4, coupling=0.15, seed=1)
    bf = brute_force_optimum(env_small, 10, 4)
    px = proxy_optimum(env_small)
    print(f"[sanity] brute-force optimum={bf:.4f} vs per-position proxy optimum={px:.4f} "
          f"(gap={bf-px:.4f}) at n=10, coupling=0.15")

    # =========================================================
    # Claim 1: fix d = n/K, vary n (and K proportionally) -> regret should
    # stay roughly flat. Contrast with K=1 (monolithic, d=n) where regret
    # should grow with n.
    # =========================================================
    print("\n=== Claim 1: regret vs n at FIXED subproblem size d (decomposed) "
          "vs MONOLITHIC (K=1, d=n) ===")
    d_fixed = 8
    T = 400
    num_actions = 5
    ns = [16, 32, 64, 128]
    for n in ns:
        K_decomp = max(n // d_fixed, 1)
        for label, K in [("decomposed(d=8)", K_decomp), ("monolithic(K=1)", 1)]:
            avg_regrets = []
            for rep in range(3):
                env = SyntheticEnv(n=n, num_actions=num_actions, coupling=0.15,
                                    seed=100 + rep)
                avg_r, cum_r, _ = regret_curve(env, n, num_actions, K, T,
                                                 seed=rng_seed_base + rep)
                avg_regrets.append(avg_r)
            m, s = np.mean(avg_regrets), np.std(avg_regrets)
            d_eff = n / K
            print(f"  n={n:4d} K={K:3d} d={d_eff:6.1f} [{label:18s}] "
                  f"avg_regret(T={T})={m:.4f} +/- {s:.4f}")
            results.append(dict(claim=1, n=n, K=K, d=d_eff, label=label,
                                 T=T, avg_regret_mean=m, avg_regret_std=s))

    # Regret vs T scaling (should grow ~sqrt(T log T) cumulative, i.e. avg
    # regret ~ sqrt(log T / T) -> 0)
    print("\n=== Claim 1: cumulative regret vs T at fixed n=64,K=8 (d=8) ===")
    Ts = [100, 200, 400, 800, 1600]
    n, K = 64, 8
    cum_finals = []
    for T in Ts:
        env = SyntheticEnv(n=n, num_actions=num_actions, coupling=0.15, seed=200)
        _, cum_r, _ = regret_curve(env, n, num_actions, K, T, seed=0)
        cum_finals.append(cum_r[-1])
        print(f"  T={T:5d}  cumulative_regret={cum_r[-1]:.3f}  "
              f"sqrt(T logT)={np.sqrt(T*np.log(max(T,2))):.2f}")
        results.append(dict(claim=1, n=n, K=K, d=n/K, label="regret_vs_T",
                             T=T, cum_regret=cum_r[-1]))
    # fit cumulative_regret ~ c * T^p
    logT = np.log(Ts)
    logR = np.log(np.maximum(cum_finals, 1e-6))
    p, c = np.polyfit(logT, logR, 1)
    print(f"  fitted power-law exponent p (cum_regret ~ T^p): {p:.3f} "
          f"(sqrt(T log T) implies p should be < 1, ~0.5-0.6 over this range)")

    # =========================================================
    # Claim 2: coupling error scaling vs K, Q, T; with vs without Lagrangian
    # coordination.
    # =========================================================
    print("\n=== Claim 2: cumulative coupling error vs K (fixed Q via alpha=0 "
          "constant overlap), WITH vs WITHOUT Lagrangian coordination ===")
    n = 64
    num_actions = 5
    T = 400
    Q_fixed = 3  # o0 passed directly as approx overlap half-width
    for K in [2, 4, 8, 16]:
        for coord in [True, False]:
            errs = []
            for rep in range(3):
                env = SyntheticEnv(n=n, num_actions=num_actions, coupling=0.2,
                                    seed=300 + rep)
                out = run_dl(n=n, num_actions=num_actions, reward_fn=env, K=K,
                              T=T, o0=Q_fixed, alpha=0.0, seed=rep,
                              use_lagrangian=coord)
                errs.append(np.sum(out["coupling_err"]))
            m, s = np.mean(errs), np.std(errs)
            print(f"  K={K:3d} Q~{Q_fixed} coordination={coord!s:5s} "
                  f"cumulative_coupling_err={m:.4f} +/- {s:.4f} "
                  f"(K^2*Q*sqrt(T) predictor={K**2*Q_fixed*np.sqrt(T):.1f})")
            results.append(dict(claim=2, n=n, K=K, Q=Q_fixed, T=T,
                                 coordination=coord, coupling_err_mean=m,
                                 coupling_err_std=s))

    print("\n=== Claim 2: cumulative coupling error vs T at fixed K=8, Q=3 "
          "(coordination ON) ===")
    K, Q_fixed = 8, 3
    couple_T = []
    for T in Ts:
        env = SyntheticEnv(n=64, num_actions=5, coupling=0.2, seed=400)
        out = run_dl(n=64, num_actions=5, reward_fn=env, K=K, T=T, o0=Q_fixed,
                      alpha=0.0, seed=0, use_lagrangian=True)
        ce = np.sum(out["coupling_err"])
        couple_T.append(ce)
        print(f"  T={T:5d} cumulative_coupling_err={ce:.4f} "
              f"sqrt(T)={np.sqrt(T):.2f}")
        results.append(dict(claim=2, n=64, K=K, Q=Q_fixed, T=T,
                             coordination=True, coupling_err_vs_T=ce))
    logT2 = np.log(Ts)
    logC = np.log(np.maximum(couple_T, 1e-6))
    p2, c2 = np.polyfit(logT2, logC, 1)
    print(f"  fitted power-law exponent p (coupling_err ~ T^p): {p2:.3f} "
          f"(sqrt(T) implies p should be ~0.5)")

    with open(HERE / "claim1_2_regret_coupling.csv",
              "w", newline="") as f:
        keys = sorted({k for r in results for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print("\nSaved -> results/claim1_2_regret_coupling.csv")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal wall-clock: {time.time()-t0:.1f}s")
