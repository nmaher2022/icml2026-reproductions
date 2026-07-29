# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
Claim 1 (Corollary 5.3), part 2: a more faithful test than the naive
"vary n at fixed d" sweep in claim1_2_regret_coupling.py.

Two things Corollary 5.3 / Theorem 5.2 actually predict that the first sweep
did NOT isolate:

(a) D&L's position-wise statistics (shape n x |A|, i.e. O(n*|A|) memory) are
    what let it dodge the O(sqrt(T|X|)) regret of a "flat" combinatorial
    bandit that must maintain one statistic per FULL solution (|X| =
    |A|^n arms). This holds for ANY K >= 1 in our implementation (K=1 in the
    first sweep was NOT the paper's "naive combinatorial bandit" baseline --
    it was still position-wise D&L, just with one subproblem covering
    everything, so of course it didn't blow up with n either). Here we
    actually build the exponential-arm flat baseline (only tractable at
    small n) and compare.

(b) R_avg(T) := (1/K) * R_total(T) (Def. right after Thm 5.2) should trade
    off: subproblem-learning term O(d*sqrt(T log T)) shrinks as K grows
    (d=n/K shrinks), while overlap-coordination term O(K*Q*R_max*sqrt(T))
    grows with K -- an interior-optimum K* = O(sqrt(n/(Q*R_max))) is
    predicted (paragraph after Thm 4.5 / Corollary 5.3). We sweep K at fixed
    n, T and look for the predicted U-shape.
"""
from pathlib import Path
import time, itertools
import numpy as np
import csv

from dl_synthetic import SyntheticEnv
from dl_core import run_dl

HERE = Path(__file__).resolve().parent


def flat_bandit_run(env, n, num_actions, T, seed):
    """UCB over the FULL solution space (one arm per full assignment) --
    only tractable for tiny n. This is the paper's contrasting baseline:
    'naive combinatorial bandits incur O(sqrt(T|X|)) where |X| may be 2^n or
    n!' (Sec 5, 'Comparison to prior work').
    """
    rng = np.random.default_rng(seed)
    arms = list(itertools.product(range(num_actions), repeat=n))
    n_arms = len(arms)
    V = np.zeros(n_arms)
    N = np.zeros(n_arms)
    r_star = env.optimal_expected()
    cum_regret = 0.0
    trace = []
    # init: play every arm once if T allows, else random subset
    order = rng.permutation(n_arms)
    for t in range(1, T + 1):
        if t <= n_arms:
            a_idx = order[t - 1]
        else:
            bonus = np.sqrt(2 * np.log(t) / np.maximum(N, 1e-9))
            a_idx = int(np.argmax(V + bonus))
        x = np.array(arms[a_idx])
        r = env(x)
        N[a_idx] += 1
        V[a_idx] += (r - V[a_idx]) / N[a_idx]
        cum_regret += r_star - env.expected(x)
        trace.append(cum_regret)
    return n_arms, np.array(trace)


def main():
    results = []

    print("=== (a) D&L (position-wise, any K) vs FLAT combinatorial-bandit "
          "baseline, tiny n where |X|=|A|^n is still tractable ===")
    num_actions = 3
    T = 300
    for n in [4, 6, 8]:
        n_arms = num_actions ** n
        env = SyntheticEnv(n=n, num_actions=num_actions, coupling=0.15, seed=42)
        r_star = env.optimal_expected()

        K = max(n // 2, 1)
        out = run_dl(n=n, num_actions=num_actions, reward_fn=env, K=K, T=T,
                      o0=1, alpha=0.0, seed=0)
        dl_cum_regret = np.cumsum(r_star - out["exp_rewards"])[-1]

        arms, flat_trace = flat_bandit_run(env, n, num_actions, T, seed=0)
        flat_cum_regret = flat_trace[-1]

        print(f"  n={n} |A|={num_actions} |X|={n_arms:6d}  T={T}  "
              f"D&L(K={K}) cum_regret={dl_cum_regret:8.2f}   "
              f"flat-bandit cum_regret={flat_cum_regret:8.2f}   "
              f"(flat needs >= {n_arms} pulls just to try every arm once; "
              f"T={T} {'covers' if T>=n_arms else 'does NOT cover'} that)")
        results.append(dict(part="a", n=n, num_actions=num_actions,
                             n_arms=n_arms, T=T, K=K,
                             dl_cum_regret=dl_cum_regret,
                             flat_cum_regret=flat_cum_regret))

    print("\n=== (b) R_avg(T) = (1/K)*cumulative_regret(T) vs K at fixed "
          "n=64,T=400 -- looking for the predicted U-shape "
          "(subproblem term down, coordination term up as K grows) ===")
    n, num_actions, T = 64, 5, 400
    Ks = [1, 2, 4, 8, 16, 32]
    for K in Ks:
        vals = []
        for rep in range(4):
            env = SyntheticEnv(n=n, num_actions=num_actions, coupling=0.2,
                                 seed=500 + rep)
            r_star = env.optimal_expected()
            out = run_dl(n=n, num_actions=num_actions, reward_fn=env, K=K, T=T,
                          o0=2, alpha=0.5, seed=rep)
            cum_regret = np.cumsum(r_star - out["exp_rewards"])[-1]
            r_avg = cum_regret / (K * T)
            vals.append(r_avg)
        m, s = np.mean(vals), np.std(vals)
        d = n / K
        print(f"  K={K:3d} d={d:5.1f}  R_avg(T)={m:.5f} +/- {s:.5f}")
        results.append(dict(part="b", n=n, K=K, d=d, T=T,
                             r_avg_mean=m, r_avg_std=s))

    with open(HERE / "claim1_tradeoff.csv",
              "w", newline="") as f:
        keys = sorted({k for r in results for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print("\nSaved -> results/claim1_tradeoff.csv")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal wall-clock: {time.time()-t0:.1f}s")
