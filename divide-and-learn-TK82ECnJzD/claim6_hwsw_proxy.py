# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""
Claim 6 (paper's Table 2): on a real-world 4-objective hardware-software
co-design task for AI accelerators (Dutta et al. 2022 simulator, ~20-dim
mixed-integer space, ~2.2e5 configs), the paper reports D&L-TS beating a
qParEGO-style MOBO baseline by ~22% hypervolume-ratio under a tight 150-eval
budget (10-seed average):
    NSGA-II        0.291 +/- 0.040
    MOBO-qNEHVI    0.287 +/- 0.015
    MOBO-qParEGO   0.34  +/- 0.012
    D&L-TS         0.372 +/- 0.006

*** We do NOT have the Dutta et al. 2022 accelerator simulator (proprietary /
specialized infra). This script is a SYNTHETIC PROXY reproduction, not a
literal replication of Table 2's numbers. *** The goal is only to test
whether D&L-TS's qualitative advantage over a qParEGO-style BO baseline
survives on a comparable synthetic stand-in problem (4-obj, ~20-dim mixed-
integer black box, conflicting objectives, mild observation noise, 150-eval
budget, 10 seeds) -- not to hit 0.372.

Methods compared:
  - D&L-TS: dl.core.run_dl(..., use_ts=True) with the Tchebycheff-scalarized
    reward_fn adapter in dl.hwsw_proxy.TchebycheffDLAdapter.
  - "qParEGO-analogue": our own simplified sequential ParEGO-style BO loop
    (random init -> {random Chebyshev-Dirichlet weight, GP surrogate on the
    scalarized augmented-Tchebycheff fitness, EI acquisition over a discrete
    candidate pool (random + local neighbor mutations of top points),
    evaluate, repeat}) using sklearn's GaussianProcessRegressor over a
    continuous-relaxation encoding of the discrete design space. botorch was
    checked (see NOTE below) and skipped in favor of this lightweight
    from-scratch analogue.
  - NSGA-II via pymoo, secondary comparison, same integer design encoding.

NOTE on botorch: `python3.13 -c "import botorch"` fails (not installed).
`pip install --dry-run botorch` resolves a GPU-oriented dependency chain
(torch 2.13 ~526MB + full nvidia CUDA toolkit wheels, several GB total) even
though this task is explicitly CPU-only with no GPU -- not "fast to set up"
as the brief's fallback condition requires. We therefore used the sklearn-GP
fallback throughout, labeled clearly as an approximate qParEGO analogue.

Evaluation-budget accounting: all three methods call
`HWSWCoDesignProblem.evaluate(x, rng)` exactly ONCE per unit of budget
consumed, so `problem.eval_count` is the single source of truth and is
asserted == 150 (or the smoke-test budget) after every run. For D&L-TS this
required disabling `run_dl`'s zeroth-order local-refine step (each refine
call costs 2 *extra* oracle calls beyond the paper's Algorithm 1 bookkeeping)
by setting `local_every = T + 1`, so exactly T = budget oracle calls are made
(one per outer bandit round) -- see dl/core.py:run_dl and dl/hwsw_proxy.py's
module docstring for the full reasoning.

Hypervolume ratio: HV_r(F) / prod|r_i - z_i|, with r (reference/nadir) and z
(ideal) derived from the elementwise min/max of the UNION of all methods' and
all seeds' observed (noiseless) objective vectors on this one fixed problem
instance -- i.e. computed once, after all runs, and reused for every run's
hv_ratio. This is documented explicitly here since the paper does not pin
down how r/z are chosen.
"""
import time
import csv
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# The isotropic-kernel GP hyperparameter search occasionally lands near a
# bound (expected with only a handful of points early in each BO run); this
# is harmless (sklearn just reports the fitted value is near a search bound)
# and floods stdout, so it is silenced here.
warnings.filterwarnings("ignore", category=UserWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

from dl_core import run_dl
from dl_hwsw_proxy import HWSWCoDesignProblem, TchebycheffDLAdapter, pareto_mask, hypervolume_ratio


# ------------------------- method runners -------------------------------

def run_dl_ts(problem, seed, T, n_weights=10):
    """One run_dl call per FIXED scalarization weight (n_weights of them,
    budget T split evenly across them), not one long run that resamples w
    every oracle call. See dl.hwsw_proxy.TchebycheffDLAdapter's docstring
    and BUGFIX_LOG.md Finding 4: a per-call-resampled weight breaks the
    stationarity assumption behind D&L's running per-position value
    estimates."""
    problem.reset_run()
    w_rng = np.random.default_rng(60_000 + seed)
    weights = [w_rng.dirichlet(np.ones(problem.n_obj)) for _ in range(n_weights)]
    T_per_weight = T // n_weights
    assert T_per_weight * n_weights == T, (T, n_weights)
    z_star = np.zeros(problem.n_obj)
    t0 = time.time()
    for wi, w in enumerate(weights):
        adapter = TchebycheffDLAdapter(problem, w=w, z_star=z_star, seed=seed * 1000 + wi)
        run_dl(n=problem.n, num_actions=problem.num_actions, reward_fn=adapter,
               K=4, T=T_per_weight, seed=seed * 1000 + wi, use_ts=True,
               local_every=T_per_weight + 1, mask_fn=adapter.mask_fn)
        z_star = adapter.z_star
    wall = time.time() - t0
    assert problem.eval_count == T, (problem.eval_count, T)
    F = np.array([h["f_true"] for h in problem.history])
    return F, wall


def _encode(problem, x):
    return np.asarray(x, dtype=float) / np.maximum(problem.levels - 1, 1)


def run_bo_qparego(problem, seed, T, n_init=20, n_cand_random=600):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    from scipy.stats import norm

    problem.reset_run()
    rng = np.random.default_rng(30_000 + seed)          # design-selection RNG
    w_rng = np.random.default_rng(35_000 + seed)         # scalarization-weight RNG
    noise_rng = np.random.default_rng(40_000 + seed)     # observation-noise RNG

    X_arch, Xc_arch, F_arch = [], [], []
    t0 = time.time()

    while problem.eval_count < T:
        if len(X_arch) < n_init:
            x = problem.random_x(rng)
        else:
            w = w_rng.dirichlet(np.ones(problem.n_obj))
            F_mat = np.array(F_arch)
            rho = 0.05
            s = np.max(w * F_mat, axis=1) + rho * np.sum(w * F_mat, axis=1)
            Xc_mat = np.array(Xc_arch)

            # Isotropic (single, not per-dim/ARD) length-scale: a 20-dim ARD
            # Matern kernel makes sklearn's L-BFGS hyperparameter search slow
            # (each of the ~130 BO iterations refits from scratch on a
            # growing dataset), which blew well past the 15-minute wall-clock
            # budget for a CPU-only, 150-eval-per-run experiment. Isotropic
            # length-scale is a standard, defensible simplification for a
            # from-scratch analogue and keeps each fit to a few ms.
            kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
                length_scale=1.0, length_scale_bounds=(1e-2, 1e1), nu=2.5
            ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-4, 1e-1))
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                           n_restarts_optimizer=0, alpha=1e-8)
            gp.fit(Xc_mat, s)
            best_s = s.max()

            cands = [problem.random_x(rng) for _ in range(n_cand_random)]
            top_idx = np.argsort(-s)[:5]
            for ti in top_idx:
                base = X_arch[ti]
                for _ in range(30):
                    cand = base.copy()
                    n_flip = int(rng.integers(1, 3))
                    idxs = rng.choice(problem.n, size=n_flip, replace=False)
                    for ii in idxs:
                        cand[ii] = int(rng.integers(0, problem.levels[ii]))
                    cands.append(cand)

            Cc = np.array([_encode(problem, c) for c in cands])
            mu, sigma = gp.predict(Cc, return_std=True)
            sigma = np.maximum(sigma, 1e-9)
            xi = 0.01
            Z = (mu - best_s - xi) / sigma
            ei = (mu - best_s - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
            x = cands[int(np.argmax(ei))]

        f = problem.evaluate(x, noise_rng)
        X_arch.append(np.array(x))
        Xc_arch.append(_encode(problem, x))
        F_arch.append(f)

    wall = time.time() - t0
    assert problem.eval_count == T, (problem.eval_count, T)
    F = np.array([h["f_true"] for h in problem.history])
    return F, wall


def run_nsga2(problem, seed, T, pop_size=10):
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.operators.repair.rounding import RoundingRepair

    problem.reset_run()
    noise_rng = np.random.default_rng(50_000 + seed)

    class Wrapped(Problem):
        def __init__(self):
            super().__init__(n_var=problem.n, n_obj=problem.n_obj,
                              xl=np.zeros(problem.n),
                              xu=(problem.levels - 1).astype(float))

        def _evaluate(self, X, out, *args, **kwargs):
            Fs = []
            for row in X:
                x = np.clip(np.rint(row).astype(int), 0, problem.levels - 1)
                f = problem.evaluate(x, noise_rng)
                Fs.append(-f)  # pymoo minimizes
            out["F"] = np.array(Fs)

    n_gen = T // pop_size
    assert n_gen * pop_size == T
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=1.0 / problem.n, eta=20, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )
    t0 = time.time()
    minimize(Wrapped(), algorithm, ("n_gen", n_gen), seed=seed, verbose=False)
    wall = time.time() - t0
    # pymoo's duplicate-elimination retries can occasionally push eval_count
    # slightly over T; trim history to the first T evaluations for a strict
    # apples-to-apples budget with the other two methods.
    if problem.eval_count > T:
        problem.history = problem.history[:T]
        problem.eval_count = T
    assert problem.eval_count == T, (problem.eval_count, T)
    F = np.array([h["f_true"] for h in problem.history])
    return F, wall


# ------------------------- experiment driver -----------------------------

METHODS = {
    "D&L-TS": run_dl_ts,
    "BO-qParEGO-analogue": run_bo_qparego,
    "NSGA-II": run_nsga2,
}


def run_all_seeds(problem, T, n_seeds, methods=METHODS):
    runs = {}  # (method, seed) -> (F, wall)
    for name, fn in methods.items():
        for seed in range(n_seeds):
            F, wall = fn(problem, seed, T)
            runs[(name, seed)] = (F, wall)
            print(f"  [{name:22s} seed={seed}] wallclock={wall:.2f}s "
                  f"evals={problem.eval_count} pareto_pts_raw={pareto_mask(F).sum()}")
    return runs


def summarize(runs, methods=METHODS):
    all_F = np.vstack([F for (F, _wall) in runs.values()])
    ref_r = all_F.min(axis=0)
    ideal_z = all_F.max(axis=0)
    print(f"\nDerived reference r (nadir, from union of all runs): {np.round(ref_r, 4)}")
    print(f"Derived ideal z (from union of all runs):            {np.round(ideal_z, 4)}")

    rows = []
    for name in methods:
        for seed in range(10 ** 6):
            key = (name, seed)
            if key not in runs:
                break
            F, wall = runs[key]
            mask = pareto_mask(F)
            front = F[mask]
            hv_ratio = hypervolume_ratio(front, ref_r, ideal_z)
            rows.append(dict(method=name, seed=seed, hv_ratio=hv_ratio,
                              final_pareto_size=int(mask.sum()), wallclock_sec=wall))
    return rows


def print_table(rows, methods=METHODS):
    print("\n=== Claim 6 synthetic-proxy summary (SYNTHETIC, not the real "
          "accelerator simulator) ===")
    print(f"{'method':22s} {'hv_ratio (mean+/-std)':24s} {'final_pareto_size':18s} "
          f"{'wallclock_sec':14s}")
    means = {}
    for name in methods:
        vals = [r["hv_ratio"] for r in rows if r["method"] == name]
        pareto_sizes = [r["final_pareto_size"] for r in rows if r["method"] == name]
        walls = [r["wallclock_sec"] for r in rows if r["method"] == name]
        if not vals:
            continue
        m, s = np.mean(vals), np.std(vals)
        means[name] = m
        print(f"{name:22s} {m:.3f} +/- {s:.3f}          "
              f"{np.mean(pareto_sizes):6.1f} +/- {np.std(pareto_sizes):4.1f}     "
              f"{np.mean(walls):6.2f}")

    if "D&L-TS" in means and "BO-qParEGO-analogue" in means:
        dl_v, bo_v = means["D&L-TS"], means["BO-qParEGO-analogue"]
        pct = (dl_v - bo_v) / bo_v * 100.0
        print(f"\nD&L-TS vs BO-qParEGO-analogue: {dl_v:.3f} vs {bo_v:.3f} "
              f"-> {pct:+.1f}% (paper claims ~+22%: MOBO-qParEGO 0.34 -> D&L-TS 0.372)")
    ranking = sorted(means.items(), key=lambda kv: -kv[1])
    print("Ranking (best -> worst hv_ratio):", " > ".join(f"{k}({v:.3f})" for k, v in ranking))


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "seed", "hv_ratio",
                                           "final_pareto_size", "wallclock_sec"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSaved -> {path}")


def main():
    t_start = time.time()

    print("=" * 70)
    print("SMOKE TEST: budget=20 evals, 2 seeds, all 3 methods")
    print("=" * 70)
    smoke_problem = HWSWCoDesignProblem(problem_seed=42)
    smoke_runs = run_all_seeds(smoke_problem, T=20, n_seeds=2)
    smoke_rows = summarize(smoke_runs)
    print_table(smoke_rows)
    print(f"\nSmoke test wallclock: {time.time() - t_start:.1f}s\n")

    print("=" * 70)
    print("FULL EXPERIMENT: budget=150 evals, 10 seeds, all 3 methods")
    print("=" * 70)
    t_full = time.time()
    problem = HWSWCoDesignProblem(problem_seed=42)  # same fixed landscape, fresh state
    runs = run_all_seeds(problem, T=150, n_seeds=10)
    rows = summarize(runs)
    print_table(rows)
    save_csv(rows, HERE / "claim6_hwsw_proxy.csv")
    print(f"\nFull experiment wallclock: {time.time() - t_full:.1f}s")
    print(f"Total wallclock (incl. smoke test): {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
