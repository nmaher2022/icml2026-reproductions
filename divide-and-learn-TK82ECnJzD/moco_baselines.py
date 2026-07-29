"""
Baselines for the MOCO reproduction (Claims 4-5). These are OUR OWN
simplified implementations, not the paper's exact solvers -- we do not have
LKH, PMOCO's trained preference-conditioned hypernetwork, or BOPR's
probabilistic reparameterization available in a CPU-only, single-session
setting. Labelled plainly wherever used:

  * ws_heuristic_*   -- proxy for the paper's "specialized solver"
                        (WS-LKH / WS-DP): weighted-sum scalarization solved
                        by a cheap constructive + local-search heuristic,
                        swept over the same weight grid as D&L.
  * nsga2_*          -- pymoo NSGA-II, standard operators.
  * bo_*             -- proxy for qParEGO/qNEHVI: GP (sklearn) + UCB search
                        over a continuous propensity-vector relaxation of the
                        combinatorial domain, decoded into a feasible
                        solution (greedy-by-propensity for Knapsack,
                        argsort-permutation for TSP). NOT botorch's
                        qNEHVI/qParEGO -- explicitly an approximate stand-in.
  * PMOCO is skipped entirely (needs RL-trained hypernetwork over an
    instance *distribution*; out of scope for single-instance CPU reproduction).
"""
from __future__ import annotations

import numpy as np

from moco_domains import (
    BiKnapsackInstance, repair_knapsack, repair_knapsack_generic,
    BiTSPInstance, tour_length, repair_permutation,
)


# --------------------------------------------------------------------------
# WS-heuristic (specialized-solver proxy)
# --------------------------------------------------------------------------

def ws_heuristic_knapsack(inst: BiKnapsackInstance, weights, n_sweeps=3,
                           noise_sigma=0.02, seed=0):
    """Every accept/reject decision is driven by a NOISY score (fresh
    Gaussian draw per query, same noise_sigma D&L's oracle uses), not the
    true objective directly -- an earlier version let this (and the other
    two baselines) search against the noiseless objective while D&L was the
    only method searching under noise, which is an unfair advantage that
    grows with n (see BUGFIX_LOG.md Finding 3). The final ARCHIVED point is
    still the true, noiseless objective of whatever x the noisy search
    landed on -- exactly mirroring how D&L's own final_x is scored."""
    n = inst.n
    archive = []
    n_evals = 0
    rng = np.random.default_rng(70_000 + seed)
    for w0, w1 in weights:
        density = (w0 * inst.profit1 + w1 * inst.profit2) / inst.weight
        order = np.argsort(-density)
        sel = np.zeros(n, dtype=bool)
        total_w = 0.0
        for idx in order:
            if total_w + inst.weight[idx] <= inst.capacity:
                sel[idx] = True
                total_w += inst.weight[idx]

        norm = max(w0 * inst.p1_sum + w1 * inst.p2_sum, 1e-9)

        def noisy_score(s):
            true = (w0 * inst.profit1[s].sum() + w1 * inst.profit2[s].sum()) / norm
            return true + rng.normal(0, noise_sigma)

        cur_score = noisy_score(sel)
        n_evals += 1
        for _ in range(n_sweeps):
            improved = False
            for i in range(n):
                trial = sel.copy()
                trial[i] = not trial[i]
                trial = repair_knapsack(inst, trial, w0, w1)
                s = noisy_score(trial)
                n_evals += 1
                if s > cur_score:
                    sel, cur_score = trial, s
                    improved = True
            if not improved:
                break
        archive.append((float(inst.profit1[sel].sum()), float(inst.profit2[sel].sum())))
    return archive, n_evals


def ws_heuristic_tsp(inst: BiTSPInstance, weights, max_2opt_sweeps=15,
                      noise_sigma=0.02, seed=0):
    """See ws_heuristic_knapsack's docstring: 2-opt accept/reject now runs on
    a noisy (fresh Gaussian draw per query) normalized tour-length score
    instead of the exact edge-weight delta, so the search itself sees the
    same noise level D&L's oracle does (BUGFIX_LOG.md Finding 3). The
    initial nearest-neighbor construction stays a deterministic function of
    the raw distance matrix (no oracle query involved, same as before)."""
    n = inst.n
    archive = []
    n_evals = 0
    rng = np.random.default_rng(71_000 + seed)
    for w0, w1 in weights:
        Dw = w0 * inst.D1 + w1 * inst.D2

        def noisy_len(perm):
            true = tour_length(Dw, perm) / inst.len_bound
            return true + rng.normal(0, noise_sigma)

        # nearest-neighbor construction from city 0
        unvisited = set(range(1, n))
        perm = [0]
        cur = 0
        while unvisited:
            nxt = min(unvisited, key=lambda c: Dw[cur, c])
            perm.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        perm = np.array(perm)
        cur_score = noisy_len(perm)
        n_evals += 1
        # 2-opt local search (first improvement) under Dw, noisy acceptance
        for _ in range(max_2opt_sweeps):
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    trial = perm.copy()
                    trial[i + 1:j + 1] = trial[i + 1:j + 1][::-1]
                    s = noisy_len(trial)
                    n_evals += 1
                    if s < cur_score:
                        perm, cur_score = trial, s
                        improved = True
            if not improved:
                break
        archive.append((tour_length(inst.D1, perm), tour_length(inst.D2, perm)))
    return archive, n_evals


# --------------------------------------------------------------------------
# NSGA-II (pymoo)
# --------------------------------------------------------------------------

def nsga2_knapsack(inst: BiKnapsackInstance, pop_size=40, n_gen=60, seed=0,
                    noise_sigma=0.02):
    """Selection fitness is corrupted by Gaussian noise (scaled to each
    objective's magnitude, same relative noise level as D&L's oracle) so
    NSGA-II's search sees the same observation noise D&L does (BUGFIX_LOG.md
    Finding 3). The archived Pareto set is recomputed from the final
    population's TRUE (noiseless) objectives, mirroring how D&L's own
    archived point is the true objective of its noisy-search-selected x."""
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.sampling.rnd import BinaryRandomSampling
    from pymoo.operators.crossover.pntx import TwoPointCrossover
    from pymoo.operators.mutation.bitflip import BitflipMutation
    from pymoo.optimize import minimize

    noise_rng = np.random.default_rng(80_000 + seed)
    sigma1 = noise_sigma * max(inst.p1_sum, 1e-9)
    sigma2 = noise_sigma * max(inst.p2_sum, 1e-9)

    class KnapsackProblem(Problem):
        def __init__(self):
            super().__init__(n_var=inst.n, n_obj=2, n_constr=0, xl=0, xu=1)

        def _evaluate(self, X, out, *args, **kwargs):
            F = np.zeros((X.shape[0], 2))
            for i, row in enumerate(X):
                sel = repair_knapsack_generic(inst, row.astype(bool))
                f1 = inst.profit1[sel].sum() + noise_rng.normal(0, sigma1)
                f2 = inst.profit2[sel].sum() + noise_rng.normal(0, sigma2)
                F[i, 0] = -f1
                F[i, 1] = -f2
            out["F"] = F

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True,
    )
    res = minimize(KnapsackProblem(), algorithm, ("n_gen", n_gen), seed=seed, verbose=False)
    X = res.X if res.X.ndim == 2 else res.X.reshape(1, -1)
    archive = []
    for row in X:
        sel = repair_knapsack_generic(inst, row.astype(bool))
        archive.append((float(inst.profit1[sel].sum()), float(inst.profit2[sel].sum())))
    n_evals = int(res.algorithm.evaluator.n_eval)
    return archive, n_evals


def nsga2_tsp(inst: BiTSPInstance, pop_size=40, n_gen=40, seed=0, noise_sigma=0.02):
    """See nsga2_knapsack's docstring: noisy selection fitness, true-objective
    archive recomputed from the final population's decision vectors."""
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.sampling.rnd import PermutationRandomSampling
    from pymoo.operators.crossover.ox import OrderCrossover
    from pymoo.operators.mutation.inversion import InversionMutation
    from pymoo.optimize import minimize

    n = inst.n
    noise_rng = np.random.default_rng(81_000 + seed)
    sigma = noise_sigma * inst.len_bound

    class TSPProblem(Problem):
        def __init__(self):
            super().__init__(n_var=n, n_obj=2, xl=0, xu=n - 1)

        def _evaluate(self, X, out, *args, **kwargs):
            F = np.zeros((X.shape[0], 2))
            for i, row in enumerate(X):
                perm = row.astype(int)
                F[i, 0] = tour_length(inst.D1, perm) + noise_rng.normal(0, sigma)
                F[i, 1] = tour_length(inst.D2, perm) + noise_rng.normal(0, sigma)
            out["F"] = F

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=PermutationRandomSampling(),
        crossover=OrderCrossover(),
        mutation=InversionMutation(),
        eliminate_duplicates=True,
    )
    res = minimize(TSPProblem(), algorithm, ("n_gen", n_gen), seed=seed, verbose=False)
    X = res.X if res.X.ndim == 2 else res.X.reshape(1, -1)
    archive = []
    for row in X:
        perm = row.astype(int)
        archive.append((tour_length(inst.D1, perm), tour_length(inst.D2, perm)))
    n_evals = int(res.algorithm.evaluator.n_eval)
    return archive, n_evals


# --------------------------------------------------------------------------
# BO baseline (sklearn GP + UCB over a continuous relaxation) -- approximate
# stand-in for qParEGO/qNEHVI, NOT the real thing.
# --------------------------------------------------------------------------

def _gp_ucb_loop(n_dim, decode_fn, score_fn, n_init, n_iter, seed, cand_pool=250, beta=2.0):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel

    rng = np.random.default_rng(seed)
    # Fixed (not re-fit) kernel hyperparameters: sklearn's default marginal-
    # likelihood optimizer (L-BFGS over kernel hyperparameters on every
    # .fit() call) dominated wall-clock in calibration (15-25s/instance) and
    # is not central to the BO baseline's *search* behaviour here, so we
    # disable it (optimizer=None) and use fixed, reasonable length-scale /
    # noise hyperparameters. This keeps the GP-UCB loop fast and its cost
    # dominated by acquisition-candidate evaluation, as intended.
    kernel = Matern(nu=2.5, length_scale=1.0) + WhiteKernel(noise_level=1e-2)
    X, y = [], []
    best = None
    for _ in range(n_init):
        p = rng.uniform(0, 1, size=n_dim)
        decoded = decode_fn(p)
        s, obj = score_fn(decoded)
        X.append(p)
        y.append(s)
        if best is None or s > best[0]:
            best = (s, obj)
    for _ in range(n_iter):
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, optimizer=None, alpha=1e-6)
        gp.fit(np.array(X), np.array(y))
        cand = rng.uniform(0, 1, size=(cand_pool, n_dim))
        mu, std = gp.predict(cand, return_std=True)
        j = int(np.argmax(mu + beta * std))
        p = cand[j]
        decoded = decode_fn(p)
        s, obj = score_fn(decoded)
        X.append(p)
        y.append(s)
        if s > best[0]:
            best = (s, obj)
    return best[1], n_init + n_iter


def bo_knapsack(inst: BiKnapsackInstance, weights, n_init=8, n_iter=25, seed=0,
                 noise_sigma=0.02):
    """The GP is fit on a NOISY scalarized score (fresh Gaussian draw per
    query, same noise_sigma/normalization as D&L's oracle), and the
    incumbent is tracked by that same noisy score -- previously the GP saw
    the exact objective directly, an advantage D&L never had (BUGFIX_LOG.md
    Finding 3). The archived point's objective is still the TRUE (noiseless)
    value of whichever x the noisy search selected as best, exactly as for
    D&L's final_x."""
    archive = []
    n_evals = 0
    for wi, (w0, w1) in enumerate(weights):
        rng_noise = np.random.default_rng(82_000 + seed * 1000 + wi)
        norm = max(w0 * inst.p1_sum + w1 * inst.p2_sum, 1e-9)

        def decode(p):
            order = np.argsort(-p)
            sel = np.zeros(inst.n, dtype=bool)
            total_w = 0.0
            for idx in order:
                if total_w + inst.weight[idx] <= inst.capacity:
                    sel[idx] = True
                    total_w += inst.weight[idx]
            return sel

        def score(sel):
            f1 = inst.profit1[sel].sum()
            f2 = inst.profit2[sel].sum()
            s = (w0 * f1 + w1 * f2) / norm + rng_noise.normal(0, noise_sigma)
            return s, (f1, f2)

        obj, evals = _gp_ucb_loop(inst.n, decode, score, n_init, n_iter, seed=seed * 1000 + wi)
        archive.append((float(obj[0]), float(obj[1])))
        n_evals += evals
    return archive, n_evals


def bo_tsp(inst: BiTSPInstance, weights, n_init=8, n_iter=25, seed=0, noise_sigma=0.02):
    """See bo_knapsack's docstring."""
    archive = []
    n_evals = 0
    for wi, (w0, w1) in enumerate(weights):
        rng_noise = np.random.default_rng(83_000 + seed * 1000 + wi)

        def decode(p):
            return np.argsort(p)

        def score(perm):
            len1 = tour_length(inst.D1, perm)
            len2 = tour_length(inst.D2, perm)
            frac1, frac2 = len1 / inst.len_bound, len2 / inst.len_bound
            s = -(w0 * frac1 + w1 * frac2) + rng_noise.normal(0, noise_sigma)
            return s, (len1, len2)

        obj, evals = _gp_ucb_loop(inst.n, decode, score, n_init, n_iter, seed=seed * 1000 + wi)
        archive.append((float(obj[0]), float(obj[1])))
        n_evals += evals
    return archive, n_evals
