"""
Synthetic proxy for Claim 6 (paper's Table 2): the 4-objective hardware-
software co-design task for AI accelerators (citing Dutta et al. 2022's
accelerator simulator). We do NOT have access to that simulator (proprietary/
specialized infra), so this module implements a clearly-labeled SYNTHETIC
STAND-IN with the same qualitative shape:

  - ~20-dimensional, mixed-integer design space (all dims collapsed to
    integer/categorical here -- a defensible simplification of "mixed-integer"
    explicitly sanctioned by the reproduction brief: abstract integer-valued
    dims with small cardinalities, standing in for cache-size choices,
    pipeline depth, numerics/precision options, etc.)
  - 4 conflicting objectives, each a fixed random nonlinear/nonconvex function
    of the design vector (mixture of 3 shifted "quadratic-bowl" Gaussian
    lobes + a sinusoidal ripple term), with a DIFFERENT random relevance
    mask/projection per objective so that no single design optimizes all 4
    at once (a nontrivial, conflicting Pareto front).
  - mild iid Gaussian observation noise on every evaluation, emulating the
    "intrinsic stochasticity from hardware-level variance" mentioned in the
    paper for this task.
  - objectives normalized (via Monte-Carlo-fitted min/max) and clipped to
    [0, 1], convention: HIGHER is better for all 4 objectives.

NOTE on configuration-space size: the paper cites ~2.2e5 total configurations
for its real ~20-dim mixed-integer space, which implies an average per-dim
cardinality of ~1.9 (2.2e5^(1/20)). That is incompatible with also using
"3-8 levels per dim" as suggested in the reproduction brief (3^20 alone is
~3.5e9). We prioritize dimensionality (20) and a realistic per-dim cardinality
range (3-6, i.e. genuinely categorical-feeling choices) over exactly matching
the raw configuration count -- none of the 3 methods under test enumerate the
space anyway (all are model-free / surrogate-based search over a black box),
so the combinatorial-size mismatch does not materially affect what's being
tested (does D&L-TS's bandit-scalarization approach beat a qParEGO-style BO
baseline under a tight 150-eval budget on a conflicting 4-objective
mixed-integer black box). This is flagged again in the experiment report.
"""
from __future__ import annotations

import numpy as np


class HWSWCoDesignProblem:
    """Fixed synthetic 4-objective, n-dim integer/categorical black box.

    One instance = one fixed "problem landscape" (like one fixed real
    accelerator co-design task). All methods and all algorithm-seeds in the
    experiment share the SAME instance (same landscape), matching the paper's
    setup of a fixed benchmark task with seeds only varying algorithm-level
    randomness/init -- not the landscape itself.
    """

    def __init__(self, n=20, n_obj=4, n_bowls=3, level_lo=3, level_hi=6,
                 noise_sigma=0.03, problem_seed=42, norm_samples=4000):
        self.n = n
        self.n_obj = n_obj
        rng = np.random.default_rng(problem_seed)
        self.levels = rng.integers(level_lo, level_hi + 1, size=n)  # per-dim cardinality
        self.num_actions = int(self.levels.max())
        self.noise_sigma = noise_sigma

        # Per-objective random landscape parameters (fixed for this instance).
        self.centers = rng.uniform(0.15, 0.85, size=(n_obj, n_bowls, n))
        self.relevance = rng.uniform(0.4, 1.4, size=(n_obj, n_bowls, n))
        self.widths = rng.uniform(0.18, 0.42, size=(n_obj, n_bowls))
        self.bowl_amp = rng.uniform(0.6, 1.4, size=(n_obj, n_bowls))
        self.ripple_freq = rng.integers(1, 4, size=(n_obj, n))
        self.ripple_phase = rng.uniform(0, 2 * np.pi, size=(n_obj, n))
        self.ripple_w = rng.uniform(0.3, 1.0, size=(n_obj, n))
        self.ripple_amp = rng.uniform(0.15, 0.35, size=n_obj)

        self.eval_count = 0
        self.history = []  # list of dicts: x, f_true (noiseless), f_noisy

        self._fit_normalization(rng, norm_samples)

    # ---- landscape ----------------------------------------------------
    def _decode(self, x):
        """Integer level indices -> continuous [0,1]^n coordinates."""
        return np.asarray(x, dtype=float) / np.maximum(self.levels - 1, 1)

    def _raw_objectives(self, xc):
        obj = np.zeros(self.n_obj)
        for o in range(self.n_obj):
            bowl_val = 0.0
            for j in range(self.centers.shape[1]):
                diff = (xc - self.centers[o, j]) * self.relevance[o, j]
                d2 = np.sum(diff ** 2)
                bowl_val += self.bowl_amp[o, j] * np.exp(-d2 / (2 * self.widths[o, j] ** 2))
            ripple = np.sum(
                self.ripple_w[o] * np.sin(2 * np.pi * self.ripple_freq[o] * xc + self.ripple_phase[o])
            ) / self.n
            obj[o] = bowl_val + self.ripple_amp[o] * ripple
        return obj

    def _fit_normalization(self, rng, n_samples):
        samples = np.zeros((n_samples, self.n_obj))
        for i in range(n_samples):
            idx = rng.integers(0, self.levels)
            xc = idx / np.maximum(self.levels - 1, 1)
            samples[i] = self._raw_objectives(xc)
        self.obj_min = samples.min(axis=0)
        self.obj_max = samples.max(axis=0)

    def expected(self, x):
        """Noiseless, normalized objective vector in [0,1]^n_obj (higher=better)."""
        xc = self._decode(x)
        raw = self._raw_objectives(xc)
        norm = (raw - self.obj_min) / np.maximum(self.obj_max - self.obj_min, 1e-9)
        return np.clip(norm, 0.0, 1.0)

    # ---- oracle (budget-consuming) -------------------------------------
    def evaluate(self, x, rng):
        """One noisy black-box evaluation. Consumes 1 unit of eval budget."""
        self.eval_count += 1
        f_true = self.expected(x)
        f_noisy = np.clip(f_true + rng.normal(0, self.noise_sigma, size=self.n_obj), 0.0, 1.0)
        self.history.append({"x": np.array(x, dtype=int), "f_true": f_true, "f_noisy": f_noisy})
        return f_noisy

    def random_x(self, rng):
        return np.array([int(rng.integers(0, L)) for L in self.levels])

    def reset_run(self):
        """Call between independent (method, seed) runs sharing this instance."""
        self.eval_count = 0
        self.history = []


class TchebycheffDLAdapter:
    """Wraps HWSWCoDesignProblem as the scalar reward_fn interface required by
    dl.core.run_dl: __call__(x)->noisy scalar reward in [0,1], expected(x)->
    noiseless scalar, random_action(i,rng)->int.

    Implements the paper's "Tchebycheff for HW-SF" (Sec 6.2) scalarization:
        g_tch(x | w, z*) = max_i w_i * |f_i(x) - z*_i|
    with z* the per-objective ideal point, estimated online as the running
    elementwise max of all noisy observations seen so far (z* only grows,
    consistent with "estimated online" -- no lookahead/cheating: at call time
    we only use observations already made). reward = 1 - g_tch, which stays
    in [0,1] since all f_i, z*_i in [0,1] and w on the simplex.

    The scalarization weight `w` is FIXED for the lifetime of one adapter
    instance (passed in at construction), not resampled per oracle call.
    An earlier version of this adapter resampled w ~ Dirichlet(1,1,1,1) on
    every single call, mirroring ParEGO's per-iteration schedule -- but
    D&L's per-position value estimates (state.V in dl/core.py) are a
    running (Welford) average that implicitly assumes a STATIONARY reward
    target for each (position, action) pair. Resampling w every call means
    the "true" reward for the same (i, a) pair changes randomly from round
    to round, so the running average silently mixes rewards computed
    against incompatible objectives -- breaking every expert's (UCB, EXP3,
    FTRL, TS) core assumption. This was confirmed as the mechanism behind
    Claim 6's D&L-TS win-at-T=20/loss-at-T=150 reversal despite n staying
    fixed at 20 for both budgets (see BUGFIX_LOG.md, Finding 4).

    To still trace a spread across the Pareto front under a fixed total
    eval budget, the driver script (claim6_hwsw_proxy.py) now constructs
    one adapter (one fixed weight) per outer scalarization weight and calls
    run_dl once per weight, splitting the total budget across weights --
    mirroring the same "one run_dl call per scalarization weight" pattern
    already used for Claims 4-5 in run_moco.py. `z_star` is carried forward
    across weights (it only grows, and represents the best per-objective
    values seen so far across the whole experiment, not just one weight's
    sub-run).
    """

    def __init__(self, problem: HWSWCoDesignProblem, w, z_star=None, seed=0):
        self.problem = problem
        self.n = problem.n
        self.num_actions = problem.num_actions
        self.noise_rng = np.random.default_rng(10_000 + seed)
        self.w = np.asarray(w, dtype=float)
        self.z_star = np.zeros(problem.n_obj) if z_star is None else np.array(z_star, dtype=float)

    def mask_fn(self, i, x):
        m = np.zeros(self.num_actions, dtype=bool)
        m[: self.problem.levels[i]] = True
        return m

    def random_action(self, i, rng):
        return int(rng.integers(0, self.problem.levels[i]))

    def __call__(self, x):
        f = self.problem.evaluate(x, self.noise_rng)
        self.z_star = np.maximum(self.z_star, f)
        g = np.max(self.w * np.abs(f - self.z_star))
        return float(np.clip(1.0 - g, 0.0, 1.0))

    def expected(self, x):
        f = self.problem.expected(x)
        g = np.max(self.w * np.abs(f - self.z_star))
        return float(np.clip(1.0 - g, 0.0, 1.0))


# ---- shared metric helpers (used by the claim-6 driver script) ---------

def pareto_mask(F):
    """Boolean mask of non-dominated rows of F (n_points, n_obj), MAXIMIZING
    all objectives."""
    F = np.asarray(F)
    n = F.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        ge = np.all(F >= F[i], axis=1)
        gt = np.any(F > F[i], axis=1)
        dominators = ge & gt
        dominators[i] = False
        if np.any(dominators):
            dominated[i] = True
    return ~dominated


def hypervolume_ratio(front_F, ref_r, ideal_z):
    """HV_r(F) / prod|r_i - z_i|, matching the paper's normalized-HV formula.
    front_F: (k, n_obj) non-dominated points (maximize convention).
    ref_r: reference point (worst per-objective corner, elementwise <= all
    observed points, in maximize convention -- a lower bound).
    ideal_z: ideal point (best per-objective corner, elementwise >= all
    observed points).
    """
    from pymoo.indicators.hv import HV

    front_F = np.asarray(front_F, dtype=float)
    ref_r = np.asarray(ref_r, dtype=float)
    ideal_z = np.asarray(ideal_z, dtype=float)

    denom = np.prod(np.maximum(ideal_z - ref_r, 1e-12))
    # pymoo HV is defined for MINIMIZATION with a reference point that is
    # dominated by (i.e. worse than) every point. Negate to convert our
    # maximize-convention front into pymoo's minimize convention: the
    # maximize-reference r (a lower bound) becomes -r (an upper bound, i.e.
    # a valid minimize-convention reference point).
    hv_calc = HV(ref_point=-ref_r)
    hv = hv_calc(-front_F)
    return float(hv / denom)
