"""
Bi-objective combinatorial domains for the D&L MOCO reproduction (Claims 4-5):
  - Bi-Knapsack (Bi-KP): Zitzler & Thiele (1999)-style bi-objective 0/1 knapsack.
  - Bi-TSP: two independent Euclidean distance matrices over the same city set.

Each domain exposes a "ScalarizedReward" object implementing the reward_fn
protocol required by dl.core.run_dl:
    __call__(x)     -> noisy scalarized reward in ~[0,1]
    expected(x)     -> noiseless scalarized reward
    random_action(i, rng) -> a random valid action at position i

Infeasibility (knapsack capacity, TSP permutation validity) is handled by
*repairing* inside the oracle (both __call__/expected and the extra
.objectives(x) helper used by our experiment driver to pull the raw
bi-objective vector for the Pareto archive), so run_dl never needs to know
about feasibility -- it only ever sees bounded scalar rewards.

These are single-instance, single-scalarization-weight environments; D&L's
outer loop (Algorithm 1, "for w in W") re-instantiates one of these per
weight vector, sharing the same underlying instance data (profits/weights or
coordinates), and the driver script collects run_dl's final_x from each into
a Pareto archive.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Bi-Knapsack
# --------------------------------------------------------------------------

class BiKnapsackInstance:
    """Random bi-objective 0/1 knapsack instance (profits1, profits2, weights
    ~ integers in [1,100], capacity = 0.5 * sum(weights)), Zitzler & Thiele
    (1999) style construction commonly used in bi-KP MOCO benchmarks.
    """

    def __init__(self, n, seed):
        rng = np.random.default_rng(seed)
        self.n = n
        self.profit1 = rng.integers(1, 101, size=n).astype(float)
        self.profit2 = rng.integers(1, 101, size=n).astype(float)
        self.weight = rng.integers(1, 101, size=n).astype(float)
        self.capacity = 0.5 * self.weight.sum()
        # loose per-weight-vector upper bounds for reward normalization
        self.p1_sum = self.profit1.sum()
        self.p2_sum = self.profit2.sum()


def repair_knapsack(inst: BiKnapsackInstance, x, density_w0, density_w1):
    """Remove items (in increasing profit-density order w.r.t. the weight
    vector (density_w0, density_w1)) until the knapsack is feasible. Returns
    a NEW boolean array; does not mutate x.
    """
    sel = x.astype(bool).copy()
    total_w = inst.weight[sel].sum()
    if total_w <= inst.capacity:
        return sel
    idx = np.flatnonzero(sel)
    density = (density_w0 * inst.profit1[idx] + density_w1 * inst.profit2[idx]) / inst.weight[idx]
    order = idx[np.argsort(density)]  # worst (lowest density) first
    for i in order:
        if total_w <= inst.capacity:
            break
        sel[i] = False
        total_w -= inst.weight[i]
    return sel


def repair_knapsack_generic(inst: BiKnapsackInstance, x):
    """Weight-agnostic repair (density w.r.t. equal-weighted profits) used by
    baselines (NSGA-II) that don't carry a scalarization weight.
    """
    return repair_knapsack(inst, x, 0.5, 0.5)


class KnapsackScalarizedReward:
    """reward_fn protocol implementation for one scalarization weight w=(w0,w1)."""

    def __init__(self, inst: BiKnapsackInstance, w, noise_sigma=0.02, seed=0):
        self.inst = inst
        self.w0, self.w1 = float(w[0]), float(w[1])
        self.noise_sigma = noise_sigma
        self.rng = np.random.default_rng(seed)
        # loose upper bound for normalization (capacity-unaware, so always an
        # over-estimate -> normalized reward stays in [0,1])
        self.norm = max(self.w0 * inst.p1_sum + self.w1 * inst.p2_sum, 1e-9)

    def _scalar(self, x):
        sel = repair_knapsack(self.inst, x, self.w0, self.w1)
        f1 = self.inst.profit1[sel].sum()
        f2 = self.inst.profit2[sel].sum()
        r = (self.w0 * f1 + self.w1 * f2) / self.norm
        return float(np.clip(r, 0.0, 1.0)), f1, f2

    def expected(self, x):
        r, _, _ = self._scalar(x)
        return r

    def __call__(self, x):
        r, _, _ = self._scalar(x)
        r += self.rng.normal(0, self.noise_sigma)
        return float(np.clip(r, 0.0, 1.0))

    def random_action(self, i, rng):
        return int(rng.integers(0, 2))

    def objectives(self, x):
        sel = repair_knapsack(self.inst, x, self.w0, self.w1)
        return float(self.inst.profit1[sel].sum()), float(self.inst.profit2[sel].sum())


# --------------------------------------------------------------------------
# Bi-TSP
# --------------------------------------------------------------------------

class BiTSPInstance:
    """n cities, two independent random Euclidean coordinate sets in
    [0,1]^2 -> two independent distance matrices (standard bi-TSP
    construction: same city *index* set, two unrelated cost structures).
    """

    def __init__(self, n, seed):
        rng = np.random.default_rng(seed)
        self.n = n
        self.coords1 = rng.uniform(0, 1, size=(n, 2))
        self.coords2 = rng.uniform(0, 1, size=(n, 2))
        self.D1 = _dist_matrix(self.coords1)
        self.D2 = _dist_matrix(self.coords2)
        # loose upper bound on a closed tour length: n edges, each <= sqrt(2)
        self.len_bound = n * np.sqrt(2.0)


def _dist_matrix(coords):
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def repair_permutation(n, x):
    """Turn a possibly-invalid vector of city ids (duplicates/missing, as can
    transiently occur during D&L's position-wise bandit search) into a valid
    permutation of 0..n-1. First occurrence of each city (scanning left to
    right) is kept; any position holding a duplicate or out-of-range value is
    reassigned the smallest not-yet-used city id. Deterministic given x.
    """
    x = np.asarray(x, dtype=int)
    seen = np.zeros(n, dtype=bool)
    out = np.empty(n, dtype=int)
    dup_positions = []
    for i in range(n):
        v = x[i]
        if 0 <= v < n and not seen[v]:
            out[i] = v
            seen[v] = True
        else:
            dup_positions.append(i)
    missing = np.flatnonzero(~seen)
    for pos, city in zip(dup_positions, missing):
        out[pos] = city
    return out


def tour_length(D, perm):
    return float(D[perm, np.roll(perm, -1)].sum())


class TSPScalarizedReward:
    """reward_fn protocol implementation for one scalarization weight w=(w0,w1)."""

    def __init__(self, inst: BiTSPInstance, w, noise_sigma=0.02, seed=0):
        self.inst = inst
        self.n = inst.n
        self.w0, self.w1 = float(w[0]), float(w[1])
        self.noise_sigma = noise_sigma
        self.rng = np.random.default_rng(seed)

    def _scalar(self, x):
        perm = repair_permutation(self.n, x)
        len1 = tour_length(self.inst.D1, perm)
        len2 = tour_length(self.inst.D2, perm)
        frac1 = len1 / self.inst.len_bound
        frac2 = len2 / self.inst.len_bound
        r = 1.0 - (self.w0 * frac1 + self.w1 * frac2)
        return float(np.clip(r, 0.0, 1.0)), len1, len2

    def expected(self, x):
        r, _, _ = self._scalar(x)
        return r

    def __call__(self, x):
        r, _, _ = self._scalar(x)
        r += self.rng.normal(0, self.noise_sigma)
        return float(np.clip(r, 0.0, 1.0))

    def random_action(self, i, rng):
        return int(rng.integers(0, self.n))

    def mask_fn(self, i, x):
        mask = np.ones(self.n, dtype=bool)
        others = np.delete(np.arange(self.n), i)
        used = x[others]
        used = used[(used >= 0) & (used < self.n)]
        mask[used] = False
        if not mask.any():
            mask[x[i] if 0 <= x[i] < self.n else 0] = True
        return mask

    def objectives(self, x):
        perm = repair_permutation(self.n, x)
        return tour_length(self.inst.D1, perm), tour_length(self.inst.D2, perm)


# --------------------------------------------------------------------------
# Oracle-call counting wrapper (shared, used for D&L and WS-heuristic; NSGA-II
# and BO count evaluations by construction in their own loops)
# --------------------------------------------------------------------------

class EvalCounter:
    """Wraps a reward_fn, counting every call to __call__ as one
    'reward-oracle evaluation' (our CPU-only proxy for computation -- we have
    no FLOP counters). random_action calls are NOT counted (no oracle query).

    expected() is NOT counted either: run_dl only calls it to log a noiseless
    regret-tracking trace for our own Claim 1/2 audits (dl.core.run_dl's
    exp_rewards return value) -- a real deployment of D&L never needs the
    noiseless value, so counting it here would double the reported budget
    for D&L/D&L-TS relative to what the algorithm actually needs, and
    silently sabotage the Claim 4 compute-efficiency comparison against BO/
    NSGA-II (which only ever see __call__).
    """

    def __init__(self, inner):
        self.inner = inner
        self.n_evals = 0

    def __call__(self, x):
        self.n_evals += 1
        return self.inner(x)

    def expected(self, x):
        return self.inner.expected(x)

    def random_action(self, i, rng):
        return self.inner.random_action(i, rng)

    def objectives(self, x):
        return self.inner.objectives(x)
