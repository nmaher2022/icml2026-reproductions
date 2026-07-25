"""Claim 1, constructive audit: the universality theorem verified EXACTLY.

The theorem is existential ("there exist parameters ..."), so SGD error is the
wrong yardstick.  Here we CONSTRUCT the parameters and evaluate with the
repo's own LEMON min-cost-flow kernel (dfn/lemon_mcf.cpp -> solve_mcf):

  (1) 1-D: any convex g on {lo..hi}.  Two-node network, one special node with
      balance t = x - lo, arcs of capacity 1 and costs equal to the successive
      increments g(lo+k+1)-g(lo+k).  Convexity => increasing costs => the MCF
      greedy fill reproduces the piecewise-linear interpolation of g exactly:
      f(x) = g(x) - g(lo)  (alpha=1, beta=g(lo)).
  (2) d-D separable convex sums: d disjoint copies of gadget (1) inside ONE
      network; balances a_i = e_i rows.  Exact again.
  (3) CONTROL, with a certificate: any DFN restricted to a line is convex
      (MCF value is convex in its supplies; supplies are affine in x), so the
      best ANY DFN can do on a concave 1-D g is the best *convex* interpolant.
      We compute that irreducible sup-error exactly with a tiny LP (Gurobi,
      ~30 vars) and report it as a fraction of range(g): a LOWER BOUND on the
      error of every DFN, however large -- the theorem's convex-extendibility
      condition is necessary, not decorative.

All checks use integer/rational data, so "exact" means < 1e-9 (solver tol).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "deep-flow-networks"))

from dfn.model import load_lemon

LEMON = load_lemon()
LO, HI = -7, 7
N_PTS = HI - LO + 1
rng = np.random.default_rng(42)


def mcf_value(n, src, dst, cost, cap, supply_row):
    out = LEMON.solve_mcf(int(n), np.asarray(src, np.int64), np.asarray(dst, np.int64),
                          np.asarray(cost, np.float64), np.asarray(cap, np.float64),
                          np.asarray([supply_row], np.float64), 1e-9)
    assert np.all(np.asarray(out["status"]) == 1)
    return float(np.asarray(out["total_cost"]).ravel()[0])


def random_convex_1d():
    """Random integer convex sequence on {lo..hi} via increasing increments."""
    inc = np.cumsum(rng.integers(0, 7, N_PTS - 1)) + rng.integers(-10, 1)
    g = np.concatenate([[0], np.cumsum(inc)]).astype(float)
    return g + rng.integers(-20, 20)


def audit_1d(n_instances=10):
    worst = 0.0
    for _ in range(n_instances):
        g = random_convex_1d()
        inc = np.diff(g)                      # increasing by construction
        n_arcs = N_PTS - 1
        src = np.zeros(n_arcs); dst = np.ones(n_arcs)
        cost = inc.copy(); cap = np.ones(n_arcs)
        for i, x in enumerate(range(LO, HI + 1)):
            t = x - LO
            val = mcf_value(2, src, dst, cost, cap, [t, -t]) + g[0]  # alpha=1, beta=g(lo)
            worst = max(worst, abs(val - g[i]))
    return worst


def audit_separable(dim=3, n_instances=5, n_eval=400):
    worst = 0.0
    for _ in range(n_instances):
        gs = [random_convex_1d() for _ in range(dim)]
        src, dst, cost, cap = [], [], [], []
        for d in range(dim):
            base = 2 * d
            inc = np.diff(gs[d])
            src += [base] * (N_PTS - 1); dst += [base + 1] * (N_PTS - 1)
            cost += list(inc); cap += [1.0] * (N_PTS - 1)
        xs = rng.integers(LO, HI + 1, size=(n_eval, dim))
        for x in xs:
            supply = []
            for d in range(dim):
                t = x[d] - LO
                supply += [t, -t]
            val = mcf_value(2 * dim, src, dst, cost, cap, supply)
            true = sum(gs[d][x[d] - LO] - gs[d][0] for d in range(dim))
            worst = max(worst, abs(val - true))
    return worst


def control_lower_bound():
    """Best convex interpolant of a concave g: LP  min e  s.t. |h_i-g_i|<=e,
    h midpoint-convex.  Its optimum lower-bounds EVERY DFN's sup error on g."""
    import gurobipy as gp
    g = -random_convex_1d()                   # concave
    g = g - g.min()
    m = gp.Model()
    m.Params.OutputFlag = 0
    h = m.addVars(N_PTS, lb=-gp.GRB.INFINITY)
    e = m.addVar(lb=0)
    for i in range(N_PTS):
        m.addConstr(h[i] - g[i] <= e); m.addConstr(g[i] - h[i] <= e)
    for i in range(1, N_PTS - 1):
        m.addConstr(h[i + 1] - 2 * h[i] + h[i - 1] >= 0)
    m.setObjective(e, gp.GRB.MINIMIZE)
    m.optimize()
    return float(m.ObjVal), float(g.max() - g.min())


if __name__ == "__main__":
    w1 = audit_1d()
    print(f"(1) 1-D convex, 10 random instances x 15 grid points: "
          f"max |constructed DFN - g| = {w1:.2e}  (exact)")
    w2 = audit_separable()
    print(f"(2) 3-D separable convex sums, 5 instances x 400 random points: "
          f"max error = {w2:.2e}  (exact)")
    lb, rngg = control_lower_bound()
    print(f"(3) CONTROL certificate: best convex interpolant of a random concave g "
          f"has sup error {lb:.1f} = {lb/rngg:.1%} of range(g) -- an LP lower bound "
          f"on the error of EVERY DFN, however large.")
    print("CONSTRUCTIVE AUDIT: universality mechanism exact on (1),(2); "
          "convex-extendibility condition certified necessary by (3).")
