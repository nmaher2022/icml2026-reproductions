"""Claim 1 numerical audit: DFNs approximate convex-extendible discrete functions.

Theorem (poster): for any convex g, finite grid I in dom g cap Z^n, eps>0, there
exist DFN params theta and scalars alpha,beta with max_{x in I} |g(x) - (alpha
f_theta(x) + beta)| <= eps.  The theorem is EXISTENTIAL; the practical audit
fits DFNs by the repo's own training loop and reports the achieved max error on
the full grid, at two model sizes (approximation power should grow with size).

Test functions on the exhaustive grid I = {-7..7}^2 (225 points):
  quad     - non-separable convex quadratic (x-c)^T Q (x-c), Q PSD       [convex]
  sepnorm  - separable convex sum_i w_i |x_i - c_i|^{1.5}                [convex]
  maxaff   - max of 6 affine functions (piecewise-linear convex)         [convex]
  CONTROL  - concave bump  -(x-c)^T Q (x-c): admits NO convex extension.

DFN + affine readout (alpha f + beta with alpha>0) is convex-extendible by
construction (min-cost-flow value is convex in the balances), so the control
CANNOT be represented -- a large control error is the audit detecting the
theorem's condition, not a training failure.  We also directly verify discrete
midpoint convexity of each trained DFN on random grid pairs.

Metric: max and mean |g - (alpha f + beta)| over ALL 1331 grid points, divided
by range(g) on the grid.  alpha,beta are absorbed by the training scaler (the
repo's fit() standardizes y; predict() undoes it -- exactly the alpha,beta of
the theorem, with alpha>0).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "deep-flow-networks"))

from dfn import DFN, fit, predict

DIM = 2
LO, HI = -7, 7
EPOCHS = 1000   # paper-strength training (run 1 with 300 epochs/batch 32 fit nothing:
                # max err ~0.7 for every function incl. control -- undertrained, kept
                # in the logbook as a negative result about SGD budget, not about DFNs)
SEED = 0

rng = np.random.default_rng(0)

grid_axes = [np.arange(LO, HI + 1)] * DIM
X = np.stack(np.meshgrid(*grid_axes, indexing="ij"), -1).reshape(-1, DIM).astype(np.float64)


def make_functions():
    c = rng.uniform(-2, 2, DIM)
    M = rng.normal(size=(DIM, DIM))
    Q = M @ M.T + 0.5 * np.eye(DIM)
    fns = {}
    fns["quad"] = ("convex", ((X - c) @ Q * (X - c)).sum(1))
    w = rng.uniform(0.5, 2.0, DIM)
    fns["sepnorm"] = ("convex", (w * np.abs(X - c) ** 1.5).sum(1))
    A = rng.normal(size=(6, DIM)) * 2
    b = rng.normal(size=6) * 3
    fns["maxaff"] = ("convex", (X @ A.T + b).max(1))
    fns["CONTROL_concave"] = ("not convex-extendible", -((X - c) @ Q * (X - c)).sum(1))
    return fns


def midpoint_convexity_violations(model, scaler, n_pairs=3000):
    """Check f((x+y)/2) <= (f(x)+f(y))/2 for random grid pairs with even sum."""
    idx = rng.integers(0, len(X), size=(n_pairs, 2))
    x, y = X[idx[:, 0]], X[idx[:, 1]]
    ok = ((x + y) % 2 == 0).all(1)
    x, y = x[ok], y[ok]
    m = (x + y) // 2
    fx = predict(model, torch.tensor(x, dtype=torch.float32), scaler).numpy()
    fy = predict(model, torch.tensor(y, dtype=torch.float32), scaler).numpy()
    fm = predict(model, torch.tensor(m, dtype=torch.float32), scaler).numpy()
    viol = fm - (fx + fy) / 2
    tol = 1e-6 * (np.abs(fx).mean() + 1)
    return int((viol > tol).sum()), len(x), float(viol.max())


def fit_and_score(name, kind, y, layer_sizes, alpha):
    t0 = time.time()
    model = DFN(input_dim=DIM, layer_sizes=layer_sizes, alpha=alpha, beta=-2.0)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    model, scaler, run = fit(model, Xt, yt, epochs=EPOCHS, batch_size=8, lr=1e-2,
                             seed=SEED, verbose=False, val_frac=0.05, test_frac=0.05)
    pred = predict(model, Xt, scaler).numpy()
    err = np.abs(pred - y)
    rng_g = y.max() - y.min()
    nviol, npairs, maxviol = midpoint_convexity_violations(model, scaler)
    print(f"[fit] {name:18s} kind={kind:22s} layers={str(layer_sizes):14s} "
          f"max_err/range={err.max()/rng_g:.4f} mean_err/range={err.mean()/rng_g:.4f} "
          f"midpoint_viol={nviol}/{npairs} (max {maxviol:.2e}) wall_s={time.time()-t0:.0f}",
          flush=True)
    return {"fn": name, "kind": kind, "layers": str(layer_sizes),
            "max_rel_err": err.max() / rng_g, "mean_rel_err": err.mean() / rng_g,
            "midpoint_violations": nviol, "midpoint_pairs": npairs}


if __name__ == "__main__":
    fns = make_functions()
    rows = []
    for name, (kind, y) in fns.items():
        sizes = [[4, 20, 4], [12, 60, 12]] if not name.startswith("CONTROL") else [[12, 60, 12]]
        for ls in sizes:
            rows.append(fit_and_score(name, kind, y, ls, alpha=5e-3))
    import csv
    with open(HERE / "claim1_universality.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nSUMMARY")
    conv = [r for r in rows if not r["fn"].startswith("CONTROL")]
    ctrl = [r for r in rows if r["fn"].startswith("CONTROL")]
    small = [r for r in conv if "20" in r["layers"]]
    big = [r for r in conv if "60" in r["layers"]]
    print(f"convex-extendible, small models: worst max_rel_err = {max(r['max_rel_err'] for r in small):.4f}")
    print(f"convex-extendible, large models: worst max_rel_err = {max(r['max_rel_err'] for r in big):.4f}")
    print(f"CONTROL (no convex extension):   max_rel_err = {ctrl[0]['max_rel_err']:.4f}")
    print(f"total midpoint-convexity violations across all fitted DFNs: "
          f"{sum(r['midpoint_violations'] for r in rows)}")
