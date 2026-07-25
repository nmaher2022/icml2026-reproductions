"""Smoke test 2: end-to-end DFN pipeline -- train, then integer-optimize the
trained surrogate with Gurobi (free size-limited license), and compare against
the true optimum of the ground-truth quadratic."""
from pathlib import Path
import sys, time

repo = Path(__file__).resolve().parent / "deep-flow-networks"
sys.path.insert(0, str(repo))

from dfn import DFN, fit
from dfn.datasets import make_quadratic_dataset
import numpy as np
from dfn.optimization import solve_dfn_ip, solve_true_opt
LB, UB = np.full(4, -10), np.full(4, 10)

X, y, gt = make_quadratic_dataset(K=500, dim=4, eigen_min=1, eigen_max=5, x_min=-10, x_max=10, seed=0)
model = DFN(input_dim=4, layer_sizes=[8, 24, 8], alpha=1e-2, beta=-2.0)
model, scaler, run = fit(model, X, y, epochs=100, batch_size=16, lr=1e-1, seed=0, verbose=False)
print('trained: test_mse_norm =', run['test_mse_norm'])

t0 = time.time()
x_dfn, obj_dfn, info = solve_dfn_ip(model, scaler, LB, UB, 0, time_limit=300, seed=0)
t_dfn = time.time() - t0
print('DFN MILP: x* =', x_dfn, 'surrogate obj =', obj_dfn, f'solve_s = {t_dfn:.2f}',
      'info:', {k: v for k, v in info.items() if k in ("status","n_vars","n_constrs","mip_gap")})

x_true, obj_true, _ = solve_true_opt(gt, LB, UB, 0, time_limit=300, seed=0)
print('TRUE opt: x* =', x_true, 'true obj =', obj_true)

from dfn.optimization import eval_true_obj
print('true value at DFN argmin:', eval_true_obj(gt, x_dfn), 'vs true optimum:', obj_true)
