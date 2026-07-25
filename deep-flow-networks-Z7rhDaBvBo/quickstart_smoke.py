"""Smoke test: exact code of notebooks/quickstart.ipynb cell 1, run headless.
Env: uv venv (python3.13, torch-cpu) + LEMON 1.3.1 headers (header-only build,
LEMON_INCLUDE_DIR) -- fallback for the official conda env (see logbook note).
"""
from pathlib import Path
import sys, time

repo = Path(__file__).resolve().parent / "deep-flow-networks"
sys.path.insert(0, str(repo))

from dfn import DFN, fit, predict
from dfn.datasets import make_quadratic_dataset

t0 = time.time()
X, y, _ = make_quadratic_dataset(K=500, dim=4, eigen_min=1, eigen_max=5, x_min=-10, x_max=10, seed=0)
model = DFN(input_dim=4, layer_sizes=[8, 100, 8], alpha=1e-2, beta=-2.0)
model, scaler, run = fit(model, X, y, epochs=100, batch_size=16, lr=1e-1, seed=0, verbose=True, log_every=10)
pred = predict(model, X[:5], scaler)
print('pred =', pred)
print('true =', y[:5])
print('best_epoch =', run['best_epoch'])
print('test_mse_norm =', run['test_mse_norm'])
print('test_mse_raw =', run['test_mse_raw'])
print(f'wall_s = {time.time()-t0:.1f}')
