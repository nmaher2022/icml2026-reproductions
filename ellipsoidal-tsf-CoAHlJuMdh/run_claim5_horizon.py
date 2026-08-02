# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///
"""Claim 5: on Lorenz-63, Fern's error growth with increasing prediction horizon is slower than
DLinear's, and geometric accuracy (SWD) degrades more gracefully than pointwise MSE at long
horizons -- a toy-scale, qualitative check of "geometry persists past the pointwise-collapse
point" (paper: DLinear collapses by horizon 96, TimeMixer/PatchTST by 192, Fern holds pointwise
accuracy to 720/~6.5 Lyapunov times). Context length is fixed; horizon grows relative to it.
Run: .venv/bin/python run_claim5_horizon.py (writes claim5_results.json).
"""
import json
import time
import torch
from data_gen import get_dataset
from fern_lib import Fern, DLinear, train_model, evaluate

SEEDS = [0, 1, 2]
N_STEPS = 9000
CONTEXT_LEN = 96
HORIZONS = [24, 48, 96, 192]
PATCH_SIZE = 24


def main():
    results = {h: {"fern": [], "dlinear": []} for h in HORIZONS}
    t0 = time.time()
    for h in HORIZONS:
        for seed in SEEDS:
            d = get_dataset("lorenz63", n_steps=N_STEPS, context_len=CONTEXT_LEN, horizon_len=h,
                             shock="base", seed=seed)
            torch.manual_seed(seed)
            fern = Fern(context_len=CONTEXT_LEN, horizon_len=h, patch_size=PATCH_SIZE,
                        n_reflections=8, dh=32, kenc=5)
            fern = train_model(fern, d["Xtr"], d["Ytr"], d["Xval"], d["Yval"], epochs=60)
            fm = evaluate(fern, d["Xte"], d["Yte"], d["train_std"])
            results[h]["fern"].append(fm)

            torch.manual_seed(seed)
            dlin = DLinear(CONTEXT_LEN, h)
            dlin = train_model(dlin, d["Xtr"], d["Ytr"], d["Xval"], d["Yval"], epochs=60)
            dm = evaluate(dlin, d["Xte"], d["Yte"], d["train_std"])
            results[h]["dlinear"].append(dm)

            print(f"[{time.time()-t0:6.1f}s] h={h:3d} seed={seed} "
                  f"fern_mse={fm['mse']:.4f} dlin_mse={dm['mse']:.4f} "
                  f"fern_swd={fm['swd']:.4f} dlin_swd={dm['swd']:.4f}", flush=True)
        with open("claim5_results.json", "w") as f:
            json.dump(results, f, indent=2)
    print(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
