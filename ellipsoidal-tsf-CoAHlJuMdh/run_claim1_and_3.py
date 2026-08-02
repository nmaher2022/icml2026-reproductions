# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///
"""Claims 1 (nonstationary robustness, up to 790x MSE gap vs baselines) and 3 (Effective
Prediction Time) -- toy scale. Trains Fern + DLinear on Lorenz-63/Roessler/Chua base and
param-shock scenarios, 3 seeds each, reports MSE/WD/SWD/EPT. See PAPER_BRIEFING.md Claims 1 & 3.
Run: .venv/bin/python run_claim1_and_3.py (writes claim1_3_results.json, ~minutes on CPU).
"""
import json
import time
import numpy as np
import torch
from data_gen import get_dataset
from fern_lib import Fern, DLinear, train_model, evaluate

SYSTEMS = ["lorenz63", "rossler", "chua"]
SHOCKS = ["base", "param"]
SEEDS = [0, 1, 2]
N_STEPS = 6000
CONTEXT_LEN = 96
HORIZON_LEN = 96
PATCH_SIZE = 24


def run_one(system, shock, seed):
    torch.manual_seed(seed)
    d = get_dataset(system, n_steps=N_STEPS, context_len=CONTEXT_LEN, horizon_len=HORIZON_LEN,
                     shock=shock, seed=seed)
    out = {}
    fern = Fern(context_len=CONTEXT_LEN, horizon_len=HORIZON_LEN, patch_size=PATCH_SIZE,
                n_reflections=8, dh=32, kenc=5)
    fern = train_model(fern, d["Xtr"], d["Ytr"], d["Xval"], d["Yval"], epochs=60)
    out["fern"] = evaluate(fern, d["Xte"], d["Yte"], d["train_std"])

    torch.manual_seed(seed)
    dlin = DLinear(CONTEXT_LEN, HORIZON_LEN)
    dlin = train_model(dlin, d["Xtr"], d["Ytr"], d["Xval"], d["Yval"], epochs=60)
    out["dlinear"] = evaluate(dlin, d["Xte"], d["Yte"], d["train_std"])
    return out


def main():
    results = {}
    t0 = time.time()
    for system in SYSTEMS:
        for shock in SHOCKS:
            key = f"{system}_{shock}"
            results[key] = []
            for seed in SEEDS:
                r = run_one(system, shock, seed)
                results[key].append(r)
                print(f"[{time.time()-t0:6.1f}s] {key} seed={seed} "
                      f"fern_mse={r['fern']['mse']:.4f} dlin_mse={r['dlinear']['mse']:.4f} "
                      f"fern_ept={r['fern']['ept']:.1f} dlin_ept={r['dlinear']['ept']:.1f}",
                      flush=True)
            with open("claim1_3_results.json", "w") as f:
                json.dump(results, f, indent=2)
    print(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
