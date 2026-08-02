# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///
"""Claim 4: ablations (Table 3 / Table 8) -- removing the encoder, the Householder rotation, or
patching each degrades Fern relative to the base config, on Lorenz-63 base scenario. 3 seeds.
Run: .venv/bin/python run_claim4_ablations.py (writes claim4_results.json).
"""
import json
import time
import torch
from data_gen import get_dataset
from fern_lib import Fern, train_model, evaluate

SEEDS = [0, 1, 2]
N_STEPS = 6000
CONTEXT_LEN = 96
HORIZON_LEN = 96
PATCH_SIZE = 24

VARIANTS = {
    "base": dict(),
    "no_rotation": dict(no_rotation=True),
    "no_encoder": dict(no_encoder=True),
    "no_patching": dict(no_patching=True),
}


def main():
    d_by_seed = {}
    results = {v: [] for v in VARIANTS}
    t0 = time.time()
    for seed in SEEDS:
        d = get_dataset("lorenz63", n_steps=N_STEPS, context_len=CONTEXT_LEN,
                         horizon_len=HORIZON_LEN, shock="base", seed=seed)
        d_by_seed[seed] = d
        for variant, kwargs in VARIANTS.items():
            torch.manual_seed(seed)
            model = Fern(context_len=CONTEXT_LEN, horizon_len=HORIZON_LEN, patch_size=PATCH_SIZE,
                         n_reflections=8, dh=32, kenc=5, **kwargs)
            model = train_model(model, d["Xtr"], d["Ytr"], d["Xval"], d["Yval"], epochs=60)
            m = evaluate(model, d["Xte"], d["Yte"], d["train_std"])
            results[variant].append(m)
            print(f"[{time.time()-t0:6.1f}s] seed={seed} {variant:12s} mse={m['mse']:.4f} "
                  f"ept={m['ept']:.1f}", flush=True)
        with open("claim4_results.json", "w") as f:
            json.dump(results, f, indent=2)
    print(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
