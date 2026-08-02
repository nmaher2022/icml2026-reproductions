# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///
"""Claim 2: Fern's Householder-factored SPD map reduces per-patch transport cost from
O(p^2) (dense SPD) to O(p) (linear in patch size p), Eq. 1-2 / Appendix A.3.2. This is a
structural/analytical claim, not compute-scale dependent, so checkable exactly (not just
toy-verified) by sweeping patch size p and comparing the two FLOP formulas from the actual
instantiated model, plus empirical wall-clock forward-pass timing as a secondary check.
Run: .venv/bin/python run_claim2_complexity.py (writes claim2_results.json, seconds).
"""
import json
import time
import numpy as np
import torch
from fern_lib import Fern

PATCH_SIZES = [8, 16, 24, 48, 96, 192, 384]
DH = 32
R = 8
G = 1  # single patch per horizon so patch_size sweep is unconfounded by patch count


def main():
    results = []
    for p in PATCH_SIZES:
        horizon = p * G
        model = Fern(context_len=horizon, horizon_len=horizon, patch_size=p, n_reflections=R, dh=DH,
                     kenc=1)
        report = model.param_flop_report()

        # empirical wall-clock forward-pass timing (secondary, illustrative check)
        x = torch.randn(64, horizon)
        with torch.no_grad():
            for _ in range(3):
                model(x)  # warmup
            t0 = time.time()
            for _ in range(20):
                model(x)
            elapsed = (time.time() - t0) / 20
        report["patch_size_swept"] = p
        report["forward_pass_wallclock_s"] = elapsed
        results.append(report)
        print(f"p={p:4d} fern_head_flops={report['fern_head_flops']:7d} "
              f"dense_spd_flops={report['dense_spd_flops']:7d} "
              f"dense/fern ratio={report['ratio']:.2f} wallclock={elapsed*1000:.3f}ms")

    with open("claim2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
