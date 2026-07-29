# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "scikit-learn", "pymoo"]
# ///
"""
Intermediate-n sanity sweep (Round 3 follow-up), NOT one of the paper's own
claims -- a diagnostic requested to test two competing explanations for the
Round 3 finding that fixing findings 8+9 makes D&L beat BO on Bi-Knapsack
(n=50/100) but not on Bi-TSP (n=20/50):
  (a) finding 11's SNR-collapse (per-position signal shrinks as n grows)
      predicts a smooth, continuous decline on BOTH domains as n grows --
      if so, n=16/32 should sit between the n=8 smoke numbers and the
      n=50/100 full-run numbers on both Bi-KP and Bi-TSP.
  (b) if the KP-vs-TSP asymmetry is instead a structural mismatch between
      D&L's mixture/local_refine machinery and permutation-style action
      spaces, we'd expect Bi-TSP to already be behind BO at small n too
      (discontinuity / domain-intrinsic, not an SNR-vs-n trend).

Reuses run_moco.py's own instance runners and D&L budget (dl_T=100, dl_K=4)
so results are directly comparable to results/claim4_5_moco.csv -- only
`n` and `n_instances` change (10 instances/cell here vs 15 in the full run,
to keep this diagnostic sweep fast).

Usage:
    uv run claim_midscale_sweep.py
"""
from __future__ import annotations

import time
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

from run_moco import (  # noqa: E402
    run_kp_instance, run_tsp_instance, process_instance, summarize, make_weights,
)

N_INSTANCES = 10
N_WEIGHTS = 12
DL_T, DL_K = 100, 4
WS_SWEEPS = 3
NSGA_POP, NSGA_GEN = 40, 60
BO_INIT, BO_ITER = 8, 25


def main():
    out_csv = HERE / "claim_midscale_sweep.csv"
    csv_rows = []
    t_start = time.perf_counter()
    weights = make_weights(N_WEIGHTS)

    for n in [16, 32]:
        for inst_i in range(N_INSTANCES):
            seed = 40000 + n * 1000 + inst_i
            t0 = time.perf_counter()
            results, maximize = run_kp_instance(
                n, seed, weights, DL_T, DL_K, WS_SWEEPS, NSGA_POP, NSGA_GEN, BO_INIT, BO_ITER)
            process_instance("Bi-KP", n, seed, results, maximize, csv_rows)
            print(f"[Bi-KP n={n} seed={seed}] done in {time.perf_counter()-t0:.1f}s "
                  f"(elapsed {time.perf_counter()-t_start:.1f}s)")

    for n in [16, 32]:
        for inst_i in range(N_INSTANCES):
            seed = 41000 + n * 1000 + inst_i
            t0 = time.perf_counter()
            results, maximize = run_tsp_instance(
                n, seed, weights, DL_T, DL_K, WS_SWEEPS, NSGA_POP, NSGA_GEN, BO_INIT, BO_ITER)
            process_instance("Bi-TSP", n, seed, results, maximize, csv_rows)
            print(f"[Bi-TSP n={n} seed={seed}] done in {time.perf_counter()-t0:.1f}s "
                  f"(elapsed {time.perf_counter()-t_start:.1f}s)")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "problem", "size", "method", "instance_seed",
            "hv_ratio", "num_nondominated", "wallclock_sec", "num_evals"])
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"\nWrote {len(csv_rows)} rows to {out_csv}")

    summarize(csv_rows)
    print(f"\nTOTAL WALLCLOCK: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
