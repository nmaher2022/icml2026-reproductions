# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "scikit-learn", "pymoo"]
# ///
"""
Driver for the D&L MOCO reproduction, Claims 4-5 (arXiv:2602.11346).

Builds Bi-Knapsack (primary) and Bi-TSP (secondary, time-permitting) Pareto
fronts with:
  - D&L        (dl.core.run_dl, use_ts=False), one run per scalarization weight
  - D&L-TS     (dl.core.run_dl, use_ts=True)
  - WS-heuristic (our proxy for the paper's specialized WS-LKH/DP solver)
  - NSGA-II    (pymoo)
  - BO         (our sklearn-GP+UCB proxy for qParEGO/qNEHVI; NOT the real thing)
  PMOCO is explicitly skipped (needs an RL-trained preference-conditioned
  hypernetwork over an instance *distribution* -- out of scope here).

Per (problem, size, method, instance seed) we record: hv_ratio (paper's
normalized-hypervolume formula, HV_r(F)/prod|r_i-z_i|, with r/z taken from
the union of all methods' raw objective values observed on that instance --
we do NOT have the paper's own reference points), num_nondominated,
wallclock_sec, and num_evals (reward-oracle calls -- our CPU-only proxy for
"computation", since we have no FLOP counters).

Usage:
    uv run run_moco.py --smoke
    uv run run_moco.py
"""
from __future__ import annotations

import csv
import time
import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")  # silence sklearn GP hyperparameter-bound
                                    # ConvergenceWarnings from the BO baseline
                                    # (cosmetic only; does not affect results)

HERE = Path(__file__).resolve().parent

from dl_core import run_dl  # noqa: E402
from moco_domains import (  # noqa: E402
    BiKnapsackInstance, KnapsackScalarizedReward, EvalCounter,
    BiTSPInstance, TSPScalarizedReward,
)
from moco_baselines import (  # noqa: E402
    ws_heuristic_knapsack, ws_heuristic_tsp,
    nsga2_knapsack, nsga2_tsp,
    bo_knapsack, bo_tsp,
)
from moco_metrics import nadir_ideal, hv_ratio  # noqa: E402


def make_weights(n_weights):
    w0 = np.linspace(0.0, 1.0, n_weights)
    return [(float(a), float(1 - a)) for a in w0]


# --------------------------------------------------------------------------
# Bi-Knapsack instance runner
# --------------------------------------------------------------------------

def run_kp_instance(n, seed, weights, dl_T, dl_K, ws_sweeps,
                     nsga_pop, nsga_gen, bo_init, bo_iter):
    inst = BiKnapsackInstance(n, seed=seed)
    results = {}

    for method, use_ts in [("D&L", False), ("D&L-TS", True)]:
        t0 = time.perf_counter()
        archive, n_evals = [], 0
        for wi, w in enumerate(weights):
            base = KnapsackScalarizedReward(inst, w, noise_sigma=0.02, seed=seed * 100 + wi)
            wrapped = EvalCounter(base)
            out = run_dl(n=inst.n, num_actions=2, reward_fn=wrapped, K=dl_K, T=dl_T,
                          seed=seed * 1000 + wi, use_ts=use_ts, mask_fn=None)
            f1, f2 = base.objectives(out["final_x"])
            archive.append((f1, f2))
            n_evals += wrapped.n_evals
        results[method] = dict(archive=archive, n_evals=n_evals,
                                wallclock=time.perf_counter() - t0)

    t0 = time.perf_counter()
    archive, n_evals = ws_heuristic_knapsack(inst, weights, n_sweeps=ws_sweeps, seed=seed)
    results["WS-heuristic"] = dict(archive=archive, n_evals=n_evals,
                                    wallclock=time.perf_counter() - t0)

    t0 = time.perf_counter()
    archive, n_evals = nsga2_knapsack(inst, pop_size=nsga_pop, n_gen=nsga_gen, seed=seed)
    results["NSGA-II"] = dict(archive=archive, n_evals=n_evals,
                               wallclock=time.perf_counter() - t0)

    t0 = time.perf_counter()
    archive, n_evals = bo_knapsack(inst, weights, n_init=bo_init, n_iter=bo_iter, seed=seed)
    results["BO"] = dict(archive=archive, n_evals=n_evals,
                          wallclock=time.perf_counter() - t0)

    return results, [True, True]  # both objectives maximized (profits)


# --------------------------------------------------------------------------
# Bi-TSP instance runner
# --------------------------------------------------------------------------

def run_tsp_instance(n, seed, weights, dl_T, dl_K, ws_sweeps,
                      nsga_pop, nsga_gen, bo_init, bo_iter):
    inst = BiTSPInstance(n, seed=seed)
    results = {}

    for method, use_ts in [("D&L", False), ("D&L-TS", True)]:
        t0 = time.perf_counter()
        archive, n_evals = [], 0
        for wi, w in enumerate(weights):
            base = TSPScalarizedReward(inst, w, noise_sigma=0.02, seed=seed * 100 + wi)
            wrapped = EvalCounter(base)
            out = run_dl(n=inst.n, num_actions=inst.n, reward_fn=wrapped, K=dl_K, T=dl_T,
                          seed=seed * 1000 + wi, use_ts=use_ts, mask_fn=base.mask_fn)
            len1, len2 = base.objectives(out["final_x"])
            archive.append((len1, len2))
            n_evals += wrapped.n_evals
        results[method] = dict(archive=archive, n_evals=n_evals,
                                wallclock=time.perf_counter() - t0)

    t0 = time.perf_counter()
    archive, n_evals = ws_heuristic_tsp(inst, weights, seed=seed)
    results["WS-heuristic"] = dict(archive=archive, n_evals=n_evals,
                                    wallclock=time.perf_counter() - t0)

    t0 = time.perf_counter()
    archive, n_evals = nsga2_tsp(inst, pop_size=nsga_pop, n_gen=nsga_gen, seed=seed)
    results["NSGA-II"] = dict(archive=archive, n_evals=n_evals,
                               wallclock=time.perf_counter() - t0)

    t0 = time.perf_counter()
    archive, n_evals = bo_tsp(inst, weights, n_init=bo_init, n_iter=bo_iter, seed=seed)
    results["BO"] = dict(archive=archive, n_evals=n_evals,
                          wallclock=time.perf_counter() - t0)

    return results, [False, False]  # both objectives minimized (tour length)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def process_instance(problem, size, seed, results, maximize, csv_rows):
    union = np.array([pt for m in results.values() for pt in m["archive"]], dtype=float)
    r, z = nadir_ideal(union, maximize)
    for method, d in results.items():
        ratio, n_nd = hv_ratio(np.array(d["archive"]), maximize, r, z)
        csv_rows.append(dict(
            problem=problem, size=size, method=method, instance_seed=seed,
            hv_ratio=ratio, num_nondominated=n_nd,
            wallclock_sec=d["wallclock"], num_evals=d["n_evals"],
        ))


def summarize(csv_rows):
    from collections import defaultdict
    groups = defaultdict(list)
    for row in csv_rows:
        key = (row["problem"], row["size"], row["method"])
        groups[key].append(row)
    print(f"\n{'problem':10s} {'size':6s} {'method':14s} {'hv_ratio':>18s} {'wallclock_sec':>18s} {'num_evals':>16s}")
    for key in sorted(groups.keys()):
        rows = groups[key]
        hv = np.array([r["hv_ratio"] for r in rows])
        wc = np.array([r["wallclock_sec"] for r in rows])
        ev = np.array([r["num_evals"] for r in rows])

        def fmt(a):
            mean, sem = a.mean(), a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
            return f"{mean:.4f}±{sem:.4f}"

        print(f"{key[0]:10s} {str(key[1]):6s} {key[2]:14s} {fmt(hv):>18s} {fmt(wc):>18s} {fmt(ev):>16s}  (n={len(rows)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out_csv = HERE / "claim4_5_moco.csv"
    csv_rows = []
    t_start = time.perf_counter()

    if args.smoke:
        print("=== SMOKE TEST (tiny scale) ===")
        kp_sizes = [8]
        tsp_sizes = [8]
        n_instances = 2
        n_weights = 5
        dl_T, dl_K = 30, 2
        ws_sweeps = 1
        nsga_pop, nsga_gen = 10, 5
        bo_init, bo_iter = 3, 5
    else:
        print("=== FULL RUN ===")
        kp_sizes = [50, 100]
        tsp_sizes = [20, 50]
        n_instances = 15
        n_weights = 12
        dl_T, dl_K = 100, 4
        ws_sweeps = 3
        nsga_pop, nsga_gen = 40, 60
        bo_init, bo_iter = 8, 25

    weights = make_weights(n_weights)

    for n in kp_sizes:
        for inst_i in range(n_instances):
            seed = 20000 + n * 1000 + inst_i
            t0 = time.perf_counter()
            results, maximize = run_kp_instance(
                n, seed, weights, dl_T, dl_K, ws_sweeps, nsga_pop, nsga_gen, bo_init, bo_iter)
            process_instance("Bi-KP", n, seed, results, maximize, csv_rows)
            print(f"[Bi-KP n={n} seed={seed}] done in {time.perf_counter()-t0:.1f}s "
                  f"(elapsed {time.perf_counter()-t_start:.1f}s)")

    tsp_nsga_pop, tsp_nsga_gen = (10, 8) if args.smoke else (40, 40)
    for n in tsp_sizes:
        for inst_i in range(n_instances):
            seed = 30000 + n * 1000 + inst_i
            t0 = time.perf_counter()
            results, maximize = run_tsp_instance(
                n, seed, weights, dl_T, dl_K, ws_sweeps, tsp_nsga_pop, tsp_nsga_gen, bo_init, bo_iter)
            process_instance("Bi-TSP", n, seed, results, maximize, csv_rows)
            print(f"[Bi-TSP n={n} seed={seed}] done in {time.perf_counter()-t0:.1f}s "
                  f"(elapsed {time.perf_counter()-t_start:.1f}s)")

    if not args.smoke:
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
