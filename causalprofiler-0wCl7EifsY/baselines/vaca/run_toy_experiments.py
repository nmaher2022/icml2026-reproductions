"""
Toy-scale evaluation of VACAMethod on two CausalProfiler space-of-interest
configs, mirroring (at drastically reduced scale, CPU-only) the paper's
Table 1 settings that Claim 3 cares about:

  - "Linear-Medium-toy": linear, continuous SCMs, ATE queries. Config is
    made IDENTICAL to the DCM baseline's Linear-Medium-toy
    (baselines/dcm/run_toy_experiments.py) so the two methods' numbers are
    directly comparable.
  - "NN-Medium-toy": same knobs, but mechanism_family=NEURAL_NETWORK -- the
    setting the paper's Table 1 reports VACA as the *best* baseline on
    (mean error 0.009 vs DCM's much worse NN-Medium score), so this is the
    qualitative pattern this toy run is trying to (weakly) probe.

Follows the same driver structure as
causal-profiler/examples/evaluation/evaluate.py (space -> seeds -> runs ->
tries) and baselines/dcm/run_toy_experiments.py, but calls VACAMethod
(vaca_method.py) instead, which trains a real VACA model per run via a
persistent subprocess in the separate `vaca` conda env (see REPORT.md).

Run from the CausalProfiler py3.11 venv:
    .venv/bin/python baselines/vaca/run_toy_experiments.py
"""
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch

from causal_profiler import (
    CausalProfiler,
    SpaceOfInterest,
    ErrorMetric,
    MechanismFamily,
    NoiseMode,
    NoiseDistribution,
    VariableDataType,
    QueryType,
)
from causal_profiler.constants import NeuralNetworkType

from vaca_method import VACAMethod

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def json_convert(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def run_space(space_name, space_kwargs, vaca_kwargs, seed_list, num_runs, num_tries,
              time_budget_per_run=180.0):
    print(f"\n=== {space_name} ===")
    space = SpaceOfInterest(**space_kwargs)
    profiler = CausalProfiler(space_of_interest=space, metric=ErrorMetric.L2)
    method = VACAMethod(**vaca_kwargs)

    results = []
    for seed in seed_list:
        set_seed(seed)
        for run in range(num_runs):
            run_id = f"{space_name}_seed{seed}_run{run}"
            data, (queries, targets), (graph, index_to_variable) = (
                profiler.generate_samples_and_queries()
            )
            n_nodes = len(index_to_variable)
            n_queries = len(queries)

            run_errors, run_failures, run_runtimes = [], [], []
            run_start = time.perf_counter()
            for try_idx in range(num_tries):
                t0 = time.perf_counter()
                estimates = [
                    method.estimate(q, data, graph, index_to_variable) for q in queries
                ]
                dt = time.perf_counter() - t0
                error, num_failed = profiler.evaluate_error(estimates, targets)
                run_errors.append(error)
                run_failures.append(int(num_failed))
                run_runtimes.append(dt)
                print(
                    f"  {run_id} try{try_idx}: nodes={n_nodes} queries={n_queries} "
                    f"error={error:.4f} failed={num_failed}/{n_queries} time={dt:.2f}s"
                )
            run_total = time.perf_counter() - run_start
            if run_total > time_budget_per_run:
                print(
                    f"  WARNING: run {run_id} took {run_total:.1f}s, exceeding "
                    f"the {time_budget_per_run}s soft budget."
                )

            result = {
                "space": space_name,
                "seed": seed,
                "run": run,
                "method": "VACAMethod",
                "n_nodes": n_nodes,
                "n_queries": n_queries,
                "run_error_mean": float(np.mean(run_errors)),
                "run_error_std": float(np.std(run_errors)),
                "errors": run_errors,
                "run_failures_mean": float(np.mean(run_failures)),
                "run_failure_rate": float(np.mean(run_failures)) / max(n_queries, 1),
                "failures_all": run_failures,
                "run_runtime_mean": float(np.mean(run_runtimes)),
                "run_runtime_std": float(np.std(run_runtimes)),
                "runtimes": run_runtimes,
                "run_total_time": run_total,
                "num_tries": num_tries,
            }
            results.append(result)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(
                os.path.join(RESULTS_DIR, f"result_{space_name}_seed{seed}_run{run}_{ts}.json"),
                "w",
            ) as f:
                json.dump(result, f, indent=2, default=json_convert)

    method._stop_worker()
    return results, method


def summarize(space_name, results, total_wall_time):
    all_errors = [e for r in results for e in r["errors"]]
    all_failures = [f for r in results for f in r["failures_all"]]
    all_runtimes = [t for r in results for t in r["runtimes"]]
    n_queries_total = sum(r["n_queries"] for r in results)
    failure_rate = (
        sum(all_failures) / (n_queries_total * results[0]["num_tries"])
        if results
        else float("nan")
    )
    summary = {
        "space": space_name,
        "n_runs": len(results),
        "mean_error": float(np.mean(all_errors)) if all_errors else float("nan"),
        "std_error": float(np.std(all_errors)) if all_errors else float("nan"),
        "max_error": float(np.max(all_errors)) if all_errors else float("nan"),
        "mean_try_runtime_s": float(np.mean(all_runtimes)) if all_runtimes else float("nan"),
        "failure_rate_pct": 100.0 * failure_rate,
        "total_wall_time_s": total_wall_time,
    }
    return summary


if __name__ == "__main__":
    # Identical to DCM's Linear-Medium-toy config
    # (baselines/dcm/run_toy_experiments.py) for direct comparability.
    linear_cfg = dict(
        number_of_nodes=(5, 6),
        variable_dimensionality=(1, 1),
        mechanism_family=MechanismFamily.LINEAR,
        expected_edges="2 * N",
        noise_mode=NoiseMode.ADDITIVE,
        noise_distribution=NoiseDistribution.GAUSSIAN,
        noise_args=[0, 0.5],
        variable_type=VariableDataType.CONTINUOUS,
        number_of_queries=3,
        query_type=QueryType.ATE,
        number_of_data_points=800,
    )
    # Same knobs, but NEURAL_NETWORK mechanisms -- the paper's Table 1
    # setting where VACA is reported as the best baseline.
    nn_cfg = dict(
        number_of_nodes=(5, 6),
        variable_dimensionality=(1, 1),
        mechanism_family=MechanismFamily.NEURAL_NETWORK,
        mechanism_args=[NeuralNetworkType.FEEDFORWARD, 8],
        expected_edges="2 * N",
        noise_mode=NoiseMode.ADDITIVE,
        noise_distribution=NoiseDistribution.GAUSSIAN,
        noise_args=[0, 0.5],
        variable_type=VariableDataType.CONTINUOUS,
        number_of_queries=3,
        query_type=QueryType.ATE,
        number_of_data_points=800,
    )

    vaca_kwargs = dict(max_epochs=15, min_epochs=3, batch_size=32, z_dim=4, verbose=False)

    seed_list = [42, 43, 44]
    num_runs = 3
    num_tries = 3

    all_summaries = {}

    t0 = time.perf_counter()
    results_linear, method_linear = run_space(
        "Linear-Medium-toy", linear_cfg, vaca_kwargs, seed_list, num_runs, num_tries
    )
    t_linear = time.perf_counter() - t0
    all_summaries["Linear-Medium-toy"] = summarize("Linear-Medium-toy", results_linear, t_linear)

    t0 = time.perf_counter()
    results_nn, method_nn = run_space(
        "NN-Medium-toy", nn_cfg, vaca_kwargs, seed_list, num_runs, num_tries
    )
    t_nn = time.perf_counter() - t0
    all_summaries["NN-Medium-toy"] = summarize("NN-Medium-toy", results_nn, t_nn)

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2, default=json_convert)

    print("\n\n=== FINAL SUMMARY ===")
    print(json.dumps(all_summaries, indent=2, default=json_convert))
    print(f"\nGRAND TOTAL WALL TIME: {t_linear + t_nn:.1f}s")
