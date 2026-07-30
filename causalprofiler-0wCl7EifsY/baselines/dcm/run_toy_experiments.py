"""
Toy-scale evaluation of DCMMethod on two CausalProfiler space-of-interest
configs, mirroring (at drastically reduced scale, CPU-only) the paper's:

  - Experiment 1 / "Linear-Medium": linear, continuous SCMs, ATE queries.
  - Experiment 2 / "Regional Discrete SCM" (Disc-C2-Reject-like): tabular,
    discrete (binary) SCMs with rejection-sampled mechanisms, Ctf-TE queries.

Follows the same driver structure as
causal-profiler/examples/evaluation/evaluate.py (space -> seeds -> runs ->
tries), but targets toy scale and saves results under
baselines/dcm/results/.
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
    FunctionSampling,
)

from dcm_method import DCMMethod

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


def run_space(space_name, space_kwargs, dcm_kwargs, seed_list, num_runs, num_tries,
              time_budget_per_run=120.0):
    print(f"\n=== {space_name} ===")
    space = SpaceOfInterest(**space_kwargs)
    profiler = CausalProfiler(space_of_interest=space, metric=ErrorMetric.L2)
    method = DCMMethod(**dcm_kwargs)

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
                "method": "DCMMethod",
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

    return results, method


def summarize(space_name, results, method, total_wall_time):
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
        "n_retrains": method.n_retrains,
        "n_retrain_failures": method.n_retrain_failures,
        "n_estimate_calls": method.n_estimate_calls,
        "n_estimate_failures": method.n_estimate_failures,
    }
    return summary


if __name__ == "__main__":
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
    discrete_cfg = dict(
        number_of_nodes=(5, 6),
        mechanism_family=MechanismFamily.TABULAR,
        variable_type=VariableDataType.DISCRETE,
        number_of_categories=2,
        expected_edges="N",
        discrete_function_sampling=FunctionSampling.SAMPLE_REJECTION,
        number_of_queries=3,
        query_type=QueryType.CTF_TE,
        number_of_data_points=800,
    )

    dcm_kwargs = dict(
        hidden_dim=32, T=50, num_epochs=8, batch_size=64, lr=1e-3,
        num_mc_samples=200, kernel_bandwidth=0.5, max_ctf_units=200,
    )

    seed_list = [42, 43, 44]
    num_runs = 3
    num_tries = 3

    all_summaries = {}

    t0 = time.perf_counter()
    results_linear, method_linear = run_space(
        "Linear-Medium-toy", linear_cfg, dcm_kwargs, seed_list, num_runs, num_tries
    )
    t_linear = time.perf_counter() - t0
    all_summaries["Linear-Medium-toy"] = summarize(
        "Linear-Medium-toy", results_linear, method_linear, t_linear
    )

    t0 = time.perf_counter()
    results_disc, method_disc = run_space(
        "Regional-Discrete-toy", discrete_cfg, dcm_kwargs, seed_list, num_runs, num_tries
    )
    t_disc = time.perf_counter() - t0
    all_summaries["Regional-Discrete-toy"] = summarize(
        "Regional-Discrete-toy", results_disc, method_disc, t_disc
    )

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2, default=json_convert)

    print("\n\n=== FINAL SUMMARY ===")
    print(json.dumps(all_summaries, indent=2, default=json_convert))
    print(f"\nGRAND TOTAL WALL TIME: {t_linear + t_disc:.1f}s")
