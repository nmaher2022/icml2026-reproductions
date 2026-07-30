"""
Toy-scale CPU evaluation of CausalNFMethod on the CausalProfiler harness.

Two settings:
  A) continuous_linear_ate      - standard continuous SCM, sanity baseline
     (CausalNF should do reasonably here, it's the setting the method was
     designed for).
  B) regional_discrete_tabular_ate / _conditional
     - TABULAR mechanism_family + DISCRETE variable_type + default
       number_of_noise_regions="V" == a Regional Discrete SCM per the paper's
       Definition E.2 (region-dependent discrete support). CausalNF is
       expected to fail much more often here since it is a continuous
       normalizing-flow method with no discrete-data support.

Toy scale: 3-6 node graphs, 3 seeds x 3 runs x 3 tries, n_samples_data in the
few hundreds, small flow (2x64 hidden units, 150 training epochs). CPU only.
Not intended to match the paper's numbers -- only to check the qualitative
direction of Claim 4 (higher CausalNF failure rate on Regional Discrete SCMs
than on continuous SCMs).
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

from causal_nf_method import CausalNFMethod

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def json_dump_convert(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


SPACES = {
    "continuous_linear_ate": dict(
        number_of_nodes=(4, 6),
        variable_dimensionality=(1, 1),
        mechanism_family=MechanismFamily.LINEAR,
        expected_edges="1.5*N",
        noise_mode=NoiseMode.ADDITIVE,
        noise_distribution=NoiseDistribution.GAUSSIAN,
        noise_args=[0, 0.5],
        variable_type=VariableDataType.CONTINUOUS,
        number_of_queries=5,
        query_type=QueryType.ATE,
        number_of_data_points=800,
    ),
    "regional_discrete_tabular_ate": dict(
        number_of_nodes=(4, 6),
        variable_dimensionality=(1, 1),
        mechanism_family=MechanismFamily.TABULAR,
        expected_edges="1.5*N",
        number_of_categories=(2, 3),
        number_of_noise_regions="V",
        variable_type=VariableDataType.DISCRETE,
        number_of_queries=5,
        query_type=QueryType.ATE,
        number_of_data_points=800,
    ),
    "regional_discrete_tabular_conditional": dict(
        number_of_nodes=(4, 6),
        variable_dimensionality=(1, 1),
        mechanism_family=MechanismFamily.TABULAR,
        expected_edges="1.5*N",
        number_of_categories=(2, 3),
        number_of_noise_regions="V",
        variable_type=VariableDataType.DISCRETE,
        number_of_queries=5,
        query_type=QueryType.CONDITIONAL,
        number_of_data_points=800,
    ),
}

SEED_LIST = [42, 43, 44]
NUM_RUNS = 3
NUM_TRIES = 3
N_SAMPLES_GROUND_TRUTH = 2000  # for CausalProfiler's internal query estimation


def evaluate_space(space_name, space_kwargs):
    print(f"\n=== Evaluating space: {space_name} ===")
    space = SpaceOfInterest(**space_kwargs)
    profiler = CausalProfiler(
        space_of_interest=space, metric=ErrorMetric.L2, n_samples=N_SAMPLES_GROUND_TRUTH
    )
    method = CausalNFMethod(
        hidden_features=(64, 64), epochs=150, lr=1e-3, n_samples_effect=1000
    )
    results = []

    for seed in SEED_LIST:
        set_seed(seed)
        for run in range(NUM_RUNS):
            run_id = f"{space_name}_seed{seed}_run{run}"
            t_gen0 = time.perf_counter()
            data, (queries, targets), (graph, index_to_variable) = (
                profiler.generate_samples_and_queries()
            )
            t_gen = time.perf_counter() - t_gen0

            if len(queries) == 0:
                print(f"  [{run_id}] no queries generated, skipping")
                continue

            run_errors, run_failures, run_runtimes = [], [], []
            for try_idx in range(NUM_TRIES):
                t0 = time.perf_counter()
                estimates = [
                    method.estimate(q, data, graph, index_to_variable) for q in queries
                ]
                dt = time.perf_counter() - t0
                error, num_failed = profiler.evaluate_error(
                    estimated=estimates, target=targets
                )
                run_errors.append(error)
                run_failures.append(int(num_failed))
                run_runtimes.append(dt)
                print(
                    f"  [{run_id}] try {try_idx + 1}/{NUM_TRIES}: "
                    f"error={error:.4f} failed={num_failed}/{len(queries)} time={dt:.2f}s"
                )

            result = {
                "space": space_name,
                "seed": seed,
                "run": run,
                "method": "CausalNFMethod",
                "num_nodes": len(index_to_variable),
                "num_queries": len(queries),
                "gen_time_s": t_gen,
                "run_error_mean": float(np.mean(run_errors)),
                "run_error_std": float(np.std(run_errors)),
                "errors": run_errors,
                "run_failures_mean": float(np.mean(run_failures)),
                "run_failures_std": float(np.std(run_failures)),
                "failures_all": run_failures,
                "run_failure_rate_mean": float(np.mean(run_failures) / len(queries)),
                "run_runtime_mean": float(np.mean(run_runtimes)),
                "num_tries": NUM_TRIES,
                "num_queries_per_try": len(queries),
            }
            results.append(result)

    summary_file = os.path.join(RESULTS_DIR, f"{space_name}_results.json")
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2, default=json_dump_convert)

    # Aggregate across all runs for this space
    all_errors = [r["run_error_mean"] for r in results]
    all_failure_rates = [r["run_failure_rate_mean"] for r in results]
    agg = {
        "space": space_name,
        "n_runs": len(results),
        "mean_error": float(np.mean(all_errors)) if all_errors else None,
        "std_error": float(np.std(all_errors)) if all_errors else None,
        "mean_failure_rate": float(np.mean(all_failure_rates)) if all_failure_rates else None,
        "std_failure_rate": float(np.std(all_failure_rates)) if all_failure_rates else None,
        "fit_stats": dict(method.stats),
    }
    print(f"  >>> {space_name} AGGREGATE: {agg}")
    return agg, results


if __name__ == "__main__":
    all_agg = {}
    all_results = {}
    for name, kwargs in SPACES.items():
        agg, results = evaluate_space(name, kwargs)
        all_agg[name] = agg
        all_results[name] = results

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agg_file = os.path.join(RESULTS_DIR, f"summary_{timestamp}.json")
    with open(agg_file, "w") as f:
        json.dump(all_agg, f, indent=2, default=json_dump_convert)
    # Also write a stable-named "latest" summary
    with open(os.path.join(RESULTS_DIR, "summary_latest.json"), "w") as f:
        json.dump(all_agg, f, indent=2, default=json_dump_convert)

    print("\n\n=== FINAL SUMMARY ===")
    for name, agg in all_agg.items():
        print(
            f"{name}: mean_error={agg['mean_error']:.4f} "
            f"mean_failure_rate={agg['mean_failure_rate']:.3f} "
            f"(n_runs={agg['n_runs']}) fit_stats={agg['fit_stats']}"
        )
    print(f"\nSaved per-space results to {RESULTS_DIR}")
    print(f"Saved aggregate summary to {agg_file}")
