"""
Supplementary stress-test: does increasing discreteness (more categories,
more noise regions -> more region-dependent stochastic support per the
paper's Definition E.2) and training for longer push CausalNF's continuous
density fit into visible divergence/NaN territory, beyond what the main toy
run (2-3 categories, 150 epochs) showed?

Not part of the primary comparison -- a secondary probe, reported separately
in REPORT.md.
"""

import json
import os
import random
import time

import numpy as np
import torch

from causal_profiler import (
    CausalProfiler,
    SpaceOfInterest,
    ErrorMetric,
    MechanismFamily,
    VariableDataType,
    QueryType,
)

from causal_nf_method import CausalNFMethod

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


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


space = SpaceOfInterest(
    number_of_nodes=(4, 6),
    variable_dimensionality=(1, 1),
    mechanism_family=MechanismFamily.TABULAR,
    expected_edges="1.5*N",
    number_of_categories=(4, 6),  # more categories -> more atoms to fit
    number_of_noise_regions=6,  # more noise regions -> more stochastic mechanism (fixed int, "V_to_PA" blows up combinatorially)
    variable_type=VariableDataType.DISCRETE,
    number_of_queries=5,
    query_type=QueryType.ATE,
    number_of_data_points=800,
)

profiler = CausalProfiler(space_of_interest=space, metric=ErrorMetric.L2, n_samples=2000)
method = CausalNFMethod(hidden_features=(64, 64), epochs=400, lr=1e-3, n_samples_effect=1000)

results = []
for seed in [42, 43, 44]:
    set_seed(seed)
    for run in range(2):
        data, (queries, targets), (graph, index_to_variable) = (
            profiler.generate_samples_and_queries()
        )
        if not queries:
            continue
        run_errors, run_failures = [], []
        for try_idx in range(2):
            t0 = time.perf_counter()
            estimates = [method.estimate(q, data, graph, index_to_variable) for q in queries]
            dt = time.perf_counter() - t0
            error, num_failed = profiler.evaluate_error(estimated=estimates, target=targets)
            run_errors.append(error)
            run_failures.append(int(num_failed))
            print(
                f"[seed{seed}_run{run}] try{try_idx}: nodes={len(index_to_variable)} "
                f"error={error:.4f} failed={num_failed}/{len(queries)} time={dt:.2f}s"
            )
        results.append(
            {
                "seed": seed,
                "run": run,
                "num_nodes": len(index_to_variable),
                "num_queries": len(queries),
                "run_error_mean": float(np.mean(run_errors)),
                "run_failures_mean": float(np.mean(run_failures)),
                "run_failure_rate_mean": float(np.mean(run_failures) / len(queries)),
            }
        )

print("\nfit stats:", method.stats)
agg = {
    "mean_error": float(np.mean([r["run_error_mean"] for r in results])),
    "mean_failure_rate": float(np.mean([r["run_failure_rate_mean"] for r in results])),
    "n_runs": len(results),
    "fit_stats": dict(method.stats),
}
print("stress-test aggregate:", agg)

with open(os.path.join(RESULTS_DIR, "regional_discrete_stress_results.json"), "w") as f:
    json.dump({"per_run": results, "aggregate": agg}, f, indent=2, default=json_dump_convert)
