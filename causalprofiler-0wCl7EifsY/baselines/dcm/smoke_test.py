"""
Tiny end-to-end smoke test for DCMMethod plugged into the CausalProfiler
harness. 3-5 node SCMs, 1 seed, 1 run, 2 tries, small n_samples. Confirms the
full pipeline runs on CPU without hanging and produces finite (or documented
NaN) output, and reports wall-clock time.
"""
import random
import time

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_smoke(name, space_kwargs, n_runs=1, n_tries=2, seed=42):
    print(f"\n=== SMOKE TEST: {name} ===")
    set_seed(seed)
    space = SpaceOfInterest(**space_kwargs)
    profiler = CausalProfiler(space_of_interest=space, metric=ErrorMetric.L2)
    method = DCMMethod(hidden_dim=16, T=20, num_epochs=5, batch_size=32, num_mc_samples=100)

    t_start = time.perf_counter()
    for run in range(n_runs):
        data, (queries, targets), (graph, index_to_variable) = (
            profiler.generate_samples_and_queries()
        )
        print(f"  run {run}: {len(index_to_variable)} nodes, {len(queries)} queries")
        for try_idx in range(n_tries):
            t0 = time.perf_counter()
            estimates = [
                method.estimate(q, data, graph, index_to_variable) for q in queries
            ]
            dt = time.perf_counter() - t0
            error, num_failed = profiler.evaluate_error(estimates, targets)
            print(
                f"    try {try_idx}: estimates={estimates} targets={targets} "
                f"error={error:.4f} failed={num_failed} time={dt:.2f}s"
            )
    total_time = time.perf_counter() - t_start
    print(
        f"  TOTAL TIME: {total_time:.2f}s | retrains={method.n_retrains} "
        f"retrain_failures={method.n_retrain_failures} "
        f"estimate_calls={method.n_estimate_calls} "
        f"estimate_failures={method.n_estimate_failures}"
    )
    return total_time


if __name__ == "__main__":
    linear_cfg = dict(
        number_of_nodes=(4, 5),
        variable_dimensionality=(1, 1),
        mechanism_family=MechanismFamily.LINEAR,
        expected_edges="2 * N",
        noise_mode=NoiseMode.ADDITIVE,
        noise_distribution=NoiseDistribution.GAUSSIAN,
        noise_args=[0, 0.5],
        variable_type=VariableDataType.CONTINUOUS,
        number_of_queries=2,
        query_type=QueryType.ATE,
        number_of_data_points=500,
    )
    discrete_cfg = dict(
        number_of_nodes=(4, 5),
        mechanism_family=MechanismFamily.TABULAR,
        variable_type=VariableDataType.DISCRETE,
        number_of_categories=2,
        expected_edges="N",
        discrete_function_sampling=FunctionSampling.SAMPLE_REJECTION,
        number_of_queries=2,
        query_type=QueryType.CTF_TE,
        number_of_data_points=500,
    )

    t1 = run_smoke("Linear-Medium-like (ATE)", linear_cfg)
    t2 = run_smoke("Regional-Discrete-like (CTF_TE)", discrete_cfg)
    print(f"\nGRAND TOTAL SMOKE TEST TIME: {t1 + t2:.2f}s")
