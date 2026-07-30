"""
Tiny end-to-end smoke test for VACAMethod plugged into the REAL CausalProfiler
harness (mirrors examples/evaluation/evaluate.py's own loop structure: seed ->
run -> try). Runs in the py3.11 CausalProfiler venv; VACAMethod internally
bridges to the separate `vaca` conda env (python 3.9) via a persistent
subprocess (adapter/run_vaca.py --serve). 1 seed, 1 run, 2 tries, small
n_samples, 4-5 node NEURAL_NETWORK-family SCM (the harness's own
generate_samples_and_queries()/CausalProfiler.evaluate_error() are used
unmodified). Confirms the full pipeline runs on CPU without hanging/crashing
and produces finite (or documented NaN) output, and reports wall-clock time.
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
)
from causal_profiler.constants import NeuralNetworkType

from vaca_method import VACAMethod


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_smoke(name, space_kwargs, n_runs=1, n_tries=2, seed=42):
    print(f"\n=== SMOKE TEST: {name} ===")
    set_seed(seed)
    space = SpaceOfInterest(**space_kwargs)
    profiler = CausalProfiler(space_of_interest=space, metric=ErrorMetric.L2)
    method = VACAMethod(max_epochs=6, min_epochs=2, batch_size=16, verbose=True)

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
    method._stop_worker()
    total_time = time.perf_counter() - t_start
    print(f"  TOTAL TIME: {total_time:.2f}s")
    return total_time


if __name__ == "__main__":
    nn_cfg = dict(
        number_of_nodes=(4, 5),
        variable_dimensionality=(1, 1),
        mechanism_family=MechanismFamily.NEURAL_NETWORK,
        mechanism_args=[NeuralNetworkType.FEEDFORWARD, 8],
        expected_edges="1.5 * N",
        noise_mode=NoiseMode.ADDITIVE,
        noise_distribution=NoiseDistribution.GAUSSIAN,
        noise_args=[0, 0.5],
        variable_type=VariableDataType.CONTINUOUS,
        number_of_queries=3,
        query_type=QueryType.ATE,
        number_of_data_points=600,
    )

    t1 = run_smoke("NN-Medium-like (ATE)", nn_cfg)
    print(f"\nGRAND TOTAL SMOKE TEST TIME: {t1:.2f}s")
