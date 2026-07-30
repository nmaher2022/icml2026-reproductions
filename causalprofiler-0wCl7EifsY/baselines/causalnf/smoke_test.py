"""
Tiny end-to-end smoke test of CausalNFMethod plugged into the real
CausalProfiler harness. Not a real evaluation run -- just checks that the
full pipeline (SpaceOfInterest -> CausalProfiler -> generate_samples_and_queries
-> CausalNFMethod.estimate -> evaluate_error) runs to completion on CPU in
reasonable time, for both a small continuous space and a small
Regional-Discrete-SCM-like space.

A crash or hang is a smoke-test failure. A high failure rate / high error is
NOT a smoke-test failure -- it may be exactly the expected result for the
discrete setting.
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

from causal_nf_method import CausalNFMethod


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_one(space, label, n_samples):
    print(f"\n=== SMOKE TEST: {label} ===")
    set_seed(42)
    profiler = CausalProfiler(space_of_interest=space, metric=ErrorMetric.L2, n_samples=n_samples)
    method = CausalNFMethod(hidden_features=(32, 32), epochs=100, n_samples_effect=500)

    data, (queries, targets), (graph, index_to_variable) = profiler.generate_samples_and_queries()
    print("nodes:", index_to_variable)
    print("graph:", graph)
    print("n queries:", len(queries))

    for try_idx in range(2):
        t0 = time.perf_counter()
        estimates = [method.estimate(q, data, graph, index_to_variable) for q in queries]
        dt = time.perf_counter() - t0
        error, num_failed = profiler.evaluate_error(estimated=estimates, target=targets)
        print(f"  try {try_idx}: error={error:.4f} failed={num_failed}/{len(queries)} time={dt:.2f}s "
              f"estimates={estimates} targets={targets}")

    print("fit stats:", method.stats)
    print(f"=== {label}: SMOKE TEST PASSED (no crash/hang) ===")


if __name__ == "__main__":
    cont_space = SpaceOfInterest(
        number_of_nodes=(3, 5),
        variable_dimensionality=(1, 1),
        mechanism_family=MechanismFamily.LINEAR,
        expected_edges="1.5*N",
        noise_mode=NoiseMode.ADDITIVE,
        noise_distribution=NoiseDistribution.GAUSSIAN,
        noise_args=[0, 0.5],
        variable_type=VariableDataType.CONTINUOUS,
        number_of_queries=3,
        query_type=QueryType.ATE,
        number_of_data_points=500,
    )
    run_one(cont_space, "continuous linear ATE (toy)", n_samples=500)

    discrete_space = SpaceOfInterest(
        number_of_nodes=(3, 5),
        variable_dimensionality=(1, 1),
        mechanism_family=MechanismFamily.TABULAR,
        expected_edges="1.5*N",
        number_of_categories=(2, 3),
        number_of_noise_regions="V",
        variable_type=VariableDataType.DISCRETE,
        number_of_queries=3,
        query_type=QueryType.ATE,
        number_of_data_points=500,
    )
    run_one(discrete_space, "regional-discrete tabular ATE (toy)", n_samples=500)

    print("\nALL SMOKE TESTS PASSED")
