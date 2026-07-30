"""
Extra toy-scale DCM run on the "NN-Medium-toy" setting, added AFTER the
original run_toy_experiments.py (which covered Linear-Medium and Regional-
Discrete, aimed at Claim 4). This one exists specifically so Claim 3
(DCM-vs-VACA on Linear-Medium/NN-Medium) has DCM numbers on the SAME
NN-Medium config VACA was evaluated on (see baselines/vaca/run_toy_experiments.py),
for a true head-to-head comparison rather than an apples-to-oranges one.
"""
import json
import os
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
    NeuralNetworkType,
)

from dcm_method import DCMMethod
from run_toy_experiments import run_space, summarize, json_convert, set_seed

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

nn_medium_cfg = dict(
    number_of_nodes=(5, 6),
    variable_dimensionality=(1, 1),
    mechanism_family=MechanismFamily.NEURAL_NETWORK,
    mechanism_args=[NeuralNetworkType.FEEDFORWARD, 8],
    expected_edges="2*N",
    noise_mode=NoiseMode.ADDITIVE,
    noise_distribution=NoiseDistribution.GAUSSIAN,
    noise_args=[0, 0.5],
    variable_type=VariableDataType.CONTINUOUS,
    number_of_queries=3,
    query_type=QueryType.ATE,
    number_of_data_points=800,
)

dcm_kwargs = dict(
    hidden_dim=32, T=50, num_epochs=8, batch_size=64, lr=1e-3,
    num_mc_samples=200, kernel_bandwidth=0.5, max_ctf_units=200,
)

seed_list = [42, 43, 44]
num_runs = 3
num_tries = 3

if __name__ == "__main__":
    t0 = time.perf_counter()
    results, method = run_space(
        "NN-Medium-toy", nn_medium_cfg, dcm_kwargs, seed_list, num_runs, num_tries
    )
    t_total = time.perf_counter() - t0
    summary = summarize("NN-Medium-toy", results, method, t_total)

    # merge into the existing summary.json rather than overwrite
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    all_summaries = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            all_summaries = json.load(f)
    all_summaries["NN-Medium-toy"] = summary
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=json_convert)

    print(json.dumps(summary, indent=2, default=json_convert))
    print(f"\nTOTAL WALL TIME: {t_total:.1f}s")
