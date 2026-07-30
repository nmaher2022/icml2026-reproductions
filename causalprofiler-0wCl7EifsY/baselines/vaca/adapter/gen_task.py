"""
Runs INSIDE the CausalProfiler venv (python 3.11,
/home/rec1/Desktop/AI_Safety/ICML_reproduce/.venv).

Generates one (data, queries, graph) sample from a CausalProfiler
SpaceOfInterest and serializes it to JSON so that a *separate* process
(running in the old-python VACA conda env) can train/evaluate VACA on it
without needing torch and causal_profiler to coexist in the same
interpreter.

Usage:
    .venv/bin/python gen_task.py --space linear_medium --seed 42 --run 0 \
        --out /path/to/task.json
"""
import argparse
import json
import random

import numpy as np

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# =======================================================================
# Toy-scale Space-of-Interest definitions.
#
# "Medium" in the paper refers to a moderate node-count SCM regime; since
# we're deliberately running at CPU/toy scale we shrink node counts and
# sample counts drastically relative to the paper, but keep the essential
# qualitative distinction the paper's Table 1 cares about:
#   - linear_medium:  mechanism_family = LINEAR
#   - nn_medium:      mechanism_family = NEURAL_NETWORK
# Both otherwise share the same structural knobs (node count range, edge
# density, noise, query type) so that any measured difference is
# attributable to the mechanism family, not incidental config drift.
# =======================================================================
def make_space(name, n_samples, number_of_nodes, number_of_queries):
    common = dict(
        number_of_nodes=number_of_nodes,
        variable_dimensionality=(1, 1),
        expected_edges="1.5 * N",
        noise_mode=NoiseMode.ADDITIVE,
        noise_distribution=NoiseDistribution.GAUSSIAN,
        noise_args=[0, 0.5],
        variable_type=VariableDataType.CONTINUOUS,
        number_of_queries=number_of_queries,
        query_type=QueryType.ATE,
        number_of_data_points=n_samples,
    )
    if name == "linear_medium":
        return SpaceOfInterest(
            mechanism_family=MechanismFamily.LINEAR,
            **common,
        )
    elif name == "nn_medium":
        return SpaceOfInterest(
            mechanism_family=MechanismFamily.NEURAL_NETWORK,
            mechanism_args=[NeuralNetworkType.FEEDFORWARD, 8],
            **common,
        )
    else:
        raise ValueError(f"Unknown space name {name}")


def jsonify(obj):
    """Recursively convert numpy types/arrays into JSON-safe python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    return obj


def query_to_dict(query):
    return {
        "type": query.type.name,  # e.g. "ATE"
        "vars": {label: [v.name for v in vs] for label, vs in query.vars.items()},
        "vars_values": jsonify(query.vars_values),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, choices=["linear_medium", "nn_medium"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_samples", type=int, default=800)
    ap.add_argument("--number_of_nodes_min", type=int, default=5)
    ap.add_argument("--number_of_nodes_max", type=int, default=6)
    ap.add_argument("--number_of_queries", type=int, default=3)
    args = ap.parse_args()

    # Seed per-run so different runs of the same space/seed actually differ
    # (mirrors the harness which reseeds once per seed, then samples fresh
    # SCMs/queries per run without touching the RNG in between; we fold
    # `run` into the seed here since gen_task.py is invoked fresh per run
    # as a subprocess and would otherwise always reproduce the same SCM).
    set_seed(args.seed * 1000 + args.run)

    space = make_space(
        args.space,
        n_samples=args.n_samples,
        number_of_nodes=(args.number_of_nodes_min, args.number_of_nodes_max),
        number_of_queries=args.number_of_queries,
    )
    profiler = CausalProfiler(
        space_of_interest=space, metric=ErrorMetric.L2, n_samples=args.n_samples
    )
    data, (queries, targets), (graph, index_to_variable) = (
        profiler.generate_samples_and_queries()
    )

    task = {
        "space_name": args.space,
        "seed": args.seed,
        "run": args.run,
        "index_to_variable": index_to_variable,
        "graph": {str(k): v for k, v in graph.items()},
        "data": {k: v.tolist() for k, v in data.items()},
        "queries": [query_to_dict(q) for q in queries],
        "targets": [float(t) for t in targets],
    }

    with open(args.out, "w") as f:
        json.dump(task, f)

    print(
        f"Wrote task: {len(index_to_variable)} nodes, "
        f"{len(queries)} queries -> {args.out}"
    )


if __name__ == "__main__":
    main()
