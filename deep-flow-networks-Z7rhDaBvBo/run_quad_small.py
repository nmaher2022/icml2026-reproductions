"""Fresh reduced-scale DFN vs MLP vs LSET head-to-head (Claim 2, independent run).

Uses the repo's own `run_experiment` pipeline (training, MILP formulations,
solution-quality protocol) unchanged; only the scale is reduced so every MILP
fits the free size-limited Gurobi license (2000 vars / 2000 constrs):

  paper Small : n=8,  K=1000, ~5k params/model, Gurobi cap 3600 s
  this run    : n=8,  K=1000, ~3k params (DFN/MLP), Gurobi cap 600 s, paper epochs (1000)

Models are approximately parameter-matched, as in the paper:
  DFN  [12,30,12]  -> 1728 arcs incl. fixed (MILP: 1736 vars, 55 constrs), ~3.0k learnable params
  MLP  [50,50]     -> ~3.0k params (big-M MILP: ~200 vars)
  LSET 100 pieces  -> ~0.9k params (license-forced: the log-sum-exp MILP expands
                      each exp/log general constraint into many auxiliaries, so 320
                      pieces exceeds the 2000-var cap; LSET is param-DISadvantaged here)

Everything retrains from scratch (fresh output dir, no committed artifacts).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE / "deep-flow-networks"
SCRIPTS = CODE_ROOT / "ICML_experiments" / "main_text" / "scripts"
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(SCRIPTS))

from main_common import run_experiment

OUTPUT_DIR = HERE / "outputs_small" / "quadratic_small"
N_SEEDS = 3
TIME_LIMIT = 600.0
TRAINING = {"epochs": 1000, "batch_size": 8, "val_frac": 0.15, "test_frac": 0.15,
            "eps": 1e-8, "device": "cpu", "weight_decay": 0.0}
MODEL_LR = {"DFN": 1e-2, "MLP": 1e-3, "LSET": 1e-3}
MODELS = {
    "DFN": {"input_dim": 8, "layer_sizes": [12, 30, 12], "alpha": 5e-3, "beta": -2.0,
            "big_cost": 1e6, "big_cap": 1e6},
    "MLP": {"in_dim": 8, "hidden_dims": [50, 50]},
    "LSET": {"in_dim": 8, "n_pieces": 100, "T": 0.05},
}
DATASET = {"K": 1000, "dim": 8, "eigen_min": 1.0, "eigen_max": 15.0, "x_min": -50, "x_max": 50}

if __name__ == "__main__":
    _, summary = run_experiment(
        name="quadratic_small",
        label="QuadSmall",
        dataset_type="quadratic",
        dataset_cfg=DATASET,
        x_min=-50,
        x_max=50,
        budget_range=(-30, 30),
        dfn_alpha=5e-3,
        training=TRAINING,
        models=MODELS,
        model_lr=MODEL_LR,
        n_seeds=N_SEEDS,
        time_limit=TIME_LIMIT,
        output_dir=OUTPUT_DIR,
        verbose=True,
    )
    print(summary.to_string(index=False))
