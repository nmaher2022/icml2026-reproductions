"""
Runs INSIDE the `vaca` conda env (python 3.9, torch==1.13.1+cpu,
torch-geometric==2.2.0, pytorch-lightning==1.4.9).

Loads a task JSON produced by gen_task.py (which runs in the CausalProfiler
py3.11 venv), trains a VACA model on it, answers every ATE query in the task
via VACA.get_interventional_distr(), computes the harness's own L2 error
formula (sqrt(mean((pred - target)**2)) over non-NaN predictions, replicated
locally -- see causal_profiler/causal_profiler.py:evaluate_error -- so this
process never needs causal_profiler installed), and writes a run-level
result JSON matching the shape produced by examples/evaluation/evaluate.py's
per-run `result` dict (run_error_mean/std, errors, run_failures_mean/std,
failures_all, run_runtime_mean/std, runtimes, num_tries).

Usage:
    vaca_env/python run_vaca.py --task /path/to/task.json \
        --out /path/to/result.json --num_tries 3 --max_epochs 15
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

VACA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "VACA"))
ADAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (VACA_ROOT, ADAPTER_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
import pytorch_lightning as pl  # noqa: E402

from vaca_dataset import build_datamodule  # noqa: E402
from models.vaca.vaca import VACA  # noqa: E402


class SilentVACA(VACA):
    """
    VACA subclass that no-ops the periodic-evaluator / TensorBoard-logging
    hooks (`on_train_epoch_end`, `on_epoch_end`, `on_fit_end`). Those hooks
    assume a `self.my_evaluator` (built in the original repo's `main.py`
    from `models._evaluator.MyEvaluator`, which needs ground-truth
    intervention machinery we don't have) and a real PL logger with a
    `save_dir` -- both irrelevant here since we run with `logger=False` and
    do our own query-answering/eval via get_interventional_distr() after
    training instead of during it.
    """

    def on_train_epoch_end(self, outputs=None):
        pass

    def on_epoch_end(self):
        pass

    def on_fit_end(self):
        pass

    def on_fit_start(self):
        pass


def build_model(dm, z_dim, h_dim_enc, h_dim_dec, max_epochs, seed):
    pl.seed_everything(seed)
    deg = dm.get_deg(indegree=True)
    model = SilentVACA(
        h_dim_list_dec=h_dim_dec,
        h_dim_list_enc=h_dim_enc,
        z_dim=z_dim,
        m_layers=1,
        deg=deg,
        edge_dim=dm.edge_dimension,
        num_nodes=dm.num_nodes,
        beta=1.0,
        annealing_beta=False,
        residual=0.0,
        drop_rate=0.0,
        dropout_adj_rate=0.0,
        dropout_adj_pa_rate=0.2,
        dropout_adj_pa_prob_keep_self=0.0,
        keep_self_loops=True,
        dropout_adj_T=0,
        act_name="relu",
        likelihood_x=dm.likelihood_list,
        distr_z="normal",
        architecture="dgnn",
        estimator="elbo",
        K=1,
        scaler=dm.scaler,
        init=None,
        is_heterogeneous=dm.is_heterogeneous,
        norm_categorical=False,
        norm_by_dim=False,
    )
    model.set_random_train_sampler(dm.get_random_train_sampler())
    model.set_optim_params(
        optim_params={"name": "adam", "params": {"lr": 0.005, "betas": [0.9, 0.999], "weight_decay": 1.2e-6}},
        sched_params={"name": "exp_lr", "params": {"gamma": 0.99}},
    )
    return model


def train_vaca(dm, model, max_epochs, min_epochs):
    trainer = pl.Trainer(
        logger=False,
        checkpoint_callback=False,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        num_sanity_val_steps=0,
        progress_bar_refresh_rate=0,
        weights_summary=None,
        deterministic=True,
        gpus=None,
    )
    trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    return model


def answer_ate_query(model, dm, all_dataset, y_name, t_name, t1_val, t0_val):
    """
    ATE = E[Y | do(T=t1)] - E[Y | do(T=t0)], estimated by running VACA's
    get_interventional_distr twice (once per intervention value) over the
    full injected dataset, in *raw* (unnormalized) data units
    (normalize=False), and taking the mean of the Y column each time.
    """
    y_idx = all_dataset.nodes_list.index(y_name)

    loader1 = dm.query_dataloader(all_dataset)
    x_gener_1, _ = model.get_interventional_distr(loader1, x_I={t_name: float(t1_val)}, normalize=False)
    y1 = x_gener_1["all"][:, y_idx].detach().cpu().numpy()

    loader0 = dm.query_dataloader(all_dataset)
    x_gener_0, _ = model.get_interventional_distr(loader0, x_I={t_name: float(t0_val)}, normalize=False)
    y0 = x_gener_0["all"][:, y_idx].detach().cpu().numpy()

    return float(np.mean(y1) - np.mean(y0))


def estimate_query(model, dm, all_dataset, query):
    """Returns a float estimate, or nan for unsupported query types / failures."""
    try:
        if query["type"] != "ATE":
            return float("nan")
        y_names = query["vars"]["Y"]
        t_names = query["vars"]["T"]
        if len(y_names) != 1 or len(t_names) != 1:
            return float("nan")
        y_name, t_name = y_names[0], t_names[0]

        t1_list, t0_list = query["vars_values"]["T"]
        t1_val = np.ravel(np.array(t1_list[0]))[0]
        t0_val = np.ravel(np.array(t0_list[0]))[0]

        est = answer_ate_query(model, dm, all_dataset, y_name, t_name, t1_val, t0_val)
        if not np.isfinite(est):
            return float("nan")
        return est
    except Exception as e:  # noqa: BLE001
        print(f"    [estimate_query] exception: {e}", file=sys.stderr)
        return float("nan")


def l2_error(pred_values, actual_values):
    """
    Replicates causal_profiler.causal_profiler.CausalProfiler.evaluate_error
    for ErrorMetric.L2 (sqrt(mean squared error) over non-NaN predictions),
    so this process never needs causal_profiler installed. See
    causal_profiler/causal_profiler.py:117-120 (read, not modified).
    """
    pred_values = np.array(pred_values, dtype=float)
    actual_values = np.array(actual_values, dtype=float)
    nan_mask = np.isnan(pred_values)
    num_failed = int(np.sum(nan_mask))
    if np.any(nan_mask):
        pred_values = pred_values[~nan_mask]
        actual_values = actual_values[~nan_mask]
    if len(pred_values) == 0:
        return 0.0, num_failed
    error = float(np.sqrt(np.mean((pred_values - actual_values) ** 2)))
    return error, num_failed


def train_from_task(task, args):
    """
    Builds the CausalProfilerSCMDataset/DataModule + trains a SilentVACA
    model from a task dict (as produced by gen_task.py). Shared by both
    one-shot mode (`main`) and server mode (`serve_main`, used by
    vaca_method.py's per-query bridge to the harness).

    Returns (dm, model, all_dataset, train_failed: bool).
    """
    nodes_list = task["index_to_variable"]
    graph = {int(k): v for k, v in task["graph"].items()}
    adj_edges = {nodes_list[k]: [nodes_list[c] for c in v] for k, v in graph.items()}

    X_all = np.concatenate(
        [np.array(task["data"][name], dtype=np.float32).reshape(-1, 1) for name in nodes_list],
        axis=1,
    )

    root_dir = os.path.join(os.path.dirname(os.path.abspath(task.get("_scratch_hint", "."))), "_vaca_scratch")
    os.makedirs(root_dir, exist_ok=True)

    try:
        dm = build_datamodule(root_dir, nodes_list, adj_edges, X_all, batch_size=args.batch_size, seed=args.seed)
        model = build_model(dm, args.z_dim, [16], [8, 8], args.max_epochs, args.seed)
        model = train_vaca(dm, model, args.max_epochs, args.min_epochs)
        all_dataset = dm.train_dataset
        return dm, model, all_dataset, False
    except Exception as e:  # noqa: BLE001
        print(f"[run_vaca] TRAINING FAILED: {e}", file=sys.stderr)
        return None, None, None, True


def serve_main(args):
    """
    Long-lived worker used by vaca_method.py: loads a task (data+graph, no
    queries needed), trains VACA ONCE, prints 'READY' (or 'TRAIN_FAILED') to
    stdout, then answers one query per line of stdin (JSON-encoded query
    dicts, same shape as gen_task.py's query_to_dict()) until it reads a
    line equal to 'DONE'. Each answer is a JSON line: {"estimate": float}.

    This lets the harness-facing VACAMethod.estimate() (running in the
    separate py3.11 causal_profiler process) retrain only once per run
    (when `data` changes) while still answering queries one at a time,
    exactly matching the harness's own per-query estimate() call pattern,
    without paying VACA's training cost on every single query/try.
    """
    with open(args.task) as f:
        task = json.load(f)

    # VACA's own dataset code (e.g. datasets/_heterogeneous.py:prepare_adj)
    # does bare `print()` logging of the graph structure straight to
    # stdout. That would corrupt our stdout-based READY/JSON-line protocol
    # below, so redirect stdout -> stderr for the whole training phase and
    # only reconnect real stdout afterwards for the request/response loop.
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        dm, model, all_dataset, train_failed = train_from_task(task, args)
    finally:
        sys.stdout = _real_stdout

    if train_failed:
        print("TRAIN_FAILED", flush=True)
    else:
        print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line or line == "DONE":
            break
        try:
            query = json.loads(line)
            if train_failed:
                est = float("nan")
            else:
                est = estimate_query(model, dm, all_dataset, query)
        except Exception as e:  # noqa: BLE001
            print(f"[run_vaca serve] exception answering query: {e}", file=sys.stderr)
            est = float("nan")
        print(json.dumps({"estimate": est}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=False)
    ap.add_argument("--num_tries", type=int, default=3)
    ap.add_argument("--max_epochs", type=int, default=15)
    ap.add_argument("--min_epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--z_dim", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--serve", action="store_true", help="Long-lived per-run worker mode; see serve_main().")
    args = ap.parse_args()

    if args.serve:
        serve_main(args)
        return

    assert args.out, "--out is required outside of --serve mode"

    with open(args.task) as f:
        task = json.load(f)

    t_start_total = time.perf_counter()
    task["_scratch_hint"] = args.out
    dm, model, all_dataset, train_failed = train_from_task(task, args)

    targets = task["targets"]
    run_errors, run_failures, run_runtimes = [], [], []

    for try_idx in range(args.num_tries):
        t0 = time.perf_counter()
        if train_failed:
            estimates = [float("nan") for _ in task["queries"]]
        else:
            estimates = [estimate_query(model, dm, all_dataset, q) for q in task["queries"]]
        duration = time.perf_counter() - t0

        error, num_failed = l2_error(estimates, targets)
        run_errors.append(error)
        run_failures.append(num_failed)
        run_runtimes.append(duration)
        print(f"  Try {try_idx + 1}/{args.num_tries}: Error={error:.4f}, Failed={num_failed}")

    result = {
        "space": task["space_name"],
        "seed": task["seed"],
        "run": task["run"],
        "method": "VACAMethod",
        "run_error_mean": float(np.mean(run_errors)),
        "run_error_std": float(np.std(run_errors)),
        "errors": run_errors,
        "run_failures_mean": float(np.mean(run_failures)),
        "run_failures_std": float(np.std(run_failures)),
        "failures_all": run_failures,
        "run_runtime_mean": float(np.mean(run_runtimes)),
        "run_runtime_std": float(np.std(run_runtimes)),
        "runtimes": run_runtimes,
        "num_tries": args.num_tries,
        "train_failed": train_failed,
        "total_wall_time_s": time.perf_counter() - t_start_total,
        "num_nodes": len(task["index_to_variable"]),
        "num_queries": len(task["queries"]),
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote result: {args.out}")


if __name__ == "__main__":
    main()
