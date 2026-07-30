"""
VACAMethod: the harness-facing plug-in adapter for VACA (Sanchez-Martin et
al. 2022, "VACA: Design of Variational Graph Autoencoders for Interventional
and Counterfactual Queries", https://github.com/psanch21/VACA), wired into
the CausalProfiler evaluation harness interface:

    class MyCausalMethod:
        def estimate(self, query, data, graph, index_to_variable) -> float: ...

This file is meant to be imported and used from WITHIN the CausalProfiler
py3.11 venv (i.e. dropped in next to my_causal_method.py and imported by an
evaluate.py-style driver), exactly like DCM's `dcm_method.py`.

Why a subprocess bridge, unlike DCM/CausalNF
----------------------------------------------
VACA's dependencies (torch==1.13.1+cpu, torch-geometric==2.2.0,
pytorch-lightning==1.4.9) only install cleanly under Python 3.9, while
causal_profiler requires Python>=3.10 -- the two cannot live in the same
interpreter/venv. So VACAMethod itself imports nothing from torch/VACA; it
only talks JSON over stdin/stdout to a long-lived worker subprocess
(`adapter/run_vaca.py --serve`) running in a separate `vaca` conda env
(python 3.9). See REPORT.md section 2-3 for the full environment story
(including the torch 1.10.0+cpu "cannot enable executable stack" issue that
forced the upgrade to 1.13.1+cpu).

Retrain policy
---------------
Like DCM's adapter, a single VACAMethod instance persists across the
harness's whole seed/run/try loop (per evaluate.py's `method = MyCausalMethod()`
being constructed once per space), but `data` only changes once per run. We
hash `data`'s raw bytes; whenever the hash changes we kill any previous
worker subprocess and start a fresh one (writes a task JSON with the new
data/graph, waits for the worker to finish training and print "READY").
Within a run, every individual `estimate()` call is answered by sending that
one query over the pipe to the already-trained worker and reading back a
JSON `{"estimate": float}` line -- so a run's `num_tries` loop reuses one
trained VACA model and just gets fresh stochastic Monte Carlo draws per call
(VACA's `get_interventional_distr` samples z from the prior and the decoder
likelihood, so repeated calls are naturally non-deterministic -- this is a
sensible interpretation of "tries" for a stochastic generative model, same
as the DCM adapter's).

Query support
--------------
Only ATE queries (Y, T each a single scalar variable) are supported --
that's what VACA's own intervention API (`get_interventional_distr`,
comparing `do(T=t1)` vs `do(T=t0)`) is built for, and it's also the only
query type used in this repro's toy configs (see REPORT.md). Any other
query type, or any exception/timeout talking to the worker, returns
float('nan'); CausalProfiler.evaluate_error counts NaNs as failures.
"""

import hashlib
import json
import os
import subprocess
import sys
import time

ADAPTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adapter")
RUN_VACA = os.path.join(ADAPTER_DIR, "run_vaca.py")
VACA_CONDA_PYTHON = "/home/rec1/anaconda3/envs/vaca/bin/python"


def _jsonify(obj):
    """Recursively convert numpy types/arrays into JSON-safe python types."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def _query_to_dict(query):
    return {
        "type": query.type.name,
        "vars": {label: [v.name for v in vs] for label, vs in query.vars.items()},
        "vars_values": _jsonify(query.vars_values),
    }


def _data_signature(data):
    h = hashlib.md5()
    for k in sorted(data.keys()):
        h.update(k.encode())
        h.update(data[k].tobytes())
    return h.hexdigest()


class VACAMethod:
    def __init__(
        self,
        max_epochs=15,
        min_epochs=3,
        batch_size=32,
        z_dim=4,
        startup_timeout_s=120,
        query_timeout_s=30,
        tasks_dir=None,
        verbose=False,
    ):
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.startup_timeout_s = startup_timeout_s
        self.query_timeout_s = query_timeout_s
        self.tasks_dir = tasks_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")
        os.makedirs(self.tasks_dir, exist_ok=True)
        self.verbose = verbose

        self._proc = None
        self._data_sig = None
        self._worker_ok = False
        self._call_idx = 0

    # ------------------------------------------------------------------
    def _log(self, msg):
        if self.verbose:
            print(f"[VACAMethod] {msg}", file=sys.stderr)

    def _stop_worker(self):
        if self._proc is not None:
            try:
                self._proc.stdin.write("DONE\n")
                self._proc.stdin.flush()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None
        self._worker_ok = False

    def _start_worker(self, data, graph, index_to_variable):
        self._stop_worker()
        self._call_idx += 1
        task_path = os.path.join(self.tasks_dir, f"_live_task_{os.getpid()}_{self._call_idx}.json")
        task = {
            "space_name": "harness_run",
            "seed": 0,
            "run": self._call_idx,
            "index_to_variable": index_to_variable,
            "graph": {str(k): v for k, v in graph.items()},
            "data": {k: v.tolist() for k, v in data.items()},
            "queries": [],
            "targets": [],
        }
        with open(task_path, "w") as f:
            json.dump(task, f)

        cmd = [
            VACA_CONDA_PYTHON, RUN_VACA, "--serve",
            "--task", task_path,
            "--max_epochs", str(self.max_epochs),
            "--min_epochs", str(self.min_epochs),
            "--batch_size", str(self.batch_size),
            "--z_dim", str(self.z_dim),
        ]
        self._log(f"starting worker: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )

        t0 = time.time()
        line = ""
        while time.time() - t0 < self.startup_timeout_s:
            line = self._proc.stdout.readline()
            if line:
                line = line.strip()
                break
            if self._proc.poll() is not None:
                break
        if line == "READY":
            self._worker_ok = True
        else:
            self._worker_ok = False
            stderr_tail = ""
            try:
                self._proc.kill()
                _, err = self._proc.communicate(timeout=5)
                stderr_tail = (err or "")[-2000:]
            except Exception:
                pass
            self._log(f"worker failed to start (got {line!r}); stderr tail:\n{stderr_tail}")
            self._proc = None

    # ------------------------------------------------------------------
    def estimate(self, query, data, graph, index_to_variable):
        sig = _data_signature(data)
        if sig != self._data_sig:
            self._data_sig = sig
            self._start_worker(data, graph, index_to_variable)

        if not self._worker_ok or self._proc is None:
            return float("nan")

        try:
            qdict = _query_to_dict(query)
            self._proc.stdin.write(json.dumps(qdict) + "\n")
            self._proc.stdin.flush()

            t0 = time.time()
            resp_line = ""
            while time.time() - t0 < self.query_timeout_s:
                resp_line = self._proc.stdout.readline()
                if resp_line:
                    break
                if self._proc.poll() is not None:
                    break
            if not resp_line:
                self._log("query timed out / worker died")
                self._worker_ok = False
                return float("nan")
            resp = json.loads(resp_line)
            return float(resp["estimate"])
        except Exception as e:  # noqa: BLE001
            self._log(f"estimate() exception: {e}")
            return float("nan")

    def __del__(self):
        try:
            self._stop_worker()
        except Exception:
            pass
