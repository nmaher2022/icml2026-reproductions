"""
DCMMethod: a plug-in adapter that wires the DCM (Diffusion-based Causal Models,
Chao et al. 2023, https://github.com/patrickrchao/DiffusionBasedCausalModels)
baseline into the CausalProfiler evaluation harness interface:

    class MyCausalMethod:
        def estimate(self, query, data, graph, index_to_variable) -> float: ...

Design notes (see REPORT.md for full discussion):

- DCM trains one small per-node diffusion model (conditioned on the node's
  parents) for every non-root endogenous variable; root variables get an
  EmpiricalDistribution (bootstrap) mechanism. This is handled by DCM's own
  `create_model_from_graph` + `dowhy.gcm.fit`.
- Retrain policy: the `DCMMethod` instance is constructed ONCE per space (per
  the harness's own loop structure) and its `estimate()` is called many times
  across seeds/runs/tries, but `data` changes once per run (see evaluate.py).
  We compute a cheap signature of `data` (md5 of the concatenated raw bytes)
  and retrain only when that signature changes from the last call, i.e. once
  per run. Within a run, the `num_tries` loop reuses the already-fitted
  per-node diffusion models and just redraws fresh stochastic MC samples per
  try -- a faithful way to interpret "tries" for a stochastic generative
  model (repeated sampling from a fixed fit, not repeated training).
- Query answering mirrors causal_profiler's own ground-truth
  `QueryEstimator` (causal_profiler/query_estimator.py) semantics, but
  substituting the trained DCM model for the true SCM:
    * ATE:  mean(Y | do(T=t1)) - mean(Y | do(T=t0)) via
            dowhy.gcm.interventional_samples.
    * CATE: same, but each arm is filtered/kernel-weighted on X using the
            exact same kernel machinery as causal_profiler
            (causal_profiler.kernels.get_kernel_function), for methodological
            consistency with the ground-truth evaluator.
    * CTF_TE: true abduction-action-prediction counterfactual via
            dowhy.gcm.counterfactual_samples, seeded with the *observed*
            training rows that match V_F (kernel/exact filtered), exactly
            mirroring evaluate_CTF_TE's 3-step procedure. This is the query
            type DCM's diffusion encode/decode machinery was purpose-built
            for.
    * CONDITIONAL: cheap fallback using only the raw observed data (no
            causal model needed at all for L1 queries) -- included for
            completeness though not used in our two toy configs.
- Failure handling: ANY exception during training or inference, or any
  non-finite estimate, causes `estimate()` to return `float('nan')`. The
  harness (`CausalProfiler.evaluate_error`) counts NaN estimates as
  failures, which is exactly how the paper defines "failure rate" ("due to
  numerical issues or exceptions").
"""

import hashlib
import sys
import os

import numpy as np
import pandas as pd
import networkx as nx

DCM_REPO = os.path.join(os.path.dirname(__file__), "DiffusionBasedCausalModels")
if DCM_REPO not in sys.path:
    sys.path.insert(0, DCM_REPO)

from model.diffusion import create_model_from_graph  # noqa: E402

import dowhy.gcm as cy  # noqa: E402
from dowhy.gcm import interventional_samples, counterfactual_samples  # noqa: E402

from causal_profiler.constants import QueryType, VariableDataType, KernelType  # noqa: E402
from causal_profiler.kernels import get_kernel_function  # noqa: E402


def _const_fn(val):
    """Returns a function usable as a dowhy.gcm hard-intervention do(node:=val)."""

    def f(x, _val=val):
        return _val

    return f


class DCMMethod:
    def __init__(
        self,
        hidden_dim=32,
        T=50,
        num_epochs=8,
        batch_size=64,
        lr=1e-3,
        num_mc_samples=200,
        kernel_bandwidth=0.5,
        max_ctf_units=200,
        verbose=False,
        seed=0,
    ):
        self.dcm_params = dict(
            hidden_dim=hidden_dim,
            t_dim=8,
            lr=lr,
            weight_decay=0.001,
            batch_size=batch_size,
            num_epochs=num_epochs,
            use_gpu_if_available=False,
            verbose=verbose,
            T=T,
        )
        self.num_mc_samples = num_mc_samples
        self.kernel_bandwidth = kernel_bandwidth
        self.kernel_fn = get_kernel_function(KernelType.GAUSSIAN)
        self.max_ctf_units = max_ctf_units
        self.rng = np.random.default_rng(seed)

        self._trained_model = None
        self._train_df = None
        self._data_sig = None

        # bookkeeping for the report
        self.n_retrains = 0
        self.n_retrain_failures = 0
        self.n_estimate_calls = 0
        self.n_estimate_failures = 0

    # ------------------------------------------------------------------ #
    # Training / retraining
    # ------------------------------------------------------------------ #
    def _data_signature(self, data):
        h = hashlib.md5()
        for k in sorted(data.keys()):
            h.update(k.encode())
            h.update(np.ascontiguousarray(data[k]).tobytes())
        return h.hexdigest()

    def _maybe_retrain(self, data, graph, index_to_variable):
        sig = self._data_signature(data)
        if sig == self._data_sig and self._trained_model is not None:
            return  # same data as last call within this run -> reuse fit
        self._data_sig = sig

        var_names = list(index_to_variable)
        g = nx.DiGraph()
        g.add_nodes_from(var_names)
        for parent_idx, children_idx in graph.items():
            pname = index_to_variable[parent_idx]
            for cidx in children_idx:
                g.add_edge(pname, index_to_variable[cidx])

        # Toy-scale assumption: variable_dimensionality == 1 (matches our
        # two chosen configs). Take the first (only) column per variable.
        df = pd.DataFrame(
            {name: np.asarray(data[name])[:, 0] for name in var_names if name in data}
        )
        df = df.astype(float)

        self.n_retrains += 1
        try:
            model = create_model_from_graph(g, self.dcm_params)
            cy.fit(model, df)
            self._trained_model = model
            self._train_df = df
        except Exception:
            self._trained_model = None
            self._train_df = None
            self.n_retrain_failures += 1

    # ------------------------------------------------------------------ #
    # Weighted filtering helper (mirrors QueryEstimator.filter_data)
    # ------------------------------------------------------------------ #
    def _weighted_filter(self, df, cond_vars, cond_values):
        n = len(df)
        mask = np.ones(n, dtype=bool)
        weights = np.ones(n)
        for var, val in zip(cond_vars, cond_values):
            val_arr = np.asarray(val).ravel().astype(float)
            col = df[var.name].to_numpy().astype(float)
            if var.variable_type == VariableDataType.DISCRETE:
                mask &= col == val_arr[0]
            else:
                w = self.kernel_fn(col.reshape(-1, 1), val_arr, self.kernel_bandwidth)
                weights *= w
                mask &= w > 1e-10
        if not np.any(mask):
            return None, None
        w_sel = weights[mask]
        if np.sum(w_sel) > 0:
            w_sel = w_sel / np.sum(w_sel)
        else:
            w_sel = np.ones(len(w_sel)) / len(w_sel)
        return mask, w_sel

    def _weighted_mean(self, df, y_name, cond_vars, cond_values):
        mask, w_sel = self._weighted_filter(df, cond_vars, cond_values)
        if mask is None:
            return float("nan")
        y_vals = df[y_name].to_numpy().astype(float)[mask]
        return float(np.sum(y_vals * w_sel))

    # ------------------------------------------------------------------ #
    # Query type handlers
    # ------------------------------------------------------------------ #
    def _estimate_ate(self, query):
        Y_name = query.vars["Y"][0].name
        T_name = query.vars["T"][0].name
        t1_list, t0_list = query.vars_values["T"]
        t1 = float(np.asarray(t1_list[0]).ravel()[0])
        t0 = float(np.asarray(t0_list[0]).ravel()[0])

        df_t1 = interventional_samples(
            self._trained_model,
            {T_name: _const_fn(t1)},
            num_samples_to_draw=self.num_mc_samples,
        )
        df_t0 = interventional_samples(
            self._trained_model,
            {T_name: _const_fn(t0)},
            num_samples_to_draw=self.num_mc_samples,
        )
        return float(df_t1[Y_name].mean() - df_t0[Y_name].mean())

    def _estimate_cate(self, query):
        Y_name = query.vars["Y"][0].name
        T_name = query.vars["T"][0].name
        t1_list, t0_list = query.vars_values["T"]
        t1 = float(np.asarray(t1_list[0]).ravel()[0])
        t0 = float(np.asarray(t0_list[0]).ravel()[0])
        X_vars = query.vars["X"]
        X_values = query.vars_values["X"]

        df_t1 = interventional_samples(
            self._trained_model,
            {T_name: _const_fn(t1)},
            num_samples_to_draw=self.num_mc_samples,
        )
        df_t0 = interventional_samples(
            self._trained_model,
            {T_name: _const_fn(t0)},
            num_samples_to_draw=self.num_mc_samples,
        )
        avg_t1 = self._weighted_mean(df_t1, Y_name, X_vars, X_values)
        avg_t0 = self._weighted_mean(df_t0, Y_name, X_vars, X_values)
        return avg_t1 - avg_t0

    def _estimate_ctf_te(self, query):
        Y_var = query.vars["Y"][0]
        Y_name = Y_var.name
        T_name = query.vars["T"][0].name
        t1_list, t0_list = query.vars_values["T"]
        t1 = float(np.asarray(t1_list[0]).ravel()[0])
        t0 = float(np.asarray(t0_list[0]).ravel()[0])
        y_val = float(np.asarray(query.vars_values["Y"][0]).ravel()[0])
        V_F_vars = query.vars["V_F"]
        V_F_values = query.vars_values["V_F"]

        df_obs = self._train_df
        mask, w_sel = self._weighted_filter(df_obs, V_F_vars, V_F_values)
        if mask is None:
            return float("nan")

        subset = df_obs[mask].reset_index(drop=True)
        if len(subset) > self.max_ctf_units:
            idx = self.rng.choice(len(subset), size=self.max_ctf_units, replace=False)
            subset = subset.iloc[idx].reset_index(drop=True)
            w_sel = w_sel[idx]
            w_sel = w_sel / np.sum(w_sel)

        cf_t1 = counterfactual_samples(
            self._trained_model, {T_name: _const_fn(t1)}, observed_data=subset
        )
        cf_t0 = counterfactual_samples(
            self._trained_model, {T_name: _const_fn(t0)}, observed_data=subset
        )

        if Y_var.variable_type == VariableDataType.DISCRETE:
            p_t1 = float(np.sum(w_sel[(cf_t1[Y_name].to_numpy() == y_val)]))
            p_t0 = float(np.sum(w_sel[(cf_t0[Y_name].to_numpy() == y_val)]))
        else:
            p_t1 = float(np.sum(cf_t1[Y_name].to_numpy() * w_sel))
            p_t0 = float(np.sum(cf_t0[Y_name].to_numpy() * w_sel))
        return p_t1 - p_t0

    def _estimate_conditional(self, query):
        # L1 query: no causal model needed, purely observational.
        Y_var = query.vars["Y"][0]
        Y_name = Y_var.name
        target_y = query.vars_values["Y"][0]
        X_vars = query.vars["X"]
        X_values = query.vars_values["X"]

        df_obs = self._train_df
        mask, w_sel = self._weighted_filter(df_obs, X_vars, X_values)
        if mask is None:
            return float("nan")

        y_vals = df_obs[Y_name].to_numpy().astype(float)[mask]
        if Y_var.variable_type == VariableDataType.DISCRETE:
            target = float(np.asarray(target_y).ravel()[0])
            return float(np.sum(w_sel[y_vals == target]))
        else:
            return float(np.sum(y_vals * w_sel))

    # ------------------------------------------------------------------ #
    # Public interface required by the harness
    # ------------------------------------------------------------------ #
    def estimate(self, query, data, graph, index_to_variable):
        self.n_estimate_calls += 1
        try:
            self._maybe_retrain(data, graph, index_to_variable)
            if self._trained_model is None:
                self.n_estimate_failures += 1
                return float("nan")

            if query.type == QueryType.ATE:
                result = self._estimate_ate(query)
            elif query.type == QueryType.CATE:
                result = self._estimate_cate(query)
            elif query.type == QueryType.CTF_TE:
                result = self._estimate_ctf_te(query)
            elif query.type == QueryType.CONDITIONAL:
                result = self._estimate_conditional(query)
            else:
                result = float("nan")

            if result is None or not np.isfinite(result):
                self.n_estimate_failures += 1
                return float("nan")
            return float(result)
        except Exception:
            self.n_estimate_failures += 1
            return float("nan")
