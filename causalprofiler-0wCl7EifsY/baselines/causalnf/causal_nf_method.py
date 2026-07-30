"""
CausalNFMethod: an adapter plugging CausalNF (Javaloy et al., 2023/2024;
repo: https://github.com/adrianjav/causal-flows, pip name "causalflows")
into the CausalProfiler evaluation harness interface:

    class MyCausalMethod:
        def estimate(self, query, data, graph, index_to_variable) -> float

Design notes (see REPORT.md for the full writeup):

- The harness instantiates the method ONCE per space, but calls
  `generate_samples_and_queries()` fresh for every run (new SCM + new data).
  Within a run, `estimate()` is called once per query x num_tries, always with
  the SAME `data` dict. We therefore train a fresh CausalNSF flow the first
  time we see a given run's data, and cache it keyed on a CONTENT HASH of
  `data` (not `id(data)` -- Python can recycle object ids across runs once the
  previous run's `data` dict is garbage collected, which would silently reuse
  a stale model). Multiple `num_tries` for the same run reuse the trained
  flow; try-to-try variance comes from the flow's own sampling stochasticity,
  not retraining.

- `index_to_variable` is already the topological order of the *visible*
  variables (verified empirically against the harness -- for every edge
  parent->child in `graph`, parent's index < child's index). We use this
  directly as the CausalNSF `order=` argument.

- CausalNF/causal-flows is a continuous normalizing-flow library (Zuko-based
  rational-quadratic splines). It has no discrete/mixed-data mode. For
  Regional-Discrete-SCM-style data (integer category codes) we deliberately do
  NOT apply any continuous relaxation or dequantization trick -- we feed the
  integer-coded data to the flow exactly as a real CausalNF user would if they
  (mis)applied the method to discrete data. If the resulting density-fitting
  objective diverges (loss/gradients become non-finite -- expected when a
  continuous density tries to concentrate on a discrete/atomic distribution)
  we ABORT TRAINING and cache `None` for that run, which makes every
  subsequent `estimate()` call for that run return NaN. NaN is treated as the
  correct/expected output in that case, not a bug to work around.

- Only ATE, CATE, and CONDITIONAL (L1/L2) queries are answered. Other query
  types (ITE, DTE, CDTE, OIP, Ctf-*) return NaN -- an explicit "not
  implemented for this toy adapter" signal, not a silent wrong answer.

- Only `variable_dimensionality == 1` per node is supported (matches all toy
  configs used here). Any run with a multi-dimensional variable makes the fit
  return None (NaN for that whole run) rather than guessing an autoregressive
  order for sub-dimensions.
"""

import hashlib
import math
from contextlib import ExitStack

import numpy as np
import torch

import causalflows

try:
    from causal_profiler.constants import QueryType, VariableDataType
except ImportError:  # pragma: no cover
    QueryType = None
    VariableDataType = None


def _hash_data(data):
    """Content hash of the data dict, robust to id() recycling across runs."""
    h = hashlib.md5()
    for name in sorted(data.keys()):
        arr = np.asarray(data[name])
        h.update(name.encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


class CausalNFMethod:
    def __init__(
        self,
        hidden_features=(64, 64),
        epochs=200,
        lr=1e-3,
        n_samples_effect=2000,
        bins=8,
        verbose=False,
    ):
        self.hidden_features = list(hidden_features)
        self.epochs = epochs
        self.lr = lr
        self.n_samples_effect = n_samples_effect
        self.bins = bins
        self.verbose = verbose
        self._cache = {}
        # Diagnostics, useful for REPORT.md / debugging, not used by the harness.
        self.stats = {"fits_attempted": 0, "fits_failed": 0, "fits_ok": 0}

    # ------------------------------------------------------------------
    # Public interface required by the harness
    # ------------------------------------------------------------------
    def estimate(self, query, data, graph, index_to_variable):
        key = _hash_data(data)
        if key not in self._cache:
            self._cache[key] = self._fit(data, graph, index_to_variable)
        model = self._cache[key]
        if model is None:
            return float("nan")
        try:
            return self._answer_query(query, model)
        except Exception:
            return float("nan")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _fit(self, data, graph, index_to_variable):
        self.stats["fits_attempted"] += 1
        try:
            var_names = list(index_to_variable)
            if len(var_names) == 0:
                self.stats["fits_failed"] += 1
                return None

            dims = [np.asarray(data[v]).shape[1] for v in var_names]
            if any(d != 1 for d in dims):
                # Scope limit: only single-dimensional variables supported.
                self.stats["fits_failed"] += 1
                return None

            X = np.concatenate(
                [np.asarray(data[v], dtype=np.float64) for v in var_names], axis=1
            )  # (n, D)
            D = X.shape[1]
            n = X.shape[0]
            if D < 1 or n < 2:
                self.stats["fits_failed"] += 1
                return None

            means = X.mean(axis=0)
            stds = X.std(axis=0)
            stds = np.where(stds < 1e-6, 1.0, stds)
            Xn = (X - means) / stds
            if not np.all(np.isfinite(Xn)):
                self.stats["fits_failed"] += 1
                return None

            Xt = torch.tensor(Xn, dtype=torch.float32)
            order = tuple(range(D))

            flow = causalflows.flows.CausalNSF(
                D, 0, order=order, hidden_features=self.hidden_features, bins=self.bins
            )
            opt = torch.optim.Adam(flow.parameters(), lr=self.lr)
            ctx = torch.zeros(Xt.shape[0], 0)

            flow.train()
            diverged = False
            last_loss = None
            for epoch in range(self.epochs):
                opt.zero_grad()
                loss = -flow(ctx).log_prob(Xt).mean()
                if not torch.isfinite(loss):
                    diverged = True
                    break
                loss.backward()
                grad_ok = all(
                    (p.grad is None) or torch.isfinite(p.grad).all()
                    for p in flow.parameters()
                )
                if not grad_ok:
                    diverged = True
                    break
                opt.step()
                last_loss = float(loss.item())
                if self.verbose and epoch % 50 == 0:
                    print(f"    [CausalNFMethod] epoch {epoch} loss {last_loss:.4f}")

            if diverged or last_loss is None or not math.isfinite(last_loss):
                self.stats["fits_failed"] += 1
                return None

            flow.eval()
            self.stats["fits_ok"] += 1
            return {
                "flow": flow,
                "var_names": var_names,
                "name_to_idx": {v: i for i, v in enumerate(var_names)},
                "means": means,
                "stds": stds,
                "D": D,
                "final_loss": last_loss,
            }
        except Exception:
            self.stats["fits_failed"] += 1
            return None

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------
    def _sample_observational(self, model, n):
        flow = model["flow"]
        ctx = torch.zeros(1, 0)
        with torch.no_grad():
            x = flow(ctx).sample((n,)).reshape(-1, model["D"])
        x = x.detach().numpy().astype(np.float64)
        return x * model["stds"] + model["means"]

    def _sample_interventional(self, model, index_value_pairs, n):
        flow = model["flow"]
        ctx = torch.zeros(1, 0)
        with torch.no_grad():
            dist = flow(ctx)
            with ExitStack() as stack:
                cur = dist
                for idx, val in index_value_pairs:
                    cur = stack.enter_context(cur.intervene(idx, float(val)))
                x = cur.sample((n,)).reshape(-1, model["D"])
        x = x.detach().numpy().astype(np.float64)
        return x * model["stds"] + model["means"]

    # ------------------------------------------------------------------
    # Query answering
    # ------------------------------------------------------------------
    def _build_intervention_pairs(self, model, t_vars, t_values):
        pairs = []
        for var, val in zip(t_vars, t_values):
            if var.name not in model["name_to_idx"]:
                return None
            idx = model["name_to_idx"][var.name]
            raw = float(np.asarray(val).ravel()[0])
            norm = (raw - model["means"][idx]) / model["stds"][idx]
            pairs.append((idx, norm))
        return pairs

    def _conditioning_weights(self, model, samples, x_vars, x_values):
        """Weight each sampled row by how well it matches X=x.
        DISCRETE X vars -> exact match (rounded); CONTINUOUS -> Gaussian kernel.
        Mirrors causal_profiler.query_estimator.filter_data's semantics.
        """
        weights = np.ones(samples.shape[0])
        for var, val in zip(x_vars, x_values):
            if var.name not in model["name_to_idx"]:
                return None
            idx = model["name_to_idx"][var.name]
            raw = float(np.asarray(val).ravel()[0])
            col = samples[:, idx]
            is_discrete = (
                VariableDataType is not None
                and getattr(var, "variable_type", None) == VariableDataType.DISCRETE
            )
            if is_discrete:
                w = (np.round(col) == round(raw)).astype(np.float64)
            else:
                std = model["stds"][idx] if model["stds"][idx] > 1e-6 else 1.0
                bw = max(0.25 * std, 1e-3)
                w = np.exp(-0.5 * ((col - raw) / bw) ** 2)
            weights = weights * w
        return weights

    def _answer_query(self, query, model):
        if QueryType is None:
            return float("nan")
        if query.type == QueryType.ATE:
            return self._answer_ate(query, model)
        elif query.type == QueryType.CATE:
            return self._answer_cate(query, model)
        elif query.type == QueryType.CONDITIONAL:
            return self._answer_conditional(query, model)
        else:
            # Out of scope for this toy adapter (ITE, DTE, CDTE, OIP, Ctf-*).
            return float("nan")

    def _answer_ate(self, query, model):
        T_vars = query.vars.get("T", [])
        Y_vars = query.vars.get("Y", [])
        if not T_vars or not Y_vars:
            return float("nan")
        Y_var = Y_vars[0]
        if Y_var.name not in model["name_to_idx"]:
            return float("nan")
        y_idx = model["name_to_idx"][Y_var.name]

        T1_values, T0_values = query.vars_values["T"]
        pairs1 = self._build_intervention_pairs(model, T_vars, T1_values)
        pairs0 = self._build_intervention_pairs(model, T_vars, T0_values)
        if pairs1 is None or pairs0 is None:
            return float("nan")

        x1 = self._sample_interventional(model, pairs1, self.n_samples_effect)
        x0 = self._sample_interventional(model, pairs0, self.n_samples_effect)
        y1 = np.nanmean(x1[:, y_idx])
        y0 = np.nanmean(x0[:, y_idx])
        if not (np.isfinite(y1) and np.isfinite(y0)):
            return float("nan")
        return float(y1 - y0)

    def _answer_cate(self, query, model):
        T_vars = query.vars.get("T", [])
        X_vars = query.vars.get("X", [])
        Y_vars = query.vars.get("Y", [])
        if not T_vars or not Y_vars:
            return float("nan")
        Y_var = Y_vars[0]
        if Y_var.name not in model["name_to_idx"]:
            return float("nan")
        y_idx = model["name_to_idx"][Y_var.name]

        T1_values, T0_values = query.vars_values["T"]
        X_values = query.vars_values.get("X", [])
        pairs1 = self._build_intervention_pairs(model, T_vars, T1_values)
        pairs0 = self._build_intervention_pairs(model, T_vars, T0_values)
        if pairs1 is None or pairs0 is None:
            return float("nan")

        x1 = self._sample_interventional(model, pairs1, self.n_samples_effect)
        x0 = self._sample_interventional(model, pairs0, self.n_samples_effect)

        avg1 = self._weighted_mean(model, x1, X_vars, X_values, y_idx)
        avg0 = self._weighted_mean(model, x0, X_vars, X_values, y_idx)
        if avg1 is None or avg0 is None:
            return float("nan")
        return float(avg1 - avg0)

    def _answer_conditional(self, query, model):
        X_vars = query.vars.get("X", [])
        Y_vars = query.vars.get("Y", [])
        if not Y_vars:
            return float("nan")
        Y_var = Y_vars[0]
        if Y_var.name not in model["name_to_idx"]:
            return float("nan")
        y_idx = model["name_to_idx"][Y_var.name]

        X_values = query.vars_values.get("X", [])
        samples = self._sample_observational(model, self.n_samples_effect)
        weights = self._conditioning_weights(model, samples, X_vars, X_values)
        if weights is None or weights.sum() < 1e-8:
            return float("nan")
        w_norm = weights / weights.sum()

        is_discrete = (
            VariableDataType is not None
            and getattr(Y_var, "variable_type", None) == VariableDataType.DISCRETE
        )
        if is_discrete:
            y_val = query.vars_values.get("Y", [None])[0]
            if y_val is None:
                return float("nan")
            raw = float(np.asarray(y_val).ravel()[0])
            matches = (np.round(samples[:, y_idx]) == round(raw)).astype(np.float64)
            return float(np.sum(matches * w_norm))
        else:
            return float(np.sum(samples[:, y_idx] * w_norm))

    def _weighted_mean(self, model, samples, x_vars, x_values, y_idx):
        if not x_vars:
            return float(np.nanmean(samples[:, y_idx]))
        weights = self._conditioning_weights(model, samples, x_vars, x_values)
        if weights is None or weights.sum() < 1e-8:
            return None
        w_norm = weights / weights.sum()
        return float(np.sum(samples[:, y_idx] * w_norm))
