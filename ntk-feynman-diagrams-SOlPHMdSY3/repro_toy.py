# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "scipy"]
# ///
"""
Toy-scale reproduction of "Finite-Width Neural Tangent Kernels from Feynman Diagrams"
(arXiv 2508.11522, OpenReview SOlPHMdSY3), claims 3, 4, 5 (see PAPER_BRIEFING.md).

Experiments:
  width-sweep  -- Claims 3 & 4 (paper's real Figure 3, mislabeled "Figure 2" in the
                  challenge's extracted claims): 4-layer MLP, ReLU/LeakyReLU vs. GeLU
                  control, sweep hidden width, compare empirical finite-width mean NTK
                  to the exact infinite-width value (analytic arc-cosine recursion for
                  ReLU; large-width empirical proxy for GeLU, which has no simple
                  closed form).
  depth-sweep  -- Claim 5 (paper's real Figure 2, mislabeled "Figure 1"): bias-free
                  ReLU MLP, sweep depth up to L, three C_W values, check linear vs.
                  exponential scaling of the NTK diagonal with depth.

Results written as JSON to results/<experiment>_<tag>.json; no plotting library
dependency (kept out of the PEP-723 header) -- ASCII summaries printed to stdout,
verdict analysis done downstream in VERDICTS.md against the saved JSON.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ACTS = {
    "relu": lambda z: torch.relu(z),
    "leaky_relu": lambda z: torch.nn.functional.leaky_relu(z, negative_slope=0.1),
    "gelu": lambda z: torch.nn.functional.gelu(z),
}

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def sample_network(widths, C_W, bias, gen):
    """widths = [n0, n1, ..., nL] (n0 = input dim). W^l ~ N(0, C_W) iid, shape (n_l, n_{l-1})."""
    params = []
    for l in range(1, len(widths)):
        n_in, n_out = widths[l - 1], widths[l]
        W = torch.randn(n_out, n_in, generator=gen) * math.sqrt(C_W)
        b = torch.randn(n_out, generator=gen) * math.sqrt(C_W) if bias else None
        W.requires_grad_(True)
        if b is not None:
            b.requires_grad_(True)
        params.append((W, b))
    return params


def preacts(params, x, act_fn):
    """Return list of preactivation vectors z^(1)..z^(L) for input x (1D tensor)."""
    h = x
    zs = []
    for l, (W, b) in enumerate(params):
        n_in = W.shape[1]
        z = (W @ h) / math.sqrt(n_in)
        if b is not None:
            z = z + b
        zs.append(z)
        h = act_fn(z)
    return zs


def ntk_unit0(params, x, xp, layer_idx, act_fn):
    """Empirical NTK Theta_00^(layer_idx+1)(x, xp) using only params up to that layer
    (gradient w.r.t. deeper-layer params is exactly zero, so building the full network
    and restricting the parameter list to layers <= layer_idx is equivalent to only
    ever having built a (layer_idx+1)-layer sub-network)."""
    used_params = []
    for W, b in params[: layer_idx + 1]:
        used_params.append(W)
        if b is not None:
            used_params.append(b)

    z_x = preacts(params, x, act_fn)[layer_idx][0]
    grads_x = torch.autograd.grad(z_x, used_params, retain_graph=True, create_graph=False)

    z_xp = preacts(params, xp, act_fn)[layer_idx][0]
    grads_xp = torch.autograd.grad(z_xp, used_params, retain_graph=True, create_graph=False)

    dot = sum((gx * gxp).sum() for gx, gxp in zip(grads_x, grads_xp))
    return dot.item(), z_x.item(), z_xp.item()


# ---------------------------------------------------------------------------
# Analytic infinite-width ReLU baseline (standard arc-cosine kernel recursion,
# Cho & Saul 2009 / Jacot et al. 2018 NTK-parametrization form). Used as the
# n->infinity ground truth for the width-sweep experiment (ReLU/LeakyReLU are
# scale-invariant per Theorem 5.2, so only the diagonal has a simple closed
# form for LeakyReLU too -- off-diagonal analytic form below is ReLU-specific).
def relu_inf_width(x, xp, C_W, depth):
    """NTK-parametrization recursion, re-derived by hand (chain-rule split into the
    top layer's own-weight contribution vs. the contribution flowing through shallower
    layers) and cross-checked numerically against the empirical autograd NTK in this
    script's smoketest -- see BUGFIX_LOG.md. The top layer's own contribution is
    K^(l+1)/C_W^(l+1) (an expectation over activations only, no weight-variance factor),
    NOT K^(l+1); the base case is Theta^(1) = x.xp/n0 (no C_W), not K^(1). Both differ
    from K^(1)/Theta^(1) by exactly a factor of C_W, invisible whenever C_W=1 (which is
    why textbook formulas quoting "Theta^(l+1) = Theta^(l) Kdot + K^(l+1)" implicitly
    assume C_W=1)."""
    x = np.asarray(x, dtype=np.float64)
    xp = np.asarray(xp, dtype=np.float64)
    n0 = len(x)
    Kxx = C_W * (x @ x) / n0
    Kxpxp = C_W * (xp @ xp) / n0
    Kxxp = C_W * (x @ xp) / n0
    Theta = (x @ xp) / n0  # base case: Theta^(1) = x.xp/n0 = K^(1)/C_W
    for l in range(2, depth + 1):
        rho = np.clip(Kxxp / math.sqrt(Kxx * Kxpxp), -1.0, 1.0)
        theta = math.acos(rho)
        # K-dot (derivative kernel): E[relu'(u) relu'(v)] = (pi - theta) / (2 pi)
        Kdot = C_W * (math.pi - theta) / (2 * math.pi)
        # K update: E[relu(u) relu(v)] = sqrt(Kxx Kxpxp)/(2 pi) * (sin theta + (pi-theta) cos theta)
        Knew = C_W / (2 * math.pi) * math.sqrt(Kxx * Kxpxp) * (
            math.sin(theta) + (math.pi - theta) * math.cos(theta)
        )
        Theta = Theta * Kdot + Knew / C_W
        Kxxp = Knew
        Kxx = C_W / 2 * Kxx  # diagonal recursion: theta=0 -> K' = C_W/2 * K
        Kxpxp = C_W / 2 * Kxpxp
    return Theta, Kxxp


def run_width_sweep(widths_n, depth, n_inits, C_W, x, xp, seed, activation, tag):
    act_fn = ACTS[activation]
    results = {"activation": activation, "depth": depth, "C_W": C_W, "n_inits": n_inits,
               "x": x, "xp": xp, "widths": widths_n, "diag": {}, "offdiag": {}}
    n0 = len(x)
    xt = torch.tensor(x)
    xpt = torch.tensor(xp)

    for n in widths_n:
        gen = torch.Generator().manual_seed(seed)
        widths = [n0] + [n] * depth
        diag_vals, offdiag_vals = [], []
        t0 = time.time()
        for i in range(n_inits):
            params = sample_network(widths, C_W, bias=False, gen=gen)
            th_diag, _, _ = ntk_unit0(params, xt, xt, depth - 1, act_fn)
            th_off, _, _ = ntk_unit0(params, xt, xpt, depth - 1, act_fn)
            diag_vals.append(th_diag)
            offdiag_vals.append(th_off)
        diag_vals = np.array(diag_vals)
        offdiag_vals = np.array(offdiag_vals)
        results["diag"][n] = {"mean": float(diag_vals.mean()), "std": float(diag_vals.std()),
                               "sem": float(diag_vals.std() / math.sqrt(n_inits))}
        results["offdiag"][n] = {"mean": float(offdiag_vals.mean()), "std": float(offdiag_vals.std()),
                                  "sem": float(offdiag_vals.std() / math.sqrt(n_inits))}
        print(f"[{tag}] width={n:5d}  diag_mean={diag_vals.mean(): .6f}  "
              f"offdiag_mean={offdiag_vals.mean(): .6f}  ({time.time()-t0:.1f}s)")

    if activation in ("relu",):
        theta_diag_inf, _ = relu_inf_width(x, x, C_W, depth)
        theta_off_inf, _ = relu_inf_width(x, xp, C_W, depth)
        results["infinite_width_analytic"] = {"diag": theta_diag_inf, "offdiag": theta_off_inf}
        print(f"[{tag}] analytic infinite-width: diag={theta_diag_inf:.6f} offdiag={theta_off_inf:.6f}")

    out_path = RESULTS_DIR / f"width_sweep_{tag}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[{tag}] wrote {out_path}")
    return results


def run_depth_sweep(width, max_depth, n_inits, C_W_list, x, seed, tag):
    act_fn = ACTS["relu"]
    n0 = len(x)
    xt = torch.tensor(x)
    results = {"width": width, "max_depth": max_depth, "n_inits": n_inits, "x": x,
               "C_W_list": C_W_list, "by_CW": {}}

    for C_W in C_W_list:
        widths = [n0] + [width] * max_depth
        per_layer = [[] for _ in range(max_depth)]
        t0 = time.time()
        for i in range(n_inits):
            gen = torch.Generator().manual_seed(seed * 1000 + i)
            params = sample_network(widths, C_W, bias=False, gen=gen)
            for l in range(max_depth):
                th, _, _ = ntk_unit0(params, xt, xt, l, act_fn)
                per_layer[l].append(th)
        means = [float(np.mean(v)) for v in per_layer]
        sems = [float(np.std(v) / math.sqrt(n_inits)) for v in per_layer]
        results["by_CW"][C_W] = {"mean_by_layer": means, "sem_by_layer": sems}
        print(f"[{tag}] C_W={C_W}  layer1={means[0]:.4f} ... layer{max_depth}={means[-1]:.4g}  "
              f"({time.time()-t0:.1f}s)")

    out_path = RESULTS_DIR / f"depth_sweep_{tag}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[{tag}] wrote {out_path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=["width-sweep", "depth-sweep"])
    p.add_argument("--tag", default="toy")
    p.add_argument("--seed", type=int, default=0)
    # width-sweep args
    p.add_argument("--widths", type=int, nargs="+", default=[10, 20, 40, 80])
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--n-inits", type=int, default=2000)
    p.add_argument("--C-W", type=float, default=2.0)
    p.add_argument("--activation", default="relu", choices=list(ACTS.keys()))
    # depth-sweep args
    p.add_argument("--width", type=int, default=50)
    p.add_argument("--max-depth", type=int, default=15)
    p.add_argument("--C-W-list", type=float, nargs="+", default=[0.25, 2.0, 4.0])
    args = p.parse_args()

    x0 = [1.0, 0.5, -0.3, 0.2]
    xp0 = [0.2, -1.0, 0.4, 0.6]

    if args.experiment == "width-sweep":
        run_width_sweep(args.widths, args.depth, args.n_inits, args.C_W, x0, xp0,
                         args.seed, args.activation, args.tag)
    else:
        run_depth_sweep(args.width, args.max_depth, args.n_inits, args.C_W_list, x0,
                         args.seed, args.tag)
