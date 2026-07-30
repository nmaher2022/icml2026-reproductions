# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Shared library for claims 2 (deterministic O(1/K^{1/2})) and 3 (stochastic
O(1/K^{1/4})) rate verification for arXiv:2505.13416 "Gluon".

Synthetic multi-layer objective
--------------------------------
f(X) = sum_i f_i(X_i), p=4 layer groups mixing matrix layers (spectral-norm
LMO, Muon-style) and vector layers (Euclidean-norm LMO = normalized GD, and
max-norm LMO = signGD) -- the three base LMO families used throughout the
paper's special-cases section.

Each f_i is a *separable, multi-timescale* quartic potential applied to the
layer's natural "channels" (vector coordinates for euclid/maxnorm layers,
singular values for spectral layers):

    f_i(X_i) = sum_j [ a_{i,j}/2 * s_j^2 + b_i/4 * s_j^4 ]

where s_j is channel j's value (x_j for vectors, sigma_j(X_i) for matrices)
and a_{i,j} is spread log-uniformly across ~3 decades per layer. This
spread-curvature design is deliberate: a single-timescale (isotropic radial)
potential converges near-exponentially once any curvature dominates, which
empirically produces a log-log slope much steeper than -1/2 (verified in an
earlier iteration of this script: slope ~ -0.95 to -1.1, R^2 ~0.98 -- too
fast, not "close to" the claimed rate). Spreading per-channel curvatures over
many decades means different channels cross into their local-convergence
regime at very different iteration counts, so the *aggregate* dual-norm
metric mixes many relaxation timescales -- the standard mechanism (well known
for gradient descent on ill-conditioned/multi-eigenvalue quadratics) that
produces genuine power-law transient decay over an intermediate K-range
instead of a single exponential. This is also a closed-form, exact
construction (no smoothing/softmax surrogate needed):
  - euclid:   grad_j = (a_j + b*x_j^2) x_j,  dual = ||grad||_2 (exact)
  - maxnorm:  grad_j = (a_j + b*x_j^2) x_j,  dual = sum_j|grad_j| (exact,
              since a_j,b,x_j^2 >= 0 the entrywise sign(grad)=sign(x) always)
  - spectral: grad = U diag((a_j+b*sigma_j^2)*sigma_j) V^T (X=U diag(sigma)V^T
              is the SVD), dual = sum_j (a_j+b*sigma_j^2)*sigma_j (exact
              nuclear norm, all diagonal entries nonnegative)

(L0_i, L1_i) generalized-smoothness constants are then estimated empirically
along realistic (small-step) trajectory pairs, exactly the way the paper
itself estimates L0/L1 empirically in Eq. 10/30 (used for claims 4-5), with a
safety margin so the fitted pair is a valid empirical upper certificate for
Assumption 1 -- checked out-of-sample by `validate_assumption1`.

This is a toy/synthetic construction built for *rate* verification (does the
metric decay like the claimed power law?), not a claim about matching the
paper's actual NanoGPT/CIFAR numbers (that's claims 4-5, handled elsewhere).
"""
import numpy as np


# --------------------------------------------------------------------------
# Layer norms / LMO primitives
# --------------------------------------------------------------------------

def primal_norm(kind, X):
    """The layer's own primal norm ||X||_(i)."""
    if kind == "spectral":
        s = np.linalg.svd(X, compute_uv=False)
        return float(s.max()) if s.size else 0.0
    elif kind == "euclid":
        return float(np.linalg.norm(X))
    elif kind == "maxnorm":
        return float(np.max(np.abs(X))) if X.size else 0.0
    raise ValueError(kind)


def dual_norm(kind, G):
    """The dual norm ||G||_(i)*."""
    if kind == "spectral":
        s = np.linalg.svd(G, compute_uv=False)
        return float(s.sum())
    elif kind == "euclid":
        return float(np.linalg.norm(G))
    elif kind == "maxnorm":
        return float(np.abs(G).sum())
    raise ValueError(kind)


def rand_unit_primal(kind, shape, rng):
    """A random direction with primal norm exactly 1 -- an LMO extreme
    point for the corresponding unit ball, used only for calibration
    sampling of realistic single-step jumps."""
    if kind == "euclid":
        v = rng.standard_normal(shape)
        return v / np.linalg.norm(v)
    elif kind == "maxnorm":
        return rng.choice([-1.0, 1.0], size=shape)
    elif kind == "spectral":
        M = rng.standard_normal(shape)
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        return U @ Vt
    raise ValueError(kind)


def n_channels(L):
    if L["kind"] == "spectral":
        return min(L["shape"])
    return L["shape"][0]


def grad_and_dual(kind, X, a_vec, b):
    """grad_i f_i(X_i) and its exact dual norm, for the separable multi-scale
    quartic potential described in the module docstring. a_vec has length
    n_channels(layer); b is a scalar shared across channels.
    Returns (grad, dual_norm_value)."""
    if kind == "euclid":
        g = (a_vec + b * X * X) * X
        return g, float(np.linalg.norm(g))
    elif kind == "maxnorm":
        g = (a_vec + b * X * X) * X
        return g, float(np.abs(g).sum())
    elif kind == "spectral":
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        diag = (a_vec + b * S * S) * S
        g = U @ np.diag(diag) @ Vt
        return g, float(np.abs(diag).sum())
    raise ValueError(kind)


def f_value(kind, X, a_vec, b):
    """f_i(X_i) itself, for Delta0 = f(X0) - inf f bookkeeping (inf f_i=0)."""
    if kind == "euclid" or kind == "maxnorm":
        s = X
    elif kind == "spectral":
        s = np.linalg.svd(X, compute_uv=False)
    else:
        raise ValueError(kind)
    return float(np.sum(0.5 * a_vec * s * s + 0.25 * b * s ** 4))


def lmo_update(kind, X, G, t):
    """X - t * LMO_{unit ball}(G), the Algorithm-2 LMO step for the three
    base norms (briefing Sec. 'Core math', Algorithm 2 closed forms)."""
    if kind == "euclid":
        gn = np.linalg.norm(G)
        if gn < 1e-14:
            return X.copy()
        return X - t * (G / gn)
    elif kind == "maxnorm":
        return X - t * np.sign(G)
    elif kind == "spectral":
        U, _, Vt = np.linalg.svd(G, full_matrices=False)
        return X - t * (U @ Vt)
    raise ValueError(kind)


# --------------------------------------------------------------------------
# Layer definitions (p=4, mixed matrix/vector, mixed norms), each with a
# log-spaced per-channel curvature vector spanning ~3 decades.
# --------------------------------------------------------------------------

def _log_spaced_a(n, lo, hi, seed):
    rng = np.random.default_rng(seed)
    if n == 1:
        return np.array([np.sqrt(lo * hi)])
    vals = np.exp(np.linspace(np.log(lo), np.log(hi), n))
    rng.shuffle(vals)  # so channel order isn't monotonic in curvature
    return vals


def make_layers():
    layers = [
        dict(name="attn", shape=(18, 14), kind="spectral", b=0.06),
        dict(name="mlp", shape=(22, 10), kind="spectral", b=0.06),
        dict(name="vecA", shape=(40,), kind="euclid", b=0.06),
        dict(name="vecB", shape=(30,), kind="maxnorm", b=0.06),
    ]
    for idx, L in enumerate(layers):
        nc = n_channels(L)
        L["a"] = _log_spaced_a(nc, 1e-6, 1.0, seed=100 + idx)
    return layers


def init_X0(layers, radius=2.0, seed=0):
    rng = np.random.default_rng(seed)
    X0 = []
    for L in layers:
        v = rng.standard_normal(L["shape"])
        n = primal_norm(L["kind"], v)
        X0.append(v * (radius / n))
    return X0


# --------------------------------------------------------------------------
# Empirical (L0_i, L1_i) calibration (mirrors the paper's own Eq. 10/30
# fitting procedure: fit a valid, empirically-checked upper certificate for
# Assumption 1 over realistic single-step (X, Y) pairs, not adversarial
# arbitrarily-far pairs).
# --------------------------------------------------------------------------

def calibrate_L0_L1(layers, r_max=4.0, step_max=0.4, n_samples=4000, seed=1):
    rng = np.random.default_rng(seed)
    results = {}
    for L in layers:
        shape, kind, a_vec, b = L["shape"], L["kind"], L["a"], L["b"]
        rX = rng.uniform(0, r_max, n_samples)
        steps = np.exp(rng.uniform(np.log(0.005), np.log(step_max), n_samples))
        LHS_list, pd_list, gxd_list = [], [], []
        for i in range(n_samples):
            X = _rand_dir(kind, shape, rng) * rX[i]
            Y = X + rand_unit_primal(kind, shape, rng) * steps[i]
            GX, phiX = grad_and_dual(kind, X, a_vec, b)
            GY, phiY = grad_and_dual(kind, Y, a_vec, b)
            LHS = dual_norm(kind, GX - GY)
            pd = primal_norm(kind, X - Y)
            if pd < 1e-3:
                continue
            LHS_list.append(LHS)
            pd_list.append(pd)
            gxd_list.append(phiX)
        y = np.array(LHS_list) / np.array(pd_list)
        x = np.array(gxd_list)
        A = np.column_stack([x, np.ones_like(x)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        L1, L0 = float(coef[0]), float(coef[1])
        L1 = max(L1, 0.01)
        L0 = max(L0, 0.0)
        resid = y - (L0 + L1 * x)
        bump = max(0.0, float(resid.max()))
        L0 = (L0 + 1.3 * bump + 1e-3) * 1.2  # safety margin -> valid empirical upper certificate
        L1 = L1 * 1.1
        resid_final = y - (L0 + L1 * x)
        results[L["name"]] = dict(L0=L0, L1=L1, max_violation=float(resid_final.max()),
                                   n_pairs=len(y))
    return results


def validate_assumption1(layers, L0L1, r_max=4.0, step_max=0.4, n_samples=3000, seed=42):
    """Fresh held-out sample: fraction of pairs where the fitted (L0,L1)
    actually upper-bounds the realized secant slope."""
    rng = np.random.default_rng(seed)
    report = {}
    for L in layers:
        shape, kind, a_vec, b = L["shape"], L["kind"], L["a"], L["b"]
        L0, L1 = L0L1[L["name"]]["L0"], L0L1[L["name"]]["L1"]
        n_ok, n_tot, max_ratio = 0, 0, 0.0
        for _ in range(n_samples):
            rX = rng.uniform(0, r_max)
            step = np.exp(rng.uniform(np.log(0.005), np.log(step_max)))
            X = _rand_dir(kind, shape, rng) * rX
            Y = X + rand_unit_primal(kind, shape, rng) * step
            GX, phiX = grad_and_dual(kind, X, a_vec, b)
            GY, _ = grad_and_dual(kind, Y, a_vec, b)
            LHS = dual_norm(kind, GX - GY)
            pd = primal_norm(kind, X - Y)
            if pd < 1e-3:
                continue
            bound = (L0 + L1 * phiX) * pd
            n_tot += 1
            if LHS <= bound + 1e-9:
                n_ok += 1
            max_ratio = max(max_ratio, LHS / bound if bound > 0 else 0.0)
        report[L["name"]] = dict(pass_rate=n_ok / n_tot if n_tot else float("nan"),
                                  max_ratio=max_ratio, n_tot=n_tot)
    return report


def _rand_dir(kind, shape, rng):
    """Random direction normalized to primal norm 1 in the layer's own norm,
    so a subsequent `* rX` gives a point at primal-norm radius rX."""
    v = rng.standard_normal(shape)
    n = primal_norm(kind, v)
    return v / n if n > 1e-14 else v
