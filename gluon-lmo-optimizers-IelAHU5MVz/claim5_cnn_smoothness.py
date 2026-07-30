# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
"""Claim 5: toy CNN / synthetic-CIFAR-style layer-wise smoothness under unScion(CNN) updates.

Paper claim (verbatim): "On a CNN trained on CIFAR-10, estimated smoothness constants also
satisfy L^0_i approximately 0, with a two-orders-of-magnitude spread in L^1_i across layers,
motivating per-layer learning-rate heterogeneity."

Paper's real setup: full CNN on full CIFAR-10, A100 GPU, unScion optimizer, full-batch
gradients, no momentum, no LR decay, ~80 epochs (Appendix E.3/E.4). This script CANNOT match
that scale (no torch in this repo, CPU only) -- it is a hand-rolled numpy CNN on a small
synthetic image-classification task, trained for many full-batch unScion-style steps. The goal
is only to check the *qualitative pattern*: L^0_i ~= 0 across layer groups, and the
classification head's L^1 sitting orders of magnitude below the conv/bias layers' L^1.

Architecture (all forward/backward hand-implemented, no autodiff):
  conv1 (Cin=2 -> Cout=4, k=3) + bias1 -> ReLU
  conv2 (Cin=4 -> Cout=4, k=3) + bias2 -> ReLU
  flatten -> head (linear, sign-updated last group, no bias)
  softmax cross-entropy loss, 3 synthetic classes.

Layer groups matching the unScion(CNN) formulas from the briefing (Eq. verbatim):
  bias_i:  X^{k+1} = X^k - t*sqrt(C_out)*g/||g||_2                      (normalized GD, Euclidean dual)
  conv_i:  X^{k+1} = X^k - t*(1/k^2)*sqrt(C_out/C_in)*U V^T             (spectral LMO / SVD of reshaped grad)
  head:    X^{k+1} = X^k - (t/n_p)*sign(g)                              (entrywise sign / max-norm LMO)

Per-layer dual/primal norms used in Eq. 10 are the ones *consistent* with each layer's LMO norm
(Euclidean self-dual for biases; nuclear/spectral pair scaled by the conv constant; entrywise
L1/Linf pair scaled by 1/n_p and n_p for the head) -- these are the norms implied by the update
rules above, not just plain Euclidean norms, since the paper's whole point is that each layer
uses its own natural norm.

Eq. 10:  L_hat_i[k] = ||grad_i^{k+1} - grad_i^k||_(i)* / ||X_i^{k+1} - X_i^k||_(i)
         L_hat_i^approx[k] = L0_i + L1_i * ||grad_i^{k+1}||_(i)*
Full-batch + deterministic here, so f_{xi^{k+1}} = f_{xi^k} = f (no stochasticity needed).

Eq. 30 fit: hinge-penalized least squares (penalize underestimation more), per layer group,
solved with scipy.optimize.minimize (2 free params L0>=0, L1>=0).

Smoketest (K=20, tiny dims) runs first with a numerical-gradient check; only if it passes clean
(no NaN, gradients flowing, loss finite) does the script scale up to the toy run (K=1500).
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import minimize
import csv

rng_global = np.random.default_rng(0)

# --------------------------------------------------------------------------------------
# Synthetic data: small multi-channel "images" with a learnable per-class spatial pattern.
# --------------------------------------------------------------------------------------
def make_data(n_per_class, H, W, C, n_classes, seed, noise=0.35, label_noise_frac=0.0):
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    templates = [
        yy / (H - 1) - 0.5,                              # class 0: horizontal gradient
        xx / (W - 1) - 0.5,                               # class 1: vertical gradient
        ((yy + xx) % 2) - 0.5,                             # class 2: checkerboard
    ]
    assert n_classes <= len(templates)
    Xs, ys = [], []
    for c in range(n_classes):
        base = templates[c]
        for _ in range(n_per_class):
            img = np.zeros((C, H, W))
            for ch in range(C):
                scale = 1.0 if ch == 0 else 0.6
                img[ch] = scale * base + rng.normal(0, noise, size=(H, W))
            Xs.append(img)
            ys.append(c)
    X = np.stack(Xs, axis=0)
    y = np.array(ys, dtype=int)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    if label_noise_frac > 0:
        # Flip a fixed (deterministic, seeded) fraction of labels to a random wrong class.
        # This creates an irreducible Bayes-error floor so full-batch training cannot drive
        # the loss/gradients all the way to float-precision zero -- keeps the trajectory in
        # an informative regime for the whole run, matching the fact that the paper's real
        # CIFAR-10 run never perfectly fits either. Deterministic given `seed`, so this is
        # still a fixed full-batch objective, not stochastic per-step noise.
        n_flip = int(round(label_noise_frac * len(y)))
        flip_idx = rng.choice(len(y), size=n_flip, replace=False)
        for i in flip_idx:
            choices = [c for c in range(n_classes) if c != y[i]]
            y[i] = rng.choice(choices)
    return X, y


# --------------------------------------------------------------------------------------
# Hand-rolled conv layer (valid, stride 1) via sliding_window_view + einsum.
# --------------------------------------------------------------------------------------
def conv_forward(X, W, b):
    N, Cin, H, Wd = X.shape
    Cout, _, k, _ = W.shape
    windows = sliding_window_view(X, (k, k), axis=(2, 3))  # (N,Cin,Ho,Wo,k,k)
    out = np.einsum("ncijpq,ocpq->noij", windows, W) + b[None, :, None, None]
    return out, windows


def conv_backward(dOut, W, windows):
    N, Cin, H, Wd = windows.shape[0], windows.shape[1], None, None
    Cout, Cin_w, k, _ = W.shape
    dW = np.einsum("noij,ncijpq->ocpq", dOut, windows)
    db = dOut.sum(axis=(0, 2, 3))
    pad = k - 1
    dOut_pad = np.pad(dOut, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    windows2 = sliding_window_view(dOut_pad, (k, k), axis=(2, 3))  # (N,Cout,H,Wd,k,k)
    Wf = W[:, :, ::-1, ::-1]
    dX = np.einsum("noyxpq,ocpq->ncyx", windows2, Wf)
    return dX, dW, db


def relu_forward(x):
    return np.maximum(x, 0.0)


def relu_backward(dOut, pre_act):
    return dOut * (pre_act > 0)


def softmax_ce(logits, y):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    N = logits.shape[0]
    loss = -np.log(np.clip(probs[np.arange(N), y], 1e-12, None)).mean()
    dlogits = probs.copy()
    dlogits[np.arange(N), y] -= 1.0
    dlogits /= N
    return loss, dlogits


# --------------------------------------------------------------------------------------
# Full model forward/backward.
# --------------------------------------------------------------------------------------
def forward_backward(params, X, y):
    W1, b1, W2, b2, Wh = params["conv1_W"], params["conv1_b"], params["conv2_W"], params["conv2_b"], params["head_W"]

    a1, win1 = conv_forward(X, W1, b1)
    h1 = relu_forward(a1)
    a2, win2 = conv_forward(h1, W2, b2)
    h2 = relu_forward(a2)
    N = X.shape[0]
    flat = h2.reshape(N, -1)
    logits = flat @ Wh.T

    loss, dlogits = softmax_ce(logits, y)

    dWh = dlogits.T @ flat
    dflat = dlogits @ Wh
    dh2 = dflat.reshape(h2.shape)

    da2 = relu_backward(dh2, a2)
    dh1, dW2, db2 = conv_backward(da2, W2, win2)

    da1 = relu_backward(dh1, a1)
    dX, dW1, db1 = conv_backward(da1, W1, win1)

    grads = {"conv1_W": dW1, "conv1_b": db1, "conv2_W": dW2, "conv2_b": db2, "head_W": dWh}
    return loss, grads


def predict_acc(params, X, y):
    W1, b1, W2, b2, Wh = params["conv1_W"], params["conv1_b"], params["conv2_W"], params["conv2_b"], params["head_W"]
    a1, _ = conv_forward(X, W1, b1)
    h1 = relu_forward(a1)
    a2, _ = conv_forward(h1, W2, b2)
    h2 = relu_forward(a2)
    flat = h2.reshape(X.shape[0], -1)
    logits = flat @ Wh.T
    return (logits.argmax(axis=1) == y).mean()


# --------------------------------------------------------------------------------------
# unScion(CNN) update rules (verbatim from briefing).
# --------------------------------------------------------------------------------------
def update_bias(b, g, t):
    n = np.linalg.norm(g) + 1e-12
    Cout = b.shape[0]
    return b - t * np.sqrt(Cout) * g / n


def update_conv(Wc, g, t):
    Cout, Cin, k, _ = Wc.shape
    G = g.reshape(Cout, Cin * k * k)
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    UVt = (U @ Vt).reshape(Cout, Cin, k, k)
    coeff = t * (1.0 / k ** 2) * np.sqrt(Cout / Cin)
    return Wc - coeff * UVt


def update_head(Wh, g, t):
    n_p = Wh.shape[1]
    return Wh - (t / n_p) * np.sign(g)


# --------------------------------------------------------------------------------------
# Per-layer dual (grad) / primal (delta) norms, consistent with each layer's LMO norm.
# --------------------------------------------------------------------------------------
def bias_dual(g, Cout):
    # ||.||_(i) = (1/sqrt(Cout)) * ||.||_2 (primal), so dual = sqrt(Cout) * ||.||_2 --
    # matches update_bias's t*sqrt(Cout)*g/||g||_2 via the same norm-scaling duality
    # verified in claim1 (unscion_cnn_bias case, c=1/sqrt(Cout)).
    return np.sqrt(Cout) * np.linalg.norm(g)


def bias_primal(d, Cout):
    return np.linalg.norm(d) / np.sqrt(Cout)


def conv_dual(g, k, Cin, Cout):
    G = g.reshape(Cout, Cin * k * k)
    s = np.linalg.svd(G, compute_uv=False)
    c = (1.0 / k ** 2) * np.sqrt(Cout / Cin)
    return c * s.sum()  # c * nuclear norm


def conv_primal(d, k, Cin, Cout):
    D = d.reshape(Cout, Cin * k * k)
    s = np.linalg.svd(D, compute_uv=False)
    c = (1.0 / k ** 2) * np.sqrt(Cout / Cin)
    return s.max() / c  # spectral norm / c


def head_dual(g, n_p):
    return np.abs(g).sum() / n_p


def head_primal(d, n_p):
    return n_p * np.abs(d).max()


# --------------------------------------------------------------------------------------
# Numerical gradient check (smoketest correctness gate).
# --------------------------------------------------------------------------------------
def gradcheck():
    rng = np.random.default_rng(1)
    X, y = make_data(2, 6, 6, 2, 2, seed=1, noise=0.3)  # tiny: 4 samples, 2 classes
    Cin, Cout1, Cout2, k = 2, 2, 2, 3
    H = X.shape[2]
    Ho1 = H - k + 1
    Ho2 = Ho1 - k + 1
    n_feat = Cout2 * Ho2 * Ho2
    params = {
        "conv1_W": rng.normal(0, 0.3, (Cout1, Cin, k, k)),
        "conv1_b": rng.normal(0, 0.1, (Cout1,)),
        "conv2_W": rng.normal(0, 0.3, (Cout2, Cout1, k, k)),
        "conv2_b": rng.normal(0, 0.1, (Cout2,)),
        "head_W": rng.normal(0, 0.3, (2, n_feat)),
    }
    loss0, grads = forward_backward(params, X, y)
    assert np.isfinite(loss0), "gradcheck: loss not finite"
    eps = 1e-5
    max_rel_err = 0.0
    for name, g in grads.items():
        p = params[name]
        idx = list(np.ndindex(p.shape))
        rng.shuffle(idx)
        for i in idx[:4]:
            orig = p[i]
            p[i] = orig + eps
            lp, _ = forward_backward(params, X, y)
            p[i] = orig - eps
            lm, _ = forward_backward(params, X, y)
            p[i] = orig
            num_g = (lp - lm) / (2 * eps)
            ana_g = g[i]
            denom = max(abs(num_g), abs(ana_g), 1e-6)
            rel_err = abs(num_g - ana_g) / denom
            max_rel_err = max(max_rel_err, rel_err)
    print(f"[gradcheck] max relative error over sampled entries: {max_rel_err:.3e}")
    assert max_rel_err < 5e-3, f"gradcheck FAILED, max rel err {max_rel_err}"
    print("[gradcheck] PASSED")


def smoketest():
    print("=== SMOKETEST: tiny CNN, K=20 steps ===")
    X, y = make_data(8, 8, 8, 2, 3, seed=2, noise=0.35)
    Cin, Cout1, Cout2, k, n_classes = 2, 3, 3, 3, 3
    H = X.shape[2]
    Ho2 = H - 2 * (k - 1)
    n_feat = Cout2 * Ho2 * Ho2
    rng = np.random.default_rng(2)
    params = {
        "conv1_W": rng.normal(0, 0.3, (Cout1, Cin, k, k)),
        "conv1_b": np.zeros(Cout1),
        "conv2_W": rng.normal(0, 0.3, (Cout2, Cout1, k, k)),
        "conv2_b": np.zeros(Cout2),
        "head_W": rng.normal(0, 0.2, (n_classes, n_feat)),
    }
    t = 0.1
    losses = []
    for step in range(20):
        loss, grads = forward_backward(params, X, y)
        assert np.isfinite(loss), f"NaN loss at smoketest step {step}"
        for name, g in grads.items():
            assert np.all(np.isfinite(g)), f"NaN grad in {name} at step {step}"
            assert np.linalg.norm(g) > 0, f"zero grad in {name} at step {step} -- not flowing"
        losses.append(loss)
        params["conv1_b"] = update_bias(params["conv1_b"], grads["conv1_b"], t)
        params["conv2_b"] = update_bias(params["conv2_b"], grads["conv2_b"], t)
        params["conv1_W"] = update_conv(params["conv1_W"], grads["conv1_W"], t)
        params["conv2_W"] = update_conv(params["conv2_W"], grads["conv2_W"], t)
        params["head_W"] = update_head(params["head_W"], grads["head_W"], t)
    print(f"[smoketest] loss trajectory: {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "smoketest: loss did not decrease at all"
    print("[smoketest] PASSED: no NaNs, gradients flow, loss decreasing.\n")


# --------------------------------------------------------------------------------------
# Main toy run + Eq. 10 trajectory logging + Eq. 30 fit.
# --------------------------------------------------------------------------------------
def main_run(K=1500, t=0.08, seed=42, noise=0.35, grad_floor_frac=1e-3, label_noise_frac=0.0,
             N_PER_CLASS=60, H=10, W=10, Cin=2, n_classes=3, Cout1=4, Cout2=4, k=3):
    print(f"=== MAIN TOY RUN: K={K} full-batch unScion(CNN) steps ===")
    X, y = make_data(N_PER_CLASS, H, W, Cin, n_classes, seed=seed, noise=noise,
                      label_noise_frac=label_noise_frac)
    N = X.shape[0]
    Ho2 = H - 2 * (k - 1)
    n_feat = Cout2 * Ho2 * Ho2
    n_p_head = n_feat  # head_W shape (n_classes, n_feat); n_p = input (column) dim

    rng = np.random.default_rng(seed)
    params = {
        "conv1_W": rng.normal(0, 0.3, (Cout1, Cin, k, k)),
        "conv1_b": np.zeros(Cout1),
        "conv2_W": rng.normal(0, 0.3, (Cout2, Cout1, k, k)),
        "conv2_b": np.zeros(Cout2),
        "head_W": rng.normal(0, 0.2, (n_classes, n_feat)),
    }

    layer_specs = {
        "bias1": ("conv1_b", "bias", dict(Cout=Cout1)),
        "conv1": ("conv1_W", "conv", dict(k=k, Cin=Cin, Cout=Cout1)),
        "bias2": ("conv2_b", "bias", dict(Cout=Cout2)),
        "conv2": ("conv2_W", "conv", dict(k=k, Cin=Cout1, Cout=Cout2)),
        "head": ("head_W", "head", dict(n_p=n_p_head)),
    }

    def dual_norm(kind, g, spec):
        if kind == "bias":
            return bias_dual(g, spec["Cout"])
        if kind == "conv":
            return conv_dual(g, spec["k"], spec["Cin"], spec["Cout"])
        return head_dual(g, spec["n_p"])

    def primal_norm(kind, d, spec):
        if kind == "bias":
            return bias_primal(d, spec["Cout"])
        if kind == "conv":
            return conv_primal(d, spec["k"], spec["Cin"], spec["Cout"])
        return head_primal(d, spec["n_p"])

    traj = {name: {"Lhat": [], "gnext": []} for name in layer_specs}
    losses = []

    loss, grad_cur = forward_backward(params, X, y)
    losses.append(loss)

    # Reference gradient scale per layer (from the first few steps, while still "learning")
    # used to detect saturation: once the model fits the toy data near-perfectly, gradients
    # collapse to float-noise level and the SVD/sign directions become meaningless (they
    # still return *a* direction, but it no longer reflects real curvature) -- those steps
    # must be excluded from the Eq. 10 trajectory used for the L0/L1 fit, or they pollute it
    # with numerically-driven noise.
    ref_scale = {name: dual_norm(kind, grad_cur[pname], spec)
                 for name, (pname, kind, spec) in layer_specs.items()}
    n_saturated_skipped = {name: 0 for name in layer_specs}

    for step in range(K):
        old_params = {name: params[pname].copy() for name, (pname, kind, spec) in layer_specs.items()}

        params["conv1_b"] = update_bias(params["conv1_b"], grad_cur["conv1_b"], t)
        params["conv1_W"] = update_conv(params["conv1_W"], grad_cur["conv1_W"], t)
        params["conv2_b"] = update_bias(params["conv2_b"], grad_cur["conv2_b"], t)
        params["conv2_W"] = update_conv(params["conv2_W"], grad_cur["conv2_W"], t)
        params["head_W"] = update_head(params["head_W"], grad_cur["head_W"], t)

        loss_next, grad_next = forward_backward(params, X, y)
        assert np.isfinite(loss_next), f"NaN loss at step {step}"
        losses.append(loss_next)

        for name, (pname, kind, spec) in layer_specs.items():
            gnext_norm = dual_norm(kind, grad_next[pname], spec)
            gcur_norm = dual_norm(kind, grad_cur[pname], spec)
            floor = grad_floor_frac * max(ref_scale[name], 1e-12)
            if gnext_norm < floor or gcur_norm < floor:
                n_saturated_skipped[name] += 1
                continue  # saturated regime: gradient ~ float noise, skip (both this and next update meaningless)
            delta = params[pname] - old_params[name]
            dgrad = grad_next[pname] - grad_cur[pname]
            num = dual_norm(kind, dgrad, spec)
            den = primal_norm(kind, delta, spec)
            if den < 1e-14:
                continue
            Lhat = num / den
            traj[name]["Lhat"].append(Lhat)
            traj[name]["gnext"].append(gnext_norm)

        grad_cur = grad_next

    acc = predict_acc(params, X, y)
    print(f"[main run] loss: {losses[0]:.4f} -> {losses[-1]:.4f}, final train acc: {acc:.3f}")
    print(f"[main run] trajectory points per layer: {[(n, len(v['Lhat'])) for n, v in traj.items()]}")
    print(f"[main run] steps skipped as saturated (grad below {grad_floor_frac:g}x initial) per layer: "
          f"{n_saturated_skipped}")
    return traj, losses, acc


# --------------------------------------------------------------------------------------
# Eq. 30 hinge-penalized least-squares fit of L0, L1 per layer.
# --------------------------------------------------------------------------------------
def fit_L0_L1(Lhat, gnext, lam=5.0):
    Lhat = np.asarray(Lhat)
    gnext = np.asarray(gnext)

    def obj(params):
        L0, L1 = params
        L0 = max(L0, 0.0)
        L1 = max(L1, 0.0)
        approx = L0 + L1 * gnext
        resid = Lhat - approx
        base = np.sum(resid ** 2)
        under = np.sum(np.maximum(0.0, resid) ** 2)  # Lhat > approx => underestimation
        return base + lam * under

    # simple nonneg init via least squares on [1, gnext]
    A = np.vstack([np.ones_like(gnext), gnext]).T
    coef, *_ = np.linalg.lstsq(A, Lhat, rcond=None)
    x0 = np.clip(coef, 0.0, None)
    res = minimize(obj, x0=x0, method="Nelder-Mead",
                    options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
    L0, L1 = max(res.x[0], 0.0), max(res.x[1], 0.0)
    return L0, L1


if __name__ == "__main__":
    gradcheck()
    smoketest()
    # Larger head input dim (n_p) matters: unScion's head norm divides the gradient dual norm
    # by n_p, so a head fed by many flattened conv features (as in a real CIFAR-10 CNN) gets a
    # much smaller effective L1 in normalized units than a toy head with few input features.
    # Also add label noise so full-batch training can't drive gradients to float-precision
    # zero within the step budget (keeps the whole trajectory informative).
    traj, losses, acc = main_run(
        K=3000, t=0.025, seed=42, noise=0.9, grad_floor_frac=1e-3, label_noise_frac=0.15,
        N_PER_CLASS=80, H=16, W=16, Cin=2, n_classes=3, Cout1=4, Cout2=8, k=3,
    )

    rows = []
    for name, data in traj.items():
        L0, L1 = fit_L0_L1(data["Lhat"], data["gnext"])
        rows.append({"layer_group": name, "L0_fit": L0, "L1_fit": L1, "n_points": len(data["Lhat"])})
        print(f"{name:10s}: L0={L0:.5f}  L1={L1:.5f}  (n={len(data['Lhat'])})")

    with open("claim5_cnn_smoothness.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["layer_group", "L0_fit", "L1_fit", "n_points"])
        w.writeheader()
        w.writerows(rows)

    # optional trajectory dump for inspection / plotting
    with open("claim5_cnn_smoothness_traj.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer_group", "step", "Lhat", "grad_dual_norm_next"])
        for name, data in traj.items():
            for i, (lh, gn) in enumerate(zip(data["Lhat"], data["gnext"])):
                w.writerow([name, i, lh, gn])

    head_L1 = next(r["L1_fit"] for r in rows if r["layer_group"] == "head")
    other_L1 = [r["L1_fit"] for r in rows if r["layer_group"] != "head" and r["L1_fit"] > 0]
    if other_L1:
        min_other = min(other_L1)
        ratio = min_other / max(head_L1, 1e-12)
        print(f"\nsmallest non-head L1 / head L1 ratio: {ratio:.2f} "
              f"(order of magnitude: {np.log10(max(ratio,1e-12)):.2f})")
    max_L0 = max(r["L0_fit"] for r in rows)
    max_L1_scale = max(r["L1_fit"] for r in rows)
    print(f"max L0 across layers: {max_L0:.5f} (vs max L1 scale {max_L1_scale:.5f})")
    print("\nDone. See claim5_cnn_smoothness.csv and claim5_cnn_smoothness_traj.csv")
