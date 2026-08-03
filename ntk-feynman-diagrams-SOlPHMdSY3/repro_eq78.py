# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "scipy"]
# ///
"""
Numeric test of Eq. 78 (Appendix G) -- the 5-diagram Feynman-diagram recursion for the
O(1/n) finite-width correction to the NTK mean, Claims 1 & 2 of
"Finite-Width Neural Tangent Kernels from Feynman Diagrams" (arXiv 2508.11522,
OpenReview SOlPHMdSY3).

Why this script exists: the original toy reproduction (repro_toy.py, Claims 3/4/5) marked
Claims 1/2 INCONCLUSIVE because the arXiv-only PDF extraction of the paper's `\Delta\Omega_d`
operator (one term in Eq. 78) showed a suspicious OHM-SIGN (U+2126) font substitution for
"Omega", raising a transcription-risk concern. A later OpenReview camera-ready PDF was
acquired and cross-checked byte-for-byte against the arXiv extraction: BOTH PDFs show the
*exact same* character-substitution pattern (identical counts of Delta-G/K/Theta/Omega/omega
occurrences), proving this is a consistent, deterministic font-encoding quirk (both PDFs are
native LaTeX text layers, not scans -- "OCR" was a misnomer), not corruption. The operator's
actual definition was recovered directly from the OpenReview text (Section 4, just before
Eq. 7/10):

    Omega_hat_{i,ab}^(l+1) = sigma_{i,a}^(l) sigma_{i,b}^(l)
                             + C_W^(l+1) Theta_{ab}^(l) sigma'_{i,a}^(l) sigma'_{i,b}^(l)
    Delta_Omega_d = Omega_hat - <Omega_hat>_{K^(l)}   (an ordinary mean-centered fluctuation)

This is simple enough to implement directly, so Claims 1/2 are no longer stuck.

Scope simplification (documented, not hidden): Eq. 78 in general involves two distinct
external inputs x1 != x2 and rank-4 tensors V/D/F with up to 16 index combinations each
(beta1..beta4 in {1,2}). To keep this tractable at toy/CPU scale, this script tests the
*single-input diagonal* case (external pair 1=2=a, one physical input x_a) instead. Every
beta-sum in Eq. 78 collapses to a single term (beta1=...=a), so each tensor reduces to one
number (Theta^{1}_aa, K^{1}_aa, V_aaaa, D_aaaa, F_aaaa). Activation is tanh (NOT scale-invariant,
so unlike Claims 3/4's ReLU/LeakyReLU case the predicted correction is genuinely nonzero --
a meaningful test, not the degenerate zero-case Theorem 5.2 already covers). The off-diagonal,
two-input case is a natural follow-up not attempted here.

Method:
  1. Analytic (quadrature-exact) infinite-width K^(l), Theta^(l) recursion for tanh, single
     diagonal input -- cross-checked against a wide-network Monte Carlo estimate first.
  2. Monte Carlo estimate, at layer l=2, of the five ingredient numbers Eq. 78 needs
     (Theta^{1(2)}, K^{1(2)}, V_aaaa, D_aaaa, F_aaaa) via a width sweep (n1=n2=n, jointly
     scaled per the paper's single-1/n-expansion convention) and linear extrapolation in 1/n.
     Neuron-index tricks (picking which layer-2 neurons coincide) isolate each tensor from
     the others in the underlying joint cumulants -- see BUGFIX_LOG.md for the derivation.
  3. Gaussian-quadrature evaluation of Eq. 78's bracket expectations (<sigma'^2>_K,
     <d^2 Omega_hat/dz^2>_K, <d^4 Omega_hat/dz^4>_K, <d^2(sigma'^2)/dz^2>_K) via exact
     autodiff derivatives of Omega_hat(z) integrated against N(0, K^(2)).
  4. Assemble Eq. 78's RHS -> predicted Theta^{1(3)}; compare to an independently-measured
     Theta^{1(3)} (separate width sweep + fit on the 3-layer network's own NTK).

Results written to results/eq78_<tag>.json.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Gauss-Hermite quadrature machinery (physicist convention, weight e^{-t^2})
GH_N = 80
_GH_T, _GH_W = np.polynomial.hermite.hermgauss(GH_N)


def gauss_expect_vals(vals):
    """E[f(z)] for z ~ N(0, sigma^2), given f already evaluated at the quadrature nodes
    z_i = sigma*sqrt(2)*t_i (see `nodes_for_variance`)."""
    return float(np.sum(_GH_W * vals) / math.sqrt(math.pi))


def nodes_for_variance(var):
    sigma = math.sqrt(max(var, 0.0))
    return sigma * math.sqrt(2.0) * _GH_T


# ---------------------------------------------------------------------------
# Analytic infinite-width recursion (single diagonal input), tanh activation
def analytic_K_Theta(x_a, C_W, depth):
    """Returns (Ks, Thetas), lists indexed 0..depth-1 for layers 1..depth.
    Base case: K^(1) = C_W*(x.x)/n0, Theta^(1) = (x.x)/n0 (no C_W -- only layer-1 weights
    contribute to the NTK gradient at layer 1, so no weight-variance factor; matches the
    convention already cross-validated for ReLU in repro_toy.py's BUGFIX_LOG)."""
    n0 = len(x_a)
    xx = float(np.dot(x_a, x_a))
    K = C_W * xx / n0
    Theta = xx / n0
    Ks, Thetas = [K], [Theta]
    for _ in range(1, depth):
        z = nodes_for_variance(K)
        s = np.tanh(z)
        sp2 = (1.0 - s ** 2) ** 2
        Enew_s2 = gauss_expect_vals(s ** 2)
        Enew_sp2 = gauss_expect_vals(sp2)
        Knew = C_W * Enew_s2
        Theta = Theta * C_W * Enew_sp2 + Knew
        K = Knew
        Ks.append(K)
        Thetas.append(Theta)
    return Ks, Thetas


def analytic_layer2_ingredients(x_a, C_W):
    """Fully analytic (zero MC noise) computation of the 5 layer-2 ingredient tensors Eq. 78
    needs, using the paper's OWN layer-to-layer recursions for V (Eq. 45), K^{1} (Eq. 47),
    D (Eq. 49), F (Eq. 5, main text) and Theta^{1} (Eq. 78 itself), each applied at the base
    transition l=1 -> l+1=2 (verified against the OpenReview PDF page images directly, not the
    text extraction, for the exact coefficients -- see BUGFIX_LOG.md).

    Because z^(1) is EXACTLY Gaussian for any finite n0 (no CLT approximation at layer 1 at
    all), every layer-1 fluctuation tensor entering these recursions (Theta^{1(1)}, K^{1(1)},
    V^(1), D^(1), F^(1)) is EXACTLY ZERO. Every recursion term proportional to one of those
    layer-1 tensors therefore vanishes identically:
      Theta^{1(2)} = 0 exactly -- every term of Eq. 78 applied at l=1 is proportional to a
        layer-1 tensor (Theta^{1(1)}, K^{1(1)}, V^(1), D^(1), or F^(1)), all zero.
      K^{1(2)} = 0 exactly -- same argument applied to Eq. 47.
    This matches (and explains) every MC width-sweep run this session: Theta1_2/K1_2 estimates
    were wildly inconsistent in sign/magnitude across debug1/debug2/scale1/crn_smoke/crn1 --
    because the true value is exactly zero and every one of those was pure sampling noise
    around it, not a real signal any amount of extra samples or CRN could have resolved.

    V^(2), D^(2), F^(2) each keep one "new" term generated fresh by the layer 1->2 finite-width
    sum (independent of any prior non-Gaussian structure, so nonzero even from an exact-Gaussian
    base layer) -- each is a pure Gauss-Hermite quadrature integral over z ~ N(0, K^(1)):
      V^(2)_aaaa = C_W^2 [ E[s^4] - E[s^2]^2 ]                        (Eq. 45, single input)
      D^(2)_aaaa = C_W [ E[s^2 * Omega_hat] - E[s^2] E[Omega_hat] ]   (Eq. 49, single input)
      F^(2)_aaaa = C_W^2 * E[s^2 * sp^2] * Theta^(1)                  (Eq. 5, single input)
    where s = tanh(z), sp = 1 - s^2, Omega_hat = s^2 + C_W * Theta^(1) * sp^2, all expectations
    under z ~ N(0, K^(1)).
    """
    K1, Th1 = analytic_K_Theta(x_a, C_W, depth=1)
    K1, Th1 = K1[0], Th1[0]

    zs = nodes_for_variance(K1)
    s = np.tanh(zs)
    sp2 = (1.0 - s ** 2) ** 2
    omega_hat = s ** 2 + C_W * Th1 * sp2

    E_s2 = gauss_expect_vals(s ** 2)
    E_s4 = gauss_expect_vals(s ** 4)
    E_s2sp2 = gauss_expect_vals(s ** 2 * sp2)
    E_omega = gauss_expect_vals(omega_hat)
    E_s2_omega = gauss_expect_vals(s ** 2 * omega_hat)

    return dict(
        Theta1_2=0.0,
        K1_2=0.0,
        V_2=(C_W ** 2) * (E_s4 - E_s2 ** 2),
        D_2=C_W * (E_s2_omega - E_s2 * E_omega),
        F_2=(C_W ** 2) * E_s2sp2 * Th1,
        K1=K1, Th1=Th1,
    )


def wide_network_KTheta_check(x_a, C_W, depth, n_wide, n_inits, seed):
    """Direct large-width MC cross-check of analytic_K_Theta (smoketest)."""
    import repro_toy as rt

    n0 = len(x_a)
    widths = [n0] + [n_wide] * depth
    xt = torch.tensor(x_a)
    gen = torch.Generator().manual_seed(seed)
    diag_by_layer = [[] for _ in range(depth)]
    Kdiag_by_layer = [[] for _ in range(depth)]
    for _ in range(n_inits):
        params = rt.sample_network(widths, C_W, bias=False, gen=gen)
        zs = rt.preacts(params, xt, torch.tanh)
        for l in range(depth):
            th, z_x, _ = rt.ntk_unit0(params, xt, xt, l, torch.tanh)
            diag_by_layer[l].append(th)
            Kdiag_by_layer[l].append(z_x ** 2)
    theta_mc = [float(np.mean(v)) for v in diag_by_layer]
    theta_sem = [float(np.std(v) / math.sqrt(n_inits)) for v in diag_by_layer]
    K_mc = [float(np.mean(v)) for v in Kdiag_by_layer]
    K_sem = [float(np.std(v) / math.sqrt(n_inits)) for v in Kdiag_by_layer]
    return K_mc, K_sem, theta_mc, theta_sem


# ---------------------------------------------------------------------------
# Autodiff derivative + Gaussian-bracket evaluator for Eq. 78's Omega_hat operator
def omega_hat_fn(z, C_W, Theta_l):
    s = torch.tanh(z)
    sp = 1.0 - s ** 2
    return s ** 2 + C_W * Theta_l * sp ** 2


def sigma_prime_sq_fn(z):
    s = torch.tanh(z)
    return (1.0 - s ** 2) ** 2


def nth_derivative_batch(f, z_vals, n):
    """n-th derivative of f, evaluated pointwise at each entry of z_vals (independent
    leaves in a 1D tensor -- exact via repeated autograd, no MC noise)."""
    z = torch.tensor(z_vals, dtype=torch.float64, requires_grad=True)
    y = f(z)
    for _ in range(n):
        y = torch.autograd.grad(y.sum(), z, create_graph=True)[0]
    return y.detach().numpy()


def bracket_deriv(f, K_variance, n):
    zs = nodes_for_variance(K_variance)
    vals = nth_derivative_batch(f, zs, n)
    return gauss_expect_vals(vals)


# ---------------------------------------------------------------------------
# Monte Carlo estimators of layer-2 ingredient tensors + independent layer-3 Theta^{1}
def preacts_full(params, x, act_fn):
    h = x
    zs = []
    for W, b in params:
        n_in = W.shape[1]
        z = (W @ h) / math.sqrt(n_in)
        if b is not None:
            z = z + b
        zs.append(z)
        h = act_fn(z)
    return zs


def one_init_measurements(params, x_a, act_fn, layer2_idx, layer3_idx):
    """Returns z0_2, z1_2 (layer-2 preacts, neurons 0,1) and Theta_00_2, Theta_11_2,
    Theta_01_2 (layer-2 NTK entries), Theta_00_3 (layer-3 NTK), all at input x_a."""
    used2 = [W for W, b in params[: layer2_idx + 1]]
    used3 = [W for W, b in params[: layer3_idx + 1]]

    zs2 = preacts_full(params, x_a, act_fn)[layer2_idx]
    z0_2, z1_2 = zs2[0], zs2[1]
    g0 = torch.autograd.grad(z0_2, used2, retain_graph=True, create_graph=False)
    g1 = torch.autograd.grad(z1_2, used2, retain_graph=True, create_graph=False)
    Theta_00_2 = sum((a * a).sum() for a in g0).item()
    Theta_11_2 = sum((a * a).sum() for a in g1).item()
    Theta_01_2 = sum((a * b).sum() for a, b in zip(g0, g1)).item()

    zs3 = preacts_full(params, x_a, act_fn)[layer3_idx]
    z0_3 = zs3[0]
    g0_3 = torch.autograd.grad(z0_3, used3, retain_graph=False, create_graph=False)
    Theta_00_3 = sum((a * a).sum() for a in g0_3).item()

    return z0_2.item(), z1_2.item(), Theta_00_2, Theta_11_2, Theta_01_2, Theta_00_3


def run_width_sweep(x_a, C_W, widths_n, n_inits, seed, tag):
    """CRN (common random numbers) width sweep. Instead of drawing an independent fresh
    random network per width (previous version -- debug1/debug2), draw ONE maximal-width
    network (n0 -> n_max -> n_max -> 1) per init and, for every width n in widths_n, use the
    top-left n-slice of those SAME weight matrices (a nested/coupled sub-network) rather than
    a fresh draw. A sub-block of an i.i.d. Gaussian matrix has exactly the same marginal
    distribution as an independent draw of that size, so this is unbiased -- it only
    *correlates* the sampling noise across widths. That's exactly what the 1/n linear fit
    needs: noise common to all widths (from shared weight entries) cancels out of the
    across-width differences the fit relies on, instead of just averaging down independent
    per-width noise. See BUGFIX_LOG.md for the debug1/debug2 runs this was needed after (R^2 on
    the key fits stayed weak even at n_inits=100000 with independent per-width sampling).
    widths = [n0, n, n, 1] (layer1: n0->n, layer2: n->n, layer3: n->1).
    layer2_idx=1 (params up to and incl. layer 2), layer3_idx=2 (params up to layer 3)."""
    n0 = len(x_a)
    xt = torch.tensor(x_a)
    act_fn = torch.tanh
    n_max = max(widths_n)
    gen = torch.Generator().manual_seed(seed * 7919)

    acc = {n: dict(z0=[], z1=[], Th00_2=[], Th11_2=[], Th01_2=[], Th00_3=[]) for n in widths_n}

    t0 = time.time()
    report_every = max(1, n_inits // 10)
    for k in range(n_inits):
        W1_full = torch.randn(n_max, n0, generator=gen) * math.sqrt(C_W)
        W2_full = torch.randn(n_max, n_max, generator=gen) * math.sqrt(C_W)
        W3_full = torch.randn(1, n_max, generator=gen) * math.sqrt(C_W)
        for n in widths_n:
            W1 = W1_full[:n, :].clone().requires_grad_(True)
            W2 = W2_full[:n, :n].clone().requires_grad_(True)
            W3 = W3_full[:, :n].clone().requires_grad_(True)
            params = [(W1, None), (W2, None), (W3, None)]
            z0, z1, th00_2, th11_2, th01_2, th00_3 = one_init_measurements(
                params, xt, act_fn, layer2_idx=1, layer3_idx=2
            )
            a = acc[n]
            a["z0"].append(z0); a["z1"].append(z1)
            a["Th00_2"].append(th00_2); a["Th11_2"].append(th11_2); a["Th01_2"].append(th01_2)
            a["Th00_3"].append(th00_3)
        if (k + 1) % report_every == 0:
            print(f"[{tag}] {k + 1}/{n_inits} CRN inits  ({time.time() - t0:.1f}s elapsed)", flush=True)

    per_width = {}
    for n in widths_n:
        a = acc[n]
        z0s = np.array(a["z0"]); z1s = np.array(a["z1"])
        Th00_2s = np.array(a["Th00_2"]); Th11_2s = np.array(a["Th11_2"]); Th01_2s = np.array(a["Th01_2"])
        Th00_3s = np.array(a["Th00_3"])

        per_width[n] = dict(
            E_z0sq=float(np.mean(z0s ** 2)),
            E_z0sq_sem=float(np.std(z0s ** 2) / math.sqrt(n_inits)),
            E_Th00_2=float(np.mean(Th00_2s)),
            E_Th00_2_sem=float(np.std(Th00_2s) / math.sqrt(n_inits)),
            E_Th00_3=float(np.mean(Th00_3s)),
            E_Th00_3_sem=float(np.std(Th00_3s) / math.sqrt(n_inits)),
            # V_aaaa ingredient: kappa4(z0,z0,z1,z1) = E[z0^2 z1^2] - E[z0^2]E[z1^2] - 2 E[z0 z1]^2
            kappa4_z0z0z1z1=float(
                np.mean(z0s ** 2 * z1s ** 2) - np.mean(z0s ** 2) * np.mean(z1s ** 2)
                - 2.0 * np.mean(z0s * z1s) ** 2
            ),
            # D_aaaa ingredient: E[z0^2 * (Th11 - E[Th11])]  (both z0, dTheta11 zero-mean)
            D_raw=float(np.mean(z0s ** 2 * (Th11_2s - Th11_2s.mean()))),
            # F_aaaa ingredient: E[z0*z1 * (Th01 - E[Th01])]
            F_raw=float(np.mean(z0s * z1s * (Th01_2s - Th01_2s.mean()))),
            n_inits=n_inits,
        )
        print(f"[{tag}] n={n:5d}  E[Th00_2]={per_width[n]['E_Th00_2']:.6f}  "
              f"E[Th00_3]={per_width[n]['E_Th00_3']:.6f}")

    print(f"[{tag}] total wall time {time.time() - t0:.1f}s for {n_inits} CRN inits x {len(widths_n)} widths")
    return per_width


def theta13_measurements(params, x_a, act_fn, layer2_idx, layer3_idx):
    """Leaner version of one_init_measurements: only Theta_00_2 (sanity check that MC recovers
    Theta1_2=0, per analytic_layer2_ingredients) and Theta_00_3 (the one quantity that still
    needs an independent MC measurement now that V/D/F/Theta1_2/K1_2 are all analytic -- see
    analytic_layer2_ingredients). Drops z1/Theta_11_2/Theta_01_2 entirely (only needed for the
    old MC-based V/D/F estimators, now unused), cutting the autograd cost per init."""
    used2 = [W for W, b in params[: layer2_idx + 1]]
    used3 = [W for W, b in params[: layer3_idx + 1]]

    z0_2 = preacts_full(params, x_a, act_fn)[layer2_idx][0]
    g0 = torch.autograd.grad(z0_2, used2, retain_graph=False, create_graph=False)
    Theta_00_2 = sum((a * a).sum() for a in g0).item()

    z0_3 = preacts_full(params, x_a, act_fn)[layer3_idx][0]
    g0_3 = torch.autograd.grad(z0_3, used3, retain_graph=False, create_graph=False)
    Theta_00_3 = sum((a * a).sum() for a in g0_3).item()

    return Theta_00_2, Theta_00_3


def run_theta13_sweep(x_a, C_W, widths_n, n_inits, seed, tag):
    """CRN width sweep measuring ONLY Theta_00_2 and Theta_00_3 (see theta13_measurements).
    Same nested/coupled-network CRN construction as run_width_sweep, just without the unused
    V/D/F ingredient bookkeeping -- cheaper per init, so more samples fit in the same budget."""
    n0 = len(x_a)
    xt = torch.tensor(x_a)
    act_fn = torch.tanh
    n_max = max(widths_n)
    gen = torch.Generator().manual_seed(seed * 7919)

    acc = {n: dict(Th00_2=[], Th00_3=[]) for n in widths_n}

    t0 = time.time()
    report_every = max(1, n_inits // 10)
    for k in range(n_inits):
        W1_full = torch.randn(n_max, n0, generator=gen) * math.sqrt(C_W)
        W2_full = torch.randn(n_max, n_max, generator=gen) * math.sqrt(C_W)
        W3_full = torch.randn(1, n_max, generator=gen) * math.sqrt(C_W)
        for n in widths_n:
            W1 = W1_full[:n, :].clone().requires_grad_(True)
            W2 = W2_full[:n, :n].clone().requires_grad_(True)
            W3 = W3_full[:, :n].clone().requires_grad_(True)
            params = [(W1, None), (W2, None), (W3, None)]
            th00_2, th00_3 = theta13_measurements(params, xt, act_fn, layer2_idx=1, layer3_idx=2)
            acc[n]["Th00_2"].append(th00_2)
            acc[n]["Th00_3"].append(th00_3)
        if (k + 1) % report_every == 0:
            print(f"[{tag}] {k + 1}/{n_inits} CRN inits  ({time.time() - t0:.1f}s elapsed)", flush=True)

    per_width = {}
    for n in widths_n:
        Th00_2s = np.array(acc[n]["Th00_2"]); Th00_3s = np.array(acc[n]["Th00_3"])
        per_width[n] = dict(
            E_Th00_2=float(np.mean(Th00_2s)), E_Th00_2_sem=float(np.std(Th00_2s) / math.sqrt(n_inits)),
            E_Th00_3=float(np.mean(Th00_3s)), E_Th00_3_sem=float(np.std(Th00_3s) / math.sqrt(n_inits)),
            n_inits=n_inits,
        )
        print(f"[{tag}] n={n:5d}  E[Th00_2]={per_width[n]['E_Th00_2']:.6f}  "
              f"E[Th00_3]={per_width[n]['E_Th00_3']:.6f}")

    print(f"[{tag}] total wall time {time.time() - t0:.1f}s for {n_inits} CRN inits x {len(widths_n)} widths")
    return per_width


def run_eq78_analytic(x_a, C_W, widths_n, n_inits, seed, tag):
    """Eq. 78 test using the ANALYTIC layer-2 ingredients (analytic_layer2_ingredients -- zero
    MC noise, see BUGFIX_LOG.md) for the predicted side, and an independent MC width sweep
    (run_theta13_sweep) ONLY for the measured Theta^{1}(3) side. Replaces run_eq78's fully-MC
    approach, which struggled to resolve the 4th-cumulant ingredients (V/D/F) and wrongly-nonzero
    Theta1_2/K1_2 estimates (both are EXACTLY zero, see analytic_layer2_ingredients) even after
    implementing CRN variance reduction."""
    ing = analytic_layer2_ingredients(x_a, C_W)
    Ks, Thetas = analytic_K_Theta(x_a, C_W, depth=3)
    K1, K2, K3 = Ks
    Th1, Th2, Th3 = Thetas

    per_width = run_theta13_sweep(x_a, C_W, widths_n, n_inits, seed, tag)
    ns = sorted(per_width.keys())
    Th1_2_i, Th1_2_s, Th1_2_r2 = extrapolate_coefficient(ns, [per_width[n]["E_Th00_2"] for n in ns], Th2)
    Th1_3_i, Th1_3_s, Th1_3_r2 = extrapolate_coefficient(ns, [per_width[n]["E_Th00_3"] for n in ns], Th3)

    E_sp2 = bracket_deriv(sigma_prime_sq_fn, K2, 0)
    bracket_d2Omega = bracket_deriv(lambda z: omega_hat_fn(z, C_W, Th2), K2, 2)
    bracket_d4Omega = bracket_deriv(lambda z: omega_hat_fn(z, C_W, Th2), K2, 4)
    bracket_d2sp2 = bracket_deriv(sigma_prime_sq_fn, K2, 2)

    term1 = C_W * ing["Theta1_2"] * E_sp2
    term2 = 0.5 * ing["K1_2"] * bracket_d2Omega
    term3 = (1.0 / 8.0) * ing["V_2"] * bracket_d4Omega
    term4 = (C_W / 2.0) * ing["D_2"] * bracket_d2sp2
    term5 = C_W * ing["F_2"] * bracket_d2sp2
    predicted = term1 + term2 + term3 + term4 + term5
    measured = Th1_3_i

    print("\n=== Eq. 78, ANALYTIC ingredients (zero MC noise) vs MC-measured Theta^{1}(3) ===")
    print(f"K^(2)={K2:.6f}  Theta^(2)={Th2:.6f}  K^(3)={K3:.6f}  Theta^(3)={Th3:.6f}")
    print(f"Theta^{{1}}(2) = {ing['Theta1_2']:.6f}  (exact)")
    print(f"K^{{1}}(2)     = {ing['K1_2']:.6f}  (exact)")
    print(f"V_aaaa        = {ing['V_2']:.6f}  (exact, quadrature)")
    print(f"D_aaaa        = {ing['D_2']:.6f}  (exact, quadrature)")
    print(f"F_aaaa        = {ing['F_2']:.6f}  (exact, quadrature)")
    print(f"\nMC sanity check -- fitted Theta1_2 from the width sweep (should be ~0): "
          f"{Th1_2_i:.6f}  (R^2={Th1_2_r2:.4f})")
    print(f"\nGaussian brackets at K^(2): <sp^2>={E_sp2:.6f}  <d2Omega>={bracket_d2Omega:.6f}  "
          f"<d4Omega>={bracket_d4Omega:.6f}  <d2(sp^2)>={bracket_d2sp2:.6f}")
    print(f"\nEq.78 terms: 1={term1:.6f} 2={term2:.6f} 3={term3:.6f} 4={term4:.6f} 5={term5:.6f}")
    print(f"\nANALYTIC PREDICTED Theta^{{1}}(3) = {predicted:.6f}  (zero MC noise)")
    print(f"MC MEASURED       Theta^{{1}}(3) = {measured:.6f}  (R^2={Th1_3_r2:.4f}, "
          f"fit slope+intercept from {len(ns)} widths x {n_inits} CRN inits)")
    rel_err = abs(predicted - measured) / (abs(measured) + 1e-12)
    print(f"relative error = {rel_err:.4f}")

    result = dict(
        x_a=x_a.tolist(), C_W=C_W, widths=widths_n, n_inits=n_inits, seed=seed,
        analytic=dict(K=Ks, Theta=Thetas),
        analytic_ingredients=ing,
        mc_sanity_Theta1_2=[Th1_2_i, Th1_2_r2],
        brackets=dict(E_sp2=E_sp2, d2Omega=bracket_d2Omega, d4Omega=bracket_d4Omega, d2sp2=bracket_d2sp2),
        terms=dict(term1=term1, term2=term2, term3=term3, term4=term4, term5=term5),
        predicted_Theta1_3=predicted,
        measured_Theta1_3=[measured, Th1_3_r2],
        relative_error=rel_err,
        per_width=per_width,
    )
    out_path = RESULTS_DIR / f"eq78_{tag}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")
    return result


def linfit_intercept(ns, ys):
    """Fit y_n * n (an O(1) quantity if y_n ~ c/n + O(1/n^2)) against 1/n; the intercept
    of that line is the leading 1/n coefficient c, robust to O(1/n^2) contamination at
    finite n. Returns (intercept, slope, r_squared)."""
    ns = np.array(ns, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    x = 1.0 / ns
    yfit = ys * ns
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, rank, sv = np.linalg.lstsq(A, yfit, rcond=None)
    slope, intercept = coef
    pred = A @ coef
    ss_res = float(np.sum((yfit - pred) ** 2))
    ss_tot = float(np.sum((yfit - yfit.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(intercept), float(slope), r2


def extrapolate_coefficient(ns, raw_vals, zero_order=0.0):
    ys = [r - zero_order for r in raw_vals]
    return linfit_intercept(ns, ys)


def run_eq78(x_a, C_W, widths_n, n_inits, seed, tag):
    per_width = run_width_sweep(x_a, C_W, widths_n, n_inits, seed, tag)
    Ks, Thetas = analytic_K_Theta(x_a, C_W, depth=3)
    K1, K2, K3 = Ks
    Th1, Th2, Th3 = Thetas

    ns = sorted(per_width.keys())
    K1_2_i, K1_2_s, K1_2_r2 = extrapolate_coefficient(ns, [per_width[n]["E_z0sq"] for n in ns], K2)
    Th1_2_i, Th1_2_s, Th1_2_r2 = extrapolate_coefficient(ns, [per_width[n]["E_Th00_2"] for n in ns], Th2)
    V_i, V_s, V_r2 = extrapolate_coefficient(ns, [per_width[n]["kappa4_z0z0z1z1"] for n in ns])
    D_i, D_s, D_r2 = extrapolate_coefficient(ns, [per_width[n]["D_raw"] for n in ns])
    F_i, F_s, F_r2 = extrapolate_coefficient(ns, [per_width[n]["F_raw"] for n in ns])
    Th1_3_i, Th1_3_s, Th1_3_r2 = extrapolate_coefficient(ns, [per_width[n]["E_Th00_3"] for n in ns], Th3)

    E_sp2 = bracket_deriv(sigma_prime_sq_fn, K2, 0)
    bracket_d2Omega = bracket_deriv(lambda z: omega_hat_fn(z, C_W, Th2), K2, 2)
    bracket_d4Omega = bracket_deriv(lambda z: omega_hat_fn(z, C_W, Th2), K2, 4)
    bracket_d2sp2 = bracket_deriv(sigma_prime_sq_fn, K2, 2)

    term1 = C_W * Th1_2_i * E_sp2
    term2 = 0.5 * K1_2_i * bracket_d2Omega
    term3 = (1.0 / 8.0) * V_i * bracket_d4Omega
    term4 = (C_W / 2.0) * D_i * bracket_d2sp2
    term5 = C_W * F_i * bracket_d2sp2
    predicted = term1 + term2 + term3 + term4 + term5
    measured = Th1_3_i

    print("\n=== Eq. 78 layer-2 ingredients (intercepts, R^2 of 1/n fit) ===")
    print(f"K^(2)={K2:.6f}  Theta^(2)={Th2:.6f}  K^(3)={K3:.6f}  Theta^(3)={Th3:.6f}")
    print(f"Theta^{{1}}(2) = {Th1_2_i:.6f}  (R^2={Th1_2_r2:.4f})")
    print(f"K^{{1}}(2)     = {K1_2_i:.6f}  (R^2={K1_2_r2:.4f})")
    print(f"V_aaaa        = {V_i:.6f}  (R^2={V_r2:.4f})")
    print(f"D_aaaa        = {D_i:.6f}  (R^2={D_r2:.4f})")
    print(f"F_aaaa        = {F_i:.6f}  (R^2={F_r2:.4f})")
    print(f"\nGaussian brackets at K^(2): <sp^2>={E_sp2:.6f}  <d2Omega>={bracket_d2Omega:.6f}  "
          f"<d4Omega>={bracket_d4Omega:.6f}  <d2(sp^2)>={bracket_d2sp2:.6f}")
    print(f"\nEq.78 terms: 1={term1:.6f} 2={term2:.6f} 3={term3:.6f} 4={term4:.6f} 5={term5:.6f}")
    print(f"\nPREDICTED Theta^{{1}}(3) = {predicted:.6f}")
    print(f"MEASURED  Theta^{{1}}(3) = {measured:.6f}  (R^2={Th1_3_r2:.4f})")
    rel_err = abs(predicted - measured) / (abs(measured) + 1e-12)
    print(f"relative error = {rel_err:.4f}")

    result = dict(
        x_a=x_a.tolist(), C_W=C_W, widths=widths_n, n_inits=n_inits, seed=seed,
        analytic=dict(K=Ks, Theta=Thetas),
        ingredients=dict(
            Theta1_2=[Th1_2_i, Th1_2_r2], K1_2=[K1_2_i, K1_2_r2], V_aaaa=[V_i, V_r2],
            D_aaaa=[D_i, D_r2], F_aaaa=[F_i, F_r2],
        ),
        brackets=dict(E_sp2=E_sp2, d2Omega=bracket_d2Omega, d4Omega=bracket_d4Omega, d2sp2=bracket_d2sp2),
        terms=dict(term1=term1, term2=term2, term3=term3, term4=term4, term5=term5),
        predicted_Theta1_3=predicted,
        measured_Theta1_3=[measured, Th1_3_r2],
        relative_error=rel_err,
        per_width=per_width,
    )
    out_path = RESULTS_DIR / f"eq78_{tag}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["smoketest-recursion", "smoketest-eq78", "full-eq78", "analytic-eq78"])
    p.add_argument("--tag", default="toy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--C-W", type=float, default=1.0)
    p.add_argument("--widths", type=int, nargs="+", default=[20, 40, 80, 160])
    p.add_argument("--n-inits", type=int, default=2000)
    args = p.parse_args()

    x0 = np.array([1.0, 0.5, -0.3, 0.2])

    if args.mode == "smoketest-recursion":
        Ks, Thetas = analytic_K_Theta(x0, args.C_W, depth=3)
        print("analytic K^(l):", Ks)
        print("analytic Theta^(l):", Thetas)
        K_mc, K_sem, th_mc, th_sem = wide_network_KTheta_check(
            x0, args.C_W, depth=3, n_wide=300, n_inits=2000, seed=args.seed
        )
        for l in range(3):
            print(f"layer {l+1}: K analytic={Ks[l]:.6f}  K_mc={K_mc[l]:.6f}+-{K_sem[l]:.6f}   "
                  f"Theta analytic={Thetas[l]:.6f}  Theta_mc={th_mc[l]:.6f}+-{th_sem[l]:.6f}")
    elif args.mode in ("smoketest-eq78", "full-eq78"):
        run_eq78(x0, args.C_W, args.widths, args.n_inits, args.seed, args.tag)
    elif args.mode == "analytic-eq78":
        run_eq78_analytic(x0, args.C_W, args.widths, args.n_inits, args.seed, args.tag)
