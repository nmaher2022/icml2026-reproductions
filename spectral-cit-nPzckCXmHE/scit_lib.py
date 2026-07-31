# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scipy"]
# ///
"""SpectralCIT: bi-level spectral representation learning + CI test statistic.

Implements Algorithm 1 and Eq. (10) from "Toward Scalable and Valid Conditional
Independence Testing with Spectral Representations" (arXiv:2512.19510v2), transcribed
from Section 3 / Section 4 / Appendix C of the local PDF (see PAPER_BRIEFING.md).

Losses (p.4, Eqs after (8)):
    L_out(theta, B) = (1/(m(m-1))) sum_{i!=j} <u_i, M v_j>^2
                     - (2/m) sum_i <u_i, M v_i>
                     + (2/(m(m-1))) sum_{i!=j} <u_i, M v_j><w_i, w_j>
    L_in(theta, B)  = (1/(m(m-1))) sum_{i!=j} <w_i, N w_j>^2
                     - (2/(m(m-1))) sum_{i!=j} <u_i, M v_j><w_i, N w_j>
where u_i = u_theta(X_i) (centered), v_i = v_theta(Y_i, Z_i) (centered),
w_i = w_theta(Z_i) (centered), M = M_theta (learned diagonal, here fixed = I_d per the
"simple test statistic" framing -- see note in `SpectralModel`), N = (M + M^T)/2.

Orthonormality regularizers (p.4):
    Omega_out = ||C_UU - I_d||_F^2 + ||C_VV - I_d||_F^2
    Omega_in  = ||C_WW - I_2d||_F^2

Algorithm 1 (p.5): alternate n_steps_inner updates of w_theta (minimizing L_in) with one
update of u_theta, v_theta (minimizing L_out); afterwards whiten u, v, w using empirical
covariances computed over the full training set.

Test statistic Eq. (10), computed on a held-out test split:
    T_n = n * || C_{U,V} - C_{U,W} C_{W,V} ||_F^2
Reject H0 at level alpha when T_n >= chi2.ppf(1-alpha, d^2).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def make_mlp(in_dim: int, out_dim: int, n_hidden: int, layer_size: int,
             activation: type[nn.Module] = nn.Tanh) -> nn.Module:
    """MLP with Xavier init, per Appendix C. Activation defaults to Tanh (the paper's
    choice, which bounds the outputs and so satisfies Assumption 4.1's sub-Gaussianity
    requirement by construction). Claim 5's ablation passes an unbounded activation
    (e.g. nn.Identity) to test whether validity degrades without that bound."""
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_hidden):
        lin = nn.Linear(d, layer_size)
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
        layers += [lin, activation()]
        d = layer_size
    lin = nn.Linear(d, out_dim)
    nn.init.xavier_uniform_(lin.weight)
    nn.init.zeros_(lin.bias)
    layers.append(lin)
    return nn.Sequential(*layers)


def center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)


def empirical_cov(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C_ab = (1/n) A^T B for already-centered A, B (n x d_a), (n x d_b)."""
    return (a.T @ b) / a.shape[0]


class SpectralModel(nn.Module):
    """u_theta: X -> R^d, v_theta: (Y,Z) -> R^d, w_theta: Z -> R^{2d}, plus the
    diagonal/matrix nuisance parameters M_theta, N_theta that appear in L_out/L_in
    (p.4). Ambiguity note (logged here, not silently guessed): Algorithm 1's pseudocode
    only lists gradient updates for w_theta (inner step) and u_theta,v_theta (outer
    step) -- M_theta, N_theta aren't given their own update rule in the box, even
    though the loss definitions on p.4 write M = M_theta (a learned diagonal matrix)
    and N = (N_theta + N_theta^T)/2. We resolve this the most literal way consistent
    with the box: M_theta, N_theta are extra learnable parameters bundled into the
    outer/inner optimizer steps respectively (M_theta trained alongside u,v since it
    only appears in L_out; N_theta trained alongside w since it only appears in
    L_in). This is a modeling decision, not a verified paper detail -- see
    BUGFIX_LOG.md.
    """

    def __init__(self, x_dim: int, y_dim: int, z_dim: int, d: int,
                 n_hidden: int = 2, layer_size: int = 128,
                 activation: type[nn.Module] = nn.Tanh):
        super().__init__()
        self.d = d
        self.u = make_mlp(x_dim, d, n_hidden, layer_size, activation)
        self.v = make_mlp(y_dim + z_dim, d, n_hidden, layer_size, activation)
        self.w = make_mlp(z_dim, 2 * d, n_hidden, layer_size, activation)
        self.M = nn.Parameter(torch.ones(d))
        self.N = nn.Parameter(torch.eye(2 * d))

    def forward(self, X, Y, Z):
        u = self.u(X)
        v = self.v(torch.cat([Y, Z], dim=1))
        w = self.w(Z)
        return u, v, w


def loss_out(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    m = u.shape[0]
    ub, vb, wb = center(u), center(v), center(w)
    uMv = (ub * M) @ vb.T  # m x m, entry i,j = <u_i, M v_j>
    ww = wb @ wb.T
    off_mask = ~torch.eye(m, dtype=torch.bool, device=u.device)
    term1 = (uMv[off_mask] ** 2).sum() / (m * (m - 1))
    term2 = 2.0 * torch.diagonal(uMv).sum() / m
    term3 = 2.0 * (uMv[off_mask] * ww[off_mask]).sum() / (m * (m - 1))
    return term1 - term2 + term3


def loss_in(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, M: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
    m = u.shape[0]
    ub, vb, wb = center(u), center(v), center(w)
    uMv = (ub * M) @ vb.T
    Nsym = (N + N.T) / 2.0
    wNw = wb @ Nsym @ wb.T
    off_mask = ~torch.eye(m, dtype=torch.bool, device=u.device)
    term1 = (wNw[off_mask] ** 2).sum() / (m * (m - 1))
    term2 = 2.0 * (uMv[off_mask] * wNw[off_mask]).sum() / (m * (m - 1))
    return term1 - term2


def omega_out(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    d = u.shape[1]
    ub, vb = center(u), center(v)
    Cuu = empirical_cov(ub, ub)
    Cvv = empirical_cov(vb, vb)
    I = torch.eye(d, device=u.device)
    return ((Cuu - I) ** 2).sum() + ((Cvv - I) ** 2).sum()


def omega_in(w: torch.Tensor) -> torch.Tensor:
    d2 = w.shape[1]
    wb = center(w)
    Cww = empirical_cov(wb, wb)
    I = torch.eye(d2, device=w.device)
    return ((Cww - I) ** 2).sum()


def train_spectral_model(
    X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor,
    d: int = 10, n_hidden: int = 2, layer_size: int = 128,
    lr_inner: float = 3e-5, lr_outer: float = 2.1e-3,
    reg_inner: float = 3.3, reg_outer: float = 1.9,
    batch_size: int = 128, n_epochs: int = 400, n_steps_inner: int = 1,
    warmup_steps: int = 100, seed: int | None = None,
    activation: type[nn.Module] = nn.Tanh,
) -> SpectralModel:
    """Algorithm 1: bi-level training of u_theta, v_theta, w_theta."""
    if seed is not None:
        torch.manual_seed(seed)
    m_train = X.shape[0]
    model = SpectralModel(X.shape[1], Y.shape[1], Z.shape[1], d, n_hidden, layer_size, activation)
    inner_params = list(model.w.parameters()) + [model.N]
    outer_params = list(model.u.parameters()) + list(model.v.parameters()) + [model.M]
    opt_in = torch.optim.Adam(inner_params, lr=lr_inner)
    opt_out = torch.optim.Adam(outer_params, lr=lr_outer)

    def sample_batch():
        idx = torch.randint(0, m_train, (min(batch_size, m_train),))
        return X[idx], Y[idx], Z[idx]

    # warm up inner model before alternation begins (Appendix C)
    for _ in range(warmup_steps):
        Xb, Yb, Zb = sample_batch()
        u, v, w = model(Xb, Yb, Zb)
        l = loss_in(u, v, w, model.M.detach(), model.N) + reg_inner * omega_in(w)
        opt_in.zero_grad()
        l.backward()
        opt_in.step()

    steps_per_epoch = max(1, m_train // batch_size)
    for _epoch in range(n_epochs):
        for _step in range(steps_per_epoch):
            for _s in range(n_steps_inner):
                Xb, Yb, Zb = sample_batch()
                u, v, w = model(Xb, Yb, Zb)
                l = loss_in(u, v, w, model.M.detach(), model.N) + reg_inner * omega_in(w)
                opt_in.zero_grad()
                l.backward()
                opt_in.step()
            Xb, Yb, Zb = sample_batch()
            u, v, w = model(Xb, Yb, Zb)
            l = loss_out(u, v, w, model.M) + reg_outer * omega_out(u, v)
            opt_out.zero_grad()
            l.backward()
            opt_out.step()
    return model


@torch.no_grad()
def whiten(model: SpectralModel, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor):
    """Post-hoc whitening (p.5): rescale u,v,w by inverse-sqrt of their own empirical
    covariance computed over the full training set, so they become empirically
    orthonormal."""
    u, v, w = model(X, Y, Z)
    ub, vb, wb = center(u), center(v), center(w)

    def inv_sqrt(cov, eps=1e-6):
        cov = cov.double()
        vals, vecs = torch.linalg.eigh(cov)
        vals = vals.clamp(min=eps)
        return (vecs @ torch.diag(vals.rsqrt()) @ vecs.T).float()

    Cuu = empirical_cov(ub, ub)
    Cvv = empirical_cov(vb, vb)
    Cww = empirical_cov(wb, wb)
    Wu, Wv, Ww = inv_sqrt(Cuu), inv_sqrt(Cvv), inv_sqrt(Cww)
    return (ub @ Wu, vb @ Wv, wb @ Ww), (Wu, Wv, Ww)


@torch.no_grad()
def test_statistic(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> float:
    """Eq. (10): T_n = n * ||C_UV - C_UW C_WV||_F^2, on whitened test-set features."""
    n = u.shape[0]
    Cuv = empirical_cov(u, v)
    Cuw = empirical_cov(u, w)
    Cwv = empirical_cov(w, v)
    diff = Cuv - Cuw @ Cwv
    return float(n * (diff ** 2).sum())


@torch.no_grad()
def test_statistic_pruned(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
                           perc_dim_prune: float = 0.9) -> tuple[float, int]:
    """Appendix C "Dimension pruning": SVD the d x d test-statistic matrix
    Delta = C_UV - C_UW C_WV, keep only the leading floor(perc_dim_prune * d) singular
    triplets, and compute T_n from those alone. Returns (T_n, retained_dim) -- the
    caller compares T_n against chi2(retained_dim^2), not chi2(d^2), per Appendix C
    ("evaluated using a corrected chi^2 distribution with degrees of freedom equal to
    the retained (pruned) dimension")."""
    n = u.shape[0]
    d = u.shape[1]
    Cuv = empirical_cov(u, v)
    Cuw = empirical_cov(u, w)
    Cwv = empirical_cov(w, v)
    diff = Cuv - Cuw @ Cwv
    svals = torch.linalg.svdvals(diff)
    k = max(1, int(perc_dim_prune * d))
    top_svals = svals[:k]
    Tn = float(n * (top_svals ** 2).sum())
    return Tn, k


@torch.no_grad()
def validation_error(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> float:
    """E_m^val (p.5): max{||C_UU-I_d||, ||C_VV-I_d||, ||C_WW-I_2d||} -- self-covariance
    (orthonormality) discrepancies, NOT cross-covariances between u/v/w."""
    d = u.shape[1]
    d2 = w.shape[1]
    Cuu = empirical_cov(u, u)
    Cvv = empirical_cov(v, v)
    Cww = empirical_cov(w, w)
    e1 = torch.linalg.matrix_norm(Cuu - torch.eye(d), ord=2)
    e2 = torch.linalg.matrix_norm(Cvv - torch.eye(d), ord=2)
    e3 = torch.linalg.matrix_norm(Cww - torch.eye(d2), ord=2)
    return float(max(e1, e2, e3))


def run_spectral_cit_trial(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray, train_frac: float = 0.8,
    perc_dim_prune: float = 0.9, **train_kwargs
) -> dict:
    """Full pipeline for one (X,Y,Z) sample: split, train, whiten, compute T_n (raw and
    dimension-pruned per Appendix C) and E_m^val."""
    N = X.shape[0]
    n_train = int(train_frac * N)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    Yt = torch.as_tensor(Y, dtype=torch.float32)
    Zt = torch.as_tensor(Z, dtype=torch.float32)
    Xtr, Xte = Xt[:n_train], Xt[n_train:]
    Ytr, Yte = Yt[:n_train], Yt[n_train:]
    Ztr, Zte = Zt[:n_train], Zt[n_train:]

    model = train_spectral_model(Xtr, Ytr, Ztr, **train_kwargs)
    (u_tr, v_tr, w_tr), (Wu, Wv, Ww) = whiten(model, Xtr, Ytr, Ztr)
    val_err = validation_error(u_tr, v_tr, w_tr)

    with torch.no_grad():
        u_te_raw, v_te_raw, w_te_raw = model(Xte, Yte, Zte)
    # apply the *training-set* whitening transforms to the test set (avoids re-fitting
    # whitening on test data, which would leak test-set structure into the statistic)
    u_te = center(u_te_raw) @ Wu
    v_te = center(v_te_raw) @ Wv
    w_te = center(w_te_raw) @ Ww

    Tn = test_statistic(u_te, v_te, w_te)
    Tn_pruned, k = test_statistic_pruned(u_te, v_te, w_te, perc_dim_prune=perc_dim_prune)
    return {"T_n": Tn, "T_n_pruned": Tn_pruned, "pruned_dim": k,
            "d": model.d, "n_test": Xte.shape[0], "E_val": val_err}
