# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///
"""Fern (Ellipsoidal TSF, arXiv 2505.17370v6) reimplemented from Algorithm 1 (Section 2) +
complexity analysis (Appendix A.3.2), plus DLinear baseline and the paper's own metrics
(Appendix A.1: W2/WD, SWD, EPT). See PAPER_BRIEFING.md for full equation transcription and the
patch-parallel OT-head design this follows.

Ablation flags (`no_rotation`, `no_encoder`, `no_patching`) reproduce Table 3/Table 8's variants.
Run via `.venv/bin/python`, not `uv run` (torch is CPU-only, pre-installed in the repo venv --
`uv run` on a torch-dependent script fetches a fresh CUDA build from PyPI instead).
"""
import math
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Metrics (Appendix A.1)
# ---------------------------------------------------------------------------

def mse(pred, true):
    return torch.mean((pred - true) ** 2).item()


def w2_1d(pred, true):
    """W2^2(y*, y) = (1/H) sum_h (y*_(h) - y_(h))^2 using per-sample sorted order statistics,
    averaged over the batch."""
    p_sorted, _ = torch.sort(pred, dim=-1)
    t_sorted, _ = torch.sort(true, dim=-1)
    return torch.mean((p_sorted - t_sorted) ** 2).item()


def sliced_wd(pred, true, n_proj=500, seed=0):
    """Project each (B, H) batch of horizon vectors onto L random unit directions in R^H, take the
    1D W2 (sorted) distance per projection, average over projections and batch."""
    B, H = pred.shape
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(H, n_proj, generator=g)
    dirs = dirs / dirs.norm(dim=0, keepdim=True)
    p_proj = pred @ dirs  # (B, n_proj)
    t_proj = true @ dirs
    p_sorted, _ = torch.sort(p_proj, dim=0)
    t_sorted, _ = torch.sort(t_proj, dim=0)
    return torch.mean((p_sorted - t_sorted) ** 2).item()


def effective_prediction_time(pred, true, train_std):
    """EPT_b = min{s : |pred - true| > eps} (eps = train-set std, single channel here), = H if
    never exceeded. Returns mean over batch."""
    err = (pred - true).abs()
    B, H = err.shape
    exceeded = err > train_std
    idx = torch.arange(1, H + 1).unsqueeze(0).expand(B, H)
    first_exceed = torch.where(exceeded, idx, torch.full_like(idx, H + 1))
    ept = first_exceed.min(dim=1).values.clamp(max=H).float()
    return ept.mean().item()


def all_metrics(pred, true, train_std):
    return dict(mse=mse(pred, true), wd=w2_1d(pred, true),
                swd=sliced_wd(pred, true), ept=effective_prediction_time(pred, true, train_std))


# ---------------------------------------------------------------------------
# Fern (Algorithm 1)
# ---------------------------------------------------------------------------

def soft_clamp(x, lo, hi):
    """Linear within [lo,hi], saturates smoothly outside -- paper Appendix A.3.3."""
    return lo + (hi - lo) * torch.sigmoid(x)


class CouplingHead(nn.Module):
    """H_*(v): feature extractor R^d_in -> R^dh, followed by a scale/shift head phi: R^dh ->
    (R^d_out, R^d_out), matching Algorithm 1's h^i = H(v^i); (s, t) = phi(h)."""

    def __init__(self, d_in, d_out, dh):
        super().__init__()
        self.feat = nn.Sequential(nn.Linear(d_in, dh), nn.Tanh(), nn.Linear(dh, dh), nn.Tanh())
        self.scale_head = nn.Linear(dh, d_out)
        self.shift_head = nn.Linear(dh, d_out)

    def forward(self, v):
        h = self.feat(v)
        s = soft_clamp(self.scale_head(h), 0.0, 5.5)
        t = 15.0 * torch.tanh(self.shift_head(h))
        return h, s, t


class OTHead(nn.Module):
    """psi(h_z) -> per-patch (Lambda, t_y, Householder vectors), applied to all g patches at once
    (parallel, matching the paper's patchwise-parallel-transport design)."""

    def __init__(self, dh, n_patches, patch_size, n_reflections):
        super().__init__()
        self.g, self.p, self.R = n_patches, patch_size, n_reflections
        out_dim = n_patches * (patch_size + patch_size + n_reflections * patch_size)
        self.net = nn.Linear(dh, out_dim)

    def forward(self, h):
        B = h.shape[0]
        out = self.net(h).view(B, self.g, 2 * self.p + self.R * self.p)
        lam_raw, ty_raw, v_raw = torch.split(out, [self.p, self.p, self.R * self.p], dim=-1)
        lam = torch.nn.functional.softplus(lam_raw)  # nonnegative eigenvalues
        ty = 15.0 * torch.tanh(ty_raw)
        v = v_raw.view(B, self.g, self.R, self.p)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)  # unit-norm Householder vectors
        return lam, ty, v


def apply_householder(v, y):
    """y: (..., p). Apply U = H_R ... H_1 to y via R reflections H_i = I - 2 v_i v_i^T."""
    R = v.shape[-2]
    out = y
    for r in range(R):
        vr = v[..., r, :]
        out = out - 2 * (out * vr).sum(-1, keepdim=True) * vr
    return out


class Fern(nn.Module):
    def __init__(self, context_len, horizon_len, patch_size=24, n_reflections=8, dh=32,
                 kenc=5, no_rotation=False, no_encoder=False, no_patching=False):
        super().__init__()
        self.n, self.H = context_len, horizon_len
        self.p = horizon_len if no_patching else patch_size
        assert self.H % self.p == 0, "horizon_len must be divisible by patch_size"
        self.g = self.H // self.p
        # Keep head capacity (R) fixed regardless of no_rotation so the ablation isolates the
        # rotation operation itself, not a smaller OT head -- R=1 here would shrink the head's
        # output dim and confound "no rotation" with "less capacity" (self-audit finding).
        self.R = n_reflections
        self.no_rotation = no_rotation
        self.no_encoder = no_encoder
        self.kenc = 0 if no_encoder else kenc

        self.Hx = CouplingHead(self.n, self.n, dh)
        self.Hz = CouplingHead(self.n, self.n, dh)
        self.final_Hz = nn.Sequential(nn.Linear(self.n, dh), nn.Tanh())
        self.ot_head = OTHead(dh, self.g, self.p, self.R)

    def forward(self, x):
        B = x.shape[0]
        z = torch.randn(B, self.n)
        xi, zi = x, z
        for _ in range(self.kenc):
            _, sz, tz = self.Hx(xi)
            zi = sz * zi + tz
            _, sx, tx = self.Hz(zi)
            xi = sx * xi + tx
        hz = self.final_Hz(zi)  # (B, dh)
        lam, ty, v = self.ot_head(hz)  # (B, g, p) / (B, g, p) / (B, g, R, p)

        # mean prediction per patch: mu_j = U_j^T Lambda_j U_j @ t_y_j  (y0 = 0 for point forecast)
        if self.no_rotation:
            mu = lam * ty  # identity rotation: elementwise diagonal scaling only
        else:
            u_ty = apply_householder(v, ty)          # U @ t_y
            scaled = lam * u_ty                        # Lambda @ (U @ t_y)
            mu = apply_householder(v.flip(-2), scaled)  # U^T @ (...)  (reverse reflection order)
        return mu.reshape(B, self.H)

    def param_flop_report(self):
        """Analytic parameter/FLOP accounting for Claim 2, counted directly from this instance's
        actual module sizes (not a separate hardcoded formula)."""
        n, p, g, R, dh = self.n, self.p, self.g, self.R, self.ot_head.net.in_features
        n_params = sum(p_.numel() for p_ in self.parameters())
        # Householder-factored OT head FLOPs (per Eq. 1-2): O(g*(p*dh_head + R*p)) for the head
        # itself, ignoring the shared encoder (same for both variants being compared).
        ot_head_flops = g * (2 * p * dh + R * p)
        dense_spd_flops = g * (p * p)  # O(g * p^2): applying a dense p x p SPD map per patch
        return dict(n_params=n_params, patch_size=p, n_patches=g, n_reflections=R,
                    fern_head_flops=ot_head_flops, dense_spd_flops=dense_spd_flops,
                    ratio=dense_spd_flops / max(ot_head_flops, 1))


# ---------------------------------------------------------------------------
# DLinear baseline (Zeng et al. 2023)
# ---------------------------------------------------------------------------

class MovingAvg(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=0)

    def forward(self, x):
        front = x[:, 0:1].repeat(1, (self.kernel_size - 1) // 2)
        end = x[:, -1:].repeat(1, self.kernel_size - (self.kernel_size - 1) // 2 - 1)
        x_pad = torch.cat([front, x, end], dim=1)
        return self.avg(x_pad.unsqueeze(1)).squeeze(1)


class DLinear(nn.Module):
    def __init__(self, context_len, horizon_len, kernel_size=25):
        super().__init__()
        self.decomp = MovingAvg(kernel_size)
        self.linear_trend = nn.Linear(context_len, horizon_len)
        self.linear_seasonal = nn.Linear(context_len, horizon_len)

    def forward(self, x):
        trend = self.decomp(x)
        seasonal = x - trend
        return self.linear_trend(trend) + self.linear_seasonal(seasonal)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(model, Xtr, Ytr, Xval, Yval, epochs=60, lr=1e-3, batch_size=64, patience=10,
                 verbose=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.HuberLoss()
    Xtr_t, Ytr_t = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    Xval_t, Yval_t = torch.from_numpy(Xval), torch.from_numpy(Yval)
    n = len(Xtr_t)
    best_val, best_state, bad_epochs = float("inf"), None, 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr_t[idx], Ytr_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_pred = model(Xval_t)
            val_loss = loss_fn(val_pred, Yval_t).item()
        if verbose:
            print(f"  epoch {ep}: train_loss={total_loss / n:.4f} val_loss={val_loss:.4f}")
        if math.isnan(val_loss):
            raise RuntimeError("NaN val loss during training")
        if val_loss < best_val - 1e-5:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate(model, Xte, Yte, train_std):
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte))
    return all_metrics(pred, torch.from_numpy(Yte), train_std)
