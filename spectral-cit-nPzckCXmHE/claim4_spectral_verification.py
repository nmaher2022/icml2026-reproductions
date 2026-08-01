# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scipy"]
# ///
"""Claim 4 verification: does Algorithm 1 actually estimate the leading spectral
directions of the partial cross-covariance operator Sigma_{X,Y|Z}, or does it just
match the pseudocode's control flow?

The README's current Claim 4 verdict ("VERIFIED (structural)") only checked that
`scit_lib.py` implements the same steps as the paper's Algorithm 1 box -- it never
checked that the *learned representations* are the thing the paper says they are:
estimates of Sigma_{X,Y|Z}'s leading singular directions (Section 3, the
partial-cross-covariance operator whose HSIC norm is the CI test statistic).

This script builds a synthetic (X, Y, Z) with an EXACT, closed-form ground truth for
Sigma_{X,Y|Z}, by construction:

    Z ~ N(0, I_dz)                       (confounder)
    S ~ N(0, I_r)                        (latent CI-breaking signal, r << dx, dy)
    X = Z @ A  + S @ Bx.T + noise_x * eps_x   (dz->dx confound + rank-r signal + noise)
    Y = Z @ A' + S @ By.T + noise_y * eps_y   (dz->dy confound + rank-r signal + noise)

    Bx = Ux @ diag(sigma), By = Uy   (Ux, Uy: orthonormal dx x r / dy x r bases)

Schur-complement identity for jointly-Gaussian (X, Y, Z):
    Sigma_{X,Y|Z} = Cov(X,Y) - Cov(X,Z) Cov(Z,Z)^-1 Cov(Z,Y)
                  = [A.T A' + Bx By.T] - A.T @ I @ A'
                  = Bx @ By.T = Ux @ diag(sigma) @ Uy.T          <- exact, any A, A'

So the TRUE leading left/right singular directions of Sigma_{X,Y|Z} are exactly Ux's
and Uy's columns, regardless of how much (nuisance) variance Z injects into X, Y
individually -- that nuisance cancels exactly in the partial covariance, but the
network never gets told this; it has to discover it from data alone.

Verification: train the real, unmodified `scit_lib.py` Algorithm 1 on this data, then
check whether the trained + whitened test-set embedding u_theta(X) preferentially
aligns (via canonical correlation) with the TRUE signal directions X @ Ux, compared to
two negative controls of matched dimension: (a) a random direction in X's orthogonal
complement of Ux (same marginal noise level, no CI-relevant signal) and (b) i.i.d.
Gaussian noise unrelated to X entirely (a CCA-implementation sanity check -- should be
~0). Mirrored for v_theta(Y,Z) vs Y @ Uy.

If u/v systematically pick out Ux/Uy over the matched-dimension noise controls, that's
real evidence Algorithm 1 estimates the claimed spectral directions, not just that the
code parses correctly. If it doesn't, Claim 4's "structural" verdict needs downgrading.
"""
from __future__ import annotations

import csv
import sys
import time

import numpy as np
import torch

from scit_lib import train_spectral_model, whiten, center

D_EMBED = 10          # embedding dim d, matches Table 2 reference hyperparameters
DZ, DX, DY = 3, 6, 6  # confounder / X / Y ambient dims
R = 2                 # true rank of Sigma_{X,Y|Z}
SIGMA = np.array([3.0, 1.5])   # true singular values of Sigma_{X,Y|Z}
Z_SCALE = 0.6         # shrinks the Z->X, Z->Y confound so it doesn't dominate
NOISE_X, NOISE_Y = 0.5, 0.5
N = 1000
N_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10

TRAIN_KW = dict(
    d=D_EMBED, n_hidden=2, layer_size=128,
    lr_inner=3e-5, lr_outer=2.1e-3, reg_inner=3.3, reg_outer=1.9,
    batch_size=128, n_epochs=400, warmup_steps=100,
)


def gen_partial_cov_signal(rng: np.random.Generator):
    """Returns X, Y, Z, Ux, Uy (true signal bases) with Sigma_{X,Y|Z} = Ux @ diag(SIGMA) @ Uy.T
    exactly (population level), by the Schur-complement cancellation in the module docstring."""
    Z = rng.normal(0, 1, size=(N, DZ))
    S = rng.normal(0, 1, size=(N, R))
    A = Z_SCALE * rng.normal(0, 1, size=(DZ, DX))
    Ap = Z_SCALE * rng.normal(0, 1, size=(DZ, DY))
    Ux, _ = np.linalg.qr(rng.normal(size=(DX, R)))
    Uy, _ = np.linalg.qr(rng.normal(size=(DY, R)))
    Bx = Ux * SIGMA[None, :]
    By = Uy
    X = Z @ A + S @ Bx.T + NOISE_X * rng.normal(size=(N, DX))
    Y = Z @ Ap + S @ By.T + NOISE_Y * rng.normal(size=(N, DY))
    return X.astype(np.float32), Y.astype(np.float32), Z.astype(np.float32), Ux, Uy


def orthogonal_complement_basis(U: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k orthonormal columns spanning a subspace of U's orthogonal complement (same
    ambient dim), for a dimension-matched 'noise direction' negative control."""
    dim = U.shape[0]
    full, _ = np.linalg.qr(np.concatenate([U, rng.normal(size=(dim, dim - U.shape[1]))], axis=1))
    return full[:, U.shape[1]:U.shape[1] + k]


def cca_correlations(A: np.ndarray, B: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Canonical correlations between A (n,p) and B (n,q): covariance-whitening + SVD
    of the cross-covariance, same inverse-sqrt-via-eigh construction as scit_lib.whiten."""
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    n = A.shape[0]
    Caa = (A.T @ A) / n
    Cbb = (B.T @ B) / n
    Cab = (A.T @ B) / n

    def inv_sqrt(cov):
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, eps, None)
        return vecs @ np.diag(vals ** -0.5) @ vecs.T

    M = inv_sqrt(Caa) @ Cab @ inv_sqrt(Cbb)
    svals = np.linalg.svd(M, compute_uv=False)
    return np.clip(svals, 0.0, 1.0)


print(f"Claim 4 verification: does u_theta(X)/v_theta(Y,Z) preferentially recover the "
      f"TRUE leading singular directions of Sigma_XY|Z (rank {R}, sigma={SIGMA.tolist()}) "
      f"over dimension-matched noise controls? dx=dy={DX}, dz={DZ}, N={N}, {N_REPS} reps.")

rng = np.random.default_rng(42)
raw_rows = []
t0 = time.time()

for rep in range(N_REPS):
    X, Y, Z, Ux, Uy = gen_partial_cov_signal(rng)
    Ux_perp = orthogonal_complement_basis(Ux, R, rng)
    Uy_perp = orthogonal_complement_basis(Uy, R, rng)

    n_train = int(0.8 * N)
    Xt, Yt, Zt = (torch.as_tensor(a) for a in (X, Y, Z))
    Xtr, Xte = Xt[:n_train], Xt[n_train:]
    Ytr, Yte = Yt[:n_train], Yt[n_train:]
    Ztr, Zte = Zt[:n_train], Zt[n_train:]

    seed = int(rng.integers(0, 2**31 - 1))
    model = train_spectral_model(Xtr, Ytr, Ztr, seed=seed, **TRAIN_KW)
    (u_tr, v_tr, w_tr), (Wu, Wv, Ww) = whiten(model, Xtr, Ytr, Ztr)
    with torch.no_grad():
        u_te_raw, v_te_raw, _ = model(Xte, Yte, Zte)
    u_te = (center(u_te_raw) @ Wu).numpy()
    v_te = (center(v_te_raw) @ Wv).numpy()

    Xte_np, Yte_np = X[n_train:], Y[n_train:]
    noise_rng = np.random.default_rng(seed)
    rand_target = noise_rng.normal(size=(Xte_np.shape[0], R))

    row = {"rep": rep}
    for side, emb, data_te, Utrue, Uperp in (
        ("u", u_te, Xte_np, Ux, Ux_perp),
        ("v", v_te, Yte_np, Uy, Uy_perp),
    ):
        cca_sig = cca_correlations(emb, data_te @ Utrue)
        cca_noise = cca_correlations(emb, data_te @ Uperp)
        cca_rand = cca_correlations(emb, rand_target)
        row[f"{side}_signal_mean"] = float(cca_sig.mean())
        row[f"{side}_signal_top"] = float(cca_sig[0])
        row[f"{side}_noise_mean"] = float(cca_noise.mean())
        row[f"{side}_noise_top"] = float(cca_noise[0])
        row[f"{side}_random_mean"] = float(cca_rand.mean())
    raw_rows.append(row)
    print(f"[{time.time()-t0:6.1f}s] rep {rep+1}/{N_REPS}: "
          f"u signal={row['u_signal_mean']:.3f} noise={row['u_noise_mean']:.3f} rand={row['u_random_mean']:.3f} | "
          f"v signal={row['v_signal_mean']:.3f} noise={row['v_noise_mean']:.3f} rand={row['v_random_mean']:.3f}")

fieldnames = ["rep", "u_signal_mean", "u_signal_top", "u_noise_mean", "u_noise_top", "u_random_mean",
              "v_signal_mean", "v_signal_top", "v_noise_mean", "v_noise_top", "v_random_mean"]
with open("claim4_raw.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(raw_rows)

summary_rows = []
for side in ("u", "v"):
    def col(name):
        return np.array([r[f"{side}_{name}"] for r in raw_rows])
    summary_rows.append({
        "side": side,
        "n_reps": N_REPS,
        "signal_mean_avg": float(col("signal_mean").mean()),
        "signal_mean_std": float(col("signal_mean").std()),
        "noise_mean_avg": float(col("noise_mean").mean()),
        "noise_mean_std": float(col("noise_mean").std()),
        "random_mean_avg": float(col("random_mean").mean()),
        "gap_signal_minus_noise": float((col("signal_mean") - col("noise_mean")).mean()),
    })
    print(f"\n{side}_theta summary: signal={summary_rows[-1]['signal_mean_avg']:.3f}"
          f"+-{summary_rows[-1]['signal_mean_std']:.3f}  noise={summary_rows[-1]['noise_mean_avg']:.3f}"
          f"+-{summary_rows[-1]['noise_mean_std']:.3f}  random={summary_rows[-1]['random_mean_avg']:.3f}"
          f"  gap={summary_rows[-1]['gap_signal_minus_noise']:.3f}")

with open("claim4_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    w.writeheader()
    w.writerows(summary_rows)

print("\nWrote claim4_raw.csv, claim4_summary.csv")
