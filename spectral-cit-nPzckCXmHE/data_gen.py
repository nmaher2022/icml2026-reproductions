# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Synthetic data generators transcribed from the paper (Section 5.1, Appendix C).

See PAPER_BRIEFING.md "Synthetic benchmarks used for Claims 1-2" for the equations and
the section/figure each generator corresponds to.
"""
from __future__ import annotations

import numpy as np


def signal_strength_ablation(N: int, d_z: int, str_cond_dep: float, null: bool,
                              str_z: float = 0.1, noise_str: float = 0.25,
                              rng: np.random.Generator | None = None):
    """Fig. 11 setup (Appendix C, cheapest synthetic benchmark -- good smoketest).

    H0: X = sin(Z + eps_X), Y = cos(Z + eps_Y)
    H1: X = sin(Z + eps_X) + eta, Y = cos(Z + eps_Y) + eta, eta ~ N(0, str_cond_dep^2 I)
    Z, eps_X, eps_Y ~ N(0, str_z^2 I_dz) [eps uses noise_str, see paper Appendix C text:
    "noise_str = 0.25" for eps_X, eps_Y].
    """
    rng = rng or np.random.default_rng()
    Z = rng.normal(0, str_z, size=(N, d_z))
    eps_X = rng.normal(0, noise_str, size=(N, d_z))
    eps_Y = rng.normal(0, noise_str, size=(N, d_z))
    X = np.sin(Z + eps_X)
    Y = np.cos(Z + eps_Y)
    if not null:
        eta = rng.normal(0, str_cond_dep, size=(N, d_z))
        X = X + eta
        Y = Y + eta
    return X, Y, Z


def post_nonlinear_model(N: int, d_z: int, null: bool, rng: np.random.Generator | None = None):
    """Fig. 2 primary synthetic benchmark (Section 5.1).

    Zbar = mean(Z), Z_i, eps_X, eps_Y, eps ~ iid N(0,1).
    H0: X = f(Zbar + eps_X/4), Y = g(Zbar + eps_Y/4)
    H1: X = f(Zbar + eps_X/4) + eps/2, Y = g(Zbar + eps_Y/4) + eps/2
    f(w) = w^3, g(w) = tanh(w)
    """
    rng = rng or np.random.default_rng()
    Z = rng.normal(0, 1, size=(N, d_z))
    Zbar = Z.mean(axis=1, keepdims=True)
    eps_X = rng.normal(0, 1, size=(N, 1))
    eps_Y = rng.normal(0, 1, size=(N, 1))
    X = (Zbar + eps_X / 4) ** 3
    Y = np.tanh(Zbar + eps_Y / 4)
    if not null:
        eps = rng.normal(0, 1, size=(N, 1))
        X = X + eps / 2
        Y = Y + eps / 2
    return X, Y, Z


def high_dim_nonsmooth(N: int, d_z: int, null: bool, rng: np.random.Generator | None = None):
    """Fig. 10 high-dimensional nonsmooth benchmark (Appendix C).

    X = f(Z/2 + eps_X), Y = g(Z/2 + eps_Y), f,g highly oscillatory near 0:
    h_k(w) = w^k for |w|>=1; cos(2pi/w) for 0<|w|<1; 1 for w=0. f=h_2, g=h_3.
    Shared noise added under H1 (paper text: "shared noise has low variance sd 0.15").
    """
    rng = rng or np.random.default_rng()

    def h(w, k):
        out = np.empty_like(w)
        abs_w = np.abs(w)
        big = abs_w >= 1
        zero = w == 0
        mid = ~big & ~zero
        out[big] = w[big] ** k
        out[mid] = np.cos(2 * np.pi / w[mid])
        out[zero] = 1.0
        return out

    Z = rng.normal(0, 1, size=(N, d_z))
    eps_X = rng.normal(0, 1, size=(N, d_z))
    eps_Y = rng.normal(0, 1, size=(N, d_z))
    X = h(Z / 2 + eps_X, 2)
    Y = h(Z / 2 + eps_Y, 3)
    if not null:
        shared = rng.normal(0, 0.15, size=(N, d_z))
        X = X + shared
        Y = Y + shared
    return X, Y, Z
