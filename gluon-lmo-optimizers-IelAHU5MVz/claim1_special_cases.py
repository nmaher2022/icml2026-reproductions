# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas"]
# ///
"""Claim 1 audit: Algorithm 1/2 special-case reductions (Section 4.1 / App. C.1).

Claim (verbatim): "The Gluon framework unifies layer-wise LMO-based optimizers,
recovering Muon (spectral norm), unScion (mixed norms for transformer blocks
and embeddings), layer-wise normalized GD, and layer-wise signGD as special
cases via Algorithm 1 (Algorithm 1, Section on special cases)."

This is a claim about ALGEBRAIC IDENTITY: each named optimizer's closed-form
update is *literally* the general LMO_{B_i^k}(g) formula

    X_i^{k+1} = X_i^k - t_i^k * s(g_i)      where s(g) = argmax_{||s||_(i)<=1} <g,s>

evaluated at a particular per-layer norm ||.||_(i). We implement:

  1. A single generic `general_lmo` that takes a "base" norm family
     (spectral / euclidean / max) plus a scalar norm multiplier c (so that
     ||.||_(i) = c * ||.||_base), and returns X0 - (t/c) * s_base(g). This is
     the literal LMO closed form for a scaled norm ball (scaling a norm by c
     scales its unit ball's radius by 1/c, which scales the LMO step by 1/c).
  2. Each special case's closed-form update exactly as printed in the
     briefing / paper (Muon's SVD update, unScion's block + embed/output
     updates for both the LLM and CNN variants, layer-wise normalized GD,
     layer-wise signGD), with NO reference to `general_lmo` internals.

For each special case we draw random gradients (and random current-iterate
X0) at several shapes/seeds/stepsizes, and check that
`general_lmo(g, family, c, t, X0)` and the case's closed form agree to
float precision. This is the actual mathematical content of claim 1: these
optimizers are literal instances of one general framework (same formula,
different norm), not just qualitatively similar heuristics.

We test the four cases named in the claim text (Muon, unScion LLM
block+embed, normalized GD, signGD) plus the paper's unScion-CNN variant
(bias / conv / head) as a bonus stress test of the same general_lmo, since
its formulas are also given verbatim in Section 4.1 / App C.1.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

TOL = 1e-8


# ----------------------------------------------------------------------------
# 1. The single generic Gluon LMO step: X^{k+1} = X0 - (t/c) * s_base(g)
#    where s_base(g) = argmax_{||s||_base <= 1} <g, s>, i.e. the LMO
#    direction for the UNSCALED base norm, and ||.||_(i) = c * ||.||_base.
# ----------------------------------------------------------------------------
def base_lmo_direction(g: np.ndarray, family: str) -> np.ndarray:
    """s_base(g) = argmax_{||s||_base<=1} <g,s> for the three base norms used
    in the paper's special cases: spectral (2->2), euclidean (2, Frobenius
    for matrices), max (entrywise sup / induced 1->infinity)."""
    if family == "spectral":
        U, _S, Vt = np.linalg.svd(g, full_matrices=False)
        return U @ Vt
    elif family == "euclidean":
        return g / np.linalg.norm(g)
    elif family == "max":
        return np.sign(g)
    else:
        raise ValueError(f"unknown norm family {family!r}")


def general_lmo(g: np.ndarray, family: str, c: float, t: float, X0: np.ndarray) -> np.ndarray:
    """LMO_{B(X0,t)}(g) for the norm ||.||_(i) = c * ||.||_base."""
    return X0 - (t / c) * base_lmo_direction(g, family)


# ----------------------------------------------------------------------------
# 2. Closed-form special cases, transcribed directly from the briefing.
# ----------------------------------------------------------------------------
def muon_closed(X0, g, t):
    # ||.||_(i) = ||.||_{2->2} (spectral norm). X^{k+1} = X0 - t * U V^T.
    U, _S, Vt = np.linalg.svd(g, full_matrices=False)
    return X0 - t * (U @ Vt)


def unscion_llm_block_closed(X0, g, t):
    # ||.||_(i) = sqrt(n_i/m_i) * ||.||_{2->2}, X_i in R^{m_i x n_i}.
    # X_i^{k+1} = X_i^k - t * sqrt(m_i/n_i) * U V^T.
    m, n = g.shape
    U, _S, Vt = np.linalg.svd(g, full_matrices=False)
    return X0 - t * np.sqrt(m / n) * (U @ Vt)


def unscion_llm_embed_closed(X0, g, t):
    # ||.||_(p) = n_p * ||.||_{1->inf}. X_p^{k+1} = X_p^k - (t/n_p)*sign(g).
    n_p = g.shape[1]
    return X0 - (t / n_p) * np.sign(g)


def normalized_gd_closed(X0, g, t):
    # ||.||_(i) = ||.||_2 (Euclidean, vector case). X^{k+1} = X0 - t*g/||g||_2.
    return X0 - t * g / np.linalg.norm(g)


def signgd_closed(X0, g, t):
    # ||.||_(i) = ||.||_infty. X^{k+1} = X0 - t*sign(g).
    return X0 - t * np.sign(g)


def unscion_cnn_bias_closed(X0, g, t):
    # biases: X^{k+1} = X0 - t*sqrt(C_out) * g/||g||_2.
    c_out = g.shape[0]
    return X0 - t * np.sqrt(c_out) * g / np.linalg.norm(g)


def unscion_cnn_conv_closed(X0, g, t, c_in, k):
    # conv kernels reshaped to C_out x (C_in*k*k):
    # X^{k+1} = X0 - t * (1/k^2) * sqrt(C_out/C_in) * U V^T.
    c_out = g.shape[0]
    U, _S, Vt = np.linalg.svd(g, full_matrices=False)
    return X0 - t * (1.0 / k ** 2) * np.sqrt(c_out / c_in) * (U @ Vt)


def unscion_cnn_head_closed(X0, g, t):
    # head: same sign-update form as the LLM embed/output group.
    n_p = g.shape[1]
    return X0 - (t / n_p) * np.sign(g)


# ----------------------------------------------------------------------------
# 3. Test harness: for each special case, compare general_lmo(...) against
#    the closed form on random (X0, g) at several shapes/seeds/stepsizes.
# ----------------------------------------------------------------------------
def run_case(name, family, c_fn, closed_fn, shape, seed, t):
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(shape)
    X0 = rng.standard_normal(shape)
    c = c_fn(g)
    general = general_lmo(g, family, c, t, X0)
    closed = closed_fn(X0, g, t)
    diff = float(np.max(np.abs(general - closed)))
    return {
        "special_case": name,
        "seed": seed,
        "shape": str(shape),
        "t": t,
        "max_abs_diff": diff,
        "pass": bool(diff < TOL),
    }


def build_jobs(seeds, ts, mat_shapes, vec_lens, embed_shapes, conv_specs):
    jobs = []

    # 1. Muon: spectral norm, c=1, on square/rectangular hidden-layer matrices.
    for shape, seed, t in itertools.product(mat_shapes, seeds, ts):
        jobs.append(("muon", "spectral", lambda g: 1.0, muon_closed, shape, seed, t))

    # 2. unScion (LLM) block layers: spectral norm scaled by sqrt(n/m).
    for shape, seed, t in itertools.product(mat_shapes, seeds, ts):
        jobs.append((
            "unscion_llm_block", "spectral",
            lambda g: np.sqrt(g.shape[1] / g.shape[0]),
            unscion_llm_block_closed, shape, seed, t,
        ))

    # 3. unScion (LLM) embed/output group: max norm scaled by n_p.
    for shape, seed, t in itertools.product(embed_shapes, seeds, ts):
        jobs.append((
            "unscion_llm_embed", "max",
            lambda g: g.shape[1],
            unscion_llm_embed_closed, shape, seed, t,
        ))

    # 4. Layer-wise normalized GD: Euclidean norm, c=1, vector layers.
    for n, seed, t in itertools.product(vec_lens, seeds, ts):
        jobs.append((
            "normalized_gd", "euclidean", lambda g: 1.0,
            normalized_gd_closed, (n,), seed, t,
        ))

    # 5. Layer-wise signGD: max norm, c=1, vector layers.
    for n, seed, t in itertools.product(vec_lens, seeds, ts):
        jobs.append((
            "signgd", "max", lambda g: 1.0,
            signgd_closed, (n,), seed, t,
        ))

    # 6. unScion (CNN) bias: Euclidean norm scaled by 1/sqrt(C_out).
    for n, seed, t in itertools.product(vec_lens, seeds, ts):
        jobs.append((
            "unscion_cnn_bias", "euclidean",
            lambda g: 1.0 / np.sqrt(g.shape[0]),
            unscion_cnn_bias_closed, (n,), seed, t,
        ))

    # 7. unScion (CNN) conv (reshaped 2D kernel): spectral norm scaled by
    #    k^2 * sqrt(C_in/C_out).
    for (c_out, c_in, k), seed, t in itertools.product(conv_specs, seeds, ts):
        shape = (c_out, c_in * k * k)

        def closed(X0, g, t, c_in=c_in, k=k):
            return unscion_cnn_conv_closed(X0, g, t, c_in, k)

        jobs.append((
            "unscion_cnn_conv", "spectral",
            lambda g, c_in=c_in, k=k: (k ** 2) * np.sqrt(c_in / g.shape[0]),
            closed, shape, seed, t,
        ))

    # 8. unScion (CNN) head: same as LLM embed/output, on head-shaped matrices.
    for shape, seed, t in itertools.product(embed_shapes, seeds, ts):
        jobs.append((
            "unscion_cnn_head", "max",
            lambda g: g.shape[1],
            unscion_cnn_head_closed, shape, seed, t,
        ))

    return jobs


def run_jobs(jobs):
    rows = []
    for name, family, c_fn, closed_fn, shape, seed, t in jobs:
        rng = np.random.default_rng(seed)
        g = rng.standard_normal(shape)
        X0 = rng.standard_normal(shape)
        c = c_fn(g)
        general = general_lmo(g, family, c, t, X0)
        closed = closed_fn(X0, g, t)
        diff = float(np.max(np.abs(general - closed)))
        rows.append({
            "special_case": name,
            "seed": seed,
            "shape": str(shape),
            "t": t,
            "max_abs_diff": diff,
            "pass": bool(diff < TOL),
        })
    return rows


def main(smoketest: bool):
    if smoketest:
        seeds = [0]
        ts = [0.1]
        mat_shapes = [(4, 3)]
        vec_lens = [5]
        embed_shapes = [(6, 4)]
        conv_specs = [(4, 2, 3)]
        out_csv = None
    else:
        seeds = [0, 1, 2, 3, 4]
        ts = [0.01, 0.1, 1.0, 3.7]
        mat_shapes = [(8, 12), (16, 16), (6, 10), (12, 6)]
        vec_lens = [8, 16, 32]
        embed_shapes = [(12, 20), (16, 16), (30, 8)]
        conv_specs = [(8, 4, 3), (16, 8, 3), (6, 3, 5)]
        out_csv = "claim1_special_cases.csv"

    jobs = build_jobs(seeds, ts, mat_shapes, vec_lens, embed_shapes, conv_specs)
    rows = run_jobs(jobs)
    df = pd.DataFrame(rows)

    print(f"Total comparisons: {len(df)}")
    summary = df.groupby("special_case")["max_abs_diff"].agg(["max", "mean", "count"])
    print(summary)
    n_fail = int((~df["pass"]).sum())
    print(f"Failures (max_abs_diff >= {TOL}): {n_fail}")
    if n_fail:
        print(df[~df["pass"]])

    if out_csv is not None:
        df.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv} ({len(df)} rows)")

    return df


if __name__ == "__main__":
    import sys

    smoke = "--full" not in sys.argv
    main(smoketest=smoke)
