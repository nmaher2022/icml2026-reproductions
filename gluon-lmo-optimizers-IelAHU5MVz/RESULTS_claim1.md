# Claim 1 — Algorithm 1/2 special-case reductions

## Claim text (verbatim, from `claims_anchored.json`)

> "The Gluon framework unifies layer-wise LMO-based optimizers, recovering
> Muon (spectral norm), unScion (mixed norms for transformer blocks and
> embeddings), layer-wise normalized GD, and layer-wise signGD as special
> cases via Algorithm 1 (Algorithm 1, Section on special cases)."

This is a claim about **algebraic identity**, not approximate similarity:
each named optimizer's update rule must literally equal the general Gluon
LMO step `X_i^{k+1} = X_i^k - t_i^k * s(g_i)`, `s(g) = argmax_{||s||_(i)<=1}
<g,s>`, for the specific per-layer norm `||.||_(i)` the paper assigns to it.

## What was run

`claim1_special_cases.py` (PEP-723, numpy + pandas only):

1. **`general_lmo(g, family, c, t, X0)`** — a single generic implementation
   of the LMO closed form for a *scaled* base norm `||.||_(i) = c *
   ||.||_base`, using only the three base directions given in the briefing:
   `spectral` (reduced SVD, `U V^T`), `euclidean` (`g/||g||_2`), `max`
   (`sign(g)`, entrywise). This is *not* copy-pasted from any special case —
   it is one routine, parameterized only by which base norm and what scalar
   multiplier `c` is used, exactly mirroring `||.||_(i) = c * ||.||_base` in
   the briefing's unScion formulas.
2. Five/eight **closed-form special-case update rules**, transcribed
   independently and directly from the briefing's formulas: Muon (SVD
   update), unScion-LLM block layers (`sqrt(m/n)`-scaled spectral), unScion-
   LLM embedding/output (`(1/n_p)`-scaled sign), layer-wise normalized GD,
   layer-wise signGD, and (bonus stress test, same section of the paper)
   unScion-CNN bias/conv/head updates.
3. For each case, random `(X0, g)` pairs at multiple shapes (matrices
   8x12, 16x16, 6x10, 12x6, 12x20, 30x8; vectors of length 8/16/32; conv
   kernels reshaped to `C_out x (C_in*k^2)` for `(C_out,C_in,k)` in
   `{(8,4,3),(16,8,3),(6,3,5)}`), 5 seeds (0-4), and 4 stepsizes
   (0.01, 0.1, 1.0, 3.7), computed both via `general_lmo` (with the
   appropriate `c`) and via the case's closed form, and recorded
   `max_abs_diff = max|general - closed|`.

**Smoketest** (1 seed, tiny shapes 4x3/5/6x4/conv(4,2,3), 1 stepsize, 8
comparisons) ran first and passed cleanly (diffs 0 to 5.6e-17) before the
full sweep — confirms no shape errors, no NaNs, no sign flips, before
scaling up.

**Full sweep**: 520 (special_case, seed, shape, t) comparisons across the 8
cases (`claim1_special_cases.csv`).

## Results

```
special_case         max_abs_diff (worst)   mean_abs_diff     count
muon                 0.0e+00                 0.0e+00            80
unscion_llm_block    8.9e-16                 8.3e-17            80
unscion_llm_embed    0.0e+00                 0.0e+00            60
normalized_gd        8.9e-16                 1.0e-16            60
signgd               0.0e+00                 0.0e+00            60
unscion_cnn_bias     1.8e-15                 3.0e-16            60
unscion_cnn_conv     2.2e-16                 4.6e-17            60
unscion_cnn_head     0.0e+00                 0.0e+00            60
```

Tolerance was `1e-8`; every single observed diff is at or below float64
machine epsilon (~1e-16 to 2e-15), i.e. the two code paths (generic LMO vs.
closed form) are computing bit-identical formulas up to ordinary
floating-point rounding. **520/520 comparisons pass, 0 failures.**

## Self-check against the claim text

The claim says Gluon "recovers Muon, unScion, layer-wise normalized GD, and
layer-wise signGD as special cases via Algorithm 1" (i.e. the same LMO
formula under different norms). The numerical evidence directly supports
exactly this: a single generic LMO routine, driven only by (a) which of the
three base norms in the paper (spectral / Euclidean / max) and (b) the
scalar norm multiplier `c_i` the paper prescribes per case, reproduces the
independently-transcribed closed-form update for Muon, both halves of
unScion (block spectral update and embedding/output sign update), layer-wise
normalized GD, and layer-wise signGD to float precision. This is a small-
tensor / synthetic-gradient audit — it does not run actual neural-network
training — but the claim itself is about an *algebraic identity between
update formulas*, which is scale-invariant: if the formulas match on
arbitrary shapes/seeds/stepsizes (as tested), they match for any real
gradient, since the LMO formula and the closed forms are pointwise
functions of `(X0, g, t)` with no dependence on how `g` was produced.

## Verdict: **VERIFIED**

All four special cases named explicitly in the claim (Muon, unScion,
layer-wise normalized GD, layer-wise signGD) — plus the CNN variant of
unScion given in the same section as a bonus check — are confirmed to be
exact instances of one general LMO update formula under the norm choices
the paper specifies, with zero failures across 520 (case, seed, shape,
stepsize) combinations and diffs at machine-precision level. This is not a
"TOY-VERIFIED" scale caveat situation: the claim is a formula-level identity
that either holds exactly or doesn't, and it holds exactly on every input
tested, including non-square/rectangular shapes that stress-test the
`sqrt(m/n)` and `1/n_p` scaling factors.

## Notes / surprises

- No discrepancies found versus the paper's stated formulas. The derivation
  that `general_lmo` uses — a norm `c * ||.||_base` has LMO step `X0 -
  (t/c) * s_base(g)` because scaling a norm by `c` scales its unit ball
  (hence the radius-`t` ball) by `1/c` — is standard but worth stating
  explicitly: it is exactly what lets the paper's `sqrt(m/n)`, `sqrt(n/m)`,
  `n_p`, `1/n_p`, `1/k^2`, `sqrt(C_out/C_in)` etc. factors fall out
  correctly, and the numerics confirm the paper's printed constants are
  internally consistent with this derivation (no off-by-reciprocal or
  off-by-sqrt errors in the briefing's transcription of the paper).
- The `unscion_cnn_conv` case additionally exercises the `k` (kernel size)
  and `C_in`/`C_out` asymmetry in the `1/k^2 * sqrt(C_out/C_in)` scaling
  factor, which wasn't strictly required by the claim's four named
  optimizers but is part of the same "special cases" section of the paper
  and passed with the same machine-precision agreement.
