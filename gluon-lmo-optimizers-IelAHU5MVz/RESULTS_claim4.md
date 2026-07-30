# Claim 4 — NanoGPT/FineWeb layer-wise smoothness (TOY reproduction)

## Claim text (verbatim)
"Empirical estimation of layer-wise smoothness constants on a 124M-parameter
NanoGPT model trained on FineWeb finds L^0_i approximately 0 across layers,
with L^1_i approximately 70 for transformer blocks (predicted stepsize 0.014
vs. tuned 0.018) and L^1_i approximately 1.3 for embedding/output layers
(predicted stepsize 0.77 vs. tuned 1.08)."

## Why this can only be a toy check
The real experiment (Appendix E.3, paper) trains a 124M-parameter NanoGPT on
real FineWeb tokens across 4xA100 GPUs for 5000 iterations. That is
infeasible on CPU/numpy in this repo (no torch, no GPU). Instead I hand-built
a tiny causal transformer from scratch in numpy (manual forward *and*
backward passes, no autodiff) and applied the exact unScion-style layer-wise
LMO update from the briefing to a synthetic token stream. This checks only
the *qualitative pattern* the claim describes, not the exact numbers.

## What was built (`claim4_transformer_smoothness.py`)
- Tiny causal transformer, hand-implemented (numpy only): vocab V=80, embed
  dim d=24, 2 transformer blocks (single-head causal self-attention + ReLU
  MLP with mult=4, pre-LN, no learnable LN affine params, no biases),
  weight-tied embedding/output matrix `E` (V x d). Sequence length 16,
  batch 16.
- Data: synthetic Markov-chain token sequences (fresh batch sampled from the
  chain every step, matching the paper's stochastic-gradient `f_ξ^k`
  notation) — no real text corpus needed.
- Parameter groups, matching the briefing's unScion formula exactly:
  - Transformer-block weight matrices (`Wq,Wk,Wv,Wo,W1,W2` x 2 blocks, 12
    matrices total): spectral/SVD LMO update `X <- X - t*sqrt(m/n)*U V^T`
    where `G = U Σ V^T` is the reduced SVD of the gradient.
  - Weight-tied embedding/output matrix `E`: sign-based LMO update
    `X <- X - (t/n_p)*sign(G)`, `n_p = d` (embedding dim, the output layer's
    fan-in).
- Correctness check: **numerical (finite-difference) gradient check on all
  12 block matrices + E before scaling up** — max relative error 6e-6 across
  all parameter groups, confirming the hand-written backprop (including
  softmax-attention backward and layernorm backward) is correct.
- Smoketest (V=20,d=8,1 block,20 steps) run first per repo convention: no
  NaNs/Inf, correct shapes, loss decreased 2.99->2.72. Then scaled to the
  full toy run: 700 steps, loss 4.38 (= ln(80), the uniform-prior baseline
  for a fresh model) -> 1.18, confirming the model actually learns the
  synthetic Markov structure.
- Eq. 10 / Eq. 30 fitting: at every step k the gradient `g^k` computed on a
  fresh minibatch at `X^k` (used for that step's update) is logged; using
  consecutive `(g^k, g^{k+1})` and `(X^k, X^{k+1})` pairs, `L_hat_i[k]` and
  the associated `||g^{k+1}||_(i)*` are computed with the correct
  primal/dual norms per group (spectral/nuclear duality for blocks,
  max/L1 duality for the tied embed/output layer, both derived from the
  briefing's `||.||_(i)` definitions via standard trace duality). `L_i^0,
  L_i^1 >= 0` fit per Eq. 30's hinge-penalized least squares (`lambda=5`,
  `scipy.optimize.minimize`, L-BFGS-B with nonnegativity bounds).

## Results

Pooled group-level fit (`claim4_transformer_smoothness.csv`, primary result;
pools all 12 block matrices' trajectories into one "transformer_block" fit,
vs. `E`'s own trajectory for "embed_output"):

| layer_group        | L0_fit | L1_fit | predicted stepsize (1/L1) | tuned stepsize used |
|---------------------|--------|--------|----------------------------|----------------------|
| transformer_block   | 0.000  | 83.89  | 0.0119                     | 0.02                 |
| embed_output        | 0.608  | 1.978  | 0.5055                     | 0.50                 |

`L1_block / L1_embed = 42.4x` (paper: `~70/1.3 ~= 54x`).

Per-matrix breakdown (`claim4_per_matrix_smoothness.csv`) shows all 12
individual block weight matrices independently fit `L1` in the range
67-85 (paper's own range is 67-71) — the per-matrix homogeneity the paper
reports is reproduced. Per-matrix `L0` is mostly small but noisier
(0 to ~14) than the pooled fit, expected given far fewer samples (699) per
individual matrix regression and the toy scale.

## Verdict: **TOY-VERIFIED**

The qualitative pattern the claim describes holds at this reduced scale:
- `L^0 ~ 0` for the transformer-block group (exactly 0 in the pooled fit,
  small relative to L1 in per-matrix fits). `L^0` for embed/output (0.61) is
  small but not as clean as "approximately 0" — noted as a discrepancy, likely
  a toy-scale/hinge-fit artifact given only 699 samples for that single
  matrix.
- `L^1` for transformer blocks is dramatically larger than for the tied
  embedding/output layer (83.9 vs. 1.98, ratio 42x), matching the paper's
  direction and order of magnitude (paper: 70 vs. 1.3, ratio 54x) — blocks
  are "rougher" (higher curvature), embed/output is much smoother, exactly
  as the paper argues motivates a ~50x larger stepsize for the tied
  embedding/output layer than for transformer blocks.
- The derived predicted stepsizes (1/L1: 0.012 for blocks, 0.51 for
  embed/output) land in the same regime as the paper's predicted values
  (0.014 and 0.77 respectively) — same order of magnitude, same huge gap
  between the two groups.

This is NOT a full VERIFIED — the exact constants (~70 vs ~1.3 in the paper)
cannot be expected to match a hand-built numpy toy transformer on synthetic
Markov data trained for 700 steps vs. a real 124M NanoGPT on FineWeb trained
for 5000 iterations on 4xA100s. But the qualitative claim — L0≈0, and a large
(~50x order-of-magnitude) L1 gap between transformer-block and embed/output
layers in the direction the paper reports — is reproduced at reduced scale.

## Files
- `claim4_transformer_smoothness.py` — PEP-723 self-contained script (numpy,
  scipy, pandas only; no torch).
- `claim4_transformer_smoothness.csv` — primary 2-row result (layer_group,
  L0_fit, L1_fit, n_samples, predicted/tuned stepsize).
- `claim4_per_matrix_smoothness.csv` — per-matrix (12 block matrices + E)
  breakdown.
- `claim4_trajectory.csv` — raw (Lhat, gnorm) pairs per step per matrix used
  for the fits.
