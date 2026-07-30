# Claim 5 — CNN/CIFAR-10 layer-wise smoothness (TOY reproduction)

## Claim text (verbatim, from `claims_anchored.json`)

> "On a CNN trained on CIFAR-10, estimated smoothness constants also satisfy
> L^0_i approximately 0, with a two-orders-of-magnitude spread in L^1_i
> across layers, motivating per-layer learning-rate heterogeneity."

Paper's real setup (Appendix E.3/E.4): full CNN on full CIFAR-10, full-batch
gradients, no momentum, no LR decay, ~80 epochs, 1xA100. Paper reports
`L_i^0 ~= 0` everywhere; conv/norm/bias layers `L_i^1 ~= 3`; classification
head `L_p^1 ~= 0.03` (~2-orders-of-magnitude gap).

## Why this can only be a toy check

This repo has no torch/GPU, so real CIFAR-10 + a production CNN is
infeasible. I hand-built a tiny CNN from scratch in numpy (manual
forward/backward, no autodiff) trained on a small synthetic image
classification task, and applied the exact unScion(CNN) layer-wise LMO
update from the briefing. This checks only the *qualitative pattern*, not
the exact constants.

## What was built (`claim5_cnn_smoothness.py`)

- Hand-implemented conv layers (`sliding_window_view` + `einsum` for forward
  and backward, valid/stride-1 convolution), ReLU, and a linear
  classification head, all with manually-derived gradients (no autodiff).
- unScion(CNN) update rules exactly as given in the briefing: biases via
  normalized-GD (`t*sqrt(C_out)*g/||g||_2`), conv kernels via spectral/SVD
  LMO on the reshaped `C_out x (C_in*k*k)` matrix (`t*(1/k^2)*sqrt(C_out/C_in)*UV^T`),
  head via entrywise sign update (`(t/n_p)*sign(g)`).
- Per-layer dual (gradient) / primal (delta) norms used for Eq. 10 are the
  ones *consistent* with each layer's LMO norm (derived by hand via standard
  norm-scaling/duality, the same derivation validated in claim 1's writeup):
  Euclidean self-dual for biases; nuclear-norm/spectral-norm pair scaled by
  `(1/k^2)*sqrt(C_out/C_in)` for conv; entrywise-L1/entrywise-Linf pair
  scaled by `1/n_p` and `n_p` for the head.
- **Correctness check**: numerical (finite-difference) gradient check on all
  5 parameter groups before scaling up — max relative error 4.2e-10.
- **Smoketest** (K=20 steps, tiny dims) passed first: no NaNs, gradients
  flowing (all nonzero), loss decreasing (1.08 -> 0.81), before any
  longer run.
- Eq. 10: full-batch + deterministic, so `grad_i f_{xi^{k+1}}(X^{k+1})` is
  just `grad_i f(X^{k+1})` — no stochasticity needed, consecutive-iterate
  differencing only, computed with 1 forward/backward per step (grad cached
  and reused between consecutive steps).
- Eq. 30: hinge-penalized least-squares fit (`lambda=5`,
  `scipy.optimize.minimize`, Nelder-Mead, `L0,L1 >= 0`), per layer group.

### Two attempts were run (both reported honestly below, not cherry-picked)

**Attempt 1** (small architecture, easy synthetic task, no label noise):
8x8x2 images shrunk to 10x10x2, `Cout1=Cout2=4`, `k=3`, `n_feat=144`
(head input dim), 3 classes, 180 samples, `K=1500-2500` full-batch unScion
steps, constant LR, no decay/momentum. **Problem found**: the toy task is
easy enough that the network reaches ~100% train accuracy / near-zero loss
within a few hundred steps. Past that point gradients collapse toward
float-precision noise, and the SVD/sign-based normalized updates keep
taking fixed-size steps in directions no longer tied to real curvature —
polluting the Eq. 10 trajectory. A relative-gradient-norm floor filter
(skip steps where either endpoint's gradient dual-norm is `< 1e-3 x` the
initial gradient scale for that layer) was added to exclude this saturated
regime. Result: `L0` not small (0.05-0.93 across layers), and the
head/non-head `L1` ratio was ~0.84-1x — essentially no gap, let alone 2
orders of magnitude.

**Attempt 2** (larger architecture + harder task, final numbers reported in
the CSV): 16x16x2 images, `Cout1=4, Cout2=8, k=3` (`n_feat=1152`, ~3800
total params — larger head fan-in on the theory that the paper's real CNN
head, fed by many flattened conv features, is exactly where the `1/n_p`
scaling in the head's norm definition should matter), 3 classes, 240
samples, **15% label noise** added (fixed/deterministic flip of a subset of
labels, seeded — creates an irreducible Bayes-error floor so full-batch
training can't drive gradients to literal float-precision zero within the
step budget, keeping the trajectory informative throughout), `K=3000`
full-batch unScion(CNN) steps, same gradient-floor filter as attempt 1.
Result below.

## Post-hoc correction: bias-layer norm bug found on independent audit

An independent code review (after this RESULTS file was first drafted) found
that `bias_dual`/`bias_primal` were missing the `sqrt(C_out)` / `1/sqrt(C_out)`
scaling factor that `update_bias` (`t*sqrt(C_out)*g/||g||_2`) implies for the
bias layers' own norm (`||.||_(i) = (1/sqrt(C_out))*||.||_2`, by the same
norm-scaling duality already verified 520/520 times in claim 1's audit — see
`claim1_special_cases.py`'s `unscion_cnn_bias` case, which uses the correct
`c=1/sqrt(C_out)`). `conv_dual/conv_primal` and `head_dual/head_primal`
already included their scaling constants correctly; the bug was isolated to
the two `bias_*` functions. Fixed in `claim5_cnn_smoothness.py` (now takes
`Cout` as an explicit argument) and the full run (gradcheck + smoketest +
K=3000 main run) was repeated end-to-end. The correction **does not change
the verdict** — if anything it strengthens the refutation (see below).

## Results (`claim5_cnn_smoothness.csv`, attempt 2 — final, post-fix)

| layer_group | L0_fit | L1_fit | n_points (post-filter) |
|---|---|---|---|
| bias1 | 12.904 | 6.890 | 998 |
| conv1 | 2.409 | 5.817 | 980 |
| bias2 | 14.704 | 7.188 | 964 |
| conv2 | 1.938 | 5.999 | 990 |
| head  | 0.128 | 7.724 | 995 |

(~2000/3000 steps per layer were filtered out as saturated even with label
noise and the larger head — the toy task remains much easier to fit than
real CIFAR-10.)

Smallest non-head `L1` / head `L1` = 5.817 / 7.724 = 0.75 (**head's L1 is
still larger than every other layer's L1**, not ~100x smaller — and the
corrected spread across all 5 layers is now an even *tighter* band, 5.82 to
7.72, than the pre-fix numbers suggested). `L0` now ranges 0.13-14.7, with
the bias layers' `L0` (12.9, 14.7) now clearly **not** small relative to
their own `L1` (6.9, 7.2) — the "approximately 0" part of the claim is even
more clearly violated post-fix than it appeared pre-fix.

Full per-step trajectory: `claim5_cnn_smoothness_traj.csv`.

## Self-check against the claim text

The claim requires two things: (1) `L0_i ~= 0` for all layer groups, and (2)
a ~2-orders-of-magnitude gap with the head's `L1` far *below* the
conv/bias/norm layers' `L1`. Neither held in this toy reproduction:

- `L0` was not close to 0 in either attempt (attempt 1: 0.05-0.93; attempt
  2 post-fix: 0.13-14.7, exceeding `L1` itself for both bias layers) — this
  contradicts the claim's first part outright at this scale.
- The head/non-head gap either didn't appear (attempt 1: ratio ~0.84-1x, no
  meaningful gap) or appeared in the **opposite direction** (attempt 2: head
  `L1`=7.72 was the *largest* of all 5 groups, not ~100x smaller).

For comparison, claim 4's structurally analogous check (a hand-built toy
transformer, same methodology: a group of spectral-LMO matrices vs. one
sign-LMO tied embedding/output matrix) **did** reproduce the paper's
qualitative pattern cleanly (L0~=0 for blocks, ~42x L1 gap in the right
direction — see `RESULTS_claim4.md`). That the same norm-derivation and
hinge-fit pipeline worked cleanly for the transformer case but not for the
CNN case is evidence the failure here is not a bug in the fitting
methodology itself, but a genuine mismatch between this toy CNN/task and
whatever property of the real, deep, CIFAR-10-scale CNN produces the head's
tiny `L1`.

## Verdict: **REFUTED** (at toy scale — does not imply the paper's real
claim is wrong, only that this reduced-scale reproduction does not confirm
it)

Two independent, principled attempts (including one specifically designed
around the mechanism that plausibly drives the paper's result — a large
head fan-in, since the head's norm scales its dual by `1/n_p`) failed to
reproduce the claimed pattern. In the second, larger attempt the direction
even inverted (head `L1` largest, not smallest). Per the task instructions,
this is reported honestly as a discrepancy rather than continuing to search
for hyperparameters that would force the "right" answer.

### Plausible explanations for the discrepancy (not verified, offered as hypotheses)
- **Depth/capacity mismatch**: the toy CNN has only 2 conv layers and a few
  thousand parameters total; the paper's real CNN is presumably deeper and
  wider, with batch/layer normalization layers of its own ("conv/norm/bias
  layers" in the paper's own claim wording implies normalization layers
  exist that aren't in this toy model at all). The head's small `L1` may be
  an emergent property of a deep feature hierarchy feeding it, not
  reproducible with a 2-layer conv stack regardless of head fan-in.
- **Task/data mismatch**: real CIFAR-10 (50k natural images, 10 classes) has
  a much richer, noisier loss landscape than this small (240-sample,
  3-class, template+Gaussian-noise) synthetic task; ~80 real epochs of
  training dynamics may explore a qualitatively different trajectory regime
  than 3000 full-batch toy steps that saturate almost immediately even with
  label noise added.
- **Numerical/statistical noise at toy scale**: with the saturation filter
  removing ~2/3 of steps, each layer's Eq. 30 fit uses under 1000 points
  from a fairly narrow, oscillatory post-transient regime (the label-noise
  floor prevents full saturation but also limits how much the gradient norm
  varies across the retained trajectory) — this is a much less-conditioned
  regression than the real paper's presumably longer, richer training
  trajectory, and than claim 4's 700-step transformer run which used nearly
  all its steps (no aggressive saturation filtering needed there).

## Files
- `claim5_cnn_smoothness.py` — PEP-723 self-contained script (numpy, scipy
  only; no torch). Includes gradcheck, smoketest, both architecture configs
  (attempt 1 accessible via `main_run()` defaults; attempt 2 is what
  `__main__` currently runs), Eq. 10 trajectory logging, and Eq. 30 fit.
- `claim5_cnn_smoothness.csv` — final (attempt 2) per-layer `L0_fit`,
  `L1_fit`, `n_points`.
- `claim5_cnn_smoothness_traj.csv` — full per-step `(layer_group, step,
  Lhat, grad_dual_norm_next)` trajectory used for the fits.
