# Gluon: Making Muon & Scion Great Again! — reproduction briefing

Paper: arXiv:2505.13416v1, "Gluon: Making Muon & Scion Great Again! (Bridging Theory and Practice
of LMO-based Optimizers for LLMs)", Riabinin/Shulgin/Gruntkowska/Richtárik (KAUST).
OpenReview id (this challenge's key): IelAHU5MVz. Local copy of the PDF:
`/home/rec1/Desktop/AI_Safety/ICML_reproduce/2505.13416v1.pdf` (46 pages, main paper pp.1-15,
appendix pp.17-46).

Challenge: HF Space `ICML-2026-agent-repro/challenge` — reproduce claims from ICML 2026 papers.
This is the `icml2026-reproductions` monorepo (nmaher2022/icml2026-reproductions pattern, though
this repro isn't in that repo yet). **Repo conventions** (already used by every other folder in
this project, e.g. `gaussian-mechanism-82Wosp2Iu1/`, `submodular-dynamic-non-monotone-tBS3uBG6Pv/`):

- Every script is **self-contained PEP-723** with an inline `uv` dependency header, e.g.:
  ```python
  # /// script
  # requires-python = ">=3.10"
  # dependencies = ["numpy"]
  # ///
  ```
  Run with `uv run <script>.py` — no repo-wide venv, no torch anywhere in this repo. Stick to
  numpy/scipy/pandas/matplotlib/plotly, matching prior reproductions. If a claim genuinely needs a
  tiny neural net, implement forward/backward by hand in numpy (this repo has no torch and that's
  intentional — keeps every script trivially runnable via `uv run`).
- **Naming**: `claimN_shortdesc.py` producing `claimN_shortdesc.csv` with the numeric results. A
  `make_figs*.py` regenerates any plots from those CSVs.
- **CRITICAL — smoketest before scaling up**: before running anything longer than ~30 seconds,
  run a tiny/fast version first (few iterations, tiny dimensions) and check for shape errors, NaNs,
  sane orders of magnitude, sign errors, etc. Only scale up once the smoketest is clean. Do not
  burn CPU-minutes on a broken script at full scale.
- All work happens in `/home/rec1/Desktop/AI_Safety/ICML_reproduce/gluon-lmo-optimizers-IelAHU5MVz/`.
  Do not touch other folders in this repo.
- End your work by writing a short `RESULTS_<claimset>.md` in this folder: what you ran, the
  numbers you got, and an explicit verdict per claim — VERIFIED (matches paper qualitatively or
  quantitatively), TOY-VERIFIED (matches at reduced scale, not claiming to hit the paper's exact
  numbers), or REFUTED (with the discrepancy). Be honest — a toy-scale "roughly consistent
  direction" is not the same as "verified", say which one you got.
- **Self-check before finishing**: reread the exact claim text below and your own numbers/plots
  side by side. Does your evidence actually support what the claim says, at the scale you ran it?
  If not, don't round up to VERIFIED.

## Core math (verbatim from the paper — Sections 2-4, Appendix C/D)

Setup: parameters `X = [X_1,...,X_p]` split into `p` layer groups, `X_i` a matrix (or vector) per
layer. `f` is the (possibly stochastic) loss. For each layer `i` there is a norm `||.||_(i)` with
dual norm `||.||_(i)*`.

**Algorithm 2 (deterministic Gluon)**, for k=0..K-1, for each layer i:
```
X_i^{k+1} = LMO_{B_i^k}(grad_i f(X^k)) := argmin_{X_i in B_i^k} <grad_i f(X^k), X_i>_(i)
where B_i^k := {X_i : ||X_i - X_i^k||_(i) <= t_i^k}
```
This LMO over a norm ball has the closed form `X_i^{k+1} = X_i^k - t_i^k * g_i / ||g_i||_(i)*` when
`||.||_(i)` is Euclidean (normalized GD); `X_i^k - t_i^k sign(g_i)` for max-norm (signGD); and for
spectral norm, `X_i^{k+1} = X_i^k - t_i^k U V^T` where `g_i = U Σ V^T` is the (reduced) SVD of the
gradient (this is Muon's update).

**Algorithm 1 (stochastic Gluon with momentum)**: same LMO step but on momentum `M_i^k = β^k
M_i^{k-1} + (1-β^k) grad_i f_ξ(X^k)` instead of the raw gradient.

**Special cases (Section 4.1 / Appendix C.1), all via the *same* Algorithm 2/13 formula, only the
norm choice `||.||_(i)` per layer changes**:
- **Muon**: `||.||_(i) = ||.||_{2→2}` (spectral norm) for all hidden layers →
  `X_i^{k+1} = X_i^k - t_i^k U_i^k (V_i^k)^T` where `grad_i f(X^k) = U_i^k Σ_i^k (V_i^k)^T` (SVD).
- **unScion (LLM training)**: for i=1..p-1 (transformer block layers), `||.||_(i) =
  sqrt(n_i/m_i) * ||.||_{2→2}` where `X_i in R^{m_i x n_i}`; for the last group `X_p` (embedding
  and output layers, tied under weight sharing), `||.||_(p) = n_p * ||.||_{1→∞}`. Update:
  ```
  X_i^{k+1} = X_i^k - t_i^k * sqrt(m_i/n_i) * U_i^k (V_i^k)^T,   i=1..p-1
  X_p^{k+1} = X_p^k - (t_p^k/n_p) * sign(grad_p f(X^k))
  ```
- **unScion (CNN training)** — biases (1D, shape `C_out`), conv kernels (4D, reshaped to
  `C_out x (C_in*k*k)`), head weights (last group, sign update):
  ```
  X_i^{k+1} = X_i^k - t_i^k * sqrt(C_i^out) * grad_i f(X^k) / ||grad_i f(X^k)||_2     (biases)
  X_i^{k+1} = X_i^k - t_i^k * (1/k^2) * sqrt(C_i^out/C_i^in) * U_i^k (V_i^k)^T        (conv, via SVD of the reshaped 2D matrix)
  X_p^{k+1} = X_p^k - (t_p^k/n_p) * sign(grad_p f(X^k))                              (head)
  ```
- **Layer-wise normalized GD**: `||.||_(i) = ||.||_2` (Euclidean, vector case n_i=1) →
  `X_i^{k+1} = X_i^k - t_i^k * grad_i f(X^k) / ||grad_i f(X^k)||_2`.
- **Layer-wise signGD**: `||.||_(i) = ||.||_∞` (n_i=1) →
  `X_i^{k+1} = X_i^k - t_i^k * sign(grad_i f(X^k))`.

**Assumption 1 (layer-wise (L^0,L^1)-smoothness)**:
`||grad_i f(X) - grad_i f(Y)||_(i)* <= (L_i^0 + L_i^1 ||grad_i f(X)||_(i)*) ||X_i - Y_i||_(i)`
for all i, X, Y.

**Theorem 1 (deterministic, Section 4.2/Appendix C.2)**: run Algorithm 2 with adaptive stepsize
`t_i^k = ||grad_i f(X^k)||_(i)* / (L_i^0 + L_i^1 ||grad_i f(X^k)||_(i)*)`. Then to reach
`min_{k<K} sum_i [(1/L_i^1) / (mean_j 1/L_j^1)] ||grad_i f(X^k)||_(i)* <= eps` it suffices to run
```
K = ceil( 2*Delta0*sum_i(L_i^0/(L_i^1)^2) / (eps^2 * (mean_j 1/L_j^1)^2)
          + 2*Delta0 / (eps * mean_j 1/L_j^1) )
```
where `Delta0 = f(X^0) - inf f`. When `L_i^0 ≈ 0` for all i (empirically observed throughout this
paper), the first term vanishes and this is effectively `K = O(1/eps)` with only the *harmonic
mean* of `L_i^1` in the constant (not the worst-case max) — this is the O(1/sqrt(K)) rate claim
once translated to "gradient norm <= eps after K steps" form (the eps^-1 iteration count <=>
eps ~ K^{-1/2} rate). A simpler, weaker version (Theorem 3, part 1) with the classic max-based
bound: `K = ceil(2*Delta0*sum_i L_i^0/eps^2 + 2*Delta0*L^1_max/eps)`, i.e. `sum_i
||grad_i f(X^k)||_(i)* = O(1/sqrt(K))` in the L^0=0 case.

**Theorem 2 (stochastic, Section 4.3/Appendix D.2)**: run Algorithm 1 with `β^k = 1-(k+1)^{-1/2}`,
`t_i^k = t_i * (k+1)^{-3/4}` for a constant `t_i > 0`, `M_i^0 = grad_i f_{ξ^0}(X^0)`. Then
```
min_{k<K} sum_i (1/(12 L_i^1)) E[||grad_i f(X^k)||_(i)*]
  <~ Delta0/K^{1/4} + (1/K^{1/4}) * sum_i [ sigma/L_i^1 + L_i^0/(L_i^1)^2 ]
```
(hides constants/log factors). I.e. the (harmonic-mean-weighted) average gradient norm decays as
`O(K^{-1/4})`. This assumes unbiased stochastic gradients with bounded variance `sigma^2`
(Assumption 2).

**Eq. 10 / Eq. 30 — the empirical smoothness-fitting procedure used for claims 4 and 5**:
```
L_hat_i[k] := ||grad_i f_{ξ^{k+1}}(X^{k+1}) - grad_i f_{ξ^k}(X^k)||_(i)* / ||X_i^{k+1} - X_i^k||_(i)
L_hat_i^approx[k] := L_i^0 + L_i^1 * ||grad_i f_{ξ^{k+1}}(X^{k+1})||_(i)*
```
Fit `L_i^0, L_i^1 >= 0` per layer group by minimizing (hinge-penalized least squares, penalizing
underestimation more, λ>=0 controls the penalty strength):
```
loss(L0,L1) = sum_k (L_hat_i[k] - L_hat_i^approx[k])^2
              + λ * sum_k max(0, L_hat_i[k] - L_hat_i^approx[k])^2
```
Paper reports (Appendix E.3/E.4, NanoGPT-124M/FineWeb, unScion, 5000 iters, no LR decay, 4xA100):
`L_i^0 ≈ 0` everywhere; transformer-block layers `L_i^1 ≈ 67-71` (predicted stepsize
`1/L_i^1 ≈ 0.014` vs. tuned `0.018`); embedding/output layer `L_p^1 ≈ 1.3` (predicted `1/L_p^1 ≈
0.77` vs. tuned `1.08`). CNN/CIFAR-10 (full-batch, no momentum, no LR decay, ~80 epochs, 1xA100):
`L_i^0 ≈ 0` everywhere; conv/norm/bias layers `L_i^1 ≈ 3`; classification head `L_p^1 ≈ 0.03` (a
~2-orders-of-magnitude gap, motivating a much larger stepsize for the head).

## The 6 extracted claims (verbatim, from claims_anchored.json)

1. "The Gluon framework unifies layer-wise LMO-based optimizers, recovering Muon (spectral norm),
   unScion (mixed norms for transformer blocks and embeddings), layer-wise normalized GD, and
   layer-wise signGD as special cases via Algorithm 1 (Algorithm 1, Section on special cases)."
2. "Under a layer-wise generalized (L^0_i, L^1_i)-smoothness condition, ||grad_i f(X) - grad_i
   f(Y)||_* <= (L^0_i + L^1_i ||grad_i f(X)||_*) ||X_i - Y_i||, Gluon with adaptive per-layer
   stepsize t_i^k = ||grad_i f(X^k)||_* / (L^0_i + L^1_i ||grad_i f(X^k)||_*) achieves O(1/K^{1/2})
   convergence in the deterministic setting (Section 4.2)."
3. "In the stochastic setting with non-Euclidean bounded variance, Gluon achieves a convergence
   rate of O(Delta^0/K^{1/4} + 1/K^{1/4} * sum_i [sigma/L^1_i + L^0_i/(L^1_i)^2]) (Theorem 1)."
   [Note: mislabeled "Theorem 1" in the extraction — this is actually Theorem 2 in the paper.]
4. "Empirical estimation of layer-wise smoothness constants on a 124M-parameter NanoGPT model
   trained on FineWeb finds L^0_i approximately 0 across layers, with L^1_i approximately 70 for
   transformer blocks (predicted stepsize 0.014 vs. tuned 0.018) and L^1_i approximately 1.3 for
   embedding/output layers (predicted stepsize 0.77 vs. tuned 1.08)."
5. "On a CNN trained on CIFAR-10, estimated smoothness constants also satisfy L^0_i approximately
   0, with a two-orders-of-magnitude spread in L^1_i across layers, motivating per-layer
   learning-rate heterogeneity."
6. **REFUTED — do not attempt.** "The zeroth-order eNTK approximation error is shown to depend on
   the model's output dimension rather than the (potentially massive) parameter dimension, in
   contrast to prior analyses that scale with parameter count." This concept (eNTK / neural
   tangent kernel / zeroth-order approximation error) does not appear anywhere in the 46-page
   paper (full-text search confirmed zero real hits) nor in the ICML poster abstract. This is a
   claim-extraction/misattribution error in the challenge pipeline, unrelated to this paper's
   actual content. Already logged as refuted — no script needed for this one.

## Cross-check: arXiv v1 vs. OpenReview submitted PDF (2026-07-30)

The user independently downloaded the OpenReview submission PDF (`31097_From_Muon_to_Gluon_Bridg_
Originally Submitted PDF.pdf`, 43 pages, double-column anonymized ICML format with line numbers,
different SHA256 than arXiv). Cross-checked both copies:
- Abstract text matches near-verbatim (one wording diff: "new LMO-based method called Gluon" in
  arXiv vs. "new LMO-based framework called Gluon" in the submission — cosmetic).
- `eNTK`/`NTK`/`tangent kernel`/`zeroth-order`/`empirical NTK` all return **zero real hits in the
  OpenReview PDF too** — claim 6 is confirmed absent from both versions of the paper, not just the
  arXiv preprint. Strengthens the REFUTED verdict.
- Theorem numbering differs by scheme only: OpenReview uses section-scoped numbers ("Theorem
  4.1" = deterministic O(1/K^{1/2}) result, "Theorem 4.3" = stochastic O(1/K^{1/4}) result, the one
  used for claim 3); arXiv renumbers globally as Theorem 1/Theorem 2 respectively. Confirms the
  claims_anchored.json label "(Theorem 1)" attached to claim 3 is wrong in **both** numbering
  schemes — it is never actually "Theorem 1" anywhere; that number belongs to a different
  (deterministic) result. This is a pre-existing citation error in the challenge's claim
  extraction, not a reason to doubt the claim's mathematical content, which does match Eq. 9 /
  Theorem 4.3 exactly.
- No other structural differences found; treat arXiv v1 and the OpenReview submission as the same
  paper for verification purposes.
