# Finite-Width Neural Tangent Kernels from Feynman Diagrams — reproduction briefing

Paper: arXiv 2508.11522v4, "Finite-Width Neural Tangent Kernels from Feynman Diagrams",
Guillen, Misof, Gerken (Chalmers University of Technology).
OpenReview id: SOlPHMdSY3. Local copy: `paper-arxiv-2508.11522v4.pdf` (53 pp, all appendices A-K
present/readable), extracted text: `paper_text.txt` (4347 lines, `pdftotext -layout`).

OpenReview/arXiv cross-check: **OpenReview fully bot-walled** on both attempts
(`forum?id=SOlPHMdSY3` and `pdf?id=SOlPHMdSY3`), consistent with the confirmed hard constraint in
this environment for every prior reproduction. arXiv 2508.11522v4 is the sole source; no
divergence could be checked, but there is no reason to suspect one — this is a preprint-style
paper, not a camera-ready-revision-prone one.

**Claim-extraction discrepancy found and noted here (not a paper error):** the challenge's
pre-extracted `claims_anchored.json` cites Figure numbers that don't match the actual paper:
- Claim 4 says "Figure 2" for the ReLU finite-width-correction experiment — the real paper has
  this as **Figure 3** ("Finite-Width Corrections for ReLU", p.9). Figure 2 in the paper is
  actually the gradient-stability plot (Claim 5's subject).
- Claim 5 says "Figure 1" for the gradient-stability experiment — the real paper has this as
  **Figure 2** ("Gradient Stability", p.9). Figure 1 in the paper is the GeLU multi-input
  finite-width-kernel-correction figure (a different experiment, Section 6.1).
This reproduction verifies the claim **text** (which is accurate) against the **correct** figures,
not the mislabeled ones — noted explicitly so a reviewer isn't confused when the cited numbers in
this write-up don't match `claims_anchored.json`'s figure references.

Challenge: HF Space `ICML-2026-agent-repro/challenge`. Lands in
`nmaher2022/icml2026-reproductions` as `ntk-feynman-diagrams-SOlPHMdSY3/`.

## Working conventions for this reproduction
- Self-contained PEP-723 Python scripts (`uv run <script>.py`), no repo-wide venv dependency.
  Needs `numpy`, `scipy`, `jax`/`jaxlib` (CPU-only — no GPU in this environment, confirmed
  blocker from [[blocker-no-gpu-hf-jobs-402]]) or plain `torch` (CPU) for autodiff over small
  MLPs — pick whichever is already installed in `.venv` to avoid a slow fresh CPU-only JAX build;
  check before committing to one.
- **Smoketest before scale**: tiny widths (n~10-20), few hundred initializations, 2-3 layers,
  before scaling to the paper's widths/depths/sample counts. Check shapes, no NaNs, right order
  of magnitude, sign of the correction.
- All work happens in `ntk-feynman-diagrams-SOlPHMdSY3/`. Don't touch other folders.
- Verdict vocabulary: VERIFIED / TOY-VERIFIED / REFUTED / BLOCKED / INCONCLUSIVE. State scale run
  next to every verdict. Never round TOY-VERIFIED up to VERIFIED.
- Self-check before finishing: reread each claim's exact wording against the source PDF side by
  side with the numbers/plots actually produced.

## Claims in scope (verbatim from `claims_anchored.json`, cross-checked against the PDF)
1. "Feynman-diagram graphical rules are introduced that simplify layer-wise recursion relations
   for finite-width Neural Tangent Kernel (NTK) statistics at order 1/n" (Section 4, rules in
   Appendix D, Theorems 4.1-4.3 + proofs in Appendix E).
2. "A recursion relation for the first-order (1/n) finite-width correction to the NTK mean is
   derived using five Feynman diagrams containing quadratic and quartic vertices" (Section 5.1,
   Eq. 15; algebraic form Eq. 78, Appendix G).
3. "For scale-invariant activation functions (ReLU and LeakyReLU), the infinite-width NTK diagonal
   Theta(x,x) has no finite-width corrections" (Section 5.3, Theorem 5.2, proof Appendix I).
4. "Experiments on four-layer ReLU MLPs sampled over 5x10^6 initializations, with hidden-layer
   widths n>=20, confirm the finite-width kernel corrections predicted by the recursion relations"
   — **actual paper Figure 3** (Section 6.3, p.9), not Figure 2 as the extracted claim states.
5. "Gradient-stability experiments on 200-hidden-unit ReLU MLPs with up to 30 layers, averaged
   over 1000 initializations, show linear scaling with depth at the critical initialization
   C_W=2" — **actual paper Figure 2** (Section 6.2, p.9), not Figure 1 as the extracted claim
   states.

## Core math / setup (transcribed from the PDF)

**Architecture / parametrization** (Section 3, Appendix B, Eq. 20):
`z_i^(l)(x) = (1/sqrt(n_{l-1})) * sum_j W_ij^(l) * sigma(z_j^(l-1)(x))`, `W_ij^(l) ~ N(0, C_W^(l))`
i.i.d., learning rate not scaled with width (standard NTK parametrization). No biases used in the
gradient-stability experiments (Section 6.2).

- Empirical NTK: `Theta_hat_ij^(l)(x,x') = sum_mu dz_i^(l)(x)/dtheta_mu * dz_j^(l)(x')/dtheta_mu`
  (Eq. 1). Empirical NNGP: `K_hat_ij^(l)(x,x') = z_i^(l)(x) z_j^(l)(x')` (Eq. 2).
- Finite-width expansions: `E[K_hat^(l)] = K^(l) + K^{1}(l)/n_{l-1} + O(1/n^2)`;
  `E[Theta_hat^(l)] = Theta^(l) + Theta^{1}(l)/n_{l-1} + O(1/n^2)` (Eqs. 21-22).
- Rank-4 tensors decomposing cumulants: V (4-pt preactivation cumulant), A/B (NTK-fluctuation
  cumulants), D/F (joint preactivation-NTK cumulants) — the building blocks of the quartic
  Feynman vertices (Eqs. 23-24, Appendix C).

**Section 4 — Feynman rules** (full list Appendix D): external vertices (solid = preactivation z,
dotted = NTK fluctuation ΔΘ̂), cubic vertices connecting two external lines to one internal line
(weight `1/n_l` or `C_W^(l+1)/n_l`), propagator = Gaussian expectation `<.>_{K^(l)}` with
covariance = NNGP Gram matrix, quartic vertices = the 10 rank-4 tensors as 4-line vertices (weight
`1/n_l`). Theorem 4.1: rules (i)-(v) uniquely determine the D/F/A/B layer recursions at order 1/n.
Theorem 4.2 extends to dNTK/ddNTK-derived tensors. Theorem 4.3: full rule set is complete at *all*
orders in 1/n, not just leading order.

**Section 5.1 — NTK mean recursion, Eq. 15 / algebraic form Eq. 78 (Appendix G):**
`Theta_12^{1}(l+1) / n_l` = sum of five diagram terms:
1. `(C_W^{(l+1)}/n_{l-1}) * Theta_12^{1}(l) * <sigma'_1^(l) sigma'_2^(l)>_{K^(l)}`
2. `(1/(2 n_{l-1})) * sum_{b1,b2} K_{b1 b2}^{1}(l) * <d^2(DeltaOmega_d,12)/dz_b1 dz_b2>_{K^(l)}`
3. `(1/(8 n_{l-1})) * sum_{b1..b4} V_{(b1 b2)(b3 b4)}^(l) * <d^4(DeltaOmega_d,12)/dz_b1..dz_b4>_{K^(l)}`
4. `(C_W^{(l+1)}/(2 n_{l-1})) * sum_{b1,b2} <d^2(sigma'_1 sigma'_2)/dz_b1 dz_b2>_{K^(l)} * D_{b1 b2 1 2}^(l)`
5. `(C_W^{(l+1)}/n_{l-1}) * sum_{b1,b2} <d^2(sigma'_1 sigma'_2)/dz_b1 dz_b2>_{K^(l)} * F_{b1 1 b2 2}^(l)`
(`DeltaOmega_d,12` is a paper-defined operator tied to the dNTK-derivative object — re-derive its
exact definition from Appendix C/D via `grep -n "Omega" paper_text.txt` during implementation if
needed; not reproduced here to avoid transcribing it wrong from OCR'd math.)

**Section 5.2 — Gradient stability (Theorem 5.1, proof App. H):** criticality operators
`(chi_perp)_{ab} = C_W^{(l+1)} <sigma'_a sigma'_b>_{K^(l)} = 1`,
`(chi_par)_{ab} = C_W^{(l+1)} <sigma''_a sigma_b>_{K^(l)} = 0` at criticality (Eq. 79). Theorem 5.1:
if the NNGP is critical, every cumulant involving the NTK is also critical (no exponential
blowup/decay with depth). For bias-free critical ReLU MLP, single-input NTK diagonal scales
*linearly* with depth: `Theta_aa^(l) = (1/n_0) ||x_a||^2 * l` (Eq. 129) — this is the "Theta = 0.7*l"
reference line in the real Figure 2 (critical C_W panel uses `||x||^2/n_0 = 0.7`).

**Section 5.3 — Theorem 5.2 (proof App. I):** scale-invariant activations
`sigma(z) = a_+ z (z>=0), a_- z (z<0)` (Eq. 99, ReLU: a+=1,a-=0; LeakyReLU: a+=1,a-=alpha) have
exact closed-form Gaussian expectations `<sigma(z)sigma(z)>_K = A*K` with `A=(a+^2+a-^2)/2`
(Eq. 100), `<sigma'(z)sigma'(z)>_K = A_tilde` = constant, independent of K (Eq. 101). These
closed forms are what force the diagonal NTK mean's 1/n correction to vanish exactly for these
activations — the mechanism to reproduce numerically is: simulate finite-width ReLU/LeakyReLU
networks at varying widths, show the *diagonal* empirical NTK mean is flat in 1/n (no correction)
while the *off-diagonal* (distinct-input) one is not.

**Section 6 — experimental setup for the claims actually in scope here:**
- **6.2 / real Figure 2 (Claim 5):** ReLU MLP, no biases, hidden layer width 200, depth up to 30
  layers, three `C_W` values `{0.25, 2.00, 4.00}` (critical `C_W=2` shows linear scaling
  `Theta=0.7*l`; off-critical values show exponential growth/decay), 1000 initializations, both
  single-input and distinct-input cases.
- **6.3 / real Figure 3 (Claim 4):** 4-layer ReLU MLP, `C_W=2`, 5e6 initializations, widths
  `n` swept from ~10 to 1000; relative deviation `|(Theta_bar_ab - Theta_ab)/Theta_ab|` for
  single-input `(0,0)` [flat/zero, confirms Theorem 5.2] and distinct-input `(0,1)` [shrinks
  toward 0 as `n -> infinity`, i.e. as `O(1/n)`]. Paper also checks LeakyReLU (same exactness for
  single-input) and GeLU (which is *not* scale-invariant — corrections present in both cases,
  Figure 25) as controls; useful as a negative-control activation for the toy run.
- Compute used by the paper: NVIDIA A40 GPUs, up to 6h/GPU for full-scale runs. Not available
  here (CPU-only, [[blocker-no-gpu-hf-jobs-402]]) — toy scale will use far smaller widths, depths,
  and initialization counts (documented per-claim in `REPRO_LOG.md`/`BUGFIX_LOG.md`), verdicts
  capped at TOY-VERIFIED accordingly for anything that would need the paper's full sample counts
  to resolve a small effect size.

## Reproduction scope / plan
- **Claims 3, 4, 5** are direct empirical claims about simulated finite-width MLPs (Section 6) —
  fully reproducible by simulating small MLPs and computing the empirical NTK/NNGP via autodiff,
  no need to implement the diagrammatic recursion solver itself. This is the primary, most
  tractable target for VERIFIED/TOY-VERIFIED verdicts with real numbers.
- **Claims 1, 2** describe the diagrammatic derivation of Eq. 15/78 itself (a symbolic/analytic
  result, not directly an "experiment" in the paper). Plan: test Eq. 78 empirically by measuring
  the layer-l cumulants (D, F, V, K^{1}) it needs as inputs directly via Monte Carlo at a given
  layer of a small simulated network, using Eq. 78 to *predict* Theta^{1}(l+1), then independently
  measuring Theta^{1}(l+1) via a two-width MC extrapolation of the same network's empirical NTK
  and comparing predicted vs. measured. Use a non-scale-invariant activation (tanh or GeLU) so the
  predicted correction is nonzero (a meaningful test, not the degenerate ReLU zero-case already
  covered by Claim 3). If the exact `DeltaOmega_d` operator definition can't be pinned down
  precisely enough from the OCR'd appendix text to implement Eq. 78 with confidence, downgrade
  Claim 2 to INCONCLUSIVE (not REFUTED) and say exactly what was missing — do not guess at a
  physics-paper's operator definition and silently risk a wrong implementation being reported as
  a verified/refuted result. Claim 1 (that the *rules* are sound, more so than one specific
  recursion output) rides on Claim 2's result plus a structural check of Theorem 4.1-4.3's proof
  sketches against the stated rules (i)-(v); note this dependency explicitly in the verdict.

## Known access blockers
- OpenReview: bot-walled, arXiv used instead (see cross-check note above). Not a claim blocker,
  just a Step-0 source note.
- No GPU: paper's own experiments used A40s; this reproduction runs CPU-only at reduced
  widths/depths/sample counts. Affects scale ceiling, not feasibility — MLPs this small run fine
  on CPU, just capped below the paper's own scale (verdicts marked TOY-VERIFIED, not VERIFIED,
  where that matters for a small-effect-size claim).
