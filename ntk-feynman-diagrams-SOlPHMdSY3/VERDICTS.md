# Verdicts — Finite-Width Neural Tangent Kernels from Feynman Diagrams (SOlPHMdSY3)

Source: `paper-arxiv-2508.11522v4.pdf` (OpenReview bot-walled both attempts, arXiv used per
Step 0; see PAPER_BRIEFING.md for the acquisition note and the Figure-numbering discrepancy found
in the challenge's `claims_anchored.json` — Claims 4 and 5 below are checked against the paper's
*actual* Figure 3 and Figure 2 respectively, not the "Figure 2"/"Figure 1" the extracted claim
text cites). All runs CPU-only (no GPU in this environment), widths/depths/sample counts well
below the paper's own scale throughout — every verdict below is capped at TOY-VERIFIED, never
rounded up. Raw results in `results/*.json`, run/bugfix history in `BUGFIX_LOG.md`.

## Claim 3 — TOY-VERIFIED

> "For scale-invariant activation functions (ReLU and LeakyReLU), the infinite-width NTK diagonal
> Theta(x,x) has no finite-width corrections" (Section 5.3, Theorem 5.2).

Ran `repro_toy.py width-sweep` for ReLU (widths 10/20/40/80/160/320, depth 4, `C_W=2`,
1500 inits/width — `results/width_sweep_relu_toy.json`) and LeakyReLU (widths
10/20/40/80/160, same depth/C_W/inits — `results/width_sweep_leakyrelu_toy.json`).

ReLU diagonal mean NTK by width: 1.358, 1.352, 1.382, 1.383, 1.392, 1.390 — flat within MC noise
(relative deviation from the analytic infinite-width value 1.380 stays in 0.2-2% at *every* width,
with no downward trend as width grows from 10 to 320; pure sampling noise at 1500 inits is ~2.6%,
consistent with this). LeakyReLU diagonal: 1.405, 1.390, 1.429, 1.431, 1.433 — same flat pattern.
Contrast with the same activations' *off-diagonal* NTK (distinct inputs), which is not
scale-invariant and does shrink toward the infinite-width value as width grows (0.300 -> 0.330
for ReLU, relative deviation 9.1% at width 10 down to 0.0% at width 320) — the diagonal-vs-
off-diagonal asymmetry is exactly the claim, and both activations show it cleanly at toy scale.

## Claim 4 (paper's real Figure 3, not "Figure 2") — TOY-VERIFIED

> "Experiments on four-layer ReLU MLPs ... confirm the finite-width kernel corrections predicted
> by the recursion relations" (Section 6.3).

Same `results/width_sweep_relu_toy.json` run as Claim 3 (paper used 5e6 inits/width across a
wider range of widths >=20; here 1500 inits, widths 10-320 — a toy scale, ~3000x fewer samples).
Single-input (diagonal) relative deviation stays flat/near-zero at every width tested (see Claim
3), matching the paper's prediction (Theorem 5.2: no correction for ReLU on the diagonal).
Distinct-input (off-diagonal) relative deviation shrinks monotonically in trend from 9.1% (n=10)
to 0.0% (n=320) as width grows, i.e. approaches the infinite-width value as `n -> infinity`,
matching the qualitative claim. Not claiming to match the paper's own numeric deviation curve at
its own sample count/width range — this is a toy-scale qualitative confirmation, not an exact
number match, hence TOY-VERIFIED not VERIFIED.

## Claim 5 (paper's real Figure 2, not "Figure 1") — TOY-VERIFIED (strong)

> "Gradient-stability experiments on ... ReLU MLPs with up to 30 layers ... show linear scaling
> with depth at the critical initialization C_W=2" (Section 6.2).

Ran `repro_toy.py depth-sweep`, bias-free ReLU MLP, width 50 (paper: 200), depth up to 15 (paper:
30), 500 inits (paper: 1000), `C_W in {0.25, 2.0, 4.0}` (matches paper) —
`results/depth_sweep_toy.json`.

- **C_W=2.0 (critical)**: NTK diagonal grows from 0.345 (layer 1) to 5.16 (layer 15). Linear fit
  (least squares, layer index vs. mean): slope 0.348, R^2 = 0.9986 — a near-perfect linear fit.
  Hand-derived analytic prediction (see `BUGFIX_LOG.md` for the derivation) for this exact input/
  `C_W`: `Theta^(l) = 0.345 * l` — the fitted slope (0.348) matches the analytic slope (0.345)
  to within 1%.
- **C_W=0.25 (sub-critical)**: decays from 0.345 (layer 1) to 1.2e-12 (layer 15). Log-linear
  (exponential) fit: R^2 = 0.999, decay rate -1.91/layer.
- **C_W=4.0 (super-critical)**: grows from 0.345 (layer 1) to 8.45e4 (layer 15). Log-linear fit:
  R^2 = 0.995, growth rate +0.86/layer.

The three-way contrast (near-perfect linear fit only at `C_W=2`, near-perfect exponential fits at
the other two) is exactly the paper's claimed qualitative picture, and the critical-case slope
matches an independently-derived analytic prediction to ~1%. Capped at TOY-VERIFIED purely because
of the reduced width/depth/init-count vs. the paper's own scale, not because of any weakness in
the fit quality.

## Claims 1 & 2 — INCONCLUSIVE (partial, with an honest scope limitation)

> Claim 1: "Feynman-diagram graphical rules are introduced that simplify layer-wise recursion
> relations for finite-width NTK statistics at order 1/n" (Section 4).
> Claim 2: "A recursion relation for the first-order (1/n) finite-width correction to the NTK
> mean is derived using five Feynman diagrams containing quadratic and quartic vertices"
> (Section 5.1, Eq. 15 / algebraic form Eq. 78, Appendix G).

**What was checked**: the paper's own Eq. 78 (Appendix G) was read directly from the extracted
PDF text and transcribed into PAPER_BRIEFING.md — five additive terms, two from quadratic
vertices (`Theta^{1}` and `K^{1}` propagation) and three from quartic vertices (`V`, `D`, `F`
tensors), matching the claim's description exactly in structure (2 quadratic + 3 quartic = 5
diagrams). This structural/textual match is confirmed.

**What was not independently re-derived**: the exact numeric prediction of Eq. 78 was not
implemented from scratch. One of its terms depends on an operator the paper writes as (OCR'd)
`\Delta\Omega_d`, tied to the paper's "dNTK" formalism (a third-derivative-type object introduced
via Roberts et al. 2022's deep learning theory framework, Appendix C) — a genuinely advanced
quantity, and the PDF's math extraction is visibly unreliable here (e.g. the "Omega" glyph itself
extracts as U+2126 OHM SIGN, a font-substitution artifact, a sign that other subscripts/exponents
in that block may not be trustworthy either). Implementing this from a possibly-corrupted
transcription risked silently producing a wrong recursion and reporting a false VERIFIED or
REFUTED verdict with unearned confidence — exactly the failure mode `verdict_checklist.md` warns
about. This was flagged as a possible outcome in PAPER_BRIEFING.md's scope section before running
any experiments, not decided post-hoc after a result looked inconvenient.

**Indirect qualitative support that does exist**: Claims 3/4/5 above independently confirm the
*outputs* the diagrammatic recursion is supposed to produce — a genuine, correctly-signed O(1/n)
correction that vanishes exactly for scale-invariant activations (Claim 3/4) and produces the
correct linear/exponential depth-scaling for the NTK mean at/away from criticality (Claim 5,
itself derived in this reproduction from a from-scratch, independently-checked NTK recursion
matching to <1%, not from the paper's diagram formalism). This is consistent with Eq. 78 being
correct, but is not the same as directly verifying Eq. 78's specific five-term algebraic content —
a network could produce the same qualitative Claims 3/4/5 behavior without validating that this
particular 5-diagram decomposition (as opposed to some other correct decomposition) is what
produces it.

**Verdict**: INCONCLUSIVE, not VERIFIED and not REFUTED. What would resolve it: either the actual
OpenReview PDF (cleaner math rendering, no OCR font substitution) to pin down `\Delta\Omega_d`'s
precise definition with confidence, or a from-scratch derivation of the dNTK formalism from
Roberts et al. (2022) independent of this paper's own (possibly OCR-garbled) notation, followed by
implementing and numerically solving Eq. 78 directly (as the paper itself does in Section 6.1 via
a custom SymPy + numerical-integration pipeline, Appendix J) rather than testing only its
downstream empirical consequences.

## Summary

| Claim | Verdict | Scale |
|---|---|---|
| 1 (Feynman rules exist/simplify recursions) | INCONCLUSIVE | structural match confirmed; full numeric re-derivation out of scope (OCR risk) |
| 2 (5-diagram NTK-mean recursion, Eq. 15/78) | INCONCLUSIVE | same as above |
| 3 (no finite-width correction, scale-invariant diag) | TOY-VERIFIED | widths 10-320, depth 4, 1500 inits (paper: widths incl. much larger, 5e6 inits) |
| 4 (ReLU kernel-correction experiment, real Fig. 3) | TOY-VERIFIED | same as Claim 3 |
| 5 (gradient stability, linear scaling, real Fig. 2) | TOY-VERIFIED | width 50, depth 15, 500 inits (paper: width 200, depth 30, 1000 inits) |

No claim is BLOCKED — all five were attempted at a fair (if reduced) scale; Claims 1/2's
INCONCLUSIVE status reflects a genuine scope limit (unreliable OCR of one specific paper-internal
operator), not a skipped or avoided claim.
