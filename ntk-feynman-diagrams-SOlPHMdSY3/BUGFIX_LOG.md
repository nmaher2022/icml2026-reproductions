# Bugfix log — ntk-feynman-diagrams-SOlPHMdSY3

## Round 1 — analytic infinite-width NTK recursion off by a factor of C_W

**Found during**: smoketest of `repro_toy.py width-sweep` (Step 3), before scaling to
the real toy run.

**Symptom**: empirical (autograd, finite-width) mean NTK came out at almost exactly
half the hand-derived analytic infinite-width value, for both the diagonal and
off-diagonal entries, at `C_W=2` (diag: empirical 1.036 vs analytic-as-first-written
2.070; offdiag: empirical 0.186 vs analytic-as-first-written 0.377 — both ratios
~0.500 = 1/C_W).

**Root cause**: `relu_inf_width()`'s recursion used the textbook-looking formula
`Theta^(l+1) = Theta^(l) * Kdot^(l+1) + K^(l+1)` with base case `Theta^(1) = K^(1)`.
Re-deriving by hand (splitting the NTK's parameter-gradient sum into the top layer's
own-weight contribution vs. the part flowing through the shallower layer's weights)
shows the top layer's own contribution is `K^(l+1) / C_W^(l+1)` — an expectation over
*activations only* (`E[sigma(z) sigma(z')]`, no weight-variance factor) — not
`K^(l+1)` itself, since the weight-variance factor `C_W` only enters `K^(l+1)` via
the *definition* `K^(l+1) = C_W * E[sigma(z)sigma(z')]`, and the NTK's chain-rule term
computes the `E[sigma sigma']` piece directly without ever squaring/summing a W with
variance C_W in that particular contraction. Likewise the base case is
`Theta^(1) = x . x' / n0` (no C_W), not `K^(1) = C_W * x.x'/n0`. Both differ from the
naive formula by exactly one factor of `C_W`, which is why the bug is invisible at
`C_W=1` (a common default in NTK-recursion code/examples) and only surfaces once a
non-unit `C_W` is used — exactly the paper's own critical-initialization setting
(`C_W=2`), which is what this reproduction needs for Claim 5.

**Fix**: base case `Theta = x @ xp / n0`; recursion step
`Theta = Theta * Kdot + Knew / C_W`. See `repro_toy.py::relu_inf_width`.

**Verification**: re-ran the same smoketest after the fix (widths 50 and 200, 3000
inits, depth 3, C_W=2, `results/width_sweep_check2.json`): empirical diag = 1.039
(w=50) / 1.037 (w=200) vs. corrected analytic 1.035; empirical offdiag = 0.187
(w=50) / 0.188 (w=200) vs. corrected analytic 0.189 — matches to within MC noise at
both widths, and the diagonal is now visibly *flat* across widths (no residual
1/n trend), consistent with Theorem 5.2's prediction that ReLU's diagonal NTK mean
has zero finite-width correction. This is exactly the kind of sign/normalization bug
the self-audit pass is meant to catch before it silently propagates into a false
verdict (the bug was in the reproduction's own analytic ground-truth, not the paper —
the empirical/autograd side was correct throughout, confirmed by the fact that only
the analytic-side numbers moved after the fix).

**Note on scope**: this bug was in my own hand-derived infinite-width closed form
(used as ground truth for Claims 3/4), not in anything transcribed from the paper's
Feynman-diagram recursion (Claims 1/2, Eq. 78) — that recursion was deliberately not
implemented from scratch given the OCR-corrupted `\Delta\Omega_d` operator notation
in the extracted appendix text (see PAPER_BRIEFING.md's scope note); Claims 1/2 are
tested via the qualitative/structural check described there instead, precisely to
avoid this exact failure mode (silently misimplementing a fragile piece of paper
notation and reporting a wrong verdict with false confidence).

**Update**: Claims 1/2 were reopened in a later session (see Round 2 below) — the OCR concern
above was resolved as a false alarm, and Eq. 78 was implemented and numerically confirmed.

## Round 2 — Claims 1/2 reopened: CRN's R² false-positive pitfall, and the analytic-marginalization
## resolution of Eq. 78

**Found during**: a later session revisiting the original INCONCLUSIVE verdict for Claims 1/2, at
the user's request to actually implement and numerically test Eq. 78 (not just check it
structurally). Full blow-by-blow run history is in `REPRO_LOG.md`; this entry documents the two
methodological findings worth remembering for future reproductions.

### Finding 1: the OCR concern was a false alarm

The original INCONCLUSIVE verdict rested on the arXiv PDF extraction rendering the paper's
`\Delta\Omega_d` operator glyph as U+2126 OHM SIGN — plausible evidence of broader font-
substitution corruption in that math block. Once the actual OpenReview PDF was obtained, a direct
byte-level comparison of Delta-symbol counts between the two extractions came back identical —
the OHM SIGN substitution is a deterministic font-encoding quirk of this specific PDF's embedded
math font, not evidence of corrupted/dropped content. **Lesson**: an OCR/extraction artifact that
*looks* alarming (a wrong-seeming Unicode codepoint) is not automatically evidence of unreliable
extraction elsewhere in the document — cross-checking against a second independently-generated
extraction (here, OpenReview vs. arXiv) is a cheap, decisive way to tell a real corruption problem
from a cosmetic one before downgrading a verdict over it.

### Finding 2: CRN (common random numbers) can produce a high R² that is NOT a converged estimate

Testing Eq. 78 requires five ingredient tensors (`Theta^{1}`, `K^{1}`, `V`, `D`, `F`) that were
initially all measured via Monte Carlo width-sweeps, fitting a `1/n` trend across widths to
extract each tensor's O(1/n) coefficient. Independent per-width sampling gave weak fits (R²
~0.0002-0.03 on the key `Theta^{1}(3)` fit even at 100k samples/width, `scale1` run) — a noise-
dominated regime, not a sample-size problem alone (some ingredients' R² got *worse* with 5x more
samples). Switching to CRN (drawing one maximal-width network per init and slicing sub-networks
per width, correlating noise across widths) looked like it fixed this: at `n_inits=2000`
(`crn_smoke`), R² on the target fit jumped to 0.90. But scaling CRN up to `n_inits=200000`
(`crn1`) made R² **drop** to 0.44, and every ingredient's fitted intercept kept drifting rather
than stabilizing. **The `crn_smoke` high R² was a small-sample fluke**: CRN's shared randomness
across widths can produce a spuriously clean-looking linear trend before enough samples have
accumulated to reveal the true (noisier) shape — the correlation CRN introduces by construction
is exactly the kind of structure that can fool an R² metric at low N. **Lesson**: never trust a
CRN-based fit's R² at face value without checking that the *point estimate* is stable as sample
size grows; a high R² at one sample size proves nothing about convergence on its own.

### Resolution: analytic marginalization replaces MC for 5 of 6 ingredient tensors

Rather than continuing to throw more MC compute at fundamentally noisy 4th-cumulant estimates,
went back to the OpenReview PDF's page *images* (not the pdftotext extraction, to eliminate any
residual OCR risk on stacked multi-line fractions) — pages 4, 16, 17 — and found the paper's own
layer-to-layer recursions: Eq. 45 (V), Eq. 47 (`K^{1}`), Eqs. 49/50 (D), Eq. 5 (F), and Eq. 78
itself (`Theta^{1}`, applicable at *any* layer transition, not just the l=2→3 step already
implemented). The key structural fact: **`z^(1)`, the first hidden layer's preactivation, is
exactly Gaussian for any finite width** (no CLT approximation needed at layer 1 — this holds
because `z^(1)` is a linear combination of the *inputs*, which are fixed, not of a previous
layer's own already-approximately-Gaussian activations). Consequently every layer-1 fluctuation
tensor (`Theta^{1(1)}`, `K^{1(1)}`, `V^(1)`, `D^(1)`, `F^(1)`) is exactly zero, and every term in
the five recursions above, when applied at l=1→2, is proportional to one of these — **except one
"new" term per equation** (present only in the V/D/F recursions, not `Theta^{1}`'s or `K^{1}`'s)
generated fresh by the current layer's finite-width sum, which reduces to a pure Gauss-Hermite
quadrature integral with zero MC noise.

**Result**: `Theta^{1}(2) = 0` and `K^{1}(2) = 0` exactly (every term in their recursions vanishes
at l=1→2); `V(2)=0.047478`, `D(2)=0.025578`, `F(2)=0.027529` from quadrature. This *explains* the
entire noisy-MC saga above: every session's MC estimate of `Theta1_2`/`K1_2` was wildly
inconsistent in sign and magnitude (across `debug1`, `debug2`, `scale1`, `crn_smoke`, `crn1`)
because the true value is exactly zero — no amount of sampling, CRN or otherwise, could ever have
"resolved" a signal that isn't there. The analytic V/D/F values were cross-checked against
`scale1`'s independent 100k-sample MC estimates and matched to 1-3% (V: 0.0475 vs 0.0477 MC; D:
0.0256 vs 0.0259; F: 0.0275 vs 0.0282) — strong confirmation the new derivation is correct, and
that the MC estimates were fine all along (their weak R² was a fit-window problem, not an
accuracy problem).

With five of the six needed quantities now exact, only `Theta^{1}(3)` (the final output, a
well-behaved 2nd-order NTK quantity, not a noisy 4th-cumulant) still needed an independent MC
measurement. Ran this at `n_inits=300000`, 7 widths (`final1`, `results/eq78_final1.json`):
**measured `-0.062614` vs. the fully analytic prediction `-0.063113` — relative error 0.8%.** The
fit's own R² (0.12) is weak, consistent with residual O(1/n²) curvature across the 10x width
range rather than noise (per-width standard errors are tiny, 5e-5 to 2e-4), and the point-estimate
agreement with a zero-noise theoretical value is the decisive evidence here, not R².

**Implementation**: `analytic_layer2_ingredients()`, `theta13_measurements()`,
`run_theta13_sweep()`, and the `analytic-eq78` CLI mode in `repro_eq78.py` (added this session;
the old fully-MC `full-eq78` mode is kept for reference but superseded for Claims 1/2 purposes).

**Verdict impact**: Claims 1/2 upgraded from INCONCLUSIVE to TOY-VERIFIED — see `VERDICTS.md`.
