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
