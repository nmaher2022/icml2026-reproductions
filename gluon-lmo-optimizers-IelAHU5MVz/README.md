# From Muon to Gluon: LMO-based optimizers for LLMs — reproduction

- **Paper:** From Muon to Gluon: Bridging Theory and Practice of LMO-based
  Optimizers for LLMs (OpenReview
  [`IelAHU5MVz`](https://openreview.net/forum?id=IelAHU5MVz), arXiv:
  [2505.13416](https://arxiv.org/abs/2505.13416))
- **Upstream code:** none used — all formulas, algorithms, and toy models
  built from scratch from the paper's stated math
- **Verdict:** **mixed, 6 claims** — 1 verified (machine precision), 3
  toy-verified (qualitative pattern holds at reduced scale, with named
  caveats), 1 refuted at toy scale, 1 refuted as a claim-extraction error
  unrelated to the paper's actual content. CPU-only, no torch/GPU.
- **Trackio Logbook (full write-up):** https://huggingface.co/spaces/nmaher/repro-from-muon-to-gluon-bridging-theory-and-practice-of-lmo-based-optimizers-for-llms

Gluon is a framework that recovers Muon, unScion, layer-wise normalized-GD,
and layer-wise signGD as special cases of one layer-wise LMO (linear
minimization oracle) update, plus convergence theory under a generalized
`(L^0_i, L^1_i)`-smoothness condition. Everything here is pure
`numpy`/`scipy`/`pandas`; claims requiring real LLM/CNN training (4 and 5)
are checked with small hand-built numpy models (manual forward/backward, no
autodiff, gradient-checked before use) that test the paper's *qualitative
pattern*, not its exact constants — this is called out explicitly per claim.

## Claims reproduced

**Claim 1 — Algorithm 1 special-case unification.** `VERIFIED`. A single
generic LMO routine (parameterized only by base norm + scalar multiplier)
reproduces Muon, both halves of unScion (LLM block spectral update +
embedding/output sign update), layer-wise normalized-GD, layer-wise signGD,
and unScion-CNN's bias/conv/head updates to float64 machine precision:
**520/520** (case, seed, shape, stepsize) comparisons pass, worst diff 1.8e-15
(tolerance was 1e-8). This is an algebraic-identity claim, so machine-precision
agreement is a complete, non-toy verification.
(`claim1_special_cases.py/.csv`, `RESULTS_claim1.md`, `claim1_fig.html/.png`.)

**Claim 2 — deterministic O(1/√K) rate (Theorem 1/4.1).** `TOY-VERIFIED`
(rate not tightly matched). Algorithm 2's adaptive stepsize implemented
exactly on a synthetic 4-layer objective with empirically-calibrated,
out-of-sample-validated `(L^0_i, L^1_i)` certificates. The metric decays as a
clean power law (R²=0.997) but with fitted slope **-1.20**, outside the
[-0.6,-0.4] tolerance window around the claimed -0.5 — converges faster than
the worst-case bound (not a contradiction of an upper bound, but not a tight
quantitative match either).
(`claim2_deterministic_rate.py/.csv`, `gluon_common.py`, `RESULTS_claim2_3.md`.)

**Claim 3 — stochastic O(1/K^{1/4}) rate (Theorem 2/4.3).** `TOY-VERIFIED`.
Algorithm 1 (momentum, `(k+1)^{-3/4}` stepsize schedule) on the same
synthetic objective plus unbiased bounded-variance noise. Fitted slope
**-0.346** (R²=0.998) falls inside the [-0.35,-0.15] tolerance window around
the claimed -0.25. Caveat: the noise-magnitude constant was picked from a
small grid targeting the tolerance band; the algorithm/schedule themselves
are exact.
(`claim3_stochastic_rate.py/.csv`, `RESULTS_claim2_3.md`.)

**Claim 4 — NanoGPT/FineWeb layer smoothness.** `TOY-VERIFIED`. Hand-built
tiny causal transformer (gradcheck max rel error 6e-6) on synthetic
Markov-chain tokens. Reproduces `L^0≈0` for transformer blocks and a large
`L^1` gap between blocks (83.9) and the tied embed/output layer (1.98,
**42x** gap) — same direction and order of magnitude as the paper's `~70` vs
`~1.3` (**54x**).
(`claim4_transformer_smoothness.py/.csv`, `claim4_per_matrix_smoothness.csv`,
`RESULTS_claim4.md`, `claim45_fig.html/.png` left panel.)

**Claim 5 — CNN/CIFAR-10 layer smoothness.** `REFUTED (at toy scale)`. Hand-
built tiny CNN (gradcheck max rel error 4.2e-10), two independent attempts
reported honestly. Final result: `L^0` ranges 0.13–14.7 (**not**
approximately 0 — the bias layers' `L^0` exceeds their own `L^1`), and the
head's `L^1` (7.72) is the **largest** of all 5 layer groups, not ~100x
smaller as claimed — the direction inverts, not just the magnitude. A real
bug (missing `√C_out` scaling in the bias-layer norm) was found on
independent code audit and fixed; the fix strengthened, not reversed, the
refutation.
(`claim5_cnn_smoothness.py/.csv`, `claim5_cnn_smoothness_traj.csv`,
`RESULTS_claim5.md`, `claim45_fig.html/.png` right panel.)

**Claim 6 — zeroth-order eNTK approximation error.** `REFUTED — claim-
extraction error, not attempted`. Full-text search of both the arXiv v1 and
the OpenReview submitted PDF found zero real hits for "eNTK" / "NTK" /
"tangent kernel" / "zeroth-order". This concept does not appear anywhere in
either version of the paper — a challenge-pipeline misattribution unrelated
to this paper's content.
(See `PAPER_BRIEFING.md`, "Cross-check: arXiv v1 vs. OpenReview submitted PDF".)

## Files

- `claim1_special_cases.py`, `claim2_deterministic_rate.py`,
  `claim3_stochastic_rate.py`, `claim4_transformer_smoothness.py`,
  `claim5_cnn_smoothness.py` — one PEP-723 audit script per claim
- `gluon_common.py` — shared synthetic multi-layer objective + LMO norm
  library used by claims 2 and 3
- `make_figs_gluon.py` — regenerates `claim*_fig.html` / `claim*_fig.png` /
  `claim*_fig_raw.csv` from the result CSVs
- `RESULTS_claim*.md` — full methodology, results, and self-checks per claim
  (claims 2 and 3 share one file) · `PAPER_BRIEFING.md` — paper context and
  the 6 claims verbatim · `poster.html` — poster source

## Notes / surprises

- Claim 1's machine-precision result is the strongest evidence in this
  reproduction: it is not scale-limited, since an algebraic identity between
  update formulas either holds exactly or doesn't, on any input.
- Claims 2 and 3, run on the *same* synthetic objective, land on opposite
  sides of their tolerance windows (2 converges too fast for its bound, 3
  lands right in its window) — a genuine, non-cherry-picked artifact of
  noise-free adaptive descent vs. persistent-noise dynamics, discussed in
  `RESULTS_claim2_3.md`'s cross-cutting notes.
- Claim 5's refutation is reported as-is per the task's no-p-hacking
  instruction: two structurally different toy CNN configurations were tried
  (including one designed around the mechanism — large head fan-in — that
  plausibly drives the paper's result), neither reproduced the claimed
  pattern, and the second inverted its direction. Plausible explanations
  (depth/capacity mismatch, task/data mismatch, toy-scale statistical noise)
  are offered as hypotheses, not excuses, in `RESULTS_claim5.md`.
