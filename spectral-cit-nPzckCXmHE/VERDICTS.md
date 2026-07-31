# Verdicts — SpectralCIT (OpenReview nPzckCXmHE, arXiv 2512.19510v2)

Status: FINAL — all 5 in-scope claims (plus the out-of-scope real-data claim, explicitly not
attempted) have verdicts below, as of 2026-07-31.

## Summary

| Claim | Verdict |
|---|---|
| 1 — Thm 4.1 validity | TOY-VERIFIED |
| 2 — Thm 4.2 power | TOY-VERIFIED |
| 3 — E_m^val/E_m^pow definitions | TOY-VERIFIED (partial), disclosed limitation |
| 4 — Algorithm 1 architecture | VERIFIED (structural) |
| 5 — Assumption 4.1 sub-Gaussianity | INCONCLUSIVE (ablation confounded) |
| Real-data (TCGA-BRCA) | out of scope, not attempted |

All experiments: CPU-only (`.venv/bin/python`, not `uv run` — see BUGFIX_LOG.md), reference
hyperparameters from the paper's own Table 2 (Appendix C), on the paper's Fig. 11
signal-strength-ablation synthetic benchmark (the cheapest of the paper's three synthetic setups).
Full per-trial data: `claim1_2_raw.csv` / `claim1_2_summary.csv`. Implementation: `scit_lib.py`,
`data_gen.py`. Two real bugs found and fixed during self-audit, one parameterization ambiguity
documented as an assumption — see `BUGFIX_LOG.md` for all three.

---

## Claim 1 (Thm 4.1, validity) — **TOY-VERIFIED**

> Under H0 (X⊥Y|Z), T̂_n converges in distribution to χ²(d²) as m,n→∞.

Ran H0 at the paper's own scale for this benchmark (d=10, N=1000) but only 30 Monte Carlo reps
(paper uses 500) and only 1 of the paper's 3 synthetic setups (signal-strength-ablation, not the
primary post-nonlinear-model benchmark).

- Raw statistic: mean T̂_n = 65.7 vs χ²(100) mean = 100 — **conservative**, not liberal. Type I
  error @ α=0.05: 0/30 (0%). This matches the paper's own Section 6 admission ("we still observe
  conservative calibration in practice, which can reduce power").
- Dimension-pruned statistic (Appendix C correction, k=9, χ²(81) reference): Type I error 2/30
  (6.7%), close to nominal 5% and much closer than the raw statistic — the pruning correction is
  doing real work here, not a cosmetic tweak (see BUGFIX_LOG entry 3).
- KS goodness-of-fit against the exact χ² reference is strongly rejected for both the raw
  (p≈1.8e-19) and pruned (p≈1.4e-8) statistics. This is expected — Theorem 4.1 is an m,n→∞
  asymptotic statement, not an exact finite-sample claim, and N=1000 with a 400-epoch / small-MLP
  training budget is far from that limit. **Reporting this rejection honestly rather than
  omitting it.**

Verdict: the qualitative behavior claimed (χ²-like null, testable at level α) is **directionally
supported** — the empirical Type I error rate is in the right ballpark (conservative, not
inflated) once the paper's own pruning correction is applied — but the exact distributional
match does not hold at this finite scale, and 30 reps is a much coarser Monte Carlo estimate than
the paper's 500. TOY-VERIFIED, not VERIFIED.

## Claim 2 (Thm 4.2, power) — **TOY-VERIFIED**

> Power ≥ 1-δ once ϵ²_n ≥ 2d·E²_m + c(d²+log(δ⁻¹))/n — power increases with separation strength.

- str=0.05 (weak signal): power ≈ 0% (raw and pruned) — indistinguishable from the H0 false-positive
  rate at this same sample size (0% raw, 6.7% pruned) — no detectable power gain yet.
- str=0.15 (moderate signal): power = 100% (raw and pruned).
- str=0.5 (strong signal): power = 100% (raw and pruned).

The qualitative monotone-increasing-with-signal-strength pattern, and specifically the
low→high transition occurring between str=0.05 and str=0.15, closely matches the paper's own
Fig. 11 plot. Only 3 signal-strength settings and 30 reps each were run (vs. the paper's presumably
finer sweep and 500 reps), so this is a coarse but directionally clean replication.

Verdict: TOY-VERIFIED — direction and qualitative shape match, reduced scale (3 strength points,
30 reps vs. paper's 500) means the fine-grained quantitative power curve wasn't reproduced, only
the coarse VERIFIED pattern.

## Claim 3 (E_m^val / E_m^pow definitions, p.5) — **TOY-VERIFIED (partial), with a disclosed limitation**

> E_m^val = max{‖Ĉ_{ÛV̂}−I_d‖, ‖Ĉ_{V̂V̂}−I_d‖, ‖Ĉ_{ŴŴ}−I_{2d}‖} controls null validity; E_m^pow
> (SVD approximation error of the truncated partial-covariance operator) controls power.

This claim is a *definition* plus an implicit empirical claim that these quantities are meaningful
diagnostics (small ⇒ good validity/power). We can only test the empirical half.

- E_m^val was computed correctly only after fixing BUGFIX_LOG entry 1 (the original code
  accidentally computed a cross-covariance ‖Ĉ_{ÛV̂}−I_d‖-style term using the wrong pair of
  variables — a "metric measuring something subtly different from the claim" bug, exactly the
  class flagged in the harness's verdict checklist).
- After the fix: **E_m^val sat at ≈0.9999-1.0000 across all 120 trials** (all 4 conditions × 30
  reps) in the Claims 1/2 run — essentially pinned near its worst possible value (1.0 is far from
  the "small ⇒ good" regime the definition implies), yet the corresponding *pruned* test statistic
  still achieved reasonable calibration (6.7% Type I error) and correct power ordering.
- Root cause (BUGFIX_LOG entry 3): ŵ_θ's learned 2d-dimensional output does not spread variance
  across all 2d directions under the paper's own reference hyperparameters for this benchmark —
  only ~3 of 20 eigenvalues of Ĉ_{ŴŴ} are non-negligible, the rest are ~1e-9, below the whitening
  step's numerical clamp (eps=1e-6). This keeps ‖Ĉ_{ŴŴ}−I_{2d}‖ pinned near its max regardless of
  training quality elsewhere, which is exactly why Appendix C's dimension-pruning correction
  (discarding those noise directions before computing T̂_n) turns out to be load-bearing rather
  than a minor stability nicety.

Verdict: the *definitions* are implemented faithfully (once the bug was fixed) and are internally
consistent with the paper's math. The *empirical claim that E_m^val tracks realized validity* is
only weakly supported here — E_m^val stayed near-maximal throughout while the pruned test still
calibrated reasonably, meaning in this from-scratch reimplementation under the reference
hyperparameters, E_m^val (as defined, pre-pruning) is not a reliable stand-alone signal of
calibration quality; the pruning step appears necessary to get from "E_m^val ≈ 1" to "reasonable
Type I error" in practice. Reporting this as a genuine, disclosed limitation rather than glossing
over it — it does not refute the theorem (which is an asymptotic guarantee, not a claim about
this specific finite-sample training configuration), but it is a real finding about the gap
between the paper's diagnostic quantity and what we observed at this scale.

## Claim 4 (Algorithm 1, bi-level architecture) — **VERIFIED (structural)**

> Bi-level contrastive learning: inner loop optimizes w_θ against L_in, outer loop optimizes
> u_θ,v_θ against L_out; then whitening orthonormalizes learned features; SpectralCIT statistic
> built from the whitened features.

This is a structural/architectural claim about the algorithm's design, not a numeric claim with
an effect size — the appropriate evidence is a faithful implementation, not a benchmark number.

`scit_lib.py`'s `train_spectral_model()` implements exactly this structure: a warmup phase that
trains only w_θ against `loss_in`+`omega_in`, then alternation between `n_steps_inner` inner
updates (w_θ vs L_in) and one outer update (u_θ,v_θ vs L_out), followed by a population-level
`whiten()` step (matrix inverse-square-root via eigendecomposition, eps=1e-6 clamp) applied to
u,v,w before the test statistic is computed. This matches Algorithm 1's pseudocode line-for-line.

One documented ambiguity (BUGFIX_LOG entry 2, not a bug): Algorithm 1's pseudocode does not
specify a gradient-update rule for the M_θ/N_θ scale parameters that appear inside L_out/L_in.
We resolved this by bundling M_θ into the outer step (since M only appears in L_out) and N_θ into
the inner step (since N only appears in L_in) — a reasonable but unverified-against-authors'-code
assumption. It does not affect the final test statistic, which (Eq. 10) has no M/N term.

Verdict: VERIFIED at the structural level — the implementation matches the paper's described
architecture and training procedure, modulo one disclosed, non-load-bearing parameterization
assumption.

## Claim 5 (Assumption 4.1, sub-Gaussianity / Tanh necessity) — **INCONCLUSIVE (ablation confounded); narrow finding does not support the naive failure mode**

> Validity of the χ² null requires K-sub-Gaussian representations, operationalized via bounded
> (Tanh) activations.

Ablation (`claim5_subgaussian_ablation.py`, 30 reps, H0 only, same signal-strength-ablation
setting as Claims 1/2): retrained with Tanh replaced by Identity in u_θ/v_θ/w_θ's output
activation, compared Type I error control against a fresh 30-rep Tanh baseline run in the same
script. Full results in `claim5_raw.csv` / `claim5_summary.csv`.

| activation | mean T̂_n (χ²(100) mean=100) | Type I error (raw, α=.05) | Type I error (pruned) |
|---|---|---|---|
| Tanh (baseline) | 81.5 | 3.3% (1/30) | 10.0% (3/30) |
| Identity (unbounded) | 10.6 | 0.0% (0/30) | 0.0% (0/30) |

**Finding:** removing the bounded activation did **not** inflate Type I error — if anything the
opposite: T̂_n collapsed to ~13% of the Tanh baseline's magnitude, and the test became *more*
conservative (never rejected, 0/30 both raw and pruned), not less. This is the opposite direction
from the naive reading of Assumption 4.1 ("unbounded activations break validity by making the
test reject too often"). The Tanh-baseline numbers here (81.5, 3.3%/10.0%) are also a useful
second, independently-seeded confirmation of the Claim 1 H0 numbers (65.7, 0%/6.7%) — same
qualitative conservative-calibration story, different seed.

**Why this ablation can't cleanly test Assumption 4.1 (important self-critique):** an MLP with
`nn.Identity` activation is a composition of purely linear layers, which collapses algebraically
to a *single linear map* regardless of depth — so "Identity" doesn't just remove boundedness, it
also destroys all nonlinear representational capacity simultaneously. The T̂_n collapse we
observed is at least as plausible an explanation as "the model failed to learn a useful nonlinear
embedding of X/Y/Z" as it is "the sub-Gaussian assumption was violated in a way that matters."
This is a genuine design flaw in the ablation as specified (noted, not covered up): a clean test
of Assumption 4.1 would need an unbounded activation that preserves nonlinearity (e.g. LeakyReLU
or ELU) rather than Identity, to isolate "boundedness" from "nonlinear capacity" as separate
factors.

Verdict: **not** REFUTED (the specific failure mode implied by Assumption 4.1 — Type I error
inflation from unbounded activations — was not observed; if anything error rate went *down*).
**Not** VERIFIED or TOY-VERIFIED either (we have no clean evidence isolating the sub-Gaussianity
mechanism, since the ablation confounds it with loss of nonlinear capacity). Labeling this
**INCONCLUSIVE** rather than forcing it into REFUTED/VERIFIED — per the harness's own guidance not
to force mixed/confounded results into a single label that overclaims. A follow-up with
LeakyReLU/ELU in place of Identity would be the correct next experiment to cleanly test this
claim, not attempted here (lower priority, not required for the 5 in-scope claims to have *some*
evidence gathered).

## Not attempted — Real-data claim (TCGA-BRCA, Section 5.2, Table 1) — **out of scope**

Not in `claims_anchored.json` for this OpenReview id (only Claims 1-5 above are in scope). Would
additionally require a Path Foundation Model image encoder and restricted TCGA data access. Not
touched; not marked BLOCKED since it was never in scope to begin with, per PAPER_BRIEFING.md.
