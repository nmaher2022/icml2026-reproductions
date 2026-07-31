# Bugfix / assumption log — spectral-cit-nPzckCXmHE

Format: one entry per finding, with before/after where numbers are affected. This log exists so a
later reviewer can trust final numbers weren't cherry-picked or silently patched.

## 1. `validation_error` computed the wrong quantity (found during initial timing run, before any
   claim-verification run)

**Bug:** `E_m^val` (paper p.5) is defined as
`max{||Ĉ_{Û₀Û₀}−I_d||, ||Ĉ_{V̂₀V̂₀}−I_d||, ||Ĉ_{Ŵ₀Ŵ₀}−I_2d||}` — three **self**-covariance
orthonormality discrepancies. My first implementation of `validation_error()` in `scit_lib.py`
computed `Ĉ_{ÛV̂}` (a **cross**-covariance between u and v) in place of `Ĉ_{ÛÛ}`, i.e. exactly the
"metric measuring something subtly different from what the claim states" bug class flagged in
`verdict_checklist.md`.

**How found:** a single near-paper-scale timing run (N=1000, d=10, d_z=3, 400 epochs) returned
`E_val=1.20`, which looked too large given the test statistic itself (`T_n=93.9` vs. a chi2(100)
mean of 100 — plausible calibration) suggested representations were reasonably well learned. Since
E_val is supposed to shrink toward 0 as training converges, a value of ~1.2 next to an apparently
well-calibrated T_n was the inconsistency that prompted rereading the exact E_m^val definition
against the code.

**Fix:** changed `Cuv = empirical_cov(u, v)` → `Cuu = empirical_cov(u, u)` (and correspondingly
`e1 = ||Cuu - I||` instead of `||Cuv - I||`) in `validation_error()`. `Cvv`, `Cww` terms were
already correct.

**Effect:** `validation_error()` is a diagnostic reported alongside verdicts (Claim 3), not an
input to the test statistic `T_n` itself (Eq. 10) — so this bug did not silently corrupt the
Claim 1/2 (validity/power) numbers, only the Claim-3 diagnostic. Re-verified after fix: on the same
trial (N=1000, d=10, d_z=3, seed=0), `E_val` post-fix is reported in `claim3_representation_error.py`'s
output — see that script's log for the corrected number.

## 3. Missing "dimension pruning" post-processing step (found via a debug probe of the corrected
   E_val, before the full calibration run)

**Finding:** even after fixing bug #1, `E_val` stayed near 1.0 across every condition, which is
suspicious for a quantity that should shrink toward 0 with training. Debug probe (direct eigen-
decomposition of `C_ŴŴ` before whitening, with d_z=3, output_dim=20 for w) showed only ~3 of 20
eigenvalues were non-negligible (~0.06-0.16); the other 17 were ~1e-9 (numerical noise). Whitening
divides by `sqrt(eigenvalue)`, clamped at `eps=1e-6` for numerical safety -- since the real
eigenvalues (~1e-9) are *below* that clamp, those directions get scaled by `1/sqrt(1e-6)=1000`, but
because the true variance there is ~1e-9, the resulting whitened variance is only ~1e-9*1e6=1e-3,
not 1 -- so `Ĉ_{ŴŴ}` stays far from `I_{2d}` in those directions, not because of a code bug but
because `w_theta` (an MLP mapping only a 3-dimensional Z into R^20) doesn't spread variance across
all 20 output dimensions under the paper's own stated reference hyperparameters
(lr_inner=3e-5 -- the smallest end of Table 2's grid) within a few-hundred-epoch budget.

**This is exactly what Appendix C's "Dimension pruning" paragraph is for** ("for added stability,
... we computed a lower-rank test statistic by performing an SVD of the test-statistic matrix and
retaining only the leading `[perc_dim_prune x output_dim]` singular triplets ... evaluated using a
corrected chi^2 distribution with degrees of freedom equal to the retained (pruned) dimension") --
I had read this paragraph while writing the briefing but initially treated it as a minor stability
tweak rather than a load-bearing step, and had not implemented it. Implemented afterward as
`test_statistic_pruned()` in `scit_lib.py`: SVD the `d x d` matrix `Delta = Ĉ_UV - Ĉ_UW Ĉ_WV`, keep
the top `floor(perc_dim_prune * d)` singular values, compute `T_n` from those alone, and compare
against `chi2(k^2)` (not `chi2(d^2)`) where `k` is the retained dimension.

**Effect:** all claim-1/2 experiment scripts report *both* the raw (`T_n`, vs. `chi2(d^2)`) and the
pruned (`T_n_pruned`, vs. `chi2(k^2)`) statistics side by side, so the verdict can state plainly
whether pruning was necessary for calibration to hold in this from-scratch reimplementation, rather
than silently only reporting whichever one looks better.

## 2. M_theta / N_theta parameterization ambiguity (documented assumption, not a bug — flagged
   during initial implementation, before any run)

Algorithm 1's pseudocode (paper p.5) only lists gradient updates for `w_theta` (inner step) and
`u_theta, v_theta` (outer step); it does not give M_theta, N_theta (the diagonal/matrix nuisance
parameters used in L_out/L_in, p.4) their own explicit update rule, even though the loss
definitions reference `M = M_theta` and `N = (N_theta + N_theta^T)/2` as if they were learned.

**Resolution used in this reproduction:** M_theta (a length-d vector, init ones) is treated as an
extra learnable parameter trained together with u_theta, v_theta in the outer step (since M only
appears in L_out); N_theta (a 2d x 2d matrix, init identity) is trained together with w_theta in
the inner step (since N only appears in L_in). This is the most literal reading of the box
consistent with which loss each nuisance parameter appears in, but it is **not verified against
the authors' own code** (which was not consulted, per this reproduction's from-scratch-only
policy) — flagged here as an assumption, not a confirmed match to the paper's exact training
procedure. Does not affect the final test statistic `T_n` (Eq. 10), which has no M/N term at all.
