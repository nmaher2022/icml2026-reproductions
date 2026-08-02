# BUGFIX_LOG.md — Ellipsoidal TSF (CoAHlJuMdh)

Self-audit pass (Step 4) run after the first full set of results, rereading `fern_lib.py` against
Algorithm 1 / Eq. 1-2 / Appendix A.1/A.3 of the paper, looking specifically for sign errors,
wrong-granularity gates/masks, unwired feedback loops, and metrics measuring something subtly
different from what the claim states.

## Real bug found and fixed: `no_rotation` ablation confounded rotation removal with reduced capacity

**Before**: `Fern.__init__` set `self.R = 1 if no_rotation else n_reflections`, so the OT head's
final linear layer (`OTHead.net`, output dim `g*(2p + R*p)`) was constructed with `R=1` for the
`no_rotation` variant instead of the base config's `R=8`. Even though `forward()` correctly bypassed
the Householder application entirely for `no_rotation` (`mu = lam * ty`, true identity rotation),
the *unused* Householder-vector outputs still shrank the head's parameter count: base model
55,776 params vs. the pre-fix `no_rotation` model's smaller head (output dim 288 vs 960 at
p=24, g=4, R=8→1). This is exactly the "gate/mask at the wrong granularity" bug class the harness
flags to check for — here the ablation flag leaked into a place (head capacity) it should not have
touched, so any MSE difference between base and `no_rotation` was partly attributable to a smaller
model, not to the rotation being removed. This would have made the ablation's evidence for Claim 4
weaker than it should be (a capacity handicap stacked on top of the intended structural handicap).

**Fix**: `self.R = n_reflections` unconditionally; only the `forward()` branch (bypass vs. apply
Householder) differs between `no_rotation` and base, so both models have identical parameter counts
(verified: 55,776 both). `run_claim4_ablations.py` rerun after the fix.

**Before/after (mean MSE % vs base, Lorenz-63, 3 seeds)**:
- Before fix: `no_rotation` +6.4%, inconsistent per-seed direction (1 of 3 seeds actually *better*
  than base).
- After fix: `no_rotation` +7.6%, consistent direction on all 3 seeds (worse than base in every
  seed). The fix didn't flip the qualitative verdict (already directionally correct on average) but
  made the effect cleaner and removed the one seed that contradicted the paper's claim.

## Reviewed, not changed: eigenvalue bound uses softplus, not the paper's stated soft-clamp

Appendix A.3.3 states "Eigenvalues and translation are parameterized with differentiable soft
bounds for numerical stability. We use a soft-clamp..." and gives explicit numeric ranges for the
elementwise coupling scale (`s ∈ [0, 5.5]`) and shifts (`t_y ∈ [-15, 15]`), which this
implementation matches exactly. No explicit numeric range is given for the eigenvalues Λ
specifically. `CouplingHead`'s and `OTHead`'s shift outputs use the paper's stated `[-15,15]`
bound; the coupling scale uses `[0,5.5]`; the eigenvalues use `softplus` (unbounded above,
nonnegative) since no upper bound is specified for Λ in the text and softplus is a standard,
stable nonnegativity parameterization. Documented as an interpretation choice, not a bug — no
NaNs/divergence observed in any of the ~60 total training runs across all four experiment scripts.

## Reviewed and confirmed correct: Householder U / U^T application order

Verified by hand that `apply_householder(v, y)` sequentially applying `v[0], v[1], ..., v[R-1]`
computes `U y` where `U = H_R...H_1` (paper's definition, Appendix A.1) when `v[0]` is identified
with the paper's `v_1`. Verified that `apply_householder(v.flip(-2), z)` then correctly computes
`U^T z = H_1...H_R z` (applying `v_{R}` first, `v_1` last) since each Householder reflection is
symmetric. `Fern.forward`'s `mu = U^T (Λ ⊙ (U t_y))` therefore matches Algorithm 1 line 13's
`y* = U_y^T Λ_y U_y (y0 + t_y)` evaluated at `y0=0`, which is the correct closed-form mean since
`E[y0]=0` and the map is affine (`E[y*] = U^T Λ U · t_y` exactly). No sign or ordering error.

## Genuine (non-bug) findings from the toy-scale runs, disclosed honestly in VERDICTS.md
Not implementation bugs, but results worth flagging here since they run counter to the paper's
headline claims and were specifically looked for as *possible* symptoms of a bug before being
accepted as genuine toy-scale findings:
- **Claim 1 (nonstationary robustness) does not hold uniformly across systems at toy scale.**
  Fern beats DLinear clearly on Lorenz-63 (both base and param-shock, ~1.4-1.5x lower MSE) but
  *loses* to DLinear on Chua's circuit (Fern's MSE is 5.5-7.7x *higher* than DLinear's, both
  scenarios) and is roughly tied/mixed on Roessler. Investigated for a scale-sensitivity bug (the
  paper doesn't describe any RevIN/instance-normalization preprocessing for the chaotic
  benchmarks — confirmed absent from the text, so raw-scale inputs are the correct reading) and
  found none; most likely a genuine toy-scale effect (DLinear is a very strong baseline for
  short-horizon, low-training-budget regimes; Fern's flexible spectral parameterization plausibly
  needs more data/epochs than this toy budget gives it to earn its structural advantage,
  especially on systems where the toy trajectory window doesn't exhibit strong nonstationarity).
- **Claim 3 (EPT) is reversed at toy scale in every one of 6 scenarios** — DLinear has *higher*
  mean EPT than Fern on both Lorenz shocks, both Roessler shocks, and both Chua shocks, opposite
  the paper's claimed ordering. Plausible explanation, not verified as the sole cause: EPT is a
  first-threshold-crossing statistic, not an average-error statistic, so a smoother/more
  conservative predictor (DLinear) can have a longer EPT even while being worse on average MSE
  (as seen on Lorenz, where Fern has *lower* MSE but *lower* EPT than DLinear simultaneously) —
  this is a real property of the EPT metric definition itself, not obviously a Fern weakness.
- **Claim 5's specific "baselines collapse, Fern's geometry persists" mechanism does not appear at
  toy-scale horizons (24-192 steps).** DLinear never collapses to mean-guessing in this range
  (unlike the paper's 96-720 step baselines-collapse-early story), so the SWD gap between Fern and
  DLinear that the paper attributes to post-collapse geometric persistence isn't observable —
  Fern's SWD advantage over DLinear actually *shrinks* from 2.44x (h=24) to ~1.0x (h=96-192) as
  horizon grows, the opposite trend from what the claim implies. Most likely needs materially
  longer horizons/trajectories (closer to the paper's own scale) to see baselines actually
  collapse — toy scale here is not long enough to test the claimed *mechanism*, only the general
  MSE ordering (which does hold: Fern's MSE stays ~1.3-1.5x lower than DLinear's at all 4
  horizons tested).
