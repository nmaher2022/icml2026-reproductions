# D&L reproduction: bug-fix log

Running record of every implementation bug found in `dl/core.py` (and
related experiment scripts) during this reproduction, what was fixed, and
what changed in the results. Kept so later investigation doesn't repeat
work or lose track of what's already been ruled out.

## Round 1 (initial implementation pass, 2026-07-28/29)

1. **`dual_update` stuck at λ=0 forever.** The multiplicative dual update
   had a fixed point at λ=0 that it never escaped, so Lagrangian
   coordination was silently inert from the start.
   Fix: corrected the update rule. Result: Claim 2's coupling-error
   coordination effect went from ~0 (mechanism dead) to a real 10-30x
   reduction at every K — this fix is confirmed working and not revisited.

2. **Coordination signal computed but never fed back into the reward.**
   `compute_xi`/`dual_update`/λ only ever fed a `coupling_err`
   *measurement* trace, never `select_action` or the reward credited in
   `update_experts` — so even with (1) fixed, coordination could not
   possibly change outcomes.
   Fix: `update_experts` now accepts a per-position penalty
   λᵢ(t−1)·ξᵢ(t−1) computed causally before crediting reward. Result:
   Claim 3b/3c showed a real (if modest) coordination effect for the
   first time.

3. **`LocalRefine` invoked K times per outer iteration instead of once.**
   Multiplied its oracle-call cost by K, inflating D&L's own reported
   compute budget.
   Fix: call once per outer iteration. Confirmed via eval-count
   bookkeeping.

4. **Eval counter double-counted bookkeeping calls.** `reward_fn.expected()`
   — a noiseless-regret-trace convenience for Claims 1/2, never queried by
   a real deployment — was being counted as a real oracle query.
   Fix: excluded from the eval counter.
   Combined effect of (3)+(4): overstated D&L's per-weight compute budget
   by ~2.6x (360→140 oracle calls/weight at T=100, K=4). This is the state
   the logbook was published with as of 2026-07-29 morning.

## Round 2 (this session, 2026-07-29 afternoon)

Found by a dedicated code-review agent tasked with explaining why Claims
4-6 (D&L vs baselines on MOCO + HW-SW) still don't replicate even after
Round 1's fixes.

5. **FTRL exploration bonus grew with visit count instead of shrinking.**
   `bonus = gamma_ftrl * sqrt(N)` (larger N → larger bonus, the opposite
   of an exploration bonus) instead of a UCB-style
   `gamma_ftrl * sqrt(log t / N)`. This is anti-exploration lock-in: the
   more a position-action pair is tried, the *more* attractive FTRL made
   it look, regardless of payoff.
   Fix: `bonus = gamma_ftrl * sqrt(log(max(t,2)) / max(N, 1e-6))`.

6. **EXP3/FTRL updates used the raw reward `r` instead of the
   Lagrangian-penalized `r_i`.** Even after Round 1 fix #2 made
   `update_experts` receive `r_i`, the EXP3 `reward_hat` and FTRL
   `loss_hat` computations still referenced the un-penalized `r` — so the
   coordination signal reached UCB/TS's value estimate but never reached
   EXP3/FTRL's weight/loss updates.
   Fix: both now use `r_i` consistently.

7. **Ablation branch routing bug (`claim3_ablation.py` / `run_dl`'s
   `experts=` override).** Single-expert-only evaluation mode only had
   explicit branches for `"ucb"` and `"ftrl"`; `"exp3"` silently fell
   through to the FTRL branch (`rho0=1.0` forces `e==2`), and `"ts"` was
   silently contaminated by EXP3 selection (`rho0=0.0` allows `e` in
   `{0,1}`, only `e==0` used TS). **This means every previously-published
   "EXP3-only" ablation number was actually re-measuring the FTRL branch
   under a different label** — Claim 3's original write-up's specific
   claim "EXP3-only 0.810, FTRL-only 0.808" was the same branch measured
   twice, not two different experts.
   Fix: added explicit UCB/EXP3/TS/FTRL branches, each using the expert's
   own real selection rule.

### Round 2 results

Re-ran Claims 1-6 in full (`rerun_all.sh`) after fixes 5-7.

- **Claims 1-2** (regret scaling, coupling-error coordination):
  qualitatively unchanged — regret exponent still ≈0.99 (near-linear, not
  the paper's ~0.5-0.6), coordination still cuts coupling error 10-30x
  across K. No numeric surprises.
- **Claim 3 ablation** (`results/claim3_ablation.csv`): numbers changed
  substantially now that EXP3/TS are evaluated correctly instead of being
  silent FTRL aliases —
  `all=0.6699, all-TS=0.6070, UCB-only=0.6899, EXP3-only=0.6421,
  FTRL-only=0.6703, TS-only=0.6211` (fraction of optimum, n reps as in
  `experiments/claim3_ablation.py`). **UCB-only is now the best single
  expert**, not EXP3/FTRL as previously (mis-)reported. The mixture still
  does not beat every single expert (UCB-only > mixture). Coordination's
  effect on 3b (light overlap) is now essentially a wash / slightly
  negative (0.6810 coordination-on vs 0.6830 off — reversed sign from the
  previous 0.662 vs 0.659, but both differences are well within noise at
  n=6 reps); 3c (heavy overlap, more reps) still shows coordination
  winning (0.6754 vs 0.6717), same direction as before.
- **Claims 4-5** (MOCO: Bi-Knapsack n=50/100, Bi-TSP n=20/50, full 15-seed
  run, `results/claim4_5_moco.csv`): **ranking unchanged.** D&L/D&L-TS
  still last of 5 methods (BO, NSGA-II, WS-heuristic beat both) on every
  domain/size, e.g. Bi-KP-50 hv_ratio: D&L 0.336, D&L-TS 0.291 vs BO
  0.537, NSGA-II 0.911, WS-heuristic 0.944. Still uses ~4.2x more oracle
  evals than BO (1680 vs 396) and runs 10-20x slower. Numbers moved by a
  few percentage points from the Round-1-published table but the
  qualitative story and full ranking are identical.
- **Claim 6** (HW-SW proxy, full 150-eval/10-seed budget,
  `results/claim6_hwsw_proxy.csv`): **ranking unchanged** — D&L-TS still
  worst of 3 (0.136 vs BO-analogue 0.178 vs NSGA-II 0.236), reversing the
  paper's claimed +22% D&L-TS-over-BO advantage into a ~24% deficit.
- **Notable new observation**: at *tiny smoke-test scale* (Bi-KP n=8,
  `run_moco.py --smoke`), the Round-2 fixes make a dramatic difference —
  D&L-TS becomes competitive/tied with the WS-heuristic baseline
  (0.8373 vs 0.8373) and close to BO/NSGA-II (0.8653/0.8889), a world away
  from "last of 5." This gap does **not** persist at the realistic problem
  sizes actually used for Claims 4-6 (n=50/100, T=150) — whatever helps at
  n=8 stops helping (or reverses) as n grows. Not yet explained; worth
  investigating directly (see open questions below).

**Conclusion after two fix rounds: the empirical falsification of Claims
4-6 is not attributable to bugs 1-7.** All seven were real, independently
confirmed bugs with clear before/after evidence, but none of them close
the gap between D&L's reported hypervolume and the baselines' at the
problem sizes the claims actually specify.

## Round 3 (2026-07-29, dedicated deep-dive investigation agent)

Dispatched specifically to explain the still-unexplained smoke-vs-full-scale
gap (Round 2's "open questions" below). Found **three more genuinely new
issues** — two clean bugs and one methodological asymmetry — plus confirmed
one of Round 2's open questions as a real structural property (not a bug).

8. **`run_dl` returned the last round's `x`, not the paper's tracked
   best-ever `x*`.** Algorithm 1 in the paper (arXiv 2602.11346) explicitly
   tracks a running incumbent (`if r^(t) > r* then x*<-x^(t)`) and *returns
   x\**, not whatever the mixture-of-experts happened to sample in the
   final round. Our `run_dl` never did this. Measured gap (Bi-Knapsack,
   noiseless expected reward, 6 seeds): last-round-x scored 0.0495-0.0977
   *lower* than the best-tracked x at every n/T tested (n=8..100).
   This directly explains a previously-unexplained anomaly from Round 2:
   Claim 3's "mixture < UCB-only" — by late rounds of a T=500 run, the
   mixture's weight has shifted toward EXP3 (~82% of position choices by
   the paper's own `u_i` formula), so the *mixture's* last round is
   disproportionately an EXP3 (randomized) sample, while "UCB-only" mode's
   last round is always the converged UCB argmax. The mixture was being
   penalized by the readout bug, not by its design.
   Fix: track `(best_r, best_x)` inside `run_dl`'s main loop using the
   realized reward `r` (same criterion the paper uses), return `best_x` as
   `final_x`. Affects Claim 3 (`claim3_ablation.py`) and Claims 4-5
   (`run_moco.py`), both of which read `out["final_x"]`. Does **not**
   affect Claim 6, whose hypervolume is computed from `problem.history`
   (every oracle call across the run), never `final_x`.

9. **MOCO baselines (Claims 4-5) searched against the noiseless objective;
   D&L was the only method fed noisy feedback.** All three baselines
   (`ws_heuristic_knapsack/tsp`, `nsga2_knapsack/tsp`, `bo_knapsack/tsp` in
   `moco_baselines.py`) computed their accept/reject or fitness-selection
   scores directly from raw instance data (`inst.profit1[sel].sum()`,
   `tour_length(...)`), completely bypassing the `noise_sigma=0.02` Gaussian
   noise that `KnapsackScalarizedReward.__call__`/`TSPScalarizedReward.__call__`
   inject into every reward D&L's search sees. This is an asymmetry that
   gets *worse* as n grows: at n=8 the injected noise is a modest fraction
   of true signal, but per Finding 10 below the true per-position signal
   itself shrinks with n while the noise floor stays fixed, so the fraction
   of the baselines' unfair advantage grows with n too.
   Fix: every baseline's search/selection loop now scores candidates
   through a noise-injected score (same `noise_sigma=0.02`, same
   normalization convention D&L's own reward functions use — fresh
   Gaussian draw per query, not a persistent per-x noise value). The final
   *archived* point for hypervolume scoring is still each baseline's TRUE,
   noiseless objective at whichever x its noisy search selected as best —
   exactly mirroring how D&L's `final_x` (post fix 8) is scored by the true
   objective of the best-by-noisy-reward point it found. Does not apply to
   Claim 6 (its harness already fed all three methods the same noisy
   `problem.evaluate(x, noise_rng)`).

10. **HW-SW proxy (Claim 6) resampled its Tchebycheff scalarization weight
    on every single oracle call**, not once per run. `TchebycheffDLAdapter`
    drew `w ~ Dirichlet(1,1,1,1)` fresh inside `__call__`/`.expected()`
    every call — fine for the BO-analogue (which refits its GP from scratch
    every iteration, so no stale cross-weight state survives) and for
    NSGA-II (never scalarizes at all), but D&L's `state.V[i,a]` is a
    cumulative running (Welford) mean over rounds, which implicitly assumes
    a *stationary* reward target for each (position, action) pair. With `w`
    re-randomized every call, the "true" target for the same (i, a) pair
    changes randomly round to round, so the running average silently mixed
    rewards computed against incompatible objectives — breaking every
    expert's (UCB, EXP3, FTRL, TS) core assumption. Mechanism match to the
    evidence: n is fixed at 20 for *both* the T=20 and T=150 HW-SW runs
    (ruling out an n-scaling explanation), and D&L-TS's win-at-T=20 /
    loss-at-T=150 reversal tracks exactly what you'd expect as more
    inconsistent-target samples accumulate into the running averages over a
    longer run.
    Fix: `TchebycheffDLAdapter` now takes a `w` fixed at construction
    (no more per-call resampling); `claim6_hwsw_proxy.py`'s `run_dl_ts`
    constructs one adapter per outer scalarization weight (10 weights,
    budget split evenly across them: `T_per_weight = T // 10`) and calls
    `run_dl` once per weight — mirroring the same "one run_dl call per
    scalarization weight" pattern Claims 4-5 already use in `run_moco.py`.
    `z_star` (the online ideal-point estimate) is carried forward across
    weights since it only grows and represents knowledge accumulated over
    the whole experiment, not just one weight's sub-run.

**Finding 11 (confirmed structural limitation, NOT a bug — answers Round
2's first open question):** per-position bandit credit assignment's
signal-to-noise ratio collapses as n grows. D&L's `update_experts` credits
one shared noisy scalar reward to every position's per-action value
estimate each round. Measured on Bi-Knapsack (30 seeds): mean
|single-item-flip Δreward| / `noise_sigma` (SNR) = 2.89 at n=8, 0.61 at
n=50, 0.39 at n=100 — 100% of single-item decisions are below the noise
floor at n=100. Mechanism: `KnapsackScalarizedReward`'s normalization
divides by a sum that scales ~linearly with n, so one item's marginal
contribution to the *normalized* reward is O(1/n), while `noise_sigma=0.02`
is a fixed constant regardless of n. GP-based BO and population-based
NSGA-II don't rely on this per-position marginal decomposition, so they
aren't hit the same way. This is a genuine property of full-bandit
position-wise credit assignment on additive-objective domains at scale,
compounded by (not solely caused by) the paper-unspecified, somewhat
arbitrary choice of a fixed `noise_sigma=0.02` — worth reporting honestly
in the logbook as a scaling limitation rather than an implementation error.

### Round 3 results

Fixes for findings 8-10 applied to `dl/core.py`, `dl/hwsw_proxy.py`,
`experiments/moco_baselines.py`, `experiments/run_moco.py`,
`experiments/claim6_hwsw_proxy.py`. Re-ran Claim 3 (`claim3_ablation.py`)
and Claim 6 (`claim6_hwsw_proxy.py`) in full; Claims 4-5
(`run_moco.py`, full — 15 instances x 4 sizes x 5 methods) is the largest
re-run and is running separately to avoid the CPU contention that produced
two false-timeout scares earlier in this round (the BO baseline's GP fits
alone take ~50s/seed at full HW-SW scale; artificially short `timeout`
values combined with 3 CPU-heavy jobs sharing 8 cores made two genuinely-
still-running processes look dead — always cross-check result-file mtimes
against a command's reported exit status before trusting either alone).

**Claim 3 ablation** (`results/claim3_ablation.csv`), Round 2 -> Round 3:

| config | Round 2 | Round 3 |
|---|---|---|
| all (mixture) | 0.6699 | 0.7110 |
| all-TS | 0.6070 | 0.6729 |
| UCB-only | 0.6899 | 0.8810 |
| EXP3-only | 0.6421 | 0.6455 |
| FTRL-only | 0.6703 | 0.8553 |
| TS-only | 0.6211 | 0.7084 |

UCB-only and FTRL-only jumped the most (~0.19 and ~0.19 absolute) since
both converge to a stable argmax by late rounds, so "last round" vs "best
round" used to matter a lot for them; EXP3-only barely moved (~0.004,
stays randomized throughout, so last-vs-best matters little). **The
mixture still trails UCB-only** (0.7110 vs 0.8810) — finding 8's fix does
not resolve the "mixture doesn't beat every single expert" anomaly, it
just changes the numbers it's measured on; this now looks like a genuine
property of the mixture-weight formula's late-run EXP3-skew rather than an
artifact of the old readout bug.

3b (light overlap, coordination on/off): 0.7152 vs 0.7200 — a wash,
consistent with Round 2 (both differences well within noise at n=6 reps).

3c (heavy overlap, more reps) — **this is a new reversal**: Round 2 showed
coordination *winning* under heavy overlap (0.6754 vs 0.6717), which was
written up as support for Theorem 4.5's overlap-severity prediction. Round
3 shows coordination *losing* (0.7554 vs 0.7706 favoring coordination off).
Since 3c's only change this round was the same best-x readout fix applied
everywhere else, the Round 2 "coordination helps more under heavy overlap"
result now looks like it was an artifact of the old last-round-x bug
interacting favorably with coordination, not a real Theorem-4.5 effect —
**this further falsifies the Lagrangian-coordination half of Claim 3**, not
just the multi-expert-mixture half.

**Claim 6 HW-SW proxy** (`results/claim6_hwsw_proxy.csv`), Round 2 -> Round 3:

| budget | method | Round 2 | Round 3 |
|---|---|---|---|
| T=20 | D&L-TS | 0.378 (1st) | 0.332 (2nd) |
| T=20 | BO-analogue | 0.319 | 0.346 (1st) |
| T=20 | NSGA-II | 0.291 | 0.301 (3rd) |
| T=150 | D&L-TS | 0.136 (3rd/last) | 0.147 (3rd/last) |
| T=150 | BO-analogue | 0.178 | 0.181 |
| T=150 | NSGA-II | 0.236 (1st) | 0.239 (1st) |

Fixing finding 10 removed the confusing win-at-T=20/lose-at-T=150
*reversal* — but not in D&L-TS's favor: it no longer wins at T=20 either
(BO does now), and it's still last of 3 at T=150, the budget the paper's
Table 2 actually specifies. **This is a cleaner, more robust negative
result than Round 2's**, not a mitigating one: there's no longer a
budget-dependent flip that needs a separate explanation, D&L-TS is simply
behind at both tested budgets once the stationarity bug is fixed.

**Claims 4-5 MOCO** (`results/claim4_5_moco.csv`, full run — 15 instances x
4 sizes x 5 methods, 300 rows, 2222s total wallclock), Round 2 -> Round 3:

| problem | size | method | Round 2 hv_ratio | Round 3 hv_ratio |
|---|---|---|---|---|
| Bi-KP | 50 | D&L | 0.336 (4th/5) | 0.4834 (**2nd/5**) |
| Bi-KP | 50 | D&L-TS | 0.291 (5th/5) | 0.3972 (4th/5) |
| Bi-KP | 50 | BO | 0.537 | 0.4449 (3rd/5) |
| Bi-KP | 50 | NSGA-II | 0.911 | 0.8188 |
| Bi-KP | 50 | WS-heuristic | 0.944 | 0.9082 |
| Bi-KP | 100 | D&L | -- | 0.2943 (**2nd/5**) |
| Bi-KP | 100 | D&L-TS | -- | 0.1847 (4th/5) |
| Bi-KP | 100 | BO | -- | 0.1943 (3rd/5) |
| Bi-KP | 100 | NSGA-II | -- | 0.6815 |
| Bi-KP | 100 | WS-heuristic | -- | 0.8913 |
| Bi-TSP | 20 | D&L | -- | 0.1983 (4th/5) |
| Bi-TSP | 20 | D&L-TS | -- | 0.1991 (3rd/5) |
| Bi-TSP | 20 | BO | -- | 0.2605 (2nd/5) |
| Bi-TSP | 20 | NSGA-II | -- | 0.5518 |
| Bi-TSP | 20 | WS-heuristic | -- | 0.7842 |
| Bi-TSP | 50 | D&L | -- | 0.0573 (5th/5) |
| Bi-TSP | 50 | D&L-TS | -- | 0.0603 (4th/5) |
| Bi-TSP | 50 | BO | -- | 0.0834 (3rd/5) |
| Bi-TSP | 50 | NSGA-II | -- | 0.2090 |
| Bi-TSP | 50 | WS-heuristic | -- | 0.8055 |

(Round 2 only reported Bi-KP-50 explicitly in the log; the qualitative
statement "D&L/D&L-TS last of 5 on every domain/size" covered all four
rows, so `--` marks cells not individually transcribed at the time.)

Fixing findings 8-9 (best-x readout + fair-noise baselines) produces a real,
qualitative change on **Bi-Knapsack only**: D&L (non-TS) now *beats BO* at
both n=50 (0.4834 vs 0.4449) and n=100 (0.2943 vs 0.1943), moving from last
of 5 to 2nd of 5. This is not noise — the finding-8 fix alone (best-tracked
x instead of last-round x) plus finding-9's fair-noise baselines together
close a gap that two rounds of other bug fixes did not touch. **On
Bi-TSP, the ranking does not flip**: D&L and D&L-TS remain behind BO at
both n=20 and n=50, and all three (D&L, D&L-TS, BO) remain far behind
NSGA-II and WS-heuristic everywhere.

Two further observations:
- **D&L-TS is now consistently worse than plain D&L** (Thompson Sampling
  expert included in the mixture) in every one of the four domain/size
  rows — a new, consistent pattern that wasn't visible in Round 2's TS-vs-
  non-TS comparison being confounded by the last-round-x bug.
- Evals/wallclock cost story is unchanged: D&L/D&L-TS still use ~4.2x more
  oracle evals than BO (1680 vs 396) and run 5-10x slower in wallclock
  (13-27s vs 2-3s), while WS-heuristic and NSGA-II dominate both hypervolume
  *and* cost. So even where D&L now beats BO on quality (Bi-KP), it does
  not do so more cheaply — directly contradicting the "90-99% less
  computation... 10-30x wall-clock speedups" half of Claim 4 regardless of
  the hypervolume outcome.

**Net effect on Claims 4-5**: the specific "D&L beats specialized solvers
by 80-98% of their hypervolume, while beating BO on computation" claim
remains falsified — WS-heuristic (our specialized-solver proxy) beats D&L
by a wide margin on every domain/size, and D&L is *more* expensive than BO,
not less, everywhere. But the picture is no longer uniformly "D&L loses to
everything": on Bi-Knapsack, three fix rounds' worth of genuine bugs were
suppressing a real quality advantage over the BO baseline specifically.
This is worth stating precisely in the eventual logbook writeup rather than
collapsing to a single "falsified" verdict for both sub-claims.

Holding all logbook updates per
the user's explicit "triple check before finalizing a falsification
verdict" instruction (this reproduction is a candidate for the $500 Best
Falsification prize category, which requires the negative result to
survive genuine adversarial scrutiny, not just two fix rounds).

### Intermediate-n sweep (n=16/32, `results/claim_midscale_sweep.csv`)

Dispatched to distinguish two explanations for the Round 3 finding that
D&L beats BO on Bi-Knapsack (n=50/100) but not Bi-TSP (n=20/50): (a) a
smooth SNR-collapse trend common to both domains, vs. (b) a hard
domain-structural discontinuity. 10 instances/cell, same D&L budget
(T=100, K=4) and baseline settings as the full run, so directly comparable
to `results/claim4_5_moco.csv`.

D&L-minus-BO hv_ratio gap by n (positive = D&L ahead):

| n | Bi-KP gap | Bi-TSP gap |
|---|---|---|
| 16 | -0.0136 | -0.0559 |
| 20 | -- | -0.0622 |
| 32 | +0.0081 | -0.0324 |
| 50 | +0.0385 | -0.0261 |
| 100 | +0.1000 | -- |

**Result: smooth trend on both domains, no discontinuity** — ruling out
explanation (b) in its strong form (there's no undiscovered separate bug
causing a sharp TSP-specific cliff). But the trends aren't identical:
Bi-KP's gap crosses zero between n=16 and n=32 and keeps growing through
n=100; Bi-TSP's gap also shrinks with n (roughly halves from n=20 to n=50)
but never crosses zero in the tested range. Most likely mechanism: BO's
evaluation budget is fixed (8 init + 25 iter = 33 evals) regardless of n,
so its GP surrogate degrades as raw dimensionality grows, in both domains
— but Bi-Knapsack's action space is fixed at 2 choices/position regardless
of n while Bi-TSP's grows to n choices/position, so D&L's own per-position
learning problem also gets proportionally harder on TSP as n grows,
partially offsetting its relative gain over BO. Not directly confirmed
(would require an ablation holding TSP's per-position action count fixed
independent of n) — flagged as a residual open question, not chased
further given diminishing returns for this reproduction's scope.

**Practical implication for the Claims 4-5 verdict**: this is not evidence
of a remaining bug. It's a genuine, moderately interesting positive result
(D&L reliably beats a fixed-budget BO baseline on binary/knapsack-style
problems once problem size is large enough) nested inside an overall
negative result (D&L never approaches the specialized-solver/WS-heuristic
baseline's hypervolume on any domain/size, and is more expensive than BO
throughout, contradicting Claim 4's compute-savings claim even where its
quality is competitive or better).

## Open questions for the next investigation pass

- **[ANSWERED, Round 3]** With findings 8-10 fixed, does the Claims 4-5
  ranking against baselines change at all at full scale (n=50/100)? —
  Partially. On Bi-Knapsack, D&L now beats BO at both n=50 and n=100
  (moves from last-of-5 to 2nd-of-5); on Bi-TSP the ranking is unchanged
  (D&L still behind BO at both n=20/50). WS-heuristic/NSGA-II beat D&L on
  every domain/size regardless. Finding 11's SNR-collapse still appears to
  dominate on TSP but not on Knapsack — an asymmetry not yet explained (see
  new open question below).
- **[ANSWERED, Round 3]** Does fixing finding 10 (stationary per-weight
  scalarization) change Claim 6's T=150 ranking? — No: it removes the
  confusing T=20-vs-T=150 win/lose reversal, but D&L-TS remains last of 3
  at T=150 (0.147 vs BO 0.181, NSGA-II 0.239), now also losing at T=20
  (which it had won pre-fix). A cleaner negative result, not a rescued one.
- **[ANSWERED, intermediate-n sweep]** Why does the finding 8+9 fix close
  the D&L-vs-BO gap on Bi-Knapsack but not on Bi-TSP, and is the underlying
  n-dependence a smooth trend or a discontinuity? See "Intermediate-n
  sweep" section below — it's a smooth trend on **both** domains (rules out
  a hard discontinuity / separate undiscovered bug), but Knapsack crosses
  over BO much earlier (between n=16 and n=32) than TSP's trend would
  project, most plausibly because TSP's per-position action space grows
  with n (`num_actions=n`) while Knapsack's stays fixed at 2 — not yet
  confirmed directly (would need e.g. an ablation that pins TSP's action
  space size independent of n, out of scope for this reproduction).
- Are `rho0`, `gamma_ftrl`, `c_ucb`, `eta_exp3` (none pinned down
  numerically by the paper) tuned/scaled sensibly as n and K grow, or are
  they fixed constants that only happen to work at toy scale?
- Is `LocalRefine`'s trigger frequency/step size appropriate at scale
  (Round 1 fixed *how often* it's counted for compute, not *whether* its
  own zeroth-order refinement logic is scale-appropriate)?

## Primary-source verification pass (2026-07-29, direct read of arXiv:2602.11346)

Downloaded and read the actual paper (101 pages incl. appendix, confirmed via
`pdfinfo`; do not trust `file`'s page count on this PDF, it misreports 12) to
check all six challenge claims against the paper's own text/theorems/tables
directly, rather than against the challenge's paraphrase or this log's prior
characterizations. `pdftotext -layout -f 12 -l 12` was used to get an exact,
verifiable token-level parse of Table 1 rather than reading a rendered image
(a 30-column table is not reliable to eyeball).

**Claims 1-4: faithful to the paper.** The regret bound, coupling-error
bound, Algorithm 1's multi-expert/Lagrangian structure, and Table 1's
80-98%/90-99%/10-30x figures all match the challenge's paraphrase of the
paper's own stated results. This log's Round 1-3 empirical falsification of
Claims 3-4 stands as a finding about *this reproduction's implementation*
vs. the paper's claims, not a misreading of what the paper claims.

**Claim 5: this log/logbook previously mischaracterized D&L's own ranking.**
The Claim 5 trackio page (`.trackio/logbook/pages/claim-5-.../page.md`)
said D&L's Table 1 numbers make it "the lowest of the six methods shown, not
the highest." Checked against the paper's actual Table 1 row (10 methods
total, not 6): D&L (0.40) and D&L-TS (0.47) are **5th and 4th of 10**,
beating PMOCO\* (0.309), NSGA-II (0.294), qParEGO (0.104), qNEHVI (0.083),
and PR (0.07) — only WS-\* (0.69), PMOCO† (0.67), and PPLS/D-C (0.63) beat
them. Fixed directly in the trackio page (see its "Correction" paragraph).
This does not change the claim-as-given's falsification verdict (the claim
still misattributes 0.69/0.67/0.63 to the wrong methods) — it only corrects
an overstatement of how badly D&L itself does on this row.

**Claim 6: a previously-undocumented "22%" misattribution, distinct from
this log's own earlier (also imprecise) "+22% D&L-TS-over-BO" phrasing at
line ~108 above.** The paper's own text (p.13) says D&L-TS achieves
"~22% improvement over baselines on average" on the 4-objective HW-SW task.
Checking the underlying Table 2 numbers: the three baselines are NSGA-II
(0.291), qNEHVI (0.287), and qParEGO (0.34) → mean 0.306; D&L-TS is 0.372.
(0.372-0.306)/0.306 ≈ 21.6% — so "22%" is genuinely an *average-over-three-
baselines* figure, correctly computed by the paper. The challenge's Claim 6
text, however, singles out D&L-TS "versus 0.34 for MOBO-qParEGO... roughly a
22% improvement" — i.e. it presents the 22% as if it were the D&L-TS-vs-
qParEGO *pairwise* gap. The actual pairwise gap is (0.372-0.34)/0.34 ≈ 9.4%,
less than half the stated figure. This is a genuine misattribution in the
challenge's claim text (not a paper error, and not one of this
reproduction's bugs) and should be flagged as such in the Claim 6 writeup:
the paper's headline number is real and correctly computed, but it is a
3-baseline average, not the specific qParEGO comparison the claim implies.
Separately, this reproduction's own from-scratch HW-SW proxy (Round 3,
`results/claim6_hwsw_proxy.csv`) falsifies the *qualitative* direction
regardless of which percentage is used: D&L-TS is last of 3 methods at
T=150 in our reconstruction, not ahead of any baseline by any percentage.
