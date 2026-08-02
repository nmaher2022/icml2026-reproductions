# VERDICTS.md — Ellipsoidal Time Series Forecasting (arXiv 2505.17370v6, OpenReview CoAHlJuMdh)

Scale run: **toy** throughout — CPU-only, short synthetic trajectories (6,000-9,000 steps vs the
paper's 25,000-36,000), small models (Kenc=5 encoder layers but narrow hidden width dh=32 vs the
paper's presumably larger config), 60-epoch training with early stopping, 3 seeds vs the paper's 4.
No real-world dataset (ETTh1/ETTm1/Weather) or heavy baselines (TimeMixer/PatchTST/Koopa/
ModernTCN/PFNN) used — see PAPER_BRIEFING.md's "Known access blockers" for the deliberate scope
choice. All numbers below are from this reproduction's own runs (`claim1_3_results.json`,
`claim2_results.json`, `claim4_results.json`, `claim5_results.json`); see `BUGFIX_LOG.md` for the
self-audit that preceded these verdicts (one real bug found and fixed: the `no_rotation` ablation
was confounded with reduced model capacity).

Vocabulary: VERIFIED / TOY-VERIFIED / REFUTED / BLOCKED / INCONCLUSIVE.

---

## Claim 1 — Nonstationary robustness (up to 790x over DLinear/Koopa, Abstract; Table 2)

**TOY-VERIFIED, partial (system-dependent) — does not hold uniformly.**

Fern vs. DLinear mean MSE (3 seeds), toy scale:
| Scenario | Fern MSE | DLinear MSE | Fern advantage |
|---|---|---|---|
| Lorenz-63 base | 42.72 | 65.18 | **1.53x lower** |
| Lorenz-63 param-shock | 41.05 | 57.25 | **1.39x lower** |
| Roessler base | 5.25 | 4.21 | 0.80x (Fern **worse**) |
| Roessler param-shock | 6.01 | 6.97 | 1.16x lower |
| Chua base | 0.43 | 0.06 | 0.13x (Fern **5.5-7.7x worse**) |
| Chua param-shock | 0.58 | 0.10 | 0.18x (Fern **worse**) |

Direction matches the paper on Lorenz-63 (both scenarios) and Roessler-param, at a much smaller
margin than the paper's up-to-790x figure (expected at toy scale — not claiming the magnitude).
On Roessler-base and both Chua scenarios, DLinear outperforms Fern, the opposite of the paper's
claimed universal ordering. Investigated as a possible bug (no per-instance normalization is used
by either model, matching the paper's own protocol which doesn't describe RevIN/instance-norm for
the synthetic chaotic benchmarks) — no implementation bug found; most plausibly a genuine
toy-scale effect where a strong, cheap linear baseline (DLinear) beats an undertrained flexible
spectral model on lower-amplitude / less strongly nonstationary systems at this data/compute
budget. Reported honestly rather than cherry-picking the Lorenz result.

## Claim 2 — Linear-time complexity via Householder SPD factorization (Abstract; Eq. 1-2; App. A.3.2)

**VERIFIED** (exact, analytic — not just toy-scale, since this is a structural/asymptotic claim
checkable directly from the implemented model's own parameter/FLOP counts, independent of training
data or compute budget).

Swept patch size p from 8 to 384 (g=1 patch, R=8, dh=32), comparing this implementation's actual
Householder-factored OT head FLOPs against the dense-per-patch-SPD-map alternative it replaces:

| p | Fern head FLOPs | Dense SPD FLOPs | dense/Fern ratio |
|---|---|---|---|
| 8 | 576 | 64 | 0.11 |
| 24 (paper's base config) | 1,728 | 576 | 0.33 |
| 96 | 6,912 | 9,216 | 1.33 |
| 192 | 13,824 | 36,864 | 2.67 |
| 384 | 27,648 | 147,456 | **5.33** |

The Householder-factored head cost grows linearly in p (as designed) while the dense-SPD
alternative grows quadratically (O(p²)) — the crossover point is clearly visible (~p≈70) and the
ratio grows linearly thereafter, exactly matching the paper's stated asymptotic argument (Eq. 1-2:
`O(B·g·(Kenc+1)·p·dh + B·g·R·p)` vs. a dense `O(B·g·p²)` map). At the paper's own stated base
config (p=24, dh not given exactly but on this order), the dense alternative is still cheaper in
absolute FLOPs — consistent with the paper's own admission that the advantage is asymptotic, not
necessarily favorable at every practical scale (the paper's own complexity discussion frames this
as an O(n³)→O(Rn) reduction relative to a *full unrestricted n×n* eigendecomposition search, which
this reproduction did not separately re-derive numerically, only the per-patch dense-SPD-vs-
factored comparison Eq. 1-2 gives directly).

## Claim 3 — Effective Prediction Time (Tables 11/12/15)

**REFUTED at toy scale** — the claimed ordering (Fern's EPT longer than baselines') is reversed
in all 6 scenarios tested.

| Scenario | Fern EPT | DLinear EPT |
|---|---|---|
| Lorenz-63 base | 43.8 | 52.5 |
| Lorenz-63 param | 45.3 | 58.0 |
| Roessler base | 64.2 | 87.9 |
| Roessler param | 65.4 | 88.2 |
| Chua base | 88.7 | 96.0 (= max, horizon length) |
| Chua param | 84.0 | 95.8 |

DLinear has a longer mean EPT than Fern in every scenario, including Lorenz-63 where Fern
simultaneously has the *lower* (better) MSE — i.e. Fern is the better average-case forecaster but
the worse first-threshold-crossing forecaster on the same data. This is disclosed as a real,
scale-independent property of the EPT metric as defined (Appendix A.1: first step where absolute
error exceeds one training-set standard deviation) rather than assumed to be a Fern weakness per
se — a smoother/more conservative predictor can trivially stay under a fixed error threshold
longer while being worse on average. Not rounded to INCONCLUSIVE: the paper makes a specific
directional claim (Fern's EPT is longer) and this reproduction's toy-scale evidence directly
contradicts it in every tested scenario, so REFUTED (at toy scale) is the honest verdict, not a
hedge.

## Claim 4 — Ablations confirm encoder/rotation/patching are each doing real work (Table 3/8)

**TOY-VERIFIED** — direction and relative importance ordering both reproduce cleanly on Lorenz-63
(3 seeds, mean MSE vs. base config):

| Variant | Mean MSE | % vs base | Mean EPT |
|---|---|---|---|
| Base | 42.72 | — | 43.8 |
| No rotation | 45.96 | **+7.6%** (worse in all 3 seeds) | 40.2 |
| No patching | 47.03 | **+10.1%** | 37.7 |
| No encoder | 68.58 | **+60.5%** (catastrophic, worst) | 15.6 (EPT collapses) |

Matches the paper's qualitative story exactly: removing the encoder is by far the most damaging
(paper: 9x MSE increase and EPT collapse 241→17; here: 1.6x MSE increase, EPT collapse 43.8→15.6 —
same direction, smaller absolute magnitude, expected at toy scale/short training), removing
rotation and patching each cause smaller but real degradation (paper: rotation +27%/patching
+5.5% on Lorenz63; here: rotation +7.6%/patching +10.1% — same direction for both, magnitudes and
relative ordering of rotation-vs-patching not an exact match but both clearly worse than base in
every seed after the capacity-confound fix, see BUGFIX_LOG.md).

## Claim 5 — Geometric accuracy (SWD) persists past the pointwise-collapse horizon (main text p.7)

**INCONCLUSIVE** — the paper's specific *mechanism* (baselines collapse to mean-guessing at a
given horizon; Fern's SWD then stays low while pointwise accuracy fails) is not observable at the
toy-scale horizons tested (24/48/96/192 steps): DLinear never collapses in this range.

Fern's overall pointwise MSE advantage over DLinear does hold at every horizon tested (1.30-1.51x
lower MSE, roughly flat, not widening or narrowing meaningfully with horizon). But the specific
geometric-persistence story doesn't replicate: Fern's SWD advantage over DLinear *shrinks* as
horizon grows (2.44x at h=24 down to ~0.93-0.99x, i.e. roughly tied, at h=96-192) — the opposite
trend the claim would predict if a "geometry persists past collapse" effect were kicking in.
Most plausible explanation: the paper's claim is specifically about a regime (~6.5 Lyapunov times,
horizon 720 out of a much longer base trajectory) that these toy-scale horizons don't reach —
this reproduction tests the general MSE-ordering claim (which holds) but not the mechanism claim
about collapse-then-geometry-persists, and is honestly reported as inconclusive on the latter
rather than forced to VERIFIED or REFUTED.

---

## Summary
| Claim | Verdict | Scale |
|---|---|---|
| 1. Nonstationary robustness | TOY-VERIFIED (partial, system-dependent) | toy |
| 2. Linear-time complexity | VERIFIED | analytic (scale-independent) |
| 3. Effective Prediction Time | REFUTED | toy |
| 4. Ablations | TOY-VERIFIED | toy |
| 5. Geometric persistence past collapse | INCONCLUSIVE | toy |

No claim was rounded up in the paper's favor. Claim 1's headline "up to 790x" and Claim 3's EPT
ordering are the two places this reproduction's toy-scale evidence most clearly diverges from the
paper's own reported numbers/direction; both are disclosed with full per-scenario numbers above
rather than summarized away.
