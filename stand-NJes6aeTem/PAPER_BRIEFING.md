# STAND: Self-Aware Precondition Induction for Interactive Task Learning — reproduction briefing

Paper: arXiv 2409.07653v2 (revised 2026-02-04), "STAND: Self-Aware Precondition Induction for
Interactive Task Learning," Daniel Weitekamp, Glen Smith, Kenneth Koedinger, Christopher
MacLellan (Georgia Tech / Carnegie Mellon University).
OpenReview id: NJes6aeTem. Local copy: `paper-arxiv-2409.07653v2.pdf` (19 pages, full text +
Appendices A-E read and confirmed complete/legible).

## CRITICAL: claims.json/claims_anchored.json are stale (v1, not v2) — do not use them as-is

Downloaded `claims.json` (3 general claims) and `claims_anchored.json` (5 anchored claims, citing
"Table 1"/"Table 2"/"Table 4") from the challenge Space. Every specific number in the anchored
claims (STAND 98.62% completeness at N=100 MC-addition vs. 96.97%/95.53%/98.01% for
DT/RF/XGBoost; 0.53% total error recurrence; 78.54% productive monotonicity on fractions vs.
50.61%/50.90%; 5.30ms fit / 0.35ms predict under a "10ms threshold") was cross-checked and found to
match **arXiv v1 (Sept 2024)** verbatim — that early single-domain-comparison version's Table 1
(p.14), Table 2 (p.16), Table 4 (p.17), and Section 3.1 fit-time paragraph (p.7), respectively.
None of these tables, numbers, or the "10ms" framing appear anywhere in v2. v2 is a substantially
rewritten paper (added synthetic-data task, hierarchical shrinkage, VSSM/Neural-Net baselines, a
Dice Adventure domain via VAL, redefined "productive monotonicity" using a >2%-change filter, all
new Table/Figure numbering) — same failure mode previously found and reported upstream for
GameDevBench (`Iexhb5lL3t`, see project memory): claim extraction ran against a pre-revision
draft, not the version OpenReview now serves under this id.

**Resolution used here** (consistent with this harness's Step 2 policy of sourcing claims from the
PDF, not the claim-extraction strings): the "Claims in scope" below are transcribed fresh from v2,
not from claims_anchored.json. The stale claims are noted so a reviewer isn't confused about why
this reproduction's headline numbers don't match the challenge site's claim text. Consider filing
an upstream correction (HF discussion on `ICML-2026-agent-repro/challenge`, same pattern as the
GameDevBench post) — not done automatically, ask the user first per standing practice for
external-visible actions.

Challenge: HF Space `ICML-2026-agent-repro/challenge`. Lands in `nmaher2022/icml2026-reproductions`
as `stand-NJes6aeTem/`.

## Working conventions for this reproduction
- Self-contained PEP-723 Python scripts (inline `uv` deps), CPU-only — STAND is a symbolic
  tree/lattice method, no GPU involved anywhere in the paper.
- **Smoketest before scale**: tiny synthetic run (few dozen samples, few features) before the
  paper's full N=100/400-feature/2000-holdout/100-repetition synthetic benchmark.
- All work happens in `stand-NJes6aeTem/`.
- Verdict vocabulary: VERIFIED / TOY-VERIFIED / REFUTED / BLOCKED. State scale run next to every
  verdict. Never round TOY-VERIFIED up to VERIFIED. Report BLOCKED claims explicitly.
- Self-check before finishing: reread each claim's exact v2 wording next to the actual numbers
  produced.

## Claims in scope (verbatim/paraphrased from v2, with section/table/figure refs)

1. **Synthetic-task accuracy**: "across all precondition learning tasks, STAND matches or has
   higher holdout set performance than the comparison models" — on the synthetic task specifically,
   "after N=100 examples, STAND (90.0%) outperforms decision trees (88.6%) and considerably
   outperforms XGBoost (84.8%)... STAND performs even better with hierarchical shrinkage (92.8%)"
   (§6.1, Table 1, Fig. 2 bottom panel). Baselines also include Random Forest (56.2%), Neural Net
   (56.7%), VSSM (55.5%).
2. **Error reoccurrence**: "excluding VSSM..., in the synthetic data, STAND with hierarchical
   shrinkage has the lowest rate of false positive reoccurrence (2.2%), and a low (but not best)
   false negative reoccurrence rate (0.8%)" (§6.2, Table 1).
3. **Productive monotonicity**: "STAND is considerably better than the comparison models... The
   alternatives show rates that are not much higher than 50%, while STAND has rates of 60-70% after
   about 10 examples" (§6.3, Fig. 3 left).
4. **Absolute precision / calibration**: "STAND aligns very near with the grey dotted line marking
   one-to-one alignment with precision and estimated probability... with hierarchical shrinkage,
   STAND's estimates of 100% probability have a nearly perfect precision of 99.71%" (§6.4, Fig. 3
   middle).
5. **Active learning utility**: "STAND with hierarchical shrinkage has the best active learning
   utility in the first 20 problems, with similar utility as XGBoost thereafter" (§6.5, Fig. 3
   right).
6. **Hyperparameter sensitivity (λp)**: "higher λp can benefit accuracy, productive monotonicity,
   and error reoccurrence. The choice of λp = 25.0 retains these benefits while maintaining high
   precision in the neighborhood of prediction probabilities close to 100%" (Appendix D, Fig. 12).
7. **Noisy/UCI benchmark comparability**: "STAND does not fail under noise, and is comparable to
   other tree models in large data scenarios" — Table 2 average accuracy: DecisionTree 85.84%,
   RandomForest 88.22%, XGBoost 88.57% (best), STAND 87.25%, STAND(heir) 86.40%, across 6 UCI
   datasets (breast-cancer, hepatitis, soybean, tic-tac-toe, vote, zoo) with an 80/20 train/test
   split (Appendix E, Table 2/3).
8. **Real ITL-domain claims** (Dice Adventure via VAL; Multi-column Addition and Fractions via
   AI2T/TutorGym, Appendix C): STAND matches/exceeds comparison models' holdout accuracy, reaching
   near-100% accuracy fastest with hierarchical shrinkage (Dice Adventure 99.51%, Fractions 99.88%,
   MC Addition 99.43% at N=25) (§6.1, Fig. 2 top three panels). **Out of scope / BLOCKED up front**:
   requires the authors' proprietary VAL and AI2T/TutorGym ITL tutoring-system infrastructure to
   generate the actual interactive precondition-induction training streams; not available in this
   environment and not something to approximate with a shaky proxy.

## Core math / setup (transcribed from v2, Sections 3-4 + Appendix B)

**STAND's general set G** (§3.1): at each node, instead of picking one best-gain literal split
`t_L = argmax_t F(X_j, Y_k | ...)` (Eq. 1) like a decision tree, STAND accepts the whole set of
near-tied literals `T_L = {t(X_j) : F(X_j,Y_k|...) >= M(1-α)}` where `M` is the max gain (Eq. 2-3),
with `α` tunable and (Appendix A.1) optionally scheduled per-node as
`α_nk = 1 - min((1-α0) + α0(N(k)/M_samples), 1)` (Eq. 13; α0=0.1, M_samples=50 base params — note
this `M` is a *different* symbol than the gain-max `M` in Eq. 2, paper reuses the letter). Node
outgoing edges that select the same sample subset are cached/reused (dedup by sample-index set),
forming a lattice rather than a tree — this is a *performance* optimization (avoids
combinatorial blowup), not required for correctness; a toy implementation may skip the caching and
still compute an equivalent (slower) lattice.

**Specific set S** (§3.2): for each leaf k, `s_k` = the set of categorical feature=value pairs
constant across all training samples filtered into that leaf (plus min/max ranges for continuous
features). Provides additional generalization criteria beyond G's literal chains.

**Certainty** (§3.3, Eq. 4-7): `Cert(y=c|x) = Cert_S(y=c|x) * Agr_G(x)`.
- `Cert_S(y=c|x)` (Eq. 5): weighted-average agreement of leaf specific-extensions with x, per leaf
  k that selects x: numerator sums `w_nk * (w_sk . 1[sk=x])` over leaves selecting x, i.e. leaf
  node-weight times the fraction of that leaf's specific-extension features x actually satisfies;
  denominator sums `w_nk * ||w_sk||_1`.
- `Agr_G(x)` (Eq. 7): weighted fraction of "in-degree opportunities" (literal-gated edges upstream
  of nodes x filters through) that x actually satisfies, weighted by node weight `w_nk`.
- Not normalized across classes (can sum >1 across c) — this is intentional (§3.3), not a bug.

**Hierarchical shrinkage** (§4, Eq. 8-12): standard leaf-to-root shrinkage
(`f_λ(x) = Ê_t0{y} + Σ_l (Ê_tl{y} - Ê_tl-1{y})/(1+λ/N(t_{l-1}))`, Eq. 8, from Agarwal et al. 2022)
applied not to leaf predictions directly but to (a) the joint probabilities `P(c,j)` used in
impurity calculations (Eq. 9, param λp — this is what most Appendix D sensitivity plots vary), (b)
node weights `w_nk` (Eq. 10, param λn, using minimum-acceptance-rate `τ_nk` and node sample count
N(k)), and (c) specific-extension weights `w_sk` via a Beta(1/2,1/2)-prior-adjusted shrunk estimate
of feature invariance probability `P*(s_k|c)` (Eq. 11-12, param λs). Paper's chosen hyperparameters
(tuned via grid search in Appendix D): λp=25, λs=25, λn=50.

**Synthetic data generation procedure** (Appendix B — full recipe used for claims 1-6 above):
1. Feature matrix X: 2100 samples × 400 integer categorical features, `2 + Poisson(1)` values/feature.
2. Target preconditions: `1 + Poisson(1)` disjunctive concepts, each an OR of two conjuncts of
   `1 + Poisson(1)` non-overlapping literals; literal values sampled uniformly.
3. Structured co-occurrence: sample 100 "distractor" conjuncts of `2 + Poisson(3)` literals
   (20% chance of overlapping literals with previously sampled ones); each applies to 80% of
   samples independently, injecting spurious feature correlations (Fig. 5).
4. Each target conjunct applied to 28% of samples (→ ~50% of data positive on average).
5. For negative samples (`Y_j != 1`): apply both target conjuncts, then with 10% probability per
   literal, resample that literal's value (at least one resampled feature guaranteed per negative
   sample) — ensures negatives aren't trivially separable by a partial-conjunct subset.
6. Split: 2100 → 100 train (only 20 negative, ordered so negatives skew earlier in the sequence —
   simulates agents making fewer mistakes as training progresses) + 2000 holdout. Repeat 100x,
   report mean ± std-error/√n.

**Baselines to implement/compare**: Decision Tree (sklearn, gini, unlimited depth — this reproduction
also fits STAND's own "expand just one random best split" DT-equivalent per v1 §5.1 item 1, but v2's
main-body comparison uses standard CART), Random Forest (100 trees, sklearn defaults), XGBoost
(sklearn-default-equivalent gradient boosting), VSSM (incremental version-space method — likely
not implementable from a from-scratch description in-scope; consider BLOCKED or a labeled
placeholder/approximation, decide during implementation), Neural Net (FC 3×100 ReLU, Adam lr=1e-3
— v2 §5 prose says "lrn. rate=1e-3" for one mention; double check against the actual figure, the
paper text has "lrn. rate=1e-3" written in a slightly garbled way — re-verify exact value while
implementing since this is a small detail worth getting right, not blocking).

## Known access blockers
- **VAL and AI2T/TutorGym** (real ITL tutoring-system infrastructure for Dice Adventure, Fractions,
  Multi-column Addition domains) — not available; claim 8 is BLOCKED from the start, not a
  discovered-later blocker. The synthetic task (claims 1-6) is the paper's own designed substitute
  "to avoid ceiling effects among models" and is fully self-contained per Appendix B, so it carries
  the bulk of the reproducible signal.
- VSSM (incremental version-space method, Hong & Tseng 1999) has no public reference
  implementation found yet as of writing this briefing — may end up BLOCKED or approximated;
  decide once implementation starts. Not critical to any single claim above (only appears as one
  baseline row in Table 1, not the focus of any claim 1-6 statement).
