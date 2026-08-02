# Bugfix log — STAND reproduction (NJes6aeTem)

Bugs found and fixed during implementation/self-audit of `data_gen.py` and `stand_lib.py`,
in the order they were found.

## 1. `data_gen.py`: distractor conjuncts overlapped with target literals instead of prior
   distractor literals

**Symptom**: holdout positive fraction ≈ 1.0 for every seed (synthetic data unusable — nearly
every sample satisfied the target concept).

**Root cause**: Appendix B says distractor conjuncts' literals "overlap with previously sampled
literals 20% of the time." Misread on first pass as referring to the *target* concept's literals
(`used_literals=all_target_literals`), which flooded the data with the target concept via spurious
correlation. Correct reading: "previously sampled" refers to literals from *earlier distractor
conjuncts in the same generation loop* — this builds realistic co-occurring feature clusters
(Fig. 5) that are unrelated to the target label, which is the actual point of the distractor step.

**Fix**: thread a `sampled_distractor_literals` accumulator through the distractor loop and pass
it as `used_literals` instead of the target literals. Verified: train is exactly 100/20-negative
per paper spec, holdout pos_frac now varies ~0.25-0.48 by seed (consistent with the paper's ~50%
average across variable disjunct counts per rep).

## 2. `stand_lib.py` `_gain()`: asymmetric hierarchical-shrinkage calculation

**Symptom**: (not directly observed in isolation — subsumed by bug 3 below, found while fixing it)

**Root cause**: the child-impurity closure inside `_gain()` called `_shrunk_joint_prob(node,
feature, value, cls)` — a helper with no `accept`/reject distinction — and used the *same* value
for both the accept-branch and reject-branch impurity calculation, while nominally treating them
as different branches. This made shrinkage's effect on split selection incoherent.

**Fix**: added an explicit `accept: bool` parameter to `_shrunk_joint_prob` so it computes
`P(y=cls, X_feature==value)` or `P(y=cls, X_feature!=value)` depending on which branch is being
evaluated, and added `_shrunk_child_impurity(node, feature, value, accept)` which computes both
classes' shrunk joint probabilities, renormalizes, and returns the branch's impurity. `_gain` now
calls this symmetrically for both branches.

## 3. `stand_lib.py` `_ancestor_nodes_and_literals()` / `certainty()`: Agr_G(x) collapsing to
   exactly 0 for reject-branch traversals through unambiguous (single-split) nodes

**Symptom**: with `hierarchical_shrinkage=True`, STAND degenerated to a trivial 3-node tree
(root splits once, two pure leaves) and predicted the training set's majority class for
*every* holdout example, i.e. holdout accuracy exactly equal to the raw positive fraction of
the holdout set — a "predict everything positive" failure mode. (Bug 2's fix alone did not
resolve this; it persisted afterward.)

**Root cause**: `certainty()`'s final line is `cert_s[1] * agr, cert_s[0] * agr` — i.e. both
class certainties are multiplied by `Agr_G(x)`, so `agr == 0.0` forces `(0.0, 0.0)` regardless of
otherwise-correct `Cert_S` values (verified directly: manually replicating the `Cert_S` formula
for a failing test example against the correct leaf/weights gave a sane nonzero result, while the
live `certainty()` call on the identical input returned exactly `(0.0, 0.0)`).

The original `_ancestor_nodes_and_literals` added one opportunity pair `(node, f, v)` for *every*
accepted split literal at every ancestor node on x's path, and `certainty()` tested
`1[t(x)] := x[f] == v` literally — i.e. "does x's raw feature value equal the literal's target
value," regardless of whether x actually traversed the accept or reject edge at that node. For a
node with only one accepted split (the common case — no alpha-tie), any example that legitimately
takes the *reject* edge scores 0 agreement at that node purely because it didn't match the
literal's positive value, even though there is no alternative candidate literal at that node to
disagree with in the first place. With a single root split and no other ancestor nodes, this drove
`den_agr > 0` but `num_agr == 0`, so `agr` was exactly `0.0` for every example on the reject side.

Re-read Eq. 7 and the surrounding prose in the PDF directly to resolve the ambiguity: "AgrG(x) is
the weighted sum of each successful opportunity (i.e., path segment that x took) divided by the
sum of all opportunity weights in Ox" — Agr_G is framed throughout Sec. 3.5 as measuring
disagreement introduced specifically by G's *lattice of near-tied alternative splits* ("how little
G needs to change to accommodate x"), not by ordinary single-path decision-tree branching. A node
with only one accepted literal has no lattice ambiguity — there is nothing for x to "disagree"
with by taking the only available branch.

**Fix**: `_ancestor_nodes_and_literals` now skips any ancestor node with fewer than 2 accepted
split literals (`len(node.splits) < 2`) — such nodes contribute no opportunity pairs, so they
cannot spuriously zero out `Agr_G`. Nodes with genuine alpha-ties (2+ near-tied literals) still
contribute one opportunity pair per literal, tested against x's raw feature value, as before.
This is a documented interpretive choice (module docstring in `stand_lib.py` updated accordingly)
since the paper's prose does not fully disambiguate this edge case; it is the reading most
consistent with Sec. 3.5's stated purpose for Agr_G and with Eq. 7's "path segment that x took"
framing.

**Verified fix**: on a full-paper-scale single synthetic rep (seed=1, 400 features, 100
train/20-negative, 2000 holdout), hierarchical-shrinkage STAND now scores 98.5% holdout accuracy
vs. plain STAND's 97.4% — h.s. STAND now outperforms plain STAND as the paper claims (previously
h.s. STAND scored ~25%, exactly the holdout's raw positive fraction).

## 4. `stand_lib.py`: `alpha=1.0` default (should be `0.1`) plus `_node_alpha()` only applying the
   dynamic Eq. 13 schedule under `hierarchical_shrinkage=True`, causing a hang / unbounded memory
   blowup on the UCI benchmark

**Symptom**: `uci_benchmark.py` hung for 30+ minutes and grew to ~9GB RSS fitting plain STAND on a
stratified 80/20 split of the breast-cancer dataset (9 low-cardinality categorical features), while
the same model trained fine on sequential-order data and on the 400-feature synthetic benchmark
(which is mostly-noise features, so few candidate splits tie).

**Root cause**: two compounding issues found by grepping the PDF text directly for "α = 0.1" and
"Appendix A":
1. `STAND.__init__`'s default was `alpha=1.0`, i.e. accept literally every split whose gain is
   >= 0 (no rejection threshold at all), even though Sec. 3.1 states plainly "α = 0.1 works well"
   and that value is used for every headline result in the paper.
2. `_node_alpha()` only replaced the fixed `self.alpha` with Eq. 13's dynamic per-node schedule
   `α_nk = 1 - min((1-α0) + α0·(N(k)/M), 1)` when `hierarchical_shrinkage=True`. But Appendix A.1
   introduces Eq. 13 as "our STAND implementation" varies the acceptance rate per node —
   unconditional phrasing, not gated on hierarchical shrinkage (a separate, orthogonal feature
   governed by λp/λs/λn). So plain (non-hs) STAND was both using the wrong fixed alpha *and* never
   getting the dynamic schedule that would have tightened acceptance at small-N nodes anyway.

With `alpha=1.0`, plain STAND accepted essentially every one of the `max_splits_per_node=4`
candidate literals with non-negative gain at every node. On UCI's low-cardinality categorical
features this produces near-complete branching (many nodes averaging close to the cap), and
because the tree has no node-count safety valve, recursion explodes combinatorially. The 400-mostly
-noise synthetic features rarely produce more than one or two positive-gain candidates per node, so
the same bug was latent but non-fatal there — this is why the bug surfaced only once the UCI claim
(Claim 7) was attempted, not during the synthetic-benchmark smoketests.

**Fix**: changed `STAND.__init__`'s default `alpha` from `1.0` to `0.1` (matching Sec. 3.1); added
a `use_dynamic_alpha=True` constructor parameter and made `_node_alpha()` apply Eq. 13's schedule
unconditionally by default (`use_dynamic_alpha=False` recovers Sec. 3.1's simpler fixed-alpha mode
for anyone who wants to isolate that variant). Also added a defensive `max_total_nodes=20000` cap
in `_grow()` (finalizes as a leaf once hit) — purely a runaway-recursion safety valve, not expected
to bind under the corrected alpha defaults, but cheap insurance against any other pathological
dataset.

**Verified fix**: re-ran the isolated breast-cancer stratified-split debug case (previously hung
30+ min / 9GB RSS) — now completes in 0.7-4.6s per model/rep. Ran a full 6-dataset × 5-model
single-rep smoketest (`breast-cancer`, `hepatitis`, `soybean`, `tic-tac-toe`, `vote`, `zoo`) — no
hangs, per-model times from <0.01s to ~32s (XGBoost on tic-tac-toe was the slowest single model).
The synthetic-benchmark run in progress when this bug was found had been using the pre-fix code for
its plain-STAND numbers (h.s.-STAND numbers are unaffected, since hs mode already always used the
dynamic schedule regardless of the `self.alpha` default) — that run was killed and both
`run_synthetic_experiment.py` and `uci_benchmark.py` were relaunched from scratch with the fix.
