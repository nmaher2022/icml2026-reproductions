# Bugfix / self-audit log — Tab-PE (WB0hLRRlcj)

Per the repro-harness Step 4 requirement: every deviation from "run the official code unmodified"
gets logged here, whether it's a bug we hit and fixed, a scope decision, or a structural
self-audit finding. Nothing here is silently absorbed into VERDICTS.md.

## 1. Claim 5 (Algorithm 2 structure) — code-vs-pseudocode self-audit

Method: read `DPSDA_upstream/pe/runner/pe.py`, `pe/population/pe_population.py`,
`pe/population/composite_population.py`, `pe/histogram/nearest_neighbors.py`,
`pe/dp/gaussian.py` in full and matched each against Algorithm 1 / Algorithm 2 as transcribed in
`PAPER_BRIEFING.md` (which was itself transcribed from `paper.txt` lines 250-330). This is a
structural/control-flow claim, not a numeric one — verified by inspection, not a script run.

**Algorithm 1 (DP_NN_HISTOGRAM)** — CONFIRMED matches:
- `NearestNeighbors.compute_histogram` (`pe/histogram/nearest_neighbors.py`): for each private
  sample, finds its nearest synthetic-sample neighbor(s) (`num_nearest_neighbors=1` in every
  script we use, matching the paper's single-nearest-neighbor histogram) and increments a vote
  counter (`Counter(list(ids.flatten()))`) → clean histogram in `CLEAN_HISTOGRAM_COLUMN_NAME`.
  This is exactly "for each private sample s, find nearest neighbor index i, increment hist[i]".
- `Gaussian.add_noise` (`pe/dp/gaussian.py`): adds i.i.d.
  `np.random.normal(scale=self._noise_multiplier, size=len(...))` to every bin of the clean
  histogram → `DP_HISTOGRAM_COLUMN_NAME`. Matches "add i.i.d. N(0,σ²) noise to every histogram
  bin", and the noise multiplier is calibrated by `get_noise_multiplier` from
  (epsilon, delta, num_iterations) via the standard Gaussian-mechanism analytic accountant —
  consistent with "sensitivity of the histogram query is 1 → Gaussian mechanism" (each private
  sample can only vote once, i.e. only affect one bin, when `num_nearest_neighbors=1`).

**Algorithm 2 (Tabular Private Evolution, per-class loop)** — CONFIRMED matches:
- `PE.run()` (`pe/runner/pe.py`): `initial()` is called once per `label_info` (per class c),
  each class's initial population sized by `_get_num_samples_per_label_id` (proportional to
  private-class fraction, matching N^(c) = N·|D_priv^(c)|/|D_priv|). Every iteration afterward
  loops `for label_id in range(len(label_info))`, computing histogram + adding noise + advancing
  population **independently per label**, then `Data.concat`-ing the per-label results back
  together at the end of the iteration. This matches "per class c ∈ C, run independently, results
  unioned" exactly — including the final union, since `Data.concat` is exactly the ∪ in
  `D_syn = D_syn ∪ D_{T-1}` (here done every iteration via concat + `syn_data.metadata.iteration`
  rather than only at the last iteration, but the accumulated per-class dataframe at the final
  iteration is what every script's `TabClassifier`/eval callback reads, so the observable claim
  behavior is identical).
- Two-stage selection schedule: `CompositePopulation` (`composite_population.py`) picks
  `self._populations[iteration]` per PE iteration. Every script we ran builds
  `[population1] * T_sampling + [population2] * (num_iterations - T_sampling)` (e.g.
  `xor_stress_test_xgb.py` line 76: `[population1] * 5 + [population2] * (num_iterations - 5)`).
  - `population1` = `PEPopulation(selection_mode="sample", next_variation_api_fold=1,
    keep_selected=False, histogram_threshold=0)` → `_select_data` with `selection_mode="sample"`
    does `np.random.choice(..., p=count/count.sum())`, i.e. weighted sampling **with
    replacement** proportional to the (clipped, since `histogram_threshold=0`) noisy histogram —
    exactly `D_t = sample N^(c) from P_t with replacement, weighted by prob`. `keep_selected=False`
    + `next_variation_api_fold=1` means `next()` returns only the single variation fold
    (`variation_data_list`, no `selected_data` appended) — exactly
    `P_{t+1} = VARIATION_API(D_t, m=1)` with no union, matching the paper's `t < T_sampling`
    branch (single variation, no union).
  - `population2` = `PEPopulation(selection_mode="rank", initial_variation_api_fold=3,
    next_variation_api_fold=3, keep_selected=True)` → `_select_data` with `"rank"` does
    `np.argsort(count)[::-1][:num_samples]`, i.e. exact top-N by histogram — exactly
    `D_t = top N^(c) of P_t by hist_t`. `keep_selected=True` + `next_variation_api_fold=3` (i.e.
    m=3) means `next()` concatenates `variation_data_list + [selected_data]` — exactly
    `P_{t+1} = VARIATION_API(D_t, m) ∪ D_t`, matching the paper's `t >= T_sampling` branch.
  - The single switch point at iteration index `T_sampling` (5 for the real/XOR datasets we use)
    reproduces the "two-stage schedule" wording in the claim text itself
    (PAPER_BRIEFING.md claim 5) precisely: sampling-with-replacement + m=1 for the first
    `T_sampling` iterations, then top-K ranking + m>1 with retention for the rest.

**Minor note (not a code bug):** my own pseudocode transcription in PAPER_BRIEFING.md (based on a
literal read of the paper's Algorithm 2 box) has two slightly different thresholds
(`t <= T_sampling` for the selection-mode switch vs. `t < T_sampling` for the variation-fold/union
switch) — a one-iteration mismatch that, taken literally, would need three regimes, not two. The
released code only has **one** switch point (`CompositePopulation`'s single list-index boundary),
which matches the claim's own prose ("two-stage schedule") rather than the finer-grained pseudocode
detail. Read as either (a) an off-by-one in the paper's own typeset pseudocode, or (b) the two
boundaries being intended to coincide in the paper's default hyperparameters. Either way, the code
we're running is unambiguous and matches the *claim text*, so this doesn't affect the Claim 5
verdict — flagged here only for completeness, not as a discrepancy between code and claim.

**Verdict input**: Claim 5 structural match is CONFIRMED by inspection — no code bug found, no
open questions remaining. Will be written up as VERIFIED (not a numeric claim, so no scale caveat
applies) in `VERDICTS.md`.

## 1b. Claim 4 (XOR stress test) — XGBoost substitution investigated further, found inadequate at n≥4

`scripts/xor_stress_test_xgb.py`'s initial run (default `xgb.XGBClassifier(objective=
"binary:logistic")`, i.e. library-default `max_depth=6`, no override) collapsed to near-random
AUC at 4 features (73.31%) and 5 features (50.74%) — the opposite of the paper's claimed AUC≈0.8
at 5 features. Before treating this as a refutation, investigated whether it was an artifact of
the classifier substitution rather than a real Tab-PE failure, since the paper's own Appendix C.1
(paper.txt lines 1061-1063) explicitly states "the max depth of XGBoost must be equal to the
number of features to achieve better-than-random accuracy" on XOR data, and our default depth (6)
wasn't matched to `num_features` per run.

**Step 1 — depth-matched re-eval on the same Tab-PE synthetic output**
(`scripts/xor_reeval_depth_matched.py`, loads each run's saved checkpoint, does NOT rerun DP
generation, just re-evaluates with `max_depth=num_features`): still collapsed —
4 features: 56.65% AUC (depth=4), 5 features: 50.24% AUC (depth=5). Full results in
`results/xor_depth_matched_eval.json`. Ruled out "wrong max_depth" as the sole explanation.

**Step 2 — sanity check on the REAL private training data** (not synthetic at all, all 35,000
train rows, same depth-matched XGBoost, same held-out test set): at 4 features, depth=5/6
succeeds well (99.98%/99.99% AUC) — confirms the classifier/pipeline is *capable* of solving 4-way
XOR given enough real data and a depth headroom of +1. But at **5 features, even training on the
full 35,000-row real (non-DP, non-synthetic) dataset, depth-matched XGBoost only gets 50.57% AUC
at depth=5 and 57.14% at depth=6** — i.e., XGBoost with default hyperparameters (no tuning beyond
max_depth) cannot reliably solve 5-way XOR in this environment/xgboost version *even given the
ground-truth real data with 35x more rows than the synthetic evaluation set*.

**Conclusion**: the near-random AUC we observed at 5 features is a property of the XGBoost
substitute classifier itself (likely default regularization/split-search defaults preventing the
greedy tree builder from finding the exact-parity split path at this feature count in this xgboost
version — a known general difficulty of greedy CART on parity functions), not evidence about
whether Tab-PE's synthetic data correctly captures the 5-way XOR structure. We have no way to
distinguish "Tab-PE fails at 5 features" from "our classifier can't detect it even when it
succeeds" with the tools available in this environment (TabPFN license-gated, TabICL not
validated on this binary/AUC task by us). **Claim 4 at 5 features is therefore BLOCKED, not
REFUTED, not VERIFIED** — logged here in full so `VERDICTS.md` states this precisely rather than
reporting the raw ~50% AUC number as if it meant something about Tab-PE. The 1-3 feature results
(where our XGBoost eval does show a real, expected degradation trend: ~100% → ~99.96% → ~99.3%
AUC) remain usable as partial, TOY-VERIFIED-quality evidence that Tab-PE's synthetic data tracks
low-order XOR correlations correctly; they just don't reach the paper's headline 5-feature claim.

## 1c. Claim 3 (compute efficiency) — Tab-PE wall-clock timing not cleanly derivable from our logs

`results/artificial_characters/log.txt` and `results/person_activity/log.txt` are cumulative
across checkpoint resumptions: the Step-3 smoketest partially ran `artificial_characters.py`
before `run_all.sh`'s later full run resumed the same checkpoint (DPSDA's checkpoint/resume
feature, working as intended). This means the first-log-line-to-last-log-line delta for these
files conflates an interrupted smoketest with the eventual full run and is not a clean single-shot
wall-clock measurement. Rather than report a confounded number as if it were comparable to our
AIM baseline's clean `time.time()`-measured runtime (144.1s AC / 1170.7s PA, measured inside
`aim_baseline.py` with no resumption involved), we are **declining to state a Tab-PE-vs-AIM
wall-clock multiplier from our own runs**. What remains cleanly true: every run in this
reproduction (Tab-PE and AIM alike) executed on CPU only, directly supporting the "no GPU needed"
half of Claim 3; the paper's own precise multipliers (~28x vs AIM, Fig. 4) are cited as a reference
point, not re-derived. This caps Claim 3 at TOY-VERIFIED (CPU-only claim) + BLOCKED (precise
multiplier claim), never a clean VERIFIED on the full claim as stated.

## 2. Disclosed substitutions / scope decisions carried over from REPRO_LOG.md (consolidated here)

- **Claim 4 (XOR stress test) classifier**: official `xor_stress_test.py` uses
  `model_name="tabpfn"`; substituted `model_name="xgboost"` in our copy
  (`scripts/xor_stress_test_xgb.py`) because the modern `tabpfn` PyPI package (PriorLabs, v2+)
  gates pretrained-weight downloads behind an interactive one-time license acceptance
  (`TabPFNLicenseError`) with no non-interactive path in this headless environment. Precedented by
  the paper's own use of XGBoost-depth sweeps for the same type of correlation-order diagnostic
  (App. A.1 / Fig. 12). This caps Claim 4's verdict at TOY-VERIFIED-with-caveat at best, never
  plain VERIFIED, regardless of the numeric outcome.
- **Claims 1/2/3 AIM baseline** (`scripts/aim_baseline.py`): uses quantile binning (20 bins) for
  numeric-column discretization, not the paper's own "PrivTree" discretization; uses a single
  degree-2 pairwise workload, not the paper's degree-2-to-5 sweep-and-report-best (App. C.3). Both
  disclosed choices, not attempts to replicate the paper's exact tuned AIM baseline — our AIM
  number is "a real, same-hardware, reasonably-configured AIM run", not "the paper's AIM
  reproduced exactly". This is what grounds any "faster than AIM" (Claim 3) or "beats AIM"
  (Claims 1/2) verdict language — will be phrased as "our own reasonably-tuned AIM baseline",
  never presented as matching the paper's tuned number.
- **AIM vendored-code compat patch**: `DPSDA_upstream_aim/mechanisms/aim.py` (cloned from
  `ryan112358/private-pgm`) called `estimation.MirrorDescent().estimate(...)`, an older class-based
  API not present in the installed `mbi==1.1.0`. Patched (local clone only, not upstream) via
  `sed -i 's/estimation\.MirrorDescent()\.estimate(/estimation.mirror_descent(/g' mechanisms/aim.py`
  — 3 call sites, functionally equivalent replacement (confirmed `MarkovRandomField`'s
  `.potentials`/`.project()`/`.synthetic_data()` still work against the new function's return
  value). This is a dependency-version-skew fix to make third-party code importable, not a change
  to the AIM algorithm itself.
