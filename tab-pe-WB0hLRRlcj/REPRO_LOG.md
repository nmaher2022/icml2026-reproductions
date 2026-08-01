# Tab-PE (WB0hLRRlcj) reproduction — recovery log

**Read this first on any cold start / session resume.**

## Status as of 2026-08-01 00:43

- Step 0 (paper acquisition): DONE. arXiv fallback (OpenReview bot-walled as usual). Full text +
  appendices readable. `paper.pdf` / `paper.txt` in this folder.
- `PAPER_BRIEFING.md`: DONE. 5 anchored claims transcribed, Algorithm 1/2 pseudocode transcribed,
  scope decisions documented (AIM via `mbi` package for a real CPU head-to-head; TabICL for
  Claims 1/2's downstream classifier, matching what the paper itself used).
- Environment: `tab-pe-WB0hLRRlcj/.venv` (uv-created, Python 3.11). Official code cloned read-only
  to `DPSDA_upstream/` (gitignored, vendored — don't commit it). Installed
  `pip install -e "./DPSDA_upstream[tabular]"` + manually pinned CPU-only
  `torch==2.13.0+cpu` / `torchvision==0.28.0+cpu` (both from
  `https://download.pytorch.org/whl/cpu` — **must reinstall torchvision from that index
  explicitly even after the extras install**, or you get an ABI mismatch
  `RuntimeError: operator torchvision::nms does not exist`; a plain `uv pip install torchvision`
  without `--index-url` silently keeps a non-cpu build even though `torch` itself was cpu). Also
  needed `azure-identity azure-storage-blob openai-cost-logger` (DPSDA's `pe/__init__.py` eagerly
  imports its text/image API modules which pull these in, even though this reproduction only uses
  the tabular path).
- **Blocker found and worked around**: the official `xor_stress_test.py` uses
  `model_name="tabpfn"` for its downstream AUC classifier. The modern `tabpfn` PyPI package (v2+,
  PriorLabs) gates pretrained-weight downloads behind an interactive one-time license acceptance
  (`TabPFNLicenseError`), no non-interactive path exists. Worked around by copying the script to
  `scripts/xor_stress_test_xgb.py` with `model_name="xgboost"` substituted (default max_depth=6,
  sufficient for up to 5-way XOR per the paper's own App. A.1 depth-vs-correlation-order argument;
  the paper itself uses XGBoost-depth sweeps for structurally the same diagnostic in Fig. 12/App
  A.1, so this is a well-precedented substitute, not an arbitrary one). Logged here and will be
  logged again in `BUGFIX_LOG.md` / `VERDICTS.md` — never silently swapped.
- `TabICL` (used for Claims 1/2's real-dataset accuracy, matches paper's own choice) has **no**
  license gate — confirmed working standalone and inside `artificial_characters.py`.
- Smoketests (Step 3): all three script families ran without shape errors, NaNs, or crashes:
  - `scripts/xor_stress_test_xgb.py --num-features 1`: full run, 38.6s wall, sane (AUC~100% at 1
    feature, expected — trivial task).
  - `scripts/artificial_characters.py`: ran 5 min before timeout cutoff, values sane and
    converging toward paper's 49.38% (was at 41.42% partway through the 15-iteration loop, i.e.
    not yet the final-iteration number). ~25 min estimated total (TabICL fit+eval ~28s/iteration +
    3x WSD computes ~23s each, x15 iterations).
  - `scripts/person_activity.py`: ran 3m20s before timeout cutoff, reached iteration 0's classifier
    eval on 5000 synthetic samples with no errors. Likely slower per-iteration than
    artificial_characters (5000 vs 1000 synthetic samples fed to TabICL) — budget more time.

## Scripts (in `scripts/`, all write to `../results/<name>/`)
- `xor_stress_test_xgb.py --num-features N` (N=1..5) — Claim 4.
- `artificial_characters.py` — Claim 1 (and part of Claim 3's runtime evidence).
- `person_activity.py` — Claim 2 (and part of Claim 3's runtime evidence).
- Claim 5 (Algorithm 2 structure) needs no script — verified by reading
  `DPSDA_upstream/pe/population/*.py` + `pe/runner/pe.py` against the paper's pseudocode (not yet
  done as of this update).
- Claim 3's "X times faster than AIM" multiplier needs a same-hardware AIM baseline. Plan: try the
  `mbi` PyPI package (AIM's own author's implementation) — **not yet attempted** as of this
  update. If infeasible within reasonable effort, fall back to citing the paper's own AIM runtime
  numbers as a reference (disclosed, not silently substituted) and rely on our own measured Tab-PE
  wall-clock time as the "no GPU required, runs fast on CPU" evidence.

## Resume instructions
All three script families are checkpoint-resumable (`SaveCheckpoints`/`checkpoint_path=` — official
library feature, "runs resume from here" per DPSDA's own README) — rerunning the same script picks
up from the last saved iteration rather than restarting.

Check `results/*/log.txt` for the last logged iteration + metrics before deciding whether a job is
still needed. Check `pgrep -af "python3 (artificial_characters|person_activity|xor_stress)"` for
anything currently running, and background job logs at `logs/*.log` (see below) for detached runs.

## UPDATE (2026-08-01 00:57): AIM baseline via `mbi`/private-pgm — WORKED, launched for real

`mbi` (PyPI, AIM's own original author Ryan McKenna's package) installs cleanly, CPU-only (jax
backend, no GPU). Cloned `ryan112358/private-pgm` (his reference `mechanisms/aim.py`) to
`DPSDA_upstream_aim/` (gitignored vendored code, same convention as `DPSDA_upstream/`). Hit one
API-version-skew bug: the cloned `aim.py` calls `estimation.MirrorDescent().estimate(...)`
(older class-based API) but the installed `mbi==1.1.0` only exposes the newer functional
`estimation.mirror_descent(...)` — patched all 3 call sites in the local clone with `sed`
(`estimation\.MirrorDescent\(\)\.estimate\(` -> `estimation.mirror_descent(`), confirmed
`MarkovRandomField`'s `.potentials`/`.project()`/`.synthetic_data()` still work with the new
function's return value. This is a local compat patch to vendored code, not something we wrote
from scratch — noted here and in `BUGFIX_LOG.md`, not silently done.

Wrote `scripts/aim_baseline.py`: discretizes numeric columns into 20 quantile bins (paper's own
AIM baseline instead uses "PrivTree" discretization — disclosed difference), builds an `mbi`
`Domain`/`Dataset`, runs AIM with degree=2 pairwise workload (paper sweeps degree 2-5 and reports
the best — we use a single reasonable value, disclosed, not an attempt to replicate their exact
tuning), times the run, and evaluates the synthetic output with the *same* TabICL harness as our
Tab-PE runs for an apples-to-apples accuracy/F1 comparison.

Smoketested successfully on Artificial Characters — ran to completion territory (budget
progressing steadily through ~70%+ before this log update, no errors) as a real (non-toy) run,
left running in the background (PID 172637, log `logs/aim_ac_smoketest.log`).

## UPDATE (2026-08-01 00:56): all remaining runs queued

Launched `scripts/run_all.sh` detached (PID 172984, log `logs/run_all.log`) — waits for the AIM
smoketest job above to finish (avoids CPU oversubscription on this 8-core machine), then runs in
sequence: `artificial_characters.py` (Tab-PE, Claim 1) -> `person_activity.py` (Tab-PE, Claim 2)
-> `xor_stress_test_xgb.py --num-features 1..5` (Claim 4) -> `aim_baseline.py --dataset
person_activity` (Claim 2/3 AIM comparison). Prints `ALL_RUNS_DONE` at the very end of
`logs/run_all.log` — check for that string before assuming everything finished. Estimated total
wall time: rough guess 1-2+ hours (artificial_characters ~25min, person_activity probably longer
given 5x more synthetic samples fed to TabICL each iteration, XOR runs ~1min each x5, AIM on
person_activity's ~115K-row/higher-cardinality domain is the biggest unknown — could be much
slower than the AC AIM run given more categorical/binned columns and larger n).

## UPDATE (2026-08-01 01:10): Claim 5 structural self-audit — DONE, confirmed match

Read `pe/runner/pe.py`, `pe/population/pe_population.py`, `pe/population/composite_population.py`,
`pe/histogram/nearest_neighbors.py`, `pe/dp/gaussian.py` in full and matched every step against
Algorithm 1/2's pseudocode. **No code bugs found; structural match confirmed** — full writeup in
`BUGFIX_LOG.md` §1 (per-class independent loop + union, two-stage `CompositePopulation` schedule
sample-with-replacement+m=1 → top-K+m=3+keep_selected matching the paper's `T_sampling` switch,
Gaussian-mechanism noise on a sensitivity-1 nearest-neighbor histogram). Claim 5 verdict input:
VERIFIED (structural claim, no scale caveat applies). Also consolidated the two already-disclosed
scope decisions (XGBoost-for-TabPFN on Claim 4, AIM binning/degree choices on Claims 1/2/3) into
`BUGFIX_LOG.md` §2 so `VERDICTS.md` has one place to pull caveats from.

As of this update: AIM's artificial_characters smoketest (PID 172637) finished the actual AIM
mechanism run (144.1s, 7152 synthetic rows) and is now in its TabICL-eval step; `run_all.sh`
(PID 172984) is still in its wait loop for that job to exit before starting
`artificial_characters.py`/`person_activity.py`/XOR/`aim_baseline.py --dataset person_activity`.

## UPDATE (2026-08-01 01:2x): AIM artificial_characters baseline — DONE

`results/artificial_characters_aim/eval.json`: our AIM baseline gets **15.92% test accuracy,
14.07% macro F1** on Artificial Characters (degree-2 workload, 20-bin quantile discretization,
ε=1.0, δ=1.575e-05, 144.1s runtime for the mechanism itself). Paper's own reported AIM number for
this dataset is 23.24±1.48% accuracy / 20.17 macro F1 (Table 1). Our baseline underperforms the
paper's AIM — expected and consistent with the disclosed scope decisions in `BUGFIX_LOG.md` §2
(single degree-2 workload vs. their degree-2-to-5 sweep-and-best, quantile binning vs. PrivTree).
This doesn't change the Claim 1 verdict direction (Tab-PE still needs to beat *some* AIM number to
support the claim) but means any "beats AIM by X%" magnitude we report must be framed as "beats
*our* reasonably-tuned AIM baseline by X%", with the paper's own AIM number cited alongside for
context, never conflated as the same measurement. `aim_baseline.py` process has exited cleanly;
`run_all.sh`'s wait-loop should unblock within its 30s poll interval and move on to
`artificial_characters.py` (Tab-PE itself).

## UPDATE (2026-08-01 01:09): Claim 1 (artificial_characters, Tab-PE) — DONE, strong match

`results/artificial_characters/log.txt`, final iteration 14 (T=15 total, 0-indexed, ε=1.0, TabICL
classifier — same setup as the paper): **47.75% test accuracy, 48.20 macro F1**. Paper's own Table
1 number: 49.38±0.46% accuracy, 48.09 macro F1. Accuracy within ~1.6pp of the paper's mean (a
single-seed run vs. their reported ± range, so this is within plausible run-to-run variance);
macro F1 essentially matches (48.20 vs 48.09). This is a **full-paper-scale match** — same
dataset, same T, same ε, same classifier, no substitutions on this script. Strong candidate for
VERIFIED on Claim 1 (pending final Claim 2/3 numbers, since Claim 3's "faster than AIM" framing
also touches this run's wall-clock time). Our own AIM baseline on the same dataset (15.92% acc /
14.07 F1, logged above) is far below both Tab-PE numbers, consistent with the claim direction
regardless of the AIM-baseline scope caveats.

`person_activity.py` (Claim 2) now running (queue moved on at 01:09:47).

## UPDATE (2026-08-01 05:16): Claim 2 (person_activity, Tab-PE) — DONE, near-exact match

`results/person_activity/log.txt`, final iteration 14 (T=15, ε=1.0, TabICL classifier, no errors
or tracebacks anywhere in the log): **63.71% test accuracy, 35.80 macro F1**. Paper's own Table 1
number: 63.72±0.18% accuracy, 35.09 macro F1. Accuracy is essentially identical (0.01pp off,
well inside their own reported ±0.18% run-to-run spread); macro F1 within ~0.7pp. Full-paper-scale
match, no substitutions on this script. Took ~4h wall clock (01:09→05:16) on this 8-core CPU
machine — consistent with Claim 3's "runs on CPU, no GPU needed" framing (feasible, not
necessarily *fast* in absolute terms, but doesn't need GPU hardware). Strong candidate for
VERIFIED on Claim 2.

Queue moved to `xor_stress_test_xgb.py --num-features 1` (Claim 4) at 05:16:13. Still need: XOR
features 1-5, and the AIM person_activity baseline (last item in `run_all.sh`) — that AIM run is
the "biggest unknown" flagged earlier given Person Activity's ~115K rows and high-cardinality
float columns; budget for it to be slow or to need troubleshooting.

## UPDATE (2026-08-01 05:20): Claim 4 (XOR, Tab-PE) — DONE running, but classifier substitute is
inadequate at 4-5 features — investigated and disclosed, not silently reported

All 5 feature-count runs finished cleanly (no errors). Raw AUCs from `xor_stress_test_xgb.py`
(default-depth XGBoost): 1 feat ~100%, 2 feat ~99.96%, 3 feat ~99.3%, 4 feat ~73.3%,
5 feat ~50.7% — looks like a refutation of the paper's "AUC≈0.8 at 5 features" claim at first
glance. Investigated before concluding that (see `BUGFIX_LOG.md` §1b for full writeup): wrote
`scripts/xor_reeval_depth_matched.py` to re-evaluate the *same already-generated* synthetic data
with `max_depth` set exactly equal to `num_features` per the paper's own Appendix C.1 guidance —
still collapsed (56.65%/50.24% AUC at 4/5 features). Then sanity-checked by training the
depth-matched classifier on the full 35,000-row **real** (non-DP) private XOR data directly (no
Tab-PE involved at all): 4 features succeeds well with enough depth (99.98% AUC at depth 5), but
**5 features stays near-random (50.57% AUC) even on real ground-truth data with 35x more rows
than our synthetic eval set**. This proves the ~50% AUC at 5 features is an artifact of the
XGBoost substitute classifier's inability to solve 5-way parity via greedy splitting in this
environment/xgboost version — not evidence about Tab-PE's synthetic data quality.

**Verdict input**: Claim 4 is BLOCKED at the 5-feature (paper's headline) data point — we lack a
working classifier (TabPFN license-gated) capable of running this specific diagnostic, and our
substitute is demonstrably inadequate (fails even on real data). Not REFUTED — the negative result
says nothing about Tab-PE. 1-3 feature results are usable as TOY-VERIFIED supporting evidence
(correct low-order-correlation trend, XGBoost is adequate at that end of the depth requirement).
4-feature result is ambiguous/inconclusive (partial classifier headroom issue, not clearly
resolved) — will note as BLOCKED alongside 5-feature rather than force a verdict.

Only remaining background job: `aim_baseline.py --dataset person_activity` (last step in
`run_all.sh`), still running as of this update — flagged as the "biggest unknown" given the
dataset's scale/cardinality.

## UPDATE (2026-08-01 05:40): background job got killed mid-eval (session teardown), resumed cleanly

`run_all.sh` and the in-flight `aim_baseline.py --dataset person_activity` process both
disappeared (no longer in `pgrep`) between updates, with `logs/run_all.log` cutting off right
after "AIM run finished in 1170.7s, produced 115402 synthetic rows" — no `ALL_RUNS_DONE`, no
Traceback. This was **not a script crash**: `results/person_activity_aim/synthetic.csv` (9MB,
115403 rows) and `timing.json` were written successfully, meaning the actual AIM mechanism run
completed; only the subsequent TabICL-eval half of `aim_baseline.py` got cut off, consistent with
the whole detached process group being torn down by something in the session/tool environment
(monitor teardown) rather than an in-script error — `nohup` detaches from terminal HUP but doesn't
protect against an explicit process-group kill.

**Resumed cheaply**: wrote `scripts/aim_eval_person_activity_resume.py` — loads the
already-completed `synthetic.csv` directly (skips redoing the expensive 1170s AIM run),
subsamples to 5000 rows (matching Tab-PE's own `person_activity.py` sample budget — disclosed in
the script's docstring), and reruns just the TabICL eval, writing `eval.json`. Launched via
`setsid nohup ... &` (survives shell/session teardown more robustly than plain `nohup &`) to
`logs/aim_pa_eval_resume.log`.

**Lesson for next time**: for the *next* reproduction in this repo, prefer `setsid nohup <cmd>
disown` (or an equivalent double-detach) over plain `nohup ... &` for anything long enough to
outlive a session boundary — plain background jobs in this environment can apparently still be
killed when the parent tooling session ends, contrary to normal nohup semantics.

## UPDATE (2026-08-01 09:57): AIM person_activity eval resumed successfully — Claim 2/3 AIM evidence complete

`results/person_activity_aim/eval.json`: our AIM baseline gets **48.32% test accuracy, 22.62 macro
F1** on Person Activity (same scope caveats as the AC AIM run: degree-2 workload, 20-bin quantile
discretization, ε=1.0, δ=7.43e-07, 1170.7s runtime for the mechanism itself, subsampled to 5000
synthetic rows for eval to match Tab-PE's own sample budget — disclosed in
`aim_eval_person_activity_resume.py`'s docstring). Paper's own reported AIM number for this dataset
is 59.53±0.47% accuracy / 30.79 macro F1 (Table 1). Our baseline underperforms the paper's AIM here
too (same expected pattern as artificial_characters) — consistent with the disclosed scope
decisions in `BUGFIX_LOG.md` §2. Tab-PE's own result (63.71%/35.80, logged above) beats our AIM
baseline by a wide margin on both datasets, supporting the claim *direction* even though the exact
"beats AIM by X%" magnitude is versus our own reasonably-tuned AIM, not the paper's tuned one.

**Claim 3 (compute efficiency) — timing evidence assembled, with a caveat on the precise multiplier:**
- AIM baseline wall-clock (clean single-shot `time.time()` measurement inside `aim_baseline.py`,
  mechanism run only, not eval): **144.1s** (artificial_characters), **1170.7s** (person_activity).
  Both AIM runs happened on the same CPU-only 8-core machine as Tab-PE, no GPU used anywhere in
  this reproduction.
- Tab-PE wall-clock: attempted to derive from `results/*/log.txt` first/last timestamps, but these
  logs are **cumulative across checkpoint resumptions** (the Step-3 smoketest partially ran
  `artificial_characters.py` before `run_all.sh`'s full run resumed the same checkpoint later), so
  a naive first-log-line-to-last-log-line delta would conflate an interrupted smoketest with the
  full run and does not represent a clean single-shot wall-clock. Rather than report a
  confounded/misleading number, **declining to state a precise Tab-PE-vs-AIM wall-clock multiplier
  for our own runs.**
- What we CAN state cleanly: (a) every run in this reproduction — Tab-PE and our AIM baseline
  alike — executed entirely on CPU, no GPU used or required anywhere, directly supporting the
  "runs entirely on CPUs" half of Claim 3; (b) our AIM baseline's own mechanism runtime (144.1s /
  1170.7s) is a real, same-hardware reference point, even though we can't cleanly divide Tab-PE's
  confounded wall-clock by it; (c) the paper's own precise multipliers (~28x vs AIM, ~10x vs
  PrivMRF at ε=1, 18.6x at 500K samples, Fig. 4) are cited as-is, not independently re-derived here.
- **Verdict input**: Claim 3 is TOY-VERIFIED-with-caveat — "no GPU needed" is directly confirmed by
  our own runs; the precise "Nx faster" multiplier is not independently reproduced (logging/
  checkpointing confound on our side, not a Tab-PE failure), so that part is BLOCKED, with the
  paper's own numbers cited for reference only.

Task #4 (run + self-audit all 5 claims) is now complete — every claim has either a numeric result,
a structural audit, or an explicit BLOCKED/caveat writeup. Moving to `VERDICTS.md` (task #5).

## What's NOT done yet (next actions)
1. Check `logs/run_all.log` for `ALL_RUNS_DONE` and `logs/aim_ac_smoketest.log` /
   `results/artificial_characters_aim/{eval,timing}.json` for the AIM AC run's final numbers. If
   AIM on Person Activity fails or is impractically slow, note that in `BUGFIX_LOG.md`/
   `VERDICTS.md` and fall back to citing the paper's own AIM number for that dataset only.
2. ~~Claim 5 structural self-audit~~ — DONE, see above and `BUGFIX_LOG.md` §1.
3. Once runs finish: pull final numbers from each `results/*/log.txt` (Tab-PE) and
   `results/*_aim/eval.json`+`timing.json` (AIM), compare against the paper's Table 1 / Fig 1 /
   Fig 4 numbers transcribed in `PAPER_BRIEFING.md`, write `VERDICTS.md` (Step 5) — pull the
   disclosed-caveat language straight from `BUGFIX_LOG.md` §2, never round a caveated result up to
   plain VERIFIED.
4. Trackio logbook + poster (Step 6), GitHub monorepo mirror (Step 7, low-effort here since this
   checkout *is* the monorepo already), harness self-audit (Step 8).

## Task tracker
This session used TaskCreate; task #3 ("smoketest") is done, task #4 ("run experiments and
self-audit") is in progress — Claim 5's self-audit is now done (`BUGFIX_LOG.md`); remaining work
under task #4 is just waiting for the background runs and then writing `VERDICTS.md` (task #5).
