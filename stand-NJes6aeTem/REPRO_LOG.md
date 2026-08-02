# STAND (NJes6aeTem) reproduction — recovery log

## Context-reset / session-teardown recovery (READ FIRST on a cold start)
1. Read auto-memory (project-icml-repro.md) -> it should point here.
2. Read STATUS + NEXT ACTIONS below.
3. Check what's already done:
   - `cat stand-NJes6aeTem/synthetic_run.log` (tail for progress; look for "Total time" = finished)
   - `ls stand-NJes6aeTem/synthetic_results.json` (exists = the main synthetic experiment, Claims
     1-5, is done)
   - `ls stand-NJes6aeTem/uci_results.json` (exists = Claim 7's UCI benchmark is done)
   - `ls stand-NJes6aeTem/lambda_p_sweep_results.json` (exists = Claim 6's hyperparameter sweep is
     done — this one likely hasn't been launched yet as of this writing, see NEXT ACTIONS)
4. `pgrep -af "run_synthetic_experiment.py\|uci_benchmark.py\|lambda_p_sweep.py"` — if a process
   that should be running is dead and its output json doesn't exist yet, relaunch it (commands
   below). These scripts are NOT currently checkpointed mid-run (they write the final JSON only at
   the very end) — a killed run must be restarted from scratch, seed 0. Keep them small enough
   (already reduced to toy scale) that a restart is cheap (~10-25 min), not catastrophic.
5. Continue with NEXT ACTIONS.

## STATUS (as of 2026-08-02, mid-run, updated after bug #4 + relaunch)

Implementation (`data_gen.py`, `stand_lib.py`) is done and debugged — see `BUGFIX_LOG.md` for the
4 bugs found and fixed (distractor-literal overlap bug in data_gen; asymmetric hierarchical-
shrinkage gain calculation; Agr_G collapsing to exactly 0 for reject-branch traversals through
unambiguous single-split nodes; **`alpha=1.0` default should be `0.1` per Sec 3.1, plus
`_node_alpha()` was only applying Eq. 13's dynamic schedule under `hierarchical_shrinkage=True`
instead of unconditionally — this caused a 30+ min hang / 9GB RSS blowup on the UCI benchmark's
low-cardinality categorical features**). All fixes verified via smoketest — see BUGFIX_LOG.md #4
for exact numbers.

**IMPORTANT — `.venv` no longer exists in this directory** (disk was at 98% full, 8.9G free, when
last checked; likely lost to disk pressure, or never needed since these are PEP-723 self-contained
scripts). Use `uv run <script>.py` instead of `.venv/bin/python <script>.py` — `uv` resolves the
inline `# /// script` dependency block itself and has 3.11/3.12/3.13 interpreters already
installed locally, so this does not require much new disk space (confirmed: "Installed 8/13
packages" in <200ms on relaunch, no meaningful disk growth).

Three experiment scripts written; **all three (re)launched from scratch with the bug-#4 fix**
since the two that had been running earlier in the session (see "UPDATE" section below) were using
the pre-fix `alpha=1.0` default for their non-hierarchical-shrinkage STAND numbers, which is wrong
per Sec 3.1's explicit "α = 0.1 works well":
- `run_synthetic_experiment.py` (Claims 1-5: accuracy, error reoccurrence, productive
  monotonicity, calibration, active learning utility) — relaunched via
  `nohup uv run run_synthetic_experiment.py > synthetic_run.log 2>&1 < /dev/null & disown`.
  Toy-scale: 15 reps (paper: 100), 10 checkpoints N=10..100 step 10 (paper: every N=1..100).
- `uci_benchmark.py` (Claim 7: 6-dataset UCI noisy benchmark) — relaunched the same way via
  `nohup uv run uci_benchmark.py > uci_run.log 2>&1 < /dev/null & disown`, output to
  `uci_results.json` when done. 10 reps (paper appears to use a single 80/20 split; 10 reps here
  is a strengthening, not a reduction — document as such). Should now complete in well under an
  hour given the smoketest showed no dataset/model pair taking more than ~32s.
- `lambda_p_sweep.py` (Claim 6: Fig. 12's λp ∈ {0,0.5,1,5,10,20,50} sweep) — **still NOT launched**
  as of this log entry; launch after the above two finish (avoid 3-way CPU contention), via
  `nohup uv run lambda_p_sweep.py > lambda_p_sweep.log 2>&1 < /dev/null & disown`. This one is
  unaffected by bug #4 in practice (hs mode already always used the dynamic schedule) but wait
  for CPU headroom anyway.

Exact paper numbers to compare against (Table 1, Table 2, Fig 12 caption — already transcribed
into `PAPER_BRIEFING.md` claims 1-2 and confirmed against the PDF directly):
- Table 1: STAND 90.0±1.1% acc / 2.6±0.1% FP reocc / 1.2±0.1% FN reocc; STAND(hs) 92.8±1.1% /
  2.2±0.1% / 0.8±0.1%; DecTree 88.6±1.1% / 7.3±0.3% / 3.8±0.2%; DecTree(hs) 87.81±0.9%; XGBoost
  84.8±1.5% / 3.4±0.1% / 0.8±0.1%; RandForest 56.2±0.8% / 2.4±0.2% / 0.2±0.1%; NeuralNet 56.7±0.7%
  / 3.3±0.1% / 1.7±0.1%; VSSM 55.5±1.5% / 0.04±0% / 0.03±0% (VSSM not implemented, BLOCKED, see
  PAPER_BRIEFING.md known-blockers section).
- Table 2 (UCI, single 80/20 split per the paper as far as we can tell): DecisionTree 85.84% avg
  (breast-cancer 74.14, hepatitis 67.74, soybean 85.40, tic-tac-toe 94.79, vote 97.70, zoo 95.24);
  RandomForest 88.22% avg; XGBoost 88.57% avg (best); STAND 87.25% avg; STAND(heir) 86.40% avg —
  per-dataset breakdown is in the paper table, transcribe into VERDICTS.md once our own numbers are
  in.

## UPDATE (mid-run): uci_benchmark.py had to be relaunched twice
First attempt was accidentally wrapped in `timeout 300` and got killed before finishing (no
output, no `uci_results.json`). Second attempt used the Bash tool's `run_in_background: true`
tracking directly — that tracked background session (along with its watcher) got torn down/killed
by an external event unrelated to this task (not OOM per `dmesg`, cause unclear) partway through,
while `run_synthetic_experiment.py` (launched via plain `nohup ... &` + `disown`, NOT tool-tracked)
survived the same event untouched. **Lesson confirmed empirically**: `run_in_background: true`
alone is not equivalent to a real detached OS process for this environment — always use
`nohup ... > log 2>&1 < /dev/null & disown` for anything that must survive session/tool-session
churn, not just for surviving a Claude session *limit*. Current (third) launch:
`nohup .venv/bin/python -u uci_benchmark.py > uci_run.log 2>&1 < /dev/null & disown`. Two
matching nohup'd watcher loops append a line to `.watchers_done` when `uci_results.json` exists /
`synthetic_run.log` contains "Total time" respectively — check
`wc -l stand-NJes6aeTem/.watchers_done` (2 lines = both done) or just check the two result files
directly; don't rely on any Bash-tool-tracked notification surviving, confirmed unreliable above.
CPU contention note: an unrelated concurrent process on this machine (`main.py -c
main-openloris-8192/bimu`, a different reproduction project per memory
`project-repro-bimu`) was consuming 350%+ CPU throughout, which is most of why reps slowed from an
initial ~30s/rep estimate to 100-500s/rep. Not something to fix here — just explains the wall-clock
budget, don't mistake it for a bug in these scripts.

## STATUS UPDATE (2026-08-02, later same day): Steps 4-6 complete

All three experiment scripts finished with the bug-#4 fix applied (`synthetic_results.json`,
`uci_results.json`, `lambda_p_sweep_results.json` all present). `VERDICTS.md` written — 6/8 claims
TOY-VERIFIED, 1 TOY-VERIFIED-weak-with-ranking-divergence (Claim 7, UCI), 1 BLOCKED (Claim 8).
`BUGFIX_LOG.md` has all 4 bugs documented.

Trackio logbook published: `nmaher/repro-stand-self-aware-precondition-induction-for-interactive-task-learning`
(https://huggingface.co/spaces/nmaher/repro-stand-self-aware-precondition-induction-for-interactive-task-learning).
Note: the repo-root `.trackio/` dir held **worldcomp2d's** leftover local state at the start of this
step (confirmed via `feedback-trackio-logbook-reattach-footgun` memory) — backed it up (moved, not
deleted) to the scratchpad before scaffolding STAND's logbook fresh with
`scaffold_icml_logbook.py`, so worldcomp2d's published Space was never touched. Poster built as a
plain single-page HTML (`poster.html`, same pattern as `worldcomp2d-WQIyx69dFg/poster.html`) and
screenshotted via `chromium-browser --headless --screenshot`, not the full posterly
measure/PDF-gate pipeline (skipped as disproportionate for this pass — no user available to answer
the design-discovery questions posterly's SKILL.md normally asks first). Reproduction-bundle
artifact logged under project `stand-njes6aetem-docs-bundle` (NOT the default project name) with
individual `add_file()` calls only — per `feedback-trackio-artifact-project-scoping` memory, never
`add_dir()` a whole reproduction folder. Verified clean afterward: 605KB / 15 files in the private
artifacts bucket (`huggingface_hub.bucket_info`), matching the curated set exactly.

## NEXT ACTIONS
1. Step 7 — mirror into `nmaher2022/icml2026-reproductions` as `stand-NJes6aeTem/` via
   `scripts/scaffold_reproduction.py add` (direct commit + push, pre-authorized).
2. Step 8 — run `harness-testing/audit_harness.py stand-NJes6aeTem`.
