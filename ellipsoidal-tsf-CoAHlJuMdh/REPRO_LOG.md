# REPRO_LOG.md — Ellipsoidal TSF (CoAHlJuMdh)

**Read this first on a cold start.**

## Status (update as work progresses)
- Step 0 (paper acquisition): DONE. arXiv 2505.17370v6 fetched (OpenReview bot-walled, as
  expected). `paper_arxiv_2505.17370v6.pdf` + `paper_text.txt` in this folder.
- Step 2 (briefing): DONE. `PAPER_BRIEFING.md` written, 5 claims transcribed from the actual PDF
  (claims_anchored.json for this paper was found STALE — see briefing's warning box — not used).
- Step 3/4 (implement + smoketest + run): IN PROGRESS.
  - `data_gen.py` (Lorenz-63/Roessler/Chua RK4 generators) — smoketested, no NaNs, std values
    plausible vs paper (Lorenz x-std ≈7.8, paper says "≈8").
  - `fern_lib.py` (Fern model per Algorithm 1, DLinear baseline, MSE/WD/SWD/EPT metrics,
    ablation flags) — smoketested, no NaNs, all ablation variants run.
  - `run_claim1_and_3.py` — nonstationary robustness (Claim 1) + EPT (Claim 3). ~53s per
    (system, shock, seed) combo, 18 combos total ≈ 16 min. Resumable: writes
    `claim1_3_results.json` incrementally after each (system, shock) group.
  - `run_claim2_complexity.py` — analytic FLOP sweep (Claim 2). Seconds, already run once
    standalone (see below) — results in `claim2_results.json`.
  - `run_claim4_ablations.py` — Table 3/8 ablations (Claim 4). ~28s per training, 12 trainings
    ≈ 5.6 min. Writes `claim4_results.json` incrementally per seed.
  - `run_claim5_horizon.py` — horizon-scaling geometric-persistence check (Claim 5). ~28s per
    training, 24 trainings ≈ 11 min. Writes `claim5_results.json` incrementally per horizon.

## How to resume
Each `run_claim*.py` script is standalone and writes its own `claim*_results.json`
incrementally (safe to re-run from scratch if interrupted — total wall clock ≈33 min for all
three heavy scripts, well within one script's own runtime even restarted). Check for
`claim1_3_results.json` / `claim4_results.json` / `claim5_results.json` and the corresponding
`*.log` files in this folder to see what's already done; `claim2_results.json` (fast, analytic)
either exists or re-run takes seconds.

Invoke via `../.venv/bin/python run_claim*.py` (NOT `uv run` — see PAPER_BRIEFING.md).

## Next actions after all 4 run scripts finish
1. Read all `claim*_results.json`, self-audit the implementation against the paper's Algorithm 1
   / Eq. 1-2 / Appendix A.1 metric definitions once more (Step 4's dedicated self-audit pass —
   look for sign errors, wrong-granularity gates, metrics measuring something subtly different).
   Log any real bugs found + before/after numbers in `BUGFIX_LOG.md`.
2. Write `VERDICTS.md` (Step 5) — one verdict per the 5 claims in `PAPER_BRIEFING.md`, using
   VERIFIED/TOY-VERIFIED/REFUTED/BLOCKED/INCONCLUSIVE vocabulary, honest about toy-scale
   magnitude gaps vs the paper's headline numbers (e.g. Fern will very likely beat DLinear by a
   small multiple at toy scale, not literally "up to 790x" — TOY-VERIFIED directionally, not
   VERIFIED).
3. Trackio logbook + poster (Step 6), monorepo mirror (Step 7 — `git remote -v` already
   confirmed origin = `nmaher2022/icml2026-reproductions`, work in place, no fresh clone needed),
   harness self-audit (Step 8).

## Known ambiguity documented
Shock timing for "param" scenarios isn't specified exactly in the paper text (only the parameter
deltas are, Table 7) — implemented as a shift applied at the train/test boundary (70% mark) so the
held-out test set is evaluated under the shocked regime, the natural reading of "nonstationary
shock" testing. See `data_gen.py` docstring.

## Step 6 complete (Trackio logbook + poster, published)
Trackio logbook published: `nmaher/repro-ellipsoidal-time-series-forecasting`
(https://huggingface.co/spaces/nmaher/repro-ellipsoidal-time-series-forecasting).
Repo-root `.trackio/` held STAND's leftover local state at the start of this step (STAND already
published, safe) — backed it up to scratchpad, scaffolded fresh with `scaffold_icml_logbook.py`.
Poster: single-page HTML (`poster.html`) + chromium headless screenshot (`poster.png`), embedded
via base64 (`poster_embed.html`, with the required `<!-- poster_embed.html -->` comment for
validate_icml_logbook.py's substring check). Reproduction bundle logged as an individual
`add_file()`-only artifact under project `ellipsoidal-tsf-coahljumdh-docs-bundle` (NOT the default
project name) per `feedback-trackio-artifact-project-scoping` memory — verified clean via
`bucket_info()`: 249,900 bytes / 19 files. `validate_icml_logbook.py --space ...` passed. Tags
confirmed: `icml2026-repro`, `paper-CoAHlJuMdh`, `arxiv:2505.17370`.

Next: Step 7 (mirror into `nmaher2022/icml2026-reproductions` monorepo via
`scripts/scaffold_reproduction.py add`), Step 8 (harness self-audit).
