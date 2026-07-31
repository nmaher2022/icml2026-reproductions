# REPRO_LOG.md — spectral-cit-nPzckCXmHE

**Read this first on a cold start / session resume.**

## What's running
`claim1_2_signal_ablation.py 30` — launched detached via `nohup`, PID/log below. Reproduces
Claims 1 (Thm 4.1 validity) & 2 (Thm 4.2 power) using the paper's Fig. 11 signal-strength-ablation
benchmark (d=10, d_z=3, N=1000, 30 reps x 4 conditions = 120 trials, ~21s/trial => ~42 min total).

- Log file: `claim1_2_run.log` (tail it for live progress — prints after every trial, so partial
  progress is visible even if the process is killed mid-run).
- Output files (written only at the very end, all-or-nothing): `claim1_2_raw.csv` (per-trial T_n /
  T_n_pruned / pruned_dim / E_val) and `claim1_2_summary.csv` (per-condition aggregates + KS test).
- Launch command: `nohup .venv/bin/python claim1_2_signal_ablation.py 30 > claim1_2_run.log 2>&1 &`
  (run from the `spectral-cit-nPzckCXmHE/` folder, using the repo-root `.venv` which has CPU torch
  already installed — do NOT use `uv run` for this script, it will try to fetch a fresh GPU torch
  build from PyPI instead of using the pre-installed CPU one, see BUGFIX_LOG note in this repro).

## How to check status
```bash
tail -20 claim1_2_run.log        # live progress
ls -la claim1_2_summary.csv      # exists only once the run has fully finished
ps aux | grep claim1_2           # confirm the process is still alive
```

## What's already done (don't redo)
- Step 0 (paper acquisition): DONE. arXiv 2512.19510v2 fetched (OpenReview bot-blocked), fully
  readable including all 3 appendices. See PAPER_BRIEFING.md.
- Step 2 (briefing): DONE. `PAPER_BRIEFING.md` has the 5 in-scope claims, transcribed math,
  hyperparameters (Appendix C Table 2), and synthetic benchmark equations.
- Step 3 (smoketest): DONE. `smoketest.py` passed (tiny scale: no NaNs, H1>H0 direction correct).
- Implementation (`scit_lib.py`, `data_gen.py`): DONE, with two real bugs/gaps found and fixed
  during self-audit *before* this run — see `BUGFIX_LOG.md`:
  1. `validation_error()` computed a cross-covariance instead of self-covariances (fixed).
  2. Dimension pruning (Appendix C, "for added stability") was missing entirely from the first
     implementation; added as `test_statistic_pruned()`. Root cause noticed: `w_theta`'s learned
     representation doesn't spread variance across all `2*d` output dims under the paper's own
     reference hyperparameters, so `Ĉ_{ŴŴ}` stays far from identity even after whitening — this is
     a real, disclosed finding, not swept under the rug.
- M_theta/N_theta parameterization ambiguity: documented as an assumption (BUGFIX_LOG entry 2),
  not resolvable from the paper text alone, doesn't affect the test statistic itself.

## claim1_2_signal_ablation.py 30 -- RESULTS (completed 2026-07-31 ~21:17, 2626s)
See `claim1_2_summary.csv` / `claim1_2_raw.csv` for full numbers. Headline:
- H0: mean T_n=65.7 (raw, vs chi2(100) mean=100) -- conservative, matches the paper's own
  Section 6 admission ("we still observe conservative calibration in practice"). Raw Type I error
  @ alpha=0.05: 0/30 (0%). Dimension-pruned (k=9, chi2(81) reference): Type I error 2/30 (6.7%),
  much closer to nominal 5%.
- Power: str=0.05 (weak) -> ~0% (both raw/pruned); str=0.15 (moderate) -> 100%; str=0.5 (strong)
  -> 100%. This exact low->high transition between str=0.05 and str=0.15 matches the paper's own
  Fig. 11 qualitative pattern closely.
- KS test rejects a clean chi2(100)/chi2(81) fit for the full null distribution shape
  (p~1.8e-19 raw, p~1.4e-8 pruned) even though the alpha=0.05 tail rejection rate is roughly
  right (pruned) -- expected at finite n=1000 for an m,n->infinity asymptotic theorem, worth
  stating honestly in the verdict rather than glossing over.
- E_val (Claim 3 diagnostic) sat at ~0.9999-1.0000 across all 120 trials (min 0.99999, max
  0.999999) -- confirmed systematic, not noise (see BUGFIX_LOG entry 3: w_theta's output doesn't
  spread variance across all 2d dims under these hyperparameters).

## Claim 5 ablation -- DONE (completed 2026-07-31 ~21:30, 616s)
`claim5_subgaussian_ablation.py 30` -- Tanh (bounded) vs Identity (unbounded), H0 only, 30 reps
each. RESULT: Identity did NOT inflate Type I error (opposite of naive expectation) -- mean T_n
collapsed to 10.6 (vs Tanh's 81.5, both far below chi2(100) mean=100), Type I error 0% both
raw/pruned (vs Tanh's 3.3%/10.0%). BUT: nn.Identity activation collapses the whole MLP to a
single linear map (composed linear layers = one linear layer), so this ablation confounds
"boundedness" with "loss of all nonlinear capacity" -- can't cleanly attribute the T_n collapse
to Assumption 4.1's sub-Gaussianity mechanism specifically. Verdict in VERDICTS.md: INCONCLUSIVE,
not forced into REFUTED/VERIFIED. A cleaner follow-up would use LeakyReLU/ELU instead of Identity
to isolate boundedness from nonlinearity -- not attempted, lower priority.

## VERDICTS.md -- DONE (2026-07-31)
All 5 in-scope claims + the out-of-scope real-data claim have final verdicts. Summary table:
Claim 1 TOY-VERIFIED, Claim 2 TOY-VERIFIED, Claim 3 TOY-VERIFIED (partial, disclosed limitation:
E_val stayed pinned ~1.0 across all 120 Claims-1/2 trials, see BUGFIX_LOG entry 3), Claim 4
VERIFIED (structural), Claim 5 INCONCLUSIVE (ablation confounded, see above). Real-data claim
(TCGA-BRCA) correctly out of scope, not attempted.

## What's NOT done yet (next actions)
1. Optionally, time/turn budget permitting: a reduced-rep version of the paper's primary Fig. 2
   post-nonlinear-model benchmark (d_Z sweep 50-300) as supplementary evidence -- not required for
   the 5 in-scope claims, lower priority.
2. Step 6 of the harness: Trackio logbook + poster (`.agents/skills/trackio/`).
3. Step 7 of the harness: GitHub monorepo mirror (`scripts/scaffold_reproduction.py`) -- requires
   explicit user confirmation before any `git push`, per harness instructions and this repo's
   established practice.

## Task tracker
This session used TaskCreate; task #2 ("select claims + smoketest") is where this run belongs,
task #3 ("run experiments and self-audit") is next.
