# WorldComp2D (WQIyx69dFg) — REPRO_LOG

**Read this first on a cold start.** This reproduction is in progress. Full context in
`PAPER_BRIEFING.md` (paper math/architecture/claims) — read that too before continuing.

## Status as of 2026-08-01 (session start)

- Paper acquired: `2605.11743v1.pdf` + `.txt` in this folder. OpenReview bot-walled, arXiv used
  (only version available — no cross-check needed/possible).
- **Official code + pretrained checkpoints found**: `../WorldComp2D/` (repo root, gitignored
  upstream clone of `github.com/JinSeongmin/WorldComp2D`). Has full training code AND pretrained
  `.pth` weights for all 3 datasets (COFW/300W/AFLW) — this means several claims can be verified
  by running the authors' own weights against real test data rather than retraining from scratch.
- `PAPER_BRIEFING.md` written: 5 claims in scope (params/FLOPs, CPU-FPS+NME, corruption
  robustness, cross-dataset transfer, latent structure), full architecture/loss transcribed from
  the paper.
- **Background agent launched** (env setup + dataset acquisition investigation): CPU-only venv at
  `worldcomp2d-WQIyx69dFg/.venv`, COFW download from Caltech (`data.caltech.edu`, confirmed
  no-auth direct download), 300W/AFLW acquisition feasibility check. Not yet returned as of this
  log entry — **check its result before redoing this work**.

## Status as of 2026-08-01 (mid-session update)

- Dataset access resolved: COFW fully accessible (grayscale + color mats, no auth,
  data.caltech.edu). 300W BLOCKED (PII-gated download form). AFLW images BLOCKED (manual
  access-request form requiring personal info — explicitly not attempted, a user decision).
  COFW-68 (Ghiasi & Fowlkes 2014) found unblocked on GitHub (`golnazghiasi/cofw68-benchmark`),
  which unblocks Claim 4 despite 300W's own test-set blocker (only need the pretrained 300W
  checkpoint + COFW-68 images/annotations, not 300W's own test data).
- **Claim 1 (params/FLOPs) — DONE.** FLOPs near-exact match (<1% error). Params ~8% high, traced
  to Localizer module specifically (real discrepancy, not a bug) — see BUGFIX_LOG.md.
- **Claim 2 FPS half — DONE.** Ordering across datasets matches exactly; absolute FPS 5-18x lower
  than paper due to genuine hardware gap (i7-8550U laptop vs i5-13400 desktop) — TOY-VERIFIED.
- **Claim 2 NME half (COFW) — DONE.** 5.165±0.052 vs paper's 5.16±0.05. Near-exact. VERIFIED.
- **Claim 4 (cross-dataset, COFW-68) — DONE.** 6.082±0.145 vs paper's 6.08±0.14. Near-exact,
  degradation 20.2% matches paper's 20.2% to one decimal. Strongest match so far. VERIFIED.
- **Claim 3 (corruption robustness, COFW) — IN PROGRESS.** `claim3_corruption_robustness.py`
  written and smoketested (baseline sub-run matched Claim 2's NME exactly: 5.165±0.052). Full
  13-condition sweep (baseline + 3 blur + 4 JPEG + 2 motion-blur + 2 occlusion) running.
- **Claim 5 (latent structure, Fig 5) — NOT STARTED.**

## Next actions (in order)
1. Check `claim3_corruption_robustness.py`'s full-sweep output, log results to BUGFIX_LOG.md
   with per-condition comparison against Table 4's COFW column, write verdict.
2. Claim 5: Fig 5a-c use the pretrained COFW PdEnc's embeddings directly on COFW test images
   (intra-class clustering, inter-class separation, latent-distance-vs-pixel-distance monotonicity
   within RF[2]'s ~54px range) — no training needed, just embedding extraction + analysis/plots.
   Fig 5d needs training a small proximity-*un*weighted ablation PdEnc from scratch on COFW
   (Eq 2 with w_ij=1) — tractable on CPU at reduced scale, report as TOY-VERIFIED for that panel.
3. Self-audit pass (`BUGFIX_LOG.md`) before writing verdicts — reread implementation against the
   paper's equations (see PAPER_BRIEFING's "Core math" section), specifically checking: is the
   FLOPs formula counted the way the paper counts it (per-fixation-point PdEnc cost x9, not once);
   is NME normalization (IO vs IP vs Diag) matched per-dataset as the paper uses it; is the
   proximity weight/contrastive loss (if Fig 5d ablation is attempted) implemented per Eq 2-3
   exactly, not a simplified approximation.
4. Verdicts -> Trackio logbook + poster -> monorepo mirror (Steps 5-8 of the harness).

## Environment note
Scripts run via `worldcomp2d-WQIyx69dFg/.venv/bin/python`, NOT `uv run` — torch-dependent PEP-723
scripts fetch fresh GPU wheels from PyPI instead of reusing a pinned CPU build (bit
`spectral-cit-nPzckCXmHE` earlier in this project).

## Status as of 2026-08-01 (complete)

All 5 claims run to completion, verdicts written (`VERDICTS.md`), Trackio logbook published, and
the reproduction mirrored into the monorepo. Fig 5d's ablation encoder (proximity-unweighted
PdEnc, would need training from scratch) was scoped out — not part of the anchored claim text
(panels a-c only), documented as "not attempted" in `VERDICTS.md` rather than silently omitted.

- Trackio logbook: https://huggingface.co/spaces/nmaher/repro-worldcomp2d-spatio-semantic-representations-of-object-identity-and-location-from-local-vie
- `harness-testing/audit_harness.py worldcomp2d-WQIyx69dFg`: PASS, 0 hard failures, 0 warnings.
- Committed and pushed to `nmaher2022/icml2026-reproductions` main (commit `ff795a3`).
- `logs/` populated with a fresh final rerun of every claim script (all numbers matched
  `VERDICTS.md` exactly except FPS, which is expected to vary run-to-run on this shared/virtualized
  host — already disclosed as TOY-VERIFIED for that reason).
