# Reproduction bundle — WorldComp2D: Spatio-semantic Representations of Object Identity and Location from Local Views

Reproduces WorldComp2D, a lightweight facial-landmark localization architecture (proximity-dependent
encoder + localizer + auxiliary refinement CNN) built for real-time CPU inference, using the paper's
own official code: [JinSeongmin/WorldComp2D](https://github.com/JinSeongmin/WorldComp2D) (gitignored
vendored clone, see `.gitignore`), unmodified, with the paper's released pretrained checkpoints.

Paper: OpenReview [WQIyx69dFg](https://openreview.net/forum?id=WQIyx69dFg)
Paper also on arXiv: [2605.11743](https://arxiv.org/abs/2605.11743) (v1).

Trackio logbook: TBD (Step 6 of the harness, not yet published).

## Verdict

| Claim | Outcome |
|---|---|
| 1. Params/FLOPs reduction vs. PoPos (Table 1) | **VERIFIED** (FLOPs, <1% error) / caveat on params magnitude (~3.7-3.8× not paper's stated 4.0×) |
| 2. Real-time CPU inference + accuracy (Table 1/2) | **VERIFIED** (COFW NME) / **TOY-VERIFIED** (FPS, correct ordering, hardware-limited magnitude) / **BLOCKED** (300W, AFLW NME — dataset access) |
| 3. Robustness to input degradation (Table 4) | **VERIFIED** (COFW, 10/12 conditions within ~0.05 NME) / **BLOCKED** (300W, AFLW — dataset access) |
| 4. Zero-shot cross-dataset transfer, 300W→COFW-68 (Table 6) | **VERIFIED** (near-exact: 6.082 vs. 6.08 NME_IO, 20.2% degradation matches exactly) |
| 5. Structured latent space (Fig 5a-c) | **VERIFIED** (Spearman ρ≥0.995 on all 3 panels; Fig 5d ablation not attempted — see `VERDICTS.md`) |

Full detail, tables, and reasoning for each verdict: `VERDICTS.md`.

**Dataset access, load-bearing for the above:** COFW (grayscale + color) fully accessible, no
auth, used for all 5 claims. 300W's and AFLW's own test images are **BLOCKED** — both require
PII-gated manual access-request forms, deliberately not attempted. Claim 4 is still fully VERIFIED
because it only needs COFW-68 (Ghiasi & Fowlkes 2014's 68-point re-annotation of COFW, found
unblocked on GitHub) plus the paper's *pretrained* 300W checkpoint — not 300W's own test data.

## Contents

- `PAPER_BRIEFING.md` — paper's claims/architecture/equations transcribed before running anything, including the dataset-access investigation.
- `REPRO_LOG.md` — chronological run log.
- `BUGFIX_LOG.md` — every bug found and fixed during implementation (MATLAB/HDF5 axis-order bugs, the pre/post-crop corruption-methodology investigation, the Loc-module parameter discrepancy), with full before/after diagnostic numbers.
- `VERDICTS.md` — final verdict for each of the 5 claims with paper-vs-ours comparison tables and reasoning.
- `claim1_params_flops.py` — Claim 1 (param count + FLOPs via `thop`, all 3 datasets, no data needed).
- `claim2_fps.py` / `claim2_nme_cofw.py` — Claim 2 (CPU inference speed on dummy inputs; COFW accuracy on the real 507-image test set).
- `claim3_corruption_robustness.py` — Claim 3 (13 conditions: baseline + Gaussian blur × 3 + JPEG × 4 + motion blur × 2 + occlusion × 2, all on real COFW test images).
- `claim4_cross_dataset.py` — Claim 4 (pretrained 300W checkpoint evaluated zero-shot on COFW-68).
- `claim5_latent_structure.py` — Claim 5 (PdEnc embedding analysis: intra/inter-class clustering, inter-class separation, distance preservation — Fig 5a-c).
- `logs/` — raw stdout from each claim script's final run, plus one named smoketest diagnostic (`claim3_smoketest_blur_precrop_vs_postcrop_diagnostic.log`, the small pre/post-crop comparison that motivated Claim 3's mixed methodology).
- `build_poster.py`, `poster.html`, `poster.png`, `poster_embed.html` — poster generation for the Trackio logbook.
- `data/`, `data_raw/` (gitignored) — COFW `.mat` files (official grayscale + color test sets from `data.caltech.edu`), re-downloadable per `PAPER_BRIEFING.md`.
- `cofw68-benchmark/` (gitignored vendored clone) — Ghiasi & Fowlkes' COFW-68 68-point re-annotation + images, used by Claim 4.
- `WorldComp2D/` (repo root, gitignored) — vendored official code + released pretrained checkpoints, not committed.

## Rerun

```bash
git clone https://github.com/JinSeongmin/WorldComp2D.git WorldComp2D
# place official pretrained checkpoints per WorldComp2D/README.md, and COFW .mat files under
# worldcomp2d-WQIyx69dFg/data/COFW/ (see PAPER_BRIEFING.md for exact download URLs)

cd worldcomp2d-WQIyx69dFg
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision h5py scipy opencv-python pillow thop

python3 claim1_params_flops.py            # Claim 1
python3 claim2_fps.py                     # Claim 2 (speed)
python3 claim2_nme_cofw.py                # Claim 2 (accuracy)
python3 claim3_corruption_robustness.py   # Claim 3
python3 claim4_cross_dataset.py           # Claim 4 (needs cofw68-benchmark/, see script header)
python3 claim5_latent_structure.py        # Claim 5
```

No GPU required for any step — all evaluation runs on CPU (`torch.device("cpu")`), consistent with
the paper's own real-time-CPU-inference framing.
