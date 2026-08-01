# WorldComp2D: Spatio-semantic Representations of Object Identity and Location from Local Views — reproduction briefing

Paper: arXiv 2605.11743v1 (12 May 2026), "WorldComp2D: Spatio-semantic Representations of Object
Identity and Location from Local Views", SeongMin Jin, Doo Seok Jeong (Hanyang University,
Republic of Korea). ICML 2026 regular contribution, CC-BY-4.0.
OpenReview id: `WQIyx69dFg`. OpenReview PDF was bot-walled (Cloudflare challenge on both
`forum?id=` and `pdf?id=`) — consistent with every prior paper in this project. Fell back to
arXiv per the harness's Step 0 rule. Local copy: `worldcomp2d-WQIyx69dFg/2605.11743v1.pdf`
(+ `2605.11743v1.txt` pdftotext extraction). Only one version was fetchable, so no cross-check
was possible/needed — arXiv is being treated as authoritative for this reproduction.

Challenge: HF Space `ICML-2026-agent-repro/challenge`. This reproduction lands in
`nmaher2022/icml2026-reproductions` as `worldcomp2d-WQIyx69dFg/`.

**Official code is available and unusually complete**: `github.com/JinSeongmin/WorldComp2D`
(Apache-2.0), cloned locally to `WorldComp2D/` (repo root, gitignored — upstream vendor clone,
not committed). It ships full training code (`Networks/{Proximity_dependent_Encoder,Localizer,
Auxiliary_localizer}/{Networks,Train,Utils}.py`) AND **pretrained `.pth` checkpoints for all
three datasets** (`Pretrained_modules/{COFW,300W,AFLW}_{Proximity_dependent_encoders,Localizers,
Auxiliary_localizers}.pth`, plus a shared `Landmark_coordinate_priors.pth`) and a ready-made
eval script `Test/Test.py` that reproduces Table 1/2 NME numbers directly. This changes the
reproduction strategy substantially from prior papers in this project: several claims can be
checked by **running the authors' own pretrained weights against real test data**, rather than
retraining from scratch — much stronger evidence than a from-scratch toy reimplementation, IF the
datasets can be acquired in the exact format the loaders expect.

## Working conventions for this reproduction
- Environment: `worldcomp2d-WQIyx69dFg/.venv` (CPU-only, no CUDA — this machine has no GPU),
  installing as close as possible to `WorldComp2D/requirements.txt`
  (torch==2.0.1, torchvision==0.15.2, pandas==1.3.4, numpy==1.22 — note `cv2==4.12.0` in that
  file is not a real PyPI package name, install `opencv-python` instead; `h5py`, `scipy`, `Pillow`
  are used by `Test/Utils.py` but missing from `requirements.txt`, add them). Any version
  substitutions must be logged, not silently made.
- Scripts for this reproduction are self-contained and run via `.venv/bin/python`, not `uv run`
  (torch-dependent PEP-723 scripts fetch fresh GPU wheels from PyPI instead of a pinned CPU build
  — this bit a prior reproduction in this project, see `spectral-cit-nPzckCXmHE`'s REPRO_LOG).
- Raw/preprocessed dataset files live in `worldcomp2d-WQIyx69dFg/data_raw/` and
  `worldcomp2d-WQIyx69dFg/data/` — both added to the repo-root `.gitignore` (large, re-fetchable,
  and in some cases third-party licensed data that must not be redistributed via this repo).
- **Smoketest before scale**: before running anything longer than ~30s-1min, run a tiny/fast
  version (few samples, 1 batch) and check for shape errors, NaNs, sane NME magnitudes. Only scale
  up (full test set) once clean.
- All work happens in `worldcomp2d-WQIyx69dFg/`. Don't touch other paper folders in this repo.
- Verdict vocabulary: VERIFIED / TOY-VERIFIED / REFUTED / BLOCKED. State the scale run next to
  every verdict. Never round a toy-scale pass up to VERIFIED. Report blocked claims explicitly.
- Self-check before finishing: reread the exact claim text and the numbers/plots produced side by
  side — does the evidence actually support what the claim says, at the scale actually run?

## Claims in scope (from `claims_anchored.json`, cross-checked against the PDF)

1. **Params/FLOPs reduction (Table 1, Sec 4.3):** WorldComp2D reduces parameter count by up to
   4.0x (9.7M -> 2.4M) and FLOPs by up to 2.2x vs the PoPos SoTA lightweight baseline (Xiang et
   al., 2025, 9.7M params / 1.2 GFLOPs — the worst case for the FLOPs ratio). Full module
   breakdown: PdEnc 1.1M params / ~15.7M FLOPs (per single fixation-point pass), Loc 1.3M params /
   ~3.0M FLOPs, AuxLoc 4.0K params / ~5.9M FLOPs per landmark. Total FLOPs formula (Sec 4.3):
   `FLOPs_tot = 9*FLOPs_PdEnc + FLOPs_Loc + |C_tot|*FLOPs_AuxLoc` (9 fixation points, |C_tot| =
   number of landmark classes: 29 COFW / 68 300W / 19 AFLW) -> 293.7M / 546.8M / 256.9M FLOPs
   respectively (Table 1 last column). This claim is entirely dataset-independent — checkable by
   directly instantiating the official `Networks.py` modules and counting parameters/FLOPs (e.g.
   via `thop`/`ptflops` or a manual FLOPs counter matching the paper's own formula), no test data
   needed.

2. **Real-time CPU inference + competitive accuracy (Table 1, Table 2, Sec 4.3):** On an
   Intel i5-13400 CPU (paper: i5-13400 2.5GHz, 128GB DRAM; we will report our own CPU spec instead
   and flag the hardware difference), FP32 batch-size-1 throughput of 138.4 FPS (COFW), 78.48 FPS
   (300W), 163.55 FPS (AFLW), with NME of 5.16+-0.05 (NME_IO, COFW), 5.06+-0.01 (NME_IO, 300W),
   1.52+-0.01 (NME_Diag, AFLW) — "competitive with but slightly behind SoTA regression methods"
   per Table 1's comparison rows (e.g. HRNet 3.32 NME_IO on 300W, PoPos 3.28). The FPS component
   is dataset-independent (needs only correctly-shaped dummy inputs run through the pretrained
   checkpoints on CPU) and reproducible regardless of dataset access; the NME component needs the
   real test sets with ground-truth landmarks.

3. **Robustness to input degradation (Table 4, Sec 4.3):** under Gaussian blur (sigma=1/2/3), JPEG
   compression (Q=80/60/40/20), motion blur (kernel=5/10), and occlusion (patch size=20/40), NME
   degrades only moderately across all three datasets, with the single largest degradation being
   +10.2% relative NME under extreme occlusion (size=40) on 300W (5.06 -> 5.58, i.e. (5.58-5.06)/
   5.06 = 10.28%). Needs real test-set images (to apply the corruptions to) plus ground truth.

4. **Zero-shot cross-dataset transfer (Table 6, Sec 4.3):** a WorldComp2D model trained on 300W
   (68 landmarks), evaluated zero-shot on COFW-68 (Ghiasi & Fowlkes 2014's 68-landmark relabeling
   of COFW, NOT the standard 29-landmark COFW used elsewhere in the paper), degrades NME by only
   20.2% (5.06 -> 6.08+-0.14), versus 27-32% degradation for regression-based SoTA baselines (LAB
   32.4%, ODN 27.1%, PIP 31.0%). This needs the separate COFW-68 annotation set, which is a
   different, less-common resource than plain COFW — flag as a likely harder acquisition than
   Claims 2/3/5 which only need the datasets already used for training.

5. **Structured latent space (Fig 5, Sec 4.2):** on COFW (29 landmarks), PdEnc's latent
   representations (a) intra-class cluster (small L2 distance between representations of the same
   landmark class across images, Fig 5a), (b) inter-class separate (larger L2 distance between the
   left-pupil landmark's representations and other classes', with nearby-in-space landmarks
   showing smaller-than-average separation, Fig 5b), and (c) preserve real-world spatial proximity
   (L2 latent distance increases monotonically with real-world pixel distance from a fixation
   point, within the ~54px range of the second-scale receptive field RF[2], Fig 5c) — contrasted
   against (d) an ablation encoder trained with proximity-*un*weighted contrastive loss (w_ij=1
   for all i,j in Eq. 2), which does NOT preserve this proximity structure (Fig 5d). This is a
   qualitative/analysis claim, reproducible via direct inspection of the pretrained COFW PdEnc's
   embeddings on COFW test images — the proximity-unweighted ablation encoder is NOT among the
   provided pretrained checkpoints, so panel (d) would need retraining a small ablation PdEnc from
   scratch on COFW (tractable: PdEnc alone, 2000 epochs per Appendix A, but on CPU may need scaling
   down and would then be TOY-VERIFIED not VERIFIED for that sub-panel).

(The unanchored `claims.json` merges these into 3 broader claims: params/FLOPs reduction,
competitive-accuracy-at-real-time-CPU-speed, and explicit latent structure — the 5-claim anchored
breakdown above is what this reproduction tracks, since it maps directly onto the paper's own
tables/figures.)

## Core math / setup (transcribed from the paper)

**Architecture (Sec 3, Fig 2, Appendix B).**
- **PdEnc** (proximity-dependent encoder): input is a two-scale local observation `o` around a
  fixation point F — patch `o[1] in R^{C x a x a}` from receptive field RF[1], and patch
  `o[2] in R^{C x sa x sa}` from a larger RF[2] (s=4), resized to `C x a x a` and concatenated
  channel-wise with `o[1]` to form `o in R^{2C x a x a}` (a=27; C=1 grayscale for COFW, C=3 RGB for
  300W/AFLW). Out-of-bounds regions use constant padding. Architecture (C32, the paper's default):
  `2C32-32C32(s=1)-32C64-64C64(s=1)-64C128-128C256-FC512-FC256-L2Norm` for COFW (channel-in=2 for
  grayscale), `6C32-...` for 300W/AFLW (channel-in=6 for RGB, since 2*C=2*3). All convs kH=kW=3,
  s=2, p=1 unless marked `(s=1)`. Output z in R^256 is L2-normalized (`z^T z = 1`, lies on a
  hypersphere). Two smaller variants C9/C16 exist for the complexity-vs-accuracy ablation
  (Table 5) — architectures also given in Appendix B.
- **Loc** (localizer): MLP `FC512-FC512-FCn-Tanh`, n = 2*|C_tot| (58 COFW, 136 300W, 38 AFLW).
  Input: N_F=9 fixation-point observations' latent vectors z (each 256-d) concatenated with their
  normalized fixation coords, vectorized to `R^{N_F*(256+2)}`. Output: coordinates for ALL |C_tot|
  landmark classes at once (not per-observation), normalized to (-1,1) via Tanh, added as an
  *offset* to a precomputed per-landmark mean coordinate ("Landmark_coordinate_priors.pth" in the
  official repo) — see Sec 4.1 "Loc predicted an offset relative to this mean."
- **AuxLoc** (optional refinement): CNN `2C24-4xDW24-C1` for COFW (grayscale), `4C24-4xDW24-C1` for
  300W/AFLW (RGB) where DW = depthwise-separable conv block. Input: first-scale patch `o[1]`
  (C x a x a) around Loc's predicted coordinate x_hat for one landmark class, concatenated with a
  1 x a x a class-conditioned embedding (values in [-1,1]) -> `R^{(C+1) x a x a}`. Output: a
  1 x a x a heatmap h; the refined coordinate is x_hat + offset-from-heatmap-peak, clamped to
  +-2px on COFW, +-1px on 300W/AFLW (Sec 4.1).
- Fixation points: N_F=9, evenly spaced at 64px intervals on a 256x256 image:
  `[[64,64],[64,128],[64,192],[128,64],[128,128],[128,192],[192,64],[192,128],[192,192]]`
  (see `Test/Test.py`'s hardcoded tensor — matches Fig 4's NF=9 layout). N_F=4/5 ablations exist
  (Table 7, Fig 4) with different fixed layouts.

**Training loss (Sec 3.1, Eq 1-3).** Proximity-weighted contrastive loss (PWConLoss):
for a proximal-object set `P_i = {c in C_i | x_c in o[2] for F_i on image i}` (objects visible
within the *second-scale* receptive field of sample i's fixation point), the loss is
`L_PWC = (-1/|B|) * sum_i (1/N_i) sum_{j in B\{i}} w_ij * 1{c_i in P_j} * l_ij
         + (1/N_i') sum_{j in B'} w_ij * 1{c_i in P_j} * l_ij`
with `l_ij = log( exp(z_i^T z_j / tau) / sum_{k in (B U B') \ {i}} exp(z_i^T z_k / tau) )`
(standard InfoNCE-style log-softmax over all other samples in the augmented batch), and proximity
weight `w_ij = 1 + exp(-0.025 * d_ij)` where `d_ij = ||x_c(i) - F_j||_2` is the real-world pixel
distance between object c_i's true coordinate and fixation point F_j of sample j — weight in
(1,2], calibrated so w=1.5 at d=27px (= the RF[1] side length). B is the base mini-batch (F=x_c
for a randomly sampled object c per image); B' is an augmented mini-batch with a single *random*
fixation point per image (not tied to any object). Ablation for Fig 5d: w_ij=1 for all i,j
(proximity-unweighted).

**Datasets (Sec 4, Table numbers as stated in the paper — used as ground truth to compare our
reproduction against):**
- **COFW** (Burgos-Artizzu et al. 2013): 1,345 train / 507 test grayscale images, 29 landmarks.
  Frequent occlusion. NME normalized by inter-ocular distance (NME_IO) or inter-pupil distance
  (NME_IP).
- **300W** (Sagonas et al. 2016): 3,148 train / 689 test RGB images, 68 landmarks. Pose/illumination
  variation. NME_IO or NME_Diag (bbox diagonal).
- **AFLW** (Koestinger et al. 2011): 20,000 train / 4,386 test RGB images, 19 landmarks. Large pose
  variation, partial visibility. NME_Diag.
- **COFW-68** (Ghiasi & Fowlkes 2014): a 68-landmark re-annotation of COFW test images, used only
  for the Claim 4 cross-dataset transfer eval (300W-trained model, zero-shot).

**Training hyperparameters (Appendix A, Table 1):** PdEnc: Adam, 2000 epochs (600 on AFLW), batch
128, lr 1e-2, decay x0.1 at epoch 1000 (300 on AFLW). Loc: Adam, 1000 epochs (400 on AFLW), batch
50, lr 5e-4, decay x0.1 at epoch 500 (200 on AFLW). AuxLoc: same schedule as Loc. Preprocessing:
crop to full head, random rescale +-5%, random horizontal flip 50%, random rotation 60% chance
+-10deg, resize to 256x256; AuxLoc's ground-truth heatmap is a 2D Gaussian (std=1.5) centered at
the true landmark within patch o[1]; samples whose true coord falls outside o[1] are dropped from
AuxLoc training. Original training used an RTX A6000 GPU workstation — irrelevant for us since
we're using their pretrained checkpoints, not retraining these three main modules (except possibly
the Fig 5d ablation encoder, which is small/CPU-feasible).

**A genuine ambiguity, not a bug, worth documenting:** the paper doesn't fully specify whether
`Landmark_coordinate_priors.pth` (the per-landmark mean coordinates Loc's output is added to as an
offset) is computed once globally per dataset or per some other grouping — Sec 4.1 just says "we
first computed the mean location for each landmark across the samples in a given dataset." Since
the official repo ships this file precomputed, we don't need to re-derive it — just use it as-is
via `Test/Models.py`'s `framework_making`.

## Known access blockers (confirmed 2026-08-01)
- OpenReview (`openreview.net`) is bot-walled from this environment for both `forum` and `pdf`
  endpoints — arXiv is the source of truth for this reproduction (see above).
- **COFW: unblocked.** Direct anonymous download from `data.caltech.edu` (`COFW.zip`, 178MB).
  `.mat` files are HDF5/v7.3, load cleanly with `h5py` (no format mismatch). Verified 507 test /
  1345 train samples match the paper's stated split sizes; `Test.Utils.COFW.__getitem__` runs
  clean on both splits. Placed at `worldcomp2d-WQIyx69dFg/data/COFW/COFW_{train,test}.mat`.
- **300W: BLOCKED.** `ibug.doc.ic.ac.uk`'s actual archive links (`ibug.zip`/`afw.zip`/`helen.zip`/
  `lfpw.zip`/`300w.zip.001-004`) all redirect to a PII-gated download form (first/last name,
  email, affiliation via POST) — not just a click-through. Separately, the exact CSV schema
  `WorldComp2D`'s loader expects (`300W_train_data.csv`/`300W_test_data_full.csv`, `image_path` +
  136 flat landmark floats) does not match any known public face-alignment preprocessing pipeline
  — a GitHub code search for the literal filenames returned 0 results. This looks like the
  authors' own private preprocessing, not a shared/reproducible artifact. Any claim requiring us
  to build the 300W *test set* from scratch is BLOCKED. Note: we still have the **pretrained
  300W-trained checkpoint weights** from the official repo (no 300W data needed to use them as a
  frozen model) — this matters for Claim 4 (see below), which only needs 300W's *train-time*
  model, not 300W's own test images.
- **AFLW: BLOCKED (images), unblocked (annotations alone).** The `AFLWinfo_release.mat` annotation
  file the loader needs is directly downloadable, no auth
  (`mmlab.ie.cuhk.edu.hk/projects/compositional/AFLWinfo_release.mat`), but useless without the
  ~25K actual Flickr images, which require a formal manual-approval access-request form on
  `tugraz.at` (Nextcloud-hosted, license agreement, implied human review) — not attempted (would
  require submitting the user's personal details to a third party; a decision for the user, not
  made unilaterally mid-task). No non-gated public image mirror found (Baidu Cloud links from
  older repos exist but require a Baidu account and weren't treated as a reliable substitute).
- **Net effect on claims**: Claims 1 (params/FLOPs, dataset-independent) and the FPS half of
  Claim 2 are already done for all three datasets (see `claim1_params_flops.py`,
  `claim2_fps.py`) since they only need the model + pretrained weights, not test data. The NME
  half of Claim 2, Claim 3 (corruption robustness), and Claim 5 (latent structure) can all be run
  on **COFW only** — 300W/AFLW portions of those claims are BLOCKED (dataset access), not
  refuted. Claim 4 (cross-dataset 300W-trained model -> COFW-68) does NOT need 300W's own test
  data (we already have the pretrained 300W checkpoint) — it only needs **COFW-68** (Ghiasi &
  Fowlkes 2014's 68-landmark relabeling of COFW), a separate resource from plain COFW. Its
  availability is being checked next; if found, Claim 4 may be fully unblocked despite 300W's
  test-set blocker. If not found quickly, Claim 4 is BLOCKED.
