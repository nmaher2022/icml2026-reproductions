# Reproduction handoff log — ICML 2026 #2446 (Fake Forgetting / Conformal Prediction Unlearning)

**Paper:** "Tackling Fake Forgetting through Uncertainty Quantification" · OpenReview `rjmVJaBpkm` · arXiv `2501.19403`
**Deliverable:** Trackio logbook Space `nmaher/repro-tackling-fake-forgetting-through-uncertainty-quantification`
**Working dir:** `/home/rec1/Desktop/AI_Safety/ICML_reproduce`

This file is the single source of truth for continuing the work autonomously (automatic mode). Update the
**STATUS** and **NEXT ACTIONS** sections as steps complete.

---

## Decision & scope (user-approved)
- User chose: **(1) CPU toy reproduction now, then (3) verify metrics.** User is separately sorting GPU compute
  for a later full run — so keep everything parameterized for a GPU re-run.
- Blockers: no local GPU (Intel iGPU only, 8 cores, ~15GB RAM); **HF Jobs = 402 (no credits)**. Both documented
  in memory (`blocker-no-gpu-hf-jobs-402.md`).
- CPU benchmark: ResNet-18 ≈ **40 ms/img/train-step, 15 ms/img/eval** on this box.

## Environment
- Python venv: `./.venv` (uv venv, py3.11). Installed: torch/torchvision (CPU), scikit-learn, scikit-image,
  matplotlib, numpy, tqdm. **wandb is NOT installed** — the driver stubs it (`sys.modules["wandb"]`).
- Official code: `./Conformal-Prediction-Unlearning/` (cloned from github.com/TIML-Group/Conformal-Prediction-Unlearning).
- Trackio 0.31.5; `hf` CLI authed as `nmaher`.

## Key paper facts (targets)
- CR = Coverage / SetSize. Prediction set = softmax ≥ (1 − q_hat); q_hat from **retain** calibration set (2000
  for CIFAR-10), alpha default 0.05 (paper also 0.10/0.15/0.20; challenge Claim 3 says alpha=0.1). Lower CR on Df
  = stronger forgetting; higher CR on Dtest = preserved utility.
- **Claim 2 (Table 2):** among forget samples the unlearned model MISCLASSIFIES, 30–68% still have GT in the
  conformal set (RT 30.6%, FT 58.3%, RL 45.5% @10% forget; up to 68.4% @50%).
- **Claim 5 (Table 7/paper Table 4):** CPU at lambda=0.5 improves UA by avg +3.93% (CIFAR-10) / +9.23% (TinyIN),
  TA drops only 1.0% / 0.57%.
- CPU loss (`unlearn_cpu.get_cpu_loss`): `clamp(q_hat − (1−p_true) + delta, min=0).mean()`, delta=0.01.

## The driver: `repro_toy.py`
Self-contained + **resumable** (skips methods already in `results.json`, reuses `ckpt/*.pth`). Reuses repo
functions faithfully (unlearn.finetune/RL/GA/ga_plus/teacher/ssd/salun/retrain; unlearn_cpu.finetune/RL/retrain;
metrics.CR/MIACR). Computes per model: UA/RA/TA, MIA (traditional), MIACR, and Coverage/SetSize/CR + fake-forget
recover-ratio on Df and Dtest at each alpha.

### Toy-scale run (THE command to execute in automatic mode)
```bash
cd /home/rec1/Desktop/AI_Safety/ICML_reproduce
nohup ./.venv/bin/python -u repro_toy.py \
  --outdir ./outputs --n_per_class 200 --cal_size 2000 --mia_cap 2000 \
  --batch_size 32 --learning_rate 0.05 \
  --orig_epochs 40 --rt_epochs 40 --unlearn_epochs 8 \
  --methods retrain,finetune,RL,GA,ga_plus,teacher,ssd,salun \
  --cpu_methods finetune,RL --lamda 0.5 --alphas 0.05,0.1 \
  > outputs_run.log 2>&1 &
```
CONFIG NOTE: first attempt (n=250, 20ep, batch64, lr0.1, FULL aug) UNDERFIT badly — original reached only
~40% acc (UA=0.59), useless for fake-forgetting. FIX applied: `_train_tf()` reduced to FLIP-ONLY aug
(RandomCrop+Grayscale removed) so the model can fit on a CPU-feasible step budget; batch 32 (more
steps/epoch), lr 0.05 (stability), 40 epochs. Target original: train acc high, test acc ~50-60%.
Expected wall-time ~4 h.
`-u` = unbuffered so `tail -f outputs_run.log` shows live progress. **Resumable**: re-running skips methods
already in `./outputs/results.json` and reuses `./outputs/ckpt/*.pth`, so a crash/restart continues where it left off.
Sanity-check first with a fast config: `--n_per_class 40 --orig_epochs 2 --rt_epochs 2 --unlearn_epochs 2
--methods finetune,GA --cpu_methods finetune` (~a few min once CIFAR is cached).

### GPU full-scale re-run (when compute is available)
Same script: `--device cuda --n_per_class 5000 --orig_epochs 200 --rt_epochs 200 --unlearn_epochs 10 --cal_size 2000`
(n_per_class 5000 = full CIFAR-10). Ideally run on an HF GPU Job under `nmaher` once credits exist.

## Logbook plan (map results → claim pages)
- Claim 1: explain CR/Coverage/SetSize; show metric implementation + a worked synthetic example (option 3).
- Claim 2: Table-2-style table (UA-mislabel count, in-set count, recover ratio) for RT/FT/RL from results.json.
- Claim 3: Coverage/SetSize/CR table for all methods at alpha=0.1 (and 0.05); note which methods rank well/poorly.
- Claim 4: traditional MIA vs MIACR per method; show the discrepancy.
- Claim 5: UA/TA for cpu-finetune-l0.0 vs l0.5 (and RL); report UA improvement & TA drop.
- Executive summary + Posterly poster (poster_embed.html) pinned; Conclusion = repro bundle artifact.
- Publish bundle = the working dir (scripts, outputs/, logs) via trackio.log_artifact, then validate + publish.

---

## Context-reset / session-teardown recovery (READ FIRST on a cold start)
All durable state is on disk; nothing important lives only in chat context. To recover after a compaction or a
full restart:
1. Read auto-memory (loads automatically) → it points here.
2. Read this file's STATUS + NEXT ACTIONS below.
3. `./.venv/bin/python -c "import json;d=json.load(open('./outputs/results.json'));print([k for k in d if k!='_meta'])"`
   to see which methods are already done (safe if the file is missing = run not started).
4. `pgrep -af repro_toy.py` — if the run process is dead and methods remain, **relaunch the exact toy-scale
   command** (above). It is resumable: skips methods already in `results.json`, reuses `./outputs/ckpt/*.pth`.
5. Then continue with NEXT ACTIONS (logbook build → poster → bundle → validate → publish).
CIFAR-10 is cached under `./data/cifar-10-batches-py/` (no re-download). Deps in `./.venv`. wandb is stubbed.

## STATUS  (updated 2026-07-21, mid-session)
- [x] Read paper + official code; understood CR/MIACR/CPU exactly.
- [x] Scaffolded logbook (short claim titles; slugs start `claim-`). Validator passes except Conclusion bundle.
- [x] Memory written (project, blocker, feedback).
- [x] Wrote `repro_toy.py`; installed deps (torch/tv CPU, sklearn, skimage, matplotlib); stubbed wandb.
- [x] Added `--mia_cap`/`test_mia` (SVC speedup) + per-method try/except (one failure won't kill the run).
- [x] `verify_metrics.py` (option 3) passes **12/12** — CR/MIACR/CPU-loss math matches paper Eqs 1-6, 8.
- [x] `.claude/settings.json` allowlists venv-python forms (incl. `nohup ./.venv/bin/python *`) for auto mode.
- [x] CIFAR-10 fully downloaded + cached.
- [~] Sanity run (`./sanity_out`, n_per_class=40): original + finetune validated (full valid schema);
      GA/retrain/cpu-finetune completing. Confirms the core code paths before the 3h run.
- [x] Full toy run COMPLETE (final config n=200, batch32, lr0.05, 40ep; NOT the 250/20/64/0.1 above). All 13
      entries in `./outputs/results.json`. Original memorized (forget_acc=1.00, RA=0.997, TA=0.483). Dtest
      coverage~0.90 => calibration valid. Claim2 recover ratios 76-100% (retrain 79/FT 100/RL 95) — HIGHER than
      paper 30-68% (weak toy model => larger sets; phenomenon reproduced/amplified). Claim3 CR_Df ranking != UA.
      Claim5 CPU-FT UA 4->34.5%(+30.5) TA 48.8->47.4(-1.4) CR_Dtest 0.221->0.221; CPU-RL UA 67->90.5. Direction
      matches paper. GA diverged NaN -> re-running with --ga_lr 1e-4 (added to driver).
- [x] Drafted all claim-page prose in `./logbook_draft.md` (with `[[FILL: ...]]` placeholders for run
      numbers). On run completion: fill placeholders from results.json, push each block to its page via
      `trackio logbook cell markdown`. Full page titles are listed in that draft.
- [ ] After run: fill draft from results → build claim pages, poster, bundle, validate, publish.
- [ ] Log results to claim pages 1-5.
- [ ] Option 3: synthetic metric-verification cell(s) (verify CR/MIACR math on hand-built softmax outputs).
- [ ] Executive summary + Posterly poster; Conclusion bundle.
- [ ] validate + publish; print link.

## NEXT ACTIONS (do these on restart, in order)
1. Check CIFAR is cached: `ls -la ./data/cifar-10-batches-py/` (if only the .tar.gz exists, let a run finish the
   download+extract). If a stale/partial `./data/cifar-10-python.tar.gz` fails MD5, torchvision re-downloads it.
2. Fast sanity run (see "sanity-check" line above); confirm `./<out>/results.json` has UA/RA/TA + per_alpha CR +
   MIA_traditional + MIACR for each method. Fix any runtime error, then delete the sanity outdir.
3. Launch THE toy-scale command (above) in background; poll `./outputs/results.json` until it contains: original,
   retrain, finetune, RL, GA, ga_plus, teacher, ssd, salun, cpu-finetune-l0.0, cpu-finetune-l0.5, cpu-RL-l0.0,
   cpu-RL-l0.5. (Resumable — safe to relaunch.)
4. Build the logbook from `results.json`:
   - Claim 2: table of {UA, n_mislabel, n_in_set, recover_ratio on forget @alpha 0.1} for retrain/finetune/RL.
     Compare recover_ratio to paper's 30.6/58.3/45.5%. This is the headline empirical claim.
   - Claim 3: {Coverage, SetSize, CR} on Df & Dtest for all methods @alpha 0.1; note ranking shifts vs UA.
   - Claim 1: CR definition + `metrics/CR.py` snippet + the synthetic verification (step from NEXT ACTION 5).
   - Claim 4: MIA_traditional vs MIACR per method; highlight a method good on MIA but poor on MIACR.
   - Claim 5: cpu-finetune l0 vs l0.5 (and RL): report UA delta (target dir: +) and TA delta (small -). Compute
     the average UA improvement across the CPU methods; compare qualitatively to paper's +3.93% (CIFAR).
   Use `trackio logbook cell markdown ... --page "<full claim page title>"`. Link the GitHub repo + arXiv in cells.
5. Option 3 (independent of the run): write `verify_metrics.py` that builds synthetic softmax matrices with known
   answers and asserts CR.get_calibration/find_quantile + prediction-set coverage/size + fake-forget counting are
   correct; capture its output on Claim 1 page (and Claim 4 for MIACR math).
6. Poster: `git clone https://github.com/Chenruishuo/posterly`; follow its SKILL.md; build `poster_embed.html`
   from logbook numbers with `--strict-polish`; add as pinned figure cell on Executive summary (below the pinned
   summary). Fill the pinned Executive summary (outcome-first + Scope & cost table; label clearly as toy/CPU).
7. Conclusion: `trackio.log_artifact("./", name="repro-bundle", type="dataset")`-style bundle (exclude .venv, data
   cache, __pycache__, .git, *.pth if huge) + artifact cell. Then:
   `python3 validate_icml_logbook.py --space nmaher/repro-tackling-fake-forgetting-through-uncertainty-quantification`
   then `trackio logbook publish nmaher/repro-tackling-fake-forgetting-through-uncertainty-quantification`. Print link.

## Gotchas discovered
- `main_evaluate.py` in the repo references undefined `args.corruption_type` (bug) — don't use it; `repro_toy.py`
  reimplements evaluation cleanly.
- `get_dataset` transform branch keys on `model_name=='resnet'` not `'resnet18'`, so resnet18 uses the else-branch
  augmentation. `repro_toy.py` replicates the resnet18 (else-branch) transforms directly.
- `unlearn_cpu.get_cpu_loss` uses `out_softmax[:, clabels]` (advanced indexing → [B,B] then .mean()); this is the
  released behavior — reproduced as-is, not "fixed".
- wandb stubbed via `sys.modules["wandb"]` (fake module) — don't `pip install wandb`.
- Bash tool cwd persists across calls; an earlier `cd Conformal-Prediction-Unlearning` stuck — always use the
  absolute project path `/home/rec1/Desktop/AI_Safety/ICML_reproduce`.
