# Reproduction bundle — ICML 2026 #2446 "Tackling Fake Forgetting through Uncertainty Quantification"

CPU toy reproduction of the official code ([TIML-Group/Conformal-Prediction-Unlearning](https://github.com/TIML-Group/Conformal-Prediction-Unlearning),
paper [arXiv:2501.19403](https://arxiv.org/abs/2501.19403)). Logbook Space:
`nmaher/repro-tackling-fake-forgetting-through-uncertainty-quantification`.

## Contents
- `repro_toy.py` — self-contained, resumable driver. Trains a ResNet-18 on a CIFAR-10 subset, runs the official
  unlearning methods (retrain/finetune/RL/GA/ga_plus/teacher/ssd/salun) and the CPU framework (unlearn_cpu), and
  evaluates UA/RA/TA, traditional MIA, MIACR, and Coverage/Set-Size/CR + fake-forget recovery on Df and Dtest.
- `verify_metrics.py` — synthetic, model-free verification that the official CR/MIACR/CPU-loss code matches the
  paper's equations (Eqs 1-6, 8). 12/12 checks pass.
- `build_tables.py` / `make_figs.py` / `build_poster.py` — turn `outputs/results.json` into the logbook tables,
  the three claim figures, and the reproduction poster.
- `outputs/results.json` — all metrics for the 13 model variants (the numbers behind every claim page).
- `figs/` — claim figures. `poster.html` — the reproduction poster source.
- `pagebodies/` — the markdown pushed to each claim page. `REPRO_LOG.md` — full handoff log + gotchas.
- `ga_rerun.log` — GA-method rerun log.

NOT included (regenerable / bulky): the rendered `poster.png`, the 391 KB `outputs_run.log`, the 13 `*.pth`
checkpoints (~45 MB each — re-created by `repro_toy.py`), the CIFAR-10
download cache, and the `.venv`.

## Rerun (CPU toy — reproduces this bundle)
```bash
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision scikit-learn scikit-image matplotlib numpy tqdm
git clone https://github.com/TIML-Group/Conformal-Prediction-Unlearning
python repro_toy.py --outdir ./outputs --n_per_class 200 --cal_size 2000 --mia_cap 2000 \
  --batch_size 32 --learning_rate 0.05 --orig_epochs 40 --rt_epochs 40 --unlearn_epochs 8 \
  --methods retrain,finetune,RL,GA,ga_plus,teacher,ssd,salun --cpu_methods finetune,RL --lamda 0.5 --alphas 0.05,0.1
python verify_metrics.py && python build_tables.py
```
The driver is resumable (skips methods already in `outputs/results.json`, reuses `outputs/ckpt/*.pth`).

## Rerun at PAPER SCALE (needs a GPU)
The same script, unchanged, reproduces the paper's setup:
```bash
python repro_toy.py --device cuda --n_per_class 5000 --orig_epochs 200 --rt_epochs 200 --unlearn_epochs 10 \
  --batch_size 64 --learning_rate 0.1 --cal_size 2000 ...
```
(`n_per_class 5000` = full CIFAR-10.) The FF method and the Tiny-ImageNet + ViT half of Claim 5 also require GPU.
Ideally run as a Hugging Face GPU Job under your namespace once credits are available.
