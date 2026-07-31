# Reproduction bundle — ICML 2026 #27304 "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"

CPU-only reproduction using the authors' own code
([kellian-cottart/active-continual-learning-bayesianbinn](https://github.com/kellian-cottart/active-continual-learning-bayesianbinn),
base commit `1b8a1a1`), paper [OpenReview SPZd0HVyiS](https://openreview.net/forum?id=SPZd0HVyiS)
(arXiv [2605.30198](https://arxiv.org/abs/2605.30198)). Full narrative write-up, run logs, and session
trace are in the published Trackio logbook:
**[nmaher/repro-active-continual-learning-with-metaplastic-binary-bayesian-neural-networks](https://huggingface.co/spaces/nmaher/repro-active-continual-learning-with-metaplastic-binary-bayesian-neural-networks)**.

## Verdict

| Claim | Outcome |
|---|---|
| 1. OpenLORIS-8192 label efficiency (32x reduction) | **Blocked** — dataset's Google Drive link requires authenticated sign-in (`gdown.download()` raises `FileURLRetrievalError`). Not attempted, not faked. |
| 2. Permuted-MNIST 1000-task accuracy & OOD AUC vs BayesBiNN | **Toy-scale (n_tasks=10) mixed result.** OOD-AUC direction matches the claim (0.996 vs 0.979 epistemic AUC, BiMU higher); accuracy-retention direction does **not** (BiMU 66.66% avg vs BayesBiNN 77.47% avg) — most likely because the paper's claimed 1000-task catastrophic-forgetting regime doesn't manifest at n=10. |
| 3. BiMU relaxation term prevents posterior saturation | **Code-level mechanism verified** against `optimizers/bimu.py` (Eq. 6-7); the long-horizon causal consequence (that this specifically prevents 1000-task collapse) is plausible but not independently tested — requires the full 1000-task run we couldn't complete on CPU. |
| 4. OpenLORIS-1024 accuracy vs BayesBiNN | **Blocked** — same OpenLORIS dataset-access issue as Claim 1. |
| 5. Animals active learning with variation-ratio querying | **Blocked** — dataset requires a Kaggle API token we do not have (confirmed still live via web search, so this is an access gap, not a dead dataset). |

Full derivations, honest-comparison discussion, and two corrections found during an independent
re-audit of the write-up (the `sum_grads` gate is **global**, not per-parameter; the OOD-AUC values
are a **per-task-boundary trajectory**, not repeated final-model probes) are in the logbook's Claim 2
and Claim 3 pages.

## Contents
- `patches/cpu-patches.diff` — the two one-line patches needed to run the authors' code CPU-only:
  `main.py` (`jax_platform_name` `gpu`→`cpu`) and `utils/gpuLoading.py` (`OpenLORIS(...)` now passes
  `device=self.device` through instead of defaulting to `cuda:0`).
- `configurations/toy-pmnist-10tasks-100neurons/{bimu,bayesbinn}.json` — the toy-scale configs used
  for Claims 2/3, identical to the paper's own `configurations/main-pmnist-1000tasks-100neurons/`
  configs except `n_tasks` (1000→10) and `max_parallel_permutation` (50/200→10).
- `logs/bimu_toy10tasks.log`, `logs/bayesbinn_toy10tasks.log` — full stdout from the two toy runs
  (~2h01m wall time each, CPU-only).
- `logs/bimu_1task_smoke.log` — a 1-task/60k-step smoke test (~6:49 wall time), used to estimate that
  the full 1000-task protocol is a multi-day single-CPU run and is infeasible here.
- `results/{bimu,bayesbinn}/accuracy/` — raw per-task test-accuracy `.npy` arrays (`split=0-task={k}-epoch=0.npy`
  and the training-iteration companions) from the toy runs.
- `results/{bimu,bayesbinn}/uncertainty/` — raw OOD (Fashion-MNIST) ROC-AUC `.npy` arrays for the
  epistemic/aleatoric/variation-ratio uncertainty signals, one file per task boundary.

NOT vendored here: the upstream repo itself (`kellian-cottart/active-continual-learning-bayesianbinn`) —
see "Rerun" below to clone and patch it. Also not included: OpenLORIS/Animals data or results (blocked,
see Claims 1/4/5 above).

## Rerun (CPU toy — reproduces this bundle)
```bash
git clone https://github.com/kellian-cottart/active-continual-learning-bayesianbinn repro-bimu
cd repro-bimu
git checkout 1b8a1a1
git apply ../active-continual-learning-bimu-SPZd0HVyiS/patches/cpu-patches.diff
# create environment per the repo's environment.yml, plus `dm_pix` (undocumented dependency
# required by utils/dataFunctions.py but missing from environment.yml)
mkdir -p configurations/toy-pmnist-10tasks-100neurons
cp ../active-continual-learning-bimu-SPZd0HVyiS/configurations/toy-pmnist-10tasks-100neurons/*.json \
   configurations/toy-pmnist-10tasks-100neurons/
python3 main.py -c toy-pmnist-10tasks-100neurons/bimu -it 1 -fits -ood fashion -v
python3 main.py -c toy-pmnist-10tasks-100neurons/bayesbinn -it 1 -fits -ood fashion -v
```
(`-fits`/`--fits_in_memory` is required for `main.py`'s Permuted-MNIST reshape path — the authors' own
`scripts/main-pmnist-table.sh` always passes it.)

## Rerun at PAPER SCALE (needs a GPU, and OpenLORIS/Animals credentials for Claims 1/4/5)
Revert the `jax_platform_name`/`device` patches (or point them at `gpu`/`cuda:0`) and use the
authors' own `configurations/main-pmnist-1000tasks-100neurons/{bimu,bayesbinn}.json`,
`configurations/main-openloris-8192/`, `configurations/main-openloris-1024/`, and the Animals active-learning
config — ideally as a Hugging Face GPU Job. OpenLORIS-Object needs authenticated Google Drive access;
the Animals dataset needs a Kaggle API token.
