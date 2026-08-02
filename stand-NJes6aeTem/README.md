# Reproduction bundle — STAND: Self-Aware Precondition Induction for Interactive Task Learning

STAND is a version-space-lattice symbolic classifier for learning action preconditions from few
examples in interactive task learning (ITL), combining a "general" set of near-tied decision
splits with a "specific" set of per-leaf constant-feature extensions into a certainty-calibrated
prediction, plus an optional hierarchical-shrinkage regularizer. No official code release was
found, so this reproduction is a from-scratch reimplementation (`stand_lib.py`, `data_gen.py`)
built directly from the paper's equations (Sections 3-4, Appendices A-B).

Paper: OpenReview [NJes6aeTem](https://openreview.net/forum?id=NJes6aeTem). arXiv
[2409.07653](https://arxiv.org/abs/2409.07653) — **note**: the challenge's own claim extraction was
generated against arXiv v1 (Sept 2024); this reproduction targets the current v2 (Feb 2026), a
substantially rewritten paper (added the synthetic benchmark, hierarchical shrinkage, new
baselines). Both PDFs are included; see `PAPER_BRIEFING.md` and `VERDICTS.md` for detail.

Trackio logbook: [HF Space](https://huggingface.co/spaces/nmaher/repro-stand-self-aware-precondition-induction-for-interactive-task-learning).

## Verdict

Toy-scale throughout: 15 reps (paper: 100) for the synthetic benchmark, 12 reps for the λp sweep,
10 reps for the UCI benchmark (paper appears to use a single 80/20 split — 10 reps here is a
strengthening). Full comparison tables and narrative analysis in `VERDICTS.md`.

| Claim | Outcome |
|---|---|
| 1. Synthetic-task accuracy (§6.1, Table 1) | **TOY-VERIFIED** — ranking exact, magnitudes uniformly ~8-11pts high |
| 2. Error reoccurrence (§6.2) | **TOY-VERIFIED (partial)** — FP direction matches, FN "not best" nuance doesn't |
| 3. Productive monotonicity (§6.3) | **TOY-VERIFIED (partial)** — STAND-family on top, but DT/XGBoost not near 50% as paper implies |
| 4. Calibration at ~100% (§6.4) | **TOY-VERIFIED** — STAND-hs near-perfect precision, NeuralNet clear outlier |
| 5. Active learning utility (§6.5) | **TOY-VERIFIED (weak)** — top/bottom tier split holds, phased sub-claim untested, high variance |
| 6. λp hyperparameter sensitivity (Appx D) | **TOY-VERIFIED** — higher λp helps accuracy + reoccurrence, matches paper's framing |
| 7. UCI noisy-dataset benchmark (Appx E) | **TOY-VERIFIED (weak)** — "doesn't fail" holds, cross-model ranking does not reproduce |
| 8. Real ITL domains (Dice Adventure, Fractions, MC Addition) | **BLOCKED** — proprietary VAL/AI2T-TutorGym infrastructure unavailable |

No claim is a clean REFUTED. Four implementation bugs were found and fixed during self-audit
(`BUGFIX_LOG.md`) — most seriously, Agr_G(x) (Eq. 7) silently collapsed to exactly 0 for
reject-branch traversals through unambiguous single-split nodes, degenerating
hierarchical-shrinkage STAND to a trivial majority-class predictor until fixed.

## Contents

- `PAPER_BRIEFING.md` — claim transcription, math summary, known access blockers.
- `VERDICTS.md` — per-claim verdicts with paper-vs-ours comparison tables.
- `BUGFIX_LOG.md` — 4 implementation bugs found and fixed, with root cause and verification.
- `REPRO_LOG.md` — recovery/status log written during the run.
- `stand_lib.py`, `data_gen.py` — the from-scratch STAND implementation and synthetic data generator.
- `run_synthetic_experiment.py` — Claims 1-5 (accuracy, error reoccurrence, productive
  monotonicity, calibration, active learning), writes `synthetic_results.json`.
- `uci_benchmark.py` — Claim 7 (6-dataset UCI noisy benchmark), writes `uci_results.json`.
- `lambda_p_sweep.py` — Claim 6 (λp hyperparameter sweep), writes `lambda_p_sweep_results.json`.
- `poster.html` / `poster.png` — summary poster.
- `paper-arxiv-2409.07653v1.pdf`, `paper-arxiv-2409.07653v2.pdf` — both paper versions (see
  staleness note above).

## Rerun

Each script is a self-contained PEP-723 file; no persistent virtualenv is required:

```
uv run run_synthetic_experiment.py   # -> synthetic_results.json (Claims 1-5)
uv run uci_benchmark.py              # -> uci_results.json (Claim 7)
uv run lambda_p_sweep.py             # -> lambda_p_sweep_results.json (Claim 6)
```

For long runs, launch detached: `nohup uv run <script>.py > <script>.log 2>&1 < /dev/null & disown`.
