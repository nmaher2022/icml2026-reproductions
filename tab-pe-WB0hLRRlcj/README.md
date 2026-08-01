# Reproduction bundle — Differentially Private Synthetic Data via APIs 4: Tabular Data

Reproduces Tab-PE, a differentially-private tabular-data synthesis method built on the Private Evolution (PE) framework, using the paper's own official code: [microsoft/DPSDA](https://github.com/microsoft/DPSDA) at commit [`9078c67`](https://github.com/microsoft/DPSDA/commit/9078c67995499e6769113780200bbf1d788d3d60), unmodified.

Paper: OpenReview [WB0hLRRlcj](https://openreview.net/forum?id=WB0hLRRlcj)
Paper also on arXiv: [2606.08259](https://arxiv.org/abs/2606.08259).

Trackio logbook: [HF Space](https://huggingface.co/spaces/nmaher/repro-differentially-private-synthetic-data-via-apis-4-tabular-data)

## Verdict

| Claim | Outcome |
|---|---|
| 1. Artificial Characters: Tab-PE beats AIM | **VERIFIED** (full paper scale) |
| 2. Person Activity: Tab-PE beats AIM | **VERIFIED** (full paper scale) |
| 3. Compute efficiency: CPU-only / ~Nx faster than AIM | **TOY-VERIFIED** (CPU-only claim) / **BLOCKED** (precise multiplier — see `BUGFIX_LOG.md` §1c) |
| 4. XOR stress test: AUC≈0.8 at 5 features | **BLOCKED** at 4-5 features (TabPFN license-gated, XGBoost substitute inconclusive — see `BUGFIX_LOG.md` §1b) / **TOY-VERIFIED** at 1-3 features |
| 5. Algorithm 2 structure (two-stage selection schedule) | **VERIFIED** (code inspection) |

Full detail, tables, and reasoning for each verdict: `VERDICTS.md`.

## Contents

- `PAPER_BRIEFING.md` — paper's claims/math as transcribed before running anything.
- `REPRO_LOG.md` — chronological run log.
- `BUGFIX_LOG.md` — every substitution, scope decision, and investigation (TabPFN → XGBoost substitution, the Claim 3 timing confound, etc.).
- `VERDICTS.md` — final verdict for each of the 5 claims with supporting numbers.
- `build_poster.py`, `poster.png`, `poster_embed.html` — poster generation for the Trackio logbook.
- `scripts/` — the actual reproduction code: `artificial_characters.py` / `person_activity.py` (Tab-PE, Claims 1/2), `aim_baseline.py` / `aim_eval_person_activity_resume.py` (AIM baseline, Claims 1/2/3), `xor_stress_test_xgb.py` / `xor_reeval_depth_matched.py` (Claim 4), `run_all.sh` (sequential driver).
- `results/` — `log.txt` + `eval.json`/timing JSON per run (per-iteration checkpoints and synthetic CSVs are gitignored — re-generatable, not evidence).
- `logs/` — raw stdout logs from the AIM baseline runs and the full `run_all.sh` sequence.
- `patches/`, `configurations/` — empty; no changes were needed to the upstream code or its default configs to reproduce these claims.
- `DPSDA_upstream/`, `DPSDA_upstream_aim/` — gitignored vendored clones of the official repo (see `.gitignore`); not committed.

## Rerun

```bash
git clone https://github.com/microsoft/DPSDA.git DPSDA_upstream
cd DPSDA_upstream && git checkout 9078c67995499e6769113780200bbf1d788d3d60 && cd ..
python3 -m venv .venv && source .venv/bin/activate
pip install -e DPSDA_upstream

cd scripts
python3 artificial_characters.py       # Claim 1 (Tab-PE, Artificial Characters)
python3 person_activity.py             # Claim 2 (Tab-PE, Person Activity)
python3 aim_baseline.py --dataset artificial_characters   # Claim 1/3 (AIM baseline)
python3 aim_baseline.py --dataset person_activity         # Claim 2/3 (AIM baseline)
for f in 1 2 3 4 5; do python3 xor_stress_test_xgb.py --num-features "$f"; done  # Claim 4
```

`run_all.sh` runs the Tab-PE + XOR steps sequentially (AIM baselines are run separately to avoid CPU oversubscription — see comments in the script). No GPU required for any step.
