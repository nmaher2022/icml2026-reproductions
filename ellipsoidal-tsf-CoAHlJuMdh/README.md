# Reproduction bundle — Ellipsoidal Time Series Forecasting

From-scratch reimplementation of Fern (Algorithm 1) — no official code release found. Toy-scale,
CPU-only. Paper: OpenReview [CoAHlJuMdh](https://openreview.net/forum?id=CoAHlJuMdh), also on
arXiv [2505.17370v6](https://arxiv.org/abs/2505.17370).

Trackio logbook: [HF Space](https://huggingface.co/spaces/nmaher/repro-ellipsoidal-time-series-forecasting).

## Verdict

| Claim | Outcome |
|---|---|
| 1. Nonstationary robustness (up to 790x over DLinear/Koopa) | **TOY-VERIFIED, partial** — clear on Lorenz-63 (1.4-1.5x lower MSE than DLinear), reversed on Chua (5.5-7.7x worse), mixed on Roessler |
| 2. Linear-time complexity via Householder SPD factorization | **VERIFIED** (exact, analytic FLOP sweep — not scale-dependent) |
| 3. Effective Prediction Time (EPT) longer than baselines | **REFUTED at toy scale** — reversed in all 6 tested scenarios |
| 4. Ablations (encoder/rotation/patching all necessary) | **TOY-VERIFIED** — direction and relative ordering both reproduce |
| 5. Geometric accuracy (SWD) persists past pointwise collapse | **INCONCLUSIVE** — the collapse mechanism isn't reached at toy-scale horizons (24-192 vs. paper's 720) |

Full numbers, per-scenario tables, and the self-audit account (one real bug found and fixed) are in
`VERDICTS.md` and `BUGFIX_LOG.md`.

## Contents
Flat-file layout (no `patches/`/`configurations/`/`logs/`/`results/` — this reproduction is a
handful of self-contained scripts, not a patch against upstream code):
- `PAPER_BRIEFING.md` — claims transcribed from the actual v6 PDF (the challenge's pre-extracted
  `claims_anchored.json` for this paper was found stale, not used — see the briefing's warning box).
- `data_gen.py` — Lorenz-63/Roessler/Chua RK4 generators (Appendix A.4.1).
- `fern_lib.py` — Fern (Algorithm 1), DLinear baseline, MSE/W2/SWD/EPT metrics (Appendix A.1),
  ablation variants.
- `run_claim1_and_3.py`, `run_claim2_complexity.py`, `run_claim4_ablations.py`,
  `run_claim5_horizon.py` — one script per claim group; each writes its own `claim*_results.json`
  and `.log`.
- `BUGFIX_LOG.md` — self-audit findings, including one real bug (an ablation confounded with a
  model-capacity difference) found and fixed before final results.
- `VERDICTS.md` — full per-claim verdicts with all numbers.
- `REPRO_LOG.md` — build/run history and resume notes.
- `poster.html` / `poster.png` / `poster_embed.html` — executive-summary poster.
- `paper_arxiv_2505.17370v6.pdf` / `paper_text.txt` — acquired paper source (arXiv fallback;
  OpenReview is bot-walled from this environment).

## Rerun
```
cd ellipsoidal-tsf-CoAHlJuMdh
../.venv/bin/python data_gen.py          # smoketest data generation
../.venv/bin/python run_claim1_and_3.py  # ~15 min
../.venv/bin/python run_claim2_complexity.py  # seconds
../.venv/bin/python run_claim4_ablations.py   # ~5 min
../.venv/bin/python run_claim5_horizon.py     # ~11 min
```
Invoke via `../.venv/bin/python`, not `uv run` — `uv run` on this torch-dependent script fetches a
fresh CUDA build from PyPI instead of reusing the repo's pre-installed CPU-only venv.
