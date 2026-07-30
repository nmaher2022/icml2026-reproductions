# CausalProfiler: Generating Synthetic Benchmarks for Rigorous and Transparent Evaluation of Causal Machine Learning — reproduction

- **Paper:** CausalProfiler: Generating Synthetic Benchmarks for Rigorous and
  Transparent Evaluation of Causal Machine Learning (OpenReview
  [`0wCl7EifsY`](https://openreview.net/forum?id=0wCl7EifsY))
- **Upstream code:** official `causal_profiler` library, used directly for
  Claims 1/2/5; baseline methods (DCM, CausalNF, VACA) are third-party
  upstream code, cloned locally but **not vendored** in this folder — see
  `baselines/*/REPORT.md` for the exact repos/commits used.
- **Verdict:** **mostly confirmed** — Claims 1, 2, 5 confirmed directly
  against the official library; Claims 3, 4 partially confirmed (qualitative
  story holds, one specific numeric comparison per claim doesn't hold at toy
  scale). CPU-only throughout.
- **GitHub (code):** https://github.com/nmaher2022/repro-causalprofiler
- **Trackio Logbook (full write-up):** https://huggingface.co/spaces/nmaher/repro-causalprofiler-generating-synthetic-benchmarks-for-rigorous-and-transparent-evaluation-of

## Claims reproduced

| # | Claim | Verdict-leaning |
|---|---|---|
| 1 | Space of Interest covers all 3 levels of Pearl's causal hierarchy (L1/L2/L3) | **Confirmed** — directly against `causal_profiler`, no baselines needed |
| 2 | Proposition 5.1 coverage guarantee | **Confirmed** — directly against `causal_profiler` |
| 3 | Experiment 1: DCM vs VACA on diverse SCMs (Linear-Medium / NN-Medium) | **Partially confirmed** — DCM's Linear-Medium advantage over VACA reproduces cleanly (0.206 vs 0.550 mean error), VACA's error drops ~8x on NN-Medium vs Linear-Medium as expected, but the paper's claim that VACA beats DCM on NN-Medium does not hold at toy scale (DCM 0.055 vs VACA 0.069) — plausibly a graph-size effect (paper: 15-20 nodes; toy: 5-6 nodes) |
| 4 | Experiment 2: CausalNF vs DCM on Regional Discrete SCMs | **Partially confirmed** — CausalNF's error rises up to ~5.7x with increasing discreteness while DCM's does not, matching the paper's qualitative story, but the paper's specific NaN-failure-rate gap did not surface at toy scale (both ~0%) |
| 5 | Regional Discrete SCMs are a genuinely novel model class | **Confirmed** — directly against `causal_profiler` |

Full narrative, per-claim methodology, and figures are in the Trackio
logbook linked above; this folder holds the code and raw results behind it.

Along the way we found and documented four implementation-level details not
obvious from the paper text alone (see `executive-summary` in the logbook
for the full list): the `graph` object from `generate_samples_and_queries()`
is index-keyed rather than name-keyed; `number_of_noise_regions=1` raises an
`AssertionError` in the installed library version for
TABULAR+DISCRETE SCMs; the `"V_to_PA"` region-count expression can blow up
combinatorially on small but densely-connected toy graphs; and VACA's pinned
`torch==1.10.0+cpu` fails to import on modern kernels, needing a newer CPU
torch/PyG build.

## Layout

- `baselines/causalnf/`, `baselines/dcm/`, `baselines/vaca/` — adapter code
  wiring each baseline method into the paper's own evaluation harness
  (`causal_nf_method.py`, `dcm_method.py`, `vaca_method.py`), toy-scale run
  scripts, results JSON, and a per-baseline `REPORT.md` with the exact
  upstream repo/commit and environment notes. Upstream baseline source code
  itself is **not** included here (see each `REPORT.md`).
- `experiments/` — one driver per claim: `claim1_pch_levels.py`,
  `claim2_coverage.py`, `claim5_regional_discrete_scm.py`.
- `results/` — raw JSON output from Claims 1, 2, 5.

## Running

```
cd causalprofiler-0wCl7EifsY
python experiments/claim1_pch_levels.py
python experiments/claim2_coverage.py
python experiments/claim5_regional_discrete_scm.py
```

Claims 1/2/5 depend only on the official `causal_profiler` library
(CPU-only, no GPU required). Claims 3/4 additionally require the baseline
methods set up per `baselines/*/REPORT.md` — VACA in particular needs an
isolated Python 3.9 conda env with a newer CPU torch/PyG build than its
pinned `torch==1.10.0`.
