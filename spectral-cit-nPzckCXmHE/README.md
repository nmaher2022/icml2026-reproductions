# Reproduction bundle — Toward Scalable and Valid Conditional Independence Testing with Spectral Representations

Independent, from-scratch reproduction of [`nPzckCXmHE`](https://openreview.net/forum?id=nPzckCXmHE)
(arXiv:[2512.19510](https://arxiv.org/abs/2512.19510)), for the
**[ICML-2026-agent-repro](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge)**
challenge. Logbook Space: [`nmaher/repro-toward-scalable-and-valid-conditional-independence-testing-with-spectral-representations`](https://huggingface.co/spaces/nmaher/repro-toward-scalable-and-valid-conditional-independence-testing-with-spectral-representations).

No baseline/authors' code was consulted; every result comes from an independent implementation of
Algorithm 1 (bi-level contrastive representation learning for conditional independence testing:
inner/outer loop training, post-hoc whitening, dimension-pruned test statistic) built by literal
reading of the paper's equations and pseudocode — CPU-only, no GPU available for this
reproduction.

## Verdict summary (5 claims)

| Claim | Topic | Verdict |
|---|---|---|
| 1 | Thm 4.1: under H0, T̂_n → χ²(d²) as m,n→∞ | **TOY-VERIFIED** — conservative calibration matching the paper's own admission; pruned-statistic Type I error 6.7% vs. nominal 5% |
| 2 | Thm 4.2: non-asymptotic power tied to E_m and n | **TOY-VERIFIED** — power climbs 0%→100% between signal strengths 0.05→0.15, matching the paper's Fig. 11 shape |
| 3 | E_m^val/E_m^pow definitions (p.5) control validity/power | **TOY-VERIFIED (partial), disclosed limitation** — definitions implemented faithfully, but E_val stayed pinned ≈1.0 across all 120 trials even where calibration was reasonable |
| 4 | Algorithm 1: bi-level training + whitening | **TOY-VERIFIED** — implementation matches the pseudocode line-for-line (structural check), *and* an added empirical check (synthetic ground-truth partial-covariance data, 10 reps) confirms the trained representations actually recover the claimed leading spectral directions of Σ_{X,Y\|Z} far better than dimension-matched noise controls (u: CCA 0.891 vs. 0.549 vs. 0.215 random; v: 0.849 vs. 0.553 vs. 0.220), not just that the code parses like the box |
| 5 | Assumption 4.1: validity requires bounded (Tanh) activations | **INCONCLUSIVE** — Tanh→Identity ablation showed no Type I error inflation, but the ablation confounds boundedness with loss of all nonlinear capacity (`nn.Identity` collapses an MLP to a linear map) |

Full write-up with all numbers, two real bugs found and fixed during self-audit, and the
documented parameterization assumption: see the
[published logbook](https://huggingface.co/spaces/nmaher/repro-toward-scalable-and-valid-conditional-independence-testing-with-spectral-representations),
`VERDICTS.md`, and `BUGFIX_LOG.md` in this folder.

Real-data claim (TCGA-BRCA, Section 5.2) is **not** in the extracted claims list for this
OpenReview id and requires a Path Foundation Model image encoder + restricted TCGA access — out
of scope, not attempted.

## Contents

- `scit_lib.py` — core implementation: `SpectralModel` (u_θ/v_θ/w_θ MLPs + M/N scale
  parameters), `train_spectral_model()` (Algorithm 1's bi-level warmup/alternation loop),
  `whiten()` (population-level inverse-square-root whitening), `test_statistic()` /
  `test_statistic_pruned()` (Eq. 10 and the Appendix-C dimension-pruning correction),
  `validation_error()` (E_m^val diagnostic).
- `data_gen.py` — synthetic benchmark generators: `signal_strength_ablation()` (Fig. 11, used for
  Claims 1/2/5), plus `post_nonlinear_model()` and `high_dim_nonsmooth()` (Figs. 2/10, implemented
  but not run — lower priority, not required for the 5 in-scope claims).
- `smoketest.py` — tiny-scale plumbing check (d=2, N=200, 5 epochs) run before any full-scale
  compute, per the harness's smoketest-before-scale rule.
- `claim1_2_signal_ablation.py` — Claims 1 & 2 (validity + power), d=10, d_z=3, N=1000, 30 reps ×
  4 conditions (H0 + 3 signal strengths). Writes `claim1_2_raw.csv` / `claim1_2_summary.csv`.
- `claim5_subgaussian_ablation.py` — Claim 5 (sub-Gaussianity ablation), Tanh vs. Identity
  activation, 30 reps each. Writes `claim5_raw.csv` / `claim5_summary.csv`.
- `claim4_spectral_verification.py` — Claim 4 empirical check (added after an external judge
  flagged the original "VERIFIED (structural)" verdict as untested behavior): synthetic
  jointly-Gaussian (X,Y,Z) with an exact closed-form Σ_{X,Y|Z} (Schur-complement construction, rank
  2, singular values [3.0, 1.5]), trains the real unmodified `scit_lib.py` Algorithm 1 on it, and
  compares learned-embedding canonical correlation against the true signal directions vs.
  dimension-matched noise-direction and random controls. 10 reps. Writes `claim4_raw.csv` /
  `claim4_summary.csv`.
- `*.csv` — raw per-trial and per-condition result tables behind every claim.
- `*.run.log` / `*_run.log` — full stdout of both background runs (live per-trial progress).
- `PAPER_BRIEFING.md` — Step 2 briefing: paper identity, transcribed math (losses, Algorithm 1,
  test statistic, Table 2 hyperparameters), the 5 in-scope claims verbatim from
  `claims_anchored.json`.
- `VERDICTS.md` — full per-claim verdict write-up (this README's summary table expanded with all
  numbers and reasoning).
- `BUGFIX_LOG.md` — 2 real bugs found and fixed during self-audit (a mis-specified E_val
  cross-covariance term; a missing Appendix-C "dimension pruning" step that turned out to be
  load-bearing for calibration) plus 1 documented parameterization assumption (M_θ/N_θ's
  gradient-update rule, not specified in Algorithm 1's pseudocode).
- `REPRO_LOG.md` — session-survival recovery log for the two long-running background jobs.

## How to run

Every script is self-contained (PEP-723 header, dependencies `numpy`, `torch`, `scipy`). This
reproduction used CPU-only torch (no GPU available); running via plain `uv run` on a machine
without a CPU-torch index pre-configured may fetch a GPU-flavored torch build instead — if that's
a problem in your environment, install `numpy`/`torch`/`scipy` into a venv first and run with that
interpreter directly instead of `uv run`.

```bash
uv run smoketest.py                        # ~5s, tiny-scale sanity check
uv run claim1_2_signal_ablation.py 30       # Claims 1-2, ~44 min (30 reps × 4 conditions)
uv run claim5_subgaussian_ablation.py 30    # Claim 5, ~10 min (30 reps × 2 activations)
uv run claim4_spectral_verification.py 10   # Claim 4, ~7 min (10 reps, synthetic ground truth)
```

Both experiment scripts accept an optional rep-count argument (default 30, the paper uses 500 —
this reproduction used a reduced Monte Carlo scale, stated explicitly in every verdict).

## Reproducibility notes

- All results use the paper's own reference hyperparameters (Table 2, Appendix C):
  `output_dim=10, n_hidden=2, layer_size=128, lr_inner=3e-5, lr_outer=2.1e-3, reg_inner=3.3,
  reg_outer=1.9, batch_size=128, n_epochs=400, warmup_steps=100, perc_dim_prune=0.9`.
- Reduced scale vs. the paper throughout: 30 Monte Carlo reps (paper: 500), 1 of the paper's 3
  synthetic benchmarks (the cheapest, Fig. 11's signal-strength-ablation), 3 signal-strength
  points instead of a finer sweep. Stated explicitly in every TOY-VERIFIED verdict, never rounded
  up to VERIFIED.
- Claim 5's ablation design has a disclosed confound (Identity activation removes nonlinearity,
  not just boundedness) — reported as INCONCLUSIVE rather than forced into REFUTED/VERIFIED; see
  `VERDICTS.md` for the honest reasoning and a suggested cleaner follow-up (LeakyReLU/ELU).
