# A studentized spherical-harmonics nonparametric two-sample test — reproduction

- **Paper:** A Studentized Spherical Harmonics–Based Nonparametric Two-Sample
  Test for Compositional and Directional Data (OpenReview
  [`QsxpsAu7l1`](https://openreview.net/forum?id=QsxpsAu7l1))
- **Upstream code:** authors' repo `nonparametric_compositional_two_sample_test`
  (cloned locally, **not vendored here**)
- CPU-only. Published to the ICML-2026-agent-repro challenge.

The paper's studentized spherical-harmonic energy-distance statistic `T_p_mn` is
audited on three fronts.

## Claims reproduced

**Claim 1 — one statistic, both data types** (`claim1_applicability.py`). A
single studentized spherical-harmonic energy-distance statistic `T_p_mn`
applies — with only a change of embedding map — to *both* compositional
(simplex) and directional (sphere) data.

**Claim 2 — asymptotic normality without calibration**
(`claim2_asymptotic_normality.py`). Under H0, `T_p_mn` converges in
distribution to `N(0,1)` **without** any permutation or bootstrap step; checked
directly against the standard normal (size at nominal level, Q–Q agreement).

**Claim 3 — power and cost** (`claim3_power_comparison.py`). The test shows
improved power in certain compositional/directional scenarios relative to
standard baselines, while being dramatically cheaper — because Claim 2 already
gives an analytic `N(0,1)` calibration, there is no permutation loop.

## Files

- `claim1_applicability.py`, `claim2_asymptotic_normality.py`,
  `claim3_power_comparison.py` — the three audits (results printed to stdout /
  captured in the published logbook)
- `poster.html` — executive-summary poster source
- `measure.json`, `GATE_REPORT.json` — poster layout / style-gate report

## Setup (not fully vendored)

Runs against the authors' released code. Clone
`nonparametric_compositional_two_sample_test` beside these scripts, then run
each `claim*.py` under the project's Python environment (`numpy`/`scipy`).
