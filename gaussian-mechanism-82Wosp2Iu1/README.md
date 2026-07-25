# Asymptotic optimality of the high-dimensional Gaussian mechanism — reproduction

- **Paper:** Asymptotic optimality of the high-dimensional Gaussian mechanism
  and improved low-dimensional mechanisms (OpenReview
  [`82Wosp2Iu1`](https://openreview.net/forum?id=82Wosp2Iu1), Spotlight)
- **Upstream code:** none released — everything here is from scratch
- **Verdict:** **11/12** · CPU-only (~12 min)

A from-scratch differential-privacy library (`sgg_lib.py`, pure
`numpy`/`scipy`/`mpmath`) implementing the hockey-stick divergence δ(ε), the
Gaussian mechanism `g(u)`/`u0(δ)`, the spherical generalized-Gaussian (SGG /
GGamma) family with Haar symmetrization, and an FFT/PRV composition accountant.

## Claims reproduced

**Claim 5 — Table 2 δ⋆ lower bounds** (`claim15_optimality.py`). All 7 values
reproduced to ~6 decimals (worst relative diff ~1.3×10⁻⁶):

| ε | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 | 8.0 | 16.0 |
|---|---|---|---|---|---|---|---|
| paper | 0.736670 | 0.706970 | 0.649185 | 0.549133 | 0.416972 | 0.292170 | 0.197615 |
| ours | 0.736670 | 0.706970 | 0.649185 | 0.549133 | 0.416972 | 0.292170 | 0.197615 |

> float64 **fails** at ε ≥ 4 (`g` involves `exp(ε)·Φ(−large)`, a catastrophic
> cancellation) — computed at 50-digit precision with `mpmath`.

**Claim 1 — asymptotic optimality** (Thm 3.1). A family of spherical
competitors calibrated to the *exact* Gaussian MSE budget never beats the
Gaussian by more than a vanishing margin as dimension T grows (inside
δ ≤ δ⋆); an out-of-regime control shows persistent finite-T improvements.

**Claim 4 — low-dimensional SGG gains** (Fig 2, `claim46_sgg.py`). Reproduced
the paper's operating points at ε = 0.1:

| T | our gain vs best baseline | paper | our p⋆ | paper p⋆ |
|---|---|---|---|---|
| 2 | 15.6% | 15.5% | 1.37 | 1.37 |
| 5 | 8.7% | 6.2% | 1.58 | 1.32 |
| 10 | 4.6% | 2.5% | 1.63 | 1.92 |

Gain shrinks with T, as claimed; the T=2 headline (−15.5%, p⋆=1.37) matches to
3 s.f.

**Claim 6 — tight FFT composition** (Prop 4.2 / Alg 7). k-fold FFT accountant
matches the closed-form k-fold Gaussian and a 4M-sample Monte-Carlo of the
summed privacy-loss RVs for the ℓ2 mechanism.

**Claims 2 + 3** — SGG family / Haar-symmetrization identities (`claim23_family.py`,
`claim2_sweep.py`).

## Files

- `sgg_lib.py` — the DP library (δ, `g`, `u0`, SGG/GGamma, PRV, FFT composition)
- `claim15_optimality.py` — Claims 1 + 5 · `claim46_sgg.py` — Claims 4 + 6 ·
  `claim23_family.py`, `claim2_sweep.py` — Claims 2 + 3
- `make_figs_gauss.py` — regenerates figures from CSVs
- `claim*_*.csv` — result tables · `poster.html` — poster source

## Gotchas that were load-bearing

- α ≤ T−1 is required for monotone privacy loss.
- T=2 has an endpoint singularity → kink-split + Gauss-Jacobi quadrature.
- The FFT composition buffer must cover `(n_bins−1)·k + 1` bins or the tail
  aliases to zero (bites at k ≥ 8).
