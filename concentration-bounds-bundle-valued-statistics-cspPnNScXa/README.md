# Sharp concentration bounds for bundle-valued statistics on manifolds — reproduction

- **Paper:** Sharp Concentration Bounds for Bundle-Valued Statistics on
  Manifolds (Das & Snasel, ICML 2026 · arXiv:2607.10592 · OpenReview
  [`cspPnNScXa`](https://openreview.net/forum?id=cspPnNScXa))
- **Upstream code:** `bundle-valued-statistics` (cloned locally, **not vendored here**)
- CPU-only. Published to the ICML-2026-agent-repro challenge (quality: medium).

The two central concentration inequalities, audited directly by Monte Carlo:

```
Theorem 1 (Hoeffding):  P(||Ȳ_n − m*|| ≥ ε) ≤ 2·exp(−n·ε² / (8·B²))
Theorem 2 (Bernstein):  P(||Ȳ_n − m*|| ≥ ε) ≤ 2·exp(−n·ε² / (2·(σ² + 2·B·ε/3)))
```

## Claims reproduced

**Claim 1 — dimension-free** (`claim1_dimension_free.py`). Both bounds depend
only on `B, σ², n, ε` — not on the ambient/fiber dimension `k`. The same
`(n, ε, B)` audit is repeated at `k = 2, 10, 100, 1000`; the bound values are
identical across `k` and the empirical tail probability never exceeds either
bound nor grows with `k`.

**Claim 2 — Hoeffding vs Bernstein** (`claim2_hoeffding_bernstein.py`). Both are
valid upper bounds; when `σ² ≪ B·ε`, Bernstein is materially tighter (the
paper's stated regime).

**Control — Assumption 1 is load-bearing.** Theorem 1 requires
`||X_i − μ|| ≤ B` *almost surely*. For a heavy-tailed source with no finite
a.s. bound, the running max `||X_i||` keeps climbing (never plateaus), and
plugging an empirically observed max in for `B` makes the Hoeffding formula fail
out-of-sample — the a.s.-bound assumption is not decorative.

`verify_bounds.py` runs Claims 1 + 2 + the control end-to-end.

## Files

- `claim1_dimension_free.py`, `claim2_hoeffding_bernstein.py`,
  `verify_bounds.py` — the audits (results printed to stdout / captured in the
  published logbook)
- `poster.html` — executive-summary poster source
- `GATE_REPORT.json` — poster style-gate report

## Setup

Pure `numpy`/`scipy` Monte Carlo; the `bundle-valued-statistics` upstream repo
is referenced but the audits here are self-contained.
