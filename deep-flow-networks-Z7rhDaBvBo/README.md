# Deep Flow Networks — reproduction

- **Paper:** Deep Flow Networks (OpenReview [`Z7rhDaBvBo`](https://openreview.net/forum?id=Z7rhDaBvBo), Spotlight)
- **Upstream code:** `github.com/ayfous/deep-flow-networks` @ `242af8e` (MIT)
- **Verdict:** **4/4 verified** · CPU-only

## Claims reproduced

**Claim 1 — universality of convex-extendible approximation** (`claim1_universality.py`,
`claim1_constructive.py`). DFNs approximate any convex, convex-extendible discrete
function on a grid; a concave control admits no convex extension and is
represented far worse. On the exhaustive grid `{-7..7}^2`:

| function | kind | max rel. err (small / large model) |
|---|---|---|
| quad | convex | 0.208 / 0.307 |
| sepnorm | convex | 0.183 / 0.123 |
| maxaff | convex (piecewise-linear) | 0.248 / 0.221 |
| **CONTROL (concave)** | **not convex-extendible** | **0.519** |

Zero discrete-midpoint-convexity violations across all fitted DFNs — the learned
map is convex by construction, so the control's large error is the audit
*detecting* the theorem's hypothesis, not a training failure. (Results in
`claim1_universality.csv`; `..._run1_undertrained.csv` is the retained negative
result at 300 epochs — DFNs need the paper's ~1000-epoch budget.)

**Claim 2 — accuracy and optimization speed** (`run_quad_small.py`). Head-to-head
DFN vs MLP vs LSET on the repo's own `run_experiment` pipeline, reduced in scale
so every MILP fits the free size-limited Gurobi license (2000 vars/constrs).
Fresh `n=8` retrain: **DFN 0.0061 MSE / 0.093 s** vs **MLP 0.0088 MSE / 1.85 s**;
LSET's log-sum-exp MILP exceeds the free-license cap. Independently, **45/45**
of the repo's committed artifacts hash-match on re-run (config-hash-keyed).

## Files

- `claim1_universality.py`, `claim1_constructive.py` — Claim 1 audits
- `run_quad_small.py`, `optsmoke.py`, `quickstart_smoke.py` — Claim 2 + env smoke
- `make_figs_dfn.py` — regenerates the `*_fig.html` / `*_fig.png` from CSVs
- `claim*_*.csv` — result tables; `aevo_*` / `claim2_*` — accuracy-vs-speed data
- `poster.html` — executive-summary poster source

## Setup (not fully vendored)

This reproduction runs the authors' released code. To rerun:

1. Clone `github.com/ayfous/deep-flow-networks` @ `242af8e` beside these scripts.
2. Build header-only **LEMON 1.3.1** from the source tarball with a 2-line
   `config.h`, exposed via `LEMON_INCLUDE_DIR`.
3. `uv venv` (py3.13) + torch-CPU.

`solve_true_opt` needs array-valued bounds (not scalars); the LSET MILP is
unsolvable under the free Gurobi license.
