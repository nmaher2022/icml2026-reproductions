# Divide and Learn: Multi-Objective Combinatorial Optimization at Scale

Independent, from-scratch reproduction of [`TK82ECnJzD`](https://openreview.net/forum?id=TK82ECnJzD)
(arXiv:[2602.11346](https://arxiv.org/abs/2602.11346)), for the
**[ICML-2026-agent-repro](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge)**
challenge. Logbook Space: `nmaher/repro-divide-and-learn-multi-objective-combinatorial-optimization-at-scale`.

No official code was released at reproduction time (arXiv comment: "Code URL
coming soon", checked 2026-07-28), so every result here comes from an
independent implementation of Algorithms 1-5 and the stated
theorems/assumptions in Sections 4-5, built by literal reading of the paper's
pseudocode and prose — **not** the authors' code.

## Verdict summary (6 claims)

| Claim | Topic | Verdict |
|---|---|---|
| 1 | Regret O(d√(T log T)), depends on subproblem dim d not full n | Qualitatively supported |
| 2 | Coupling error O(K²QR_max√T) via Lagrangian coordination | Qualitatively supported |
| 3 | Multi-expert (UCB/EXP3/FTRL) mixture + coordination beats any single expert | **Falsified** |
| 4 | 80-98% specialized HV at 90-99% less compute, 10-30x faster than BO | **Falsified overall** (D&L beats our BO baseline on Bi-Knapsack specifically, not Bi-TSP; specialized-solver and compute-efficiency comparisons falsified everywhere) |
| 5 | Bi-TSP-100: D&L HV 0.69 vs specialized 0.67 vs PMOCO 0.63 | **Falsified** as stated (misattributes the paper's own Table 1 numbers) |
| 6 | D&L-TS HV 0.372±0.006 vs 0.34±0.012 qParEGO, 150-eval budget | Paper's own numbers confirmed accurate; **does not replicate** on our synthetic proxy (D&L-TS worst of 3) |

Full write-up, six real bugs found and fixed across three rounds, an
intermediate-n diagnostic sweep, and two paper-verification corrections: see
the [published logbook](https://huggingface.co/spaces/nmaher/repro-divide-and-learn-multi-objective-combinatorial-optimization-at-scale)
and `BUGFIX_LOG.md` in this folder.

## Contents

- `dl_core.py`, `dl_synthetic.py`, `dl_hwsw_proxy.py` — the D&L implementation
  (Algorithm 1: position-wise multi-expert bandits + Lagrangian dual
  coordination + zeroth-order local refinement) and its synthetic
  decomposable-with-bounded-coupling test environment.
- `moco_domains.py`, `moco_baselines.py`, `moco_metrics.py` — Bi-Knapsack/
  Bi-TSP instance generators, the WS-heuristic/NSGA-II/BO baselines, and the
  paper's normalized-hypervolume metric.
- `claim1_2_regret_coupling.py`, `claim1_tradeoff.py` — Claims 1-2 (regret
  scaling, coupling error, the K-tradeoff U-shape).
- `claim3_ablation.py` — Claim 3 (multi-expert + coordination ablation).
- `run_moco.py` — Claims 4-5 (Bi-Knapsack/Bi-TSP vs. WS-heuristic/NSGA-II/BO).
- `claim6_hwsw_proxy.py` — Claim 6 (4-objective HW-SW co-design synthetic
  proxy vs. a qParEGO-style BO analogue).
- `claim_midscale_sweep.py` — diagnostic-only intermediate-n (16/32) sweep,
  not one of the paper's claims, run to check whether the Bi-Knapsack/Bi-TSP
  split found in Claim 4 is a smooth n-driven trend or a hidden bug.
- `make_figures.py` — regenerates all `*.png` figures from the `*.csv` result
  files in this folder.
- `*.csv` — raw result tables behind every claim/figure.
- `*.png` — figures embedded in the logbook and poster.
- `poster.html` — the reproduction poster source (Chenruishuo/posterly
  format); the rendered/base64-embedded variant is omitted to keep this repo
  lean.
- `BUGFIX_LOG.md` — full bug-fix history (6 bugs, 3 rounds) and the
  primary-source verification pass against the paper's own tables.

## How to run

Every script is self-contained (PEP-723 header, run with
[`uv`](https://docs.astral.sh/uv/)), no manual environment setup needed:

```bash
uv run claim1_2_regret_coupling.py
uv run claim1_tradeoff.py
uv run claim3_ablation.py
uv run run_moco.py            # Claims 4-5, full run (~35 min on 1 CPU core)
uv run claim6_hwsw_proxy.py
uv run claim_midscale_sweep.py
uv run make_figures.py
```

All CPU-only; no GPU-dependent baselines (botorch qNEHVI/qParEGO, PMOCO's
RL-trained hypernetwork) were attempted — smaller from-scratch analogues are
substituted throughout and labeled as such in every claim's write-up.

## Reproducibility notes

- MOCO/HW-SW benchmarks use synthetic instances/landscapes, not the paper's
  exact (unreleased) instances, seeds, or — for HW-SW — its proprietary
  accelerator simulator.
- Total compute: ~2 hours across three fix-and-rerun rounds plus the
  diagnostic sweep, single CPU core, no paid infrastructure.

## License

MIT (see the monorepo's top-level `LICENSE`) — covers this reproduction/audit
code. No third-party code is vendored here.
