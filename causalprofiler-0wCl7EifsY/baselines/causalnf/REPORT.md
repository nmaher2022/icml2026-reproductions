# CausalNF baseline plug-in for CausalProfiler — toy-scale, CPU-only

This covers the CausalNF half of Claim 4 (CausalNF has a much higher failure
rate than DCM on Regional Discrete SCMs). DCM is handled by a parallel effort;
this report only speaks to CausalNF's own behavior.

## 1. What CausalNF needed, API-wise

- Repo: [`adrianjav/causal-flows`](https://github.com/adrianjav/causal-flows)
  (pip package name `causalflows`), cloned into
  `baselines/causalnf/causal-flows`. Requires Python >= 3.11 (the system
  `python3.10` had no `_ssl` module, so I built venvs from
  `anaconda3/envs/ai-safety`'s Python 3.11 instead — CPU-only `torch==2.13+cpu`
  throughout, no GPU touched anywhere).
- It's a small wrapper around [Zuko](https://github.com/probabilists/zuko)
  normalizing flows. The relevant class is `causalflows.flows.CausalNSF`
  (Causal Neural Spline Flow): a continuous, real-valued, invertible
  density model. Construction: `CausalNSF(features, context_features,
  order=<topological order tuple>, hidden_features=[...])`. Training is
  plain maximum likelihood: `loss = -flow(context).log_prob(X).mean()`.
  Interventions are done via a context manager:
  `with flow(context).intervene(index, value) as int_flow: int_flow.sample((n,))`,
  which can be nested for multiple simultaneous interventions.
- It has **no discrete or mixed-data mode**. The library, README, and tests
  are 100% continuous-density-oriented (rational-quadratic splines over
  R). This is a load-bearing fact for the rest of this report.
- Repo's own smoke test: `pytest tests/` -> 6/7 passed. The one failure
  (`test_interventions`) is a strict `torch.allclose` check tripped by a
  float32 rounding artifact (`-1.49e-08` vs `0.0`, i.e. numerically zero) —
  a pre-existing test-tolerance issue in the library, unrelated to my setup
  and not a functional problem. I additionally hand-wrote a
  README-style example (3-node continuous chain SCM, `CausalNSF(3, 0,
  order=(0,1,2))`, 200 training steps) and recovered an ATE estimate of
  **-1.62** against a ground truth of **-1.6** — confirms the library
  trains and answers interventional queries correctly on CPU.

## 2. Adapter design (`causal_nf_method.py`)

`CausalNFMethod` implements `estimate(self, query, data, graph,
index_to_variable) -> float`.

**Training/caching strategy.** The harness instantiates the method once per
`SpaceOfInterest`, but calls `generate_samples_and_queries()` fresh every
run, then calls `estimate()` once per query x `num_tries`, always against the
same `data` for that run. I cache a trained flow keyed on a **content hash**
of `data` (MD5 over sorted `(name, shape, dtype, bytes)`), not `id(data)` —
Python can and does recycle object ids once the previous run's `data` dict is
garbage collected, which would silently reuse a stale model for unrelated
data. The flow is therefore trained exactly once per run (on the first
`estimate()` call for that run) and reused for the remaining queries/tries;
try-to-try variance comes from the flow's own sampling stochasticity, not
retraining — this matches what `num_tries` seems designed to measure.

**Graph handling.** I verified empirically (not assumed) that
`index_to_variable` is already the topological order of the visible
variables, and that `graph`'s indices respect parent < child. This is used
directly as the `order=` argument to `CausalNSF` — no extra topological
sort needed. Only `variable_dimensionality == 1` per node is supported; if
any variable has dimension > 1, `_fit` returns `None` (all queries for that
run -> NaN) rather than guessing a multi-dimensional autoregressive order.
This never triggered in the toy configs used here (all use
`variable_dimensionality=(1,1)`), but is an explicit, documented scope
limit rather than a silent bug.

**Discrete data: no relaxation trick, on purpose.** For Regional-Discrete
data (integer category codes), the adapter feeds the raw integer-coded
values to the flow after simple standardization (zero mean / unit std),
exactly as a real CausalNF user would do if they applied a continuous
method to discrete data without modification. No dequantization, no
one-hot, no rounding-aware objective — the point is to observe CausalNF's
*actual* behavior under a genuine train/test mismatch, not to paper over it.

**Failure detection (graceful NaN, not a crash).** During training, after
every gradient step the adapter checks `torch.isfinite(loss)` and that all
parameter gradients are finite. On the first non-finite loss/gradient,
training aborts and the run's model is cached as `None`; every subsequent
`estimate()` call for that run then returns `NaN` immediately. All of
`_fit` and `_answer_query` are additionally wrapped in broad
`try/except Exception` so any other library-level numerical error (e.g. a
spline domain issue) also degrades to `NaN` rather than crashing the
harness loop. `NaN` here is the intended signal for "CausalNF could not
handle this," not a bug to work around.

**Query types answered.** `ATE`, `CATE`, `CONDITIONAL` — chosen because
their ground-truth semantics (read from `causal_profiler/query_estimator.py`)
are well defined for both discrete and continuous Y, and they're the
central L1/L2 query types:
- `ATE = mean(Y | do(T=t1)) - mean(Y | do(T=t0))`, using each arm's
  interventional samples from the trained flow (raw value mean, matches
  the harness's own ground-truth definition even for discrete Y — it's
  not treated as a probability).
- `CATE` = same, but each arm's samples are additionally weighted by a
  conditioning filter on X: exact match for DISCRETE X, Gaussian-kernel
  soft weighting for CONTINUOUS X (mirroring
  `query_estimator.filter_data`'s own logic).
- `CONDITIONAL` = observational samples from the flow, same X-weighting;
  for DISCRETE Y, weighted proportion of exact matches (a probability); for
  CONTINUOUS Y, weighted mean.
- All other query types (`ITE`, `DTE`, `CDTE`, `OIP`, `Ctf-DE/IE/TE`)
  explicitly return NaN — out of scope for this toy adapter, documented as
  such rather than silently mishandled.

## 3. Smoke test

`smoke_test.py`: 3-5 node graphs, 1 seed, `n_samples=500`, 2 tries, run
directly through the real harness (`CausalProfiler` +
`generate_samples_and_queries` + `evaluate_error`), for both a small
continuous-linear-ATE space and a small tabular-discrete-ATE space.

**Result: PASSED.** No crash, no hang. Both settings completed in a couple
of seconds each and produced finite errors with 0/3 queries failed (a
"high failure rate" outcome would also have counted as a pass per the
brief — a crash or hang is the only real failure mode for a smoke test —
but at this very small scale nothing diverged).

## 4. Toy-scale evaluation

`run_toy_eval.py`. All CPU. Config used and rationale:

- **Graphs**: 4-6 nodes, `variable_dimensionality=(1,1)`, `expected_edges="1.5*N"`.
  Small enough to train a flow in ~0.2-5s per fit on CPU.
- **Setting A — `continuous_linear_ate`** (sanity baseline; CausalNF should
  do reasonably here since this is exactly its intended domain):
  `mechanism_family=LINEAR`, `variable_type=CONTINUOUS`,
  `noise_mode=ADDITIVE`, `noise_distribution=GAUSSIAN`, `noise_args=[0,0.5]`,
  `query_type=ATE`.
- **Setting B — `regional_discrete_tabular_ate` / `_conditional`**
  (Regional-Discrete-SCM-like; there is no dedicated "Regional Discrete SCM"
  flag in `SpaceOfInterest` — I confirmed by reading `space_of_interest.py`,
  `sampler.py`, and `mechanism.py`, and cross-checking against the paper's
  Definition E.2, that this class is exactly `mechanism_family=TABULAR` +
  `variable_type=DISCRETE` with `number_of_noise_regions` controlling the
  region-dependent discrete support — the code literally partitions each
  noise variable's support into regions via `np.digitize` against sampled
  thresholds, matching the paper's definition of a "regional discrete
  mechanism"). `number_of_categories=(2,3)`, `number_of_noise_regions="V"`
  (the library default expression), two sub-settings with `query_type=ATE`
  and `query_type=CONDITIONAL`.
- **Data/flow scale**: `number_of_data_points=800` (training data per run),
  `n_samples=2000` for the harness's own internal ground-truth Monte Carlo
  estimation (overridden down from the 10000 default), flow
  `hidden_features=(64,64)`, `epochs=150`, `n_samples_effect=1000` samples
  drawn from the trained flow when answering each query.
- **Repetition**: `seed_list=[42,43,44]` x `num_runs=3` x `num_tries=3` = 27
  (query-set, try) combinations per space, 5 queries per run.

### Results (mean over 9 runs per space, each run's error/failure averaged
over 3 tries first)

| Setting | mean L2 error | mean failure (NaN) rate | fits diverged |
|---|---|---|---|
| `continuous_linear_ate` (baseline) | **0.084** (+/- 0.075) | **0.0%** | 0/9 |
| `regional_discrete_tabular_ate` | **0.199** (+/- 0.116) | **0.0%** | 0/9 |
| `regional_discrete_tabular_conditional` | **0.219** (+/- 0.144) | **0.7%** | 0/9 |

Raw per-run JSON: `results/continuous_linear_ate_results.json`,
`results/regional_discrete_tabular_ate_results.json`,
`results/regional_discrete_tabular_conditional_results.json`. Aggregate:
`results/summary_latest.json` (also timestamped copy in the same folder).

### Supplementary stress probe (not part of the primary comparison)

Since the primary discrete setting showed almost no literal NaN failures, I
ran one additional, smaller probe (`run_stress_discrete.py`, 3 seeds x 2
runs x 2 tries, `results/regional_discrete_stress_results.json`) pushing
discreteness further — `number_of_categories=(4,6)`, `number_of_noise_regions=6`
(fixed int; the `"V_to_PA"` expression from the paper's own definition
blew up combinatorially into a >2 PiB array allocation request for a 4-6
node graph with several categorical parents per node — an artifact of the
`causal_profiler` library's combinatorial region-count formula on toy-sized
but densely-connected graphs, not something in scope to fix here since
`causal_profiler` is read-only; I substituted a fixed integer region count
instead) — and 400 training epochs instead of 150:

| Setting | mean L2 error | mean failure (NaN) rate | fits diverged |
|---|---|---|---|
| stress: 4-6 categories, 6 noise regions | **0.477** | **0.0%** | 0/6 |

## 5. Does the qualitative pattern hold?

**Partially, and honestly it's more nuanced than a clean yes/no.**

- **Error magnitude**: yes, clearly, and it scales monotonically with
  discreteness/stochasticity: continuous baseline 0.084 -> regional-discrete
  (2-3 categories) 0.199-0.219 (~2.4-2.6x) -> stress regional-discrete (4-6
  categories, more noise regions) 0.477 (~5.7x the continuous baseline).
  This is consistent with the paper's underlying claim that CausalNF is
  substantially worse on Regional Discrete SCMs — a continuous normalizing
  flow has no principled way to represent a genuinely discrete, region-
  dependent distribution, and the estimation error reflects that
  increasingly as the discrete structure gets richer.
- **Literal NaN failure rate**: no, not at this toy scale. My adapter's
  divergence detector (abort training on non-finite loss/gradient) almost
  never triggered — failure rate stayed at 0.0-0.7% in every setting,
  including the continuous baseline. With only 2-6 categories, ~800 training
  points, and 150-400 epochs, a spline flow can still fit a "blurred"
  continuous approximation to a small discrete distribution without its
  training loss actually diverging to -infinity/NaN within that short a
  training budget — the theoretical failure mode (density trying to
  concentrate infinite mass on point atoms) is more of an asymptotic
  phenomenon than something toy-scale training reliably hits. It's plausible
  the paper's larger-scale runs (more categories, more training, more nodes,
  possibly stricter numerical guards or a different failure criterion than
  "did the loss go non-finite") surface the failure-rate gap more directly
  than my finite-loss/grad check does at this scale.
- **Bottom line**: the *direction* of Claim 4 (CausalNF struggles much more
  on Regional Discrete SCMs than on standard continuous SCMs) is visible and
  strengthens with increasing discreteness in this toy reproduction, but
  primarily through **estimation error**, not through the **literal NaN/
  crash failure-rate metric** my adapter tracks. I want to be explicit that
  this is a real limitation of the toy scale (and possibly of my
  particular divergence-detection heuristic) rather than claiming a clean
  confirmation of the paper's specific failure-rate numbers — those would
  need a larger-scale run (more categories/regions, larger graphs, longer
  training, possibly thousands of samples) to have a chance of actually
  reproducing the same failure-rate magnitude the paper reports.

## Files

- Adapter: `/home/rec1/repro-causalprofiler/baselines/causalnf/causal_nf_method.py`
- Smoke test: `/home/rec1/repro-causalprofiler/baselines/causalnf/smoke_test.py`
- Toy eval driver: `/home/rec1/repro-causalprofiler/baselines/causalnf/run_toy_eval.py`
- Stress probe: `/home/rec1/repro-causalprofiler/baselines/causalnf/run_stress_discrete.py`
- Results: `/home/rec1/repro-causalprofiler/baselines/causalnf/results/`
- CausalNF clone: `/home/rec1/repro-causalprofiler/baselines/causalnf/causal-flows`
- Venvs (CPU-only, not committed): `venv_harness` (causal_profiler +
  causalflows together, used to run everything above),
  `venv_causalnf` (causal-flows in isolation, used only for its own
  pytest smoke test / README example)
