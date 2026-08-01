# ICML 2026 paper reproductions

Independent reproductions of ten ICML 2026 papers, produced for the
**[ICML-2026-agent-repro](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge)**
challenge (Hugging Face × alphaXiv). Each reproduction re-derives the paper's
central claims from scratch or from the authors' released code, on **CPU only**,
and reports the achieved numbers against the paper's.

Every folder here is self-contained: audit scripts (PEP-723, run with
[`uv`](https://docs.astral.sh/uv/)), the result CSVs they produce, and the
figures built from those CSVs. The full narrative write-ups live as published
trackio logbooks (linked per paper).

## Index

| Paper | OpenReview | Claims reproduced | Verdict | Folder | Trackio Logbook |
|---|---|---|---|---|---|
| Deep Flow Networks | [`Z7rhDaBvBo`](https://openreview.net/forum?id=Z7rhDaBvBo) (Spotlight) | 2/2 (4 sub-claims) | **4/4 verified** | [`deep-flow-networks-Z7rhDaBvBo/`](deep-flow-networks-Z7rhDaBvBo/) | [HF Space](https://huggingface.co/spaces/nmaher/repro-deep-flow-networks) |
| Asymptotic optimality of the high-dimensional Gaussian mechanism | [`82Wosp2Iu1`](https://openreview.net/forum?id=82Wosp2Iu1) (Spotlight) | 6 | **11/12** | [`gaussian-mechanism-82Wosp2Iu1/`](gaussian-mechanism-82Wosp2Iu1/) | [HF Space](https://huggingface.co/spaces/nmaher/repro-asymptotic-optimality-of-the-high-dimensional-gaussian-mechanism-and-improved-low-dimensio) |
| A studentized spherical-harmonics nonparametric two-sample test | [`QsxpsAu7l1`](https://openreview.net/forum?id=QsxpsAu7l1) | 3 | published | [`spherical-harmonics-two-sample-test-QsxpsAu7l1/`](spherical-harmonics-two-sample-test-QsxpsAu7l1/) | [HF Space](https://huggingface.co/spaces/nmaher/repro-a-studentized-spherical-harmonics-based-nonparametric-two-sample-test-for-compositional-an) |
| Sharp concentration bounds for bundle-valued statistics on manifolds | [`cspPnNScXa`](https://openreview.net/forum?id=cspPnNScXa) | 2 + control | published (medium) | [`concentration-bounds-bundle-valued-statistics-cspPnNScXa/`](concentration-bounds-bundle-valued-statistics-cspPnNScXa/) | [HF Space](https://huggingface.co/spaces/nmaher/repro-sharp-concentration-bounds-for-bundle-valued-statistics-on-manifolds) |
| Tackling fake forgetting through uncertainty quantification | [`rjmVJaBpkm`](https://openreview.net/forum?id=rjmVJaBpkm) | 5 (CPU toy) | published (medium) | [`fake-forgetting-uncertainty-rjmVJaBpkm/`](fake-forgetting-uncertainty-rjmVJaBpkm/) | [HF Space](https://huggingface.co/spaces/nmaher/repro-tackling-fake-forgetting-through-uncertainty-quantification) |
| Improved dynamic algorithm for non-monotone submodular maximization | [`tBS3uBG6Pv`](https://openreview.net/forum?id=tBS3uBG6Pv) | 3 | 3/6 (medium) | [`submodular-dynamic-non-monotone-tBS3uBG6Pv/`](submodular-dynamic-non-monotone-tBS3uBG6Pv/) | [HF Space](https://huggingface.co/spaces/nmaher/repro-improved-dynamic-algorithm-for-non-monotone-submodular-maximization-under-cardinality-cons) |
| Divide and Learn: Multi-Objective Combinatorial Optimization at Scale | [`TK82ECnJzD`](https://openreview.net/forum?id=TK82ECnJzD) | 6 | 2 TOY-VERIFIED / 4 REFUTED (pending judge score) | [`divide-and-learn-TK82ECnJzD/`](divide-and-learn-TK82ECnJzD/) | [HF Space](https://huggingface.co/spaces/nmaher/repro-divide-and-learn-multi-objective-combinatorial-optimization-at-scale) |
| CausalProfiler: Generating Synthetic Benchmarks for Rigorous and Transparent Evaluation of Causal ML | [`0wCl7EifsY`](https://openreview.net/forum?id=0wCl7EifsY) | 5 | mostly confirmed (1/2/5 confirmed; 3/4 partially confirmed) | [`causalprofiler-0wCl7EifsY/`](causalprofiler-0wCl7EifsY/) | [HF Space](https://huggingface.co/spaces/nmaher/repro-causalprofiler-generating-synthetic-benchmarks-for-rigorous-and-transparent-evaluation-of) |
| From Muon to Gluon: Bridging Theory and Practice of LMO-based Optimizers for LLMs | [`IelAHU5MVz`](https://openreview.net/forum?id=IelAHU5MVz) | 6 | mixed (1 verified; 2/3/4 toy-verified; 5 refuted at toy scale; 6 refuted — claim-extraction error) | [`gluon-lmo-optimizers-IelAHU5MVz/`](gluon-lmo-optimizers-IelAHU5MVz/) | [HF Space](https://huggingface.co/spaces/nmaher/repro-from-muon-to-gluon-bridging-theory-and-practice-of-lmo-based-optimizers-for-llms) |
| Active Continual Learning with Metaplastic Binary Bayesian Neural Networks | [`SPZd0HVyiS`](https://openreview.net/forum?id=SPZd0HVyiS) | 5 (2 toy-verified/mixed; 3 blocked by data access) | mixed (2/3 toy-scale mixed; 1/4/5 blocked — dataset access) | [`active-continual-learning-bimu-SPZd0HVyiS/`](active-continual-learning-bimu-SPZd0HVyiS/) | [HF Space](https://huggingface.co/spaces/nmaher/repro-active-continual-learning-with-metaplastic-binary-bayesian-neural-networks) |
| Toward Scalable and Valid Conditional Independence Testing with Spectral Representations | [`nPzckCXmHE`](https://openreview.net/forum?id=nPzckCXmHE) | 5 | 4 TOY-VERIFIED (1 partial), 1 INCONCLUSIVE | [`spectral-cit-nPzckCXmHE/`](spectral-cit-nPzckCXmHE/) | [HF Space](https://huggingface.co/spaces/nmaher/repro-toward-scalable-and-valid-conditional-independence-testing-with-spectral-representations) |
| Differentially Private Synthetic Data via APIs 4: Tabular Data | [`WB0hLRRlcj`](https://openreview.net/forum?id=WB0hLRRlcj) | 5 | **3/5 verified** (1 blocked @ 4-5 features, 1 split CPU-only/multiplier) | [`tab-pe-WB0hLRRlcj/`](tab-pe-WB0hLRRlcj/) | [HF Space](https://huggingface.co/spaces/nmaher/repro-differentially-private-synthetic-data-via-apis-4-tabular-data) |
| WorldComp2D: Spatio-semantic Representations of Object Identity and Location from Local Views | [`WQIyx69dFg`](https://openreview.net/forum?id=WQIyx69dFg) | 5 | 4/5 verified (COFW); 300W/AFLW dataset access blocked | [`worldcomp2d-WQIyx69dFg/`](worldcomp2d-WQIyx69dFg/) | [HF Space](https://huggingface.co/spaces/nmaher/repro-worldcomp2d-spatio-semantic-representations-of-object-identity-and-location-from-local-vie) |

*Verdicts are the challenge judge's scores (verified/falsified = 2 pt, toy =
1 pt, inconclusive = 0), against the published logbook, except where noted
as pending.*

## Harness

The workflow behind these reproductions (paper acquisition, briefing, smoketest-before-scale,
self-audit, honest verdicts, logbook/poster, GitHub mirror) is packaged as a reusable skill at
[`.agents/skills/repro-harness/`](.agents/skills/repro-harness/). Its design rationale, build
status, and audit process live in [`harness-testing/`](harness-testing/) — that folder tracks the
pipeline itself, not any one paper's claims.

## How to run

The audit scripts carry their dependencies inline (PEP-723 headers), so no
manual environment setup is needed:

```bash
uv run gaussian-mechanism-82Wosp2Iu1/claim15_optimality.py
```

The Deep Flow Networks reproduction additionally depends on the authors'
released code (`github.com/ayfous/deep-flow-networks`) and the header-only
LEMON graph library; those are **not** vendored here — see that folder's README
for setup. Everything else is pure `numpy`/`scipy`/`mpmath`/`torch`-CPU.

## Reproducibility notes

- **CPU-only.** None of these need a GPU; wall times are minutes.
- Figures (`*.png`, `*.html`) are regenerated from the committed CSVs by each
  folder's `make_figs*.py`.
- Poster HTML (`poster.html`) is the source of each logbook's executive-summary
  poster; the large rendered/base64-embedded variants are omitted to keep the
  repo lean.

## License

MIT (see [`LICENSE`](LICENSE)) — covers the reproduction/audit code in this
repository. Third-party code (e.g. the Deep Flow Networks repo, LEMON) retains
its own upstream license and is not redistributed here.
