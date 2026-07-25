# ICML 2026 paper reproductions

Independent reproductions of three ICML 2026 papers, produced for the
**[ICML-2026-agent-repro](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge)**
challenge (Hugging Face × alphaXiv). Each reproduction re-derives the paper's
central claims from scratch or from the authors' released code, on **CPU only**,
and reports the achieved numbers against the paper's.

Every folder here is self-contained: audit scripts (PEP-723, run with
[`uv`](https://docs.astral.sh/uv/)), the result CSVs they produce, and the
figures built from those CSVs. The full narrative write-ups live as published
trackio logbooks (linked per paper).

## Index

| Paper | OpenReview | Claims reproduced | Verdict | Folder |
|---|---|---|---|---|
| Deep Flow Networks | [`Z7rhDaBvBo`](https://openreview.net/forum?id=Z7rhDaBvBo) (Spotlight) | 2/2 (4 sub-claims) | **4/4 verified** | [`deep-flow-networks-Z7rhDaBvBo/`](deep-flow-networks-Z7rhDaBvBo/) |
| Asymptotic optimality of the high-dimensional Gaussian mechanism | [`82Wosp2Iu1`](https://openreview.net/forum?id=82Wosp2Iu1) (Spotlight) | 6 | **11/12** | [`gaussian-mechanism-82Wosp2Iu1/`](gaussian-mechanism-82Wosp2Iu1/) |
| Improved dynamic algorithm for non-monotone submodular maximization | [`tBS3uBG6Pv`](https://openreview.net/forum?id=tBS3uBG6Pv) | 3 | see folder | [`submodular-dynamic-non-monotone-tBS3uBG6Pv/`](submodular-dynamic-non-monotone-tBS3uBG6Pv/) |

*Verdicts are the challenge judge's scores (verified/falsified = 2 pt, toy =
1 pt, inconclusive = 0), against the published logbook.*

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
