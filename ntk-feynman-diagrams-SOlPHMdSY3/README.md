# Finite-Width Neural Tangent Kernels from Feynman Diagrams — reproduction

- **Paper:** Finite-Width Neural Tangent Kernels from Feynman Diagrams (arXiv
  [`2508.11522v4`](https://arxiv.org/abs/2508.11522), OpenReview
  [`SOlPHMdSY3`](https://openreview.net/forum?id=SOlPHMdSY3))
- **Upstream code:** none released — everything here is from scratch
- **Verdict:** **3/5 TOY-VERIFIED, 2/5 INCONCLUSIVE** · CPU-only (~5 min compute)
- **Trackio Logbook (full write-up):** https://huggingface.co/spaces/nmaher/repro-finite-width-neural-tangent-kernels-from-feynman-diagrams

A toy-scale, CPU-only reproduction implementing finite-width MLPs (PyTorch autograd) and an
independently hand-derived analytic infinite-width NTK recursion, to test the paper's empirical
claims about finite-width corrections to the Neural Tangent Kernel.

**Note on the challenge's extracted claims**: `claims_anchored.json` cites Figure 2 for Claim 4
and Figure 1 for Claim 5 — these are mislabeled. The paper's actual figures for those experiments
are **Figure 3** and **Figure 2** respectively (verified directly against the source PDF; see
`PAPER_BRIEFING.md`).

## Claims reproduced

| # | Claim (paper section) | Verdict | Scale |
|---|---|---|---|
| 1 | Feynman-diagram rules simplify layer-wise recursions (Sec. 4) | INCONCLUSIVE | structural match confirmed; full numeric re-derivation blocked by OCR-corrupted operator notation |
| 2 | 5-diagram NTK-mean recursion, Eq. 15/78 (Sec. 5.1) | INCONCLUSIVE | same as above |
| 3 | No finite-width correction for scale-invariant activations, Theorem 5.2 (Sec. 5.3) | **TOY-VERIFIED** | widths 10-320, depth 4, 1500 inits/width |
| 4 | ReLU kernel-correction experiment, real Fig. 3 (Sec. 6.3) | **TOY-VERIFIED** | same run as Claim 3 |
| 5 | Gradient stability, linear depth-scaling at C_W=2, real Fig. 2 (Sec. 6.2) | **TOY-VERIFIED** | width 50, depth 15, 500 inits; linear fit R²=0.9986 |

**Claim 5** is the strongest result: at critical `C_W=2`, the NTK diagonal grows linearly with
depth (fitted slope 0.348, matching an independent analytic prediction of 0.345 to <1%, R²=0.9986
over 15 layers), while sub-/super-critical `C_W` values show near-perfect exponential decay/growth
(R²>0.99 both). **Claims 3/4** confirm the diagonal-vs-off-diagonal asymmetry Theorem 5.2 predicts
for ReLU/LeakyReLU: the diagonal NTK mean is flat across a 32× width range while the off-diagonal
visibly converges toward its infinite-width value as width grows.

**Claims 1/2** (the diagrammatic derivation itself) are marked INCONCLUSIVE rather than forced to
a verdict: one term of Eq. 78 depends on an operator (`ΔΩ_d`, tied to the paper's dNTK formalism)
whose extraction from the only available PDF source (arXiv — OpenReview was bot-walled) showed
clear font-substitution corruption. Implementing a possibly-misread physics-paper operator risked
a false verdict, so this was left open rather than guessed at — see `VERDICTS.md` for the full
reasoning and what would resolve it.

A bugfix caught during the mandatory smoketest-before-scale step is logged in `BUGFIX_LOG.md`: this
reproduction's own hand-derived analytic infinite-width NTK formula (used as ground truth for
Claims 3/4) was off by a factor of `C_W`, invisible at `C_W=1` and only surfacing at the paper's
own critical value `C_W=2` — fixed and reverified before any reported number was produced.

## Contents

- `paper-arxiv-2508.11522v4.pdf`, `paper_text.txt` — acquired paper (OpenReview bot-walled both
  attempts) and its extracted text.
- `PAPER_BRIEFING.md` — claims in scope, transcribed math/setup, reproduction plan and scope
  decisions (written before any code).
- `repro_toy.py` — self-contained PEP-723 script (`torch`, `numpy`, `scipy`): finite-width MLP
  construction, empirical NTK via autograd, and an independent analytic infinite-width ReLU NTK
  recursion, for both the width-sweep (Claims 3/4) and depth-sweep (Claim 5) experiments.
- `results/*.json` — raw run outputs for every experiment reported above.
- `BUGFIX_LOG.md` — the self-audit bugfix (analytic ground-truth normalization error) with
  before/after numbers.
- `VERDICTS.md` — full per-claim reasoning and pre-submission self-check.
- `build_poster.py`, `poster.html`, `poster.png`, `poster_embed.html` — executive-summary poster
  (Playwright-rendered HTML card, base64-embedded into the Trackio logbook).

## Rerunning

```bash
cd ntk-feynman-diagrams-SOlPHMdSY3
uv run repro_toy.py width-sweep --tag relu_toy --widths 10 20 40 80 160 320 --depth 4 \
  --n-inits 1500 --C-W 2.0 --activation relu
uv run repro_toy.py depth-sweep --tag toy --width 50 --max-depth 15 --n-inits 500 \
  --C-W-list 0.25 2.0 4.0
```

No GPU required — every run above completes in well under two minutes on an 8-core CPU.
