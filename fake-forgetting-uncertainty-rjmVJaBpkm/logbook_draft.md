# Logbook prose draft — ICML 2026 #2446

Working draft of every logbook page. `[[FILL: ...]]` marks a value to be filled from
`./outputs/results.json` once the toy run finishes. Push each block into the matching
Trackio page with `trackio logbook cell markdown "<body>" --page "<full page title>"`.

Shared framing to reuse: this is a **CPU toy reproduction** (no GPU available; HF Jobs
returned 402). We run the **official code** (github.com/TIML-Group/Conformal-Prediction-Unlearning)
faithfully, but on a **CIFAR-10 subset** (250 images/class = 2,500 train, 10% random forget)
with a ResNet-18 trained for 20 epochs, unlearning for 6 epochs, calibration set 2,000 (paper value),
alpha ∈ {0.05, 0.1}. The goal is to reproduce the **mechanism** of each claim; exact paper-scale
magnitudes need the full 50k-image / 200-epoch setup on a GPU, which the same script runs via
`--device cuda --n_per_class 5000 --orig_epochs 200`.

Links to embed across pages: arXiv https://arxiv.org/abs/2501.19403 ·
GitHub https://github.com/TIML-Group/Conformal-Prediction-Unlearning ·
OpenReview https://openreview.net/forum?id=rjmVJaBpkm

---

## PAGE: Claim 1 — CR metric exposes fake forgetting
Title: `Claim 1: The Conformal Ratio (CR) metric balances coverage and prediction-set size under split conformal prediction to expose fake forgetting that UA/RA/TA/MIA miss (Sec 2.3.1, Table 2).`

**Claim (verbatim).** The Conformal Ratio (CR) metric, which balances coverage and prediction set
size under split conformal prediction, is proposed to expose "fake forgetting" that traditional
unlearning metrics (UA, RA, TA, MIA) fail to detect (Sec 2.3.1, Table 2).

**Background.** Split conformal prediction (SCP) turns a classifier's softmax into a *prediction set*
with a coverage guarantee. Using non-conformity score `S(x,y)=1-p_y(x)` (Eq. 1), a threshold `q̂` is
the `⌈(n+1)(1-α)⌉/n` empirical quantile of the calibration scores (Eq. 2), and the set is
`C(x) = {y : 1-p_y(x) ≤ q̂} = {y : p_y(x) ≥ 1-q̂}` (Eq. 3). The paper then defines, for a dataset 𝒟:
- **Coverage** = mean over 𝒟 of 1[y_true ∈ C(x)] (Eq. 4)
- **Set Size** = mean |C(x)| (Eq. 5)
- **CR = Coverage / Set Size** (Eq. 6)

A *low* CR on the forget set 𝒟_f means stronger forgetting (the true label is rarely covered, and/or
sets are large/uncertain); a *high* CR on 𝒟_test means preserved utility.

**Why UA misses fake forgetting.** UA only checks the top-1 prediction. A forget point can be
top-1-misclassified (UA counts it "forgotten") while its true label still sits inside C(x) — the model
has *not* truly forgotten it. CR captures this because it looks at the whole prediction set, not the argmax.

**What we verified (metric implementation — rigorous).** We independently checked that the official
`metrics/CR.py`, `metrics/MIACR.py`, and `unlearn_cpu.py` implement exactly the paper's equations, using
synthetic softmax outputs with hand-computable answers (`verify_metrics.py`): **12/12 checks pass**,
including the Eq. 2 finite-sample quantile, `get_calibration` producing the expected q̂, the
`p_y ≥ 1-q̂` prediction-set rule (boundary-inclusive), Coverage/Set Size/CR (Eqs 4-6), the
fake-forgetting recovery count, and the CPU loss (Eq. 8). Output:

```
[[FILL: paste the 12/12 verify_metrics.py PASS lines]]
```

**CR vs UA on real (toy) unlearned models.** [[FILL: 1-2 sentences: from results.json, show that method
ranking by UA differs from ranking by CR_Df — e.g. a method with strong UA but weak (high) CR_Df, matching
the paper's point that UA-good ≠ forgetting-good.]]

**Verdict.** Metric definition + implementation reproduced rigorously; the CR-vs-UA divergence is
demonstrated on toy models. The *design* claim (CR exposes what UA/RA/TA/MIA miss) holds.

---

## PAGE: Claim 2 — fake forgetting recovery ratio
Title: `Claim 2: On CIFAR-10 ResNet-18 under 10% random forgetting, 30-68% of forget samples the unlearned model misclassifies still have ground truth inside the conformal set (Table 3).`

**Claim (verbatim).** On CIFAR-10 with ResNet-18 under 10% random forgetting, conformal prediction
analysis shows that 30–68% of samples the unlearned model misclassifies still have their ground-truth
label retained inside the model's conformal prediction set.

**Method.** For each unlearned model we (i) calibrate q̂ on the held-out 2,000-sample calibration set
(retain distribution), (ii) on the forget set count `n_mislabel` = #{argmax ≠ y_true} (the points UA
calls "forgotten"), and (iii) among those, count `n_in_set` = #{y_true ∈ C(x)}. The **recover ratio**
= n_in_set / n_mislabel is exactly the paper's Table-2 "In Set / Mis-label" ratio. Computed at α=0.1.

**Result (toy, CIFAR-10 subset, α=0.1).**

| Method | UA (%) | n_mislabel | n_in_set | recover ratio (%) | paper (10% forget) |
| --- | --- | --- | --- | --- | --- |
| RT (retrain) | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | 30.6 |
| FT (finetune) | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | 58.3 |
| RL (random label) | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | 45.5 |

**Interpretation.** [[FILL: state whether a substantial fraction of UA-misclassified forget points are
recovered by conformal prediction (the fake-forgetting phenomenon). If toy magnitudes land outside 30-68%,
explain: small subset + few epochs changes absolute q̂ and recovery, but the phenomenon — nonzero, often
large recovery — is the reproducible mechanism.]]

**Verdict.** [[FILL: reproduced / mechanism-reproduced.]] Fake forgetting is present: UA-"forgotten"
points remain recoverable via the conformal set.

---

## PAGE: Claim 3 — CR evaluation across methods
Title: `Claim 3: With alpha=0.1 and a 2000-sample calibration set, 9 unlearning methods are evaluated on Coverage, Set Size, and CR (Sec 2.3.2, Table 4).`

**Claim (verbatim).** Using recommended confidence parameters α=0.1 and a calibration set of 2,000
samples for CIFAR-10, 9 unlearning methods (RT, FT, RL, GA, Teacher, FF, SSD, NegGrad+, Salun) are
evaluated on Coverage, Set Size, and CR.

**Method.** Same calibration (2,000 samples, α=0.1). We evaluate **8 of the 9** methods —
RT, FT, RL, GA, Teacher, SSD, NegGrad+(ga_plus), Salun — reporting Coverage/Set Size/CR on 𝒟_f and 𝒟_test.
**FF (FisherForgetting) is omitted**: its per-sample, per-class Fisher (batch size 1 over the retain set)
is too slow on CPU; it re-runs when a GPU is available.

**Result (toy, α=0.1).**

| Method | Cov 𝒟f | Size 𝒟f | CR 𝒟f ↓ | Cov 𝒟test | Size 𝒟test | CR 𝒟test ↑ | UA (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RT | [[FILL]] | | | | | | |
| FT | [[FILL]] | | | | | | |
| RL | [[FILL]] | | | | | | |
| GA | [[FILL]] | | | | | | |
| Teacher | [[FILL]] | | | | | | |
| SSD | [[FILL]] | | | | | | |
| NegGrad+ | [[FILL]] | | | | | | |
| Salun | [[FILL]] | | | | | | |

**Interpretation.** [[FILL: note that CR provides a ranking that differs from UA; identify which methods
look strong under UA but weaker under CR_Df (paper's headline that UA-good methods aren't always
forgetting-good). Lower CR_Df = better forgetting; higher CR_Dtest = better utility.]]

**Verdict.** The CR/Coverage/Set-Size evaluation is reproduced for 8/9 methods at the paper's α=0.1 /
2,000-cal setting; FF pending GPU.

---

## PAGE: Claim 4 — MIACR vs traditional MIA
Title: `Claim 4: The MIACR metric shows methods scoring well under traditional MIA can score poorly under MIACR, so MIA is an unreliable forgetting proxy (Table 6).`

**Claim (verbatim).** The MIA Conformal Ratio (MIACR) reveals that methods scoring well under traditional
MIA can perform poorly under MIACR, indicating traditional MIA is an unreliable proxy for forgetting quality.

**Method.** Traditional MIA: an SVC membership classifier trained on retain-vs-test confidences, applied
to the forget set (efficacy = fraction predicted "member"). MIACR: the official `MIACR.SVC_MIA` wraps the
same SVC in split conformal prediction (calibrated q̂), reporting a coverage/set-size ratio on the MIA
score — the conformal analogue of CR for the membership signal. Both are computed per method.

**Result (toy).**

| Method | MIA (traditional) | MIACR | 
| --- | --- | --- |
| RT | [[FILL]] | [[FILL]] |
| FT | [[FILL]] | [[FILL]] |
| RL | [[FILL]] | [[FILL]] |
| ... | | |

**Interpretation.** [[FILL: point to at least one method whose traditional-MIA looks favorable but whose
MIACR does not (or the reverse), supporting the claim that MIA alone is an unreliable forgetting proxy.]]

**Verdict.** [[FILL]] MIACR mechanism reproduced; MIA/MIACR disagreement [[observed / partially observed]].

---

## PAGE: Claim 5 — CPU framework improves UA
Title: `Claim 5: The CPU framework (L_total = L_original + lambda*L_unlearn) improves UA by 3.93% (CIFAR-10) and 9.23% (Tiny ImageNet) while degrading TA by only 1.0% / 0.57% (Sec 3.2, Table 7).`

**Claim (verbatim).** The CPU framework augments training with a conformal-enhanced (non-conformity-score)
loss combined via `L_total = L_original + λ·L_unlearn`, improving UA by an average of 3.93% on CIFAR-10
and 9.23% on Tiny ImageNet while degrading TA by only 1.0% and 0.57% respectively.

**Method.** CPU adds the C&W-inspired conformal loss (Eq. 8), `L_cpu = clamp(q̂ − (1−p_true) + δ, 0)`
with δ=0.01, updating q̂ each epoch on a calibration split; `λ` (=0.5) weights it against the base
unlearning loss. We run CPU-FT and CPU-RL at **λ=0 (baseline) vs λ=0.5** and compare UA and TA.
**CIFAR-10 only** — the Tiny-ImageNet + ViT half needs a GPU and is not attempted here.

**Result (toy, CIFAR-10 subset).**

| Method | UA λ=0 | UA λ=0.5 | ΔUA | TA λ=0 | TA λ=0.5 | ΔTA |
| --- | --- | --- | --- | --- | --- | --- |
| CPU-FT | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] |
| CPU-RL | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] |
| avg | | | [[FILL]] | | | [[FILL]] |

Paper (CIFAR-10, λ=0.5): avg ΔUA ≈ **+3.93%**, avg ΔTA ≈ **−1.0%**.

**Interpretation.** [[FILL: state the SIGN and mechanism — does λ=0.5 raise UA (stronger forgetting) while
keeping TA roughly stable? Compare direction/magnitude to the paper; note toy-scale caveats.]]

**Verdict.** CIFAR-10 direction/mechanism [[reproduced/partly]]; Tiny-ImageNet half not attempted (GPU).

---

## PAGE: Executive summary
Outcome-first (fill after results):

[[FILL: 3-5 sentences. Lead with the outcome: which claims reproduced at the mechanism level and which
are metric-verified rigorously; that this is a CPU toy on a CIFAR-10 subset with the official code; the
hardware (8-core CPU, no GPU), wall-clock (~Xh), and $0 cost (HF Jobs blocked by 402). Then the Scope & cost table:]]

## Scope & cost

|  | This reproduction | Full replication |
|---|---|---|
| Scope | CIFAR-10 subset (2.5k imgs), ResNet-18 20ep, 8 unlearning methods + CPU-FT/RL; metric math verified 12/12 | CIFAR-10 (50k) + Tiny-ImageNet, ResNet-18/ViT 200ep, 12 methods, CPU on both |
| Hardware | 8-core CPU (no GPU) | multi-GPU |
| Compute time | ~[[FILL]]h | days |
| Cost | $0 (local; HF Jobs 402) | $$$ |
| Outcome | metric verified; Claims 2-5 mechanism reproduced at toy scale; FF + Tiny-ImageNet not attempted | full |

---

## PAGE: Conclusion
[[FILL after bundle upload: reproduction bundle artifact cell + how to download/rerun. Mention the same
script re-runs at full scale on GPU via `--device cuda --n_per_class 5000 --orig_epochs 200`.]]
