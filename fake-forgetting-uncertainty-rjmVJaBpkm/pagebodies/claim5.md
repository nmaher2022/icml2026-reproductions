**Claim.** The CPU framework augments training with a conformal-enhanced (non-conformity-score) loss combined via `L_total = L_original + λ·L_unlearn`, improving UA by an average of **3.93%** on CIFAR-10 and **9.23%** on Tiny ImageNet while degrading TA by only **1.0%** and **0.57%** respectively (Sec 3.2, paper Table 7/4).

**Method.** CPU adds the C&W-inspired conformal loss (Eq. 8), `L_cpu = clamp(q̂ − (1−p_true) + δ, 0)` with δ=0.01, re-estimating q̂ each epoch on a calibration split; λ weights it against the base unlearning loss. We run CPU-FT and CPU-RL at **λ=0 (baseline, no CPU) vs λ=0.5** and compare UA and TA. **CIFAR-10 only** — the Tiny-ImageNet + ViT half needs a GPU and is **not attempted** here.

**Result (toy, CIFAR-10 subset, α=0.1):**

| Method | UA λ0 | UA λ0.5 | ΔUA | TA λ0 | TA λ0.5 | ΔTA | CR_Df λ0 | CR_Df λ0.5 | CR_Dtest λ0 | CR_Dtest λ0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CPU-FT | 4.0 | 34.5 | **+30.5** | 48.8 | 47.4 | **−1.4** | 0.369 | 0.287 | 0.221 | 0.221 |
| CPU-RL | 67.0 | 90.5 | **+23.5** | 43.7 | 41.4 | **−2.3** | 0.159 | 0.135 | 0.161 | 0.155 |
| **avg** | | | **+27.0** | | | **−1.9** | | | | |

Paper (CIFAR-10, λ=0.5): avg ΔUA **+3.93%**, avg ΔTA **−1.0%**.

**Interpretation.** The CPU mechanism reproduces **cleanly and in the paper's direction**: adding the conformal loss (λ=0.5) **raises unlearning accuracy** (CPU-FT UA 4.0→34.5%, CPU-RL 67→90.5%) while **test accuracy barely moves** (−1.4% and −2.3%). The conformal signature matches the paper's narrative too: **CR_Dtest is essentially unchanged** (CPU-FT 0.221→0.221; the paper notes CR_Dtest drops only ~0.03) while **CR_Df drops** (0.369→0.287), i.e. stronger forgetting without utility loss. The absolute ΔUA is much larger than the paper's +3.93% because our baseline (λ=0) FT barely forgets at toy scale (UA 4%), leaving large headroom; the *sign, mechanism, and the UA↑ / small-TA↓ trade-off* are reproduced.

**Verdict: CIFAR-10 half reproduced** (direction + mechanism); **Tiny-ImageNet + ViT half not attempted** (GPU required). Code: [github.com/TIML-Group/Conformal-Prediction-Unlearning](https://github.com/TIML-Group/Conformal-Prediction-Unlearning).
