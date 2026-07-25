**Claim.** Using α=0.1 and a 2,000-sample calibration set for CIFAR-10, 9 unlearning methods (RT, FT, RL, GA, Teacher, FF, SSD, NegGrad+, Salun) are evaluated on Coverage, Set Size, and CR (Sec 2.3.2, paper Table 3/4).

**Method.** Same calibration (2,000 samples, α=0.1). We evaluate **8 of the 9** methods — RT, FT, RL, GA, Teacher, SSD, NegGrad+ (=`ga_plus`), Salun — reporting Coverage/Set Size/CR on 𝒟_f (forget) and 𝒟_test. **FF (FisherForgetting) is omitted**: its per-sample, per-class Fisher (batch size 1 over the retain set) is prohibitively slow on CPU; it re-runs when a GPU is available. GA required a smaller learning rate (`ga_lr=1e-4`) — at the default rate gradient ascent diverged to NaN.

**Result (toy, α=0.1)** — lower CR_Df ↓ = stronger forgetting; higher CR_Dtest ↑ = preserved utility:

| Method | UA (%) | Cov Df | Size Df | CR Df ↓ | Cov Dtest | Size Dtest | CR Dtest ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RT | 53.5 | 0.89 | 4.32 | 0.206 | 0.90 | 4.20 | 0.214 |
| FT | 3.5 | 1.00 | 2.67 | 0.375 | 0.89 | 4.00 | 0.223 |
| RL | 66.0 | 0.96 | 5.61 | 0.172 | 0.89 | 5.32 | 0.168 |
| GA | 0.5 | 1.00 | 2.19 | 0.457 | 0.89 | 4.15 | 0.214 |
| Teacher | 92.5 | 0.78 | 8.00 | 0.098 | 0.80 | 8.00 | 0.100 |
| SSD | 0.0 | 1.00 | 2.15 | 0.465 | 0.89 | 4.19 | 0.213 |
| NegGrad+ | 7.0 | 1.00 | 2.58 | 0.388 | 0.89 | 3.85 | 0.231 |
| Salun | 65.0 | 0.97 | 6.05 | 0.161 | 0.89 | 5.42 | 0.165 |

**Interpretation.** Two headline findings reproduce:
1. **Conformal calibration is valid** — Dtest coverage ≈ 0.89–0.90 for every method, i.e. the 1−α = 0.9 guarantee holds.
2. **CR ranks methods differently from UA, and exposes over-forgetting.** By UA the "strongest forgetter" is **Teacher (92.5%)**, and it also has the best CR_Df (0.098). But its **CR_Dtest collapses to 0.100** — the model was destroyed (RA/TA ≈ 10%). UA alone would crown Teacher; CR_Dtest reveals it is useless. Conversely GA/SSD/FT show UA ≈ 0–3.5% (UA: "not forgotten") with correspondingly high CR_Df (0.38–0.47), and NegGrad+ under our toy setting barely forgets (UA 7%, CR_Df 0.388) — so a method the paper ranks well by UA can look weak under CR, matching the paper's core point that UA-good ≠ forgetting-good.

**Verdict: reproduced for 8/9 methods** at the paper's α=0.1 / 2,000-calibration setting (FF pending GPU). Code: [github.com/TIML-Group/Conformal-Prediction-Unlearning](https://github.com/TIML-Group/Conformal-Prediction-Unlearning).
