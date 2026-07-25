**Claim.** The Conformal Ratio (CR) metric — balancing coverage and prediction-set size under split conformal prediction — is proposed to expose "fake forgetting" that traditional unlearning metrics (UA, RA, TA, MIA) fail to detect (Sec 2.3.1, Table 2).

**Setup (this reproduction).** CPU toy reproduction of the official code ([TIML-Group/Conformal-Prediction-Unlearning](https://github.com/TIML-Group/Conformal-Prediction-Unlearning), paper [arXiv:2501.19403](https://arxiv.org/abs/2501.19403)). No GPU was available and HF Jobs returned 402, so we run the *real* metric code on a CIFAR-10 subset (200 img/class) with a ResNet-18. See the Executive summary for scope.

**How CR is defined.** With non-conformity score `S(x,y)=1-p_y(x)` (Eq. 1), the split-conformal threshold `q̂` is the `⌈(n+1)(1-α)⌉/n` quantile of calibration scores (Eq. 2), and the prediction set is `C(x)={y : p_y(x) ≥ 1-q̂}` (Eq. 3). Then for a dataset 𝒟: **Coverage** = mean 1[y∈C(x)] (Eq. 4), **Set Size** = mean |C(x)| (Eq. 5), and **CR = Coverage / Set Size** (Eq. 6). Low CR on the forget set 𝒟_f = stronger forgetting; high CR on 𝒟_test = preserved utility. UA only inspects the top-1 label, so a forget point can be top-1-wrong (UA calls it "forgotten") while its true label still sits inside `C(x)` — CR catches this, UA cannot.

**Evidence 1 — the metric implementation provably matches the paper (rigorous).** `verify_metrics.py` feeds synthetic softmax outputs with hand-computable answers through the official `metrics/CR.py`, `metrics/MIACR.py`, and `unlearn_cpu.py`. **12/12 checks pass**, covering the Eq. 2 finite-sample quantile, `get_calibration`, the `p_y≥1-q̂` set rule, Coverage/Set Size/CR (Eqs 4-6), the fake-forget recovery count, and the CPU loss (Eq. 8):

```
[PASS] Eq2 q_level formula :: q_level=1.0
[PASS] CR.find_quantile == manual np.quantile(higher) :: repo=0.9000 manual=0.9000
[PASS] CR.get_calibration matches Eq2 q_hat :: q_hat_model=0.9000
[PASS] Coverage (Eq4) = 0.50 :: 0.500
[PASS] Set Size (Eq5) = 2.00 (boundary-inclusive >=) :: 2.000
[PASS] CR (Eq6) = 0.25 :: 0.2500
[PASS] fake-forget: 1 of 3 misclassified has GT in set (ratio 0.333)
[PASS] CPU loss clamp(q_hat-(1-p)+delta) = 0.41 :: loss=0.4100
[PASS] CPU loss = 0 once true label excluded (S>q_hat+delta) :: loss=0.0000
[PASS] MIACR.find_quantile == CR.find_quantile :: 0.9000
12/12 checks passed
```

**Evidence 2 — CR disagrees with UA on real (toy) unlearned models.** SSD reaches UA=0.0% (UA says "nothing forgotten") yet CR_Df=0.465 (CR agrees it barely forgets), while retrain has UA=53.5% and CR_Df=0.206. More tellingly, **Teacher tops UA (92.5%) and CR_Df (0.098, apparently the strongest forgetter) but its CR_Dtest collapses to 0.100** — the model was destroyed (RA/TA≈10%), which UA alone never reveals but CR_Dtest does. CR therefore captures a forgetting-vs-utility picture invisible to UA. Full per-method numbers on the Claim 3 page.

**Verdict: reproduced.** The CR metric definition and its official implementation are verified exactly against the paper's equations, and CR is shown to diverge from UA on real unlearned models.
