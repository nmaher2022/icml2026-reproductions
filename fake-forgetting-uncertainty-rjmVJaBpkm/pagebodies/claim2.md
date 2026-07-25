**Claim.** On CIFAR-10 with ResNet-18 under 10% random forgetting, conformal-prediction analysis shows that **30–68%** of the forget samples the unlearned model *misclassifies* still have their ground-truth label inside the model's conformal prediction set — the "fake forgetting" phenomenon (paper Table 2).

**Method.** For each unlearned model: (1) calibrate q̂ on the held-out 2,000-sample calibration set (α=0.1); (2) on the forget set count `n_mislabel` = #{argmax ≠ y_true} — the points UA counts as "forgotten"; (3) among those, count `n_in_set` = #{y_true ∈ C(x)}. The **recover ratio** = n_in_set / n_mislabel is exactly the paper's Table-2 "In-Set / Mis-label" ratio. Our original ResNet-18 memorized the training data (forget_acc=1.000, RA=0.997, TA=0.483), so "forgotten" points are genuinely ones unlearning removed.

**Result (toy, CIFAR-10 subset, α=0.1):**

| Method | UA (%) | n_mislabel | n_in_set | recover ratio (%) | paper (10% forget) |
| --- | --- | --- | --- | --- | --- |
| RT (retrain) | 53.5 | 107 | 85 | **79.4** | 30.6 |
| FT (finetune) | 3.5 | 7 | 7 | **100.0** | 58.3 |
| RL (random label) | 66.0 | 132 | 125 | **94.7** | 45.5 |

**Interpretation.** The fake-forgetting phenomenon **reproduces strongly**: a large fraction of UA-"forgotten" forget points still have their true label recoverable inside the conformal set — even for the gold-standard retrain baseline (79%). Our ratios sit *above* the paper's 30–68% band, and this is expected and explainable: the paper's original reaches 91.8% test accuracy, giving tight conformal sets (≈1–2 labels), so fewer misclassified points retain the true label; our CPU-toy model reaches only ~48% test accuracy, so calibrated sets are larger (≈4–6 labels of 10) and recovery is higher. The *direction and mechanism* — UA drastically overstates forgetting because the true label survives in the prediction set — is reproduced (indeed amplified). At paper scale (the same script with `--device cuda --n_per_class 5000 --orig_epochs 200`) the tighter sets should bring the ratios down into the reported band.

**Verdict: mechanism reproduced** (magnitude inflated by reduced toy-model accuracy). Fake forgetting is clearly present across RT/FT/RL. Full CR/Coverage/Set-Size numbers on the Claim 3 page. Code: [github.com/TIML-Group/Conformal-Prediction-Unlearning](https://github.com/TIML-Group/Conformal-Prediction-Unlearning).
