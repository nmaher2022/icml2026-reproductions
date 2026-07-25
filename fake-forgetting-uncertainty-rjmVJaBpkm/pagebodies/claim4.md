**Claim.** The MIA Conformal Ratio (MIACR) reveals that methods scoring well under traditional membership-inference-attack (MIA) evaluation can perform poorly under MIACR, indicating that traditional MIA is an unreliable proxy for forgetting quality (paper Table 6).

**Method.** Traditional MIA: an SVC membership classifier trained on retain-vs-test confidences, applied to the forget set; the reported value is the fraction of forget points predicted "member" — **lower = forget data looks more test-like = better forgetting**. MIACR: the official `MIACR.SVC_MIA` wraps that same SVC in split conformal prediction (calibrated q̂) and reports a coverage/set-size ratio on the membership signal — the conformal analogue of CR. Both are computed per method (α=0.1, 2,000-sample calibration).

**Result (toy):**

| Method | MIA (traditional) | MIACR |
| --- | --- | --- |
| RT | 0.43 | 0.018 |
| FT | 0.83 | 0.000 |
| RL | 0.15 | 0.018 |
| GA | 0.94 | 0.003 |
| Teacher | **0.00** | **0.039** |
| SSD | 0.96 | 0.000 |
| NegGrad+ | 0.81 | 0.000 |
| Salun | 0.15 | 0.031 |

**Interpretation.** The two metrics **disagree — indeed they invert**, exactly as the claim states:
- **Teacher** looks like the *best* method under traditional MIA (0.00, forget data appears fully non-member) but has the **highest / worst MIACR (0.039)**. A method that "wins" on MIA is flagged by MIACR.
- **Salun and RL** also score well on MIA (0.15) yet carry relatively high MIACR (0.031, 0.018).
- Conversely **FT, SSD, NegGrad+** score badly on MIA (0.81–0.96) but have the *lowest* MIACR (0.000).

So ranking methods by traditional MIA gives essentially the opposite ordering to MIACR. This supports the paper's conclusion that **traditional MIA is an unreliable proxy for forgetting quality** — a strong-MIA score does not imply genuine forgetting once uncertainty is accounted for.

**Verdict: mechanism reproduced** — the MIA/MIACR disagreement is clear and in the claimed direction. (Note: the SVC-MIA estimates are noisier at toy scale; the *inversion* is the reproducible signal, not the exact values.) Code: [github.com/TIML-Group/Conformal-Prediction-Unlearning](https://github.com/TIML-Group/Conformal-Prediction-Unlearning).
