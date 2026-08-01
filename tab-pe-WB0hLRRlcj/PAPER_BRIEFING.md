# Differentially Private Synthetic Data via APIs 4: Tabular Data (Tab-PE) — reproduction briefing

Paper: arXiv 2606.08259v1, "Differentially Private Synthetic Data via APIs 4: Tabular Data",
Toan Tran, Arturs Backurs, Zinan Lin, Victor Reis, Li Xiong, Sergey Yekhanin (Emory University /
Microsoft Research). ICML 2026.
OpenReview id: WB0hLRRlcj. Local copy of the PDF: `tab-pe-WB0hLRRlcj/paper.pdf`
(`tab-pe-WB0hLRRlcj/paper.txt` for the pdftotext extraction used for quotes below).
OpenReview PDF was bot-walled (consistent with this environment's known hard constraint); arXiv
used as the (only available, and here sufficient) source. No OpenReview/arXiv diff possible.

Challenge: HF Space `ICML-2026-agent-repro/challenge`. This reproduction lands in
`nmaher2022/icml2026-reproductions` (this repo) as `tab-pe-WB0hLRRlcj/` (current working folder,
already at repo root — `git remote -v` confirms this checkout *is* the target monorepo, so no
separate clone/copy step is needed for Step 7).

**Official code exists and is used as the primary implementation**: `github.com/microsoft/DPSDA`
(Microsoft's "Private Evolution" library), cloned read-only into
`tab-pe-WB0hLRRlcj/DPSDA_upstream/` (gitignored — vendored upstream code, not committed, per
`no_vendored_code` harness convention). `example/tabular/{artificial_characters,person_activity,
xor_stress_test}.py` are the paper authors' own scripts that generate exactly the Table 1 / Fig 1
numbers this reproduction checks. This is an "official code" reproduction for Tab-PE itself, not a
from-scratch reimplementation.

## Working conventions for this reproduction
- Python venv at `tab-pe-WB0hLRRlcj/.venv` (separate from the repo-root `.venv`, since this needs
  torch + DPSDA's `[tabular]` extras that other reproductions don't). Install via
  `pip install -e ".[tabular]"` from inside `DPSDA_upstream/`, run scripts with
  `.venv/bin/python <script>.py` directly, **not** `uv run` — a prior reproduction in this repo
  (`spectral-cit-nPzckCXmHE`) found `uv run` on a torch-dependent PEP-723 script fetches a fresh
  GPU-flavored torch build from PyPI instead of a CPU-only one; avoid that here too. Force CPU
  torch explicitly (`pip install torch --index-url https://download.pytorch.org/whl/cpu` if the
  default resolves a CUDA build).
- **Smoketest before scale**: before running anything longer than ~30s-1min, run the smallest
  variant (XOR with 1 feature, or a script with `num_iterations`/`num_samples` cut way down) and
  check for shape errors, NaNs, sane magnitudes, sign errors. Only scale up once clean.
- All work happens in `tab-pe-WB0hLRRlcj/`. Don't touch other folders in this repo.
- Verdict vocabulary: VERIFIED / TOY-VERIFIED / REFUTED / BLOCKED. State the scale run next to
  every verdict. Never round a toy-scale pass up to VERIFIED. Report blocked claims explicitly,
  never fake or silently skip them.
- Self-check before finishing: reread the exact claim text and our own numbers/plots side by side
  — does the evidence actually support what the claim says, at the scale we ran it?
- **Baseline scope decision**: the paper's own baselines (PrivMRF/GEM/RAP++/PrivGSD) are marked
  "(GPU)" in Fig. 4's legend in the paper itself — they don't *need* GPU algorithmically but the
  paper ran them on GPU for speed. Reimplementing all of them is out of scope. **AIM** (the paper's
  own "best baseline", McKenna et al. 2022) is *not* GPU-tagged and has an official, actively
  maintained, pip-installable CPU implementation (`mbi`, by AIM's own original author Ryan
  McKenna) — use it for a real same-hardware head-to-head against Tab-PE on Claims 1/2/3, rather
  than just quoting the paper's AIM numbers. If `mbi`'s API doesn't map cleanly onto these datasets
  within reasonable effort, fall back to citing the paper's reported AIM numbers as a reference
  point and say so explicitly in the verdict (not silently).
- The downstream classifier the paper actually used for Table 1 is **TabICL** (Qu et al. 2025,
  transformer tabular foundation model, in-context learning, no training) — confirmed at
  paper.txt line ~1250 ("We use the SOTA classifier TabICL ... for all datasets. ... For all
  methods, we fit TabICL on the generated [data]"), and matches the official example scripts'
  default (`model_name="tabicl"`). Use TabICL, not XGBoost, for the two real-dataset accuracy
  claims (1 and 2) to match what actually produced the paper's numbers. TabICL runs on CPU (slower,
  no training needed — just in-context inference) via the `tabicl` PyPI package (torch-based,
  pretrained weights pulled from HF Hub). If TabICL proves infeasible on CPU at real dataset scale
  within a reasonable session budget, fall back to XGBoost and disclose the substitution plainly
  (this changes what's being measured, so it would cap the verdict at TOY-VERIFIED with a stated
  caveat, not VERIFIED).

## Claims in scope (verbatim from claims_anchored.json, cross-checked against the paper's own text)

1. **Artificial Characters, ε=1.0**: Tab-PE 49.38±0.46% accuracy vs AIM 23.24±1.48% (Table 1,
   Section 5.4, page 6/paper.txt line ~424-425). Also reports Macro F1 48.09 vs 20.17, and
   Section 5.4 prose states "+9.02% accuracy and +8.99% macro F1" — note this delta is versus the
   *second*-best baseline (PrivGSD 40.36%), not AIM; the paper's headline "vs AIM" framing in the
   challenge's claim text is really "Tab-PE beats every baseline including AIM by a wide margin."
2. **Person Activity, ε=1.0**: Tab-PE 63.72±0.18% vs AIM 59.53±0.47% (Table 1, same section).
   Macro F1 35.09 vs 30.79.
3. **Compute efficiency, no GPU**: "Tab-PE runs entirely on CPUs" (Section 5.5, page 8, quoted
   exactly in Step 0's report) and is ~28x faster than AIM, ~10x faster than PrivMRF at ε=1.0
   (Fig. 4a), and 18.6x faster than the leading baseline at 500K synthetic samples (Fig. 4c,
   paper.txt line ~511-527). Baselines other than AIM/PrivSyn are GPU-tagged in the paper's own
   legend — see baseline scope decision above.
4. **XOR stress test, 5 features**: at ε=1.0, Tab-PE achieves AUC≈0.8 while all marginal-query
   baselines (set up with the *ideal* marginal degree K=num_features+1) collapse to random-guess
   performance (Section 5.2, Fig. 1, paper.txt line ~444-456: "all the baselines fail completely at
   5 features, delivering a downstream performance of random guess. In contrast, Tab-PE
   successfully yields an AUC score of 0.8 for 5 features.").
5. **Algorithm 2 structure**: Tab-PE's generation loop = RANDOM_API (init) → T iterations of
   VARIATION_API + DP_NN_HISTOGRAM scoring → selection, with a two-stage selection schedule:
   sampling-with-replacement (with variation degree m=1) for the first `T_sampling` iterations,
   then top-K ranking selection (with variation degree m>1, keeping prior selected samples too) for
   the rest (Section 4, Algorithm 1 + Algorithm 2, paper.txt lines 250-330). This is a structural/
   code claim, verified by matching the official implementation's control flow against the
   pseudocode, not by a single numeric result.

## Core math / setup (transcribed from the paper)

**Algorithm 1 — DP_NN_HISTOGRAM**(D_priv, Population P, noise multiplier σ):
for each private sample s, find nearest neighbor index i = argmin_j distance(s, P[j]), increment
hist[i]; after scanning all of D_priv, add i.i.d. N(0, σ²) noise to every histogram bin. Sensitivity
of the histogram query is 1 (each private sample can only affect one bin) → Gaussian mechanism.

**Distance metric** (Eq. 4, mixed categorical+numerical):
`distance(a,b) = λ·Σ_i 1[a_cat(i) ≠ b_cat(i)] + Σ_j ((a_num(j) - b_num(j)) / (max(X_num(j)) - min(X_num(j))))²`
λ is a hyperparameter balancing categorical vs numerical contribution.

**VARIATION_API** (Eq. 2/3, random-walk perturbation): categorical attribute i resampled uniformly
from its domain with probability µ_cat (else kept); numerical attribute j perturbed by
`x' = Π_{X_num(j)}(x + φ)`, `φ ~ N(0, τ²)`, `τ = µ_num · (max(X_num(j)) - min(X_num(j)))`, projected
back into the valid range. Both µ_cat and µ_num follow polynomial decay:
`µ(t) = µ_init - (µ_init - µ_final)·(t/T)^γ`.

**Algorithm 2 — Tabular Private Evolution** (per class c ∈ C, run independently, results unioned):
```
P_0 = RANDOM_API(N^(c))                      # N^(c) = N * |D_priv^(c)| / |D_priv|
for t = 0 to T-1:
  hist_t = DP_NN_HISTOGRAM(D_priv^(c), P_t, σ)
  if t <= T_sampling:
    hist_t[i] = max(0, hist_t[i]); prob[i] = hist_t[i] / sum(hist_t)
    D_t = sample N^(c) from P_t with replacement, weighted by prob
  else:
    D_t = top N^(c) of P_t by hist_t
  if t < T_sampling:
    P_{t+1} = VARIATION_API(D_t, m=1)          # small population -> higher avg histogram counts -> lower noise sensitivity
  else:
    P_{t+1} = VARIATION_API(D_t, m) ∪ D_t       # keep_selected=True in the official code
D_syn = D_syn ∪ D_{T-1}
```
Default hyperparameters (App. C.3, Table 4, and matching the official example scripts):
T=15 (real datasets) or T=20 (XOR), T_sampling≈5 (first 5 iterations use `population1` in the
official code's `CompositePopulation`), µ_init=0.5, µ_final=0.01-0.02, polynomial decay γ=0.2,
ε=1.0, δ=1/(|D_priv|·ln|D_priv|), variation degree m=3 in the ranking stage
(`initial_variation_api_fold=3` in the official artificial_characters.py). Synthetic sample counts:
1000 for Artificial Characters, presumably 5000 for Person Activity (check `person_activity.py`),
2000 for simulation datasets by default (paper's App C.3; verify against the actual example script
args when reading it, since the paper text and the released code may not agree on the exact number
— note any divergence found).

**Privacy analysis** (App B.1): reused from the original PE paper's Gaussian-mechanism +
composition-theorem proof (Lin et al. 2024, their Section 4.3) — not independently re-derived here,
out of scope for a claim about numeric results.

## Known access blockers (fill in as discovered)
- OpenReview PDF: bot-walled, worked around via arXiv (not a blocker for claim verification, just
  for Step 0 sourcing — noted above).
- TabICL feasibility on CPU at Person Activity's scale (164,860 samples, though only the 1000-15000
  *synthetic* samples plus the test split get fed to TabICL, not the full private set) — unverified
  until actually run; log the outcome here once known.
- AIM (`mbi` package) integration effort/feasibility — unverified until attempted; log outcome here.
