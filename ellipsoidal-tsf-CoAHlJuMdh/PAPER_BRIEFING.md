# Ellipsoidal Time Series Forecasting — reproduction briefing

Paper: arXiv 2505.17370v6, "Ellipsoidal time series forecasting", Qilin Wang.
OpenReview id: CoAHlJuMdh. Local copy: `paper_arxiv_2505.17370v6.pdf` (23 pages), full text
extracted to `paper_text.txt`.

**OpenReview/arXiv cross-check**: OpenReview is bot-walled from this environment (confirmed,
consistent with every prior reproduction in this repo) — could not fetch the OpenReview PDF at
all. arXiv v6 (2505.17370v6, dated 2 May 2026) is the sole and therefore authoritative source.
Appendix confirmed present and readable (pp. 12/13–23, Tables 4–15).

Challenge: HF Space `ICML-2026-agent-repro/challenge`. This reproduction lands in
`nmaher2022/icml2026-reproductions` as `ellipsoidal-tsf-CoAHlJuMdh/`.

**IMPORTANT — claims.json/claims_anchored.json staleness found and NOT used as-is.** The
pre-extracted `claims_anchored.json` entries for `CoAHlJuMdh` cite specific numbers (Lorenz63 MSE
11.4 vs TimeMixer 27.3 vs DLinear 65.4; Rossler MSE 0.02 vs TimeMixer 2.52 vs DLinear 4.58; "274
of 336 steps ≈ 2.5 Lyapunov times"; an "ANF ablation" raising MSE 51%; a rotation ablation raising
MSE 35%) that **do not match any table in the actual v6 PDF**. Table 2 (main stress test) shows
Rossler-Base fr=0.019 vs tm=1.03 vs dl=5.42; Lorenz-Base fr=21.66 vs tm=43.21 vs dl=76.55 — an
order of magnitude off from the anchored claim's numbers, and Fern's Lorenz63 MSE is ~21-22
throughout the paper, never ~11.4. "ANF" appears exactly once in the text, as a citation for prior
work the encoder is "inspired by" (Huang et al. 2020), not as a named ablation arm — the paper's
actual ablation table (Table 3 / Table 8) only has: Base, No encoder & no mean updates, Only
encoder, No rotation, No patching, Reflections={2,24}. This is the same claims-extraction
version-staleness pattern already seen and documented for GameDevBench and STAND in this repo
(prior extraction likely ran against an older draft). Per this harness's Step 0 rule, claims below
are transcribed directly from the v6 PDF, not from the anchored JSON. Not filed upstream this
session (would need to ask the user first, same as the GameDevBench precedent).

## Working conventions for this reproduction
- CPU-only local machine (no CUDA) — see `blocker-no-gpu-hf-jobs-402` memory. Everything below is
  toy-scale by design: short trajectories, small models, few epochs.
- PEP-723 self-contained Python scripts. Torch is CPU-only in `../.venv` — **invoke scripts via
  `../.venv/bin/python script.py`, not `uv run`** (prior sessions found `uv run` on a
  torch-dependent script fetches a fresh 500MB+ CUDA build from PyPI instead of reusing the
  pre-installed CPU venv).
- All work happens in `ellipsoidal-tsf-CoAHlJuMdh/`. Don't touch other folders in this repo.
- **Smoketest before scale**: before running anything longer than ~30s-1min, run a tiny/fast
  version (few timesteps, tiny model, 1-2 epochs) and check for shape errors, NaNs, sane
  magnitudes, sign errors. Only scale up once clean.
- Verdict vocabulary: **VERIFIED** (matches paper's own scale) / **TOY-VERIFIED** (matches
  directionally/qualitatively at reduced scale, not claiming paper's exact numbers) / **REFUTED**
  (ran fairly, contradicts the claim) / **BLOCKED** (not attempted, state exactly why) /
  **INCONCLUSIVE** (ran it, result doesn't cleanly support or contradict). Never round
  TOY-VERIFIED up to VERIFIED. State the scale run next to every verdict. Never silently skip a
  blocked claim.
- Self-check before finishing: reread each claim's exact text and the actual numbers/plots
  side by side — does the evidence really support what the claim says, at the scale run?

## Claims in scope (verbatim/closely paraphrased from the v6 PDF, section/table refs)

1. **Nonstationary robustness (headline claim).** "Fern demonstrates exceptional stability,
   outperforming baselines like DLinear and Koopa by over two orders of magnitude (up to 790×) on
   nonstationary settings" (Abstract). Table 2 (main text, p.6): Fern is best MSE/WD in 19 of 21
   stress-test scenarios (stochastic + chaotic systems under base/param/state/switch shocks),
   including "98% lower MSE than TimeMixer on Rössler."

2. **Linear-time complexity via Householder SPD factorization.** "This formulation reduces the
   computational cost of eigen-decomposition from cubic to linear time while providing
   interpretable, geometry-aware projections" (Abstract). Full derivation Section 2 + Appendix
   A.3.2: dense per-patch SPD application/generation is O(B·g·p²) (or O(n³) for full
   eigendecomposition without patching, n=g·p), Fern's Householder-factored SPD map costs
   `T_Fern = O(B·g·(Kenc·p·dh + p·dh + R·p))` (Eq. 1-2), i.e. linear in patch size p for fixed R.
   Table 10 reports empirical training time and FLOPs (Fern: 1.025M params, 0.0035 GFLOPs on
   Lorenz63, vs. TimeMixer 0.886M params / 0.0463 GFLOPs, PatchTST 2.008M / 0.0308 GFLOPs).

3. **Effective Prediction Time (EPT) — Fern stays reliable longer than baselines on chaotic
   systems.** EPT is defined (Appendix A.1) as the first horizon step at which |error| exceeds one
   training-set standard deviation ε_d, per dimension, averaged: `EPT_{b,d} = min{s : |y_pred -
   y_true| > ε_d}` (= H if never exceeded). Tables 11 (Chua) / 12 (Rossler) / 15 (Lorenz) report
   average EPT per model. E.g. Table 12 (Rossler, dt=1e-2): avg EPT Fern=324.2 vs
   TimeMixer=251.8, PatchTST=223.9, DLinear=232.4. Table 11 (Chua): avg EPT Fern=274.4 vs
   TimeMixer=262.7, PatchTST=213.0, DLinear=185.6.

4. **Ablations confirm the encoder, Householder rotation, and patching are each doing real work**
   (Table 3 main text / Table 8 appendix, prediction length 192). Removing the encoder and mean
   updates is catastrophic on Lorenz63 (MSE 21.66 → 194.43, ~9x worse, EPT collapses 241.25 →
   17.10). Removing rotation (no Householder SPD, presumably identity/diagonal-only transport)
   increases Lorenz63 MSE 21.66 → 27.62 (~27%) and ETTh1 MSE 10.96 → 11.84 (~8%). Removing
   patching also degrades both (Lorenz63 21.66 → 22.86, ETTh1 10.96 → 10.99).

5. **Geometric accuracy persists on Lorenz-63 past the horizon where pointwise prediction becomes
   provably impossible.** Main text (p.7): baselines collapse to mean-guessing early — "DLinear at
   horizon 96, TimeMixer/PatchTST at 192 (Table 15)" — while "Fern maintains pointwise accuracy
   until horizon 720... roughly 6.5 Lyapunov times, where errors amplify ≈650× and pointwise
   prediction is provably impossible — geometry persists: SWD 4.89 versus 10–40 for baselines."

## Core math / setup (transcribed from the PDF)

**Model (Algorithm 1, Section 2, p.3):** Given windowed input x, sample z, y0 ~ N(0, I). For
i = 1..Kenc=5 encoder layers: compute shared features h^i_x = H_x(x^i), h^i_z = H_z(z^i) via a
bidirectional coupling network (inspired by ANF/affine coupling flows — the encoder's own
architectural inspiration, not an ablation arm), generate affine scale/shift heads (s^i_z, t^i_z)
= φ_x(h^i_x) and update z^{i+1} = s^i_z ⊙ z^i + t^i_z, then (s^i_x, t^i_x) = φ_z(h^i_z) and update
x^{i+1} = s^i_x ⊙ x^i + t^i_x. After Kenc layers, final feature h_z = H_z(z^{Kenc+1}) feeds an OT
head ψ producing (Λ, t_y, U): Λ = nonnegative diagonal eigenvalues (patch-wise, dim p), t_y =
shift, U = orthogonal matrix built from R Householder reflections H_i = I − 2v_iv_i^T (unit-norm
v_i, R≤p; base config R=8). Output y* = U_y^T Λ_y U_y (y0 + t_y) — the Brenier/W2-optimal affine
map from N(0,I) to N(µ,Σ) with Σ = AA^T, A = UΛU^T. Training: "simple Huber-loss" on the predicted
mean vs ground truth, **no direct supervision on eigenvalues** — the spectral structure emerges
from minimizing the point-forecast loss alone (this is the whole point of the abstract's
"interpretable projections" claim, tested qualitatively in Claim 1/Fig.1's speed-correlated
max-eigenvalue pattern — out of scope numerically here, in-scope only via the ablation/complexity
claims above).

**Patching:** input horizon n split into g patches of size p (n = g·p; base config patch size
p=24). Each patch's transport is computed independently → parallel, and the complexity analysis
(Claim 2) is per-patch.

**Metrics (Appendix A.1):**
- MSE: standard.
- W2/WD (1D): sort predicted and true values within a horizon (ignoring time order) and take mean
  squared difference of order statistics: `W2²(y*, y) = (1/H) Σ_h (y*_(h) - y_(h))²`.
- SWD: sliced W2, project onto L=500 random directions, average 1D W2.
- EPT: `EPT_{b,d} = min{s ∈ {1..H} : |y_pred_{b,d,s} - y_true_{b,d,s}| > ε_d}` where ε_d is the
  training-set std of dimension d; = H if never exceeded. Report mean over batch/dims.

**Synthetic chaotic systems (Appendix A.4.1, Table 7) — exact equations/params:**
- Lorenz-63: ẋ=σ(y−x), ẏ=x(ρ−z)−y, ż=xy−βz; σ=10, ρ=28, β=8/3; dt=0.01, RK4, float64→float32.
- Rössler: ẋ=−y−z, ẏ=x+ay, ż=b+z(x−c); a=0.2, b=0.2, c=5.7; dt=0.01, RK4.
- Chua's circuit: ẋ=α(y−x−h(x)), ẏ=x−y+z, ż=−βy, h(x)=m1·x + ½(m0−m1)(|x+1|−|x−1|);
  α=15.6, β=28.0, m0=−8/7, m1=−5/7; dt=0.005, RK4.
- "Param" shocks perturb these constants partway through the trajectory (see Table 7 for exact
  deltas per system); "state"/"switch" shocks perturb initial conditions or make a mid-trajectory
  jump.

**Baselines available for comparison**: DLinear (Zeng et al. 2023 — simple moving-average
decomposition + per-component linear layer, easy to implement from scratch) is the primary
baseline to reimplement given CPU/toy scope; TimeMixer/PatchTST/Koopa/ModernTCN/PFNN are heavier
and out of scope to reimplement faithfully at toy scale — their absence doesn't block Claim 1 (a
"beats DLinear by orders of magnitude" check is directly testable with just DLinear as baseline;
the 790x figure blends multiple baselines/scenarios, treat as directional).

## Toy-scale reproduction plan
- Implement Fern (Algorithm 1) and DLinear from scratch in PyTorch (CPU), plus the three chaotic
  generators (Lorenz-63, Rössler, Chua) and the MSE/W2/SWD/EPT metrics, all self-contained.
- Toy scale: short trajectories (~3-6k steps vs paper's 25-36k), small models (encoder width/depth
  reduced), few epochs, single seed for the smoketest then 3 seeds for the toy runs (vs paper's 4).
- Claim 1: train Fern + DLinear on Lorenz/Rossler/Chua base + param-shock scenarios, compare MSE —
  check Fern beats DLinear by a large margin, direction and rough order of magnitude, not exact
  790x.
- Claim 2: analytic FLOPs/param-count check from the actual instantiated model (count Householder
  vs dense-SPD parameter/FLOP cost directly from code) — this is a structural/analytical claim, not
  compute-scale-dependent, so checkable exactly even at toy scale. Also empirical wall-clock scaling
  vs patch size p as a secondary check.
- Claim 3: compute EPT for Fern vs DLinear on the toy runs, check Fern's EPT ≥ DLinear's,
  directionally.
- Claim 4: implement the ablation variants (no-rotation = identity U; no-encoder = skip encoder
  loop, feed raw x/z; no-patching = single patch g=1) and rerun, check MSE moves in the same
  direction reported (each ablation increases MSE vs base).
- Claim 5: on Lorenz-63, compare per-horizon MSE/SWD growth for Fern vs DLinear across increasing
  prediction horizons at toy scale, check Fern's error growth is slower / SWD stays lower at long
  horizons — qualitative check of "geometry persists past the pointwise-collapse point," not a
  literal Lyapunov-time replication (toy trajectories are far shorter than the paper's).

## Known access blockers
- OpenReview PDF unreachable (bot wall) — arXiv v6 used as sole source, see above.
- TimeMixer/PatchTST/Koopa/ModernTCN/PFNN baselines not reimplemented at toy scale (heavy
  transformer/mixer baselines, disproportionate effort for a toy-scale CPU check) — Claim 1 tested
  against DLinear only, disclosed as a scope limitation, not silently substituted.
- ETTh1/ETTm1/Weather real-world datasets not used — all 5 in-scope claims are checkable on the
  paper's own synthetic chaotic benchmarks, which the paper generates itself (no external data
  access needed), so this isn't a blocker for any claim, just a scope choice to avoid unnecessary
  real-dataset fetching.
