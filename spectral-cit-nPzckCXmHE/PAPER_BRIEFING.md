# "Toward Scalable and Valid Conditional Independence Testing with Spectral Representations" — reproduction briefing

Paper: arXiv 2512.19510v2, "Toward Scalable and Valid Conditional Independence Testing with
Spectral Representations", Alek Fröhlich, Vladimir R. Kostić, Karim Lounici, Daniel Perazzo,
Daniel Tiezzi, Massimiliano Pontil. ICML 2026 (PMLR 306). Code repo (authors'):
https://github.com/alekfrohlich/SCIT (not used directly — reproduction is from-scratch, per
challenge rules).

OpenReview id: nPzckCXmHE. OpenReview was bot-wall-blocked (browser-verification challenge on both
the forum page and the PDF endpoint) — this reproduction is sourced entirely from arXiv v2. Local
copy: `paper-arxiv-2512.19510.pdf` (34 pages, full text + all 3 appendices confirmed readable
end-to-end by the Step 0 acquisition subagent, no OpenReview/arXiv cross-check possible since
OpenReview never rendered).

Challenge: HF Space `ICML-2026-agent-repro/challenge`. This reproduction lands in
`nmaher2022/icml2026-reproductions` as `spectral-cit-nPzckCXmHE/`.

## Working conventions for this reproduction
- PEP-723 self-contained Python scripts run via `uv run` (this repo's existing convention — see
  `divide-and-learn-TK82ECnJzD/`, `gaussian-mechanism-82Wosp2Iu1/` etc. for examples). CPU-only
  torch is already installed in `./.venv`.
- **Smoketest before scale**: before any run longer than ~30s-1min, use tiny settings (d_Z small,
  few reps, few training steps) and check for shape errors, NaNs, sane test-statistic magnitudes
  (should be O(d^2) under H0), and that whitening actually produces near-identity empirical
  covariances. Only scale up once clean.
- All work happens in `spectral-cit-nPzckCXmHE/`. Don't touch other folders in this repo.
- Verdict vocabulary: VERIFIED / TOY-VERIFIED / REFUTED / BLOCKED. State the scale run next to
  every verdict. Never round a toy-scale pass up to VERIFIED. Report BLOCKED claims explicitly.
- Self-check before finishing: reread the exact claim text and the numbers/plots side by side —
  does the evidence actually support what's claimed, at the scale actually run?

## Compute note
Unlike prior reproductions in this repo (BiMU, D&L — deep ResNet/ViT training needing GPU-scale
compute, see memory `blocker-no-gpu-hf-jobs-402`), **this paper's claims are all statistical/
theoretical, tested via small MLPs (`n_hidden` 1-4 layers, `layer_size` 128-512) and Monte Carlo
repetition, not large network training.** The paper's own compute was CPU-based (2x Xeon E5-2695,
no GPU needed for SpectralCIT itself — GPUs in their setup were for the TCGA image encoder, not
for this method). This reproduction should be feasible **at or near the paper's own scale** on the
local CPU-only machine, budget permitting session time — no HF Jobs / GPU credit blocker expected
to bind here.

## Claims in scope (verbatim from claims_anchored.json, cross-checked against the PDF)
1. **(Thm 4.1, validity)** Under H0 (X⊥Y|Z), the test statistic T̂_n converges in distribution to
   a chi-squared distribution with d² degrees of freedom as m,n→∞.
2. **(Thm 4.2, power)** Non-asymptotic power guarantee: the test achieves power ≥ 1-δ once the
   separation threshold ϵ²_n ≥ 2d·E²_m + c(d²+log(δ⁻¹))/n, tying power jointly to representation
   error E_m and sample size n.
3. **(the E_m definitions, p.5)** The representation-learning errors E_m^val (validity) and
   E_m^pow (power) are each a max over operator-norm discrepancy terms — E_m^val over
   ‖Ĉ_{ÛV̂}−I_d‖, ‖Ĉ_{V̂V̂}−I_d‖, ‖Ĉ_{ŴŴ}−I_{2d}‖ (orthonormality of learned features); E_m^pow
   is ‖[Σ_{XẎ·Z}]_d − U_θ M_θ V_θ*‖ (SVD approximation quality) — and these single quantities are
   claimed to control null validity and alternative power respectively.
4. **(Algorithm 1)** Bi-level contrastive learning with learned representations u_θ, v_θ, w_θ
   (MLPs) plus empirical whitening estimates the leading spectral features of the partial
   covariance operator Σ_{XY·Z}: inner loop optimizes w_θ against L_in, outer loop optimizes
   u_θ,v_θ against L_out, then a post-hoc whitening step orthonormalizes the learned features.
5. **(Assumption 4.1)** Validity of the chi-squared null requires the learned representations
   û_θ(X), v̂_θ(Ẏ), ŵ_θ(Z) to be K-sub-Gaussian — operationalized in the paper by using bounded
   (Tanh) activations.

Real-data claim (TCGA-BRCA breast cancer, Section 5.2, Table 1) is **not** in the extracted claims
list and requires a Path Foundation Model image encoder + restricted TCGA access — out of scope,
will be marked BLOCKED if touched at all, not treated as a claim to verify.

## Core math / setup (transcribed from the paper)

**Objects.** X, Y, Z random vectors; Ẏ := (Y,Z). Partial cross-covariance operator
Σ_{XẎ·Z} = Σ_{XẎ} − Σ_{XZ}Σ_{ZẎ} (Eq. 3). H0: X⊥Y|Z ⟺ ‖Σ_{XẎ·Z}‖_HS = 0 (Eq. 9).

**Representation learning (Section 3, Eqs 5-8, Algorithm 1).** Parametrize u_θ:X→ℝ^d,
v_θ:Ẏ→ℝ^d, w_θ:Z→ℝ^{2d} as MLPs (Tanh activations, Xavier init). Two losses computed via
U-statistics over a training batch (ū, v̄, w̄ = empirically centered outputs; M = M_θ diag matrix
of singular values, N = (N_θ+N_θᵀ)/2):

  L_out(θ,B) = (1/(m(m-1)))Σ_{i≠j}⟨ū_i,Mv̄_j⟩² − (2/m)Σ_i⟨ū_i,Mv̄_i⟩
             + (2/(m(m-1)))Σ_{i≠j}⟨ū_i,Mv̄_j⟩⟨w̄_i,w̄_j⟩

  L_in(θ,B) = (1/(m(m-1)))Σ_{i≠j}⟨w̄_i,Nw̄_j⟩² − (2/(m(m-1)))Σ_{i≠j}⟨ū_i,Mv̄_j⟩⟨w̄_i,Nw̄_j⟩

Plus orthonormality regularizers Ω_out(θ) = ‖Ĉ_{U_θU_θ}−I_d‖²_F + ‖Ĉ_{V_θV_θ}−I_d‖²_F and
Ω_in(θ) = ‖Ĉ_{W_θW_θ}−I_{2d}‖²_F, weighted by γ.

**Algorithm 1 (bi-level training loop):**
```
for t in 1..n_steps:
    for s in 1..n_steps_inner:           # inner loop: warm up w_θ
        sample minibatch B; g_in = ∇_θ[L_in(θ,B) + γΩ_in(θ,B)]; update w_θ
    sample minibatch B'; g_out = ∇_θ[L_out(θ,B') + γΩ_out(θ,B')]; update u_θ,v_θ
# post-hoc whitening (population-level, over the full training set):
û_θ, v̂_θ, ŵ_θ ← Ĉ_{ŨU}^{-1/2} ũ_θ,  Ĉ_{ṼV}^{-1/2} ṽ_θ,  Ĉ_{W̃W}^{-1/2} w̃_θ
```
Table 2 (Appendix C) reference hyperparameters used in the paper's own Fig. 2/5 experiments
(selected via Weights & Biases Bayesian search, budget 30 trials): output_dim (d) = 10 (nominal),
n_hidden ∈ {1,2,3,4}, layer_size ∈ {128,256,512}, lr_inner/outer ~ LogUniform(3e-5,1e-2)
(reference: (3e-5, 2.1e-3)), reg_str_inner/outer ~ LogUniform(1,100) (reference: (3.3,1.9)),
batch_size ∈ {128,256,512,1024}, perc_dim_prune ~ Uniform(0.85,1). Training: 400 epochs, inner
loop warmed up 100 steps before alternation begins, 80/20 train/test split.

**Test statistic (Eq. 10, computed on the held-out test set D_n^test after training on
D_m^train):**
  T̂_n = n‖Ĉ_{ÛV̂} − Ĉ_{ÛŴ}Ĉ_{ŴV̂}‖²_F

Reject H0 at level α when T̂_n ≥ c_α = q_{1-α}(χ²(d²)) (the 1-α quantile of chi-squared with d²
degrees of freedom).

**Dimension pruning (Appendix C).** For stability, all models trained at fixed output_dim, then at
test time an SVD of the test-statistic matrix retains only the leading
⌊perc_dim_prune × output_dim⌋ singular triplets; the χ² reference distribution's degrees of
freedom are corrected to match the retained (pruned) dimension.

**Synthetic benchmarks used for Claims 1-2 (Section 5.1, Appendix C):**
- *Post-nonlinear model* (primary benchmark, Fig. 2): Z̄=(1/d_Z)Σz_i, Z_i,ε_X,ε_Y,ε iid N(0,1),
  d_Z ∈ {50,...,300}. H0: X=f(Z̄+ε_X/4), Y=g(Z̄+ε_Y/4). H1: X=f(Z̄+ε_X/4)+ε/2, Y=g(Z̄+ε_Y/4)+ε/2.
  f(w)=w³, g(w)=tanh(w). N=1000, 100 reps per d_Z, α=0.05.
- *Signal-strength ablation* (Fig. 11, Appendix C) — cheapest/smallest of the synthetic
  experiments, good smoketest/toy target: fixed d_Z=3, str_z=0.1, noise_str=0.25.
  H0: X=sin(Z+ε_X), Y=cos(Z+ε_Y). H1: X=sin(Z+ε_X)+η, Y=cos(Z+ε_Y)+η, η~N(0,str_cond_dep²·I),
  str_cond_dep ∈ {0.05,0.15,0.5}. Z,ε_X,ε_Y ~ N(0,0.1²I_{d_Z}). 500 reps.
- *High-dim nonsmooth benchmark* (Fig. 10): X=f(Z/2+ε_X), Y=g(Z/2+ε_Y) with f,g highly
  oscillatory near 0 (defined piecewise via cos(2π/w) for 0<|w|<1). d_Z ∈{50,100,200,300}, 5 reps.

## Known access blockers
- OpenReview forum/PDF: bot-detection wall, could not be fetched at all (Step 0). Reproduction
  sourced from arXiv v2 only.
- TCGA-BRCA real-data experiment (Section 5.2): requires restricted-access TCGA WSIs + a
  pathology foundation model (Path-FM) encoder — out of scope, not in claims list, would be
  BLOCKED if attempted.
- Baseline methods (KCIT, RCIT, GCIT, DGCIT, NNLSCIT) are used for *comparison* in the paper's
  figures but are not themselves claims in `claims_anchored.json` for this OpenReview id — scope
  is SpectralCIT's own behavior (Claims 1-5), not a head-to-head baseline bake-off. May implement
  a minimal RCIT-style baseline opportunistically if time allows, but it is not required for any
  in-scope claim.
