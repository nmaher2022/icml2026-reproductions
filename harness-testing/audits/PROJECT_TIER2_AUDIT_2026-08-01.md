# Tier-2 qualitative audit — 2026-08-01

Independent second-reader pass per `harness-testing/AUDIT.md`. Six folders reviewed (exceeds the
4-5 minimum), prioritized by claimed-result size/surprise and by mandate (BiMU). `nPzckCXmHE` and
`WB0hLRRlcj` already have Step-8 audits and were not re-reviewed. All Tier-1 runs below were
re-executed live (not taken on faith) via `uv run harness-testing/audit_harness.py <folder>`.

## active-continual-learning-bimu-SPZd0HVyiS (mandatory re-check)

**Clean — both prior corrections verified present and numerically exact.** README's Claim 2 row
cites 0.996 vs 0.979 epistemic-AUC and 66.66% vs 77.47% accuracy-retention. Recomputed both
independently from raw `.npy` files:
- Epistemic AUC: `mean(roc-auc-epistemic-task=0..9)` = 0.99586 (bimu) / 0.97854 (bayesbinn) — matches
  "0.996 vs 0.979" exactly.
- Accuracy retention: `mean(results/{method}/accuracy/split=0-task=9-epoch=0.npy)` (the final
  10-element per-task-accuracy vector) = 0.66655 (bimu) / 0.77471 (bayesbinn) — matches "66.66% vs
  77.47%" exactly.

The README explicitly states the two corrections from the earlier independent re-audit ("`sum_grads`
gate is global, not per-parameter" and "OOD-AUC values are a per-task-boundary trajectory, not
repeated final-model probes") and the committed data structure matches: 100 separate
`task={k}-epoch=0.npy` files per method/metric confirm the trajectory reading, not a single
repeated-probe file. These are exactly the two bug classes `verdict_checklist.md` calls out
(gate granularity, metric-measures-something-else) — good evidence the corrections were real, not
cosmetic. BLOCKED claims (1, 4, 5) all name a concrete, still-live obstacle (Google Drive auth,
Kaggle token) rather than "didn't get to it." Claim 3's code-level verification against
`optimizers/bimu.py` Eq. 6-7 could not be independently re-checked here since the upstream repo
isn't vendored in this folder (by design — "link, don't vendor") — taken on faith, correctly
disclosed as "plausible but not independently tested" for the causal (1000-task) consequence.
**Confidence: high** — the two headline numbers were traced to raw data byte-for-byte, not just
skimmed.

## divide-and-learn-TK82ECnJzD

Most heavily self-audited folder in the repo (4 rounds, 15 numbered findings in `BUGFIX_LOG.md`,
556 lines) but has **no folder-level `README.md` or `PAPER_BRIEFING.md` at all** — verdicts live
only as a terse top-level-README summary ("2 qualitatively supported / 4 falsified") and prose
inside `BUGFIX_LOG.md`. Live Tier-1 re-run confirms: 0 hard failures, but `verdict_vocabulary` WARNs
0/6 (BUGFIX_LOG uses "falsified"/"qualitatively supported," never the canonical
VERIFIED/TOY-VERIFIED/REFUTED/BLOCKED terms), and `briefing_exists` WARNs. Traced Round 3's revised
Claim-3-ablation table against `claim3_ablation.csv`: `all=0.7110, all-TS=0.6729, UCB-only=0.8810,
EXP3-only=0.6455, FTRL-only=0.8553, TS-only=0.7084` — every value matches the CSV to 4 decimals.
The primary-source verification pass (direct arXiv:2602.11346 read, `pdftotext -f 12 -l 12` for
Table 1) is a genuine strength — it caught and corrected a real prior overstatement (D&L's own
Table-1 ranking was previously mischaracterized as "lowest of 6" when it's actually 5th of 10) and
flagged a challenge-claim-text misattribution (the paper's "22%" is a 3-baseline average, not the
specific pairwise gap the challenge's Claim 6 text implies) — the reproduction did not just accept
the challenge's paraphrase. **Finding**: the missing README means this folder's REFUTED verdicts
are not independently checkable by anyone reading only the folder (as opposed to the 556-line log)
— a structural gap even though the underlying work is unusually rigorous. **Confidence: high** on
the numbers that were checked; the REFUTED conclusion itself reads as well-earned given four
adversarial rounds explicitly aimed at rescuing it, none of which succeeded.

## gluon-lmo-optimizers-IelAHU5MVz

**Clean.** `PAPER_BRIEFING.md` shows genuine PDF engagement, not paraphrase: it cross-checked both
arXiv v1 and the OpenReview-submitted PDF independently, caught that the challenge's Claim 3 cites
"Theorem 1" when the actual theorem is 4.3 (both numbering schemes), and confirmed Claim 6
("zeroth-order eNTK approximation") is a genuine claim-extraction/misattribution error — text search
found zero hits for "eNTK"/"NTK"/"tangent kernel" in either PDF version. This is exactly the kind of
concrete, falsifiable check Tier-2 item 5 asks for (briefing reads like it came from the PDF, not a
claims-list paraphrase). Numeric spot-checks: Claim 2 slope -1.199970 (file) vs "-1.20" (README);
Claim 3 slope -0.346234 vs "-0.346"; Claim 1's "520/520 pass, worst diff 1.8e-15" reproduced exactly
from `claim1_special_cases.csv` (520 rows, 520 passes, max diff 1.7764e-15); Claim 5's
`L^0` range "0.13-14.7" and head `L^1` "7.72" both appear verbatim in `RESULTS_claim5.md`. No gate/
mask-granularity or metric-mismatch bugs found. **Confidence: high** — every checked number traced
exactly.

## deep-flow-networks-Z7rhDaBvBo

Numbers check out exactly (Claim 1's 4-row error table and Claim 2's DFN/MLP MSE-and-time figures
both match `claim1_fig_raw.csv` / `claim2_fig_raw.csv` to full precision). One **overstatement-
adjacent finding**: Claim 2's "4/4 verified" blends two different evidence types without flagging
the distinction in the top-level README. `claim2_fig_raw.csv`'s `experiment` column labels the
Quad-n=16/RA/MDVSP rows "(official artifacts)" — i.e., these come from the *authors' own committed*
`main_text/outputs/*/*_summary.csv` files (confirmed via `make_figs_dfn.py` lines 28-30), re-hashed
for reproducibility (the "45/45 hash-match" claim), not independently retrained. Only the headline
DFN-vs-MLP numbers quoted in the README's Claim 2 table (`QuadSmall`, "FRESH retrain") are a truly
independent from-scratch run. This is disclosed correctly inside the folder's own README
("Independently, 45/45 of the repo's committed artifacts hash-match on re-run") but the top-level
index's bare "4/4 verified" doesn't carry that nuance forward — a reader relying only on the
top-level table could mistake "hash-matches the authors' own bundled results" for "we independently
reran and got the paper's numbers." Also worth noting: `run_quad_small.py`'s docstring shows this
"reduced scale" run actually matches the paper's own declared "Small" config (n=8, K=1000) except
for a shorter Gurobi time cap (600s vs 3600s) — so this is not a scale-down below what the paper
itself reports, just a license-driven timeout change; LSET's 3/3 failure at that cap is disclosed
honestly, not hidden. **Confidence: medium-high** — numbers traced cleanly; the artifact-provenance
distinction is a real but minor clarity gap, not a fabricated result.

## gaussian-mechanism-82Wosp2Iu1

Highest claim-count folder (score "11/12," presumably points-based: 6 claims x up to 2 pts each,
per the top-level README's judge-scoring footnote) but **no `BUGFIX_LOG.md`, no self-audit language,
and no canonical per-claim verdict labels anywhere in the folder** (Tier-1 re-run: WARN, 4 warnings
including `self_audit_log` and `verdict_vocabulary`). Nowhere in the folder or repo is it stated
*which* claim cost the missing point — an un-auditable score. Numeric spot-checks of what *is*
present: Claim 5's Table-2 δ⋆ values match `claim5_table2.csv` to the stated ~6 decimals; Claim 4's
Fig-2 gains (15.6/8.7/4.6% vs paper's 15.5/6.2/2.5%, our p⋆ 1.37/1.58/1.63 vs paper 1.37/1.32/1.92)
match `claim4_fig2.csv` exactly — and this is itself the likely source of the missing point, since
the T=5/T=10 magnitudes deviate substantially from the paper (8.7 vs 6.2%, ~40% relative gap) even
though the direction (shrinking gain) matches; this looks like a legitimate TOY/partial call, just
never labeled as such in the README prose.

**Concrete finding**: Claim 1's "asymptotic optimality... an out-of-regime control shows persistent
finite-T improvements" is not clearly supported by `claim1_asymptotic.csv` at the T range actually
tested (T=2..256). The "inside-regime" margin (which the paper predicts should vanish) shrinks from
-1.0 (T=2) to -0.0365 (T=256); the "outside-regime control" margin (claimed to persist, i.e. not
vanish) shrinks from -0.01374 to -0.000222 over the same range — a >60x reduction. Checking the
decay rate over the last doubling (T=128→256) in both regimes: inside ratio = 0.503 (halves),
outside ratio = 0.507 (also halves) — the two regimes are decaying at statistically indistinguishable
rates at the T values actually tested, which undercuts the "vanishing vs. persistent" qualitative
distinction the prose draws. This may simply mean T=256 is still too small to see the outside
regime's asymptote (a legitimate toy-scale caveat), but the README doesn't flag that ambiguity — it
states the "persistent" framing as settled fact. This is exactly the sort of thing Step 4's
self-audit is supposed to catch, and this folder shows no evidence that self-audit happened.
**Confidence: medium** — the numeric CSV values are exactly traced; the *interpretation* concern is
my own re-derivation from those numbers, not a discovered internal inconsistency, so treat it as a
flag for a closer look rather than a proven refutation.

## submodular-dynamic-non-monotone-tBS3uBG6Pv

Honest and unusually self-aware about its own low score: README explains the earlier 1/6 was a
logbook-page-truncation artifact (one 287KB claim page starved the judge's token budget for the
other two), not missing work, and that trimming pages recovered 3/6 with all three claims now
independently assessed (still capped at "toy" since the paper released no code). Numeric checks:
Claim 1's "minimum expected ratio ≈0.4956" = exact global min of `exp_min_ratio` across all 96 rows
of `claim1_A1_expect.csv` (0.4956337...) — and is clearly labeled as a minimum in the prose, good
practice. Claim 3's "≈0.5964" is also the exact global minimum of `claim3_A2.csv`'s 96
`exp_min_ratio` values (a single covcost/k=3/seed=1 run) — **but the prose says "the improved
variant reaches an expected ratio ≈0.5964," with no "minimum" qualifier**, unlike Claim 1's parallel
treatment. The dataset's actual mean `exp_min_ratio` is 0.8554 — 43% higher than the quoted number.
This isn't a fabricated number (it traces exactly to real data) and doesn't flip the qualitative
verdict (0.5964 minimum is still above Claim 1's 0.4956 minimum, so "above the base variant" holds
either way, and if anything A2's true typical performance of ~0.86 is understated, not overstated) —
but it's an inconsistent-disclosure pattern worth fixing: Claim 1 clearly flags "minimum ... over
the sweep," Claim 3 doesn't, even though both numbers are computed the same way. Claim 1's
supermodular-control "≈0.115" is only approximately traceable — the closest matching raw aggregate
(`mean(min_ratio)` across the 24 control rows) is 0.1123, a ~2.4% relative gap from "0.115"; plausible
under "≈" and rounding, but not an exact match like the other numbers checked here. **Confidence:
medium-high** — three of four checked numbers traced exactly; one (supermod control) only
approximately; the Claim-3 minimum-vs-mean labeling gap is concrete and reproducible.

## Overall summary

6 folders reviewed in depth (BiMU, divide-and-learn, gluon-lmo, deep-flow-networks,
gaussian-mechanism, submodular), plus live Tier-1 re-runs on 5 more (causalprofiler,
concentration-bounds, spherical-harmonics, fake-forgetting, deep-flow-networks) to check a
systemic pattern. No fabricated numbers were found anywhere — every headline figure I attempted to
trace landed in a raw CSV/`.npy` file, usually to 4+ decimal places. 2 of 6 folders (gluon-lmo, BiMU)
came back fully clean. 4 of 6 had a real, concrete, non-fabrication finding: a missing per-claim
verdict trail (divide-and-learn), an unlabeled evidence-provenance blend (deep-flow-networks), an
unlabeled/unaudited possible-toy-call plus an unsupported "persistent vs. vanishing" qualitative
claim (gaussian-mechanism), and an inconsistent minimum-vs-mean disclosure (submodular).

**Systemic pattern, not isolated mistakes**: live Tier-1 re-runs show 7 of the 10 audited-here
folders (all except BiMU, gluon-lmo, and — partially — divide-and-learn) WARN on both
`self_audit_log` and `verdict_vocabulary` — they predate the `BUGFIX_LOG.md` + canonical-4-term-
vocabulary convention that the two prior Step-8 audits (`nPzckCXmHE`, `WB0hLRRlcj`) and this one's
BiMU/gluon-lmo checks show working well where it's actually used. Every finding in this audit that
wasn't a pure labeling nit (gaussian-mechanism's Claim 1, submodular's Claim 3) came from a folder
in this older cohort with no self-audit trail — consistent with the hypothesis that the self-audit
step, when actually run, catches exactly this class of issue (as `nPzckCXmHE`'s and BiMU's own
histories already show), and folders that never got it are the ones where an independent reader
still finds something. **Recommendation**: retrofitting a lightweight self-audit pass (even without
a full BUGFIX_LOG rewrite) to the 7 pre-convention folders, specifically rereading each headline
number's exact framing against verdict_checklist.md item 1 (min vs. mean, toy vs. verified), would
likely be higher-value than further per-number spot-checking — the numbers themselves are reliably
real; it's the *labeling and disclosure* around them that lags in the older cohort.
