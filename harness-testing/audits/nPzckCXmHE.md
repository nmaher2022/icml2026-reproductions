# Step 8 audit — spectral-cit-nPzckCXmHE (Toward Scalable and Valid CIT with Spectral Representations)

Tier 1: see `nPzckCXmHE.json` — overall **WARN**, 0 hard failures, 2 soft warnings
(`smoketest_evidence`, `raw_results_present`). Tier 2 qualitative review below.

## Process finding: Step 7 was run against a redundant clone, not in place (real bug, fixed)

The working repo at `/home/rec1/Desktop/AI_Safety/ICML_reproduce` **is itself** a clone of the
target monorepo (`git remote -v` → `origin = nmaher2022/icml2026-reproductions`, confirmed by
`git log`/`git fetch` matching). Step 7 was nonetheless executed by `gh repo clone`-ing a *second*,
fresh copy into a scratchpad directory and mirroring/committing/pushing from there.

Consequence: after that push, this working repo ended up 1 commit behind `origin/main`, with an
orphaned untracked duplicate `spectral-cit-nPzckCXmHE/` folder here (missing `README.md`, still
containing the un-gitignored PDF and `__pycache__`, since those had only been excluded by hand
during the scratchpad copy, not by `.gitignore`). `git pull --ff-only` then refused, listing every
file in the duplicate folder as a would-be-overwritten untracked file. Verified byte-identical
(modulo CRLF→LF normalization) before deleting the duplicate and completing the pull — no data was
actually lost, but this is exactly the shape of conflict that could lose work with a less careful
merge.

Root cause: `SKILL.md`'s Step 7 section never tells the agent to check whether the *current*
working directory already is the target monorepo before cloning. `scripts/scaffold_reproduction.py`
already auto-detects this correctly (`--repo-path` omitted walks up from cwd for a README with an
`## Index` table) — the redundant clone bypassed that auto-detection instead of relying on it.

**Fix applied:** none to the skill file yet (see recommendation below — this should go in
`SKILL.md` Step 7, not just this note). The immediate repo-state issue (behind-by-1, duplicate
folder, stray backup dirs) was resolved by hand this session.

**Recommendation:** add one line to `SKILL.md` Step 7: before cloning anything, run `git remote -v`
in the current working directory; if it already points at the target monorepo, run
`scripts/scaffold_reproduction.py add` in place (no `--repo-path`) instead of cloning a second copy.

## Gate finding: `raw_results_present` false-warns on the repo's actual convention

Re-ran Tier 1 against two other completed reproductions to check whether this WARN was
paper-specific:

| Folder | `raw_results_present` |
|---|---|
| `divide-and-learn-TK82ECnJzD` | WARN (despite substantial raw CSVs present) |
| `gluon-lmo-optimizers-IelAHU5MVz` | WARN (same) |
| `active-continual-learning-bimu-SPZd0HVyiS` | PASS |
| `spectral-cit-nPzckCXmHE` (this one) | WARN |

The gate only looks for a directory literally named `results` (or containing "result"). The
established convention actually used by nearly every reproduction in this repo — confirmed
earlier by inspecting `divide-and-learn-TK82ECnJzD` directly — is flat CSV/log files at the folder
root, not a `results/` subfolder; BiMU is the outlier that happens to match the scaffold script's
literal default skeleton. So this gate WARNs on the *normal* case and only PASSes on the
exception. Fixed in `audit_harness.py` this session: the gate now also accepts non-empty
`*.csv`/`*.log`/`*.json` files sitting directly in the folder root as raw-results evidence.

## Gate finding: `paper_source`'s arXiv-cross-check check is negation-blind

`PAPER_BRIEFING.md` for this paper says, honestly: "no OpenReview/arXiv cross-check possible since
[OpenReview was bot-blocked, fetched via arXiv instead]." The gate's cross-check sub-regex matches
on the bare substring "cross-check" appearing anywhere in the folder's markdown, with no
negation-awareness — so this sentence *denying* a cross-check happened was scored as if one *had*
happened, and the gate PASSed. The PASS is accidentally correct here (no cross-check was actually
necessary or possible, and that fact is disclosed honestly), but the check can't tell that sentence
apart from an equally plausible one like "OpenReview inaccessible, arXiv used instead" that
contains none of the trigger words — which would have WARNed for an equally legitimate case. Not
fixed this session (lower priority than the two above — it's a soft gate that isn't currently
producing a wrong verdict, just an unreliable one); flagging for a future pass: replace the
keyword match with a check for an explicit positive-confirmation pattern instead of any mention of
"cross-check."

## Vocabulary finding: Claim 5's "INCONCLUSIVE" is a 5th, uncanonical term

`verdict_checklist.md`'s vocabulary is VERIFIED / TOY-VERIFIED / REFUTED / BLOCKED. Claim 5's
Tanh-vs-Identity ablation came back confounded (Identity collapses the MLP to a linear map, so the
result can't isolate Assumption 4.1's sub-Gaussianity mechanism from loss of nonlinear capacity) —
genuinely neither VERIFIED/TOY-VERIFIED nor REFUTED (the claimed failure mode wasn't observed), and
not BLOCKED either (BLOCKED means access/resource-blocked, not "ran fine but the ablation design
doesn't isolate the variable"). "INCONCLUSIVE" was introduced ad hoc to cover this. This correctly
tripped `verdict_vocabulary`'s WARN (4/5 rows, not 5/5) — the gate did its job. The open question is
a real harness-design gap, not a mistake in this reproduction: does the canonical vocabulary need a
5th term for "confounded/uninterpretable-by-design," or should this class of result map onto one of
the existing 4 with stronger guidance? Recommend resolving explicitly in `verdict_checklist.md`
rather than leaving every reproduction to invent its own ad hoc term.

## Positive finding: Step 0's subagent + hard-exit update worked as intended

This reproduction was also the first real test of the updated Step 0 (paper acquisition run as a
subagent, hard-exit instead of pausing to ask for a manual PDF on failure). OpenReview was in fact
bot-blocked for this paper, and the acquisition subagent fell through to arXiv (v2, all 3
appendices readable) automatically, without interrupting the user. This is a genuine, non-trivial
exercise of the new behavior, not a no-op test — it validates the change in production.

## Positive finding: self-audit caught a real instance of a named bug class

`BUGFIX_LOG.md` entry 1 (`E_m^val` computed as a cross-covariance between the wrong pair of
variables instead of the self-covariances the definition requires) is exactly
`verdict_checklist.md`'s "metric measuring something subtly different from what's claimed" failure
mode — caught during Step 4's self-audit, before verdicts were written, not via a later correction.
Confirms the self-audit step is pulling real weight here, consistent with (but independent from)
the BiMU claim 2/3 precedent that motivated adding this checklist item in the first place.

## Not done in this pass (disclosed gap in the audit itself)

Tier 2 item 1 — spot-checking `PAPER_BRIEFING.md`'s transcribed equations directly against the
arXiv PDF pages — was not performed in this audit. Flagging as an open item rather than silently
skipping it.

## Addendum: code-level re-audit of the two soft (INCONCLUSIVE / partial) results

Follow-up pass, independently re-deriving `scit_lib.py`'s loss functions, whitening, and test
statistic against `PAPER_BRIEFING.md`'s transcribed equations, specifically targeting Claim 5
(INCONCLUSIVE) and Claim 3 (TOY-VERIFIED, partial). **No new bugs found** — both existing
explanations hold up:

- Claim 3's E_val≈1.0: re-derived the `eps=1e-6` clamp-in-`inv_sqrt` mechanism independently from
  first principles (a true eigenvalue ~1e-9 gets under-corrected by a clamp-based `rsqrt`, leaving
  ~1e-3 whitened variance instead of 1) — matches BUGFIX_LOG entry 3 exactly. Confirmed
  `validation_error()` is correctly called on post-whitening (hat) features per the paper's own
  notation, not a location bug.
- Claim 5's collapse: confirmed algebraically that `make_mlp(..., activation=nn.Identity)`
  produces a literal composition of 3 affine layers with no intervening nonlinearity, i.e. a single
  linear map — the "collapses to linear" explanation is not just plausible, it's mechanically
  exact given the code.
- All loss/statistic/whitening code checked term-for-term against the paper's transcribed
  equations (index conventions, `N=(N+Nᵀ)/2`, floor via `int()` truncation, SVD's descending-order
  convention for "leading" singular values) — no transcription errors found.
- One minor, non-bug design note: `claim5_subgaussian_ablation.py` draws both activation arms'
  datasets from one shared `rng` stream sequentially rather than a paired same-data design. Adds
  minor unnecessary variance to the comparison; doesn't affect the INCONCLUSIVE verdict given the
  ~8x effect size. Worth a paired design in any follow-up ablation.
