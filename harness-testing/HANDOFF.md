# Handoff — repro-harness skill build

**Deliverable:** `.agents/skills/repro-harness/` (skill packaging the 8-step reproduction
pipeline) + `.agents/skills/repro-harness/scripts/scaffold_reproduction.py` (Step 7 automation).
**Working dir:** `/home/rec1/Desktop/AI_Safety/ICML_reproduce` (this IS the checked-out
`nmaher2022/icml2026-reproductions` monorepo — there is no separate remote-only copy).

This file is the single source of truth for continuing this specific piece of work (building the
harness) after a context reset. It is about the *harness build itself*, not about any one paper
reproduction — for that, see the per-paper `REPRO_LOG.md`/`BUGFIX_LOG.md` pattern the harness
itself now documents.

---

## What's done

- `SKILL.md` — the 8-step pipeline: Step 0 paper acquisition (hard gate) → Step 1 selection →
  Step 2 briefing → Step 3 smoketest-before-scale → Step 4 run + self-audit/bugfix loop → Step 5
  honest per-claim verdicts → Step 6 logbook + poster → Step 7 GitHub mirror → Step 8 audit the
  harness itself.
- `paper_acquisition.md` — Step 0 in full: OpenReview PDF first, diff against arXiv if one exists.
  **Updated 2026-07-31** (commit `3d5f2b1`): acquisition now runs as a **foreground subagent**
  (context isolation — PDF text/retries stay out of the main agent's context) and, if OpenReview is
  unreachable or the arXiv fallback is incomplete (esp. a missing/garbled methodology appendix),
  Step 0 **hard-exits and reports failure** rather than pausing with `AskUserQuestion` for a
  manually-supplied PDF — that stop-and-ask branch was removed. (The original stop-and-ask design
  was the user's specific addition, per `design-notes.md`'s BiMU Claim-2 precedent; the 2026-07-31
  change tightens the failure path further after real use showed the ask-branch just delayed an
  inevitable "pick a different paper" decision.) **Validated in production** on `nPzckCXmHE` — see
  "First live end-to-end run" below.
- `briefing_template.md` — starting structure for a per-paper `PAPER_BRIEFING.md`.
- `verdict_checklist.md` — VERIFIED / TOY-VERIFIED / REFUTED / BLOCKED vocabulary + a
  pre-submission self-check, including two named bug classes worth re-checking (wrong-granularity
  gates; metrics measuring something subtly different from the claim).
- `scripts/scaffold_reproduction.py` — `init` (bootstrap a monorepo skeleton: README + Index
  table + LICENSE) and `add` (create `<slug>-<orid>/` + insert its Index row) subcommands.
  Repo-path auto-detected by walking up from cwd for a README with an `## Index` table; column
  schema read from that table's own header, not hardcoded. Never runs `git commit`/`git push` —
  only stages and prints the command. **Tested**: dry-run `add` against the real repo's README
  (correct row assembly, correct auto-detected repo root); full `init` → `add` cycle against a
  scratch tmp directory (skeleton creation, folder+subdir creation, row insertion, staging all
  verified by inspecting the resulting files and `git status`).
- `design-notes.md` (this folder) — the conversation turn that led to the skill, saved verbatim
  as design rationale.
- Memory updated: `feedback-openreview-paper-required.md` (the Step 0 hard-gate rule),
  `reference-repro-harness-skill.md` (pointer to the skill + its files), `MEMORY.md` index rows
  added for both.

## First live end-to-end run: `spectral-cit-nPzckCXmHE` (2026-07-31/08-01)

The skill was pressure-tested Steps 0–8, in full, on "Toward Scalable and Valid Conditional
Independence Testing with Spectral Representations" (OpenReview `nPzckCXmHE`) — the first real use
of the pipeline as written, not just the pattern-matched precedent from 10 prior reproductions.
Landed as commits `7acbac6` (the reproduction + monorepo mirror), `91f1caa` (Step 8 harness fixes),
`1e8b20c` (code-level re-audit addendum). Full details: `spectral-cit-nPzckCXmHE/VERDICTS.md`,
`audits/nPzckCXmHE.json`, `audits/nPzckCXmHE.md`.

**Step 0 validated for real**: OpenReview was bot-wall-blocked; the acquisition subagent fell
through to arXiv (v2, all 3 appendices readable) automatically and reported success without
pausing — the first genuine exercise of the 2026-07-31 subagent/hard-exit change, not a no-op.

**Real process bug found (Step 7) and fixed**: the agent cloned a *fresh* copy of the monorepo
into a scratchpad directory to do the mirroring step, not realizing the working directory already
*is* that monorepo (this doc's own header line 5–6 says so, but `SKILL.md` itself didn't tell the
agent to check). Consequence: after the scratchpad clone was pushed, this working repo ended up
behind `origin/main` with an orphaned, untracked, uncommitted duplicate folder that collided on the
next `git pull --ff-only`. No data was lost (verified byte-identical before reconciling), but it's
exactly the shape of conflict that could lose work with a less careful merge. `SKILL.md` Step 7 now
says to check `git remote -v` before cloning anything. Full writeup: `audits/nPzckCXmHE.md`.

**Two Step 8 gate bugs found and fixed in `audit_harness.py`**, confirmed by re-running against
`divide-and-learn-TK82ECnJzD` and `gluon-lmo-optimizers-IelAHU5MVz` to rule out paper-specific
false positives:
- `raw_results_present` only recognized a `results/`-named subdirectory, false-warning on the
  actual convention used by nearly every reproduction in this repo (flat CSV/log files at the
  folder root). Fixed to also accept non-empty `*.csv`/`*.log`/`*.json` at the folder root.
- `paper_source`'s arXiv-cross-check check matched the bare substring "cross-check" with no
  negation awareness — a sentence *denying* a cross-check happened (`nPzckCXmHE`'s own
  `PAPER_BRIEFING.md` honestly says "no OpenReview/arXiv cross-check possible") scored as if one had
  occurred. The PASS was accidentally correct here but the check is unreliable as written. **Not
  fixed yet** — flagged as an open item below.

**Open design question surfaced, not resolved**: Claim 5's verdict used "INCONCLUSIVE," a 5th term
outside `verdict_checklist.md`'s canonical VERIFIED/TOY-VERIFIED/REFUTED/BLOCKED vocabulary —
legitimately, since the ablation's result was genuinely confounded (`nn.Identity` collapses an MLP
to a linear map, so it can't cleanly isolate Assumption 4.1's sub-Gaussianity mechanism), and none
of the 4 canonical terms cleanly cover "ran fine, but the result is uninterpretable by design."
Tripped `verdict_vocabulary`'s WARN correctly (4/5 rows) — the gate did its job; the vocabulary
itself has a real gap. See open items below.

## Not done / open items

- **`paper_source`'s cross-check sub-check is negation-blind** (see above) — replace the bare
  keyword match with a check for an explicit positive-confirmation pattern. Lower priority: it's a
  soft gate and hasn't yet produced a wrong overall verdict, just an unreliable one.
- **`verdict_checklist.md`'s vocabulary has no term for a confounded/uninterpretable result.**
  `nPzckCXmHE` invented "INCONCLUSIVE" ad hoc for Claim 5. Decide explicitly: add a 5th canonical
  term, or give guidance on mapping this case onto one of the existing 4 (not BLOCKED — that means
  access/resource-blocked, not "ran fine but confounded by design").
- **No `--commit`/`--push` automation, by design** — matches the git-safety posture used
  throughout this project; stays manual/agent-confirmed every time. Two commits (`91f1caa`,
  `1e8b20c`) are currently ahead of `origin/main`, pending an explicit push confirmation.
- **No hook/enforcement layer.** Step 8's `audit_harness.py` is a *post-hoc* check, not a
  pre-flight one — nothing stops an agent from skipping the smoketest or writing "VERIFIED" without
  the self-check *before* Step 8 gets run. If that's wanted later, the natural next piece is a
  pre-flight version of the same gates, run before Step 6/7 rather than after.
- **`init`'s LICENSE/README bootstrap has only been exercised on a throwaway repo** — never
  actually needed here since this monorepo is already bootstrapped (now 11 papers, README with a
  populated Index table already exists).
- **Tier 2 audit item 1 (spot-check the briefing's transcribed math against the source PDF pages)
  still hasn't been done for `nPzckCXmHE`** — the code-level re-audit (loss functions, whitening,
  test statistic vs. `PAPER_BRIEFING.md`'s transcribed equations) was done and found no
  discrepancies, but nobody has re-opened the actual arXiv PDF to check the briefing's transcription
  against it independently.

## Recovery / cold-start steps

1. Read auto-memory (loads automatically) → `reference-repro-harness-skill` and
   `feedback-openreview-paper-required` point back here and to the skill files.
2. Read this file's "Not done / open items" above — that's the actual TODO list.
3. `git status -sb` in this directory to see what's still untracked vs. what (if anything) got
   committed since this doc was written.
4. Read `.agents/skills/repro-harness/SKILL.md` for the pipeline itself before using or modifying
   it further.

## Next actions

1. Push the 2 commits currently ahead of `origin/main` (`91f1caa`, `1e8b20c`) once the user
   confirms — everything up through the `nPzckCXmHE` audit addendum is committed locally.
2. Resolve the vocabulary open item (add a 5th canonical term, or explicit guidance mapping
   confounded results onto the existing 4) in `verdict_checklist.md` before the next reproduction
   hits the same ambiguity Claim 5 did.
3. Fix `paper_source`'s negation-blind cross-check sub-check in `audit_harness.py` (lower
   priority — see "Not done" above).
4. Per `AUDIT.md`'s own cadence guidance, don't re-run Step 8 on every single reproduction going
   forward — next trigger is either the next 2-3 new reproductions, or any time a result gets
   challenged/corrected.
5. Still open from before: `scaffold_reproduction.py add` was run for real (non-dry-run) for
   `nPzckCXmHE`, but only via the now-fixed redundant-clone path — confirm the *in-place*
   invocation (no `--repo-path`, run directly in this working directory per the new `SKILL.md`
   guidance) on the next paper, since `nPzckCXmHE` never actually tested that path.
