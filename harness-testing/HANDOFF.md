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
  honest per-claim verdicts → Step 6 logbook + poster → Step 7 GitHub mirror.
- `paper_acquisition.md` — Step 0 in full: OpenReview PDF first, diff against arXiv if one
  exists, and if OpenReview is unreachable, **stop and ask the user via `AskUserQuestion`** rather
  than proceeding on a claims.json paraphrase alone. This was the user's specific addition to the
  pipeline (see `design-notes.md` for the full rationale and the BiMU Claim-2
  precedent that motivated it).
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

## Not done / open items

- **Nothing was committed as of this doc's original writing** — the user has since asked for
  `harness-testing/` and the skill to be committed/pushed (see the "Committed" note added below
  once that lands). When staging, stage only the intended files (`.agents/skills/repro-harness/`,
  `harness-testing/`, the top-level `README.md` pointer edit) — not the unrelated loose files
  already sitting untracked in this working tree, e.g. `data/`, `Conformal-Prediction-Unlearning/`,
  various PDFs/logs from other in-progress work.
- **`scaffold_reproduction.py add` has never been run in real (non-dry-run) mode against the live
  monorepo README** — only against a scratch/tmp repo. The dry-run against the real README parsed
  correctly, but a real write + `git diff` review on the actual `README.md` hasn't happened. Do
  that once before trusting it unattended on the live table (11 rows currently).
- **No `--commit`/`--push` automation, by design** — matches the git-safety posture used
  throughout this project; stays manual/agent-confirmed every time.
- **The skill hasn't been used end-to-end on an actual new paper yet.** Steps 0–6 are written
  from the pattern in 10 already-completed reproductions, but haven't been pressure-tested by
  actually running them as a checklist on an 11th paper. First real use should surface whether any
  step needs adjustment (especially Step 0's AskUserQuestion flow, which has never actually fired).
- **No hook/enforcement layer.** Nothing currently stops an agent from skipping the smoketest, or
  writing "VERIFIED" without running the `verdict_checklist.md` self-check — it's all
  read-and-follow, not machine-checked. If that's wanted later, a lightweight pre-flight grep (e.g.
  flag a RESULTS/README that says "verified" without "toy-scale" nearby, or a folder with no
  `BUGFIX_LOG.md`/audit note) would be the natural next piece.
- **`init`'s LICENSE/README bootstrap has only been exercised on a throwaway repo** — never
  actually needed here since this monorepo is already bootstrapped (10 papers, README with a
  populated Index table already exists).

## Recovery / cold-start steps

1. Read auto-memory (loads automatically) → `reference-repro-harness-skill` and
   `feedback-openreview-paper-required` point back here and to the skill files.
2. Read this file's "Not done / open items" above — that's the actual TODO list.
3. `git status -sb` in this directory to see what's still untracked vs. what (if anything) got
   committed since this doc was written.
4. Read `.agents/skills/repro-harness/SKILL.md` for the pipeline itself before using or modifying
   it further.

## Next actions

1. Decide with the user whether/when to commit the skill + design notes + this handoff doc to
   local git history (note: committing here *is* committing to `icml2026-reproductions`, no
   separate push step needed to "land" it in that repo — pushing to the remote is the only thing
   still gated on explicit confirmation).
2. Do one real (non-dry-run) `scaffold_reproduction.py add` against the live README — ideally as
   part of actually landing the next paper reproduction — and review the diff before committing.
3. Run the full pipeline (Steps 0–7) on the next paper end-to-end as the first live test of the
   skill as written; note anywhere it didn't match reality and fix the skill files, not just the
   one-off situation.
