---
name: repro-harness
description: "Reproduce an ICML/NeurIPS/ICLR paper's claims end-to-end for the ICML-2026-agent-repro challenge (or any independent paper-reproduction task): acquire and verify the source paper, write a briefing, smoketest before scaling, run an honest self-audit/bugfix loop, write per-claim verdicts, publish a Trackio logbook, and mirror the bundle into the icml2026-reproductions GitHub monorepo. Use whenever asked to reproduce, verify, or audit a paper's claims/results."
allowed-tools: Bash(*), Read, Write, Edit, Grep, Glob, WebFetch, WebSearch, AskUserQuestion
---

# repro-harness — paper reproduction pipeline

Distilled from 10 completed reproductions in this repo (see `harness-testing/design-notes.md`
for the full rationale). Each step below is a stage that showed up independently,
convention-only, across those 10 projects — this skill just makes the convention explicit and
enforced instead of re-derived every time.

Composes with existing skills rather than replacing them: `.agents/skills/trackio/` for the
logbook, `.agents/skills/hf-cli/` for Hub auth/downloads, `posterly/` for the executive-summary
poster. This skill owns the steps *around* those — selection, paper acquisition, briefing,
running, auditing, and the GitHub mirror.

## Step 0 — Acquire the paper (hard gate — run as a subagent, do not skip)

Every claim you verify or refute must trace back to the paper's actual text, not to a
pre-extracted `claims.json` string, an abstract, or your own memory of the paper.

**Run this step in a subagent** (`Agent` tool, `subagent_type: general-purpose`, foreground —
Step 2 needs its result before it can start). PDF text, WebFetch noise, and retry attempts belong
in the subagent's context, not the main agent's. Give the subagent the paper's OpenReview id (and
title) and this exact order of operations:

1. **Try the OpenReview submission PDF first** (`https://openreview.net/forum?id=<orid>`,
   PDF at `https://openreview.net/pdf?id=<orid>`). This is the primary source — it is the exact
   version the challenge's claims were extracted against, and may be a camera-ready revision that
   postdates the arXiv listing (added ablations, changed numbers, reworded claims).
2. **If OpenReview is blocked** (auth wall, `WebFetch`/`curl` blocked, 403/bot-challenge, forum
   page won't render, rate-limited, whatever), **fall back to arXiv**: use the arXiv id if known,
   otherwise search by title.
3. **If the arXiv paper also can't be read completely** — importantly, check that the
   **methodology appendix, if the paper has one, is present and readable**, not just the main
   body — **exit the process**. Stop immediately: do not fall back to a `claims.json` string or
   an abstract and proceed as if that's equivalent, and do not pause to ask the user mid-task.
   Return a clear failure report (which source failed, why, what was/wasn't readable) to the main
   agent.
4. **On success**, save the acquired PDF(s) into the working reproduction folder and, if both
   OpenReview and arXiv versions were fetched, diff them for anything about to be cited as a
   claim (same numbers, same table entries, same wording) — note any divergence in the briefing
   and treat OpenReview as authoritative.

**The main agent must treat a Step 0 failure report as a hard stop for this paper** — report the
failure to the user directly and do not continue to Step 1+. (This supersedes the older
stop-and-`AskUserQuestion` behavior: acquisition failure now ends the attempt rather than pausing
for a manually-supplied PDF.)

Full detail: `paper_acquisition.md`.

## Invoking this skill

You don't need to paste claim text from the challenge site into the prompt — a paper title or
OpenReview id is enough. Pull `claims.json`/`claims_anchored.json` yourself (see memory
`reference-challenge-leaderboard-api` for the download command) for the pre-extracted claim list;
Step 0 still requires reading the actual PDF as the authoritative source regardless, so
hand-pasted claim text would just get superseded anyway.

## Step 1 — Select (only if the paper isn't already assigned)

Rank unclaimed candidates by `score = claims x ease`, cross-checked against the live
`icml2026-repro`-tagged Space leaderboard so you don't duplicate an already-claimed paper (see
memory `reference-challenge-leaderboard-api` for the query, or `top100_candidates.md` /
`top100_verified.csv` at the repo root for the existing ranked list and how it was built).

## Step 2 — Write the briefing

Before touching code, write `PAPER_BRIEFING.md` in the paper's working folder. Use
`briefing_template.md` as the starting structure. It must fix, up front:
- Paper identity (title, OpenReview id, arXiv id if any, local PDF path from Step 0).
- Repo conventions this reproduction will follow (PEP-723 scripts / whatever the target repo
  uses, naming pattern, where results get written).
- The paper's actual math/setup for the claims in scope, transcribed from the PDF — not
  paraphrased from a challenge-provided claim string.
- The verdict vocabulary (below) restated so it's visible mid-task.
- The smoketest-before-scale rule (below) restated for the same reason.

## Step 3 — Smoketest before scaling

Every prior briefing repeats this near-verbatim — keep doing it: before running anything longer
than ~30s-1min, run a tiny/fast version (few iterations, tiny dimensions, 1 task instead of N)
and check for shape errors, NaNs, sane orders of magnitude, sign errors. Only scale up once that's
clean. Never burn compute on a broken script at full scale.

## Step 4 — Run, then self-audit

**Before launching anything that will run longer than a session comfortably lasts** (this project
runs on Claude Pro, which has session/usage limits — a multi-hour toy run *will* outlive a
session): launch it detached (`nohup ... &`), write a `REPRO_LOG.md`-style recovery file *before*
starting, make the script resumable if at all possible, and save a memory pointer to the recovery
file immediately. See `session_survival.md` for the full pattern — skipping this is how a session
limit turns into lost work instead of a 30-second resume.

Run the toy- and (if feasible) full-scale experiments. Then, before writing verdicts, run a
dedicated self-audit pass — reread the implementation against the paper's actual equations,
looking specifically for: sign errors, gates/masks applied at the wrong granularity (per-parameter
vs. global — the exact bug class the BiMU `sum_grads` correction caught), feedback loops that were
computed but never wired into what they're supposed to affect (the exact bug class the D&L
`BUGFIX_LOG.md` round 1 caught), and metrics that measure something subtly different from what the
claim states (the exact bug class the BiMU OOD-AUC "repeated probe vs. per-task-boundary
trajectory" correction caught). Log findings and before/after numbers in a running
`BUGFIX_LOG.md` (see `divide-and-learn-TK82ECnJzD/BUGFIX_LOG.md` for the format) — don't silently
fix and move on; the log is what lets a later session or reviewer trust the final numbers weren't
cherry-picked.

If a long-running job needs to survive a session boundary, write a `REPRO_LOG.md` handoff (see
`fake-forgetting-uncertainty-rjmVJaBpkm/REPRO_LOG.md`) with a "read this first on a cold start"
section: how to check what's already done, how to resume, what the next actions are.

## Step 5 — Write honest verdicts

One verdict per claim, using exactly this vocabulary (see `verdict_checklist.md` for the full
self-check):
- **VERIFIED** — matches the paper's claim at the paper's own scale.
- **TOY-VERIFIED** — matches directionally/qualitatively at a reduced scale; explicitly not
  claiming to hit the paper's exact numbers.
- **REFUTED** — ran it at a fair scale and it contradicts the claim; state the discrepancy.
- **BLOCKED** — not attempted, and say exactly why (auth-gated dataset, missing compute, etc.) —
  never faked, never silently skipped.

Never round a TOY-VERIFIED up to VERIFIED. Never let a BLOCKED claim go unmentioned in the final
summary. State what scale you ran at next to every verdict.

## Step 6 — Logbook + poster

Use the `hugging-face-trackio` skill for the logbook (`scaffold_icml_logbook.py` /
`validate_icml_logbook.py` at the repo root scaffold and validate the canonical ICML structure:
pinned executive summary, one page per claim, conclusion) and `posterly` for the executive-summary
poster. Publish, then validate before calling it done.

## Step 7 — Mirror into the GitHub monorepo

Per memory `reference-icml-monorepo-conventions`: create `<short-slug>-<orid>/` at the repo root
of the target monorepo (`nmaher2022/icml2026-reproductions` for this project), containing
patches/configs/logs/results/README but **not** vendored third-party code (link + clone
instructions instead). Add a row to the top-level README's Index table. Direct-commit to `main` —
no PR workflow in this repo. **Push is pre-authorized for this repo** (user confirmed 2026-08-01):
commit and `git push origin main` without pausing to ask — this repo is a solo-owned public
mirror with no collaborators to disrupt, and the user has explicitly opted out of a per-run
confirmation gate here. Still follow the standing git safety rules otherwise (no force-push, no
`--no-verify`, review the diff for secrets before pushing).

**Before cloning anything**, run `git remote -v` in the current working directory. If it already
points at the target monorepo, work in place (`scripts/scaffold_reproduction.py add` with no
`--repo-path`, since omitting it auto-detects the repo root) instead of cloning a fresh copy
elsewhere. A redundant clone caused a real divergence during the `nPzckCXmHE` reproduction — see
`harness-testing/audits/nPzckCXmHE.md` — where the working repo ended up behind the just-pushed
commit with an orphaned, uncommitted duplicate folder that collided on the next `git pull`.

Use `scripts/scaffold_reproduction.py` for the mechanical part of this step:

```bash
# One-time, only if the target repo doesn't exist yet or has no README Index table:
uv run scripts/scaffold_reproduction.py init --repo-path <path> --title "..." --author "..."

# Every reproduction:
uv run scripts/scaffold_reproduction.py add --repo-path <path> \
  --slug <short-slug> --orid <orid> --title "<paper title>" [--arxiv <id>] \
  --field "Paper=<paper title>" \
  --field 'OpenReview=[`<orid>`](https://openreview.net/forum?id=<orid>)' \
  --field "Claims reproduced=<n>" \
  --field "Verdict=<summary>" \
  --field 'Trackio Logbook=[HF Space](<url>)'
```

`--repo-path` is optional for `add` — omitted, it walks up from the current directory looking for
a README.md with an `## Index` table (the same repo-root search trick `.trackio/` uses), so the
script isn't hardcoded to any one repo or path. `--field` keys must match whatever columns the
target README's Index table actually has (it reads the header, not a fixed schema) — a
Folder-shaped column is auto-filled from `--slug`/`--orid` if you don't pass one. The script only
`git add`s what it writes; it never commits or pushes — review the diff and run the printed
commit command yourself (or have the agent run it only after you've confirmed).

It creates the folder skeleton (`patches/`, `configurations/`, `logs/`, `results/`) and a
`README.md` stub with TODOs for the verdict table, but does **not** generate the narrative
content — that's Step 5's honest verdict writing, done by hand against the actual results.

## Step 8 — Audit the harness itself

This step is about the pipeline's own reliability, not any one paper's claims. After finishing a
reproduction (or periodically across several), run
`harness-testing/audit_harness.py <paper-folder>` — it checks structural evidence that Steps 0-7
were actually followed (paper acquisition source, smoketest-before-scale evidence, a self-audit
log, verdict-vocabulary compliance, Index-row formatting) and writes a JSON gate report. Follow it
with the qualitative review pass in `harness-testing/AUDIT.md` for the judgment calls a script
can't make (did the self-audit actually catch real bugs, do the verdicts hold up against the
source PDF). Findings that indicate the *skill itself* needs a fix (not just the one paper) go
back into these `SKILL.md`/`*.md` files, not just into that paper's bugfix log.

## Files in this skill
- `paper_acquisition.md` — Step 0 in full, with the concrete AskUserQuestion prompt to use when
  OpenReview access fails, and how to diff an arXiv version against it.
- `briefing_template.md` — starting structure for `PAPER_BRIEFING.md`.
- `verdict_checklist.md` — the verdict vocabulary plus a pre-submission self-check.
- `session_survival.md` — how to avoid losing work when a Claude session limit is hit mid-run, or
  the internet drops mid-run (detached jobs, resumable scripts, write-the-recovery-file-first,
  resumable transfers).

## Harness-testing project (outside this skill folder)
- `harness-testing/design-notes.md` — the conversation that produced this skill, kept as rationale.
- `harness-testing/HANDOFF.md` — build status / open items for the skill and scaffold script.
- `harness-testing/AUDIT.md` — the Step 8 audit methodology in full.
- `harness-testing/audit_harness.py` — the automated Step 8 gate script.
- `harness-testing/audits/` — one JSON gate report per audited paper folder.
- `scripts/scaffold_reproduction.py` — Step 7 automation: `init` bootstraps a monorepo skeleton,
  `add` creates a paper folder + inserts its Index-table row. Repo-path auto-detected, column
  schema read from the target README rather than hardcoded. Never commits or pushes.
