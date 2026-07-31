# Repro-harness design notes

Saved conversation turn that led to the `repro-harness` skill (`.agents/skills/repro-harness/`).
Kept verbatim as the design rationale — if the skill's steps look arbitrary later, this is why
each one is there.

## User prompt

> Based on the completed projects in the github, can you summarise key steps/prompts shared
> acrosse projects ? My goal is create an efficient harness that could include good prompts,
> relevant skills and hooks

## Assistant response

Looking across all 10 completed reproductions, a clear pipeline repeats every time, and several
pieces of a harness already exist in this repo (they're just not packaged as a skill yet). Here's
the pattern:

### The recurring pipeline

1. **Selection** — `top100_candidates.md`/`top100_verified.csv` rank unclaimed papers by
   `score = claims x ease`, cross-checked against the live leaderboard so you don't duplicate
   someone else's claim (the `reference-challenge-leaderboard-api` memory documents that API).
2. **Briefing** — before touching code, write a `PAPER_BRIEFING.md` (see
   `gluon-lmo-optimizers-IelAHU5MVz/`) that fixes: repo conventions, exact math transcribed from
   the paper, and — critically — the **verdict vocabulary** used everywhere: `VERIFIED` /
   `TOY-VERIFIED` / `REFUTED` / `BLOCKED`, never rounding a toy-scale pass up to full verification.
3. **Smoketest before scale** — every briefing repeats the same rule near-verbatim: run a
   tiny/fast version first, check shapes/NaNs/sign errors, only then scale up. This shows up in
   `repro_toy.py`'s sanity-check config, the BiMU 1-task smoke run, etc.
4. **Self-audit / bugfix loop** — `divide-and-learn-TK82ECnJzD/BUGFIX_LOG.md` is the sharpest
   example: a dedicated code-review pass (sometimes a subagent) explaining *why* a result doesn't
   replicate, finding real implementation bugs, logged round-by-round with before/after numbers.
   The BiMU claim-2/3 re-audit did the same thing informally.
5. **Honest verdict writing** — the constant discipline across every README/RESULTS file: state
   the claim verbatim, state what was actually run and at what scale, and explicitly separate
   "direction matches" from "magnitude matches." Blocked claims (auth-gated datasets) are reported
   as blocked, never faked or silently skipped.
6. **Logbook + poster** — `scaffold_icml_logbook.py` / `validate_icml_logbook.py` at the repo root
   already codify the canonical Trackio structure (pinned executive summary, one page per claim,
   conclusion), and `.agents/skills/trackio/logbook.md` + `posterly/` are full skills for that and
   for the poster.
7. **GitHub mirror** — a folder per paper, `<slug>-<orid>/`, patches/configs/logs/results only (no
   vendored third-party code), added to the top-level README Index table, direct-commit not PR
   (per the monorepo-conventions memory).

### What's already reusable as harness pieces
- **Skills**: `.agents/skills/trackio/` (logging/logbook), `.agents/skills/hf-cli/`, `posterly/` —
  all drop-in.
- **Scripts**: `scaffold_icml_logbook.py`, `validate_icml_logbook.py` — the closest thing to a
  "hook" already, just not wired to fire automatically.
- **Selection tooling**: the top100 ranking + leaderboard-cross-check logic, currently ad hoc, is
  a good candidate for a small script.

### What's missing (the gap a harness would fill)
- No `SKILL.md` packaging the pipeline itself (steps 2-5 above) the way `posterly` packages
  poster-building — right now it's convention + memory, not an invokable skill.
- No hook enforcing the smoketest-before-scale rule or the verdict vocabulary (e.g. a pre-flight
  check that a RESULTS.md doesn't say "verified" without a "toy-scale/full-scale" qualifier next
  to it).
- `BUGFIX_LOG.md`/`REPRO_LOG.md`-style handoff files are reinvented per-project rather than
  templated.

## Follow-up requirement added by the user

The user asked for the summary to be saved (this file) and for it to become a `repro-harness`
skill, with one addition called out as important: **when the OpenReview paper itself can't be
accessed, the skill must stop and ask the user to download it manually rather than proceeding
without it.** Rationale given: the OpenReview version is the source of the claims being
reproduced, and should be diffed against the arXiv version when one exists (camera-ready OpenReview
revisions can add/change ablations or numbers relative to the arXiv preprint — a claim-extraction
paraphrase already produced one real discrepancy, documented in the BiMU reproduction's Claim 2
page: "task-boundary-dependent BayesBiNN baseline" turned out not to match the paper's own Table 1
setup). This became Step 0 (hard gate) of the skill.
