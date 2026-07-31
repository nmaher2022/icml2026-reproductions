# Auditing the harness

This is an audit of the **pipeline**, not of a paper's claims. A paper's claims get audited by
Step 4/5 of `repro-harness` itself (the self-audit/bugfix loop, the verdict self-check). This
document is one level up: after using the harness on a paper, did the harness's own steps actually
get followed, and did following them produce a trustworthy result? If a gap shows up here, the fix
belongs in the skill files (`SKILL.md`, `paper_acquisition.md`, etc.), not just in that one paper's
folder.

Two tiers, because some questions are checkable by a script and some aren't.

## Tier 1 — Automated structural gates (`audit_harness.py`)

Fast, deterministic, no judgment calls. Run it against a reproduction folder:

```bash
uv run harness-testing/audit_harness.py <folder> [--repo-path .] [--orid <orid>] [--json-out harness-testing/audits/<orid>.json]
```

It checks for **structural evidence** that each step ran, not whether the content is *good* —
that's Tier 2. Gates and what they look for:

| Gate | Checks | Severity |
|---|---|---|
| `paper_source` | A `PAPER_BRIEFING.md` (or folder README) mentions an OpenReview PDF path/URL for this paper's `orid`, and — if an arXiv id is also mentioned anywhere — that a cross-check against it is noted | hard |
| `briefing_exists` | `PAPER_BRIEFING.md` present in the folder | soft |
| `smoketest_evidence` | `logs/` (or equivalent) contains something that looks like a smoketest — a small/short log, or a filename containing `smoke`/`sanity`/`toy`, sitting alongside a full-scale run log | soft |
| `self_audit_log` | A `BUGFIX_LOG.md`, or a "correction"/"re-audit"/"corrected" mention in the README/logbook pages, is present | soft |
| `verdict_vocabulary` | The folder README's verdict section uses the four-term vocabulary (VERIFIED / TOY-VERIFIED / REFUTED / BLOCKED) or a clearly equivalent labeling — flags prose verdicts that don't map cleanly to one of the four | soft |
| `blocked_claims_disclosed` | Every claim labeled blocked/not-attempted has a stated reason (not just "blocked") | hard |
| `index_row_present` | The top-level README's Index table has a row whose folder-link column points at this folder | hard |
| `raw_results_present` | A `results/`-equivalent directory contains actual data files, or (the more common convention in this repo) non-empty `*.csv`/`*.log`/`*.json` files sit directly at the folder root | soft |
| `no_vendored_code` | The folder doesn't contain a nested `.git` or an obviously-vendored third-party tree | hard |

`hard` gate failures mean the audit's overall status is FAIL. `soft` gate failures mean WARN — the
kind of thing worth a human glance but not disqualifying (older reproductions, e.g., predate the
`PAPER_BRIEFING.md`/`BUGFIX_LOG.md` convention and will legitimately WARN on those; that's
informative, not a bug in the audit).

Output is a JSON gate report (schema mirrors `posterly`'s `GATE_REPORT.json` shape) written to
`harness-testing/audits/<orid>.json` so results accumulate and can be diffed/compared across
reproductions over time.

## Tier 2 — Qualitative review (human, or an independent subagent)

The things a script can't check because they require actually reading and judging content.
Run this after Tier 1 passes, ideally by whoever *didn't* write the reproduction (or a fresh
subagent with no memory of writing it) — the BiMU claim 2/3 corrections in this repo were only
found because a second, independent read caught what the first pass missed.

1. **Spot-check the briefing's math against the source PDF.** Pick 1-2 equations/claims
   transcribed into `PAPER_BRIEFING.md` and verify them against the actual PDF page, not against
   the briefing's own paraphrase of itself.
2. **Re-run the Step 5 self-check independently.** Reread each claim's exact wording next to the
   reported numbers. Does the evidence actually support the stated verdict, at the scale it was
   run? Would a skeptical reader agree, or does a "toy-scale, direction matches" result get
   overstated anywhere?
3. **Check Step 0 actually fired when it should have.** If any claim ended up BLOCKED for "data
   access," did paper acquisition genuinely hit that wall, or did the reproduction route around a
   fixable problem without asking? (This can't be checked structurally — a `BLOCKED` label with a
   plausible-sounding reason still needs a human to ask "did we really try, or is Kaggle auth
   actually a 5-minute fix?")
4. **Check for the two named bug classes from `verdict_checklist.md`**: a gate/mask claimed at
   the wrong granularity, and a metric measuring something subtly different from what's claimed.
   These are exactly the two mistakes an independent re-read has already caught once in this repo
   (BiMU Claims 2/3) — worth explicitly re-checking every time, not just hoping the first pass
   caught them.
5. **Judge whether the harness steps themselves were followed in spirit, not just in form.** A
   `PAPER_BRIEFING.md` that exists but transcribes claims from `claims.json` instead of the PDF
   passes the `paper_source` gate's presence check but fails the actual intent of Step 0 — Tier 1
   can't catch that, a reader can.

Record findings as a short note appended to `harness-testing/audits/<orid>.json`'s sibling — either
inline in the JSON under a `qualitative_notes` field you add by hand, or as a markdown note in the
same `audits/` folder. If a finding implicates the skill itself (not just this one paper), open it
as a change to the relevant `SKILL.md`/`*.md` file directly.

## When to run this

- Once after the harness is used on the *next* new paper (the first real pressure test — see
  `HANDOFF.md`'s next actions).
- Periodically thereafter (e.g., every 2-3 new reproductions) rather than every single time, once
  the gates are consistently passing — the point is catching drift, not re-litigating settled
  process every run.
- Any time a reproduction's results get challenged or corrected (like the BiMU re-audit) — that's
  a signal the harness's own checks should have caught it, so run Tier 1+2 on it and see which gate
  should have flagged the issue but didn't; fix that gate.
