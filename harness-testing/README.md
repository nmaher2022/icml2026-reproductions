# Harness testing

This folder is the project home for the **repro-harness** meta-effort: building and validating a
reusable pipeline for reproducing ICML 2026 papers, on top of the 10 reproductions already done in
this repo. It is not a paper reproduction itself, so it doesn't get a row in the top-level
README's Index table — see the "Harness" note in that README instead.

The skill itself lives at [`.agents/skills/repro-harness/`](../.agents/skills/repro-harness/)
(skills have to live there to be discoverable/invokable) — this folder holds everything *about*
that skill: why it's built the way it is, its current build status, and how we check whether it's
actually working.

## Contents

- [`design-notes.md`](design-notes.md) — the conversation that produced the skill, kept verbatim
  as rationale for why each pipeline step exists.
- [`HANDOFF.md`](HANDOFF.md) — build status and open items for the skill and its scaffold script.
  Read this first for "what's actually done vs. still open."
- [`AUDIT.md`](AUDIT.md) — how we audit the harness itself: automated structural gates plus a
  qualitative review pass, run after each paper the harness is used on.
- [`audit_harness.py`](audit_harness.py) — the automated half of the audit; run against a finished
  (or in-progress) reproduction folder, writes a JSON gate report.
- [`audits/`](audits/) — one gate report per audited paper, named `<orid>.json`.
- [`candidates.md`](candidates.md) — the current pick for "next paper to test the harness on" and
  why, refreshed against the live challenge leaderboard each time a new candidate is chosen.

## Why this is separate from a normal paper folder

Every other top-level folder in this repo is a finished reproduction of one paper's claims. This
one is infrastructure: it doesn't reproduce a paper, it tracks whether the *process* used to
reproduce papers is sound. Keeping it separate means the Index table stays exactly "one row per
paper" and this project's own churn (skill revisions, audit reports) doesn't pollute that table's
git history.
