# Step 8 audit — Tab-PE (WB0hLRRlcj)

Tier-1: PASS, 0 hard failures, 0 warnings (see sibling `WB0hLRRlcj.json`) — but only after a fix.

## Harness gap found and fixed

`no_vendored_code` initially FAILed on `DPSDA_upstream_aim/.git`, a nested git dir from cloning
the official `microsoft/DPSDA` repo for the AIM-baseline runs. The gate did a raw filesystem
`rglob(".git")` scan with no awareness of `.gitignore` — it couldn't distinguish a genuinely
vendored (committed) third-party tree from a gitignored, untracked clone kept around only for
local reruns (exactly the "link, don't vendor" pattern Step 7 asks for). Confirmed via
`git status --short` and `git check-ignore -v` that both `DPSDA_upstream/` and
`DPSDA_upstream_aim/` are correctly gitignored and untracked in this reproduction.

Fixed `audit_harness.py`'s `gate_no_vendored_code` to shell out to `git check-ignore` on each
nested `.git`'s parent directory and only hard-fail on ones that are *not* ignored. Verified the
fix both ways: it now PASSes this reproduction's two legitimately-gitignored clones, and still
correctly FAILs a synthetic test folder with a nested `.git` that has no gitignore rule covering
it (tested against a scratch dir outside the repo, not committed anywhere).

## Tier-2 — light spot-check (not full checklist)

Per `AUDIT.md`'s cadence guidance, a full Tier-2 pass isn't due yet (this is only the reproduction
immediately after the `nPzckCXmHE` audit, not the 2-3rd one). Did a light independent check
instead of the full 5-item list:

- Cross-checked `PAPER_BRIEFING.md`'s two headline numbers (AC: Tab-PE 49.38% vs AIM 23.24%; PA:
  Tab-PE 63.72% vs AIM 59.53%) against the paper text — both transcribed correctly.
- Cross-checked `VERDICTS.md`'s reported numbers against the actual raw evidence files
  (`results/*/tabular_classifier_tabicl_filter_*_test_acc.csv`, `results/*_aim/eval.json`) rather
  than trusting the summary tables at face value — all four figures traced exactly to the raw
  files, including the one place where our result *diverges* from the paper (our AIM baseline on
  Person Activity: 48.32% vs paper's 59.53%). That divergence is disclosed openly in `VERDICTS.md`
  with a stated cause (untuned/subsampled AIM baseline), not glossed over or hidden behind the
  "beats AIM" headline — a good sign the self-audit was genuine rather than post-hoc rationalized.

No further Tier-2 items (briefing math spot-check beyond the two numbers above, Step-0-genuinely-
hit-a-wall check for the TabPFN/Claim-4 substitution, etc.) were run this pass — deferred to the
next scheduled full Tier-2, per cadence.

## Verdict

Harness gap fixed at the tool level (benefits all future reproductions, not just this one).
Reproduction itself: no issues found in the spot-check. Safe to mirror and commit.
