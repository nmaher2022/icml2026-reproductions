# Surviving a Claude session limit mid-reproduction

No skill can prevent a Claude Pro usage/session limit from being hit — that's a platform-level
constraint, not something client-side instructions can bypass. What the harness *can* do is make
hitting one cost close to nothing: the compute keeps running unattended, and the next session (or
a manual restart) picks up exactly where it left off from one file, instead of re-deriving context
or restarting a multi-hour job from scratch. Two separate problems, two separate mitigations.

## Problem 1 — the session dies while a long job is running

If a training/eval run is launched as a foreground blocking call, it dies with the session. Fix:
**never launch a long-running job in the foreground.** Detach it so the OS process outlives the
Claude session:

```bash
nohup ./.venv/bin/python -u repro_toy.py --outdir ./outputs [...] > outputs_run.log 2>&1 &
```

(`-u`/unbuffered so `tail -f outputs_run.log` shows live progress from a fresh session with no
memory of launching it.) This is already the pattern used in
`fake-forgetting-uncertainty-rjmVJaBpkm/REPRO_LOG.md` — keep using it for anything that runs
longer than a few minutes. A new session recovers visibility with:

```bash
pgrep -af repro_toy.py      # still running?
tail -n 50 outputs_run.log  # how far did it get?
```

## Problem 2 — work needs to resume across sessions regardless

Even with a detached job, a fresh session needs to know the job exists, what it's for, and what to
do next. Fix: **write the recovery file *before* starting the long job, not after finishing it.**
If the session dies one message after launch, the recovery file is the only thing standing between
"pick up in 30 seconds" and "re-derive everything from scratch."

Use the `REPRO_LOG.md` pattern already established in this repo — one file per paper folder (or
at the working root if the folder doesn't exist yet), with, at minimum:

```markdown
## Context-reset / session-teardown recovery (READ FIRST on a cold start)
1. Read auto-memory -> it points here.
2. Read STATUS + NEXT ACTIONS below.
3. <exact command to check what's already done, e.g. inspect a results.json / checkpoint dir>
4. `pgrep -af <script>` -- if the process is dead and work remains, relaunch the exact command
   below. It should be resumable (see "Make it resumable" below).
5. Continue with NEXT ACTIONS.

## STATUS (updated <timestamp>)
<what's running, what's done, what's blocked>

## NEXT ACTIONS
<ordered list>
```

**Save a memory pointer to this file as soon as it exists** — right when the long job launches,
not at the end of the reproduction. Auto-memory is what survives a session boundary; if the
recovery file exists on disk but nothing points to it from memory, a fresh session has no way to
discover it without the user re-explaining. A one-line project memory ("mid-run: see
`<folder>/REPRO_LOG.md` for status/resume") costs nothing to write and is the difference between
an automatic resume and a manual one.

## Make it resumable, not just restartable

A relaunch that redoes already-completed work wastes exactly the compute time a session limit
already cost you. Prefer scripts that checkpoint progress and skip what's done — `repro_toy.py`'s
pattern: write completed results into a `results.json` keyed by method/config, and check that file
before starting each unit of work, so a second launch of the identical command only computes what's
still missing. Save model checkpoints (`ckpt/*.pth` or equivalent) alongside, so a training run
that was interrupted mid-epoch doesn't need to restart from epoch 0 either — reload the last
checkpoint if the framework supports it, or fall back to per-unit resumability (skip whole
finished tasks/configs) if mid-run checkpointing isn't feasible in the time available.

## Where this fits in the pipeline

This applies within **Step 4** (run + self-audit) whenever a run is expected to take longer than
one session comfortably allows: launch detached, write the recovery file first, make the script
resumable, save the memory pointer immediately. It's not a separate step because it's not
optional scaffolding — for anything long-running, it's part of how Step 4 should be done from the
start, not bolted on after the first session dies mid-run.
