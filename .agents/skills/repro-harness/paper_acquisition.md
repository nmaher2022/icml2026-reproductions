# Paper acquisition (Step 0 detail)

Why this is a hard gate, not a nice-to-have: the challenge's claims are extracted from a specific
paper version, and every downstream step (briefing, code audit, verdict) is only as trustworthy as
the text it was checked against. A claim string pulled from `claims.json` alone is already one
level removed from the paper — it can paraphrase, drop qualifiers, or misattribute a setting (this
happened for real: the BiMU reproduction's Claim 2 was phrased as "task-boundary-dependent
BayesBiNN baseline," but the paper's own Table 1 runs BayesBiNN with `Task bounds = NO` — the
extracted claim string didn't match the paper's actual experimental setup). Reading the source PDF
is the only way to catch that class of error before it propagates into a false verdict.

## Run this as a subagent

Spawn it with the `Agent` tool, `subagent_type: general-purpose`, run in the **foreground**
(`run_in_background: false`) — Step 2 (the briefing) can't start until acquisition either succeeds
or the whole attempt is aborted, so the main agent needs the result synchronously. The point of
delegating is context isolation: PDF text, failed `WebFetch`/`curl` attempts, and retries all pile
up fast, and none of it needs to sit in the main agent's context once a clean PDF path (or a
failure report) comes back.

Brief the subagent with: the paper title, the OpenReview id, an arXiv id if already known, the
working reproduction folder path to save into, and the exact order of operations below (don't
paraphrase it — the appendix check in step 3 is the part most likely to get silently skipped if
summarized).

## Order of operations

1. **Try OpenReview first.**
   ```
   WebFetch("https://openreview.net/forum?id=<orid>", "find the submission PDF link and any
   revision history / camera-ready note")
   ```
   Then fetch the PDF itself (OpenReview PDF URLs are typically
   `https://openreview.net/pdf?id=<orid>`). Save it locally, e.g.
   `<working-folder>/paper-openreview-<orid>.pdf`, so later steps can re-read specific sections
   without re-fetching.

2. **If OpenReview is blocked** (auth wall, `WebFetch` errors out, forum page won't render, PDF
   link 404s/rate-limits, bot-challenge/403 on either the web UI or the API — this has been a
   confirmed hard constraint in this environment before, don't burn more than one or two retries
   on it) — **fall back to arXiv**. Use a known arXiv id if the OpenReview page (or challenge
   metadata) surfaced one; otherwise search by title:
   ```
   https://export.arxiv.org/api/query?search_query=ti:"<exact title>"
   ```
   Fetch and save the PDF, e.g. `<working-folder>/paper-arxiv-<id>.pdf`.

3. **Check the arXiv paper is completely readable — specifically the methodology appendix, if
   the paper has one.** A paper that only fully renders its main body but truncates, drops, or
   garbles the appendix where the actual method/algorithm/hyperparameters live is *not* a usable
   source — the briefing and audit steps need that detail. If the arXiv version is unreadable, or
   readable only in part (missing/garbled appendix in particular), **exit the process
   immediately**:
   - Do not fall back to `claims.json` text or the abstract and proceed as if that's equivalent.
   - Do not pause with `AskUserQuestion` to ask for a manually-supplied PDF — this step no longer
     has a stop-and-ask branch. Acquisition either succeeds cleanly from OpenReview or arXiv, or
     the attempt ends here.
   - Return a short, specific failure report to the main agent: which source(s) were tried, what
     failed at each (blocked/404/rate-limited/incomplete), and — if arXiv partially rendered —
     exactly what was missing (e.g. "Appendix B [hyperparameters] present but Appendix C
     [proof of Theorem 2] truncated after page 14").

4. **If both sources are readable**, spot-check the claims you're about to work on against both:
   - Same headline numbers in the relevant table?
   - Same experimental setting (dataset split, baseline configuration, hyperparameters) named in
     the claim?
   - Any section/ablation present in one but not the other?
   If they diverge on anything you're about to cite as "the claim," note the divergence explicitly
   in `PAPER_BRIEFING.md` and treat the **OpenReview version as authoritative** — it's what the
   challenge is actually scoring against.

5. Once you have a readable PDF (local file), use the `Read` tool directly on it (it handles PDFs,
   `pages` param for long ones) rather than re-summarizing from web search snippets — the audit
   steps later need exact wording, table numbers, and equation numbers.

## What the main agent does with the result

A **success** result from the subagent is: the local PDF path(s) it saved, and — if both sources
were fetched — any noted OpenReview/arXiv divergence. Proceed to Step 1/2 using that path.

A **failure** result is a hard stop for this paper, not a degrade-and-continue. Report the failure
to the user directly (which source(s) failed and why) and do not proceed to Step 1+ for this
paper. Pick a different candidate, or wait for the user to supply a PDF directly before retrying
Step 0 — but that's a new decision made by the user in a later turn, not an in-task pause built
into this step.
