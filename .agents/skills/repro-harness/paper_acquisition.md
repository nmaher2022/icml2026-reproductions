# Paper acquisition (Step 0 detail)

Why this is a hard gate, not a nice-to-have: the challenge's claims are extracted from a specific
paper version, and every downstream step (briefing, code audit, verdict) is only as trustworthy as
the text it was checked against. A claim string pulled from `claims.json` alone is already one
level removed from the paper — it can paraphrase, drop qualifiers, or misattribute a setting (this
happened for real: the BiMU reproduction's Claim 2 was phrased as "task-boundary-dependent
BayesBiNN baseline," but the paper's own Table 1 runs BayesBiNN with `Task bounds = NO` — the
extracted claim string didn't match the paper's actual experimental setup). Reading the source PDF
is the only way to catch that class of error before it propagates into a false verdict.

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

2. **If an arXiv id is known or discoverable** (from the OpenReview page, the challenge metadata,
   or a web search), fetch that too and save it alongside, e.g.
   `<working-folder>/paper-arxiv-<id>.pdf`. Spot-check the claims you're about to work on against
   both:
   - Same headline numbers in the relevant table?
   - Same experimental setting (dataset split, baseline configuration, hyperparameters) named in
     the claim?
   - Any section/ablation present in one but not the other?
   If they diverge on anything you're about to cite as "the claim," note the divergence explicitly
   in `PAPER_BRIEFING.md` and treat the **OpenReview version as authoritative** — it's what the
   challenge is actually scoring against.

3. **If OpenReview access fails** (auth wall, `WebFetch` errors out, forum page won't render,
   PDF link 404s/rate-limits, etc.), do not substitute the arXiv version silently and do not
   proceed on the claims.json text alone. Stop and ask:

   ```
   AskUserQuestion({
     questions: [{
       question: "I can't reach the OpenReview PDF for <orid> (<error summary>). The paper's
         actual text is the source of the claims I'd be verifying, so I don't want to proceed
         from a paraphrase alone. Could you download the PDF (openreview.net/pdf?id=<orid>) and
         drop it somewhere I can read, or paste a path if you already have it?",
       header: "Paper access",
       options: [
         {label: "I'll download it now", description: "Pause and wait; tell me the path once it's saved."},
         {label: "Use the arXiv version instead", description: "Proceed on arXiv only, with a note that OpenReview-specific revisions couldn't be checked."},
         {label: "Proceed on claims.json text only", description: "Not recommended — flag every verdict as based on the extracted claim string, not the source paper."}
       ],
       multiSelect: false
     }]
   })
   ```
   If the user picks a fallback option, record that choice explicitly in `PAPER_BRIEFING.md` and
   carry the caveat into every affected claim's verdict (e.g. "verified against the arXiv v1 text;
   OpenReview camera-ready revisions, if any, not cross-checked").

4. Once you have a readable PDF (local file), use the `Read` tool directly on it (it handles PDFs,
   `pages` param for long ones) rather than re-summarizing from web search snippets — the audit
   steps later need exact wording, table numbers, and equation numbers.
