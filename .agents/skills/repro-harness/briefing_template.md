# PAPER_BRIEFING.md template

Copy this into the working reproduction folder and fill it in before writing any code. Source
every fact from the acquired PDF(s) (Step 0), not from a claims.json string alone.

```markdown
# <Paper title> — reproduction briefing

Paper: <arXiv id/version if any>, "<title>", <authors> (<affiliation>).
OpenReview id: <orid>. Local copy of the PDF: `<path from Step 0>`.
<If an arXiv version was also checked:> arXiv/OpenReview cross-check: <"consistent" | list of
divergences found and which version is treated as authoritative>.

Challenge: HF Space `ICML-2026-agent-repro/challenge`. This reproduction lands in
`nmaher2022/icml2026-reproductions` as `<slug>-<orid>/` (see `reference-icml-monorepo-conventions`
memory / repo README for the folder convention).

## Working conventions for this reproduction
- <PEP-723 self-contained scripts run via `uv run`, or whatever this specific reproduction's
  environment actually needs — state it precisely, including any undocumented dependencies found>
- **Smoketest before scale**: before running anything longer than ~30s-1min, run a tiny/fast
  version (few iterations, tiny dims) and check for shape errors, NaNs, sane magnitudes, sign
  errors. Only scale up once clean.
- All work happens in `<working-folder>/`. Don't touch other folders in this repo.
- Verdict vocabulary (see `verdict_checklist.md`): VERIFIED / TOY-VERIFIED / REFUTED / BLOCKED /
  INCONCLUSIVE.
  State the scale run next to every verdict. Never round a toy-scale pass up to VERIFIED. Report
  blocked claims explicitly, never fake or silently skip them.
- Self-check before finishing: reread the exact claim text and your own numbers/plots side by
  side — does the evidence actually support what the claim says, at the scale you ran it?

## Claims in scope (verbatim from the paper, with section/table/equation refs)
1. <claim 1, quoted or closely paraphrased with page/section number>
2. ...

## Core math / setup (transcribed from the paper, not from a claim-extraction summary)
<Equations, algorithm boxes, table setups needed to implement or check the claims above — copy
these from the PDF directly, cite section numbers.>

## Known access blockers (fill in as discovered)
<e.g. "Dataset X requires a Google Drive sign-in" / "Dataset Y needs a Kaggle API token" — note
here as soon as found so Step 4/5 can mark the relevant claims BLOCKED without re-discovering it.>
```
