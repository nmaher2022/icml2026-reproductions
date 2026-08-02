# Verdict vocabulary and pre-submission self-check

## Vocabulary (use exactly these five; don't invent synonyms)

- **VERIFIED** — reproduced at (or acceptably close to) the paper's own scale, and the direction
  *and* magnitude match the claim.
- **TOY-VERIFIED** — reproduced at a reduced scale (fewer tasks/steps/samples than the paper),
  and the result is *directionally* consistent with the claim. Explicitly not the same as
  VERIFIED — say what scale you ran and why the full scale wasn't feasible.
- **REFUTED** — ran at a fair scale (paper's own scale, or a scale large enough that the claimed
  effect should already be visible) and the result contradicts the claim. State the discrepancy
  in numbers, and consider whether the claim might still hold at a different scale you didn't
  reach — say so if plausible, don't overclaim refutation either.
- **BLOCKED** — not attempted, because of a concrete access/compute obstacle (dataset auth wall,
  no GPU, missing credential). Name the obstacle. Confirm it's still real (e.g. a quick web check
  that the dataset isn't simply dead) so "blocked" isn't a stand-in for "didn't get to it."
- **INCONCLUSIVE** — attempted at a fair scale, but the result neither clearly supports nor
  clearly contradicts the claim, and forcing it into VERIFIED/TOY-VERIFIED/REFUTED would overstate
  the evidence either way. Two legitimate cases seen in practice (confirmed twice, SpectralCIT's
  Claim 3/5 and Ellipsoidal TSF's Claim 5): (1) the run was too small/short to reach the regime the
  claim is actually about (e.g. a claim about behavior "past a collapse point" when nothing
  collapsed at toy scale), or (2) the metric pins at a degenerate value across all conditions,
  giving no signal either way. Don't use INCONCLUSIVE as a soft landing for "didn't look closely
  enough" — it still requires the same self-check rigor as the other four, and the writeup must
  say specifically what would need to change (more scale, a different metric) to get a real
  verdict.

Mixed results (e.g. one sub-metric matches, another doesn't — see the BiMU Claim 2 OOD-AUC-matches
but accuracy-doesn't case) don't get forced into a single label: state each sub-result's verdict
separately, and give the honest combined read in prose rather than picking whichever verdict looks
better.

## Pre-submission self-check (run this before writing the final verdict)

1. Reread the claim's exact wording (from the source PDF, not a paraphrase) side by side with your
   own numbers/plots. Does the evidence actually support what the claim says, at the scale you ran
   it?
2. If you ran at reduced scale: is there a specific reason (stated in the writeup) why the
   paper's claimed effect might not be expected to appear at this scale? ("10 tasks is too short a
   sequence for a catastrophic-forgetting mode that's claimed to emerge over 1000 tasks" is a
   valid reason; "we didn't have time to check" is not a reason to round up the verdict, just an
   honest scope statement.)
3. If a metric name in your code doesn't obviously match the claim's metric: re-derive what it's
   actually measuring from the code (not the variable name) before trusting it. Two bug classes
   from past reproductions to check for specifically:
   - A gate/mask claimed to be per-parameter that's actually computed once globally (or vice
     versa) — re-read the actual boolean/array shape, don't trust a comment or the first
     description written.
   - A metric that's described as "a repeated probe of the final model" but is actually a
     per-checkpoint/per-task-boundary trajectory (or vice versa) — check what data each recorded
     value was actually computed against.
4. Did you check for implementation bugs that would make a claim mechanistically impossible to
   verify even if the numbers happen to look plausible? (E.g. a coordination signal computed but
   never fed into the thing it's supposed to affect — the result can look "roughly right" by
   coincidence while the mechanism under test isn't actually running.) A dedicated code-review
   pass (self-review or a subagent) specifically hunting for this before finalizing verdicts has
   caught real bugs in past reproductions — don't skip it for claims backed by a nontrivial
   algorithm.
5. Are all BLOCKED claims named in the final summary, not just omitted from the verdict table?
6. Full run logs and raw result files (not just the numbers you quote) are saved in the bundle —
   so results are checkable, not "trust me" numbers.
