#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Tier-1 automated audit of the repro-harness pipeline (see AUDIT.md for Tier 2).

Checks a finished (or in-progress) reproduction folder for *structural evidence* that
Steps 0-7 of `.agents/skills/repro-harness/SKILL.md` were actually followed -- not
whether the content is good (that's the qualitative Tier-2 review). Writes a JSON gate
report mirroring the shape of posterly's GATE_REPORT.json.

This audits the harness's own reliability, not a paper's claims -- a hard gate failing
here means "the pipeline skipped a step," not "the reproduction is wrong."

Usage:
  uv run audit_harness.py <folder> [--repo-path .] [--orid ORID] [--json-out PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

VERDICT_TERMS = ("VERIFIED", "TOY-VERIFIED", "REFUTED", "BLOCKED")
INDEX_HEADING = "## Index"


def find_repo_root(start: Path) -> Path | None:
    start = start.resolve()
    for d in (start, *start.parents):
        readme = d / "README.md"
        if readme.is_file():
            try:
                if INDEX_HEADING in readme.read_text(encoding="utf-8", errors="ignore"):
                    return d
            except OSError:
                continue
    return None


def infer_orid(folder: Path) -> str | None:
    m = re.search(r"-([A-Za-z0-9]{8,12})$", folder.name)
    return m.group(1) if m else None


def read_all_text(folder: Path, patterns: tuple[str, ...] = ("*.md",)) -> str:
    chunks = []
    for pattern in patterns:
        for p in sorted(folder.rglob(pattern)):
            try:
                chunks.append(p.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
    return "\n".join(chunks)


def gate(name: str, severity: str, status: str, summary: str) -> dict:
    assert severity in ("hard", "soft")
    assert status in ("PASS", "WARN", "FAIL")
    return {"name": name, "severity": severity, "status": status, "summary": summary}


def gate_paper_source(folder: Path, orid: str | None) -> dict:
    text = read_all_text(folder)
    has_openreview = bool(re.search(r"openreview\.net/(pdf|forum)\?id=", text, re.I)) or \
        bool(re.search(r"openreview[\s\-_]*(pdf|paper)", text, re.I))
    if orid and re.search(re.escape(orid), text):
        has_openreview = has_openreview or bool(re.search(rf"{re.escape(orid)}.{{0,80}}(pdf|openreview)", text, re.I))
    has_arxiv = bool(re.search(r"arxiv\.org/abs/|arxiv[:\s]", text, re.I))
    if not has_openreview:
        return gate("paper_source", "hard", "FAIL",
                     "No OpenReview PDF reference found in this folder's markdown -- Step 0 "
                     "should cite where the paper text came from.")
    if has_arxiv and not re.search(r"(cross-check|consistent|diverg|arXiv version|both versions)", text, re.I):
        return gate("paper_source", "hard", "WARN",
                     "OpenReview PDF referenced, and an arXiv version is also mentioned, but no "
                     "cross-check/divergence note was found -- confirm Step 0's arXiv diff happened.")
    return gate("paper_source", "hard", "PASS", "OpenReview PDF reference found.")


def gate_briefing_exists(folder: Path) -> dict:
    if list(folder.glob("PAPER_BRIEFING.md")):
        return gate("briefing_exists", "soft", "PASS", "PAPER_BRIEFING.md present.")
    return gate("briefing_exists", "soft", "WARN",
                 "No PAPER_BRIEFING.md -- fine for reproductions that predate this convention, "
                 "otherwise Step 2 was skipped.")


def gate_smoketest_evidence(folder: Path) -> dict:
    log_dirs = [d for d in folder.rglob("*") if d.is_dir() and d.name.lower() in ("logs", "log")]
    candidates: list[Path] = []
    for d in log_dirs:
        candidates.extend(p for p in d.iterdir() if p.is_file())
    candidates.extend(p for p in folder.glob("*.log") if p.is_file())
    if not candidates:
        return gate("smoketest_evidence", "soft", "WARN", "No log files found to check for a smoketest.")
    name_hit = any(re.search(r"smoke|sanity|toy", p.name, re.I) for p in candidates)
    size_hit = False
    if len(candidates) >= 2:
        sizes = sorted(p.stat().st_size for p in candidates)
        size_hit = sizes[0] > 0 and sizes[-1] / max(sizes[0], 1) >= 5
    if name_hit or size_hit:
        return gate("smoketest_evidence", "soft", "PASS", "Found a log that looks like a smoketest.")
    return gate("smoketest_evidence", "soft", "WARN",
                 "Log files exist but none look like a smoketest (by name or by being much smaller "
                 "than the largest log) -- confirm Step 3 actually ran a small config first.")


def gate_self_audit_log(folder: Path) -> dict:
    if list(folder.glob("BUGFIX_LOG.md")):
        return gate("self_audit_log", "soft", "PASS", "BUGFIX_LOG.md present.")
    text = read_all_text(folder)
    if re.search(r"(?i)(correct(ed|ion)s?\b|re-?audit)", text):
        return gate("self_audit_log", "soft", "PASS",
                     "No BUGFIX_LOG.md, but found correction/re-audit language in the README.")
    return gate("self_audit_log", "soft", "WARN",
                 "No BUGFIX_LOG.md and no correction/re-audit language found -- Step 4's self-audit "
                 "may not have happened, or wasn't documented.")


def gate_verdict_vocabulary(folder: Path) -> dict:
    readme = folder / "README.md"
    if not readme.is_file():
        return gate("verdict_vocabulary", "soft", "WARN", "No folder README.md to check.")
    text = readme.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"##\s*Verdict.*?(?=\n##\s|\Z)", text, re.S | re.I)
    if not m:
        return gate("verdict_vocabulary", "soft", "WARN", "No '## Verdict' section found in README.md.")
    section = m.group(0)
    rows = [l for l in section.splitlines() if l.strip().startswith("|") and not re.match(r"^\|[\s:|-]+\|\s*$", l)]
    rows = rows[1:] if rows else rows  # drop header row
    if not rows:
        return gate("verdict_vocabulary", "soft", "WARN", "Verdict section has no table rows to check.")
    hits = sum(1 for r in rows if any(term in r.upper() for term in VERDICT_TERMS))
    frac = hits / len(rows)
    if frac >= 0.8:
        return gate("verdict_vocabulary", "soft", "PASS", f"{hits}/{len(rows)} verdict rows use the standard vocabulary.")
    return gate("verdict_vocabulary", "soft", "WARN",
                 f"Only {hits}/{len(rows)} verdict rows clearly use VERIFIED/TOY-VERIFIED/REFUTED/BLOCKED "
                 "-- check whether prose verdicts still map cleanly to one of the four.")


def _paragraphs(text: str) -> list[str]:
    """Join soft-wrapped prose lines into paragraphs; keep table rows/headings as their own unit.

    Markdown source in this repo hard-wraps prose across raw newlines (see the BiMU README's
    "NOT vendored" note, where a single sentence's "blocked," and its reason land on different
    physical lines) -- scanning line-by-line would split a reason from its trigger word. Table
    rows can't span multiple physical lines in valid markdown, so those are kept one-per-unit.
    """
    paras: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("|", "#")):
            if buf:
                paras.append(" ".join(buf))
                buf = []
            if stripped.startswith("|"):
                paras.append(stripped)
        else:
            buf.append(stripped)
    if buf:
        paras.append(" ".join(buf))
    return paras


def gate_blocked_claims_disclosed(folder: Path) -> dict:
    readme = folder / "README.md"
    if not readme.is_file():
        return gate("blocked_claims_disclosed", "hard", "PASS", "No folder README.md to check.")
    text = readme.read_text(encoding="utf-8", errors="ignore")
    bad = []
    for para in _paragraphs(text):
        if re.search(r"(?i)\bblocked\b", para):
            after = re.split(r"(?i)blocked", para, maxsplit=1)[1]
            after = re.sub(r"^[\s*_—\-,:)]+", "", after)
            if len(after.strip()) < 15:
                bad.append(para.strip())
    if bad:
        return gate("blocked_claims_disclosed", "hard", "FAIL",
                     f"{len(bad)} 'blocked' mention(s) with no stated reason, e.g.: {bad[0][:120]!r}")
    return gate("blocked_claims_disclosed", "hard", "PASS", "Every 'blocked' mention has an accompanying reason.")


def gate_index_row_present(folder: Path, repo_root: Path | None) -> dict:
    if repo_root is None:
        return gate("index_row_present", "hard", "WARN", "Could not find a repo root README.md to check.")
    readme = repo_root / "README.md"
    text = readme.read_text(encoding="utf-8", errors="ignore")
    if re.search(re.escape(folder.name) + r"/", text):
        return gate("index_row_present", "hard", "PASS", "Folder is linked from the top-level README's Index table.")
    return gate("index_row_present", "hard", "FAIL",
                 f"No link to '{folder.name}/' found in {readme} -- Step 7's Index row is missing.")


def gate_raw_results_present(folder: Path) -> dict:
    result_dirs = [d for d in folder.rglob("*") if d.is_dir() and "result" in d.name.lower()]
    for d in result_dirs:
        if any(p.is_file() and p.stat().st_size > 0 for p in d.rglob("*")):
            return gate("raw_results_present", "soft", "PASS", f"Non-empty results found under {d.relative_to(folder)}/.")
    if result_dirs:
        return gate("raw_results_present", "soft", "WARN", "results/-like directory exists but appears empty.")
    # Most reproductions in this repo don't use a results/ subfolder at all -- raw CSV/log/JSON
    # outputs sit flat at the folder root instead (see harness-testing/audits/nPzckCXmHE.md).
    # Only checking for a results/ dir false-warns on that (the majority) convention.
    flat_hits = [p for p in folder.glob("*") if p.is_file() and p.suffix.lower() in (".csv", ".log", ".json")
                 and p.stat().st_size > 0]
    if flat_hits:
        return gate("raw_results_present", "soft", "PASS",
                     f"Non-empty raw result files found at folder root (e.g. {flat_hits[0].name}).")
    return gate("raw_results_present", "soft", "WARN",
                 "No results/-like directory and no *.csv/*.log/*.json files at the folder root -- "
                 "fine for a pure derivation/math reproduction, otherwise raw outputs may be missing.")


def gate_no_vendored_code(folder: Path) -> dict:
    nested_git = list(folder.rglob(".git"))
    tracked = [p for p in nested_git if not _is_git_ignored(p.parent)]
    if tracked:
        return gate("no_vendored_code", "hard", "FAIL",
                     f"Found nested .git at {tracked[0].relative_to(folder)} that is NOT gitignored -- "
                     "third-party code may be vendored instead of linked (Step 7 says link + clone "
                     "instructions, don't vendor).")
    if nested_git:
        return gate("no_vendored_code", "hard", "PASS",
                     f"Found {len(nested_git)} nested .git dir(s), but all are gitignored (linked clones, "
                     "not vendored).")
    return gate("no_vendored_code", "hard", "PASS", "No nested .git directories found.")


def _is_git_ignored(path: Path) -> bool:
    path = path.resolve()
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            cwd=path.parent, capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def run_audit(folder: Path, repo_path: str | None, orid: str | None) -> dict:
    orid = orid or infer_orid(folder)
    repo_root = Path(repo_path).resolve() if repo_path else find_repo_root(folder.parent)

    gates = [
        gate_paper_source(folder, orid),
        gate_briefing_exists(folder),
        gate_smoketest_evidence(folder),
        gate_self_audit_log(folder),
        gate_verdict_vocabulary(folder),
        gate_blocked_claims_disclosed(folder),
        gate_index_row_present(folder, repo_root),
        gate_raw_results_present(folder),
        gate_no_vendored_code(folder),
    ]

    hard_failures = sum(1 for g in gates if g["severity"] == "hard" and g["status"] == "FAIL")
    warnings = sum(1 for g in gates if g["status"] == "WARN")
    overall = "FAIL" if hard_failures else ("WARN" if warnings else "PASS")

    return {
        "schema_version": 1,
        "audit": "repro-harness Tier-1 (see harness-testing/AUDIT.md)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "folder": str(folder.resolve()),
        "orid": orid,
        "overall": overall,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "gates": gates,
        "note": "Tier-1 only -- run the Tier-2 qualitative review in AUDIT.md before trusting this alone.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="path to the reproduction folder to audit")
    parser.add_argument("--repo-path", help="monorepo root (default: search upward for a README with an Index table)")
    parser.add_argument("--orid", help="OpenReview id (default: inferred from the folder name's -<orid> suffix)")
    parser.add_argument("--json-out", help="where to write the JSON report (default: print to stdout only)")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"error: {folder} is not a directory", file=sys.stderr)
        return 1

    report = run_audit(folder, args.repo_path, args.orid)

    for g in report["gates"]:
        marker = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL"}[g["status"]]
        print(f"[{g['severity']:4}] {marker:4} {g['name']:24} {g['summary']}")
    print(f"\noverall: {report['overall']}  (hard_failures={report['hard_failures']}, warnings={report['warnings']})")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out_path}")
    else:
        print("\n(pass --json-out PATH to save this report)")

    return 1 if report["overall"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
