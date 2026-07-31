#!/usr/bin/env python3
"""Validate ICML 2026 reproduction logbook structure before publish."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from trackio import logbook as lb

    _HAS_TRACKIO = True
except ImportError:
    lb = None
    _HAS_TRACKIO = False

ROOT_SLUG = "index"
OPENREVIEW_ID_RE = re.compile(r"^[A-Za-z0-9]{8,12}$")
HF_PAPER_URL_RE = re.compile(r"https://huggingface\.co/papers/\S+")
OPENREVIEW_URL_RE = re.compile(r"https://openreview\.net/forum\?id=\S+")
HUB_URL_RE = re.compile(
    r"https://huggingface\.co/(models|datasets|spaces|jobs|buckets)/[^\s<>\"'`]+"
)
GITHUB_REPO_RE = re.compile(r"https://github\.com/[^\s<>\"'`]+")


def _fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def _repo_name(space_id: str | None) -> str | None:
    if not space_id or "/" not in space_id:
        return None
    return space_id.partition("/")[2]


def _looks_like_openreview_repo(name: str) -> bool:
    if OPENREVIEW_ID_RE.fullmatch(name):
        return True
    if name.startswith("repro-"):
        suffix = name[6:]
        if OPENREVIEW_ID_RE.fullmatch(suffix):
            return True
    return False


def validate_with_trackio(space_id: str | None) -> int:
    proj = lb.require_project_dir()
    result = lb.validate_logbook(proj, profile="icml2026", space_id=space_id)
    for warning in result["warnings"]:
        _warn(warning)
    if result["errors"]:
        for err in result["errors"]:
            _fail(err)
        return 1
    print("Logbook validation passed.")
    return 0


def _find_project_dir(start: Path | None = None) -> Path | None:
    start = Path(start or Path.cwd()).resolve()
    for d in (start, *start.parents):
        candidate = d / ".trackio"
        if (candidate / "logbook" / "pages" / "index.md").is_file():
            return candidate
    return None


def _link_order(index_path: Path) -> list[str]:
    text = index_path.read_text(encoding="utf-8")
    seen: list[str] = []
    for slug in re.findall(r"\(#/([A-Za-z0-9._-]+)\)", text):
        if slug not in seen:
            seen.append(slug)
    return seen


def _index_prose(text: str) -> str:
    cell_re = re.compile(
        r"(^|\n)---\n<!-- trackio-cell\n([\s\S]*?)\n-->\n([\s\S]*?)"
        r"(?=\n---\n<!-- trackio-cell\n|\s*$)"
    )
    return cell_re.sub("", text).strip()


def _title_of(index_path: Path) -> str:
    for line in index_path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"#\s+(.+)", line.strip())
        if m:
            return m.group(1).strip()
    return ""


def _parse_cells(text: str) -> list[dict]:
    cell_re = re.compile(
        r"(^|\n)---\n<!-- trackio-cell\n([\s\S]*?)\n-->\n([\s\S]*?)"
        r"(?=\n---\n<!-- trackio-cell\n|\s*$)"
    )
    cells = []
    for match in cell_re.finditer(text):
        try:
            meta = json.loads(match.group(2))
        except json.JSONDecodeError:
            continue
        cells.append(
            {
                "type": meta.get("type"),
                "title": meta.get("title"),
                "pinned": meta.get("pinned"),
                "body": match.group(3),
            }
        )
    return cells


def validate_standalone(space_id: str | None) -> int:
    proj = _find_project_dir()
    if proj is None:
        _fail(
            "No logbook in this directory. Scaffold first with scaffold_icml_logbook.py."
        )
        return 1

    errors = 0
    warnings = 0
    root = proj / "logbook"
    metadata_path = proj / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    tags = metadata.get("tags") or []

    if "icml2026-repro" not in tags:
        _fail('metadata.json tags must include "icml2026-repro".')
        errors += 1
    paper_tags = [t for t in tags if t.startswith("paper-")]
    if not paper_tags:
        _fail('metadata.json tags must include "paper-<openreview-id>".')
        errors += 1

    repo = _repo_name(space_id or metadata.get("space_id"))
    if not repo:
        _fail("Pass --space username/repro-<slugified-title> or set metadata.space_id.")
        errors += 1
    elif not repo.startswith("repro-"):
        _fail(f'Space repo name must start with "repro-" (got "{repo}").')
        errors += 1
    elif _looks_like_openreview_repo(repo):
        _fail(f'Space slug "{repo}" looks like an OpenReview id.')
        errors += 1

    index_path = root / "pages" / "index.md"
    index_text = index_path.read_text(encoding="utf-8")
    index_prose = _index_prose(index_text)
    title = _title_of(index_path)
    if not re.match(r"^Reproduction:\s+.+\S", title, re.I):
        _fail('Index heading must be "# Reproduction: <paper title>".')
        errors += 1
    if not HF_PAPER_URL_RE.search(index_prose) and not OPENREVIEW_URL_RE.search(
        index_prose
    ):
        _fail("Index must link to HF papers or OpenReview.")
        errors += 1

    toc_slugs = _link_order(index_path)
    disk_slugs = [
        d.name
        for d in (root / "pages").iterdir()
        if d.is_dir() and (d / "page.md").is_file()
    ]
    if set(disk_slugs) != set(toc_slugs):
        _fail("Index Pages table must match sidebar pages exactly.")
        errors += 1
    if not toc_slugs:
        _fail("Index Pages table is empty.")
        errors += 1
    else:
        if toc_slugs[0] != "executive-summary":
            _fail('First page must be Executive summary (slug executive-summary).')
            errors += 1
        if toc_slugs[-1] != "conclusion":
            _fail('Last page must be Conclusion.')
            errors += 1
        for slug in toc_slugs[1:-1]:
            if not re.match(r"^claim-\d+", slug):
                _fail(f'Claim page slug must start with "claim-" (got "{slug}").')
                errors += 1

    exec_path = root / "pages" / "executive-summary" / "page.md"
    if not exec_path.is_file():
        _fail('Missing page "Executive summary".')
        errors += 1
    else:
        exec_cells = _parse_cells(exec_path.read_text(encoding="utf-8"))
        if not any(
            c["type"] == "markdown"
            and c.get("pinned")
            and (c.get("title") or "").strip().lower() == "executive summary"
            for c in exec_cells
        ):
            _fail('Need pinned markdown cell titled "Executive summary".')
            errors += 1
        if not any(
            c["type"] == "figure"
            and c.get("pinned")
            and "poster_embed.html" in (c.get("body") or "")
            for c in exec_cells
        ):
            _fail("Need pinned figure cell referencing poster_embed.html.")
            errors += 1

    concl_path = root / "pages" / "conclusion" / "page.md"
    if not concl_path.is_file():
        _fail('Missing page "Conclusion".')
        errors += 1
    else:
        concl_cells = _parse_cells(concl_path.read_text(encoding="utf-8"))
        if not any(c["type"] == "artifact" for c in concl_cells):
            _fail("Conclusion needs a reproduction bundle artifact cell.")
            errors += 1

    combined = ""
    has_artifact = False
    for page_file in root.rglob("*.md"):
        text = page_file.read_text(encoding="utf-8")
        combined += text + "\n"
        has_artifact = has_artifact or any(
            c["type"] == "artifact" for c in _parse_cells(text)
        )
    has_hub = bool(HUB_URL_RE.search(combined))
    has_github = bool(GITHUB_REPO_RE.search(combined))
    if not has_hub and not has_github and not has_artifact:
        _warn(
            "No Hugging Face URLs, GitHub repos, or artifact cells found yet."
        )
        warnings += 1

    if errors:
        return 1
    if warnings:
        print("Logbook validation passed with warnings.")
    else:
        print("Logbook validation passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate ICML reproduction logbook structure."
    )
    parser.add_argument(
        "--space",
        dest="space_id",
        help="Publish target username/repro-<slugified-title>",
    )
    args = parser.parse_args()

    if _HAS_TRACKIO and hasattr(lb, "validate_logbook"):
        try:
            return validate_with_trackio(args.space_id)
        except lb.LogbookError as exc:
            _fail(str(exc))
            return 1

    if not _HAS_TRACKIO:
        _warn("trackio not installed; using built-in validator.")
    else:
        _warn("trackio is older than validate API; using built-in validator.")
    return validate_standalone(args.space_id)


if __name__ == "__main__":
    raise SystemExit(main())
