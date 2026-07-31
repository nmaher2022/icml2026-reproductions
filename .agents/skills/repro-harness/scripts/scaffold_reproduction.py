#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Scaffold an icml2026-reproductions-style GitHub monorepo.

Two subcommands:
  init  Bootstrap a brand-new monorepo skeleton (README.md with an "## Index"
        table, LICENSE). Safe to run against an existing repo too -- only
        fills in what's missing, never overwrites existing content.
  add   Add one finished reproduction: create `<slug>-<orid>/` with the
        standard subfolders + a README stub, and insert a row into the
        nearest ancestor README.md's "## Index" table.

Never runs `git commit` or `git push` -- only `git add`s the files it wrote and
prints the exact commit command for a human (or the calling agent, with
explicit user confirmation) to run.

Column-schema agnostic: the Index table's own header row defines what columns
exist. `add` reads them from the table and requires a `--field "Col=Value"`
for each one (a Folder-ish column is auto-filled if omitted), so this works
against any repo's table shape, not just this project's specific six columns.

Usage:
  uv run scaffold_reproduction.py init --repo-path . --title "My Paper Reproductions" --author "Jane Doe"
  uv run scaffold_reproduction.py add --slug gluon-lmo-optimizers --orid IelAHU5MVz \\
      --field "Paper=From Muon to Gluon: ..." \\
      --field 'OpenReview=[`IelAHU5MVz`](https://openreview.net/forum?id=IelAHU5MVz)' \\
      --field "Claims reproduced=6" \\
      --field "Verdict=mixed" \\
      --field 'Trackio Logbook=[HF Space](https://huggingface.co/spaces/...)'
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

INDEX_HEADING = "## Index"
DEFAULT_COLUMNS = [
    "Paper",
    "OpenReview",
    "Claims reproduced",
    "Verdict",
    "Folder",
    "Trackio Logbook",
]

MIT_LICENSE_TEMPLATE = """MIT License

Copyright (c) {year} {author}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

FOLDER_README_STUB = """# Reproduction bundle — {title}

<!-- TODO: one-line description, upstream code link (if any) + base commit, paper links (OpenReview + arXiv). -->

Paper: OpenReview [{orid}](https://openreview.net/forum?id={orid}){arxiv_line}
<!-- TODO: link the published Trackio logbook once it exists. -->

## Verdict

<!-- TODO: one row per claim. Use exactly VERIFIED / TOY-VERIFIED / REFUTED / BLOCKED
     (see the repro-harness skill's verdict_checklist.md) and state the scale run. -->

| Claim | Outcome |
|---|---|
| 1. ... | ... |

## Contents
<!-- TODO: list what's under patches/, configurations/, logs/, results/ -->

## Rerun
<!-- TODO: exact commands to reproduce this bundle from a clean checkout. -->
"""


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def find_repo_root(start: Path) -> Path | None:
    """Walk up from `start` looking for a README.md containing an Index table (like a .git search)."""
    start = start.resolve()
    for d in (start, *start.parents):
        readme = d / "README.md"
        if readme.is_file():
            try:
                text = readme.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if INDEX_HEADING in text:
                return d
    return None


def parse_index_table(readme_text: str) -> tuple[list[str], int, int]:
    """Return (header_columns, first_data_row_line_idx, table_end_line_idx) for the Index table."""
    lines = readme_text.splitlines()
    try:
        heading_idx = next(i for i, l in enumerate(lines) if l.strip() == INDEX_HEADING)
    except StopIteration:
        raise SystemExit(f"error: no '{INDEX_HEADING}' heading found in README.md")

    i = heading_idx + 1
    while i < len(lines) and not lines[i].strip().startswith("|"):
        i += 1
    if i >= len(lines):
        raise SystemExit(f"error: no markdown table found under '{INDEX_HEADING}'")

    header_line = lines[i]
    sep_idx = i + 1
    if sep_idx >= len(lines) or not re.match(r"^\|[\s:|-]+\|\s*$", lines[sep_idx].strip()):
        raise SystemExit("error: Index table is missing its '|---|---|...' separator row")

    columns = [c.strip() for c in header_line.strip().strip("|").split("|")]

    j = sep_idx + 1
    while j < len(lines) and lines[j].strip().startswith("|"):
        j += 1
    return columns, sep_idx + 1, j


def cmd_init(args: argparse.Namespace) -> int:
    repo_path = Path(args.repo_path).resolve()
    repo_path.mkdir(parents=True, exist_ok=True)

    is_repo = (repo_path / ".git").is_dir()
    if not is_repo:
        _run(["git", "init"], cwd=repo_path)
        print(f"git-initialized {repo_path}")

    columns = args.columns.split(",") if args.columns else DEFAULT_COLUMNS
    columns = [c.strip() for c in columns]

    readme_path = repo_path / "README.md"
    touched: list[str] = []

    if not readme_path.is_file():
        title = args.title or repo_path.name
        header = " | ".join(columns)
        sep = " | ".join("---" for _ in columns)
        readme_path.write_text(
            f"# {title}\n\n"
            f"<!-- TODO: one-paragraph description of what this repo collects. -->\n\n"
            f"{INDEX_HEADING}\n\n"
            f"| {header} |\n"
            f"| {sep} |\n\n"
            f"## License\n\n"
            f"MIT (see [`LICENSE`](LICENSE)).\n",
            encoding="utf-8",
        )
        touched.append("README.md (created)")
    else:
        text = readme_path.read_text(encoding="utf-8")
        if INDEX_HEADING not in text:
            header = " | ".join(columns)
            sep = " | ".join("---" for _ in columns)
            insertion = f"\n{INDEX_HEADING}\n\n| {header} |\n| {sep} |\n"
            license_idx = text.find("\n## License")
            if license_idx != -1:
                text = text[:license_idx] + insertion + text[license_idx:]
            else:
                text = text.rstrip("\n") + "\n" + insertion
            readme_path.write_text(text, encoding="utf-8")
            touched.append("README.md (added Index table to existing file)")
        else:
            print("README.md already has an Index table -- left untouched.")

    license_path = repo_path / "LICENSE"
    if not license_path.is_file() and args.license == "mit":
        author = args.author or "<AUTHOR>"
        if author == "<AUTHOR>":
            print("warning: no --author given; LICENSE written with a '<AUTHOR>' placeholder -- fix before committing.")
        license_path.write_text(
            MIT_LICENSE_TEMPLATE.format(year=args.year or date.today().year, author=author),
            encoding="utf-8",
        )
        touched.append("LICENSE (created)")
    elif not license_path.is_file():
        print(f"note: --license={args.license} not auto-generated; add LICENSE manually.")

    if touched:
        _run(["git", "add", *[
            "README.md" if t.startswith("README.md") else "LICENSE" for t in touched
        ]], cwd=repo_path)
        print("Staged: " + ", ".join(touched))
        print(f"\nNext: review the diff, then commit yourself, e.g.\n"
              f"  cd {repo_path} && git commit -m \"Initialize reproductions monorepo\"")
    else:
        print("Nothing to do -- repo already has a README with an Index table and a LICENSE.")

    print(f"\nrepo root: {repo_path}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    repo_root = (
        Path(args.repo_path).resolve() if args.repo_path else find_repo_root(Path.cwd())
    )
    if repo_root is None:
        print(
            "error: couldn't find a README.md with an '## Index' table starting from the "
            "current directory (or --repo-path). Run `init` first, or pass --repo-path "
            "pointing at an existing monorepo.",
            file=sys.stderr,
        )
        return 1

    readme_path = repo_root / "README.md"
    if not readme_path.is_file():
        print(f"error: {readme_path} does not exist.", file=sys.stderr)
        return 1

    folder_name = args.folder_name or f"{args.slug}-{args.orid}"
    folder = repo_root / folder_name

    readme_text = readme_path.read_text(encoding="utf-8")
    columns, data_start, data_end = parse_index_table(readme_text)

    fields: dict[str, str] = {}
    for raw in args.field or []:
        if "=" not in raw:
            print(f"error: --field must be 'Column=Value', got: {raw!r}", file=sys.stderr)
            return 1
        key, _, value = raw.partition("=")
        fields[key.strip()] = value.strip()

    folder_link = f"[`{folder_name}/`]({folder_name}/)"
    for col in columns:
        if "folder" in col.lower() and col not in fields:
            fields[col] = folder_link

    missing = [c for c in columns if c not in fields]
    if missing:
        print(
            "error: this Index table has columns " + ", ".join(f"'{c}'" for c in columns) +
            " -- missing --field for: " + ", ".join(f"'{c}'" for c in missing),
            file=sys.stderr,
        )
        return 1

    new_row = "| " + " | ".join(fields[c] for c in columns) + " |"

    if args.dry_run:
        print("[dry-run] would insert row into README.md Index table:")
        print("  " + new_row)
    else:
        lines = readme_text.splitlines()
        lines.insert(data_end, new_row)
        readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Inserted Index row into {readme_path}")

    created: list[str] = []
    if not args.dry_run:
        for sub in ([] if args.no_subdirs else ["patches", "configurations", "logs", "results"]):
            d = folder / sub
            if not d.is_dir():
                d.mkdir(parents=True, exist_ok=True)
                created.append(str(d.relative_to(repo_root)))

        folder_readme = folder / "README.md"
        if not folder_readme.is_file() and not args.skip_readme_stub:
            arxiv_line = f"\nPaper also on arXiv: [{args.arxiv}](https://arxiv.org/abs/{args.arxiv}).\n" if args.arxiv else "\n"
            folder_readme.write_text(
                FOLDER_README_STUB.format(
                    title=args.title or args.slug, orid=args.orid, arxiv_line=arxiv_line
                ),
                encoding="utf-8",
            )
            created.append(str(folder_readme.relative_to(repo_root)))

        _run(["git", "add", str(folder.relative_to(repo_root)), "README.md"], cwd=repo_root)

    print(f"\nfolder: {folder}")
    if created:
        print("created: " + ", ".join(created))
    if not args.dry_run:
        print(
            f"\nStaged. Fill in {folder_name}/README.md's TODOs (verdict table, contents, rerun "
            f"instructions -- see the repro-harness skill's verdict_checklist.md for the vocabulary), "
            f"then review and commit yourself, e.g.\n"
            f"  cd {repo_root} && git commit -m \"Add <Paper Title> ({args.orid}) reproduction\""
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="bootstrap a new monorepo skeleton (README + LICENSE)")
    p_init.add_argument("--repo-path", default=".", help="path to the repo root (created if missing)")
    p_init.add_argument("--title", help="repo title for a newly created README.md")
    p_init.add_argument("--columns", help="comma-separated Index table columns (default: the 6-column ICML convention)")
    p_init.add_argument("--author", help="copyright holder for LICENSE")
    p_init.add_argument("--year", type=int, help="copyright year (default: current year)")
    p_init.add_argument("--license", default="mit", choices=["mit", "none"], help="license template to write")
    p_init.set_defaults(func=cmd_init)

    p_add = sub.add_parser("add", help="add one finished reproduction: folder + Index row")
    p_add.add_argument("--repo-path", help="repo root (default: search upward from cwd for a README with an Index table)")
    p_add.add_argument("--slug", required=True, help="short paper slug, e.g. gluon-lmo-optimizers")
    p_add.add_argument("--orid", required=True, help="OpenReview forum id, e.g. IelAHU5MVz")
    p_add.add_argument("--folder-name", help="override the folder name (default: <slug>-<orid>)")
    p_add.add_argument("--title", help="paper title, used in the folder README stub")
    p_add.add_argument("--arxiv", help="arXiv id, if any, used in the folder README stub")
    p_add.add_argument(
        "--field", action="append",
        help="'Column=Value' for the Index table, repeatable -- one per column in that table's header "
             "(a Folder-ish column is auto-filled from --slug/--orid if omitted)",
    )
    p_add.add_argument("--no-subdirs", action="store_true", help="skip creating patches/configurations/logs/results/")
    p_add.add_argument("--skip-readme-stub", action="store_true", help="don't write a folder README.md stub")
    p_add.add_argument("--dry-run", action="store_true", help="print what would change without writing/staging")
    p_add.set_defaults(func=cmd_add)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
