#!/usr/bin/env python3
"""Scaffold a canonical ICML 2026 reproduction Trackio logbook."""

from __future__ import annotations

import argparse
import json
import re
import sys


def repro_slug_from_title(title: str, *, max_len: int = 96) -> str:
    clean = re.sub(r"^Reproduction:\s*", "", title.strip(), flags=re.I).strip()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", clean.lower()).strip("-") or "page"
    base = f"repro-{slug}"
    if len(base) <= max_len:
        return base
    return base[:max_len].rstrip("-")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the canonical ICML reproduction logbook template."
    )
    parser.add_argument("--title", required=True, help="Paper title")
    parser.add_argument("--orid", required=True, help="OpenReview forum id")
    parser.add_argument("--arxiv", help="arXiv id (for HF paper link)")
    parser.add_argument("--openreview-url", help="Full OpenReview paper URL")
    parser.add_argument(
        "--hf-indexed",
        action="store_true",
        help="Paper is indexed on huggingface.co/papers",
    )
    parser.add_argument(
        "--claims-json",
        help='JSON array of claim strings, e.g. \'["headline accuracy"]\'',
    )
    parser.add_argument(
        "--username",
        help="HF username for metadata.space_id (username/repro-<slug>)",
    )
    args = parser.parse_args()

    try:
        from trackio import logbook as lb
    except ImportError:
        print(
            "trackio is required. Install with: uv pip install --upgrade trackio",
            file=sys.stderr,
        )
        return 1

    if hasattr(lb, "scaffold_icml_logbook"):
        claims = None
        if args.claims_json:
            try:
                claims = json.loads(args.claims_json)
            except json.JSONDecodeError as exc:
                print(f"Invalid --claims-json: {exc}", file=sys.stderr)
                return 1
            if not isinstance(claims, list):
                print("--claims-json must be a JSON array.", file=sys.stderr)
                return 1
        try:
            info = lb.scaffold_icml_logbook(
                title=args.title,
                orid=args.orid,
                claims=claims,
                arxiv_id=args.arxiv,
                openreview_url=args.openreview_url,
                hf_indexed=args.hf_indexed,
                username=args.username,
            )
        except lb.LogbookError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        _print_next_steps(info)
        return 0

    claims = None
    if args.claims_json:
        try:
            claims = json.loads(args.claims_json)
        except json.JSONDecodeError as exc:
            print(f"Invalid --claims-json: {exc}", file=sys.stderr)
            return 1
        if not isinstance(claims, list):
            print("--claims-json must be a JSON array.", file=sys.stderr)
            return 1

    paper_title = re.sub(r"^Reproduction:\s*", "", args.title.strip(), flags=re.I).strip()
    logbook_title = f"Reproduction: {paper_title}"
    slug = repro_slug_from_title(paper_title)
    space_id = f"{args.username}/{slug}" if args.username else None
    orid = args.orid.strip()
    if not orid:
        print("OpenReview id (--orid) is required.", file=sys.stderr)
        return 1

    if lb.find_project_dir() and (
        lb.logbook_root(lb.find_project_dir()) / "pages" / "index.md"
    ).exists():
        print("A logbook already exists in this directory.", file=sys.stderr)
        return 1

    try:
        proj = lb.create_logbook(title=logbook_title, space_id=space_id)
        paper_link = (
            f"[HF paper page](https://huggingface.co/papers/{args.arxiv.strip()})"
            if args.hf_indexed and args.arxiv
            else f"[OpenReview paper]({(args.openreview_url or f'https://openreview.net/forum?id={orid}').strip()})"
        )
        claim_specs = []
        for i, claim in enumerate(claims or [], start=1):
            claim_text = str(claim).strip()
            if not claim_text:
                continue
            if not re.match(r"^Claim\s+\d+\s*:", claim_text, re.I):
                claim_text = f"Claim {i}: {claim_text}"
            claim_specs.append((claim_text, lb.ensure_page(proj, claim_text)))

        exec_slug = lb.ensure_page(proj, "Executive summary")
        concl_slug = lb.ensure_page(proj, "Conclusion")
        index_lines = [
            f"# {logbook_title}",
            "",
            paper_link,
            "",
            lb.TOC_HEADING,
            "",
            lb.TOC_HEADER,
            lb.TOC_SEP,
            f"| [Executive summary](#/{exec_slug}) |",
        ]
        for claim_title, claim_slug in claim_specs:
            index_lines.append(f"| [{claim_title}](#/{claim_slug}) |")
        index_lines += [f"| [Conclusion](#/{concl_slug}) |", ""]
        (lb._pages_dir(proj) / "index.md").write_text(
            "\n".join(index_lines), encoding="utf-8"
        )

        metadata = lb.read_metadata(proj)
        metadata["tags"] = ["icml2026-repro", f"paper-{orid}"]
        if args.arxiv:
            metadata["paper"] = {"arxiv_id": args.arxiv.strip()}
        if space_id:
            metadata["space_id"] = space_id
        lb.write_metadata(proj, metadata)

        summary_body = (
            "Write a 3–5 sentence outcome-first summary here.\n\n"
            "## Scope & cost\n\n"
            "| Item | Value |\n"
            "| --- | --- |\n"
            "| GPU / compute | |\n"
            "| Wall time | |\n"
            "| Feasibility | |\n"
        )
        lb.add_markdown_cell(proj, exec_slug, summary_body, title="Executive summary")
        summary_id = lb.last_cell_id(proj, page=exec_slug)
        if summary_id:
            lb.set_cell_pinned(proj, summary_id, pinned=True, page=exec_slug)

        lb.add_figure_cell(
            proj,
            exec_slug,
            html=(
                "<p>Build a reproduction poster with "
                '<a href="https://github.com/Chenruishuo/posterly">Chenruishuo/posterly</a> '
                "and replace this cell with <code>poster_embed.html</code>.</p>"
            ),
            title="Reproduction poster (poster_embed.html)",
        )
        poster_id = lb.last_cell_id(proj, page=exec_slug)
        if poster_id:
            lb.set_cell_pinned(proj, poster_id, pinned=True, page=exec_slug)

        lb.add_markdown_cell(
            proj,
            concl_slug,
            (
                "Add a reproduction bundle artifact cell here after running:\n\n"
                "```bash\n"
                f'trackio.log_artifact("./repro_{slug[6:]}/", name="repro-bundle", type="dataset")\n'
                f"trackio logbook cell artifact {slug}/repro-bundle:v0 "
                '--page "Conclusion" --title "Reproduction bundle" --type dataset\n'
                "```"
            ),
            title="Reproduction bundle",
        )
        for claim_title, claim_slug in claim_specs:
            lb.add_markdown_cell(
                proj,
                claim_slug,
                f"Document setup, runs, and results for **{claim_title}**.",
                title=claim_title,
            )
        lb.write_site_files(proj)
    except lb.LogbookError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    _print_next_steps(
        {
            "title": logbook_title,
            "slug": slug,
            "space_id": space_id or f"<username>/{slug}",
        }
    )
    return 0


def _print_next_steps(info: dict) -> None:
    print(f"Scaffolded logbook: {info['title']}")
    print(f"Publish slug: {info['slug']}")
    print(f"Publish target: {info['space_id']}")
    print(
        "\nNext steps:\n"
        "  1. Reproduce each claim (log commands, Hub assets, results)\n"
        "  2. Fill Executive summary + poster_embed.html (Chenruishuo/posterly)\n"
        "  3. Add reproduction bundle artifact on Conclusion\n"
        "  4. curl -sL …/validate_icml_logbook.py | python3 - --space "
        f"{info['space_id']}\n"
        f"  5. trackio logbook publish {info['space_id']}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
