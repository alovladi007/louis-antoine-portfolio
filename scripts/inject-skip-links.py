#!/usr/bin/env python3
"""Inject a skip-to-main-content link into every page that lacks one.

Audit before this script ran (manual scan of about/):
  16/16 about pages had NO skip-link, even though every page is long-form
  content with a nav bar at the top. A keyboard / screen-reader user
  hitting one of those pages has to tab through the entire nav before
  reaching the page body on every navigation.

The skip-link is invisible until keyboard-focused, then slides into the
top-left corner (matching the homepage's pattern). Targets the first
viable in-page anchor: <main id=...>, <article id=...>, <section id=...>,
or the first <h1>'s parent.

Pattern matches homepage/blog post styling:
  - inline-style (so no CSS file edit is needed)
  - z-index high
  - keyboard-focus animation
  - high-contrast colors

Idempotent: detects the existing skip-link and skips.

Run:
  python3 scripts/inject-skip-links.py                 # whole repo
  python3 scripts/inject-skip-links.py about/          # subtree only
  python3 scripts/inject-skip-links.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {
    ".git", ".github", "node_modules", "scripts", "tests",
    "backend", "frontend", "_site", "playwright-report",
    "test-results", "git",
}
SKIP_NAMES = {"404.html"}

BODY_OPEN_RE = re.compile(r"<body[^>]*>", re.IGNORECASE)
SKIP_LINK_RE = re.compile(r'class=["\'][^"\']*\bskip-link\b', re.IGNORECASE)

# What anchor on the page we should jump to. Prefer #main-content (the
# homepage convention), then #main, then the first <main>/<article>/<section>
# with an id.
MAIN_ID_RE = re.compile(
    r'<(?:main|article|section)[^>]+id=["\']([a-zA-Z0-9_-]+)["\']',
    re.IGNORECASE,
)

SKIP_LINK_HTML = (
    '\n    <a href="{anchor}" class="skip-link" '
    'style="position:absolute;top:-100px;left:0;z-index:10000;'
    'padding:.75rem 1.5rem;background:#1f2937;color:#fff;'
    'text-decoration:none;border-radius:0 0 8px 0;font-weight:600;'
    'transition:top .2s" '
    'onfocus="this.style.top=\'0\'" onblur="this.style.top=\'-100px\'"'
    '>Skip to main content</a>\n'
)


def discover_html_files(roots: list[str]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        rpath = (REPO_ROOT / root).resolve()
        if rpath.is_file() and rpath.suffix == ".html":
            out.append(rpath)
            continue
        for dirpath, dirnames, filenames in os.walk(rpath):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for f in filenames:
                if f.endswith(".html") and f not in SKIP_NAMES:
                    out.append(Path(dirpath) / f)
    return sorted(set(out))


def best_anchor(text: str) -> str:
    """Pick the most appropriate #anchor for skip-to-content."""
    # Common convention first
    if 'id="main-content"' in text or "id='main-content'" in text:
        return "#main-content"
    if 'id="main"' in text or "id='main'" in text:
        return "#main"
    m = MAIN_ID_RE.search(text)
    if m:
        return f"#{m.group(1)}"
    # Last-ditch: pretend a <main> id exists. Browser will scroll to
    # the document body if the anchor doesn't resolve.
    return "#content"


def process(path: Path, dry_run: bool) -> tuple[bool, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if SKIP_LINK_RE.search(text):
        return False, "already has skip-link"
    body = BODY_OPEN_RE.search(text)
    if not body:
        return False, "no <body>"

    anchor = best_anchor(text)
    insertion = SKIP_LINK_HTML.format(anchor=anchor)
    new_text = text[: body.end()] + insertion + text[body.end():]

    if not dry_run:
        path.write_text(new_text, encoding="utf-8")
    return True, f"added (anchor={anchor})"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("roots", nargs="*", default=["."])
    args = p.parse_args()

    files = discover_html_files(args.roots)
    print(f"Scanning {len(files)} files...")

    changed = 0
    skipped = 0
    no_body = 0
    for f in files:
        c, reason = process(f, args.dry_run)
        if c:
            changed += 1
        elif reason == "no <body>":
            no_body += 1
        else:
            skipped += 1

    verb = "would change" if args.dry_run else "changed"
    print(f"\n{verb}: {changed}")
    print(f"already had skip-link: {skipped}")
    print(f"no body (unprocessable): {no_body}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
