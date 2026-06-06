#!/usr/bin/env python3
"""Inject meta description + Open Graph tags into every HTML page that's
missing them.

Audit before this script ran:
  about/        0/16  have meta description
  projects/*    ~4/930 have meta description
  OG image      0/930 have og:image

Each page's title is the source of truth for the description (the
authoritative summary the page author already wrote). We mirror it
into meta description + og:title + og:description, add a stable
og:image pointing at assets/og-image.png (the existing branded card),
and set og:type / og:url / twitter:card so links rendered in Slack,
LinkedIn, X, etc. get the proper card treatment.

Idempotent: re-running on a page that already has each tag is a no-op.
Doesn't touch pages that already define the tag (respects manual
overrides).

Run:
  python3 scripts/inject-meta-tags.py             # whole repo
  python3 scripts/inject-meta-tags.py --dry-run   # report what would change
  python3 scripts/inject-meta-tags.py projects/quantum/  # subtree only
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

SITE_URL = "https://alovladi007.github.io/louis-antoine-portfolio"
SITE_NAME = "Louis Vladimir Antoine — Engineering Portfolio"
DEFAULT_OG_IMAGE = f"{SITE_URL}/assets/og-image.png"

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories we never touch
SKIP_DIRS = {
    ".git", ".github", "node_modules", "scripts", "tests",
    "backend", "frontend", "_site", "playwright-report",
    "test-results", "git",
}

TITLE_RE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)
HEAD_OPEN_RE = re.compile(r"<head[^>]*>", re.IGNORECASE)

# Tag presence detectors (loose; we just need to know "is the tag defined")
META_DESC_RE = re.compile(r'<meta\s+[^>]*name=["\']description["\']', re.IGNORECASE)
OG_TITLE_RE = re.compile(r'<meta\s+[^>]*property=["\']og:title["\']', re.IGNORECASE)
OG_DESC_RE = re.compile(r'<meta\s+[^>]*property=["\']og:description["\']', re.IGNORECASE)
OG_IMAGE_RE = re.compile(r'<meta\s+[^>]*property=["\']og:image["\']', re.IGNORECASE)
OG_URL_RE = re.compile(r'<meta\s+[^>]*property=["\']og:url["\']', re.IGNORECASE)
OG_TYPE_RE = re.compile(r'<meta\s+[^>]*property=["\']og:type["\']', re.IGNORECASE)
TWITTER_CARD_RE = re.compile(r'<meta\s+[^>]*name=["\']twitter:card["\']', re.IGNORECASE)


def clean_title(raw: str) -> str:
    """Strip whitespace + collapse internal newlines + drop trailing ' | site'
    so we can append the site name uniformly."""
    t = re.sub(r"\s+", " ", raw).strip()
    # Drop common trailing site-name suffixes so we don't double up.
    for suffix in (
        " | Louis Vladimir Antoine — Engineering Portfolio",
        " | Louis Vladimir Antoine - Engineering Portfolio",
        " | Louis Antoine Portfolio",
        " | Louis Antoine",
    ):
        if t.endswith(suffix):
            t = t[: -len(suffix)]
            break
    return t


def description_for(title: str) -> str:
    """Generate a meta description from a page title. Title is the
    author's curated summary; we just append the portfolio context."""
    return f"{title} — part of the engineering portfolio of Louis Vladimir Antoine, covering hardware, semiconductors, photonics, and software systems."


def og_url_for(rel: str) -> str:
    if rel == "index.html":
        return f"{SITE_URL}/"
    return f"{SITE_URL}/{rel}"


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
                if f.endswith(".html"):
                    out.append(Path(dirpath) / f)
    return sorted(set(out))


def build_meta_block(title: str, rel: str) -> str:
    """Return the formatted meta-tag block to inject after <head>."""
    desc = description_for(title)
    url = og_url_for(rel)
    return (
        f'\n    <!-- Meta tags injected by scripts/inject-meta-tags.py -->\n'
        f'    <meta name="description" content="{html_escape(desc)}">\n'
        f'    <meta property="og:type" content="article">\n'
        f'    <meta property="og:title" content="{html_escape(title)}">\n'
        f'    <meta property="og:description" content="{html_escape(desc)}">\n'
        f'    <meta property="og:url" content="{url}">\n'
        f'    <meta property="og:image" content="{DEFAULT_OG_IMAGE}">\n'
        f'    <meta property="og:site_name" content="{SITE_NAME}">\n'
        f'    <meta name="twitter:card" content="summary_large_image">\n'
    )


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def process_file(path: Path, dry_run: bool) -> tuple[bool, str]:
    """Return (changed, reason)."""
    text = path.read_text(encoding="utf-8", errors="ignore")

    title_m = TITLE_RE.search(text)
    if not title_m:
        return False, "no <title>"

    title = clean_title(title_m.group(1))
    if not title:
        return False, "empty <title>"

    head_m = HEAD_OPEN_RE.search(text)
    if not head_m:
        return False, "no <head>"

    # Check which tags already exist; only inject the missing ones.
    has = {
        "description":  bool(META_DESC_RE.search(text)),
        "og:title":     bool(OG_TITLE_RE.search(text)),
        "og:desc":      bool(OG_DESC_RE.search(text)),
        "og:image":     bool(OG_IMAGE_RE.search(text)),
        "og:url":       bool(OG_URL_RE.search(text)),
        "og:type":      bool(OG_TYPE_RE.search(text)),
        "twitter":      bool(TWITTER_CARD_RE.search(text)),
    }
    if all(has.values()):
        return False, "all tags already present"

    rel = path.relative_to(REPO_ROOT).as_posix()
    desc = description_for(title)
    url = og_url_for(rel)

    additions: list[str] = []
    if not has["description"]:
        additions.append(f'    <meta name="description" content="{html_escape(desc)}">')
    if not has["og:type"]:
        additions.append('    <meta property="og:type" content="article">')
    if not has["og:title"]:
        additions.append(f'    <meta property="og:title" content="{html_escape(title)}">')
    if not has["og:desc"]:
        additions.append(f'    <meta property="og:description" content="{html_escape(desc)}">')
    if not has["og:url"]:
        additions.append(f'    <meta property="og:url" content="{url}">')
    if not has["og:image"]:
        additions.append(f'    <meta property="og:image" content="{DEFAULT_OG_IMAGE}">')
        additions.append(f'    <meta property="og:site_name" content="{SITE_NAME}">')
    if not has["twitter"]:
        additions.append('    <meta name="twitter:card" content="summary_large_image">')

    if not additions:
        return False, "nothing to add"

    block = (
        "\n    <!-- SEO + OG (auto-injected by scripts/inject-meta-tags.py) -->\n"
        + "\n".join(additions)
        + "\n"
    )

    insert_at = head_m.end()
    new_text = text[:insert_at] + block + text[insert_at:]

    if not dry_run:
        path.write_text(new_text, encoding="utf-8")

    added = [
        name for name, present in has.items() if not present
    ]
    return True, f"added: {', '.join(added)}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="report changes without writing")
    p.add_argument("roots", nargs="*", default=["."], help="paths to walk (default: repo root)")
    args = p.parse_args()

    files = discover_html_files(args.roots)
    print(f"Scanning {len(files)} files...")

    changed = 0
    skipped = 0
    no_title = 0
    for f in files:
        c, reason = process_file(f, args.dry_run)
        if c:
            changed += 1
        elif reason in {"no <title>", "empty <title>", "no <head>"}:
            no_title += 1
        else:
            skipped += 1

    verb = "would change" if args.dry_run else "changed"
    print(f"\n{verb}: {changed}")
    print(f"already complete: {skipped}")
    print(f"unprocessable (no title/head): {no_title}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
