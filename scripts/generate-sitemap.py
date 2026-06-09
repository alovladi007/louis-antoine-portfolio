#!/usr/bin/env python3
"""Generate sitemap.xml from the filesystem.

Walks the repository for *.html files, classifies each by path, and emits
a sitemap.xml with appropriate priority and changefreq for each tier:

  Homepage         priority 1.0  changefreq weekly
  Top-level hubs   priority 0.8  changefreq monthly
  Sub-hub indexes  priority 0.7  changefreq monthly
  Blog posts       priority 0.7  changefreq monthly
  Project pages    priority 0.6  changefreq monthly
  About pages      priority 0.5  changefreq yearly
  Demos            priority 0.4  changefreq yearly

Pages excluded:
  - 404.html (deliberately not indexed)
  - Anything under git/, node_modules/, .git/, scripts/
  - *-backup-* or *-archive-* (snapshots, not canonical pages)

Run:
  python3 scripts/generate-sitemap.py
  # writes sitemap.xml at repo root

Re-run whenever new project pages are added.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SITE_URL = "https://alovladi007.github.io/louis-antoine-portfolio"
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "sitemap.xml"

# Paths to skip entirely
SKIP_DIRS = {
    ".git", ".github", "node_modules", "scripts", "tests",
    "git", "backend", "frontend", "playwright-report", "test-results",
}

SKIP_NAMES = {"404.html"}


def classify(relpath: str) -> tuple[float, str]:
    """Return (priority, changefreq) for a given relative HTML path."""
    parts = relpath.split("/")

    # Homepage
    if relpath == "index.html":
        return (1.0, "weekly")

    # Top-level hub aggregators
    top_hubs = {
        "electronics-projects.html",
        "photonics-projects.html",
        "innovation-projects.html",
    }
    if relpath in top_hubs:
        return (0.8, "monthly")

    # Sub-hub indexes (projects/<hub>/<hub>-projects.html)
    if len(parts) == 3 and parts[0] == "projects" and parts[-1].endswith("-projects.html"):
        return (0.7, "monthly")

    # Blog posts
    if parts[0] == "blog":
        return (0.7, "monthly")

    # About + experience pages
    if parts[0] == "about":
        return (0.5, "yearly")

    # Demos
    if parts[0] == "demos":
        return (0.4, "yearly")

    # Anything else under projects/ — individual project pages
    if parts[0] == "projects":
        return (0.6, "monthly")

    # Top-level fallback for stray pages
    return (0.5, "yearly")


def _pending_changes() -> set:
    """Set of relative paths with staged or unstaged changes — files that
    will be part of the NEXT commit. Used to break a chicken-and-egg
    where `git log -1` returns the OLD lastmod for a file you're about
    to commit, which then drifts the moment the commit lands and CI
    runs against HEAD."""
    try:
        out = subprocess.run(
            ["git", "diff", "HEAD", "--name-only"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        )
        return set(out.stdout.split())
    except (subprocess.SubprocessError, OSError):
        return set()


_PENDING: set | None = None


def last_modified(path: Path) -> str | None:
    """Return ISO date of last git commit touching `path`, or today if
    the file has uncommitted changes (so a local run agrees with the
    post-commit CI run)."""
    global _PENDING
    if _PENDING is None:
        _PENDING = _pending_changes()

    rel = str(path.relative_to(REPO_ROOT))
    if rel in _PENDING:
        from datetime import date
        return date.today().isoformat()

    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%cs", "--", rel],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        date = out.stdout.strip()
        return date if date else None
    except (subprocess.SubprocessError, OSError):
        return None


def discover_html_files() -> list[str]:
    """Walk REPO_ROOT and return sorted list of relative html paths."""
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        # Prune skip-dirs in place
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for f in filenames:
            if not f.endswith(".html"):
                continue
            if f in SKIP_NAMES:
                continue
            rel = os.path.relpath(os.path.join(dirpath, f), REPO_ROOT)
            # Normalize separators to forward-slash (URL form)
            rel = rel.replace(os.sep, "/")
            found.append(rel)
    found.sort()
    return found


def main() -> None:
    files = discover_html_files()
    print(f"Discovered {len(files)} HTML pages")

    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']

    for rel in files:
        priority, changefreq = classify(rel)
        url = f"{SITE_URL}/{rel}" if rel != "index.html" else f"{SITE_URL}/"
        lastmod = last_modified(REPO_ROOT / rel)

        lines.append("  <url>")
        lines.append(f"    <loc>{url}</loc>")
        if lastmod:
            lines.append(f"    <lastmod>{lastmod}</lastmod>")
        lines.append(f"    <changefreq>{changefreq}</changefreq>")
        lines.append(f"    <priority>{priority:.1f}</priority>")
        lines.append("  </url>")

    lines.append("</urlset>")
    lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT.relative_to(REPO_ROOT)} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
