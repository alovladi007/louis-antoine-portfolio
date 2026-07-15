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

# Root-level directories to skip entirely.
#
# CRITICAL: this MUST stay in sync with the rsync exclude list in
# .github/workflows/static.yml. This script walks the SOURCE TREE, but the
# sitemap describes the DEPLOYED SITE. Any directory the deploy excludes but
# this script includes produces a sitemap entry that 404s in production —
# and because .github/workflows/sitemap.yml regenerates from the same tree,
# the drift check would happily re-add the bad entries on every run rather
# than catch them.
#
# That is exactly what happened with `docs/`: it is excluded from the deploy
# artifact but was absent here, so the sitemap advertised 5 URLs
# (docs/index.html, docs/additional-projects.html, docs/ml-projects.html,
# docs/self-driving-vision.html, docs/projects/self-driving-vision/index.html)
# that do not exist on the live site.
#
# These are matched at the ROOT only (see discover_html_files), mirroring the
# leading-slash anchoring in static.yml, so nested dirs that merely share a
# name (e.g. riscv-soc-files/docs) are still published.
SKIP_ROOT_DIRS = {
    ".git", ".github", ".claude", "node_modules", "_site",
    "scripts", "tests", "backend", "archives", "logs",
    "docs",              # excluded by static.yml — must not be sitemapped
    "git", "playwright-report", "test-results",
    # frontend/ DOES ship, but its only page is a meta-refresh redirect
    # stub linked from nowhere — indexing a redirect is bad SEO, so it is
    # deliberately omitted. (Allowed: sitemap ⊆ artifact, not equality.)
    "frontend",
}

# Directory names skipped at ANY depth (mirrors the unanchored static.yml
# patterns).
SKIP_ANY_DIRS = {"node_modules", "__pycache__"}

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
    """Walk REPO_ROOT and return sorted list of relative html paths.

    Invariant: the result must be a SUBSET of what static.yml publishes.
    Listing a URL the deploy excludes gives Google a guaranteed 404;
    listing fewer than we publish is harmless (a sitemap is "what should be
    indexed", not "every file that ships").

    Root-only skips are pruned at depth 0 only, mirroring the leading-slash
    anchoring in static.yml, so a nested directory that merely shares a name
    (riscv-soc-files/docs, medimetrics-nest/scripts) is still published AND
    sitemapped.
    """
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        at_root = Path(dirpath).resolve() == REPO_ROOT
        pruned = []
        for d in dirnames:
            if d.startswith("."):
                continue
            if d in SKIP_ANY_DIRS:
                continue
            if at_root and d in SKIP_ROOT_DIRS:
                continue
            pruned.append(d)
        dirnames[:] = pruned

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
