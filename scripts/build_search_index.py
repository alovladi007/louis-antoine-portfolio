#!/usr/bin/env python3
"""
Build a search index from every project/demo/research/blog/about/pages
HTML file in the repo. Output: assets/data/search-index.json.

Each entry:
  {
    "url":         "projects/comms/mmwave-rf-complete.html",
    "title":       "...",          # from <title> or <h1>
    "category":    "comms",        # parent folder, human-friendly
    "section":     "projects",     # top-level section
    "description": "First useful paragraph...",
    "tags":        ["...", "..."]  # auto-derived from filename + folder
  }

Re-run this whenever new pages are added so the search index stays fresh.
"""
import os, re, json, html
from pathlib import Path
from typing import Optional, List

ROOT = Path("/Users/vladimirantoine/New Portfolio/louis-antoine-portfolio")
OUTPUT = ROOT / "assets/data/search-index.json"

# Folders to crawl (top-level)
SECTIONS = {
    "projects": "projects",
    "demos":    "demos",
    "research": "research",
    "blog":     "blog",
    "about":    "about",
    "pages":    "pages",
}

# Skip: redirect stubs, the search page itself, util pages
SKIP_FILES = {
    "pages/portfolio-search.html",
    "pages/sitemap.html",
    "pages/coming-soon.html",
    "pages/clear-cache.html",
}
SKIP_FOLDERS = {"_archive", "_internal"}

# Friendly category names
CATEGORY_NAMES = {
    "cmp":              "Chemical-Mechanical Polishing",
    "comms":            "Communications & RF",
    "computer-vision":  "Computer Vision & AR/VR",
    "climate":          "Climate & Energy",
    "finance":          "Finance & Trading",
    "iot":              "IoT & Distributed Systems",
    "medical":          "Medical & Biomedical",
    "misc":             "Misc & Multi-domain",
    "ml-ai":            "Machine Learning & AI",
    "navigation":       "Navigation & Autonomy",
    "photonics":        "Photonics & Metamaterials",
    "power-electronics":"Power Electronics",
    "quantum":          "Quantum",
    "semiconductor":    "Semiconductor Process",
    "self-driving-vision": "Self-Driving Vision",
    "pcm-gst-research": "PCM-GST Research",
}

# Regexes - simple, fast, good enough
RE_TITLE       = re.compile(r"<title>([^<]*)</title>", re.I | re.S)
RE_DESC        = re.compile(r'<meta\s+name=["\']description["\']\s+content=["\']([^"\']+)["\']', re.I)
RE_H1          = re.compile(r"<h1[^>]*>(.*?)</h1>", re.I | re.S)
RE_H2          = re.compile(r"<h2[^>]*>(.*?)</h2>", re.I | re.S)
RE_P           = re.compile(r"<p[^>]*>(.*?)</p>", re.I | re.S)
RE_TAG_STRIP   = re.compile(r"<[^>]+>")
RE_WS          = re.compile(r"\s+")
RE_REFRESH     = re.compile(r'http-equiv=["\']refresh["\']', re.I)


def clean(s: str, limit: int = 220) -> str:
    if not s:
        return ""
    s = RE_TAG_STRIP.sub(" ", s)
    s = html.unescape(s)
    s = RE_WS.sub(" ", s).strip()
    if len(s) > limit:
        s = s[: limit - 1].rstrip() + "…"
    return s


def derive_tags(filename: str, category: str) -> List[str]:
    base = filename.replace(".html", "")
    parts = re.split(r"[-_]", base)
    tags = [p for p in parts if p and len(p) > 2 and p.lower() not in {"the", "and", "for"}]
    if category and category not in tags:
        tags.insert(0, category)
    return tags[:8]


def is_redirect_stub(content: str) -> bool:
    head = content[:1500]
    return bool(RE_REFRESH.search(head)) and len(content) < 2000


def extract(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
    except Exception:
        return None

    if is_redirect_stub(content):
        return None  # skip redirect-only pages

    rel_url = str(path.relative_to(ROOT))
    if rel_url in SKIP_FILES:
        return None

    parts = rel_url.split(os.sep)
    section = parts[0]
    # category for projects/<cat>/foo.html, otherwise section name
    if section == "projects" and len(parts) >= 3:
        category_slug = parts[1]
    else:
        category_slug = section
    category = CATEGORY_NAMES.get(category_slug, category_slug.replace("-", " ").title())

    # Title: prefer <title>, fall back to first <h1>
    title = ""
    m = RE_TITLE.search(content)
    if m:
        title = clean(m.group(1), 180)
    if not title:
        m = RE_H1.search(content)
        if m:
            title = clean(m.group(1), 180)
    if not title:
        title = parts[-1].replace(".html", "").replace("-", " ").title()

    # Strip the common " - Louis Antoine" / " - Technical Insights" suffixes
    for sep in [" - Louis Antoine Portfolio", " - Louis Antoine", " | Louis Antoine"]:
        if title.endswith(sep):
            title = title[: -len(sep)]

    # Description: <meta>, then first non-empty <p>, then first <h2>
    description = ""
    m = RE_DESC.search(content)
    if m:
        description = clean(m.group(1), 240)
    if not description:
        for m in RE_P.finditer(content):
            cand = clean(m.group(1), 240)
            if len(cand) >= 30:
                description = cand
                break
    if not description:
        m = RE_H2.search(content)
        if m:
            description = clean(m.group(1), 240)

    return {
        "url":         rel_url,
        "title":       title,
        "section":     section,
        "category":    category,
        "categorySlug": category_slug,
        "description": description,
        "tags":        derive_tags(parts[-1], category_slug),
    }


def main():
    index = []
    for section in SECTIONS:
        section_dir = ROOT / section
        if not section_dir.is_dir():
            continue
        for path in sorted(section_dir.rglob("*.html")):
            if any(p in SKIP_FOLDERS for p in path.parts):
                continue
            entry = extract(path)
            if entry:
                index.append(entry)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as fh:
        json.dump({"generated": "auto", "count": len(index), "items": index}, fh, ensure_ascii=False, separators=(",", ":"))

    # Stats
    by_section = {}
    by_category = {}
    for entry in index:
        by_section[entry["section"]] = by_section.get(entry["section"], 0) + 1
        by_category[entry["category"]] = by_category.get(entry["category"], 0) + 1

    print(f"Indexed {len(index)} pages")
    print()
    print("=== By section ===")
    for k, n in sorted(by_section.items(), key=lambda x: -x[1]):
        print(f"  {n:5d}  {k}")
    print()
    print("=== By category (top 20) ===")
    for k, n in sorted(by_category.items(), key=lambda x: -x[1])[:20]:
        print(f"  {n:5d}  {k}")
    print()
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
