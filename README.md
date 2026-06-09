# Louis Antoine — Engineering Portfolio

[![Deploy](https://github.com/alovladi007/louis-antoine-portfolio/actions/workflows/static.yml/badge.svg?branch=main)](https://github.com/alovladi007/louis-antoine-portfolio/actions/workflows/static.yml)
[![Smoke tests](https://github.com/alovladi007/louis-antoine-portfolio/actions/workflows/smoke.yml/badge.svg?branch=main)](https://github.com/alovladi007/louis-antoine-portfolio/actions/workflows/smoke.yml)
[![Lighthouse](https://github.com/alovladi007/louis-antoine-portfolio/actions/workflows/lighthouse.yml/badge.svg?branch=main)](https://github.com/alovladi007/louis-antoine-portfolio/actions/workflows/lighthouse.yml)
[![Sitemap drift](https://github.com/alovladi007/louis-antoine-portfolio/actions/workflows/sitemap.yml/badge.svg?branch=main)](https://github.com/alovladi007/louis-antoine-portfolio/actions/workflows/sitemap.yml)
[![Live site](https://img.shields.io/badge/live-alovladi007.github.io%2Flouis--antoine--portfolio-2563eb?logo=githubpages&logoColor=white)](https://alovladi007.github.io/louis-antoine-portfolio/)
[![License](https://img.shields.io/badge/license-MIT%20code%20%2F%20RR%20content-informational)](LICENSE)

A multi-section engineering portfolio covering semiconductor process work, photonics, quantum optics, machine learning, autonomy, and adjacent applied-physics topics.

The site is a static HTML/CSS/JS app deployed to GitHub Pages, with **940 interactive simulators, tools, and project write-ups** organized by domain.

**Live site →** <https://alovladi007.github.io/louis-antoine-portfolio/>

## Section hubs

- Homepage — [`index.html`](index.html)
- Electronics — [`electronics-projects.html`](electronics-projects.html)
- Photonics — [`photonics-projects.html`](photonics-projects.html)
- Machine Learning & Data Science — [`projects/ml-ai/ml-projects.html`](projects/ml-ai/ml-projects.html)
- Innovation — [`innovation-projects.html`](innovation-projects.html)

## Repository layout

```
.
├── index.html                       # Homepage
├── electronics-projects.html        # Hub: electronics
├── photonics-projects.html          # Hub: photonics
├── innovation-projects.html         # Hub: innovation
│
├── projects/                        # All project pages, grouped by domain
│   ├── cmp/                         # Chemical-mechanical polishing (~23)
│   ├── semiconductor/               # Process, lithography, deposition, etching (~240)
│   ├── photonics/                   # Metamaterials & integrated photonics (~14)
│   ├── comms/                       # Communications & RF (~51)
│   ├── navigation/                  # GNSS, autonomy, undersea (~70)
│   ├── computer-vision/             # CV, AR/VR, self-driving (~44)
│   ├── ml-ai/                       # ML/DL/RL/generative (~136)
│   ├── medical/                     # Biomedical, imaging, clinical (~46)
│   ├── quantum/                     # CV-QKD, squeezing, nonlinear (~34)
│   ├── power-electronics/           # GaN, RISC-V SoC, hardware (~33)
│   ├── finance/                     # Trading, backtesting, risk (~16)
│   ├── iot/                         # IoT, distributed, cluster (~34)
│   ├── climate/                     # Climate/energy/sensors (~19)
│   └── misc/                        # Themed multi-domain projects (~31)
│
├── about/                           # Bio: ASML, coursework, research, skills, certs
├── blog/                            # Blog (CV-quantum, GaN-SiC, transformers)
├── research/                        # Research hubs + algorithm simulators
├── demos/                           # Standalone demos and showcase pages
├── pages/                           # Site utilities (FAQ, sitemap, legal, dashboards)
├── tests/                           # Dev/debug pages (not deployed)
│
├── assets/
│   ├── images/                      # Photos, illustrations, backgrounds
│   ├── pdfs/                        # Resume, diplomas, certifications
│   └── data/                        # Datasets, captions
│
├── backend/                         # Flask app for self-driving demo (not deployed)
├── docs/                            # Status notes, project guides (not deployed)
├── scripts/                         # Helper scripts (not deployed; see below)
│   ├── generate-sitemap.py          # Walks the file tree → sitemap.xml
│   └── inject-meta-tags.py          # Idempotent meta + OG tag injector
│
├── sitemap.xml                      # 939 URLs, regenerated on every push
├── robots.txt                       # Allows crawlers, points to sitemap
│
├── styles.css, styles-advanced.css  # Site-wide CSS
├── script.js, script-advanced.js,
├── enhance.js, i18n.js              # Site-wide JS
├── service-worker.js, manifest.json # PWA config (must stay at root)
├── package.json, requirements.txt   # Dev tooling
├── LICENSE                          # MIT for code / all-rights-reserved for content
└── .nojekyll                        # Disable Jekyll on GitHub Pages
```

## Tech stack

- **Runtime libs** (CDN-loaded, no bundler) — Three.js, GSAP, AOS, Swiper, Typed.js, particles.js, Font Awesome
- **PWA** — `manifest.json` + `service-worker.js`
- **Internationalization** — [`i18n.js`](i18n.js) (EN / FR / ES)
- **Backend demo** — Flask + OpenCV DNN ([`backend/`](backend/))

The site ships as plain HTML; there is no build step. All runtime libraries load from CDN inside the HTML files. `package.json` exists only for the `npm run serve` helper.

## Local development

```bash
# Static-server preview
python3 -m http.server 8765
# then open http://localhost:8765/
```

## Self-driving demo (Flask backend)

[`backend/app.py`](backend/app.py) is a standalone Flask app implementing the self-driving vision page — real-time lane detection (Canny + probabilistic Hough), object detection (YOLOv3-tiny via OpenCV DNN), and collision risk scoring.

```bash
pip install -r requirements.txt
python backend/app.py
# then open frontend/index.html in a browser
```

YOLO weights/config/labels go in `backend/model/` — these are gitignored; fetch from <https://pjreddie.com/darknet/yolo/>.

## Deployment

Push to `main` → [.github/workflows/static.yml](.github/workflows/static.yml) assembles a trimmed `_site/` directory (excludes `backend/`, `scripts/`, `docs/`, `tests/`, internal `*.md`, dev tooling, and VCS metadata) and uploads it to GitHub Pages.

CSS is minified in-flight via [csso](https://github.com/css/csso) inside the deploy step — repo source stays readable, only the deployed copy is compressed.

## CI workflows

| Workflow | Trigger | What it checks |
|---|---|---|
| [`static.yml`](.github/workflows/static.yml)  | push to main | Assemble + minify + deploy to GitHub Pages |
| [`smoke.yml`](.github/workflows/smoke.yml)    | post-deploy + daily cron | Playwright smoke tests + axe-core a11y against the live site |
| [`lighthouse.yml`](.github/workflows/lighthouse.yml) | post-deploy + daily cron | Lighthouse CI on the deployed homepage; perf, a11y, SEO, best-practices floors |
| [`sitemap.yml`](.github/workflows/sitemap.yml) | every push/PR | Regenerates `sitemap.xml` and fails the build if it drifts from the file tree |

## SEO + social sharing

| What | Status |
|---|---|
| `sitemap.xml` | 939 URLs (homepage, 3 top hubs, 6 sub-hubs, 6 blog posts, 781 project pages, 119 about/about-related, 29 demos). Auto-generated from the file tree. |
| `robots.txt` | Allow all, points to the sitemap |
| `meta description` | 940/940 pages (100%) |
| `og:title`, `og:description`, `og:image`, `og:url`, `og:site_name`, `og:type`, `twitter:card` | 940/940 pages — LinkedIn / Slack / X / Discord all render proper link previews |
| Schema.org `Person` JSON-LD | Homepage only (the rest are project pages, not personal pages) |
| Open Graph image | [`assets/og-image.png`](assets/og-image.png) — custom branded card |

The `sitemap.yml` workflow guards against forgotten regenerations: if you add a new project page and forget to re-run `python3 scripts/generate-sitemap.py`, CI fails with a clear "sitemap.xml is stale" message and the exact command to run.

The meta-tag injector ([`scripts/inject-meta-tags.py`](scripts/inject-meta-tags.py)) is **idempotent** and **non-destructive**: it only adds tags that are missing. Pages with hand-curated meta tags (e.g. the [PQC SE project page](projects/power-electronics/pqc-riscv-system.html) with its custom description and og:title) are preserved verbatim — the script just fills in the gaps the author didn't write.

## Recovery

Every reorganization is committed in small, revertible commits:

```bash
git log --oneline | head -20            # find the commit
git revert <commit-sha>                  # safely undo it
```

Renames are preserved (`git log --follow path/to/file`).

## License & attribution

See [LICENSE](LICENSE). In short:

- **Code** (`*.py`, `*.js`, `*.ts`, `*.css`, `*.sv`, build files, HTML scaffolding) — MIT.
- **Personal content** (photos, diplomas, resume PDFs, biographical text, employer affiliations) — all rights reserved.

Project pages credit upstream papers and reference implementations where applicable.
