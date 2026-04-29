# Louis Antoine — Engineering Portfolio

The source for [alovladi007.github.io/louis-antoine-portfolio](https://alovladi007.github.io/louis-antoine-portfolio/) — a multi-section portfolio covering semiconductor process engineering, photonics, quantum optics, machine learning, autonomy, and adjacent applied-physics work.

The site is a static HTML/CSS/JS app deployed via GitHub Pages, with hundreds of interactive simulators, tools, and project write-ups organized by domain.

## Live site

- **Homepage** — [`index.html`](index.html)
- **Electronics projects** — [`electronics-projects.html`](electronics-projects.html)
- **Photonics projects** — [`photonics-projects.html`](photonics-projects.html)
- **Machine Learning & Data Science** — [`machine-learning-projects.html`](machine-learning-projects.html)
- **Innovation projects** — [`innovation-projects.html`](innovation-projects.html)

## Repository layout

```
.
├── index.html                       # Homepage
├── electronics-projects.html        # Hub: electronics
├── photonics-projects.html          # Hub: photonics
├── machine-learning-projects.html   # Hub: ML/DS
├── innovation-projects.html         # Hub: innovation
│
├── projects/                        # All project pages, grouped by domain
│   ├── cmp/                         # Chemical-mechanical polishing (23)
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
├── tests/                           # Dev/debug pages
│
├── assets/
│   ├── images/                      # Photos, illustrations, backgrounds
│   ├── pdfs/                        # Resume, diplomas, certifications
│   └── data/                        # Datasets, captions
│
├── docs/                            # Status notes, READMEs, project guides, notebooks
├── scripts/                         # Helper scripts (.py, .sh, .mac, .jsl, .sql)
├── archives/                        # Old project bundles (.zip, .tar.gz)
├── logs/                            # Server log files
│
├── styles.css, styles-advanced.css  # Site-wide CSS
├── script.js, script-advanced.js, enhance.js, i18n.js  # Site-wide JS
├── service-worker.js, manifest.json # PWA config (must stay at root)
├── package.json, requirements.txt   # Dependencies
└── .nojekyll                        # Disable Jekyll on GitHub Pages
```

## Tech stack

- **Build**: Vite (configured via [`package.json`](package.json))
- **3D / visualization**: Three.js, Plotly.js
- **Animation**: GSAP, Lottie-web, AOS, Particles.js
- **ML in-browser**: TensorFlow.js
- **PWA**: `manifest.json` + `service-worker.js`
- **Internationalization**: [`i18n.js`](i18n.js) (EN / FR / ES)

## Local development

```bash
# Static-server preview (no build step required)
python3 -m http.server 8765
# then open http://localhost:8765/

# Vite dev server (with HMR)
npm install
npm run dev

# Production build
npm run build
```

## Self-driving demo (Flask backend)

Inside [`backend/`](backend/) is a standalone Flask app implementing the self-driving vision project page — real-time lane detection (Canny + probabilistic Hough), object detection (YOLOv3-tiny via OpenCV DNN), and collision risk scoring.

```bash
pip install -r requirements.txt
python backend/app.py
# then open frontend/index.html in a browser
```

YOLO weights/config/labels go in `backend/model/` — see [`backend/`](backend/) for sources.

## Recovery

Every reorganization in this repo's history is committed in small, revertible commits. To roll back any single migration:

```bash
git log --oneline | head -20            # find the commit
git revert <commit-sha>                  # safely undo it
```

Project content moves through git history are preserved with rename detection (`git log --follow path/to/file`).

## License & attribution

Personal portfolio. Project pages credit upstream papers and reference implementations where applicable.
