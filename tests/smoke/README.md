# Smoke tests

Minimal Playwright suite that verifies the live GitHub Pages deploy at
<https://alovladi007.github.io/louis-antoine-portfolio/> is healthy.

What it checks:

- Homepage returns < 400 and renders the hero (`Louis Vladimir Antoine`).
- Four hub pages (electronics, photonics, innovation, ML/AI) resolve and
  carry the right title.
- The Community Hub still carries the "Demo only" banner introduced in
  the Phase 1 hygiene pass.
- `manifest.json` is served with a JSON MIME type (PWA install requirement).
- `projects/pcm-gst-research.zip` still resolves — it's wired as a download
  button on four project pages and must not 404.

## Run locally

```bash
cd tests/smoke
npm install
npm run install-browsers   # one-time
npm test                   # hits the live site by default
```

To run against a local server instead:

```bash
SMOKE_BASE_URL=http://localhost:8765 npm test
```

## Run in CI

GitHub Actions runs the suite:

- on every push to `main` (after the Pages deploy completes),
- on a daily schedule (catches CDN regressions and link rot).

See [.github/workflows/smoke.yml](../../.github/workflows/smoke.yml).

## What this is not

This is a deploy-health check, not an end-to-end functional test. It does
not click through every project, it does not exercise the YouTube background
music, and it does not validate visual layout. Add targeted tests next to
the feature you're changing when you need richer coverage.
