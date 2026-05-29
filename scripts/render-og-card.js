/**
 * render-og-card.js
 *
 * Renders assets/og/og-card.html to a 1200x630 PNG at
 * assets/images/og-card.png. Requires Playwright (already in
 * tests/smoke as a devDependency).
 *
 * Usage from repo root:
 *     node scripts/render-og-card.js
 *
 * The script starts a tiny static server on a random port, navigates a
 * headless Chromium to the template, sets the viewport to the OG spec
 * size, and screenshots. Network fonts (Font Awesome from CDN) are
 * given a chance to load via a small networkidle wait.
 */

const fs = require('fs');
const http = require('http');
const path = require('path');
const { chromium } = require('playwright');

const ROOT = path.resolve(__dirname, '..');
const TEMPLATE = '/assets/og/og-card.html';
const OUT = path.join(ROOT, 'assets', 'images', 'og-card.png');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css':  'text/css; charset=utf-8',
  '.js':   'application/javascript; charset=utf-8',
  '.png':  'image/png',
  '.jpg':  'image/jpeg',
  '.svg':  'image/svg+xml',
  '.json': 'application/json',
  '.ico':  'image/x-icon',
};

function staticServer(rootDir) {
  return http.createServer((req, res) => {
    const safePath = path.normalize(decodeURIComponent(req.url.split('?')[0]));
    const file = path.join(rootDir, safePath);
    if (!file.startsWith(rootDir)) {
      res.writeHead(403); res.end('Forbidden'); return;
    }
    fs.stat(file, (err, stat) => {
      if (err || !stat.isFile()) {
        res.writeHead(404); res.end('Not found'); return;
      }
      res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
      fs.createReadStream(file).pipe(res);
    });
  });
}

(async () => {
  const server = staticServer(ROOT);
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  const url = `http://127.0.0.1:${port}${TEMPLATE}`;
  console.log(`Static server: ${url}`);

  const browser = await chromium.launch();
  try {
    const ctx = await browser.newContext({
      viewport: { width: 1200, height: 630 },
      // deviceScaleFactor: 1 — OG image must stay under ~1 MB to render
      // reliably on Twitter / X. 2x doubled the file size to 1.8 MB,
      // which is above the safe threshold. The spec-recommended 1200x630
      // at 1x renders sharply enough for the unfurl preview.
      deviceScaleFactor: 1,
    });
    const page = await ctx.newPage();
    await page.goto(url, { waitUntil: 'networkidle', timeout: 30_000 });

    // Give web fonts (Font Awesome) one more beat to settle so the icons
    // aren't rendered as boxes in the PNG.
    await page.waitForTimeout(800);

    await fs.promises.mkdir(path.dirname(OUT), { recursive: true });
    await page.screenshot({
      path: OUT,
      type: 'png',
      clip: { x: 0, y: 0, width: 1200, height: 630 },
      omitBackground: false,
    });
    const sz = (await fs.promises.stat(OUT)).size;
    console.log(`Wrote ${OUT} (${(sz / 1024).toFixed(1)} KB)`);
  } finally {
    await browser.close();
    server.close();
  }
})();
