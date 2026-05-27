import { test, expect, type Page } from '@playwright/test';

// Block third-party media + analytics so the smoke test only measures the
// site itself. The homepage embeds a YouTube iframe for background music;
// we don't want a YT outage to fail our deploy checks.
async function blockThirdParty(page: Page) {
  await page.route('**/*', (route) => {
    const url = route.request().url();
    if (
      url.includes('youtube.com') ||
      url.includes('youtube-nocookie.com') ||
      url.includes('googlevideo.com') ||
      url.includes('doubleclick.net') ||
      url.includes('googletagmanager.com')
    ) {
      return route.abort();
    }
    return route.continue();
  });
}

test.describe('portfolio smoke', () => {
  test.beforeEach(async ({ page }) => {
    await blockThirdParty(page);
  });

  test('homepage loads and renders the hero', async ({ page }) => {
    const response = await page.goto('./', { waitUntil: 'domcontentloaded' });
    expect(response?.status(), 'homepage HTTP status').toBeLessThan(400);
    await expect(page).toHaveTitle(/Louis Vladimir Antoine/i);
    await expect(page.locator('.hero-title')).toHaveText(/Louis Vladimir Antoine/i);
  });

  test('homepage has accessibility landmarks (skip-link + main)', async ({ page }) => {
    // Regression guard for the a11y pass: skip-link must be present and
    // point at a real #main-content landmark. If a future redesign removes
    // either, this test will catch it.
    await page.goto('./', { waitUntil: 'domcontentloaded' });
    const skipLink = page.locator('a.skip-link');
    await expect(skipLink).toHaveAttribute('href', '#main-content');
    await expect(page.locator('main#main-content')).toBeAttached();
  });

  // Hub pages — if any of these 404 the front-page nav is broken.
  const hubs: Array<{ path: string; titleMatch: RegExp; selector?: string }> = [
    { path: 'electronics-projects.html', titleMatch: /electronics/i },
    { path: 'photonics-projects.html',   titleMatch: /photonics/i },
    { path: 'innovation-projects.html',  titleMatch: /innovation/i },
    { path: 'projects/ml-ai/ml-projects.html', titleMatch: /(machine learning|ml|ai)/i },
  ];

  for (const hub of hubs) {
    test(`hub page resolves: ${hub.path}`, async ({ page }) => {
      const response = await page.goto(hub.path, { waitUntil: 'domcontentloaded' });
      expect(response?.status(), `${hub.path} HTTP status`).toBeLessThan(400);
      await expect(page).toHaveTitle(hub.titleMatch);
    });
  }

  test('community demo carries the demo banner', async ({ page }) => {
    const response = await page.goto('pages/community.html', {
      waitUntil: 'domcontentloaded',
    });
    expect(response?.status()).toBeLessThan(400);
    // Phase 1 hygiene added a sticky "Demo only." banner. If a future
    // change wires a real backend the banner should go; until then it
    // must be visible so visitors aren't misled.
    await expect(page.getByRole('note', { name: /demo notice/i })).toContainText(
      /demo only/i,
    );
  });

  test('PWA manifest is served with the correct MIME', async ({ request }) => {
    const response = await request.get('manifest.json');
    expect(response.status()).toBe(200);
    const ctype = response.headers()['content-type'] ?? '';
    expect(ctype).toMatch(/application\/(manifest\+)?json/);
  });

  // Downloadable project bundles live in the v1-downloads GitHub Release,
  // not in the repo. Four project pages link to pcm-gst-research.zip and
  // one to riscv_soc_paper_assets.zip; if either 302→404s the buttons are
  // broken. HEAD follows the GitHub → S3 redirect with maxRedirects.
  const releaseAssets = [
    'pcm-gst-research.zip',
    'riscv_soc_paper_assets.zip',
  ];
  for (const asset of releaseAssets) {
    test(`release asset still resolves: ${asset}`, async ({ request }) => {
      const url = `https://github.com/alovladi007/louis-antoine-portfolio/releases/download/v1-downloads/${asset}`;
      const response = await request.head(url, { maxRedirects: 5 });
      expect(response.status(), `${asset} HTTP status`).toBeLessThan(400);
    });
  }
});
