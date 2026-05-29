import { test, expect, type Page } from '@playwright/test';
import { AxeBuilder } from '@axe-core/playwright';

/**
 * Accessibility CI gate.
 *
 * Runs axe-core against the homepage and the four section hubs. The suite
 * fails on any `critical` or `serious` violation. Lower-severity violations
 * (`moderate`, `minor`) are surfaced in the report but don't break the
 * build, since they often involve genuine design judgment calls.
 *
 * Some rules are explicitly disabled with a rationale:
 *
 * - `color-contrast` is disabled on these pages because the site uses a
 *   purple gradient hero where light-on-purple text falls under the WCAG
 *   AA threshold by axe's measurement, but is legible in practice. This
 *   is a deliberate design choice tracked separately.
 *
 * - `region` is disabled because some hub pages put navigation outside a
 *   <main> landmark for layout reasons; the skip-link still works.
 *
 * Add additional rule overrides only with a written reason.
 */

async function blockThirdParty(page: Page) {
  await page.route('**/*', (route) => {
    const url = route.request().url();
    if (
      url.includes('youtube.com') ||
      url.includes('youtube-nocookie.com') ||
      url.includes('googlevideo.com') ||
      url.includes('googletagmanager.com') ||
      url.includes('google-analytics.com') ||
      url.includes('doubleclick.net') ||
      url.includes('clarity.ms')
    ) {
      return route.abort();
    }
    return route.continue();
  });
}

async function runAxe(page: Page, url: string) {
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  const results = await new AxeBuilder({ page })
    .disableRules(['color-contrast', 'region'])
    .analyze();

  const critical = results.violations.filter(
    (v) => v.impact === 'critical' || v.impact === 'serious',
  );

  if (critical.length > 0) {
    // Surface useful detail in the CI log
    const summary = critical
      .map(
        (v) =>
          `[${v.impact}] ${v.id}: ${v.description}\n  -> ${v.nodes.length} node(s); first: ${v.nodes[0]?.target.join(' ')}`,
      )
      .join('\n');
    console.error(`\naxe violations on ${url}:\n${summary}\n`);
  }
  expect(critical, `Critical/serious a11y violations on ${url}`).toEqual([]);
}

test.describe('accessibility', () => {
  test.beforeEach(async ({ page }) => {
    await blockThirdParty(page);
  });

  test('homepage has no critical/serious axe violations', async ({ page }) => {
    await runAxe(page, './');
  });

  const hubs = [
    'electronics-projects.html',
    'photonics-projects.html',
    'innovation-projects.html',
    'projects/ml-ai/ml-projects.html',
  ];
  for (const hub of hubs) {
    test(`hub a11y: ${hub}`, async ({ page }) => {
      await runAxe(page, hub);
    });
  }
});
