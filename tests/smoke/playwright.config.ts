import { defineConfig, devices } from '@playwright/test';

// Must end with a trailing slash. Without it, page.goto('foo.html')
// would resolve against the parent path and 404. With it, the project
// prefix /louis-antoine-portfolio/ is treated as a directory.
const BASE_URL =
  process.env.SMOKE_BASE_URL ??
  'https://alovladi007.github.io/louis-antoine-portfolio/';

export default defineConfig({
  testDir: '.',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 2 : undefined,
  reporter: process.env.CI ? [['github'], ['list']] : 'list',
  use: {
    baseURL: BASE_URL,
    trace: 'retain-on-failure',
    // The homepage autoplays background music via the YouTube embed.
    // Block third-party media so the test isn't dependent on YT being up.
    extraHTTPHeaders: { 'User-Agent': 'louis-antoine-portfolio-smoke/1.0' },
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});
