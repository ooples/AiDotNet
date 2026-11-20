import { defineConfig } from '@playwright/test';
import base from './playwright.config';

// Run via npm run test:preview. It builds with isolated fixture configuration;
// real-backend tests deliberately remain in the main config.
const executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
const launchOptions = executablePath ? { executablePath } : undefined;
export default defineConfig({
  ...base,
  // Isolated browser contexts and fixture backends share no customer records.
  workers: process.env.CI ? 2 : 4,
  reporter: [['list'], ['html', { outputFolder: 'playwright-report/preview', open: 'never' }]],
  outputDir: 'test-results/preview',
  use: { ...base.use, launchOptions },
  projects: base.projects?.filter((p) => ['anon-desktop', 'anon-mobile', 'auth-anon'].includes(p.name!))
    .map((p) => ({ ...p, use: { ...p.use, launchOptions } })),
  webServer: {
    command: 'npm run preview -- --host 127.0.0.1 --port 4322',
    url: 'http://127.0.0.1:4322/',
    reuseExistingServer: false,
    timeout: 30_000,
  },
});
