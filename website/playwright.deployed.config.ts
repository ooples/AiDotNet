import { defineConfig } from '@playwright/test';
import base from './playwright.config';

const executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
const launchOptions = executablePath ? { executablePath } : undefined;
const selected = ['anon-desktop', 'anon-mobile', 'auth-anon', 'user.setup', 'admin.setup', 'user-auth', 'admin-auth', 'journeys'];

export default defineConfig({
  ...base,
  reporter: [['list'], ['html', { outputFolder: 'playwright-report/deployed', open: 'never' }]],
  outputDir: 'test-results/deployed',
  use: { ...base.use, launchOptions, storageState: 'tests/e2e/.auth/deployment.json', trace: 'off' },
  projects: [
    { name: 'deployment.setup', testMatch: /deployment\.setup\.ts/, use: { storageState: { cookies: [], origins: [] } } },
    ...base.projects!.filter((project) => selected.includes(project.name!)).map((project) => ({
      ...project,
      dependencies: [...(project.dependencies ?? []), 'deployment.setup'],
      testIgnore: [...(Array.isArray(project.testIgnore) ? project.testIgnore : project.testIgnore ? [project.testIgnore] : []),
        '**/billing-preview.spec.ts', '**/crawler.spec.ts'],
      use: { ...project.use, launchOptions, trace: 'off' as const },
    })),
  ],
});
