import { defineConfig } from '@playwright/test';
import base from './playwright.config';
import { DEPLOYMENT_STORAGE_STATE } from './tests/e2e/helpers/auth-state';

const executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
const launchOptions = executablePath ? { executablePath } : undefined;
const selected = ['anon-desktop', 'anon-mobile', 'auth-anon', 'user.setup', 'admin.setup', 'user-auth', 'admin-auth', 'journeys'];

export default defineConfig({
  ...base,
  reporter: [['./tests/e2e/helpers/redact-reporter.ts'], ['list'], ['html', { outputFolder: 'playwright-report/deployed', open: 'never' }]],
  outputDir: 'test-results/deployed',
  use: { ...base.use, launchOptions, storageState: DEPLOYMENT_STORAGE_STATE, trace: 'off' },
  projects: [
    { name: 'deployment.setup', testMatch: /deployment\.setup\.ts/, use: { storageState: { cookies: [], origins: [] }, screenshot: 'off' } },
    ...base.projects!.filter((project) => selected.includes(project.name!)).map((project) => ({
      ...project,
      dependencies: [...(project.dependencies ?? []), 'deployment.setup'],
      testIgnore: [...(Array.isArray(project.testIgnore) ? project.testIgnore : project.testIgnore ? [project.testIgnore] : []),
        '**/billing-preview.spec.ts', '**/crawler.spec.ts'],
      use: { ...project.use, launchOptions, trace: 'off' as const },
    })),
  ],
});
