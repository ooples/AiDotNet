import { defineConfig, devices } from '@playwright/test';

// Storage-state files produced by the auth.setup.ts projects. Each
// auth-required test project `dependencies`-chains on the matching setup
// project, so Playwright runs the login sequence once per suite run
// instead of re-logging in for every spec file.
const USER_STORAGE_STATE = 'tests/e2e/.auth/user.json';
const ADMIN_STORAGE_STATE = 'tests/e2e/.auth/admin.json';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  // CI: 1 worker to keep DB-touching tests (admin license issue, API-key
  // create) deterministic — they read-modify-write the same rows and
  // parallel workers would race on the at-most-one-active-license index.
  workers: process.env.CI ? 1 : undefined,
  // dot = one-line-per-test on CI; GitHub annotations are layered on top
  // for failure output. The stock 'github' reporter produces confusing
  // empty-body annotations when every test passes.
  reporter: process.env.CI
    ? [['dot'], ['github'], ['html', { open: 'never' }]]
    : 'html',
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL || 'https://www.aidotnet.dev',
    headless: true,
    screenshot: 'only-on-failure',
    trace: 'on-first-retry',
    // Most tests run under Chromium; Firefox/WebKit surface real-world
    // differences (notably webcrypto semantics and cookie handling)
    // but at 3x the CI runtime. Stick to Chromium for the baseline and
    // add cross-browser explicitly for critical paths if needed.
    ...devices['Desktop Chrome'],
  },
  projects: [
    // ---------- Anonymous projects (existing test suite) ----------
    {
      name: 'anon-desktop',
      testIgnore: ['**/user/**', '**/admin/**', '**/auth/**'],
      use: { viewport: { width: 1440, height: 900 } },
    },
    {
      name: 'anon-mobile',
      testIgnore: ['**/user/**', '**/admin/**', '**/auth/**'],
      use: { viewport: { width: 390, height: 844 } },
    },

    // ---------- Auth setup projects ----------
    //
    // These run first, log into the site via supabase.auth.signInWithPassword,
    // and persist the resulting Supabase auth token to a storageState file.
    // Downstream auth-required projects attach that storage state so tests
    // start already signed-in. Keeps the auth cost O(1) per full run
    // instead of O(n) per test.
    {
      name: 'user.setup',
      testMatch: /user\.setup\.ts/,
    },
    {
      name: 'admin.setup',
      testMatch: /admin\.setup\.ts/,
    },

    // ---------- Authenticated projects ----------
    {
      name: 'user-auth',
      testDir: './tests/e2e/user',
      use: {
        viewport: { width: 1440, height: 900 },
        storageState: USER_STORAGE_STATE,
      },
      dependencies: ['user.setup'],
    },
    {
      name: 'admin-auth',
      testDir: './tests/e2e/admin',
      use: {
        viewport: { width: 1440, height: 900 },
        storageState: ADMIN_STORAGE_STATE,
      },
      dependencies: ['admin.setup'],
    },
  ],
});

// Exports so test files can reference the storage paths when they need
// to write a fresh session (e.g., token expiry tests) without guessing
// the filename.
export { USER_STORAGE_STATE, ADMIN_STORAGE_STATE };
