import { defineConfig, devices } from '@playwright/test';

// Storage-state files produced by the auth.setup.ts projects. Each
// auth-required test project `dependencies`-chains on the matching setup
// project, so Playwright runs the login sequence once per suite run
// instead of re-logging in for every spec file.
const USER_STORAGE_STATE = 'tests/e2e/.auth/user.json';
const ADMIN_STORAGE_STATE = 'tests/e2e/.auth/admin.json';

// Production-mutating safety: the auth-backed specs perform real writes
// (API key create/revoke, profile update, license issuance). Rather than
// defaulting to https://www.aidotnet.dev and turning a bare
// `npx playwright test` into a production-data-mutating command, we
// require PLAYWRIGHT_BASE_URL to be set explicitly. CI sets it in the
// post-deploy smoke job; local devs set it to their preview URL or a
// dev server. No default == no accidental prod writes.
const baseURL = process.env.PLAYWRIGHT_BASE_URL;
if (!baseURL) {
  throw new Error(
    'PLAYWRIGHT_BASE_URL is required. Set it to the environment you want ' +
    'to test against — e.g. http://localhost:4321 for a local dev server, ' +
    'a Vercel preview URL for a PR, or https://www.aidotnet.dev for ' +
    'production smoke (only do this deliberately — the auth-backed specs ' +
    'mutate real accounts).',
  );
}

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  // One worker always. DB-touching tests (admin license issue, API-key
  // create/revoke, profile save-and-restore in settings.spec.ts)
  // read-modify-write the same seeded accounts, and parallel workers
  // would race on the at-most-one-active-license index + interleave
  // the restore step of settings.spec.ts with another run's write.
  //
  // The parallelism risk is the same locally and in CI — the single
  // user.setup / admin.setup projects emit one storageState each and
  // every auth-backed spec attaches that same session. A second worker
  // would submit a revoke on the same key the first worker just
  // created. Cap globally so the default `npx playwright test`
  // invocation can't hit that race by accident.
  workers: 1,
  // dot = one-line-per-test on CI; GitHub annotations are layered on top
  // for failure output. The stock 'github' reporter produces confusing
  // empty-body annotations when every test passes.
  reporter: process.env.CI
    ? [['dot'], ['github'], ['html', { open: 'never' }]]
    : 'html',
  use: {
    baseURL,
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
      // Base on Pixel 5's real device config (correct isMobile +
      // hasTouch + user agent) rather than spreading Desktop Chrome +
      // overriding only the viewport. Code paths that gate on
      // `window.matchMedia('(hover: none)')`, touch events, or the
      // mobile UA regex won't fire on a desktop-UA fake-narrow window.
      use: { ...devices['Pixel 5'] },
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
