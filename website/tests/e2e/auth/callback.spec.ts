import { test, expect } from '@playwright/test';

/**
 * Regression coverage for /auth/callback's URL-error handling. The
 * page reads three layers of failure (search-param error, hash-param
 * error, getSession exception) and surfaces them as actionable user
 * messages — these specs pin the surfaced text + hints so a regression
 * that drops a branch back to the silent-timeout state fails CI.
 *
 * No login required: every assertion exercises the unauthenticated
 * error-rendering path. Runs under the auth-anon project (see
 * playwright.config.ts) so it picks up no storageState.
 */

test.describe('/auth/callback URL-error handling', () => {
  test('search-param server_error renders fatal-error UI with provider-misconfig hint', async ({ page }) => {
    await page.goto('/auth/callback/?error=server_error&error_code=server_error&error_description=Invalid%20client%20secret');

    // Headline + technical details visible.
    await expect(page.getByText('Sign-in failed')).toBeVisible();
    await expect(page.getByText('error: server_error')).toBeVisible();
    await expect(page.getByText('error_code: server_error')).toBeVisible();
    // URLSearchParams already percent-decodes — assert the *decoded*
    // form to confirm we're not double-decoding.
    await expect(page.getByText('error_description: Invalid client secret')).toBeVisible();

    // Server-error hint mentions client-secret rotation / callback drift.
    await expect(page.getByText(/OAuth provider is misconfigured/)).toBeVisible();
  });

  test('search-param access_denied renders the user-cancelled hint', async ({ page }) => {
    await page.goto('/auth/callback/?error=access_denied&error_description=The%20user%20declined');

    await expect(page.getByText('Sign-in failed')).toBeVisible();
    await expect(page.getByText('error: access_denied')).toBeVisible();
    await expect(page.getByText(/declined to grant the requested permissions/)).toBeVisible();
  });

  test('error=unsupported_provider (no error_code) still surfaces disabled-provider hint', async ({ page }) => {
    // Pins the fix from PR #1258 review comment #4: the disabled-
    // provider hint must trigger when the field is on `error` not just
    // `error_code`. Some IdPs / Supabase configs split the two
    // inconsistently.
    await page.goto('/auth/callback/?error=unsupported_provider&error_description=GitHub%20provider%20not%20enabled');

    await expect(page.getByText(/sign-in provider is currently disabled/)).toBeVisible();
  });

  test('hash-param error takes precedence over search-param error', async ({ page }) => {
    // Implicit-flow errors arrive in the URL hash. parseAuthError()
    // gives them precedence over search params because the hash form
    // is more specific (carries error_code from the IdP).
    await page.goto('/auth/callback/?error=ignored#error=hash_error&error_description=Hash%20wins');

    await expect(page.getByText('error: hash_error')).toBeVisible();
    await expect(page.getByText('error_description: Hash wins')).toBeVisible();
  });

  test('per-source precedence: hash error_description wins, search description ignored', async ({ page }) => {
    // PR #1258 review: per-FIELD merge could splice an `error` from
    // hash with `error_description` from search. Per-SOURCE precedence
    // (current behaviour after the precedence-rewrite commit) MUST take
    // ALL fields from hash when hash carries an error, ignoring search
    // entirely. This spec pins that contract: a hash `error` with NO
    // hash `error_description` must NOT pick up the search's
    // `error_description`.
    await page.goto('/auth/callback/?error=search_err&error_description=search%20description#error=hash_err');

    await expect(page.getByText('error: hash_err')).toBeVisible();
    // Critically, the search-side description must NOT appear.
    await expect(page.getByText('error_description: search description')).toHaveCount(0);
  });
});

test.describe('/auth/callback non-URL failure paths', () => {
  // PR #1258 review #6/#7: every failure layer is supposed to leave a
  // console.error trace and surface the fatal-error UI. Previously the
  // tests only exercised URL-param failures; these specs cover the
  // session-exchange error path and the 10-second timeout path that
  // gated the silent-stall regression #1258 was created to fix.

  test('getSession() error renders "Session exchange failed."', async ({ page }) => {
    // Intercept the Supabase client lib that the page imports and force
    // its `auth.getSession()` to return an error tuple. Routing the
    // import to a small inline shim is sufficient — the page only uses
    // `auth.getSession()` and `auth.onAuthStateChange()` from the lib.
    await page.route('**/lib/supabase*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/javascript',
        body: `
          export const supabase = {
            auth: {
              getSession: async () => ({ data: { session: null }, error: { message: 'Forced session-exchange failure for test' } }),
              onAuthStateChange: () => ({ data: { subscription: { unsubscribe: () => {} } } }),
            },
          };
        `,
      });
    });

    const consoleErrors: string[] = [];
    page.on('console', m => { if (m.type() === 'error') consoleErrors.push(m.text()); });

    await page.goto('/auth/callback/');

    await expect(page.getByText('Session exchange failed.')).toBeVisible();
    await expect(page.getByText('Forced session-exchange failure for test')).toBeVisible();
    // PR #1258 promise: every failure path leaves a console.error.
    expect(consoleErrors.some(e => e.includes('getSession() error'))).toBeTruthy();
  });

  test('10-second timeout renders fatal-error UI when no SIGNED_IN event fires', async ({ page }) => {
    // Force getSession() to return an empty session AND the auth-state
    // listener to never fire — the only escape hatch is the 10-second
    // setTimeout fallback. Use page.clock.fastForward to skip the wall-
    // clock wait so the spec runs in <1s.
    await page.clock.install();
    await page.route('**/lib/supabase*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/javascript',
        body: `
          export const supabase = {
            auth: {
              getSession: async () => ({ data: { session: null }, error: null }),
              onAuthStateChange: () => ({ data: { subscription: { unsubscribe: () => {} } } }),
            },
          };
        `,
      });
    });

    const consoleErrors: string[] = [];
    page.on('console', m => { if (m.type() === 'error') consoleErrors.push(m.text()); });

    await page.goto('/auth/callback/');
    // Skip the 10-second hard timeout deterministically.
    await page.clock.fastForward(10_500);

    await expect(page.getByText('Sign-in did not complete in time.')).toBeVisible();
    await expect(page.getByText(/SIGNED_IN event within 10 seconds/)).toBeVisible();
    // PR #1258 review #2/#7: timeout path must console.error too.
    expect(consoleErrors.some(e => e.includes('timed out waiting for SIGNED_IN'))).toBeTruthy();
  });

  test('error_description with literal % does not crash the handler', async ({ page }) => {
    // PR #1258 review comment #1+#3: the previous code called
    // decodeURIComponent on a value URLSearchParams had already
    // decoded, so a description containing a literal '%' (e.g. an
    // encoded '%25') would throw URIError on the second decode and
    // fall into the generic "An unexpected error occurred" branch.
    // The fix removes the second decode entirely; this spec pins
    // that the literal '%' message renders intact.
    await page.goto('/auth/callback/?error=server_error&error_description=quota%2025%25%20exceeded');

    // After URLSearchParams.get(), the value is "quota 25% exceeded".
    await expect(page.getByText('error_description: quota 25% exceeded')).toBeVisible();
    // Critically, NOT the generic fallback message.
    await expect(page.getByText('An unexpected error occurred')).toHaveCount(0);
  });
});
