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
