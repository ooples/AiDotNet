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

    // Headline + hint always visible (mapped, not raw IdP text).
    await expect(page.getByText('Sign-in failed')).toBeVisible();
    await expect(page.getByText(/OAuth provider is misconfigured/)).toBeVisible();
    // Per PR #1258 review #5: raw IdP text now lives behind a
    // <details> "Show technical details" disclosure to mitigate
    // phishing — the technical lines aren't in the rendered DOM
    // until the summary is clicked.
    await expect(page.getByText('Show technical details')).toBeVisible();
    await page.getByText('Show technical details').click();
    await expect(page.getByText('error: server_error')).toBeVisible();
    await expect(page.getByText('error_code: server_error')).toBeVisible();
    // URLSearchParams already percent-decodes — assert the *decoded*
    // form to confirm we're not double-decoding.
    await expect(page.getByText('error_description: Invalid client secret')).toBeVisible();
  });

  test('search-param access_denied renders the user-cancelled hint', async ({ page }) => {
    await page.goto('/auth/callback/?error=access_denied&error_description=The%20user%20declined');

    await expect(page.getByText('Sign-in failed')).toBeVisible();
    await expect(page.getByText(/declined to grant the requested permissions/)).toBeVisible();
    await page.getByText('Show technical details').click();
    await expect(page.getByText('error: access_denied')).toBeVisible();
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

    await page.getByText('Show technical details').click();
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

    await page.getByText('Show technical details').click();
    await expect(page.getByText('error: hash_err')).toBeVisible();
    // Critically, the search-side description must NOT appear.
    await expect(page.getByText('error_description: search description')).toHaveCount(0);
  });

  test('errorCode-only callback (no error field) still renders fatal-error UI', async ({ page }) => {
    // PR #1258 review comment #4: the parser now treats any of
    // `error`, `error_code`, `error_description` as a failure marker.
    // A callback that returns ONLY `error_code` (e.g. otp_expired
    // recovery flows) must NOT fall through to the 10-second timeout.
    await page.goto('/auth/callback/?error_code=otp_expired&error_description=Magic%20link%20expired');

    await expect(page.getByText('Sign-in failed')).toBeVisible();
    await page.getByText('Show technical details').click();
    await expect(page.getByText('error_code: otp_expired')).toBeVisible();
    await expect(page.getByText('error_description: Magic link expired')).toBeVisible();
  });

  test('errorCode === unsupported_provider variant surfaces the disabled-provider hint', async ({ page }) => {
    // PR #1258 review comment #8: the implementation accepts the
    // disabled-provider marker in EITHER `error` or `error_code` (some
    // IdPs split inconsistently). The existing spec covers the `error`
    // path; this one pins the `error_code` path so a regression that
    // drops one branch fails CI.
    await page.goto('/auth/callback/?error=oauth_failure&error_code=unsupported_provider&error_description=GitHub%20provider%20not%20enabled');

    await expect(page.getByText(/sign-in provider is currently disabled/)).toBeVisible();
  });

  test('error_description with HTML metacharacters is HTML-escaped (no XSS)', async ({ page }) => {
    // PR #1258 review comment #7: the safe() escaping helper is now a
    // security-critical guardrail because URL-controlled text is
    // rendered via innerHTML. Pin that `<` / `>` survive as escaped
    // entities — a regression that bypassed safe() would render the
    // <script> as actual markup.
    await page.goto('/auth/callback/?error=server_error&error_description=%3Cscript%3Ealert(1)%3C%2Fscript%3E');

    await page.getByText('Show technical details').click();
    // The literal escaped text appears verbatim in the DOM.
    await expect(page.getByText('<script>alert(1)</script>', { exact: false })).toBeVisible();
    // No alert dialog fired (script tag was escaped, not parsed).
    let dialogFired = false;
    page.on('dialog', () => { dialogFired = true; });
    // Give any (broken) script a tick to dispatch alert before asserting.
    await page.waitForTimeout(200);
    expect(dialogFired).toBe(false);
  });
});

test.describe('/auth/callback success paths', () => {
  // PR #1258 review comment #6: the e2e suite previously only covered
  // failure + timeout. The PR also changes both the immediate-session
  // redirect and the SIGNED_IN/clearTimeout race, so the success paths
  // need pinning too.

  test('immediate getSession() success redirects to default account page', async ({ page }) => {
    // Force getSession() to return a valid session synchronously so the
    // page hits the "data.session" branch (line ~189) and redirects
    // without waiting for an auth-state event.
    await page.route('**/lib/supabase*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/javascript',
        body: `
          export const supabase = {
            auth: {
              getSession: async () => ({ data: { session: { access_token: 't', user: { id: 'u' } } }, error: null }),
              onAuthStateChange: () => ({ data: { subscription: { unsubscribe: () => {} } } }),
            },
          };
        `,
      });
    });
    await page.goto('/auth/callback/');
    // Sanitized fallback redirect = ${base}account/. Wait for the URL
    // change rather than asserting on /account/ content (which would
    // require auth state we don't actually have in this stub).
    await expect(page).toHaveURL(/\/account\/?$/, { timeout: 5_000 });
  });

  test('redirect query param routing honours same-origin paths only', async ({ page }) => {
    // sanitizeRedirect must accept ?redirect=/foo/ but reject
    // protocol-relative (//evil.com) and absolute URLs.
    await page.route('**/lib/supabase*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/javascript',
        body: `
          export const supabase = {
            auth: {
              getSession: async () => ({ data: { session: { access_token: 't', user: { id: 'u' } } }, error: null }),
              onAuthStateChange: () => ({ data: { subscription: { unsubscribe: () => {} } } }),
            },
          };
        `,
      });
    });

    // Whitelisted same-origin path → respected.
    await page.goto('/auth/callback/?redirect=/settings/api-keys/');
    await expect(page).toHaveURL(/\/settings\/api-keys\/?$/, { timeout: 5_000 });
  });

  test('redirect with protocol-relative URL falls back to default account page (open-redirect guard)', async ({ page }) => {
    // PR #1258 review comment #1: open-redirect guard must reject
    // //evil.example which would otherwise resolve to https://evil.example.
    await page.route('**/lib/supabase*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/javascript',
        body: `
          export const supabase = {
            auth: {
              getSession: async () => ({ data: { session: { access_token: 't', user: { id: 'u' } } }, error: null }),
              onAuthStateChange: () => ({ data: { subscription: { unsubscribe: () => {} } } }),
            },
          };
        `,
      });
    });
    await page.goto('/auth/callback/?redirect=//evil.example/phish');
    // Must land on default /account/ — NOT navigate off-domain.
    await expect(page).toHaveURL(/\/account\/?$/, { timeout: 5_000 });
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
