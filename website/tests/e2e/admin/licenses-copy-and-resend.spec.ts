import { test, expect, Route } from '@playwright/test';

// Coverage for the per-row Copy and Resend controls added to /admin/licenses
// in PR #1256. Each test stubs the side-effecting boundary it cares about
// (clipboard for copy, the supabase functions endpoint for resend, the
// license_keys REST query for the row source) so this suite is safe to
// run against any seeded admin environment without mutating real user
// state or sending real email — including a freshly-seeded environment
// with zero license rows. Closes review-comment #1256.hteW.

/**
 * Stubs the supabase REST query that loads the license_keys table so the
 * /admin/licenses page is guaranteed to render at least one row regardless
 * of the underlying DB state. Without this, `.copy-key-btn.first()` and
 * `.resend-email-btn.first()` would never resolve on an empty environment
 * and the tests would fail on locator timeout, masking the actual control
 * behavior the suite is supposed to verify.
 */
async function stubLicenseRows(
  page: import('@playwright/test').Page,
): Promise<void> {
  const cannedRow = {
    id: 'fixture-license-id-0001',
    license_key: 'aidn.fixturekey1234.fixturesig5678abcdef',
    user_id: null,
    customer_email: 'fixture@example.com',
    product: 'aidotnet',
    tier: 'community',
    status: 'active',
    activations_used: 0,
    activations_limit: 1,
    issued_at: new Date().toISOString(),
    expires_at: null,
    revoked_at: null,
    organization_name: null,
    profiles: { full_name: 'Fixture Owner', email: 'fixture@example.com' },
  };
  await page.route('**/rest/v1/license_keys**', async (route) => {
    const method = route.request().method();
    // Stub GETs (the read path that loads the table). For everything else
    // — POST, PATCH, DELETE, OPTIONS preflight — let the request continue
    // to the real network so a future test that exercises a mutation
    // (suspend, revoke, reactivate) sees the genuine response and can
    // assert on real failure modes. Synthesizing a 204 for non-GETs would
    // mask those failures and produce false-green tests.
    // Closes review-comment #1268.h3tm.
    if (method !== 'GET') {
      await route.continue();
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([cannedRow]),
    });
  });
}

test.describe('Admin — licenses copy + resend buttons', () => {
  // ---------------------------------------------------------------------
  // Copy button
  // ---------------------------------------------------------------------

  test('copy button writes the FULL license key (not the truncated preview) to clipboard', async ({ page }) => {
    // Stub navigator.clipboard.writeText BEFORE any page script attaches so
    // the click handler picks up the stub. addInitScript runs in every
    // frame on every navigation, which is what we need here.
    await page.addInitScript(() => {
      (window as unknown as { __copiedTexts: string[] }).__copiedTexts = [];
      // Many environments expose navigator.clipboard read-only; redefine
      // to be safe across browsers.
      Object.defineProperty(navigator, 'clipboard', {
        configurable: true,
        value: {
          writeText: async (text: string) => {
            (window as unknown as { __copiedTexts: string[] }).__copiedTexts.push(text);
          },
        },
      });
    });

    await stubLicenseRows(page);
    await page.goto('/admin/licenses/');

    // Wait for the table to populate. The stat counter `#active-count`
    // flipping off the placeholder `--` is the same readiness signal the
    // sibling spec uses; reusing it keeps the readiness contract single-
    // sourced.
    await expect(page.locator('#active-count')).not.toHaveText('--', { timeout: 15_000 });

    const firstCopyBtn = page.locator('.copy-key-btn').first();
    await expect(firstCopyBtn).toBeVisible();

    // The DOM only carries the truncated preview (e.g. `aidn.abc1...wxyz`).
    // The full key lives in the in-memory allLicenses array and is only
    // surfaced via clipboard. Snapshot the preview so we can verify the
    // copied string is a SUPERSET of it (i.e., the full key, not the
    // truncated form the page displays).
    const preview = (await firstCopyBtn.textContent() ?? '').trim();
    expect(preview, 'copy button should render a non-empty preview').not.toEqual('');

    await firstCopyBtn.click();

    // page.click() dispatches the event but does NOT await the async
    // handler — by the time control returns, navigator.clipboard.writeText
    // may not have run yet. Poll until __copiedTexts is populated before
    // reading it; otherwise the assertion can race the click handler and
    // see length 0 on a fast machine. Same pattern the alert test below
    // uses for window.alert.
    await expect
      .poll(
        () => page.evaluate(() => (window as unknown as { __copiedTexts: string[] }).__copiedTexts.length),
        { timeout: 3_000 },
      )
      .toBe(1);

    const copied = await page.evaluate(() => {
      return (window as unknown as { __copiedTexts: string[] }).__copiedTexts;
    });
    expect(copied).toHaveLength(1);
    const copiedKey = copied[0];

    // The copied string must NOT be just the preview — the whole point of
    // the button is to recover the full key. The preview contains a `...`
    // separator from the substring(0,8) + '...' + substring(-4) format;
    // the full key never contains that ellipsis.
    expect(copiedKey, 'clipboard should receive the full key, not the truncated preview')
      .not.toContain('...');
    expect(copiedKey.length).toBeGreaterThan(preview.length);

    // The "copied" feedback flash must actually render. The previous
    // regex `/copied|aidn|harm/i` was non-enforcing — it would pass even
    // if the flash never fired, because the original button label
    // contains a key preview that starts with `aidn` or `harm`. Tightened
    // to require the literal "copied" text so a regression that drops
    // the visual feedback fails the test.
    await expect(firstCopyBtn).toContainText(/copied/i, { timeout: 2000 });
  });

  test('copy button surfaces a clipboard-permission alert when writeText throws', async ({ page }) => {
    await page.addInitScript(() => {
      Object.defineProperty(navigator, 'clipboard', {
        configurable: true,
        value: {
          writeText: async () => { throw new Error('Permission denied'); },
        },
      });
    });

    // Capture the alert text. window.alert is a blocking dialog — Playwright's
    // dialog handler fires before the JS resumes so we both record the
    // message and dismiss the modal.
    const alerts: string[] = [];
    page.on('dialog', async (d) => {
      alerts.push(d.message());
      await d.dismiss();
    });

    await stubLicenseRows(page);
    await page.goto('/admin/licenses/');
    await expect(page.locator('#active-count')).not.toHaveText('--', { timeout: 15_000 });

    await page.locator('.copy-key-btn').first().click();

    // Allow the click handler's async catch path to run and dispatch alert().
    await expect.poll(() => alerts.length, { timeout: 3_000 }).toBeGreaterThan(0);

    const msg = alerts[0];
    // Message must point at REAL workarounds. The earlier version
    // suggested "Reveal the key via the activations modal" which the
    // activations modal does not actually do; PR #1256 review feedback
    // surfaced that as misleading and we fixed it. Lock the fix in.
    expect(msg.toLowerCase()).not.toContain('activations modal');
    expect(msg.toLowerCase()).toMatch(/permission|browser|https|secure/);
  });

  // ---------------------------------------------------------------------
  // Resend button
  // ---------------------------------------------------------------------

  // Helper: route handler for the resend edge function. Captures every
  // body that hits it and lets the caller specify a canned response.
  async function stubResend(
    page: import('@playwright/test').Page,
    resp: { status: number; body: Record<string, unknown> },
  ): Promise<{ requests: Array<{ body: unknown }> }> {
    const captured: Array<{ body: unknown }> = [];
    await page.route('**/functions/v1/admin-resend-license-email**', async (route: Route) => {
      // Filter to POST only. supabase.functions.invoke() sends an
      // Authorization header which triggers a CORS preflight OPTIONS
      // request before the actual POST; if we counted that too, the
      // `expect(stub.requests).toHaveLength(1)` assertions below would
      // intermittently see length 2 and fail. Reply 204 (no content) to
      // the preflight without recording it.
      if (route.request().method() !== 'POST') {
        await route.fulfill({ status: 204 });
        return;
      }

      let parsed: unknown = null;
      try {
        parsed = JSON.parse(route.request().postData() ?? 'null');
      } catch { /* leave null */ }
      captured.push({ body: parsed });
      await route.fulfill({
        status: resp.status,
        contentType: 'application/json',
        body: JSON.stringify(resp.body),
      });
    });
    return { requests: captured };
  }

  test('resend button POSTs license_id and shows "sent" on a 200 response', async ({ page }) => {
    const stub = await stubResend(page, {
      status: 200,
      body: { success: true, recipient: 'test@example.com', message: 'OK' },
    });

    // Auto-accept the confirm() dialog the click handler shows before
    // making the request. Without this, the request never fires.
    page.on('dialog', (d) => d.accept());

    await stubLicenseRows(page);
    await page.goto('/admin/licenses/');
    await expect(page.locator('#active-count')).not.toHaveText('--', { timeout: 15_000 });

    const firstResend = page.locator('.resend-email-btn').first();
    await expect(firstResend).toBeVisible();
    const expectedId = await firstResend.getAttribute('data-id');
    expect(expectedId, 'resend button must carry the license row id on data-id').not.toBeNull();

    await firstResend.click();

    // Surface the "sent" feedback. The handler sets innerHTML to a span;
    // assert by visible text rather than DOM structure to keep the test
    // resilient to future UI tweaks.
    await expect(firstResend).toContainText(/sent/i, { timeout: 5_000 });

    // Exactly one POST should have hit the route, with our license id.
    expect(stub.requests).toHaveLength(1);
    expect(stub.requests[0].body).toEqual({ license_id: expectedId });
  });

  test('resend button shows "failed" with the server message visible inline on a 422', async ({ page }) => {
    const stub = await stubResend(page, {
      status: 422,
      body: {
        success: false,
        error: 'no_recipient_email',
        message: 'Edit the license customer_email before retrying.',
      },
    });

    page.on('dialog', (d) => d.accept());

    await stubLicenseRows(page);
    await page.goto('/admin/licenses/');
    await expect(page.locator('#active-count')).not.toHaveText('--', { timeout: 15_000 });

    const firstResend = page.locator('.resend-email-btn').first();
    await firstResend.click();

    await expect(firstResend).toContainText(/failed/i, { timeout: 5_000 });

    // Server message is rendered inline as visible text inside a
    // role="alert" span so it's accessible to keyboard, touch, and
    // screen-reader users (review-comment #1268.h3tx / .h3t6 — the
    // previous tooltip-only placement was inaccessible).
    const alertText = firstResend.locator('[role="alert"]');
    await expect(alertText).toBeVisible({ timeout: 5_000 });
    await expect(alertText).toContainText('customer_email');

    expect(stub.requests).toHaveLength(1);
  });

  test('resend dismiss button restores the original label on click and is keyboard-accessible', async ({ page }) => {
    // PR #1268 review-comment #h3uM: pin the new dismiss interaction.
    // Before this PR, failure auto-reverted via setTimeout and there was
    // no dismiss control; this test ensures (a) the dismiss button
    // appears on failure, (b) clicking it restores the idle label,
    // (c) the dismiss control is a real <button> (not a clickable
    // <span>), and (d) it is reachable AND activatable from the
    // keyboard (tab to focus, Enter or Space to activate) — review-
    // comment #1268.iCxY/g/k/o flagged that earlier revisions only
    // checked attributes, which would let a regression that broke
    // keyboard reach (display:none focus, tabindex=-1, etc.) pass.
    await stubResend(page, {
      status: 500,
      body: { success: false, error: 'send_failed', message: 'Resend provider returned 503.' },
    });
    page.on('dialog', (d) => d.accept());

    await stubLicenseRows(page);
    await page.goto('/admin/licenses/');
    await expect(page.locator('#active-count')).not.toHaveText('--', { timeout: 15_000 });

    const firstResend = page.locator('.resend-email-btn').first();
    const originalLabel = (await firstResend.textContent() ?? '').trim();
    await firstResend.click();

    // Failure renders.
    await expect(firstResend).toContainText(/failed/i, { timeout: 5_000 });

    // Dismiss control is a real <button> (not a clickable <span>) with
    // a screen-reader-friendly aria-label. The dismiss is rendered as a
    // SIBLING of the resend button (button-within-button is invalid
    // markup), so locate it via the resend button's parent rather than
    // as a descendant of the resend button.
    const resendRow = firstResend.locator('xpath=..');
    const dismiss = resendRow.locator('> button.resend-dismiss');
    await expect(dismiss).toBeVisible();
    await expect(dismiss).toHaveAttribute('type', 'button');
    await expect(dismiss).toHaveAttribute('aria-label', /dismiss/i);

    // Real keyboard reachability: tab from the resend button to the
    // dismiss control (it should be the very next focusable element in
    // DOM order since it's an immediate sibling). If anything along
    // the way regresses (tabindex=-1, hidden, focusable: false), this
    // expect-focus assertion fails and locks down the regression.
    await firstResend.focus();
    await page.keyboard.press('Tab');
    await expect(dismiss).toBeFocused();

    // Real keyboard activation: pressing Enter on a focused <button>
    // fires a click event. Without a proper <button> element (e.g., if
    // someone regresses back to a clickable <span>), Enter would do
    // nothing and the failure state would remain — catching that is
    // the whole point of this assertion.
    await page.keyboard.press('Enter');
    await expect(firstResend).not.toContainText(/failed/i, { timeout: 2_000 });
    // The button text returns to whatever was originally there. The
    // original label is short ("resend") so equality is fine.
    const restored = (await firstResend.textContent() ?? '').trim();
    expect(restored).toBe(originalLabel);

    // Sanity: a second failure → dismiss cycle, this time using Space
    // (the OTHER conventional button-activation key). A regression that
    // bound a keypress handler to Enter only would fail here.
    await firstResend.click();
    await expect(firstResend).toContainText(/failed/i, { timeout: 5_000 });
    const dismiss2 = resendRow.locator('> button.resend-dismiss');
    await expect(dismiss2).toBeVisible();
    await dismiss2.focus();
    await page.keyboard.press('Space');
    await expect(firstResend).not.toContainText(/failed/i, { timeout: 2_000 });
  });

  test('resend button makes no request when the admin cancels the confirm dialog', async ({ page }) => {
    let invoked = false;
    await page.route('**/functions/v1/admin-resend-license-email**', async (route: Route) => {
      // Same preflight filter as stubResend — without it, a CORS OPTIONS
      // request from the browser would set invoked=true even when the
      // user dismissed the confirm and the actual POST never fired.
      // Stub functions endpoint preflights with 204 so we never count
      // them; this is the resend endpoint, not the rest/v1 stub.
      if (route.request().method() !== 'POST') {
        await route.fulfill({ status: 204 });
        return;
      }
      invoked = true;
      await route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
    });

    // Dismiss the confirm() dialog this time. The click handler should
    // short-circuit and never enqueue a network call.
    page.on('dialog', (d) => d.dismiss());

    await stubLicenseRows(page);
    await page.goto('/admin/licenses/');
    await expect(page.locator('#active-count')).not.toHaveText('--', { timeout: 15_000 });

    await page.locator('.resend-email-btn').first().click();

    // Give any (broken) request a tick to slip through before asserting it
    // didn't. Without the small wait an immediate assertion could pass
    // even if the dismiss path were broken.
    await page.waitForTimeout(500);
    expect(invoked, 'cancelled confirm must not invoke the resend endpoint').toBe(false);
  });
});
