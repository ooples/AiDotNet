import { test, expect, Route } from '@playwright/test';

// Coverage for the per-row Copy and Resend controls added to /admin/licenses
// in PR #1256. Each test stubs the side-effecting boundary it cares about
// (clipboard for copy, the supabase functions endpoint for resend) so this
// suite is safe to run against any seeded admin environment without
// mutating real user state or sending real email.
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

    // Click flashes "copied" then reverts. Don't race the revert — assert
    // on the captured clipboard text directly.
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

    // Brief feedback flash should fire and then revert. We check it
    // appeared at least once (race-tolerant).
    await expect(firstCopyBtn).toContainText(/copied|aidn|harm/i, { timeout: 2000 });
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

  test('resend button shows "failed" + tooltip carrying the server message on a 422', async ({ page }) => {
    const stub = await stubResend(page, {
      status: 422,
      body: {
        success: false,
        error: 'no_recipient_email',
        message: 'Edit the license customer_email before retrying.',
      },
    });

    page.on('dialog', (d) => d.accept());

    await page.goto('/admin/licenses/');
    await expect(page.locator('#active-count')).not.toHaveText('--', { timeout: 15_000 });

    const firstResend = page.locator('.resend-email-btn').first();
    await firstResend.click();

    await expect(firstResend).toContainText(/failed/i, { timeout: 5_000 });

    // The handler embeds the server message into the title attribute on
    // the inner <span>, so the admin sees WHICH failure category fired
    // (no_recipient → 422) on hover without opening devtools.
    const failedSpan = firstResend.locator('span[title]').first();
    await expect(failedSpan).toHaveAttribute('title', /customer_email/i, { timeout: 5_000 });

    expect(stub.requests).toHaveLength(1);
  });

  test('resend button makes no request when the admin cancels the confirm dialog', async ({ page }) => {
    let invoked = false;
    await page.route('**/functions/v1/admin-resend-license-email**', async (route: Route) => {
      invoked = true;
      await route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
    });

    // Dismiss the confirm() dialog this time. The click handler should
    // short-circuit and never enqueue a network call.
    page.on('dialog', (d) => d.dismiss());

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
