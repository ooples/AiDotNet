import { test, expect, devices } from '@playwright/test';

test.describe('Admin — licenses', () => {
  test('renders header, filter dropdowns, and the issue-license modal', async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });

    await page.goto('/admin/licenses/');

    await expect(page.getByRole('heading', { name: 'License Keys', level: 1 })).toBeVisible();

    // Three status counts + a "Issue License" button together form the
    // header row. If the counts stay at "--" after the initial fetch,
    // the license_keys RLS policy blocked the admin read.
    await expect(page.locator('#active-count')).not.toHaveText('--', { timeout: 15_000 });

    // Product filter must list all supported products. Keep this in
    // sync with the PRODUCTS array in admin/licenses/index.astro — adding
    // a new product without updating the options regresses admin filter
    // UX silently.
    const productOptions = await page.locator('#filter-product option').allTextContents();
    expect(productOptions).toEqual(['All Products', 'AiDotNet', 'Harmonic Engine', 'Token Optimizer']);

    const tierOptions = await page.locator('#filter-tier option').allTextContents();
    expect(tierOptions).toEqual(['All Tiers', 'Community', 'Professional', 'Enterprise']);

    // Open the Issue License modal. The empty "Select a product"
    // placeholder must be present and disabled/selected — this is the
    // guard that prevents accidental AiDotNet issuance when the admin
    // meant to issue Harmonic Engine.
    await page.getByRole('button', { name: 'Issue License' }).click();
    await expect(page.locator('#issue-modal')).toBeVisible();

    const placeholderOption = page.locator('#issue-product option').first();
    const placeholderValue = await placeholderOption.getAttribute('value');
    expect(placeholderValue).toBe('');
    const placeholderDisabled = await placeholderOption.getAttribute('disabled');
    expect(placeholderDisabled).not.toBeNull();
    // Also assert the <select> resolves to the empty-value placeholder
    // on open. If only `disabled` stayed on the option but `selected`
    // disappeared, browsers would auto-pick PRODUCTS[0] (AiDotNet) and
    // the guard this test protects evaporates silently.
    await expect(page.locator('#issue-product')).toHaveValue('');

    // Cancel closes the modal. The real issuance path (issue, email,
    // resend, SDK validation, revoke) is covered by license-lifecycle.spec.ts.
    await page.getByRole('button', { name: 'Cancel' }).click();
    await expect(page.locator('#issue-modal')).toBeHidden();

    expect(consoleErrors, 'admin licenses should emit zero console errors').toEqual([]);
  });
});

test.describe('Phone admin license dialog', () => {
  const phone = devices['Pixel 5'];
  test.use({ viewport: phone.viewport, userAgent: phone.userAgent, deviceScaleFactor: phone.deviceScaleFactor, isMobile: true, hasTouch: true });

  test('the issuance form scrolls and its Cancel button accepts a real phone tap', async ({ page }) => {
    const unexpectedIssues: string[] = [];
    await page.route('**/functions/v1/issue-license', async route => {
      unexpectedIssues.push(route.request().method());
      await route.fulfill({ status: 409, json: { error: 'phone_cancel_test_blocks_issuance' } });
    });
    await page.goto('/admin/licenses/');
    await expect(page.locator('#licenses-loading')).toBeHidden();
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(393);
    await page.getByRole('button', { name: 'Issue License', exact: true }).click();
    const modal = page.locator('#issue-modal');
    await expect(modal).toBeVisible();
    await page.locator('#issue-product').selectOption('aidotnet');
    const sendEmail = page.locator('#issue-send-email');
    await sendEmail.scrollIntoViewIfNeeded();
    await sendEmail.tap();
    await expect(sendEmail).not.toBeChecked();
    await page.locator('#issue-email').fill('delivered+e2e-phone-cancel@resend.dev');
    await page.locator('#issue-customer-name').fill('E2E phone dialog cancellation');
    await page.locator('#issue-notes').fill('The Cancel action must be reachable without zooming or horizontal panning.');
    const cancel = page.locator('#cancel-issue-btn');
    const panel = modal.locator(':scope > div');
    await panel.evaluate(el => { el.scrollTop = 0; });
    const session = await page.context().newCDPSession(page);
    try {
      for (let i = 0; i < 8; i++) {
        try { await expect(cancel).toBeInViewport({ ratio: 1, timeout: 100 }); break; } catch { /* swipe to reach it */ }
        const box = await panel.boundingBox();
        if (!box) throw new Error('Issue dialog has no touch target');
        await session.send('Input.synthesizeScrollGesture', {
          x: box.x + box.width / 2, y: box.y + box.height - 25,
          yDistance: -Math.min(300, box.height - 50), speed: 700, gestureSourceType: 'touch',
        });
      }
      await expect(cancel).toBeInViewport({ ratio: 1 });
    } finally { await session.detach(); }
    await cancel.tap();
    await expect(modal).toBeHidden();
    expect(unexpectedIssues).toEqual([]);
  });
});
