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
  test.use({ ...devices['Pixel 5'] });

  test('the issuance form scrolls and its Cancel button accepts a real phone tap', async ({ page }) => {
    await page.goto('/admin/licenses/');
    await expect(page.locator('#licenses-loading')).toBeHidden();
    await page.getByRole('button', { name: 'Issue License', exact: true }).click();
    const modal = page.locator('#issue-modal');
    await expect(modal).toBeVisible();
    await page.locator('#issue-product').selectOption('aidotnet');
    await page.locator('#issue-email').fill('delivered+e2e-phone-cancel@resend.dev');
    await page.locator('#issue-customer-name').fill('E2E phone dialog cancellation');
    await page.locator('#issue-notes').fill('The Cancel action must be reachable without zooming or horizontal panning.');
    const cancel = page.locator('#cancel-issue-btn');
    await cancel.scrollIntoViewIfNeeded();
    await expect(cancel).toBeInViewport();
    await cancel.click();
    await expect(modal).toBeHidden();
  });
});
