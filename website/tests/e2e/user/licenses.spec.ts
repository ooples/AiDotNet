import { test, expect } from '@playwright/test';

test.describe('Logged-in user — licenses', () => {
  test('/account/licenses/ renders header and Get-a-License CTA', async ({ page }) => {
    await page.goto('/account/licenses/');

    await expect(page.getByRole('heading', { name: 'License Keys', level: 1 })).toBeVisible();

    // The "Get a License" CTA is the primary conversion path from this page —
    // if the href regresses to /signup/ or /account/billing/ instead of
    // /pricing/, a paying-customer flow breaks. Assert on the link's href
    // rather than the text alone.
    const ctaLink = page.getByRole('link', { name: 'Get a License' });
    await expect(ctaLink).toBeVisible();
    await expect(ctaLink).toHaveAttribute('href', /\/pricing\/?$/);
  });

  test('renders either the empty state or the license list (not both, and not the loader)', async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });

    await page.goto('/account/licenses/');

    // Wait for the loader to disappear. Which terminal state we land on
    // depends on whether the test user happens to hold any license keys,
    // so we assert the *invariant* — loader done, and exactly one of the
    // two terminal UIs visible — instead of picking a branch.
    await expect(page.locator('#licenses-loading')).toBeHidden({ timeout: 10_000 });

    const noLicensesVisible = await page.locator('#no-licenses:not(.hidden)').count();
    const containerVisible = await page.locator('#licenses-container:not(.hidden)').count();

    expect(noLicensesVisible + containerVisible,
      'exactly one of #no-licenses or #licenses-container should be visible').toBe(1);

    // Setup instructions should only appear when the user actually has a
    // license to configure. If they show up in the empty-state branch, the
    // page is leaking DOM from an earlier render.
    if (containerVisible > 0) {
      await expect(page.locator('#setup-section')).toBeVisible();
      // The env-var snippet should be populated with a real key, not the
      // literal placeholder ("your-key-here").
      const envCode = await page.locator('#env-var-code').textContent();
      expect(envCode).toMatch(/^AIDOTNET_LICENSE_KEY=.+/);
      expect(envCode).not.toContain('your-key-here');
    } else {
      await expect(page.locator('#setup-section')).toBeHidden();
    }

    expect(consoleErrors, 'licenses page should emit zero console errors').toEqual([]);
  });

  test('post-checkout ?new=true flow shows the provisioning banner', async ({ page }) => {
    // The spinner banner is gated on `?new=true` in the URL. We don't
    // actually trigger a Stripe checkout here — we just verify that the
    // banner's three pre-rendered icons exist and that the loading icon is
    // the one that's visible first. If a future edit accidentally deletes
    // any of the icons, this test catches it before a real customer hits
    // the post-checkout flow.
    await page.goto('/account/licenses/?new=true');

    await expect(page.locator('#new-license-banner')).toBeVisible();
    await expect(page.locator('#new-license-icon-loading')).toBeVisible();
    // The success/warning icons must exist in the DOM (they're toggled via
    // `hidden` class, not innerHTML replacement) — that's how the page
    // avoids an XSS surface. If they get refactored away, the markup-only
    // contract breaks.
    await expect(page.locator('#new-license-icon-success')).toBeAttached();
    await expect(page.locator('#new-license-icon-warning')).toBeAttached();
  });
});
