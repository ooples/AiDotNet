import { test, expect } from '@playwright/test';

test.describe('Logged-in user — billing', () => {
  test('renders the current plan, upgrade CTA, and FAQ', async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });

    await page.goto('/account/billing/');
    await expect(page.getByRole('heading', { name: 'Billing', level: 1 })).toBeVisible();

    // Gate on the explicit load-complete marker set by loadBilling().
    // Without it, the plan regex below could pass on the pre-hydration
    // default ("Free" is baked into the HTML) and we'd silently accept
    // a broken profile fetch for a Pro/Enterprise account.
    await expect(page.locator('#billing-root')).toHaveAttribute(
      'data-billing-loaded',
      'true',
      { timeout: 10_000 },
    );

    // The plan name is populated from profiles.subscription_tier, and it
    // must be one of the three supported tiers — any other value means
    // the tier mapping in billing/index.astro has regressed.
    await expect(page.locator('#billing-plan')).toHaveText(/^(Free|Pro|Enterprise)$/);

    // The upgrade CTA sits on every tier (its label changes), so assert
    // the button exists rather than the text. For Free users the link
    // points to /pricing/; for Pro it's still /pricing/ but the label is
    // "Manage Plan"; for Enterprise the href is mailto:support.
    const upgradeLink = page.locator('#billing-upgrade-btn');
    await expect(upgradeLink).toBeVisible();

    // FAQ details are collapsed by default. Expanding one proves the
    // native <details> behavior isn't broken by overriding CSS (which
    // *has* happened before — transform: none on summary can break it).
    const firstFaq = page.locator('details').first();
    const summary = firstFaq.locator('summary');
    await summary.click();
    await expect(firstFaq).toHaveAttribute('open', '');

    expect(consoleErrors, 'billing page should emit zero console errors').toEqual([]);
  });

  test('Stripe portal section is shown only for paying customers', async ({ page }) => {
    await page.goto('/account/billing/');

    // Wait on the explicit load-complete marker rather than text content,
    // because "Free" is the pre-hydration default — a naive text-match
    // on #billing-plan would pass before loadBilling() has actually read
    // the profile. The data-billing-loaded flag is set last in
    // loadBilling(), so once it flips we know the portal-section toggle
    // has also run.
    await expect(page.locator('#billing-root')).toHaveAttribute(
      'data-billing-loaded',
      'true',
      { timeout: 10_000 },
    );

    const plan = await page.locator('#billing-plan').textContent();
    const portalSectionHidden = await page.locator('#portal-section').evaluate((el) =>
      el.classList.contains('hidden'),
    );

    if (plan?.trim() === 'Free') {
      // Free users have no Stripe customer ID yet — the portal section
      // must stay hidden so we don't render Manage-Subscription / View-
      // Invoices links that go nowhere.
      expect(portalSectionHidden).toBe(true);
    } else {
      // Paying customers must see the portal links, and both links must
      // resolve to a real (not "#") href once PUBLIC_STRIPE_PORTAL_URL is
      // configured. If the build-time env is missing, the links degrade
      // to "#" — we don't fail the test on that case because the portal
      // URL is optional until the customer portal is formally opened.
      expect(portalSectionHidden).toBe(false);
    }
  });
});
