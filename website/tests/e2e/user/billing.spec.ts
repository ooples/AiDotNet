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

    await expect(page.locator('#billing-change-plan')).toHaveText(/^(Upgrade|Change plan)$/);

    // FAQ details are collapsed by default. Expanding one proves the
    // native <details> behavior isn't broken by overriding CSS (which
    // *has* happened before — transform: none on summary can break it).
    const firstFaq = page.locator('details').first();
    const summary = firstFaq.locator('summary');
    await summary.click();
    await expect(firstFaq).toHaveAttribute('open', '');

    expect(consoleErrors, 'billing page should emit zero console errors').toEqual([]);
  });

  test('every billing control leads somewhere real for this account', async ({ page }) => {
    await page.goto('/account/billing/');
    await expect(page.locator('#billing-root')).toHaveAttribute('data-billing-loaded', 'true', { timeout: 20_000 });
    await expect(page.locator('#billing-error')).toBeHidden();
    const plan = (await page.locator('#billing-plan').textContent())?.trim();
    const hasAccount = (await page.locator('#billing-root').getAttribute('data-billing-account')) === 'true';

    if (hasAccount) {
      // A Stripe customer: both portal controls must open a real Stripe portal session.
      for (const id of ['#manage-subscription-btn', '#view-invoices-btn']) {
        await Promise.all([page.waitForURL(/^https:\/\/billing\.stripe\.com\//, { timeout: 30_000 }), page.locator(id).click()]);
        await page.goBack();
        await expect(page.locator('#billing-root')).toHaveAttribute('data-billing-loaded', 'true', { timeout: 20_000 });
      }
    } else if (plan !== 'Free') {
      // A paid plan with no Stripe customer: say how to change it instead of showing dead links.
      await expect(page.locator('#portal-section')).toBeHidden();
      await expect(page.locator('#billing-no-account')).toBeVisible();
    } else {
      await expect(page.locator('#portal-section')).toBeHidden();
    }

    await page.locator('#billing-change-plan').click();
    await expect(page).toHaveURL(/\/pricing\/$/);
  });
});
