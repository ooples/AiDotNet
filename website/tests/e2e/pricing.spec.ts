import { test, expect } from '@playwright/test';

const BASE = '';

test.describe('Pricing Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`${BASE}/pricing/`);
  });

  test('page loads successfully', async ({ page }) => {
    await expect(page.locator('h1')).toBeVisible();
  });

  test('3 tier names render', async ({ page }) => {
    const bodyText = await page.locator('body').textContent() || '';
    expect(bodyText).toContain('Free');
    expect(bodyText).toContain('Pro');
    expect(bodyText).toContain('Enterprise');
  });

  test('monthly/yearly toggle exists', async ({ page }) => {
    const toggle = page.locator('#billing-toggle');
    await expect(toggle).toBeVisible();
    // Also check the labels
    const monthlyLabel = page.locator('#label-monthly');
    const yearlyLabel = page.locator('#label-yearly');
    await expect(monthlyLabel).toBeVisible();
    await expect(yearlyLabel).toBeVisible();
  });

  test('Pro tier shows price', async ({ page }) => {
    const bodyText = await page.locator('body').textContent() || '';
    expect(bodyText).toMatch(/\$29|\$24/);
  });

  test('CTA buttons render', async ({ page }) => {
    // PricingCard renders the Stripe-checkout CTAs (Subscribe on Pro /
    // Enterprise tiers, Get Free License on Community) as <button>
    // elements — the click handler triggers a JS-driven Stripe Checkout
    // flow, not a navigation. Plain non-Stripe CTAs (e.g., Contact)
    // render as <a> with an href. Match either role so the test reflects
    // the real page semantics; if `getByRole('link', …)` were the only
    // selector, healthy pages would fail because Subscribe is correctly
    // a button.
    const namePattern = /Install|Subscribe|Contact|Get Started|Get Free License/i;
    const ctas = page
      .getByRole('button', { name: namePattern })
      .or(page.getByRole('link', { name: namePattern }));
    const count = await ctas.count();
    expect(count).toBeGreaterThanOrEqual(2);
  });

  test('FAQ section renders', async ({ page }) => {
    const bodyText = await page.locator('body').textContent() || '';
    expect(bodyText).toContain('Frequently asked questions');
  });
});
