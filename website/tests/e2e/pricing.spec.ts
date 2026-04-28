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
    // PricingCard renders the Stripe-checkout CTAs as <button> elements
    // (the click handler runs the JS Stripe Checkout flow, not a
    // navigation), so we accept either button or link role. Each tier's
    // CTA is asserted individually with an exact-name regex and an
    // explicit count: 1× "Get Free License" (Community) + 2×
    // "Subscribe" (Professional + Enterprise). A broad regex with a
    // ≥ 2 floor would false-pass when an actual tier CTA breaks (e.g.,
    // an unrelated "Get Started" elsewhere on the page would still
    // satisfy a 2-CTA threshold even if Subscribe disappeared).
    const freeLicenseCta = page
      .getByRole('button', { name: /^Get Free License$/i })
      .or(page.getByRole('link', { name: /^Get Free License$/i }));
    const subscribeCtas = page
      .getByRole('button', { name: /^Subscribe$/i })
      .or(page.getByRole('link', { name: /^Subscribe$/i }));

    await expect(freeLicenseCta).toHaveCount(1);
    await expect(subscribeCtas).toHaveCount(2);
  });

  test('FAQ section renders', async ({ page }) => {
    const bodyText = await page.locator('body').textContent() || '';
    expect(bodyText).toContain('Frequently asked questions');
  });
});
