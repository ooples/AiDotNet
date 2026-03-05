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
    const ctaButtons = page.getByRole('link', { name: /Install|Subscribe|Contact|Get Started/i });
    const count = await ctaButtons.count();
    expect(count).toBeGreaterThanOrEqual(2);
  });

  test('FAQ section renders', async ({ page }) => {
    const bodyText = await page.locator('body').textContent() || '';
    expect(bodyText).toContain('Frequently asked questions');
  });
});
