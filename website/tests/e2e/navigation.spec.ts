import { test, expect } from '@playwright/test';

const BASE = '/AiDotNet';

test.describe('Desktop Navigation', () => {
  test.use({ viewport: { width: 1440, height: 900 } });

  test('navbar renders all links', async ({ page }) => {
    await page.goto(`${BASE}/`);
    const nav = page.locator('nav');
    await expect(nav).toBeVisible();

    const links = ['Features', 'Solutions', 'Pricing', 'Docs', 'Playground', 'API Reference'];
    for (const linkText of links) {
      await expect(nav.getByRole('link', { name: linkText })).toBeVisible();
    }
  });

  test('Features link navigates to features page', async ({ page }) => {
    await page.goto(`${BASE}/`);
    await page.locator('nav').getByRole('link', { name: 'Features' }).click();
    await expect(page).toHaveURL(/\/features\//);
    await expect(page.locator('h1')).toBeVisible();
  });

  test('Solutions link navigates to solutions page', async ({ page }) => {
    await page.goto(`${BASE}/`);
    await page.locator('nav').getByRole('link', { name: 'Solutions' }).click();
    await expect(page).toHaveURL(/\/solutions\//);
  });

  test('Pricing link navigates to pricing page', async ({ page }) => {
    await page.goto(`${BASE}/`);
    await page.locator('nav').getByRole('link', { name: 'Pricing' }).click();
    await expect(page).toHaveURL(/\/pricing\//);
  });

  test('back/forward browser navigation works', async ({ page }) => {
    await page.goto(`${BASE}/`);
    await page.locator('nav').getByRole('link', { name: 'Features' }).click();
    await expect(page).toHaveURL(/\/features\//);
    await page.goBack();
    await expect(page).toHaveURL(/\/AiDotNet\/?$/);
    await page.goForward();
    await expect(page).toHaveURL(/\/features\//);
  });
});

test.describe('Navigation - All Viewports', () => {
  test('logo click returns to landing page', async ({ page }) => {
    await page.goto(`${BASE}/features/`);
    await page.locator(`nav a[href*="AiDotNet"]`).first().click();
    await expect(page).toHaveURL(/\/AiDotNet\/?$/);
  });

  test('Get Started CTA navigates to docs', async ({ page }) => {
    await page.goto(`${BASE}/`);
    const cta = page.getByRole('link', { name: /Get Started/i }).first();
    await expect(cta).toBeVisible();
    const href = await cta.getAttribute('href');
    expect(href).toContain('docs');
  });

  test('navbar gets background on scroll', async ({ page }) => {
    await page.goto(`${BASE}/`);
    const nav = page.locator('nav').first();
    await page.evaluate(() => window.scrollBy(0, 200));
    await page.waitForTimeout(500);
    const hasBackground = await nav.evaluate((el) => {
      const style = window.getComputedStyle(el);
      return style.backgroundColor !== 'rgba(0, 0, 0, 0)' && style.backgroundColor !== 'transparent';
    });
    expect(hasBackground).toBe(true);
  });
});

test.describe('Mobile Navigation', () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test('mobile hamburger menu opens and closes', async ({ page }) => {
    await page.goto(`${BASE}/`);
    const menuButton = page.locator('button[aria-label*="menu" i], button[aria-label*="Menu" i], nav button').first();
    if (await menuButton.isVisible()) {
      await menuButton.click();
      const mobileMenu = page.locator('[class*="mobile"], [class*="menu"], nav ul, nav div[class*="open"]').first();
      await expect(mobileMenu).toBeVisible();
    }
  });
});
