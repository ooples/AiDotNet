import { test, expect } from '@playwright/test';

const BASE = '/AiDotNet';

test.describe('Responsive Design - Desktop', () => {
  test.use({ viewport: { width: 1440, height: 900 } });

  test('landing page renders full navbar', async ({ page }) => {
    await page.goto(`${BASE}/`);
    const nav = page.locator('nav');
    await expect(nav).toBeVisible();
    const links = nav.locator('a:visible');
    const count = await links.count();
    expect(count).toBeGreaterThanOrEqual(4);
  });

  test('no horizontal overflow on landing page', async ({ page }) => {
    await page.goto(`${BASE}/`);
    const hasOverflow = await page.evaluate(() => {
      return document.documentElement.scrollWidth > document.documentElement.clientWidth;
    });
    expect(hasOverflow).toBe(false);
  });
});

test.describe('Responsive Design - Tablet', () => {
  test.use({ viewport: { width: 768, height: 1024 } });

  test('landing page renders without overflow', async ({ page }) => {
    await page.goto(`${BASE}/`);
    const hasOverflow = await page.evaluate(() => {
      return document.documentElement.scrollWidth > document.documentElement.clientWidth;
    });
    expect(hasOverflow).toBe(false);
  });

  test('pricing page renders without overflow', async ({ page }) => {
    await page.goto(`${BASE}/pricing/`);
    const hasOverflow = await page.evaluate(() => {
      return document.documentElement.scrollWidth > document.documentElement.clientWidth;
    });
    expect(hasOverflow).toBe(false);
  });
});

test.describe('Responsive Design - Mobile', () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test('landing page renders without overflow', async ({ page }) => {
    await page.goto(`${BASE}/`);
    const hasOverflow = await page.evaluate(() => {
      return document.documentElement.scrollWidth > document.documentElement.clientWidth;
    });
    expect(hasOverflow).toBe(false);
  });

  test('features catalog renders without overflow', async ({ page }) => {
    await page.goto(`${BASE}/features/`);
    const hasOverflow = await page.evaluate(() => {
      return document.documentElement.scrollWidth > document.documentElement.clientWidth;
    });
    expect(hasOverflow).toBe(false);
  });

  test('pricing page renders without overflow', async ({ page }) => {
    await page.goto(`${BASE}/pricing/`);
    const hasOverflow = await page.evaluate(() => {
      return document.documentElement.scrollWidth > document.documentElement.clientWidth;
    });
    expect(hasOverflow).toBe(false);
  });

  test('solution detail page renders without overflow', async ({ page }) => {
    await page.goto(`${BASE}/solutions/finance/`);
    const hasOverflow = await page.evaluate(() => {
      return document.documentElement.scrollWidth > document.documentElement.clientWidth;
    });
    expect(hasOverflow).toBe(false);
  });

  test('font sizes are readable on mobile', async ({ page }) => {
    await page.goto(`${BASE}/`);
    const bodyFontSize = await page.evaluate(() => {
      const body = document.querySelector('body');
      return body ? parseFloat(window.getComputedStyle(body).fontSize) : 0;
    });
    expect(bodyFontSize).toBeGreaterThanOrEqual(14);
  });
});
