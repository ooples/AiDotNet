import { test, expect } from '@playwright/test';

const BASE = '/AiDotNet';

test.describe('Accessibility', () => {
  test('all images have alt text or decorative role', async ({ page }) => {
    await page.goto(`${BASE}/`);
    const images = page.locator('img');
    const count = await images.count();

    for (let i = 0; i < count; i++) {
      const alt = await images.nth(i).getAttribute('alt');
      const role = await images.nth(i).getAttribute('role');
      expect(alt !== null || role === 'presentation', `Image ${i} should have alt text or role="presentation"`).toBe(true);
    }
  });

  test('buttons have accessible labels', async ({ page }) => {
    await page.goto(`${BASE}/`);
    const buttons = page.locator('button');
    const count = await buttons.count();

    for (let i = 0; i < count; i++) {
      const button = buttons.nth(i);
      const text = await button.textContent();
      const ariaLabel = await button.getAttribute('aria-label');
      const ariaLabelledBy = await button.getAttribute('aria-labelledby');
      const title = await button.getAttribute('title');

      const hasAccessibleName = (text && text.trim().length > 0) ||
        ariaLabel !== null ||
        ariaLabelledBy !== null ||
        title !== null;

      expect(hasAccessibleName, `Button ${i} should have an accessible name`).toBe(true);
    }
  });

  test('keyboard navigation works', async ({ page }) => {
    await page.goto(`${BASE}/`);
    await page.keyboard.press('Tab');
    const focusedElement = page.locator(':focus');
    const count = await focusedElement.count();
    expect(count).toBeGreaterThan(0);
  });

  test('page has exactly one h1', async ({ page }) => {
    await page.goto(`${BASE}/`);
    const h1Count = await page.locator('h1').count();
    expect(h1Count).toBe(1);
  });

  test('feature detail pages have one h1', async ({ page }) => {
    await page.goto(`${BASE}/features/neural-networks/`);
    const h1Count = await page.locator('h1').count();
    expect(h1Count).toBe(1);
  });

  test('solution pages have one h1', async ({ page }) => {
    await page.goto(`${BASE}/solutions/finance/`);
    const h1Count = await page.locator('h1').count();
    expect(h1Count).toBe(1);
  });

  test('pricing page has one h1', async ({ page }) => {
    await page.goto(`${BASE}/pricing/`);
    const h1Count = await page.locator('h1').count();
    expect(h1Count).toBe(1);
  });
});
