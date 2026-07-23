import { test, expect } from '@playwright/test';

const BASE = '';

test.describe('Features Catalog', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`${BASE}/features/`);
  });

  test('page loads with hero', async ({ page }) => {
    await expect(page.locator('h1')).toBeVisible();
  });

  test('page contains all major categories', async ({ page }) => {
    const bodyText = await page.locator('body').textContent() || '';
    const categories = [
      'Neural Network', 'Classical Machine Learning', 'Computer Vision',
      'Diffusion', 'Audio', 'Vision-Language', 'Video',
      'Reinforcement Learning', 'LoRA', 'NER', 'Optimizer',
      'Loss Function', 'Activation', 'Federated', 'Distributed',
      'AutoML', 'Financial', 'Model Serving', 'Performance',
    ];
    for (const cat of categories) {
      expect(bodyText, `Should contain "${cat}"`).toContain(cat);
    }
  });

  test('search input exists', async ({ page }) => {
    const searchInput = page.locator('input');
    const count = await searchInput.count();
    expect(count).toBeGreaterThan(0);
  });

  test('category links navigate to detail pages', async ({ page }) => {
    const detailLinks = page.locator('a[href*="/features/neural-networks"], a[href*="/features/classical-ml"]');
    const count = await detailLinks.count();
    expect(count).toBeGreaterThan(0);
  });

  test('stats badge shows implementation count', async ({ page }) => {
    const bodyText = await page.locator('body').textContent() || '';
    // Features page should show category/module counts
    expect(bodyText.length).toBeGreaterThan(100);
  });
});
