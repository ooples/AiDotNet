import { test, expect } from '@playwright/test';

const BASE = '/AiDotNet';

test.describe('Landing Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`${BASE}/`);
  });

  test('hero section renders headline and CTA buttons', async ({ page }) => {
    await expect(page.locator('h1')).toBeVisible();
    const ctaButtons = page.getByRole('link', { name: /Get Started|Install|GitHub/i });
    await expect(ctaButtons.first()).toBeVisible();
  });

  test('stats badge shows correct implementation count', async ({ page }) => {
    const statsText = await page.locator('body').textContent();
    expect(statsText).toContain('7,300');
  });

  test('NuGet badge renders', async ({ page }) => {
    const nugetBadge = page.locator('a[href*="nuget.org"], img[alt*="NuGet"]');
    await expect(nugetBadge.first()).toBeVisible();
  });

  test('feature cards grid renders with correct counts', async ({ page }) => {
    const bodyText = await page.locator('body').textContent();
    // Check key feature card titles exist
    expect(bodyText).toContain('Neural Network');
    expect(bodyText).toContain('Classical ML');
    expect(bodyText).toContain('Computer Vision');
  });

  test('Browse all features CTA link works', async ({ page }) => {
    const browseLink = page.getByRole('link', { name: /Browse all|View all|See all/i }).first();
    if (await browseLink.isVisible()) {
      const href = await browseLink.getAttribute('href');
      expect(href).toContain('features');
    }
  });

  test('industry solutions section renders', async ({ page }) => {
    const bodyText = await page.locator('body').textContent();
    expect(bodyText).toContain('Finance');
    expect(bodyText).toContain('Healthcare');
    expect(bodyText).toContain('Document');
  });

  test('code examples contain AiModelBuilder pattern', async ({ page }) => {
    const codeBlocks = page.locator('pre, code');
    const allCode = await codeBlocks.allTextContents();
    const hasAiModelBuilder = allCode.some(text => text.includes('AiModelBuilder'));
    expect(hasAiModelBuilder).toBe(true);
  });

  test('comparison section renders', async ({ page }) => {
    const bodyText = await page.locator('body').textContent();
    expect(bodyText).toContain('TorchSharp');
  });

  test('footer renders with links', async ({ page }) => {
    const footer = page.locator('footer');
    await expect(footer).toBeVisible();
    const footerLinks = footer.locator('a');
    const linkCount = await footerLinks.count();
    expect(linkCount).toBeGreaterThan(0);
  });
});
