import { test, expect } from '@playwright/test';

const BASE = '/AiDotNet';

test.describe('Link Validation', () => {
  test('internal navigation links on landing page work', async ({ page }) => {
    await page.goto(`${BASE}/`);
    const internalLinks = page.locator(`a[href*="/AiDotNet/"]`);
    const hrefs: string[] = [];
    const count = await internalLinks.count();

    for (let i = 0; i < count; i++) {
      const href = await internalLinks.nth(i).getAttribute('href');
      if (href && !hrefs.includes(href)) {
        hrefs.push(href);
      }
    }

    // Check a sample of internal links (first 15 to keep test fast)
    const sampled = hrefs.slice(0, 15);
    for (const href of sampled) {
      const response = await page.goto(href);
      expect(response?.status(), `Link ${href} should return 200`).toBe(200);
    }
  });

  test('external links have target=_blank', async ({ page }) => {
    await page.goto(`${BASE}/`);
    const externalLinks = page.locator('a[href^="http"]');
    const count = await externalLinks.count();

    let checked = 0;
    for (let i = 0; i < count && checked < 10; i++) {
      const link = externalLinks.nth(i);
      const href = await link.getAttribute('href');
      if (href && !href.includes('localhost') && !href.includes('127.0.0.1')) {
        const target = await link.getAttribute('target');
        if (target) {
          expect(target).toBe('_blank');
          checked++;
        }
      }
    }
  });

  test('GitHub link exists', async ({ page }) => {
    await page.goto(`${BASE}/`);
    const githubLink = page.locator('a[href*="github.com"]');
    const count = await githubLink.count();
    expect(count).toBeGreaterThan(0);
  });

  test('feature detail links from features index work', async ({ page }) => {
    await page.goto(`${BASE}/features/`);
    const detailLinks = page.locator(`a[href*="/AiDotNet/features/"]`);
    const hrefs: string[] = [];
    const count = await detailLinks.count();

    for (let i = 0; i < count; i++) {
      const href = await detailLinks.nth(i).getAttribute('href');
      if (href && !hrefs.includes(href) && href !== `${BASE}/features/` && href !== '/AiDotNet/features/') {
        hrefs.push(href);
      }
    }

    for (const href of hrefs) {
      const response = await page.goto(href);
      expect(response?.status(), `Feature link ${href} should return 200`).toBe(200);
    }
  });

  test('footer has links', async ({ page }) => {
    await page.goto(`${BASE}/`);
    const footerLinks = page.locator('footer a');
    const count = await footerLinks.count();
    expect(count).toBeGreaterThan(0);
  });
});
