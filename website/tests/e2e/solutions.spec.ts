import { test, expect } from '@playwright/test';

const BASE = '';

test.describe('Solutions Hub', () => {
  test('solutions index page loads', async ({ page }) => {
    const response = await page.goto(`${BASE}/solutions/`);
    expect(response?.status()).toBe(200);
  });

  test('solutions index contains all 3 solution types', async ({ page }) => {
    await page.goto(`${BASE}/solutions/`);
    const bodyText = await page.locator('body').textContent() || '';
    expect(bodyText).toContain('Finance');
    expect(bodyText).toContain('Healthcare');
    expect(bodyText).toContain('Document');
  });

  test('solution links navigate to detail pages', async ({ page }) => {
    await page.goto(`${BASE}/solutions/`);
    const financeLink = page.locator('a[href*="solutions/finance"]');
    const healthcareLink = page.locator('a[href*="solutions/healthcare"]');
    const documentLink = page.locator('a[href*="solutions/document"]');

    expect(await financeLink.count()).toBeGreaterThan(0);
    expect(await healthcareLink.count()).toBeGreaterThan(0);
    expect(await documentLink.count()).toBeGreaterThan(0);
  });
});

const solutionPages = [
  { path: '/solutions/finance/', name: 'Financial Services' },
  { path: '/solutions/healthcare/', name: 'Healthcare' },
  { path: '/solutions/document-ai/', name: 'Document AI' },
];

for (const solution of solutionPages) {
  test.describe(`Solution Detail: ${solution.name}`, () => {
    test(`loads without errors`, async ({ page }) => {
      const response = await page.goto(`${BASE}${solution.path}`);
      expect(response?.status()).toBe(200);
    });

    test(`renders hero section`, async ({ page }) => {
      await page.goto(`${BASE}${solution.path}`);
      const title = page.locator('h1');
      await expect(title).toBeVisible();
    });

    test(`renders capabilities with model tags`, async ({ page }) => {
      await page.goto(`${BASE}${solution.path}`);
      const bodyText = await page.locator('body').textContent() || '';
      // Each solution page mentions specific model names
      const hasModels = /BERT|GPT|SAM|OCR|GARCH|DeepAR|Chronos|LayoutLM/.test(bodyText);
      expect(hasModels).toBe(true);
    });

    test(`renders workflow steps`, async ({ page }) => {
      await page.goto(`${BASE}${solution.path}`);
      const bodyText = await page.locator('body').textContent() || '';
      // Solution pages have workflow sections with step descriptions
      const hasWorkflow = /deployment|training|ingestion|preparation/i.test(bodyText);
      expect(hasWorkflow).toBe(true);
    });

    test(`renders code example with AiModelBuilder`, async ({ page }) => {
      await page.goto(`${BASE}${solution.path}`);
      const codeBlock = page.locator('pre, code');
      const allCode = await codeBlock.allTextContents();
      const hasAiModelBuilder = allCode.some(text => text.includes('AiModelBuilder'));
      expect(hasAiModelBuilder).toBe(true);
    });

    test(`renders CTA links`, async ({ page }) => {
      await page.goto(`${BASE}${solution.path}`);
      const cta = page.getByRole('link', { name: /Get Started|Install|Learn More|Documentation|Browse|GitHub/i });
      const count = await cta.count();
      expect(count).toBeGreaterThanOrEqual(1);
    });
  });
}
