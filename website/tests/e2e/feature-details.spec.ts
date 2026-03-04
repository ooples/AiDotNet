import { test, expect } from '@playwright/test';

const BASE = '/AiDotNet';

const featurePages = [
  { path: '/features/neural-networks/', name: 'Neural Network Layers' },
  { path: '/features/classical-ml/', name: 'Classical ML' },
  { path: '/features/computer-vision/', name: 'Computer Vision' },
  { path: '/features/diffusion-image/', name: 'Image Generation' },
  { path: '/features/diffusion-video/', name: 'Video Generation' },
  { path: '/features/diffusion-audio/', name: 'Audio Generation' },
  { path: '/features/audio-speech/', name: 'Audio & Speech' },
  { path: '/features/vision-language/', name: 'Vision-Language' },
  { path: '/features/video-processing/', name: 'Video Processing' },
  { path: '/features/reinforcement-learning/', name: 'Reinforcement Learning' },
  { path: '/features/lora-finetuning/', name: 'LoRA & Fine-Tuning' },
  { path: '/features/ner-nlp/', name: 'NER & NLP' },
  { path: '/features/optimizers/', name: 'Optimizers' },
  { path: '/features/loss-functions/', name: 'Loss Functions' },
  { path: '/features/activations/', name: 'Activation Functions' },
  { path: '/features/federated-learning/', name: 'Federated Learning' },
  { path: '/features/distributed-training/', name: 'Distributed' },
  { path: '/features/automl-nas/', name: 'AutoML' },
  { path: '/features/finance-ai/', name: 'Financial AI' },
  { path: '/features/model-serving/', name: 'Model Serving' },
  { path: '/features/performance/', name: 'Performance' },
];

for (const featurePage of featurePages) {
  test.describe(`Feature Detail: ${featurePage.name}`, () => {
    test(`loads without errors`, async ({ page }) => {
      const response = await page.goto(`${BASE}${featurePage.path}`);
      expect(response?.status()).toBe(200);
    });

    test(`renders hero with title`, async ({ page }) => {
      await page.goto(`${BASE}${featurePage.path}`);
      const title = page.locator('h1');
      await expect(title).toBeVisible();
      const titleText = await title.textContent();
      expect(titleText?.length).toBeGreaterThan(0);
    });

    test(`renders subcategories`, async ({ page }) => {
      await page.goto(`${BASE}${featurePage.path}`);
      // Feature detail pages have multiple subcategory sections
      const headings = page.locator('h2, h3');
      const count = await headings.count();
      expect(count).toBeGreaterThanOrEqual(2);
    });

    test(`renders code example with AiModelBuilder`, async ({ page }) => {
      await page.goto(`${BASE}${featurePage.path}`);
      const codeBlock = page.locator('pre, code');
      const allCode = await codeBlock.allTextContents();
      const hasAiModelBuilder = allCode.some(text => text.includes('AiModelBuilder'));
      expect(hasAiModelBuilder).toBe(true);
    });

    test(`has navigation links`, async ({ page }) => {
      await page.goto(`${BASE}${featurePage.path}`);
      // Should have links back to features or to docs
      const links = page.locator('a[href*="features"], a[href*="docs"]');
      const count = await links.count();
      expect(count).toBeGreaterThanOrEqual(1);
    });
  });
}
