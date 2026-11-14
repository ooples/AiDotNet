import { test, expect } from '@playwright/test';

// Every signed-out control that had a handler but no test: theme toggle, pricing interval
// toggle, features search / clear / category expand / copy, homepage code tabs, OAuth buttons,
// and the mobile menu. Each test asserts the control's effect, not just that it rendered.

const desktopOnly = (name: string) => name !== 'anon-desktop';

test('theme toggle switches dark mode and the choice survives a reload', async ({ page }, info) => {
  test.skip(desktopOnly(info.project.name), 'desktop only');
  await page.goto('/');
  const isDark = () => page.evaluate(() => document.documentElement.classList.contains('dark'));
  const before = await isDark();
  await page.locator('#theme-toggle').click();
  await expect.poll(isDark).toBe(!before);
  await page.reload();
  await expect.poll(isDark).toBe(!before);
  await page.locator('#theme-toggle').click();
  await expect.poll(isDark).toBe(before);
});

test('pricing monthly/yearly toggle swaps every price and back', async ({ page }, info) => {
  test.skip(desktopOnly(info.project.name), 'desktop only');
  await page.goto('/pricing/');
  const prices = page.locator('[data-monthly-price][data-yearly-price]');
  const n = await prices.count();
  expect(n, 'paid cards carry both prices').toBeGreaterThan(0);
  const pairs = await prices.evaluateAll((els) => els.map((e) => [e.getAttribute('data-monthly-price'), e.getAttribute('data-yearly-price')]));
  for (let i = 0; i < n; i++) await expect(prices.nth(i)).toContainText(pairs[i][0]!);
  await page.locator('#billing-toggle').click();
  for (let i = 0; i < n; i++) await expect(prices.nth(i)).toContainText(pairs[i][1]!);
  await page.locator('#billing-toggle').click();
  for (let i = 0; i < n; i++) await expect(prices.nth(i)).toContainText(pairs[i][0]!);
});

test('features search filters, reports no results, clears, and expands categories', async ({ page, context, baseURL }, info) => {
  test.skip(desktopOnly(info.project.name), 'desktop only');
  await page.goto('/features/');
  const categories = page.locator('.feature-category');
  const total = await categories.count();
  expect(total).toBeGreaterThan(3);

  // A term that matches nothing hides every category and shows the empty state.
  await page.locator('#feature-search').fill('zzqqxx-no-such-feature');
  await expect(page.locator('#no-results')).toBeVisible();
  await expect(categories.filter({ visible: true })).toHaveCount(0);
  await page.locator('#clear-search').click();
  await expect(page.locator('#feature-search')).toHaveValue('');
  await expect(page.locator('#no-results')).toBeHidden();
  await expect(categories.filter({ visible: true })).toHaveCount(total);

  // A real model name narrows the list and auto-expands its category.
  const tag = (await page.locator('.model-tag').first().textContent())!.trim();
  await page.locator('#feature-search').fill(tag);
  const visible = categories.filter({ visible: true });
  await expect.poll(() => visible.count()).toBeGreaterThan(0);
  expect(await visible.count()).toBeLessThanOrEqual(total);
  await expect(visible.first().locator('.category-content')).toBeVisible();
  await page.locator('#feature-search').fill('');

  // The category header toggles its content.
  const first = categories.first();
  const toggle = first.locator('.category-toggle');
  const content = first.locator('.category-content');
  const wasOpen = await content.isVisible();
  await toggle.click();
  await expect(content).toBeVisible({ visible: !wasOpen });
  await toggle.click();
  await expect(content).toBeVisible({ visible: wasOpen });

  // Copy buttons put the snippet on the clipboard.
  const copy = page.locator('.copy-btn').first();
  if (await copy.count()) {
    await context.grantPermissions(['clipboard-read', 'clipboard-write'], { origin: new URL(baseURL!).origin });
    await copy.scrollIntoViewIfNeeded();
    await copy.click();
    await expect.poll(() => page.evaluate(() => navigator.clipboard.readText())).not.toBe('');
  }
});

test('homepage code tabs switch panels and selection state', async ({ page }, info) => {
  test.skip(desktopOnly(info.project.name), 'desktop only');
  await page.goto('/');
  const tabs = page.locator('.code-tab');
  const n = await tabs.count();
  expect(n).toBeGreaterThan(1);
  for (let i = n - 1; i >= 0; i--) {
    const tab = tabs.nth(i);
    await tab.click();
    const panelId = await tab.getAttribute('aria-controls');
    await expect(tab).toHaveAttribute('aria-selected', 'true');
    await expect(page.locator(`#${panelId}`)).toBeVisible();
    await expect(page.locator('.code-panel').filter({ visible: true })).toHaveCount(1);
  }
});

for (const [path, ids] of [['/login/', ['github-login', 'google-login']], ['/signup/', ['github-signup', 'google-signup']]] as const) {
  test(`${path} OAuth buttons start the provider sign-in with our callback`, async ({ page, baseURL }, info) => {
    test.skip(desktopOnly(info.project.name), 'desktop only');
    for (const id of ids) {
      const provider = id.split('-')[0];
      let authorize: URL | undefined;
      // Stop at Supabase's authorize endpoint: following it would leave for GitHub/Google.
      await page.route('**/auth/v1/authorize**', async (route) => {
        authorize = new URL(route.request().url());
        await route.fulfill({ status: 200, contentType: 'text/html', body: '<p>provider redirect (stubbed by test)</p>' });
      });
      await page.goto(path);
      await page.locator(`#${id}`).click();
      await expect.poll(() => authorize?.href, { timeout: 15_000 }).toBeTruthy();
      expect(authorize!.searchParams.get('provider')).toBe(provider);
      expect(authorize!.searchParams.get('redirect_to')).toBe(`${new URL(baseURL!).origin}/auth/callback/`);
      await page.unroute('**/auth/v1/authorize**');
    }
  });
}

test('mobile menu opens and closes', async ({ page }, info) => {
  test.skip(info.project.name !== 'anon-mobile', 'mobile only');
  await page.goto('/');
  const menu = page.locator('#mobile-menu');
  await expect(menu).toBeHidden();
  await page.locator('#mobile-menu-btn').click();
  await expect(menu).toBeVisible();
  await expect(menu.getByRole('link').first()).toBeVisible();
  await page.locator('#mobile-menu-btn').click();
  await expect(menu).toBeHidden();
});
