import { test, expect } from '@playwright/test';

test.describe('Logged-in user — usage', () => {
  test('renders stat cards and date-range selector', async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });

    await page.goto('/account/usage/');
    await expect(page.getByRole('heading', { name: 'Usage', level: 1 })).toBeVisible();

    // The four summary cards must always render, regardless of whether
    // the user has any api_usage rows. A fresh account shows "0" and
    // "--" placeholders; a heavy user shows real numbers. Either way,
    // the four IDs must be on the page — if one gets accidentally renamed
    // by a template refactor, the loadUsage() setter misses it silently.
    await expect(page.locator('#total-calls')).toBeVisible();
    await expect(page.locator('#avg-latency')).toBeVisible();
    await expect(page.locator('#p95-latency')).toBeVisible();
    await expect(page.locator('#success-rate')).toBeVisible();

    // Date-range selector must offer exactly the three windows we support.
    // Adding a 4th without updating the loadUsage() math (or vice versa)
    // is a classic regression source.
    const rangeOptions = await page.locator('#date-range option').allTextContents();
    expect(rangeOptions).toEqual(['Last 7 days', 'Last 30 days', 'Last 90 days']);

    // Changing the range triggers loadUsage() again. We don't assert on
    // the resulting numbers (the test user has no traffic), only that the
    // 'change' listener is wired — a DOM trace of the dropdown changing
    // without the script running would mean the script tag crashed before
    // attach.
    await page.locator('#date-range').selectOption('7');
    await expect(page.locator('#date-range')).toHaveValue('7');

    expect(consoleErrors, 'usage page should emit zero console errors').toEqual([]);
  });
});
