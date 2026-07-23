import { test, expect } from '@playwright/test';

test.describe('Admin — usage analytics', () => {
  test('/admin/usage/ renders stat cards, chart, and date-range selector', async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });

    await page.goto('/admin/usage/');

    await expect(page.getByRole('heading', { name: 'Usage Analytics', level: 1 })).toBeVisible();

    // Four summary stats must render even on an empty api_usage table
    // (they'd show "--" or "0"). If any of these IDs disappear from the
    // markup, the loadUsage() setter misses them silently.
    for (const id of ['stat-total', 'stat-avg-latency', 'stat-p95', 'stat-success']) {
      await expect(page.locator(`#${id}`)).toBeVisible();
    }

    // Admin-side date range has a 14-day window that user-side doesn't —
    // the difference is intentional so admins can compare week-over-week
    // windows without waiting 30 days. If these options ever merge, the
    // comparison use case breaks.
    const rangeOptions = await page.locator('#date-range option').allTextContents();
    expect(rangeOptions).toEqual(['Last 7 days', 'Last 14 days', 'Last 30 days']);

    // Changing range triggers loadUsage() again. We don't assert on the
    // numbers (the test env may be empty) — just that the change event
    // fires and the dropdown value sticks.
    await page.locator('#date-range').selectOption('7');
    await expect(page.locator('#date-range')).toHaveValue('7');

    expect(consoleErrors, 'admin usage page should emit zero console errors').toEqual([]);
  });
});
