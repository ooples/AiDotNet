import { test, expect } from '@playwright/test';

test.describe('Admin — API keys', () => {
  test('/admin/api-keys/ renders header, filters, and keys table', async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });

    await page.goto('/admin/api-keys/');

    await expect(page.getByRole('heading', { name: 'API Keys', level: 1 })).toBeVisible();

    // Loader resolving to the table (or an empty-state for a fresh env)
    // means the initial user_api_keys fetch got through. We don't require
    // rows because a test env could legitimately have zero API keys.
    await expect(page.locator('#keys-loading')).toBeHidden({ timeout: 15_000 });

    await expect(page.locator('#search-input')).toBeVisible();
    const statusOptions = await page.locator('#filter-status option').allTextContents();
    expect(statusOptions).toEqual(['All Status', 'Active', 'Revoked']);

    // Active/Revoked counts should resolve to integers. If the
    // user_api_keys RLS policy blocks admin reads, they stick at "--".
    await expect(page.locator('#active-count')).not.toHaveText('--', { timeout: 15_000 });
    await expect(page.locator('#revoked-count')).not.toHaveText('--', { timeout: 15_000 });

    expect(consoleErrors, 'admin api-keys page should emit zero console errors').toEqual([]);
  });
});
