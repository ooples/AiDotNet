import { test, expect } from '@playwright/test';

test.describe('Logged-in user — dashboard', () => {
  test('/account/ renders greeting + sidebar + zero console errors', async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });

    await page.goto('/account/');

    await expect(page.getByRole('heading', { name: 'Dashboard', level: 1 })).toBeVisible();

    // Sidebar exposes the full nav. Asserting each item catches a regression
    // where AccountLayout silently drops a link (e.g., from a CSS conflict
    // or a build-time conditional that only surfaces on prod).
    for (const name of ['Dashboard', 'Licenses', 'API Keys', 'Usage', 'Billing', 'Settings']) {
      await expect(page.getByRole('link', { name, exact: true })).toBeVisible();
    }
    await expect(page.getByRole('button', { name: 'Sign Out' })).toBeVisible();

    // Dashboard pulls the user's email into the greeting. If Supabase's
    // session propagation regresses (e.g., an expired token not being
    // refreshed), the greeting shows "Welcome back, undefined".
    await expect(page.getByText(/Welcome back,/)).toBeVisible();

    expect(consoleErrors, 'dashboard should emit zero console errors').toEqual([]);
  });

  test('quick-actions navigate to the right pages', async ({ page }) => {
    await page.goto('/account/');

    // "Generate API Key" card → /account/api-keys/
    await page.getByRole('link', { name: /Generate API Key/ }).click();
    await expect(page).toHaveURL(/\/account\/api-keys\/?$/);

    await page.goBack();

    await page.getByRole('link', { name: /View Usage/ }).click();
    await expect(page).toHaveURL(/\/account\/usage\/?$/);

    await page.goBack();

    await page.getByRole('link', { name: /Manage Billing/ }).click();
    await expect(page).toHaveURL(/\/account\/billing\/?$/);
  });

  test('signed-in users are bounced from /login and /signup', async ({ page }) => {
    // These pages auto-redirect signed-in users to /account/ so the "switch
    // accounts" flow can't accidentally create a second profile. If the
    // redirect regresses, paid subscribers could end up seeing the signup
    // form by mistake.
    await page.goto('/login/');
    await expect(page).toHaveURL(/\/account\/?$/);

    await page.goto('/signup/');
    await expect(page).toHaveURL(/\/account\/?$/);
  });
});
