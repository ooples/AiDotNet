import { test as setup, expect } from '@playwright/test';
import { ADMIN_STORAGE_STATE } from '../../../playwright.config';

const email = process.env.PLAYWRIGHT_ADMIN_EMAIL;
const password = process.env.PLAYWRIGHT_ADMIN_PASSWORD;

/**
 * Same shape as user.setup.ts but for the admin account. After sign-in,
 * asserts that /admin/ loads — that route is gated behind
 * profiles.role = 'admin', so this doubles as a verification that the
 * admin test account's role is correctly set (and that the admin guard
 * on the frontend is still wired).
 */
setup('authenticate as admin', async ({ page }) => {
  if (!email || !password) {
    throw new Error(
      'PLAYWRIGHT_ADMIN_EMAIL and PLAYWRIGHT_ADMIN_PASSWORD must be set. ' +
      'In CI they come from GitHub secrets; locally export them before ' +
      'running `npx playwright test`.',
    );
  }

  await page.goto('/login/');
  await page.getByLabel('Email address').fill(email);
  await page.getByLabel('Password').fill(password);
  await page.getByRole('button', { name: 'Sign In' }).click();

  await expect(page).toHaveURL(/\/account\/?$/, { timeout: 15_000 });

  // Verify admin-gate via a real navigation — if role='admin' somehow
  // got rolled back on the test user, the frontend would redirect to
  // /account/ instead of rendering the admin layout. The heading assertion
  // below catches both cases.
  await page.goto('/admin/');
  await expect(page).toHaveURL(/\/admin\/?$/);
  // "Admin Panel" matches three nodes (mobile nav link, desktop nav link,
  // sidebar badge). We scope to the sidebar's <aside> so a hidden nav-link
  // fallback doesn't satisfy the assertion when the sidebar actually
  // failed to render.
  await expect(page.getByRole('complementary').getByText('Admin Panel')).toBeVisible();

  await page.context().storageState({ path: ADMIN_STORAGE_STATE });
});
