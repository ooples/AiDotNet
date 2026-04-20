import { test as setup, expect } from '@playwright/test';
import { USER_STORAGE_STATE } from '../../../playwright.config';

const email = process.env.PLAYWRIGHT_USER_EMAIL;
const password = process.env.PLAYWRIGHT_USER_PASSWORD;

/**
 * Logs in as the regular test user once per test-run and persists the
 * resulting Supabase session (localStorage + cookies) to
 * USER_STORAGE_STATE. All specs in the `user-auth` project attach that
 * storage state so they start already signed-in.
 *
 * Why not service-role JWT injection? Supabase's JS client writes to
 * localStorage under `sb-<ref>-auth-token`, and recomputes the token via
 * refresh on every tab focus — the cleanest way to avoid token-shape
 * drift is to do a real email+password login and let the client store
 * whatever it wants to store.
 */
setup('authenticate as regular user', async ({ page }) => {
  if (!email || !password) {
    throw new Error(
      'PLAYWRIGHT_USER_EMAIL and PLAYWRIGHT_USER_PASSWORD must be set. ' +
      'In CI they come from GitHub secrets; locally export them before ' +
      'running `npx playwright test`.',
    );
  }

  await page.goto('/login/');
  await page.getByLabel('Email address').fill(email);
  await page.getByLabel('Password').fill(password);
  await page.getByRole('button', { name: 'Sign In' }).click();

  // Sign-in redirects to /account/ on success. Asserting on URL +
  // dashboard heading catches both login-form errors ("Invalid email or
  // password") and silent-failure states where the form just doesn't
  // submit (which would otherwise produce a useless "timeout waiting
  // for navigation" downstream).
  await expect(page).toHaveURL(/\/account\/?$/, { timeout: 15_000 });
  await expect(page.getByRole('heading', { name: 'Dashboard', level: 1 })).toBeVisible();

  await page.context().storageState({ path: USER_STORAGE_STATE });
});
