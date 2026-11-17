import { test, expect } from '@playwright/test';

// Fail fast at module load if the env var is missing. Defaulting to ''
// hid the config error behind a cryptic "locator resolved to an empty
// string" failure on the profile-email assertion. The user.setup
// project already enforces this at login time, but replaying the check
// here keeps it meaningful when someone runs this spec in isolation.
const PLAYWRIGHT_USER_EMAIL = process.env.PLAYWRIGHT_USER_EMAIL;
if (!PLAYWRIGHT_USER_EMAIL) {
  throw new Error(
    'PLAYWRIGHT_USER_EMAIL is required for tests/e2e/user/settings.spec.ts — ' +
    'set it to the seeded test-user email (locally export it; in CI it comes ' +
    'from the matching GitHub secret).',
  );
}

test.describe('Logged-in user — settings', () => {
  test('profile section populates from session and saves a name update', async ({ page }) => {
    await page.goto('/account/settings/');
    await expect(page.getByRole('heading', { name: 'Settings', level: 1 })).toBeVisible();

    // The profile section lazy-loads from supabase.auth.getUser(), so the
    // email text starts as "Loading..." and flips to the real value. If
    // the getUser() call regresses, "Loading..." sticks forever.
    await expect(page.locator('#profile-email')).toHaveText(PLAYWRIGHT_USER_EMAIL, {
      timeout: 10_000,
    });

    // "Signed in with email" confirms the app_metadata.provider is set to
    // 'email' for this user — i.e., it was created via the password flow
    // rather than Google/GitHub OAuth. This matters because the
    // Change-Password section only renders for email users.
    await expect(page.locator('#profile-provider')).toContainText(/Signed in with email/i);
    await expect(page.locator('#password-section')).toBeVisible();

    // Save a new name, then restore the original in a finally block.
    // Two reasons to restore rather than leave a timestamp suffix:
    //   1. This is a post-deploy production smoke. Leaving every run's
    //      "Playwright User <epoch>" on the profile accumulates drift
    //      and makes the admin-users table noisier over time.
    //   2. If a future test asserts on the seeded profile name
    //      explicitly, the leftover stale name would break it.
    const fullNameInput = page.getByLabel('Full Name');
    const originalName = await fullNameInput.inputValue();
    const newName = `Playwright User ${Date.now()}`;

    try {
      await fullNameInput.fill(newName);
      await page.getByRole('button', { name: 'Save Changes' }).click();
      await expect(page.locator('#profile-success')).toBeVisible();
      await expect(page.locator('#profile-success')).toContainText(
        /Profile updated successfully/i,
      );

      // Reload to prove the update actually persisted to auth.users +
      // public.profiles, not just reflected in local form state.
      await page.reload();
      await expect(page.getByLabel('Full Name')).toHaveValue(newName, { timeout: 10_000 });
    } finally {
      // Restore even on assertion failure. If originalName was empty
      // (first-ever run against a fresh seed), we still write the
      // empty value back explicitly so the restore path itself is
      // exercised every run.
      await page.getByLabel('Full Name').fill(originalName);
      await page.getByRole('button', { name: 'Save Changes' }).click();
      await expect(page.locator('#profile-success')).toBeVisible();
    }
  });

  test('password mismatch surfaces a friendly error without calling the API', async ({ page }) => {
    await page.goto('/account/settings/');
    await expect(page.locator('#password-section')).toBeVisible({ timeout: 10_000 });

    await page.getByLabel('New Password', { exact: true }).fill('aaaaaaaa');
    await page.getByLabel('Confirm New Password').fill('bbbbbbbb');
    await page.getByRole('button', { name: 'Update Password' }).click();

    // Client-side check short-circuits before calling supabase.auth
    // .updateUser, so we should see the local error and *not* Supabase's
    // own "Password should be at least..." variant.
    await expect(page.locator('#password-error')).toBeVisible();
    await expect(page.locator('#password-error')).toContainText(/Passwords do not match/i);
  });

  test('Delete Account asks for DELETE, and Cancel leaves the account untouched', async ({ page }) => {
    // The seeded account is shared by every run, so this never confirms: the real deletion (refused
    // while a subscription bills, then done) is account-billing.spec.ts on a throwaway customer.
    const deleteCalls: string[] = [];
    page.on('request', (r) => { if (r.url().includes('/functions/v1/delete-account')) deleteCalls.push(r.method()); });
    page.on('dialog', (d) => { throw new Error(`unexpected browser dialog: ${d.message()}`); });
    await page.goto('/account/settings/');

    await page.getByRole('button', { name: 'Delete Account' }).click();
    const dialog = page.locator('#delete-account-modal');
    await expect(dialog).toBeVisible();
    await expect(dialog).toContainText('revokes every license key');
    for (let i = 0; i < 6; i++) {
      await page.keyboard.press(i % 2 ? 'Tab' : 'Shift+Tab');
      expect(await page.evaluate(() => !!document.activeElement?.closest('#delete-account-modal')), 'keyboard focus stays in the deletion dialog').toBe(true);
    }
    const confirm = page.locator('#delete-confirm-btn');
    await expect(confirm).toBeDisabled();
    await page.locator('#delete-confirm-input').fill('delete');
    await expect(confirm, 'only the exact word enables deletion').toBeDisabled();
    await page.locator('#delete-confirm-input').fill('DELETE');
    await expect(confirm).toBeEnabled();

    await page.locator('#delete-cancel-btn').click();
    await expect(dialog).toBeHidden();
    await expect(page.locator('#delete-account-btn')).toBeFocused();
    // Reopening starts clean: the typed confirmation does not carry over.
    await page.locator('#delete-account-btn').click();
    await expect(page.locator('#delete-confirm-input')).toHaveValue('');
    await expect(confirm).toBeDisabled();
    await page.keyboard.press('Escape');
    await expect(dialog).toBeHidden();
    expect(deleteCalls, 'no delete request was sent').toEqual([]);
  });
});
