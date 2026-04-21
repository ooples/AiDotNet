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

  test('Delete Account shows a confirm dialog and then a support-contact alert', async ({ page }) => {
    await page.goto('/account/settings/');

    let dialogCount = 0;
    const dialogMessages: string[] = [];
    page.on('dialog', async (dialog) => {
      dialogCount++;
      dialogMessages.push(dialog.message());
      if (dialog.type() === 'confirm') await dialog.accept();
      else await dialog.dismiss();
    });

    await page.getByRole('button', { name: 'Delete Account' }).click();

    // Two dialogs in sequence: a confirm() ("Are you sure...") followed
    // by an alert() directing to support. This is intentional — we don't
    // offer self-service delete yet — so we assert the two-step shape
    // explicitly so an accidental silent-delete doesn't slip through.
    await expect.poll(() => dialogCount).toBe(2);
    expect(dialogMessages[0]).toMatch(/Are you sure you want to delete your account/i);
    expect(dialogMessages[1]).toMatch(/support@aidotnet\.dev/);
  });
});
