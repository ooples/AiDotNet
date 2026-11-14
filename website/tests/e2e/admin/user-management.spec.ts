import { test, expect } from '@playwright/test';
import {
  journeySkipReason, testUserEmail, randomPassword, createConfirmedUser, deleteUser, restSelect, restInsert,
  type AdminUser,
} from '../helpers/backend';

// Admin actions on another user's account, against the real backend. Each run creates a
// throwaway user through the admin API (Resend test-inbox address), acts on it in the UI, checks
// the database, and deletes it. The nightly job deletes any leftovers.

let user: AdminUser | undefined;

test.describe.configure({ mode: 'serial' });

test.beforeAll(async ({ request }) => {
  if (journeySkipReason()) return;
  user = await createConfirmedUser(request, testUserEmail('admin'), randomPassword(), `E2E Admin Target ${Date.now()}`);
});

test.afterAll(async ({ request }) => {
  if (user) await deleteUser(request, user.id);
});

test.beforeEach(() => {
  const skip = journeySkipReason();
  test.skip(!!skip, skip);
});

test('admin edits a user: Cancel discards, Save persists role and tier', async ({ page, request }) => {
  const confirmations: string[] = [];
  page.on('dialog', async (d) => { confirmations.push(d.message()); await d.accept(); });
  await page.goto('/admin/users/');
  // The users search matches name or id (not email).
  await page.locator('#search-input').fill(user!.id);
  const edit = page.locator(`.edit-user-btn[data-id="${user!.id}"]`);
  await expect(edit).toBeVisible({ timeout: 15_000 });

  // Cancel leaves the profile untouched.
  await edit.click();
  await expect(page.locator('#edit-modal')).toBeVisible();
  await page.locator('#edit-tier').selectOption('enterprise');
  await page.locator('#modal-cancel').click();
  await expect(page.locator('#edit-modal')).toBeHidden();
  let [profile] = await restSelect<Array<{ role: string; subscription_tier: string }>>(
    request, `profiles?select=role,subscription_tier&id=eq.${user!.id}`);
  expect(profile.subscription_tier).not.toBe('enterprise');

  // Save writes both fields.
  await edit.click();
  await page.locator('#edit-tier').selectOption('pro');
  await page.locator('#edit-role').selectOption('user');
  await page.locator('#modal-save').click();
  await expect(page.locator('#edit-modal')).toBeHidden();
  await expect(page.locator('#modal-error')).toBeHidden();
  [profile] = await restSelect<Array<{ role: string; subscription_tier: string }>>(
    request, `profiles?select=role,subscription_tier&id=eq.${user!.id}`);
  expect(profile).toEqual({ role: 'user', subscription_tier: 'pro' });
  expect(confirmations).toEqual([]);
});

test('admin users list pages forward and back', async ({ page }) => {
  await page.goto('/admin/users/');
  const info = page.locator('#page-info');
  await expect(info).not.toHaveText('', { timeout: 15_000 });
  const next = page.locator('#next-btn');
  if (await next.isDisabled()) {
    // One page of users: both buttons must be disabled, not dead-but-enabled.
    await expect(page.locator('#prev-btn')).toBeDisabled();
    return;
  }
  const first = await info.textContent();
  await next.click();
  await expect(info).not.toHaveText(first!);
  await page.locator('#prev-btn').click();
  await expect(info).toHaveText(first!);
});

test("admin revokes a user's API key", async ({ page, request }) => {
  const keyName = `e2e-admin-revoke-${Date.now()}`;
  const key = await restInsert<{ id: string }>(request, 'user_api_keys', {
    user_id: user!.id,
    key_name: keyName,
    key_prefix: 'e2e_',
    key_hash: 'e2e-not-a-real-hash',
  });
  page.on('dialog', (d) => d.accept());
  await page.goto('/admin/api-keys/');
  // The API-key search matches key name, prefix or owner name.
  await page.locator('#search-input').fill(keyName);
  const revoke = page.locator(`.revoke-btn[data-id="${key.id}"]`);
  await expect(revoke).toBeVisible({ timeout: 15_000 });
  await revoke.click();
  await expect.poll(async () => {
    const [row] = await restSelect<Array<{ is_active: boolean }>>(request, `user_api_keys?select=is_active&id=eq.${key.id}`);
    return row.is_active;
  }, { timeout: 15_000 }).toBe(false);
});
