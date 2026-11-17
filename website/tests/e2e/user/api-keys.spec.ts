import { test, expect } from '@playwright/test';

test.describe('Logged-in user — API keys', () => {
  test('full create → reveal → revoke round-trip', async ({ page, context, baseURL }) => {
    // The key name carries the test-run timestamp so repeated CI runs
    // don't collide on the same row, and so we can grep for this row
    // later to make assertions like "only our key was revoked".
    const keyName = `playwright-e2e-${Date.now()}`;

    await page.goto('/account/api-keys/');
    await expect(page.getByRole('heading', { name: 'API Keys', level: 1 })).toBeVisible();

    // Wait for loadKeys() to finish attaching the form submit handler —
    // the handler captures currentUserId from supabase.auth.getUser(),
    // so a click landing before that resolves is a no-op. Asserting that
    // the Keys table stops "loading" (i.e., a settled render) is the
    // most reliable proxy for "page JS fully bootstrapped".
    await page.waitForLoadState('networkidle');

    // --- Create ---
    await page.getByRole('button', { name: /Create New Key/ }).click();
    await expect(page.locator('#create-modal')).toBeVisible();

    await page.getByLabel('Key Name').fill(keyName);
    // 'Inference' scope is pre-checked; leave the defaults to prove the
    // default-scope contract still holds.
    await page.locator('#create-key-form button[type="submit"]').click();

    // --- Reveal (one-time-only modal) ---
    // If the INSERT fails (RLS, constraint violation, network error), the
    // page surfaces the message in #create-key-error. Race the two
    // possible outcomes so a failure produces a useful error message
    // ("Failed to create API key: <actual message>") instead of a bare
    // "locator never became visible" timeout.
    // Race the two possible outcomes so a genuine INSERT failure (RLS,
    // constraint violation) surfaces as a readable error rather than a
    // bare "waiting for locator" timeout.
    const newKeyModal = page.locator('#new-key-modal');
    const errorBox = page.locator('#create-key-error');
    await Promise.race([
      newKeyModal.waitFor({ state: 'visible', timeout: 10_000 }),
      errorBox.waitFor({ state: 'visible', timeout: 10_000 }),
    ]);
    if (await errorBox.isVisible()) {
      throw new Error(`Key creation failed: ${(await errorBox.textContent()) ?? ''}`);
    }
    await expect(newKeyModal).toBeVisible();

    // The full key is rendered exactly once, then the user has to click
    // "I've copied my key" — if we ever ship a bug that shows the key
    // *after* dismissal, a pentester would find it. Assert the prefix
    // shape here; the mask/reveal behavior on the list row is covered
    // below.
    const revealedKey = await page.locator('#new-key-value').textContent();
    expect(revealedKey, 'created key should use the adn_ prefix and be 68 chars total').toMatch(
      /^adn_[0-9a-f]{64}$/,
    );

    await context.grantPermissions(['clipboard-read', 'clipboard-write'], { origin: new URL(baseURL!).origin });
    await page.locator('#copy-key-btn').click();
    await expect.poll(() => page.evaluate(() => navigator.clipboard.readText())).toBe(revealedKey);
    await page.locator('#close-new-key').click();
    await expect(newKeyModal).toBeHidden();

    // --- Appears in the list with a masked preview, not the full key ---
    const ourRow = page.locator('#keys-body tr', { hasText: keyName });
    await expect(ourRow).toBeVisible();
    // The row shows only the first 12 chars (key_prefix) + "...". The
    // full 68-char key must never make it into the table's DOM once the
    // reveal modal dismisses — that's our post-creation confidentiality
    // invariant.
    const rowHtml = await ourRow.innerHTML();
    expect(rowHtml).not.toContain(revealedKey as string);
    expect(rowHtml).toContain((revealedKey as string).substring(0, 12));

    // --- Revoke ---
    // The revoke handler uses window.confirm, so we need to pre-accept
    // the dialog before clicking the button.
    page.once('dialog', (dialog) => dialog.accept());
    await ourRow.getByRole('button', { name: 'Revoke' }).click();

    // After revocation, the row re-renders with a "Revoked" status label and no
    // Revoke button. If the row-refresh regresses, the button sticks as "Revoke"
    // forever and a second click would silently re-run the UPDATE.
    await expect(ourRow.getByText('Revoked', { exact: true })).toBeVisible({ timeout: 5_000 });
    await expect(ourRow.getByRole('button', { name: 'Revoke' })).toHaveCount(0);
  });

  test('create-modal Cancel button dismisses the modal without creating a key', async ({ page }) => {
    const loaded = page.waitForResponse((r) => r.request().method() === 'GET' && r.url().includes('/rest/v1/user_api_keys'));
    await page.goto('/account/api-keys/');
    await loaded;
    await page.waitForLoadState('networkidle');

    const rowCountBefore = await page.locator('#keys-body tr').count();
    const inserts: string[] = [];
    page.on('request', (r) => { if (r.method() === 'POST' && r.url().includes('/rest/v1/user_api_keys')) inserts.push(r.url()); });

    await page.getByRole('button', { name: /Create New Key/ }).click();
    await expect(page.locator('#create-modal')).toBeVisible();
    await page.getByLabel('Key Name').fill(`cancelled-key-${Date.now()}`);
    await page.getByRole('button', { name: 'Cancel' }).click();
    await expect(page.locator('#create-modal')).toBeHidden();

    // Cancel must not trigger an INSERT. If a refactor accidentally wires
    // the Cancel button to the form's submit handler, we'd see a new row
    // appear. Re-count to confirm nothing was added.
    const rowCountAfter = await page.locator('#keys-body tr').count();
    expect(rowCountAfter).toBe(rowCountBefore);
    expect(inserts, 'Cancel must not submit an insert').toEqual([]);
  });
});
