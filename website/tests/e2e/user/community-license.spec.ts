import { test, expect } from '@playwright/test';

// Community registration against the REAL register-community-license function, as a signed-in
// user clicking the pricing page button. The function is idempotent per user: the first run
// creates the test user's community license, every later run gets the same key back
// ("is_existing"), so this is safe to run on every deploy and nightly.
//
// This path broke silently when a redeploy rebuilt the function against a newer gotrue-js:
// every signed-in caller got 401 "Invalid or expired session." and no test called it.

test('signed-in user registers (or retrieves) a community license from the pricing page', async ({ page }) => {
  await page.goto('/pricing/');

  const response = page.waitForResponse(
    (r) => r.url().includes('/functions/v1/register-community-license') && r.request().method() === 'POST',
    { timeout: 30_000 },
  );
  await page.locator('.checkout-btn[data-tier="community"]').first().click();
  const res = await response;
  const body = await res.json();
  // Never log keys or tokens: CI logs of this public repo are world-readable.
  const safe = JSON.stringify({ ...body, license_key: body.license_key ? '<redacted>' : undefined, bootstrap_token: body.bootstrap_token ? '<redacted>' : undefined });
  // 201 when the license is created, 200 when the user already had one.
  expect([200, 201], `register-community-license: HTTP ${res.status()} ${safe}`).toContain(res.status());
  expect(body.success, safe).toBe(true);
  expect(typeof body.license_key === 'string' && body.license_key.length > 0, 'a license key is returned').toBe(true);

  // The page shows the key in the success (new) or existing-license banner, never the error banner.
  const shown = body.is_existing ? '#existing-key-display' : '#community-key-display';
  await expect(page.locator(shown)).toHaveText(body.license_key);
  await expect(page.locator('#error-banner')).toBeHidden();
});
