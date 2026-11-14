import { test, expect, type Page, type Request } from '@playwright/test';
import { createPublicKey, verify, randomBytes } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

// End-to-end license lifecycle against the REAL backend. Nothing here is stubbed: the
// admin issues a key in the UI, the issuance email and a resend go through the
// admin-resend-license-email function to Resend, the SDK endpoints (issue-license,
// validate-license, get-revocations) are called the way the SDK calls them, and the
// key is revoked in the UI.
//
// The UI-only specs (licenses-copy-and-resend.spec.ts) stub the function, which is why
// they stayed green while every real resend returned 401 "Invalid or expired session."
//
// Production writes are bounded and labelled:
//   - the email goes to Resend's test inbox (delivered@resend.dev), which accepts mail
//     through the real API path without delivering it anywhere;
//   - the license row is named "e2e-test <timestamp>" and is always revoked, by the UI
//     step or by the afterEach safety net.

const RUN = `${Date.now()}`;
const CUSTOMER_NAME = `e2e-test ${RUN}`;
const CUSTOMER_EMAIL = `delivered+e2e-${RUN}@resend.dev`;

// The SDK's embedded verification key. The tokens and the CRL must verify against exactly
// this key, or every customer's offline check fails.
const PUBLIC_KEYS = JSON.parse(
  readFileSync(fileURLToPath(new URL('../../../../src/BuildKey/LicensePublicKey.json', import.meta.url)), 'utf8'),
) as { keys: Array<{ kid: string; kty: string; crv: string; x: string }> };

function verifyEd25519(kid: string, message: Buffer, signatureB64Url: string): boolean {
  const jwk = PUBLIC_KEYS.keys.find((k) => k.kid === kid);
  if (!jwk) throw new Error(`kid '${kid}' is not in src/BuildKey/LicensePublicKey.json`);
  const key = createPublicKey({ key: jwk, format: 'jwk' });
  return verify(null, message, key, Buffer.from(signatureB64Url, 'base64url'));
}

interface Backend {
  functionsUrl: string;
  restUrl: string;
  apikey: string;
  authorization: string;
}

interface Issued {
  id: string;
  key: string;
  backend: Backend;
}

let issued: Issued | undefined;

/** Collects alert()/confirm() texts and accepts them, so a failure alert fails the test with its text. */
function recordDialogs(page: Page): string[] {
  const alerts: string[] = [];
  page.on('dialog', async (d) => {
    if (d.type() === 'alert') alerts.push(d.message());
    await d.accept();
  });
  return alerts;
}

test.describe.configure({ mode: 'serial' });

test.describe('License lifecycle (real backend)', () => {
  test.afterEach(async ({ request }) => {
    // Safety net: never leave an active test license behind, even when a step failed.
    if (!issued) return;
    const { restUrl, apikey, authorization } = issued.backend;
    await request.patch(`${restUrl}/license_keys?id=eq.${issued.id}&status=neq.revoked`, {
      headers: { apikey, authorization, 'content-type': 'application/json', prefer: 'return=minimal' },
      data: { status: 'revoked', revoked_at: new Date().toISOString() },
    });
  });

  test('admin issues a key, the email is sent, the SDK endpoints accept it, and revoking it denies it online and offline', async ({ page, request }) => {
    test.setTimeout(120_000);
    // A short laptop viewport (1280x650 is 1366x768 minus browser chrome): the issue dialog
    // used to overflow here with its submit button out of reach.
    await page.setViewportSize({ width: 1280, height: 650 });
    const alerts = recordDialogs(page);

    await page.goto('/admin/licenses/');
    await expect(page.locator('#active-count')).not.toHaveText('--', { timeout: 15_000 });

    // ---- Issue through the dialog ----
    await page.getByRole('button', { name: 'Issue License' }).click();
    await expect(page.locator('#issue-modal')).toBeVisible();
    await page.locator('#issue-product').selectOption('aidotnet');
    await page.locator('#issue-email').fill(CUSTOMER_EMAIL);
    await page.locator('#issue-customer-name').fill(CUSTOMER_NAME);
    await page.locator('#issue-tier').selectOption('professional');
    await page.locator('#issue-max-activations').fill('2');
    await page.locator('#issue-notes').fill('Automated e2e lifecycle test; revoked by the test.');
    await expect(page.locator('#issue-send-email')).toBeChecked();

    const insertRequest = page.waitForRequest(
      (r) => r.url().includes('/rest/v1/license_keys') && r.method() === 'POST',
    );
    const issueEmailResponse = page.waitForResponse(
      (r) => r.url().includes('/functions/v1/admin-resend-license-email') && r.request().method() === 'POST',
      { timeout: 30_000 },
    );
    const submit = page.locator('#issue-form button[type="submit"]');
    await submit.scrollIntoViewIfNeeded();
    await submit.click();

    const insert: Request = await insertRequest;
    const insertBody = insert.postDataJSON() as { license_key: string };
    const insertResponse = await insert.response();
    expect(insertResponse?.status(), 'license insert').toBe(201);
    const inserted = (await insertResponse!.json()) as { id: string };
    const headers = await insert.allHeaders();
    const origin = new URL(insert.url()).origin;
    issued = {
      id: inserted.id,
      key: insertBody.license_key,
      backend: {
        functionsUrl: `${origin}/functions/v1`,
        restUrl: `${origin}/rest/v1`,
        apikey: headers['apikey'],
        authorization: headers['authorization'],
      },
    };
    expect(issued.key).toMatch(/^aidn\./);

    // ---- The issuance email really went out ----
    const emailResponse = await issueEmailResponse;
    expect(
      emailResponse.status(),
      `issuance email: ${await emailResponse.text()}`,
    ).toBe(200);
    expect(alerts, 'no failure alert after issuing').toEqual([]);

    // ---- Resend from the row ----
    await page.locator('#search-input').fill(issued.key);
    const resend = page.locator(`.resend-email-btn[data-id="${issued.id}"]`);
    await expect(resend).toBeVisible();
    const resendResponse = page.waitForResponse(
      (r) => r.url().includes('/functions/v1/admin-resend-license-email') && r.request().method() === 'POST',
      { timeout: 30_000 },
    );
    await resend.click();
    const resent = await resendResponse;
    expect(resent.status(), `resend: ${await resent.text()}`).toBe(200);
    await expect(resend).toHaveText('sent');

    // ---- SDK path: offline token, then online validation ----
    const machine = randomBytes(32).toString('hex');
    const { functionsUrl, apikey } = issued.backend;
    const sdkHeaders = { apikey, 'content-type': 'application/json' };

    const issueRes = await request.post(`${functionsUrl}/issue-license`, {
      headers: sdkHeaders,
      data: { license_key: issued.key, machine_id_hash: machine, hostname: 'e2e', os_description: 'e2e' },
    });
    const issueJson = await issueRes.json();
    expect(issueRes.status(), `issue-license: ${JSON.stringify(issueJson)}`).toBe(200);
    const token = String(issueJson.offline_token);
    const [prefix, claimsB64, sigB64] = token.split('.');
    expect(prefix).toBe('aidn2');
    const claimsBytes = Buffer.from(claimsB64, 'base64url');
    const claims = JSON.parse(claimsBytes.toString('utf8'));
    expect(verifyEd25519(claims.kid, claimsBytes, sigB64), 'offline token signature').toBe(true);
    expect(claims).toMatchObject({ tier: 'professional', jti: issued.id, mach: machine, alg: 'EdDSA' });
    expect(claims.caps).toEqual(expect.arrayContaining(['model:save', 'model:load', 'tensors:save', 'tensors:load']));

    const validateRes = await request.post(`${functionsUrl}/validate-license`, {
      headers: sdkHeaders,
      data: { license_key: issued.key, machine_id_hash: machine, package: 'AiDotNet' },
    });
    const validateJson = await validateRes.json();
    expect(validateRes.status(), `validate-license: ${JSON.stringify(validateJson)}`).toBe(200);
    expect(validateJson.valid).toBe(true);

    // ---- Revoke in the UI ----
    await page.locator(`.revoke-btn[data-id="${issued.id}"]`).click();
    await expect(page.locator('tr', { has: page.locator(`[data-id="${issued.id}"]`) })).toContainText('revoked');
    expect(alerts, 'no failure alert after revoking').toEqual([]);

    // Online: the next validation is refused.
    const revokedRes = await request.post(`${functionsUrl}/validate-license`, {
      headers: sdkHeaders,
      data: { license_key: issued.key, machine_id_hash: machine, package: 'AiDotNet' },
    });
    expect(revokedRes.status()).toBe(403);
    expect((await revokedRes.json()).valid).toBe(false);

    // Offline: the CRL is signed by the SDK's key and lists the license, so cached tokens die too.
    // The query string skips any cached copy (the endpoint sends max-age=3600).
    const crlRes = await request.get(`${functionsUrl}/get-revocations?e2e=${Date.now()}`, { headers: { apikey } });
    expect(crlRes.status()).toBe(200);
    const crl = (await crlRes.json()) as { kid: string; payload: string; sig: string };
    const payloadBytes = Buffer.from(crl.payload, 'base64url');
    expect(verifyEd25519(crl.kid, payloadBytes, crl.sig), 'CRL signature').toBe(true);
    const payload = JSON.parse(payloadBytes.toString('utf8')) as { exp: number; rjti: string[] };
    expect(payload.exp * 1000).toBeGreaterThan(Date.now());
    expect(payload.rjti, 'revoked license must be on the CRL').toContain(issued.id);
  });
});
