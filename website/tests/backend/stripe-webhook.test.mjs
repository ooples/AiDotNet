import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { webcrypto } from 'node:crypto';
import ts from 'typescript';

// Execute the real edge handler with boundary doubles. No network, secrets,
// or claims about Stripe signature cryptography / database RLS are involved.
const source = readFileSync(new URL('../../supabase/functions/stripe-webhook/index.ts', import.meta.url), 'utf8');
const code = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
}).outputText;

function fixture({ insertError, profileError, existingLicense = false } = {}) {
  let handler;
  let issued = existingLicense;
  const writes = [], emails = [];
  const client = {
    from(table) {
      const query = {
        select() { return query; },
        eq() { return query; },
        update(data) { writes.push({ table, operation: 'update', data }); return query; },
        insert(data) {
          writes.push({ table, operation: 'insert', data });
          if (!insertError) issued = true;
          return Promise.resolve({ error: insertError ?? null });
        },
        maybeSingle() { return Promise.resolve({ data: issued ? { id: 'license-id', license_key: 'existing-key' } : null, error: null }); },
        then(resolve, reject) { return Promise.resolve({ error: profileError ?? null }).then(resolve, reject); },
      };
      return query;
    },
  };
  class Stripe {
    static createFetchHttpClient() { return {}; }
    webhooks = { constructEventAsync: async (body, signature) => {
      if (signature !== 'fixture-signature') throw new Error('invalid signature');
      return JSON.parse(body);
    } };
  }
  const env = {
    STRIPE_SECRET_KEY: 'sk_test_fixture', STRIPE_WEBHOOK_SECRET: 'whsec_fixture',
    SUPABASE_URL: 'http://fixture.invalid', SUPABASE_SERVICE_ROLE_KEY: 'fixture-service',
  };
  runInNewContext(code, {
    exports: {}, Request, Response, crypto: webcrypto, Uint8Array,
    console: { log() {}, warn() {}, error() {} },
    Deno: { env: { get: (key) => env[key] } },
    require(id) {
      if (id.includes('deno.land')) return { serve: (fn) => { handler = fn; } };
      if (id.includes('supabase-js')) return { createClient: () => client };
      if (id.includes('stripe@')) return Stripe;
      if (id.endsWith('/email.ts')) return { sendLicenseKeyEmail: async (data) => { emails.push(data); } };
      throw new Error(`Unexpected import: ${id}`);
    },
  });
  const session = {
    id: 'cs_fixture', client_reference_id: 'user-id', customer: 'cus_fixture', subscription: 'sub_fixture',
    metadata: { tier: 'pro' }, payment_status: 'paid', customer_details: { email: 'preview@example.test' },
  };
  return {
    writes, emails,
    async send(type = 'checkout.session.completed', overrides = {}, signature = 'fixture-signature') {
      return handler(new Request('http://fixture.invalid/webhook', {
        method: 'POST', headers: signature ? { 'stripe-signature': signature } : {},
        body: JSON.stringify({ type, data: { object: { ...session, ...overrides } } }),
      }));
    },
  };
}

test('unpaid checkout is acknowledged without changing entitlements', async () => {
  const f = fixture();
  const response = await f.send('checkout.session.completed', { payment_status: 'unpaid' });
  assert.equal(response.status, 200);
  assert.equal(f.writes.length, 0);
  assert.equal(f.emails.length, 0);
});

for (const type of ['checkout.session.completed', 'checkout.session.async_payment_succeeded']) {
  test(`${type} provisions the canonical license and retry does not issue again`, async () => {
    const f = fixture();
    assert.equal((await f.send(type)).status, 200);
    const inserts = f.writes.filter((w) => w.operation === 'insert');
    assert.equal(inserts.length, 1);
    assert.equal(inserts[0].data.tier, 'professional');
    assert.equal(inserts[0].data.user_id, 'user-id');
    assert.match(inserts[0].data.license_key, /^AIDN-PROD-PRO-[A-Fa-f0-9]{32}$/);
    assert.equal(f.emails.length, 1);
    assert.equal((await f.send(type)).status, 200);
    assert.equal(f.writes.filter((w) => w.operation === 'insert').length, 1);
    assert.equal(f.emails.length, 1);
  });
}

test('no-payment-required checkout can provision a legitimate trial', async () => {
  const f = fixture();
  assert.equal((await f.send('checkout.session.completed', { payment_status: 'no_payment_required' })).status, 200);
  assert.equal(f.writes.filter((w) => w.operation === 'insert').length, 1);
});

test('a delayed payment grants access only after its success event', async () => {
  const f = fixture();
  assert.equal((await f.send('checkout.session.completed', { payment_status: 'unpaid' })).status, 200);
  assert.equal(f.writes.length, 0);
  assert.equal((await f.send('checkout.session.async_payment_succeeded')).status, 200);
  assert.equal(f.writes.filter((w) => w.operation === 'insert').length, 1);
});

test('an unknown payment status fails closed', async () => {
  const f = fixture();
  assert.equal((await f.send('checkout.session.completed', { payment_status: null })).status, 500);
  assert.equal(f.writes.length, 0);
});

test('an issuance failure returns 500 so Stripe retries', async () => {
  const f = fixture({ insertError: { message: 'database unavailable' } });
  assert.equal((await f.send()).status, 500);
  assert.equal(f.emails.length, 0);
});

for (const signature of ['', 'invalid']) {
  test(`a ${signature || 'missing'} signature never changes entitlements`, async () => {
    const f = fixture();
    assert.equal((await f.send('checkout.session.completed', {}, signature)).status, 400);
    assert.equal(f.writes.length, 0);
  });
}
