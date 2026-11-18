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

function fixture({ insertError, profileError, existingLicense = false, revokedLicense = false, stripeError = false, currentStatus = 'active', currentTier = 'pro', otherLicenses = [] } = {}) {
  let handler, beforeInsert, beforeLicenseUpdate;
  const defaults = { user_id: 'user-id', product: 'aidotnet', tier: 'professional', status: 'active', stripe_subscription_id: null, stripe_checkout_session_id: null };
  const licenses = otherLicenses.map((row, i) => ({ ...defaults, id: `other-${i}`, ...row }));
  if (existingLicense || revokedLicense) licenses.push({ ...defaults, id: 'license-id', license_key: 'existing-key', stripe_subscription_id: 'sub_fixture', status: revokedLicense ? 'revoked' : 'active' });
  const profiles = [{ id: 'user-id', email: 'preview@example.test', stripe_customer_id: 'cus_fixture', subscription_tier: 'free', subscription_status: 'active' }];
  let status = currentStatus, tier = currentTier, unavailable = stripeError;
  const writes = [], emails = [], retrievedIds = [];
  const client = {
    async rpc(name, args) {
      assert.equal(name, 'sync_aidotnet_billing_profile');
      if (profileError) return { error: profileError };
      for (const profile of profiles.filter(p => args.p_user_id ? p.id === args.p_user_id : p.stripe_customer_id === args.p_customer_id)) {
        const active = licenses.filter(l => l.user_id === profile.id && l.product === 'aidotnet' && l.status === 'active');
        const rank = Math.max(-1, ...active.map(l => ({ enterprise: 2, professional: 1, community: 0 }[l.tier] ?? 0)),
          args.p_fallback_status === 'active' ? ({ enterprise: 2, pro: 1, free: 0 }[args.p_fallback_tier] ?? 0) : -1);
        const data = { subscription_tier: rank === 2 ? 'enterprise' : rank === 1 ? 'pro' : 'free', subscription_status: rank >= 0 ? 'active' : args.p_fallback_status };
        Object.assign(profile, data); writes.push({ table: 'profiles', operation: 'update', data });
      }
      return { error: null };
    },
    from(table) {
      const filters = [];
      const rows = table === 'profiles' ? profiles : licenses;
      const matches = row => filters.every(([op,k,v]) => op === 'neq' ? row[k] !== v : op === 'is' ? (row[k] ?? null) === v : row[k] === v);
      const query = {
        select() { return query; }, eq(k,v) { filters.push(['eq',k,v]); return query; },
        neq(k,v) { filters.push(['neq',k,v]); return query; }, is(k,v) { filters.push(['is',k,v]); return query; },
        limit() { return query; },
        update(data) { writes.push({ table, operation: 'update', data }); query.change = data; return query; },
        async insert(data) {
          writes.push({ table, operation: 'insert', data });
          if (beforeInsert) { const hook = beforeInsert; beforeInsert = null; await hook(); }
          if (insertError) return { error: insertError };
          if (licenses.some(l => l.product === data.product && ((data.stripe_subscription_id && l.stripe_subscription_id === data.stripe_subscription_id) || (data.stripe_checkout_session_id && l.stripe_checkout_session_id === data.stripe_checkout_session_id)))) return { error: { code: '23505' } };
          licenses.push({ ...defaults, id: `license-${licenses.length}`, ...data });
          return { error: null };
        },
        maybeSingle() { return Promise.resolve({ data: rows.find(matches) ?? null, error: table === 'profiles' ? profileError ?? null : null }); },
        async then(resolve, reject) {
          try {
            if (table === 'license_keys' && query.change && beforeLicenseUpdate) { const hook = beforeLicenseUpdate; beforeLicenseUpdate = null; await hook(); }
            if (table === 'profiles' && profileError) return resolve({ data: null, error: profileError });
            if (query.change) for (const row of rows.filter(matches)) Object.assign(row, query.change);
            return resolve({ data: query.change ? null : rows.filter(matches), error: null });
          } catch (error) { return reject(error); }
        },
      };
      return query;
    },
  };
  class Stripe {
    static createFetchHttpClient() { return {}; }
    subscriptions = { retrieve: async id => {
      retrievedIds.push(id); assert.equal(id, 'sub_fixture', 'only the requested fixture subscription may be retrieved');
      if (unavailable) throw new Error('Stripe unavailable');
      return { id, customer: 'cus_fixture', status, items: { data: [{ price: { metadata: { tier } } }] } };
    } };
    webhooks = { constructEventAsync: async (body, signature) => {
      if (signature !== 'fixture-signature') throw new Error('invalid signature');
      return JSON.parse(body);
    } };
  }
  const env = { STRIPE_SECRET_KEY: 'sk_test_fixture', STRIPE_WEBHOOK_SECRET: 'whsec_fixture', SUPABASE_URL: 'http://fixture.invalid', SUPABASE_SERVICE_ROLE_KEY: 'fixture-service' };
  runInNewContext(code, {
    exports: {}, Request, Response, crypto: webcrypto, Uint8Array, console: { log() {}, warn() {}, error() {} },
    Deno: { env: { get: key => env[key] } },
    require(id) {
      if (id.includes('deno.land')) return { serve: fn => { handler = fn; } };
      if (id.includes('supabase-js')) return { createClient: () => client };
      if (id.includes('stripe@')) return Stripe;
      if (id.endsWith('/email.ts')) return { sendLicenseKeyEmail: async data => { emails.push(data); } };
      throw new Error(`Unexpected import: ${id}`);
    },
  });
  const session = { id: 'cs_fixture', client_reference_id: 'user-id', customer: 'cus_fixture', subscription: 'sub_fixture', metadata: { tier: 'pro' }, payment_status: 'paid', customer_details: { email: 'preview@example.test' } };
  return {
    writes, emails, licenses, profiles, retrievedIds,
    setSubscription(nextStatus, nextTier = tier) { status = nextStatus; tier = nextTier; },
    setStripeError(value) { unavailable = value; },
    setBeforeInsert(hook) { beforeInsert = hook; }, setBeforeLicenseUpdate(hook) { beforeLicenseUpdate = hook; },
    async send(type = 'checkout.session.completed', overrides = {}, signature = 'fixture-signature') {
      return handler(new Request('http://fixture.invalid/webhook', { method: 'POST', headers: signature ? { 'stripe-signature': signature } : {}, body: JSON.stringify({ type, data: { object: { ...session, ...overrides } } }) }));
    },
  };
}

for (const event of ['checkout.session.completed', 'checkout.session.async_payment_succeeded']) {
  test(`a late ${event} cannot grant access after subscription deletion`, async () => {
    const f = fixture({ currentStatus: 'canceled' });
    assert.equal((await f.send('customer.subscription.deleted', { id: 'sub_fixture' })).status, 200);
    assert.equal((await f.send(event)).status, 200);
    assert.ok(f.licenses.every(license => license.status === 'expired'));
    assert.equal(f.writes.some(w => w.data.status === 'active' || w.data.subscription_status === 'active'), false);
    assert.equal(f.emails.length, 0);
  });
}

for (const event of ['invoice.paid', 'customer.subscription.updated']) {
  test(`a late ${event} reconciles the canceled subscription instead of reactivating it`, async () => {
    const f = fixture({ currentStatus: 'canceled' });
    assert.equal((await f.send(event, { id: 'sub_fixture', status: 'active' })).status, 200);
    assert.equal(f.writes.some(w => w.data.status === 'active' || w.data.subscription_status === 'active'), false);
    assert.ok(f.writes.some(w => w.data.status === 'expired'));
  });
}

test('Stripe lookup failure leaves a disabled durable receipt that can recover without Checkout replay', async () => {
  const f = fixture({ stripeError: true });
  assert.equal((await f.send()).status, 500);
  assert.equal(f.licenses.length, 1);
  assert.equal(f.licenses[0].status, 'suspended');
  f.setStripeError(false);
  assert.equal((await f.send('invoice.paid')).status, 200);
  assert.equal(f.licenses[0].status, 'active');
});

test('replayed checkout cannot replace a revoked subscription license', async () => {
  const f = fixture({ revokedLicense: true });
  assert.equal((await f.send()).status, 200);
  assert.equal(f.writes.length, 0);
  assert.equal(f.emails.length, 0);
});

test('a modern invoice subscription reference is reconciled as well', async () => {
  const f = fixture({ currentStatus: 'canceled' });
  assert.equal((await f.send('invoice.paid', { subscription: undefined, parent: { subscription_details: { subscription: 'sub_fixture' } } })).status, 200);
  assert.ok(f.writes.some(w => w.data.status === 'expired'));
  assert.equal(f.writes.some(w => w.data.status === 'active'), false);
});

for (const event of ['checkout.session.completed', 'customer.subscription.updated', 'customer.subscription.deleted']) {
  test(`a profile failure during ${event} is retried instead of acknowledging stale billing state`, async () => {
    const f = fixture({ profileError: { message: 'profile write unavailable' } });
    assert.equal((await f.send(event, { id: 'sub_fixture' })).status, 500);
    assert.equal(f.writes.filter(w => w.operation === 'insert').length, 0);
    assert.equal(f.emails.length, 0);
  });
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
  const f = fixture({ currentStatus: 'trialing' });
  assert.equal((await f.send('checkout.session.completed', { payment_status: 'no_payment_required' })).status, 200);
  assert.equal(f.writes.filter((w) => w.operation === 'insert').length, 1);
});

for (const recovery of ['invoice.paid', 'customer.subscription.updated']) {
  test(`a deferred paid entitlement recovers through ${recovery} without another Checkout event`, async () => {
    const f = fixture({ currentStatus: 'past_due' });
    assert.equal((await f.send('checkout.session.async_payment_succeeded')).status, 200);
    assert.equal(f.licenses.length, 1);
    assert.equal(f.licenses[0].status, 'suspended');
    f.setSubscription('active');
    assert.equal((await f.send(recovery, { id: 'sub_fixture' })).status, 200);
    assert.equal(f.licenses.length, 1);
    assert.equal(f.licenses[0].status, 'active');
  });
}

test('Checkout replay repairs the customer link even when a subscription license already exists', async () => {
  const f = fixture({ existingLicense: true });
  assert.equal((await f.send()).status, 200);
  assert.ok(f.writes.some(w => w.table === 'profiles' && w.data.stripe_customer_id === 'cus_fixture'));
  assert.equal(f.licenses.length, 1);
  assert.equal(f.writes.filter(w => w.operation === 'insert').length, 0);
});

test('Checkout replay after a tier change reconciles the one subscription license', async () => {
  const f = fixture();
  assert.equal((await f.send()).status, 200);
  f.setSubscription('active', 'enterprise');
  assert.equal((await f.send()).status, 200);
  assert.equal(f.licenses.length, 1);
  assert.equal(f.licenses[0].tier, 'enterprise');
  assert.equal((await f.send('customer.subscription.updated', { id: 'sub_fixture' })).status, 200);
  assert.equal(f.licenses.length, 1);
  assert.equal(f.licenses[0].tier, 'enterprise');
});

test('a late checkout uses the current plan rather than stale Checkout tier metadata', async () => {
  const f = fixture({ currentTier: 'enterprise' });
  assert.equal((await f.send()).status, 200);
  assert.equal(f.licenses[0].tier, 'enterprise');
  assert.ok(f.writes.filter(w => w.table === 'license_keys' && w.data.status === 'active').every(w => w.data.tier === 'enterprise'));
});

test('a delayed payment grants access only after its success event', async () => {
  const f = fixture();
  assert.equal((await f.send('checkout.session.completed', { payment_status: 'unpaid' })).status, 200);
  assert.equal(f.writes.length, 0);
  assert.equal((await f.send('checkout.session.async_payment_succeeded')).status, 200);
  assert.equal(f.writes.filter((w) => w.operation === 'insert').length, 1);
});

for (const prior of [null, 'sub_other']) test(`a paid subscription keeps its own entitlement beside existing grant ${prior}`, async () => {
  const f = fixture({ currentStatus: 'past_due', otherLicenses: [{ stripe_subscription_id: prior }] });
  assert.equal((await f.send()).status, 200);
  assert.equal(f.licenses.length, 2);
  assert.equal(f.licenses.find(l => l.stripe_subscription_id === 'sub_fixture').status, 'suspended');
  f.setSubscription('active');
  assert.equal((await f.send('invoice.paid')).status, 200);
  assert.equal(f.licenses.filter(l => l.status === 'active').length, 2);
  f.setSubscription('canceled');
  assert.equal((await f.send('customer.subscription.deleted', { id: 'sub_fixture' })).status, 200);
  assert.equal(f.licenses.filter(l => l.status === 'active').length, 1);
  assert.equal(f.profiles[0].subscription_tier, 'pro');
  assert.equal(f.profiles[0].subscription_status, 'active');
});

for (const recovery of ['active', 'canceled']) test(`Checkout reconciles a ${recovery} event that completed before its insert`, async () => {
  const f = fixture({ currentStatus: 'past_due' });
  f.setBeforeInsert(async () => {
    assert.equal(f.licenses.length, 0);
    f.setSubscription(recovery);
    assert.equal((await f.send(recovery === 'active' ? 'invoice.paid' : 'customer.subscription.deleted', { id: 'sub_fixture' })).status, 200);
  });
  assert.equal((await f.send()).status, 200);
  assert.equal(f.licenses.length, 1);
  assert.equal(f.licenses[0].status, recovery === 'active' ? 'active' : 'expired');
  if (recovery === 'canceled') assert.equal(f.emails.length, 0);
});

test('a cancellation interleaving reconciliation cannot be overwritten by the older active snapshot', async () => {
  const f = fixture();
  f.setBeforeLicenseUpdate(async () => {
    f.setSubscription('canceled');
    assert.equal((await f.send('customer.subscription.deleted', { id: 'sub_fixture' })).status, 200);
  });
  assert.equal((await f.send()).status, 200);
  assert.equal(f.licenses[0].status, 'expired');
  assert.equal(f.profiles[0].subscription_tier, 'free');
  assert.equal(f.profiles[0].subscription_status, 'expired');
  assert.equal(f.emails.length, 0);
});

test('one-time paid receipts remain independent and idempotent beside manual grants', async () => {
  const f = fixture({ otherLicenses: [{}] });
  assert.equal((await f.send('checkout.session.completed', { subscription: null })).status, 200);
  assert.equal(f.licenses.length, 2);
  assert.equal((await f.send('checkout.session.completed', { subscription: null })).status, 200);
  assert.equal(f.licenses.length, 2);
  assert.equal(f.emails.length, 1);
});

test('the Stripe boundary rejects an unknown subscription instead of returning the fixture', async () => {
  const f = fixture();
  assert.equal((await f.send('invoice.paid', { subscription: 'sub_wrong' })).status, 500);
  assert.deepEqual(f.retrievedIds, ['sub_wrong']);
  assert.equal(f.writes.length, 0);
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
