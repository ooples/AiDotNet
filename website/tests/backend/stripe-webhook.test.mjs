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

function fixture({ insertError, profileError, rpcError, withoutIntent = false, existingLicense = false, revokedLicense = false, stripeError = false, currentStatus = 'active', currentTier = 'pro', otherLicenses = [] } = {}) {
  let handler, beforeInsert, beforeLicenseUpdate;
  const defaults = { user_id: 'user-id', product: 'aidotnet', tier: 'professional', status: 'active', stripe_subscription_id: null, stripe_checkout_session_id: null };
  const licenses = otherLicenses.map((row, i) => ({ ...defaults, id: `other-${i}`, ...row }));
  if (existingLicense || revokedLicense) licenses.push({ ...defaults, id: 'license-id', license_key: 'existing-key', stripe_subscription_id: 'sub_fixture', status: revokedLicense ? 'revoked' : 'active' });
  const profiles = [{ id: 'user-id', email: 'preview@example.test', stripe_customer_id: 'cus_fixture', subscription_tier: 'free', subscription_status: 'active' }];
  const intents = withoutIntent ? [] : [{ id: '11111111-1111-4111-8111-111111111111', user_id:'user-id', tier:'pro', created_at:'2020-01-01', expires_at:'2030-01-01' }];
  const customers = [{ customer_id:'cus_fixture', user_id:'user-id' }];
  const bindings = [];
  let linkFailure = profileError, mirrorFailure = rpcError ?? profileError;
  let status = currentStatus, tier = currentTier, unavailable = stripeError;
  const writes = [], emails = [], retrievedIds = [];
  const client = {
    async rpc(name, args) {
      if (name === 'bind_aidotnet_checkout') {
        const owner = customers.find(row=>row.customer_id===args.p_customer_id);
        const binding = bindings.find(row=>row.session_id===args.p_session_id);
        if ((owner && owner.user_id !== args.p_user_id) || (binding && (binding.user_id !== args.p_user_id || binding.customer_id !== args.p_customer_id))) return {error:{message:'ownership conflict'}};
        if (args.p_customer_id && !owner) customers.push({customer_id:args.p_customer_id,user_id:args.p_user_id});
        if (!binding) bindings.push({session_id:args.p_session_id,user_id:args.p_user_id,customer_id:args.p_customer_id});
        return {error:null};
      }
      assert.equal(name, 'sync_aidotnet_billing_profile');
      if (mirrorFailure) return { error: mirrorFailure };
      for (const profile of profiles.filter(p => args.p_user_id ? p.id === args.p_user_id : (p.stripe_customer_id === args.p_customer_id || customers.some(c=>c.user_id===p.id&&c.customer_id===args.p_customer_id)))) {
        const active = licenses.filter(l => l.user_id === profile.id && l.product === 'aidotnet' && l.status === 'active');
        const rank = Math.max(-1, ...active.map(l => ({ enterprise: 2, professional: 1, community: 0 }[l.tier] ?? 0)));
        if (!profile.stripe_customer_id && customers.some(c=>c.user_id===profile.id&&c.customer_id===args.p_customer_id)) profile.stripe_customer_id=args.p_customer_id;
        const data = { subscription_tier: rank === 2 ? 'enterprise' : rank === 1 ? 'pro' : 'free', subscription_status: rank >= 0 ? 'active' : args.p_fallback_status };
        Object.assign(profile, data); writes.push({ table: 'profiles', operation: 'update', data });
      }
      return { error: null };
    },
    from(table) {
      const filters = [];
      const rows = ({profiles,license_keys:licenses,billing_customers:customers,billing_checkout_intents:intents,billing_checkout_bindings:bindings})[table];
      assert.ok(rows, 'Unknown table '+table);
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
        maybeSingle() { return Promise.resolve({ data: rows.find(matches) ?? null, error: table === 'profiles' ? linkFailure ?? null : null }); },
        async then(resolve, reject) {
          try {
            if (table === 'license_keys' && query.change && beforeLicenseUpdate) { const hook = beforeLicenseUpdate; beforeLicenseUpdate = null; await hook(); }
            if (table === 'profiles' && linkFailure) return resolve({ data: null, error: linkFailure });
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
      if (id.endsWith('/checkout-ownership.ts')) {
        const exports = {};
        const source = readFileSync(new URL('../../supabase/functions/_shared/checkout-ownership.ts', import.meta.url),'utf8');
        runInNewContext(ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText,{exports,console},{filename:'checkout-ownership.ts'});
        return exports;
      }
      if (id.endsWith('/email.ts')) return { sendLicenseKeyEmail: async data => { emails.push(data); } };
      throw new Error(`Unexpected import: ${id}`);
    },
  });
  const session = { id: 'cs_fixture', created: 1700000000, client_reference_id: '11111111-1111-4111-8111-111111111111', customer: 'cus_fixture', subscription: 'sub_fixture', metadata: { tier: 'pro' }, payment_status: 'paid', customer_details: { email: 'preview@example.test' } };
  return {
    writes, emails, licenses, profiles, retrievedIds, customers, bindings, intents,
    setProfileError(value) { linkFailure=value; }, setMirrorError(value) { mirrorFailure=value; },
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
    assert.equal(f.licenses.length, event === 'checkout.session.completed' ? 1 : 0);
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
  f.profiles[0].stripe_customer_id = null;
  assert.equal((await f.send()).status, 200);
  assert.equal(f.profiles[0].stripe_customer_id, 'cus_fixture');
  assert.ok(f.writes.some(w => w.table === 'profiles' && w.data.stripe_customer_id === 'cus_fixture'));
  assert.equal(f.licenses.length, 1);
  assert.equal(f.writes.filter(w => w.operation === 'insert').length, 0);
});

for (const reference of ['user-id','22222222-2222-4222-8222-222222222222',null]) test(`public Checkout fields cannot select an account: ${reference}`, async () => {
  const f = fixture();
  assert.equal((await f.send('checkout.session.completed',{client_reference_id:reference})).status,500);
  assert.equal(f.licenses.length,0);
  assert.equal(f.writes.length,0);
});

for (const [name, created] of [
  ['after the intent expires', Date.parse('2031-01-01') / 1000],
  ['before the intent creation tolerance', Date.parse('2020-01-01') / 1000 - 2],
]) test(`Checkout created ${name} cannot write an entitlement`, async () => {
  const f = fixture();
  assert.equal((await f.send('checkout.session.completed', { created })).status, 500);
  assert.equal(f.writes.length, 0);
});

test('a stored Checkout binding rejects a different customer without writes', async () => {
  const f = fixture();
  f.bindings.push({session_id:'cs_fixture',user_id:'user-id',customer_id:'cus_other'});
  assert.equal((await f.send()).status, 500);
  assert.equal(f.writes.length, 0);
});

test('a customer already owned by another account cannot be rebound through Checkout', async () => {
  const f = fixture();
  f.customers[0].user_id='another-user';
  assert.equal((await f.send()).status,500);
  assert.equal(f.licenses.length,0);
});

test('an old Checkout cannot replace a newer preferred customer or remove its grant', async () => {
  const f = fixture({currentStatus:'canceled',otherLicenses:[{stripe_subscription_id:'sub_new',stripe_customer_id:'cus_new'}]});
  f.profiles[0].stripe_customer_id='cus_new';
  f.customers.push({customer_id:'cus_new',user_id:'user-id'});
  assert.equal((await f.send()).status,200);
  assert.equal(f.profiles[0].stripe_customer_id,'cus_new');
  assert.equal(f.profiles[0].subscription_tier,'pro');
  assert.equal(f.customers.length,2);
});

test('a failed profile link repair leaves a durable key recoverable by invoice alone', async () => {
  const f = fixture({profileError:{message:'link unavailable'}});
  f.profiles[0].stripe_customer_id=null;
  assert.equal((await f.send()).status,500);
  const key=f.licenses[0].id;
  assert.equal(f.licenses[0].status,'suspended');
  f.setProfileError(null); f.setMirrorError(null);
  assert.equal((await f.send('invoice.paid')).status,200);
  assert.equal(f.licenses[0].id,key);
  assert.equal(f.licenses[0].status,'active');
  assert.equal(f.profiles[0].stripe_customer_id,'cus_fixture');
  assert.equal(f.profiles[0].subscription_tier,'pro');
});

test('an RPC failure after a committed license change recovers the same key on replay', async () => {
  const f=fixture({existingLicense:true,currentTier:'enterprise',rpcError:{message:'mirror unavailable'}});
  assert.equal((await f.send('invoice.paid')).status,500);
  assert.equal(f.licenses[0].tier,'enterprise');
  assert.equal(f.profiles[0].subscription_tier,'free');
  f.setMirrorError(null);
  assert.equal((await f.send('invoice.paid')).status,200);
  assert.equal(f.licenses.length,1);
  assert.equal(f.licenses[0].id,'license-id');
  assert.equal(f.profiles[0].subscription_tier,'enterprise');
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
