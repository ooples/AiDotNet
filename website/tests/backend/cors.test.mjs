import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runInNewContext } from 'node:vm';
import { webcrypto } from 'node:crypto';
import ts from 'typescript';

function loadHandler(name, dependencies = {}) {
  let handler;
  const cache = new Map();
  function load(path) {
    if (cache.has(path)) return cache.get(path);
    const exports = {};
    cache.set(path, exports);
    const code = ts.transpileModule(readFileSync(path, 'utf8'), {
      compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
    }).outputText;
    runInNewContext(code, {
      exports, Request, Response, Headers, crypto: webcrypto, TextEncoder, TextDecoder, Uint8Array, atob, btoa,
      Deno: { env: { get: (key) => dependencies.env?.[key] } }, console,
      require(id) {
        if (id.includes('deno.land')) return { serve: (fn) => { handler = fn; } };
        if (id.includes('supabase-js')) return { createClient: dependencies.createClient ?? (() => { throw new Error('CORS checks must not call the backend'); }) };
        if (id.includes('stripe@')) return dependencies.Stripe ?? class Stripe {};
        if (id.startsWith('.')) return load(resolve(dirname(path), id));
        throw new Error(`Unexpected dependency: ${id}`);
      },
    }, { filename: path });
    return exports;
  }
  load(fileURLToPath(new URL(`../../supabase/functions/${name}/index.ts`, import.meta.url)));
  return handler;
}

test('billing portal returns customers to the originating approved website', async () => {
  const sessions = [];
  class Stripe {
    static createFetchHttpClient() { return {}; }
    prices = { async *list() {} };
    billingPortal = {
      configurations: { async *list() { yield { id: 'config', metadata: { tag: 'aidotnet-plans-v1', products: '[]' } }; } },
      sessions: { async create(params) { sessions.push(params); return { url: 'https://billing.stripe.com/session' }; } },
    };
  }
  const handler = loadHandler('billing-portal', {
    Stripe,
    env: { SUPABASE_URL: 'https://fixture.invalid', SUPABASE_ANON_KEY: 'anon', SUPABASE_SERVICE_ROLE_KEY: 'service', STRIPE_SECRET_KEY: 'stripe' },
    createClient: () => ({
      auth: { getUser: async () => ({ data: { user: { id: 'user' } } }) },
      from: () => ({ select: () => ({ eq: () => ({ single: async () => ({ data: { stripe_customer_id: 'customer' } }) }) }) }),
    }),
  });
  for (const origin of ['https://test.aidotnet.dev', 'https://www.aidotnet.dev', 'https://attacker.example']) {
    const response = await handler(new Request('https://fixture.invalid', {
      method: 'POST', headers: { Origin: origin, Authorization: 'Bearer session' },
      body: JSON.stringify({ action: 'manage', return_path: '/pricing/' }),
    }));
    assert.equal(response.status, 200);
    assert.equal(sessions.at(-1).return_url, `${origin.includes('attacker') ? 'https://www.aidotnet.dev' : origin}/pricing/`);
  }
});

test('billing portal rejects a duplicate cancellation scheduled through cancel_at', async () => {
  class Stripe {
    static createFetchHttpClient() { return {}; }
    prices = { async *list() {} };
    subscriptions = { async *list() { yield { id: 'subscription', status: 'trialing', created: 1, cancel_at_period_end: false, cancel_at: 200 }; } };
    billingPortal = {
      configurations: { async *list() { yield { id: 'config', metadata: { tag: 'aidotnet-plans-v1', products: '[]' } }; } },
      sessions: { async create() { throw new Error('A duplicate cancellation must not open a new portal session'); } },
    };
  }
  const handler = loadHandler('billing-portal', {
    Stripe,
    env: { SUPABASE_URL: 'https://fixture.invalid', SUPABASE_ANON_KEY: 'anon', SUPABASE_SERVICE_ROLE_KEY: 'service', STRIPE_SECRET_KEY: 'stripe' },
    createClient: () => ({
      auth: { getUser: async () => ({ data: { user: { id: 'user' } } }) },
      from: () => ({ select: () => ({ eq: () => ({ single: async () => ({ data: { stripe_customer_id: 'customer' } }) }) }) }),
    }),
  });
  const response = await handler(new Request('https://fixture.invalid', {
    method: 'POST', headers: { Authorization: 'Bearer session' }, body: JSON.stringify({ action: 'cancel' }),
  }));
  assert.equal(response.status, 409);
  assert.equal((await response.json()).error, 'already_cancelling');
});

for (const scheduled of [150, 200, 250, null]) {
  test(`moving a later cancellation before renewal: existing end ${scheduled}`, async () => {
    const updates = [];
    const subscription = { id: 'owned-subscription', status: 'trialing', created: 1, current_period_end: 200, cancel_at_period_end: false, cancel_at: scheduled,
      items: { data: [{ quantity: 1, price: { metadata: { tier: 'pro' }, recurring: { interval: 'month' }, currency: 'usd', unit_amount: 2900 } }] } };
    class Stripe {
      static createFetchHttpClient() { return {}; }
      subscriptions = {
        async *list(params) { assert.equal(params.customer, 'owned-customer'); yield subscription; },
        async update(id, params) { assert.equal(id, 'owned-subscription'); updates.push(params); return { ...subscription, cancel_at: params.cancel_at }; },
      };
    }
    const handler = loadHandler('billing-portal', {
      Stripe,
      env: { SUPABASE_URL: 'https://fixture.invalid', SUPABASE_ANON_KEY: 'anon', SUPABASE_SERVICE_ROLE_KEY: 'service', STRIPE_SECRET_KEY: 'stripe' },
      createClient: () => ({
        auth: { getUser: async () => ({ data: { user: { id: 'user' } } }) },
        from: () => ({ select: () => ({ eq: () => ({ single: async () => ({ data: { stripe_customer_id: 'owned-customer' } }) }) }) }),
      }),
    });
    const response = await handler(new Request('https://fixture.invalid', {
      method: 'POST', headers: { Authorization: 'Bearer session' }, body: JSON.stringify({ action: 'cancel_before_renewal', subscription_id: 'untrusted-other-subscription' }),
    }));
    assert.equal(response.status, scheduled === 250 ? 200 : 409);
    if (scheduled === 250) {
      assert.deepEqual(updates.map(p => ({ ...p })), [{ cancel_at: 200, proration_behavior: 'none' }]);
      assert.equal((await response.json()).subscription.cancel_at, 200);
    } else assert.equal(updates.length, 0);
  });
}

for (const scenario of [
  { name: 'refuses an ongoing subscription', subscriptions: [{ status: 'trialing' }], status: 409, calls: [] },
  { name: 'allows legacy cancellation at period end', subscriptions: [{ status: 'active', cancel_at_period_end: true }], status: 200, calls: ['revoke', 'delete'] },
  { name: 'allows the portal cancel_at date at renewal', subscriptions: [{ status: 'trialing', cancel_at: 200 }], status: 200, calls: ['revoke', 'delete'] },
  { name: 'allows cancellation before renewal', subscriptions: [{ status: 'active', cancel_at: 150 }], status: 200, calls: ['revoke', 'delete'] },
  { name: 'refuses a cancellation after another bill would occur', subscriptions: [{ status: 'active', cancel_at: 250 }], status: 409, calls: [] },
  { name: 'checks later subscriptions as well', subscriptions: [{ status: 'canceled' }, { status: 'active' }], status: 409, calls: [] },
  { name: 'fails closed when Stripe cannot be checked', subscriptions: [], stripeError: true, status: 502, calls: [] },
  { name: 'fails closed when the profile cannot be read', subscriptions: [], profileError: true, status: 503, calls: [] },
  { name: 'does not delete when license revocation fails', subscriptions: [], revokeError: true, status: 500, calls: ['revoke'] },
  { name: 'reports an Auth deletion failure after revocation', subscriptions: [], deleteError: true, status: 500, calls: ['revoke', 'delete'] },
]) {
  test(`account deletion: ${scenario.name}`, async () => {
    const calls = [];
    class Stripe {
      static createFetchHttpClient() { return {}; }
      subscriptions = { async *list() {
        if (scenario.stripeError) throw new Error('fixture Stripe failure');
        for (const subscription of scenario.subscriptions) yield { current_period_end: 200, cancel_at_period_end: false, cancel_at: null, ...subscription };
      } };
    }
    const handler = loadHandler('delete-account', {
      Stripe,
      env: { SUPABASE_URL: 'https://fixture.invalid', SUPABASE_ANON_KEY: 'anon', SUPABASE_SERVICE_ROLE_KEY: 'service', STRIPE_SECRET_KEY: 'stripe' },
      createClient: () => ({
        auth: {
          getUser: async () => ({ data: { user: { id: 'fixture-user' } } }),
          admin: { deleteUser: async (id) => { assert.equal(id, 'fixture-user'); calls.push('delete'); return { error: scenario.deleteError ? {} : null }; } },
        },
        from: (table) => table === 'profiles'
          ? { select: () => ({ eq: () => ({ maybeSingle: async () => ({ data: { stripe_customer_id: 'customer' }, error: scenario.profileError ? {} : null }) }) }) }
          : { update: () => { calls.push('revoke'); return { eq: (field, id) => { assert.equal(field, 'user_id'); assert.equal(id, 'fixture-user'); return { neq: async () => ({ error: scenario.revokeError ? {} : null }) }; } }; } },
      }),
    });
    const response = await handler(new Request('https://fixture.invalid', {
      method: 'POST', headers: { Authorization: 'Bearer session' }, body: JSON.stringify({ confirm: 'DELETE' }),
    }));
    assert.equal(response.status, scenario.status);
    assert.deepEqual(calls, scenario.calls);
  });
}

for (const name of ['billing-portal', 'delete-account', 'register-community-license', 'issue-license', 'admin-resend-license-email', 'validate-license']) {
  test(`${name}: production and preview preflight and error responses have correct CORS`, async () => {
    const handler = loadHandler(name);
    for (const origin of ['https://www.aidotnet.dev', 'https://test.aidotnet.dev']) {
      for (const method of ['OPTIONS', 'GET']) {
        const res = await handler(new Request('https://fixture.invalid/', { method, headers: { Origin: origin } }));
        assert.equal(res.status, method === 'OPTIONS' ? 204 : 405);
        assert.equal(res.headers.get('access-control-allow-origin'), origin);
        assert.match(res.headers.get('vary'), /Origin/i);
        if (method === 'OPTIONS') assert.match(res.headers.get('access-control-allow-methods'), /POST/);
      }
    }
    // Concurrent requests cannot retain another request's allowed Origin.
    const origins = ['https://test.aidotnet.dev', 'https://attacker.example', 'null', 'https://test.aidotnet.dev.attacker.example'];
    const responses = await Promise.all(origins.map((origin) => handler(new Request('https://fixture.invalid/', {
      method: 'OPTIONS', headers: { Origin: origin },
    }))));
    assert.deepEqual(responses.map((res) => res.headers.get('access-control-allow-origin')), [origins[0], null, null, null]);
  });
}
