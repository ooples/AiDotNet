import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runInNewContext } from 'node:vm';
import { webcrypto, createHmac } from 'node:crypto';
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
      exports, Request, Response, Headers, URL, crypto: webcrypto, TextEncoder, TextDecoder, Uint8Array, atob, btoa,
      Deno: { env: { get: (key) => dependencies.env?.[key] } }, console,
      fetch: dependencies.fetch ?? (() => { throw new Error('Unexpected network access in a handler contract'); }),
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

function queryRows(rows) {
  let selected = rows;
  const query = {
    select: () => query,
    eq: (key,value) => { selected=selected.filter(row=>row[key]===value); return query; },
    single: async () => ({ data:selected[0] ?? null,error:null }),
    maybeSingle: async () => ({ data:selected[0] ?? null,error:null }),
    then: resolve => resolve({data:selected,error:null}),
  };
  return query;
}

test('billing portal returns customers to the originating approved website', async () => {
  const sessions = [];
  class Stripe {
    static createFetchHttpClient() { return {}; }
    prices = { async *list() {} };
    subscriptions = { async *list() {} };
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
      from: table => table === 'billing_customers' ? queryRows([{user_id:'user',customer_id:'customer'}]) : queryRows([{id:'user',stripe_customer_id:'customer'}]),
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
      from: table => table === 'billing_customers' ? queryRows([{user_id:'user',customer_id:'customer'}]) : queryRows([{id:'user',stripe_customer_id:'customer'}]),
    }),
  });
  const response = await handler(new Request('https://fixture.invalid', {
    method: 'POST', headers: { Authorization: 'Bearer session' }, body: JSON.stringify({ action: 'cancel' }),
  }));
  assert.equal(response.status, 409);
  assert.equal((await response.json()).error, 'already_cancelling');
});

for (const scenario of [
  { scheduled: 150, expectedError: 'no_later_cancellation' },
  { scheduled: 200, expectedError: 'no_later_cancellation' },
  { scheduled: 250, expectedError: undefined },
  { scheduled: null, expectedError: 'no_later_cancellation' },
  { scheduled: 250, expectedId: 'other-subscription', expectedError: 'billing_state_changed' },
  { scheduled: 250, expectedRenewal: 199, expectedError: 'billing_state_changed' },
  { scheduled: 250, expectedId: null, expectedError: 'billing_state_changed' },
  { scheduled: 250, expectedRenewal: null, expectedError: 'billing_state_changed' },
  { scheduled: 250, expectedCancelAt: 260, expectedError: 'billing_state_changed' },
  { scheduled: 250, expectedCancelAt: undefined, expectedError: 'billing_state_changed' },
]) {
  const { scheduled } = scenario;
  const mismatch = ['expectedId','expectedRenewal','expectedCancelAt'].filter(key => key in scenario).map(key => `${key}=${scenario[key]}`).join(', ');
  test(`moving cancellation with end ${scheduled}: ${mismatch || 'matching snapshot'}`, async () => {
    const updates = [];
    const subscription = { id: 'owned-subscription', status: 'trialing', created: 1, current_period_end: 200, cancel_at_period_end: false, cancel_at: scheduled,
      items: { data: [{ quantity: 1, price: { metadata: { tier: 'pro' }, recurring: { interval: 'month' }, currency: 'usd', unit_amount: 2900 } }] } };
    class Stripe {
      static createFetchHttpClient() { return {}; }
      prices = { async *list() {} };
      billingPortal = {
        configurations: { async *list() { yield { id:'config',metadata:{tag:'aidotnet-plans-v1',products:'[]'} }; } },
        sessions: { async create() { return {url:'https://billing.stripe.com/review'}; } },
      };
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
        from: table => table === 'billing_customers' ? queryRows([{user_id:'user',customer_id:'owned-customer'}]) : queryRows([{id:'user',stripe_customer_id:'owned-customer'}]),
      }),
    });
    const response = await handler(new Request('https://fixture.invalid', {
      method: 'POST', headers: { Authorization: 'Bearer session' }, body: JSON.stringify({ action: 'cancel_before_renewal', expected_subscription_id: 'expectedId' in scenario ? scenario.expectedId : 'owned-subscription', expected_renewal: 'expectedRenewal' in scenario ? scenario.expectedRenewal : 200, expected_cancel_at: 'expectedCancelAt' in scenario ? scenario.expectedCancelAt : scheduled }),
    }));
    const body = await response.json();
    assert.equal(response.status, scenario.expectedError ? 409 : 200);
    if (!scenario.expectedError) {
      assert.equal(updates.length,0,'Stripe owns the final cancellation write');
      assert.equal(body.url,'https://billing.stripe.com/review');
    } else {
      assert.equal(body.error, scenario.expectedError);
      assert.equal(updates.length, 0);
    }
  });
}

for (const scenario of [
  { name: 'refuses an ongoing subscription', subscriptions: [{ status: 'trialing' }], status: 409, calls: [] },
  { name: 'allows legacy cancellation at period end', subscriptions: [{ status: 'active', cancel_at_period_end: true }], status: 200, calls: ['revoke', 'delete'] },
  { name: 'allows the portal cancel_at date at renewal', subscriptions: [{ status: 'trialing', cancel_at: 200 }], status: 200, calls: ['revoke', 'delete'] },
  { name: 'allows cancellation before renewal', subscriptions: [{ status: 'active', cancel_at: 150 }], status: 200, calls: ['revoke', 'delete'] },
  { name: 'refuses a cancellation after another bill would occur', subscriptions: [{ status: 'active', cancel_at: 250 }], status: 409, calls: [] },
  { name: 'checks later subscriptions as well', subscriptions: [{ status: 'canceled' }, { status: 'active' }], status: 409, calls: [] },
  { name: 'checks a replacement customer while the preferred customer is canceled', customers:['customer','replacement'], subscriptions:[], byCustomer:{customer:[{status:'canceled'}],replacement:[{status:'active'}]}, status:409,calls:[] },
  { name: 'fails closed when Stripe cannot be checked', subscriptions: [], stripeError: true, status: 502, calls: [] },
  { name: 'fails closed when the profile cannot be read', subscriptions: [], profileError: true, status: 503, calls: [] },
  { name: 'does not delete when license revocation fails', subscriptions: [], revokeError: true, status: 500, calls: ['revoke'] },
  { name: 'reports an Auth deletion failure after revocation', subscriptions: [], deleteError: true, status: 500, calls: ['revoke', 'delete'] },
]) {
  test(`account deletion: ${scenario.name}`, async () => {
    const calls = [];
    class Stripe {
      static createFetchHttpClient() { return {}; }
      subscriptions = { async *list(params) {
        if (scenario.stripeError) throw new Error('fixture Stripe failure');
        for (const subscription of scenario.byCustomer?.[params.customer] ?? scenario.subscriptions) yield { current_period_end: 200, cancel_at_period_end: false, cancel_at: null, ...subscription };
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
        from: (table) => table === 'billing_customers' ? queryRows((scenario.customers ?? ['customer']).map(customer_id=>({user_id:'fixture-user',customer_id}))) : table === 'profiles'
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

for (const existing of [false, true]) test(`Community issuance uses the serialized RPC (existing=${existing})`, async () => {
  let issuance;
  const client = {
    auth: { getUser: async () => ({ data: { user: { id: 'community-user' } } }) },
    from: () => {
      const query = { select: () => query, eq: () => query, order: () => query, limit: () => query, maybeSingle: async () => ({ data: null, error: null }) };
      return query;
    },
    rpc: async (name, args) => {
      assert.equal(name, 'issue_aidotnet_community_license');
      issuance = args;
      return { data: { id: 'community-license', license_key: existing ? 'existing-key' : args.p_license_key, is_existing: existing }, error: null };
    },
  };
  const handler = loadHandler('register-community-license', {
    env: { SUPABASE_URL: 'https://fixture.invalid', SUPABASE_ANON_KEY: 'anon', SUPABASE_SERVICE_ROLE_KEY: 'service' },
    createClient: () => client,
  });
  const response = await handler(new Request('https://fixture.invalid', { method: 'POST', headers: { Authorization: 'Bearer user-token', Origin: 'https://test.aidotnet.dev' } }));
  assert.equal(response.status, existing ? 200 : 201);
  const body = await response.json();
  assert.equal(body.is_existing, existing);
  assert.equal(body.license_key, existing ? 'existing-key' : issuance.p_license_key);
  assert.equal(issuance.p_user_id, 'community-user');
  assert.match(issuance.p_license_key, /^AIDN-PROD-COMMUNITY-[0-9a-f]{32}$/);
});

function billingOwnershipFixture(options={}) {
  const user={id:'signed-in-user',email:'owner@example.test',email_confirmed_at:'2026-01-01',...options.user};
  const customers=options.customers ?? [];
  const bindings=options.bindings ?? [];
  const intents=[], deliveries=[];
  const session={id:'cs_test_claim',created:1700000000,status:'complete',payment_status:'paid',client_reference_id:null,
    customer:'cus_claim',customer_details:{email:'owner@example.test'},metadata:{tier:'pro'},...options.session};
  const client={
    auth:{getUser:async()=>({data:{user}})},
    from(table) {
      if(table==='profiles') return queryRows([{id:user.id,stripe_customer_id:options.preferred ?? null}]);
      if(table==='billing_customers') return queryRows(customers);
      if(table==='billing_checkout_bindings') return queryRows(bindings);
      if(table==='billing_checkout_intents') return {insert:row=>{const saved={id:'11111111-1111-4111-8111-111111111111',...row};intents.push(saved);return queryRows([saved]);}};
      throw new Error('Unexpected table '+table);
    },
    async rpc(name,args) {
      assert.equal(name,'bind_aidotnet_checkout');
      const customer=customers.find(c=>c.customer_id===args.p_customer_id);
      if(customer&&customer.user_id!==args.p_user_id) return {error:{message:'foreign customer'}};
      if(!customer&&args.p_customer_id) customers.push({customer_id:args.p_customer_id,user_id:args.p_user_id});
      bindings.push({session_id:args.p_session_id,user_id:args.p_user_id,customer_id:args.p_customer_id});
      return {error:null};
    },
  };
  class Stripe {
    static createFetchHttpClient(){return {};}
    subscriptions={async *list(params){for(const sub of options.subscriptions?.[params.customer] ?? []) yield sub;}};
    checkout={sessions:{retrieve:async id=>{assert.equal(id,session.id);return session;}}};
  }
  const handler=loadHandler('billing-portal',{
    Stripe,createClient:()=>client,
    env:{SUPABASE_URL:'https://fixture.invalid',SUPABASE_ANON_KEY:'anon',SUPABASE_SERVICE_ROLE_KEY:'service',STRIPE_SECRET_KEY:'stripe',STRIPE_WEBHOOK_SECRET:'fixture-signing-secret'},
    fetch:async(url,request)=>{
      assert.equal(url,`https://fixture.invalid/functions/v1/stripe-webhook${options.preview ? '-preview' : ''}`);
      const signature=request.headers['stripe-signature'];
      const timestamp=signature.match(/^t=(\d+),v1=/)?.[1];
      assert.equal(signature,`t=${timestamp},v1=${createHmac('sha256','fixture-signing-secret').update(timestamp+'.'+request.body).digest('hex')}`);
      deliveries.push(JSON.parse(request.body));
      return Response.json({received:true});
    },
  });
  return {intents,deliveries,bindings,customers,send:body=>handler(new Request(`https://fixture.invalid/functions/v1/billing-portal${options.preview ? '-preview' : ''}`,{method:'POST',headers:{Authorization:'Bearer owner-session'},body:JSON.stringify(body)}))};
}

test('Checkout references bind the authenticated account, ignoring browser-supplied identity',async()=>{
  const f=billingOwnershipFixture();
  const response=await f.send({action:'checkout_reference',tier:'pro',interval:'month',user_id:'victim'});
  assert.equal(response.status,200);
  assert.equal(f.intents[0].user_id,'signed-in-user');
  assert.equal((await response.json()).reference,f.intents[0].id);
});
test('another owned subscription blocks Checkout even when an inactive customer is selected',async()=>{
  const f=billingOwnershipFixture({customers:[{user_id:'signed-in-user',customer_id:'old'},{user_id:'signed-in-user',customer_id:'new'}],subscriptions:{old:[],new:[{id:'sub_new',status:'active',created:1}]}});
  assert.equal((await f.send({action:'checkout_reference',customer_id:'old',tier:'pro',interval:'month'})).status,409);
  assert.equal(f.intents.length,0);
});
test('a verified account can explicitly claim an unbound completed checkout',async()=>{
  const f=billingOwnershipFixture();
  assert.equal((await f.send({action:'claim_checkout',checkout_session_id:'cs_test_claim'})).status,200);
  assert.equal(f.bindings[0].user_id,'signed-in-user');
  assert.equal(f.deliveries.length,1);
  assert.equal(f.deliveries[0].data.object.id,'cs_test_claim');
});
test('a preview claim fulfills through the matching preview webhook',async()=>{
  const f=billingOwnershipFixture({preview:true});
  assert.equal((await f.send({action:'claim_checkout',checkout_session_id:'cs_test_claim'})).status,200);
  assert.equal(f.deliveries.length,1);
});
for(const scenario of [
  {name:'unverified email',user:{email_confirmed_at:null},status:403},
  {name:'another payer email',session:{customer_details:{email:'another@example.test'}},status:403},
  {name:'unpaid checkout',session:{payment_status:'unpaid'},status:409},
  {name:'unfinished checkout',session:{status:'open'},status:409},
  {name:'another bound account',bindings:[{session_id:'cs_test_claim',customer_id:'cus_claim',user_id:'another-user'}],status:403},
]) test(`claim refuses ${scenario.name} without fulfillment`,async()=>{
  const f=billingOwnershipFixture(scenario);
  assert.equal((await f.send({action:'claim_checkout',checkout_session_id:'cs_test_claim'})).status,scenario.status);
  assert.equal(f.deliveries.length,0);
});

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
