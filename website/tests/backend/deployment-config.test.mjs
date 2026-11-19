import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { configureStripeWebhook } from '../../scripts/configure-stripe-webhook.mjs';
import { syncFunctionMailKey } from '../../scripts/sync-function-mail-key.mjs';

const target = 'https://yfkqwpgjahoamlgckjib.supabase.co/functions/v1/stripe-webhook';
const event = 'checkout.session.async_payment_succeeded';
test('Stripe verification preserves events by refusing an unsafe list replacement', async () => {
  const endpoint = { id: 'we_fixture', url: target, status: 'enabled', livemode: true, enabled_events: ['invoice.paid', 'customer.subscription.deleted'] };
  let posts = 0;
  await assert.rejects(configureStripeWebhook({ key: 'fixture', fetcher: async (url, options) => {
    if (options.method === 'POST') {
      posts++;
      assert.fail('Verification must not write to any Stripe endpoint');
    }
    assert.equal(url, 'https://api.stripe.com/v1/webhook_endpoints?limit=100');
    return Response.json(url.includes('?') ? { data: [endpoint], has_more: false } : endpoint);
  } }), /No configuration was changed/);
  assert.equal(posts, 0);
  assert.deepEqual(endpoint.enabled_events, ['invoice.paid', 'customer.subscription.deleted']);
});
for (const enabled_events of [['*'], [event]]) {
  test(`Stripe configuration does not rewrite an already subscribed endpoint: ${enabled_events}`, async () => {
    const result = await configureStripeWebhook({ key: 'fixture', fetcher: async (_, options) => {
      assert.equal(options.method, undefined);
      return Response.json({ data: [{ id: 'we_fixture', url: target, status: 'enabled', livemode: true, enabled_events }], has_more: false });
    } });
    assert.equal(result.changed, false);
  });
}
test('Stripe configuration cannot alter another product or create a missing endpoint', async () => {
  await assert.rejects(configureStripeWebhook({ key: 'fixture', fetcher: async (_, options) => {
    assert.equal(options.method, undefined);
    return Response.json({ data: [{ id: 'we_other', url: 'https://other.example/webhook', status: 'enabled', enabled_events: [] }], has_more: false });
  } }), /found 0/);
});
test('live Stripe verification rejects a sandbox endpoint even when already subscribed', async () => {
  await assert.rejects(configureStripeWebhook({ key: 'fixture', fetcher: async (_, options) => {
    assert.equal(options.method, undefined);
    return Response.json({ data: [{ id: 'we_sandbox', url: target, status: 'enabled', livemode: false, enabled_events: [event] }], has_more: false });
  } }), /Expected a live/);
});
test('mail verification retries a transient read after the successful write', async () => {
  const key = 'fixture-mail-key';
  let reads = 0, writes = 0;
  const result = await syncFunctionMailKey({ token: 'fixture', key, fetcher: async (_, options) => {
    if (options.method === 'POST') { writes++; return Response.json({}); }
    reads++;
    if (reads === 1) return Response.json([]);
    if (reads === 2) return new Response('unavailable', { status: 503 });
    return Response.json([{ name: 'RESEND_API_KEY', value: createHash('sha256').update(key).digest('hex') }]);
  } });
  assert.equal(result.changed, true);
  assert.equal(writes, 1);
  assert.equal(reads, 3);
});
test('an already synchronized mail key does not require secret-write access', async () => {
  const key = 'fixture-mail-key';
  const result = await syncFunctionMailKey({ token: 'fixture', key, fetcher: async (_, options) => {
    assert.equal(options.method, undefined);
    return Response.json([{ name: 'RESEND_API_KEY', value: createHash('sha256').update(key).digest('hex') }]);
  } });
  assert.equal(result.changed, false);
});
test('mail synchronization does not hide a mismatched key when write permission is absent', async () => {
  await assert.rejects(syncFunctionMailKey({ token: 'fixture', key: 'fixture-mail-key', fetcher: async (_, options) => {
    return options.method === 'POST' ? new Response('sensitive provider details', { status: 403 }) : Response.json([]);
  } }), error => /edge_functions_secrets_write/.test(error.message) && !error.message.includes('sensitive provider details'));
});
