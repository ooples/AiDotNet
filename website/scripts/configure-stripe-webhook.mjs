import { pathToFileURL } from 'node:url';

const PROJECT = 'https://yfkqwpgjahoamlgckjib.supabase.co';
const REQUIRED_EVENT = 'checkout.session.async_payment_succeeded';

/** Preserve other events and the signing secret on the one existing website endpoint. */
export async function configureStripeWebhook({ key, projectUrl = PROJECT, mode = 'live', fetcher = fetch }) {
  if (!key || projectUrl !== PROJECT) throw new Error('Expected the existing website project and a Stripe server key');
  if (mode !== 'live' && mode !== 'sandbox') throw new Error('Explicit live or sandbox mode is required');
  const target = `${PROJECT}/functions/v1/stripe-webhook`;
  async function request(path, options = {}) {
    const response = await fetcher(`https://api.stripe.com/v1${path}`, {
      ...options, signal: AbortSignal.timeout(30_000),
      headers: { Authorization: `Bearer ${key}`, 'Stripe-Version': '2023-10-16', 'Content-Type': 'application/x-www-form-urlencoded' },
    });
    if (!response.ok) throw new Error(`Stripe endpoint configuration failed: HTTP ${response.status}`);
    return response.json();
  }
  const endpoints = [];
  let after = '';
  do {
    const page = await request(`/webhook_endpoints?limit=100${after ? `&starting_after=${encodeURIComponent(after)}` : ''}`);
    endpoints.push(...page.data);
    if (!page.has_more) break;
    const next = page.data.at(-1)?.id;
    if (!next || next === after) throw new Error('Invalid Stripe endpoint pagination');
    after = next;
  } while (true);
  const matches = endpoints.filter(e => e.status === 'enabled' && e.url.replace(/\/$/, '') === target);
  if (matches.length !== 1) throw new Error(`Expected one enabled website webhook endpoint; found ${matches.length}. No endpoint was created or removed.`);
  const endpoint = matches[0];
  if (endpoint.livemode !== (mode === 'live')) throw new Error(`Expected a ${mode} Stripe endpoint`);
  if (!Array.isArray(endpoint.enabled_events)) throw new Error('Stripe did not return the subscribed events');
  if (endpoint.enabled_events.includes('*') || endpoint.enabled_events.includes(REQUIRED_EVENT)) return { changed: false };
  // Stripe replaces the complete list and provides no conditional update.
  throw new Error(`Add ${REQUIRED_EVENT} to ${endpoint.id} at https://dashboard.stripe.com/webhooks, preserving existing events, then rerun verification. No configuration was changed.`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    const result = await configureStripeWebhook({ key: process.env.STRIPE_SECRET_KEY, projectUrl: process.env.SUPABASE_URL });
    console.log(result.changed ? 'Enabled delayed-payment success on the existing website webhook.' : 'The existing website webhook already receives delayed-payment success.');
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
