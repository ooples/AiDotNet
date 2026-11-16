import type { APIRequestContext } from '@playwright/test';

// Real Stripe objects for the signed-in personas. STRIPE_SECRET_KEY (a restricted key is enough:
// Customers write, Subscriptions write, Prices read) comes from the repo secret of that name.
//
// A persona's subscription is a 7-day trial with no payment method, set to cancel itself if the
// trial ever ends, so nothing is charged even in live mode. Each customer carries
// metadata.e2e=true and is deleted at the end of the test (which also cancels the subscription);
// the nightly job deletes any customer an aborted run left behind.

const API = 'https://api.stripe.com/v1';

function form(params: Record<string, string | number | boolean>): string {
  return Object.entries(params).map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`).join('&');
}

async function stripe<T>(request: APIRequestContext, method: 'GET' | 'POST' | 'DELETE', path: string, params: Record<string, string | number | boolean> = {}): Promise<T> {
  const key = process.env.STRIPE_SECRET_KEY;
  if (!key) throw new Error('STRIPE_SECRET_KEY is not set');
  const body = form(params);
  const url = method === 'GET' && body ? `${API}${path}?${body}` : `${API}${path}`;
  const res = await request.fetch(url, {
    method,
    headers: { authorization: `Bearer ${key}`, 'content-type': 'application/x-www-form-urlencoded' },
    data: method === 'GET' ? undefined : body,
  });
  const json = await res.json();
  if (!res.ok()) throw new Error(`Stripe ${method} ${path}: HTTP ${res.status()} ${json?.error?.message ?? ''}`);
  return json as T;
}

export interface StripePrice { id: string; unit_amount: number; recurring: { interval: 'month' | 'year' }; metadata: Record<string, string> }

/** The live plan price the webhook and billing-portal recognise (price.metadata.tier). */
export async function planPrice(request: APIRequestContext, tier: 'pro' | 'enterprise', interval: 'month' | 'year' = 'month'): Promise<StripePrice> {
  const list = await stripe<{ data: StripePrice[] }>(request, 'GET', '/prices', { active: true, type: 'recurring', limit: 100 });
  const names = tier === 'pro' ? ['pro', 'professional'] : ['enterprise'];
  const hit = list.data.find((p) => names.includes((p.metadata.tier ?? '').toLowerCase()) && p.recurring.interval === interval);
  if (!hit) throw new Error(`No active ${interval}ly Stripe price has metadata.tier=${tier}`);
  return hit;
}

export interface StripeFixture { customerId: string; subscriptionId: string; price: StripePrice }

export async function createTrialSubscription(request: APIRequestContext, email: string, userId: string, tier: 'pro' | 'enterprise'): Promise<StripeFixture> {
  const price = await planPrice(request, tier);
  const customer = await stripe<{ id: string }>(request, 'POST', '/customers', {
    email, name: 'E2E persona', 'metadata[e2e]': 'true', 'metadata[supabase_user_id]': userId,
  });
  const sub = await stripe<{ id: string; status: string }>(request, 'POST', '/subscriptions', {
    customer: customer.id,
    'items[0][price]': price.id,
    trial_period_days: 7,
    'trial_settings[end_behavior][missing_payment_method]': 'cancel',
    'metadata[e2e]': 'true',
  });
  if (sub.status !== 'trialing') throw new Error(`Expected a trialing subscription, got ${sub.status}`);
  return { customerId: customer.id, subscriptionId: sub.id, price };
}

export async function getSubscription(request: APIRequestContext, id: string) {
  return stripe<{ status: string; cancel_at_period_end: boolean; items: { data: Array<{ price: StripePrice }> } }>(request, 'GET', `/subscriptions/${id}`);
}

export async function deleteCustomer(request: APIRequestContext, id: string): Promise<void> {
  try { await stripe(request, 'DELETE', `/customers/${id}`); } catch (e) {
    if (!String(e).includes('No such customer')) throw e;
  }
}
