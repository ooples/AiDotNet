import type { APIRequestContext } from '@playwright/test';
import { serverRequest } from '../../../scripts/secure-backend-http.mjs';

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
  const res = await serverRequest(url, {
    method,
    // Match the version used by billing-portal: subscription period fields moved in Basil.
    headers: { authorization: `Bearer ${key}`, 'content-type': 'application/x-www-form-urlencoded', 'Stripe-Version': '2023-10-16' },
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

/** Legacy Checkout fixture: a real card-free trial, completed only through Stripe's UI. */
export async function createTrialCheckout(request: APIRequestContext, email: string, userId: string, origin: string) {
  const price=await planPrice(request,'pro');
  const customer=await stripe<{id:string}>(request,'POST','/customers',{
    email,name:'E2E legacy checkout claim','metadata[e2e]':'true','metadata[supabase_user_id]':userId,
  });
  try {
    const session=await stripe<{id:string;url:string}>(request,'POST','/checkout/sessions',{
      customer:customer.id,mode:'subscription','line_items[0][price]':price.id,'line_items[0][quantity]':1,
      payment_method_collection:'if_required','payment_method_types[0]':'card',
      'subscription_data[trial_period_days]':7,'subscription_data[trial_settings][end_behavior][missing_payment_method]':'cancel',
      'subscription_data[metadata][e2e]':'true','metadata[e2e]':'true','metadata[tier]':'pro',
      client_reference_id:userId,success_url:`${origin}/account/licenses/?new=true`,cancel_url:`${origin}/pricing/?cancelled=true`,
    });
    return {sessionId:session.id,url:session.url,customerId:customer.id};
  } catch (error) {
    try { await deleteCustomer(request,customer.id); }
    catch (cleanupError) { throw new AggregateError([error,cleanupError],'Checkout fixture creation and cleanup failed'); }
    throw error;
  }
}

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
  return stripe<{ status: string; cancel_at_period_end: boolean; cancel_at: number | null; current_period_end: number; items: { data: Array<{ price: StripePrice }> } }>(request, 'GET', `/subscriptions/${id}`);
}

/** Moves the subscription's only item to another price, as the portal does once it has a card. */
export async function changeSubscriptionPrice(request: APIRequestContext, id: string, priceId: string): Promise<void> {
  const sub = await stripe<{ items: { data: Array<{ id: string }> } }>(request, 'GET', `/subscriptions/${id}`);
  await stripe(request, 'POST', `/subscriptions/${id}`, {
    'items[0][id]': sub.items.data[0].id, 'items[0][price]': priceId, proration_behavior: 'none',
  });
}

/** Fixture setup only: exercise a scheduled cancellation beyond the next renewal. */
export async function scheduleLaterTrialCancellation(request: APIRequestContext, id: string): Promise<void> {
  const sub = await getSubscription(request, id);
  if (sub.status !== 'trialing') throw new Error('This fixture must remain a card-free trial');
  await stripe(request, 'POST', `/subscriptions/${id}`, { cancel_at: sub.current_period_end + 86400, proration_behavior: 'none' });
}

export async function deleteCustomer(request: APIRequestContext, id: string): Promise<void> {
  try { await stripe(request, 'DELETE', `/customers/${id}`); } catch (e) {
    if (!String(e).includes('No such customer')) throw e;
  }
}
