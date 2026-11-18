import { supabase } from './supabase';

// Client side of the billing-portal edge function. Every billing control on the site goes through
// here so a failure always surfaces as a message instead of a dead click.

export type PaidTier = 'pro' | 'enterprise';
export type PlanTier = 'free' | PaidTier;

export interface SubscriptionSummary {
  id: string;
  status: string;
  tier: PaidTier | null;
  interval: 'month' | 'year' | null;
  unit_amount: number | null;
  currency: string | null;
  quantity: number;
  current_period_end: number;
  cancel_at_period_end: boolean;
  cancel_at?: number | null;
  trial_end: number | null;
}

export interface BillingState {
  signedIn: boolean;
  tier: PlanTier;
  status: string;
  hasBillingAccount: boolean;
  subscription: SubscriptionSummary | null;
  billingAccounts?: Array<{ id: string; selected: boolean }>;
  /** Set when the profile or the subscription could not be read. The rest of the state is then a
   *  guess: callers must not treat it as "no subscription" (that would offer a second checkout). */
  error?: string;
}

export type PortalAction =
  | { action: 'manage'; customer_id?: string }
  | { action: 'cancel' }
  | { action: 'change_plan'; tier: PaidTier; interval?: 'month' | 'year' };

/** profiles.subscription_tier still holds the legacy 'professional' on some rows. */
export function normalizeTier(raw: string | null | undefined): PlanTier {
  if (raw === 'pro' || raw === 'professional') return 'pro';
  if (raw === 'enterprise') return 'enterprise';
  return 'free';
}

async function invoke<T>(body: Record<string, unknown>): Promise<{ data?: T; error?: string }> {
  const { data, error } = await supabase.functions.invoke(import.meta.env.PUBLIC_BILLING_FUNCTION || 'billing-portal', { method: 'POST', body });
  if (!error) return { data: data as T };
  // FunctionsHttpError carries the Response; our function always answers with { message }.
  let message = 'Billing is temporarily unavailable. Please try again or email support@aidotnet.dev.';
  try {
    const res = (error as { context?: Response }).context;
    const payload = res && typeof res.json === 'function' ? await res.json() : null;
    if (payload?.message) message = payload.message;
  } catch { /* keep the generic message */ }
  return { error: message };
}

/** Profile tier plus, for customers with a Stripe account, the live subscription. */
export async function loadBillingState(): Promise<BillingState> {
  const { data: { session } } = await supabase.auth.getSession();
  if (!session?.user) return { signedIn: false, tier: 'free', status: 'active', hasBillingAccount: false, subscription: null };
  const { data: profile, error: profileError } = await supabase
    .from('profiles')
    .select('subscription_tier, subscription_status, stripe_customer_id')
    .eq('id', session.user.id)
    .single();
  if (profileError || !profile) {
    return {
      signedIn: true, tier: 'free', status: 'unknown', hasBillingAccount: false, subscription: null,
      error: 'Could not load your plan. Please refresh the page or email support@aidotnet.dev.',
    };
  }
  const state: BillingState = {
    signedIn: true,
    tier: normalizeTier(profile?.subscription_tier),
    status: profile?.subscription_status ?? 'active',
    hasBillingAccount: !!profile?.stripe_customer_id,
    subscription: null,
  };
  {
    const { data, error } = await invoke<{ subscription: SubscriptionSummary | null; billing_account?: boolean; billing_accounts?: Array<{id:string;selected:boolean}> }>({ action: 'summary' });
    if (error) state.error = `Could not load your subscription from Stripe. ${error}`;
    else {
      state.subscription = data?.subscription ?? null;
      state.hasBillingAccount = data?.billing_account ?? state.hasBillingAccount;
      state.billingAccounts = data?.billing_accounts ?? [];
    }
  }
  return state;
}

/** Opens the Stripe portal for the action, or returns the reason it could not. */
export async function openPortal(req: PortalAction, returnPath = window.location.pathname): Promise<string | undefined> {
  const { data, error } = await invoke<{ url: string }>({ ...req, return_path: returnPath });
  if (error) return error;
  if (!data?.url) return 'Stripe did not return a portal link. Please try again.';
  window.location.assign(data.url);
  return undefined;
}

/** Called only after confirmation to shorten a later scheduled cancellation. */
export async function cancelBeforeRenewal(subscriptionId: string, renewal: number, scheduled: number): Promise<string | undefined> {
  const { data, error } = await invoke<{ url?: string }>({ action: 'cancel_before_renewal', expected_subscription_id: subscriptionId, expected_renewal: renewal, expected_cancel_at: scheduled });
  if (error) return error;
  if (!data?.url) return 'Stripe did not return a cancellation review link. Refresh your billing details.';
  window.location.assign(data.url);
  return undefined;
}

/** Reference is created by the authenticated server and contains no browser-supplied account ID. */
export async function checkoutReference(tier: PaidTier, interval: 'month' | 'year') {
  return invoke<{ reference: string }>({ action: 'checkout_reference', tier, interval });
}

export async function claimCheckout(sessionId: string) {
  return invoke<{ claimed: boolean }>({ action: 'claim_checkout', checkout_session_id: sessionId });
}

export function formatMoney(amount: number | null, currency: string | null): string {
  if (amount == null) return '';
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: (currency || 'usd').toUpperCase(), minimumFractionDigits: amount % 100 ? 2 : 0 }).format(amount / 100);
}

export function formatDate(unixSeconds: number): string {
  return new Date(unixSeconds * 1000).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
}

export const TIER_LABEL: Record<PlanTier, string> = { free: 'Free', pro: 'Pro', enterprise: 'Enterprise' };

/** Stripe's portal can schedule the end through cancel_at while leaving the legacy flag false. */
export function cancellationDate(subscription: SubscriptionSummary | null | undefined): number | null {
  return subscription?.cancel_at ?? (subscription?.cancel_at_period_end ? subscription.current_period_end : null);
}
