import { supabase } from './supabase';

// Client side of the billing-portal edge function. Every billing control on the site goes through
// here so a failure always surfaces as a message instead of a dead click.

export type PaidTier = 'pro' | 'enterprise';
export type PlanTier = 'free' | PaidTier;

export interface SubscriptionSummary {
  status: string;
  tier: PaidTier | null;
  interval: 'month' | 'year' | null;
  unit_amount: number | null;
  currency: string | null;
  quantity: number;
  current_period_end: number;
  cancel_at_period_end: boolean;
  trial_end: number | null;
}

export interface BillingState {
  signedIn: boolean;
  tier: PlanTier;
  status: string;
  hasBillingAccount: boolean;
  subscription: SubscriptionSummary | null;
}

export type PortalAction =
  | { action: 'manage' }
  | { action: 'cancel' }
  | { action: 'change_plan'; tier: PaidTier; interval?: 'month' | 'year' };

/** profiles.subscription_tier still holds the legacy 'professional' on some rows. */
export function normalizeTier(raw: string | null | undefined): PlanTier {
  if (raw === 'pro' || raw === 'professional') return 'pro';
  if (raw === 'enterprise') return 'enterprise';
  return 'free';
}

async function invoke<T>(body: Record<string, unknown>): Promise<{ data?: T; error?: string }> {
  const { data, error } = await supabase.functions.invoke('billing-portal', { method: 'POST', body });
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
  const { data: profile } = await supabase
    .from('profiles')
    .select('subscription_tier, subscription_status, stripe_customer_id')
    .eq('id', session.user.id)
    .single();
  const state: BillingState = {
    signedIn: true,
    tier: normalizeTier(profile?.subscription_tier),
    status: profile?.subscription_status ?? 'active',
    hasBillingAccount: !!profile?.stripe_customer_id,
    subscription: null,
  };
  if (state.hasBillingAccount) {
    const { data } = await invoke<{ subscription: SubscriptionSummary | null }>({ action: 'summary' });
    state.subscription = data?.subscription ?? null;
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

export function formatMoney(amount: number | null, currency: string | null): string {
  if (amount == null) return '';
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: (currency || 'usd').toUpperCase(), minimumFractionDigits: amount % 100 ? 2 : 0 }).format(amount / 100);
}

export function formatDate(unixSeconds: number): string {
  return new Date(unixSeconds * 1000).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
}

export const TIER_LABEL: Record<PlanTier, string> = { free: 'Free', pro: 'Pro', enterprise: 'Enterprise' };
