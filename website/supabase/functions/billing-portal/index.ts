import { serve as serveHttp } from "https://deno.land/std@0.177.0/http/server.ts";
import { withCors, websiteOrigin } from "../_shared/cors.ts";
const serve = (handler: (req: Request) => Response | Promise<Response>) => serveHttp(withCors(handler));
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";
import Stripe from "https://esm.sh/stripe@14.14.0?target=deno";
import { checkoutOwner, rememberCheckoutOwner, ownedBillingCustomers } from "../_shared/checkout-ownership.ts";

// Signed-in customers' window into their Stripe subscription.
//
//   POST { action: "summary" }      -> the live subscription (plan, price, renewal, cancel state)
//   POST { action: "manage" }       -> Stripe Customer Portal session URL (payment method, invoices)
//   POST { action: "change_plan", tier: "pro" | "enterprise", interval?: "month" | "year" }
//                                   -> portal URL that opens straight on the plan-change confirm step
//   POST { action: "cancel" }       -> portal URL that opens straight on the cancel step
//
// The selected customer must be in the service-owned billing_customers ledger.
// A legacy profile preference is never ownership evidence.
// Portal sessions use a configuration this function owns (found by metadata, created on first use)
// so plan changes work without anyone hand-configuring the dashboard's default portal.

const ALLOWED_ORIGIN = Deno.env.get("ALLOWED_ORIGIN") ?? "https://www.aidotnet.dev";
const corsHeaders = {
  "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};
const PORTAL_CONFIG_TAG = "aidotnet-plans-v1";
const PAID_TIERS = ["pro", "enterprise"] as const;
type PaidTier = typeof PAID_TIERS[number];
const LIVE_STATUSES = new Set(["active", "trialing", "past_due", "unpaid", "incomplete"]);

let stripe: Stripe | null = null;
function getStripe(): Stripe {
  if (stripe) return stripe;
  const key = Deno.env.get("STRIPE_SECRET_KEY");
  if (!key) throw new Error("missing_stripe_secret_key");
  stripe = new Stripe(key, { apiVersion: "2023-10-16", httpClient: Stripe.createFetchHttpClient() });
  return stripe;
}

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { ...corsHeaders, "Content-Type": "application/json" } });
}

// Price metadata.tier is the same vocabulary the webhook reads ('pro' | 'professional' | 'enterprise').
function tierOfPrice(price: Stripe.Price | null | undefined): PaidTier | null {
  const raw = (price?.metadata?.tier ?? "").toLowerCase();
  if (raw === "pro" || raw === "professional") return "pro";
  if (raw === "enterprise") return "enterprise";
  return null;
}

async function planPrices(s: Stripe): Promise<Stripe.Price[]> {
  const out: Stripe.Price[] = [];
  for await (const p of s.prices.list({ active: true, type: "recurring", limit: 100 })) {
    if (tierOfPrice(p)) out.push(p);
  }
  return out;
}

async function portalConfiguration(s: Stripe, prices: Stripe.Price[]): Promise<string> {
  const products = new Map<string, string[]>();
  for (const p of prices) {
    const prod = typeof p.product === "string" ? p.product : p.product.id;
    products.set(prod, [...(products.get(prod) ?? []), p.id]);
  }
  const features: Stripe.BillingPortal.ConfigurationCreateParams.Features = {
    customer_update: { enabled: true, allowed_updates: ["email", "address", "tax_id"] },
    invoice_history: { enabled: true },
    payment_method_update: { enabled: true },
    subscription_cancel: { enabled: true, mode: "at_period_end" },
    subscription_update: {
      enabled: products.size > 0,
      default_allowed_updates: ["price", "quantity"],
      proration_behavior: "create_prorations",
      products: [...products].map(([product, ids]) => ({ product, prices: ids })),
    },
  };
  const wanted = JSON.stringify(features.subscription_update.products);
  for await (const c of s.billingPortal.configurations.list({ active: true, limit: 100 })) {
    if (c.metadata?.tag !== PORTAL_CONFIG_TAG) continue;
    if (c.metadata?.products === wanted.slice(0, 500)) return c.id;
    await s.billingPortal.configurations.update(c.id, { features, metadata: { tag: PORTAL_CONFIG_TAG, products: wanted.slice(0, 500) } });
    return c.id;
  }
  const created = await s.billingPortal.configurations.create({
    features,
    business_profile: { headline: "AiDotNet subscription" },
    metadata: { tag: PORTAL_CONFIG_TAG, products: wanted.slice(0, 500) },
  });
  return created.id;
}

async function liveSubscription(s: Stripe, customer: string): Promise<Stripe.Subscription | null> {
  // Every page: the live subscription can sit behind any number of ended ones.
  let newest: Stripe.Subscription | null = null;
  for await (const x of s.subscriptions.list({ customer, status: "all", limit: 100, expand: ["data.items.data.price"] })) {
    if (LIVE_STATUSES.has(x.status) && (!newest || x.created > newest.created)) newest = x;
  }
  return newest;
}

function summarize(sub: Stripe.Subscription | null) {
  if (!sub) return { subscription: null };
  const item = sub.items.data[0];
  const price = item?.price;
  return {
    subscription: {
      id: sub.id,
      status: sub.status,
      tier: tierOfPrice(price),
      interval: price?.recurring?.interval ?? null,
      unit_amount: price?.unit_amount ?? null,
      currency: price?.currency ?? null,
      quantity: item?.quantity ?? 1,
      current_period_end: sub.current_period_end,
      cancel_at_period_end: sub.cancel_at_period_end,
      cancel_at: sub.cancel_at ?? null,
      trial_end: sub.trial_end,
    },
  };
}

serve(async (req: Request) => {
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: corsHeaders });
  if (req.method !== "POST") return json({ error: "method_not_allowed" }, 405);

  const authHeader = req.headers.get("Authorization");
  if (!authHeader) return json({ error: "unauthorized", message: "Sign in to manage billing." }, 401);
  const url = Deno.env.get("SUPABASE_URL"), anon = Deno.env.get("SUPABASE_ANON_KEY"), service = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  if (!url || !anon || !service) return json({ error: "server_configuration_error" }, 500);

  // Pass the token explicitly: see admin-resend-license-email for why argument-less getUser() fails.
  const jwt = authHeader.replace(/^Bearer\s+/i, "");
  const { data: { user }, error: authError } = await createClient(url, anon).auth.getUser(jwt);
  if (authError || !user) return json({ error: "unauthorized", message: "Invalid or expired session." }, 401);

  let body: Record<string, unknown>;
  try { body = await req.json(); } catch { return json({ error: "invalid_json" }, 400); }
  const action = body.action;
  if (action !== "summary" && action !== "manage" && action !== "change_plan" && action !== "cancel" && action !== "cancel_before_renewal" && action !== "checkout_reference" && action !== "claim_checkout") {
    return json({ error: "invalid_action" }, 400);
  }
  const returnPath = typeof body.return_path === "string" && /^\/(account|pricing)\/[\w\-/]*$/.test(body.return_path)
    ? body.return_path : "/account/billing/";
  const returnUrl = `${websiteOrigin(req) ?? ALLOWED_ORIGIN}${returnPath}`;

  const admin = createClient(url, service);
  const { data: profile, error: profileError } = await admin
    .from("profiles").select("stripe_customer_id, subscription_tier").eq("id", user.id).single();
  if (profileError || !profile) return json({ error: "profile_not_found" }, 404);
  try {
    const customers = await ownedBillingCustomers(admin, user.id);
    const unresolved = !!profile.stripe_customer_id && !customers.includes(profile.stripe_customer_id);
    const ownershipMessage = "A legacy billing account needs ownership verification. Claim your completed checkout from Account → Licenses, or email support@aidotnet.dev. Your verified billing accounts remain available.";
    if (typeof body.customer_id === "string" && !customers.includes(body.customer_id)) {
      return json({ error: "not_your_billing_account" }, 403);
    }
    let selected: Stripe.Subscription | null = null;
    let ongoing = false;
    let customer = typeof body.customer_id === "string" ? body.customer_id
      : !unresolved && profile.stripe_customer_id ? profile.stripe_customer_id : customers[0] ?? null;
    const s = customers.length || action === "claim_checkout" ? getStripe() : null;
    for (const id of customers) {
      const candidate = await liveSubscription(s!, id);
      if (candidate) ongoing = true;
      if (body.customer_id && id !== body.customer_id) continue;
      if (candidate && (!selected || candidate.created > selected.created)) { selected = candidate; customer = id; }
    }
    if (action === "checkout_reference") {
      if (unresolved) return json({ error: "billing_ownership_unverified", message: ownershipMessage }, 409);
      if (body.tier !== "pro" && body.tier !== "enterprise") return json({ error: "invalid_tier" }, 400);
      if (body.interval !== "month" && body.interval !== "year") return json({ error: "invalid_interval" }, 400);
      if (ongoing) return json({ error: "existing_subscription", message: "Manage your existing subscription instead of starting another checkout." }, 409);
      const { data, error } = await admin.from("billing_checkout_intents")
        .insert({ user_id: user.id, tier: body.tier, interval: body.interval }).select("id").single();
      if (error || !data) throw error ?? new Error("Checkout reference was not saved");
      return json({ reference: data.id });
    }
    if (action === "claim_checkout") {
      if (typeof body.checkout_session_id !== "string" || !/^cs_[A-Za-z0-9_]+$/.test(body.checkout_session_id)) return json({ error: "invalid_checkout" }, 400);
      if (!user.email || !user.email_confirmed_at) return json({ error: "verified_email_required", message: "Verify your account email before claiming a checkout." }, 403);
      const session = await getStripe().checkout.sessions.retrieve(body.checkout_session_id);
      if (session.status !== "complete" || !["paid","no_payment_required"].includes(session.payment_status)) {
        return json({ error: "checkout_not_paid", message: "This checkout has not completed payment." }, 409);
      }
      if (session.customer_details?.email?.toLowerCase() !== user.email.toLowerCase()) {
        return json({ error: "checkout_email_mismatch", message: "Sign in with the verified email used for this checkout." }, 403);
      }
      const owner = await checkoutOwner(admin, session);
      if (owner && owner !== user.id) return json({ error: "checkout_already_claimed" }, 403);
      await rememberCheckoutOwner(admin, user.id, session);
      const secret = Deno.env.get("STRIPE_WEBHOOK_SECRET");
      if (!secret) throw new Error("Webhook fulfillment is not configured");
      const payload = JSON.stringify({ id: `evt_claim_${crypto.randomUUID()}`, object: "event",
        type: "checkout.session.completed", data: { object: session } });
      const timestamp = Math.floor(Date.now()/1000);
      const signingKey = await crypto.subtle.importKey("raw", new TextEncoder().encode(secret), { name:"HMAC",hash:"SHA-256" }, false, ["sign"]);
      const signature = Array.from(new Uint8Array(await crypto.subtle.sign("HMAC",signingKey,new TextEncoder().encode(`${timestamp}.${payload}`)))).map(byte=>byte.toString(16).padStart(2,"0")).join("");
      const webhook = new URL(req.url).pathname.split('/').at(-1) === 'billing-portal-preview' ? 'stripe-webhook-preview' : 'stripe-webhook';
      const fulfilled = await fetch(`${url}/functions/v1/${webhook}`, { method:"POST",
        headers: { "Content-Type":"application/json","stripe-signature":`t=${timestamp},v1=${signature}` }, body:payload });
      if (!fulfilled.ok) throw new Error("Verified checkout fulfillment will need a retry");
      await fulfilled.body?.cancel();
      return json({ claimed:true });
    }
    if (action === "summary") {
      return json({ billing_account: customers.length > 0, billing_accounts: customers.map(id=>({id,selected:id===customer})),
        ownership_resolution_required: unresolved, ownership_message: unresolved ? ownershipMessage : null, ...summarize(selected) });
    }
    if (!customer && unresolved) return json({ error: "billing_ownership_unverified", message: ownershipMessage }, 409);
    if (!customer || !s) return json({ error: "no_billing_account", message: "There is no Stripe billing account to manage. Contact support@aidotnet.dev for team-granted plans." }, 409);

    if (action === "cancel_before_renewal") {
      const sub = selected;
      if (!sub || body.expected_subscription_id !== sub.id || body.expected_renewal !== sub.current_period_end || body.expected_cancel_at !== sub.cancel_at) {
        return json({ error: "billing_state_changed", message: "Your subscription, renewal, or scheduled cancellation changed. Review the refreshed billing details and confirm again." }, 409);
      }
      if (!sub || sub.cancel_at == null || sub.cancel_at <= sub.current_period_end) {
        return json({ error: "no_later_cancellation", message: "There is no later cancellation to move. Refresh your billing details." }, 409);
      }
      // Stripe owns the final confirmation and write. Our earlier read must not
      // overwrite a cancellation that another session changed in the meantime.
      const configuration = await portalConfiguration(s, await planPrices(s));
      const session = await s.billingPortal.sessions.create({ customer, return_url:returnUrl, configuration });
      return json({ url: session.url });
    }

    const prices = await planPrices(s);
    const configuration = await portalConfiguration(s, prices);
    const params: Stripe.BillingPortal.SessionCreateParams = { customer, return_url: returnUrl, configuration };

    if (action === "change_plan" || action === "cancel") {
      const sub = await liveSubscription(s, customer);
      if (!sub) {
        return json({ error: "no_active_subscription", message: "No active subscription was found on your billing account." }, 409);
      }
      if (action === "cancel") {
        if (sub.cancel_at_period_end || sub.cancel_at != null) {
          const endsAt = sub.cancel_at ?? sub.current_period_end;
          return json({ error: "already_cancelling", cancel_at: endsAt, message: `Your subscription is already set to end on ${new Date(endsAt * 1000).toISOString().slice(0, 10)}. Refresh your billing details to manage this cancellation.` }, 409);
        }
        params.flow_data = {
          type: "subscription_cancel",
          subscription_cancel: { subscription: sub.id },
          after_completion: { type: "redirect", redirect: { return_url: returnUrl } },
        };
      } else {
        const tier = body.tier;
        if (tier !== "pro" && tier !== "enterprise") return json({ error: "invalid_tier" }, 400);
        const item = sub.items.data[0];
        const interval = body.interval === "year" || body.interval === "month" ? body.interval : item?.price?.recurring?.interval ?? "month";
        const target = prices.find((p) => tierOfPrice(p) === tier && p.recurring?.interval === interval)
          ?? prices.find((p) => tierOfPrice(p) === tier);
        if (!target) return json({ error: "plan_unavailable", message: `The ${tier} plan is not available for online purchase.` }, 409);
        if (item?.price?.id === target.id) {
          return json({ error: "already_on_plan", message: "You are already on this plan." }, 409);
        }
        params.flow_data = {
          type: "subscription_update_confirm",
          subscription_update_confirm: {
            subscription: sub.id,
            items: [{ id: item.id, price: target.id, quantity: item.quantity ?? 1 }],
          },
          after_completion: { type: "redirect", redirect: { return_url: returnUrl } },
        };
      }
    }

    const session = await s.billingPortal.sessions.create(params);
    return json({ url: session.url });
  } catch (err) {
    console.error(`billing-portal ${action} failed for user ${user.id}:`, err);
    return json({ error: "stripe_error", message: "Stripe could not open your billing portal. Please try again or email support@aidotnet.dev." }, 502);
  }
});
