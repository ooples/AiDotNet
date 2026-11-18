import { serve } from "https://deno.land/std@0.177.0/http/server.ts";
import { createClient, SupabaseClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";
import Stripe from "https://esm.sh/stripe@14.14.0?target=deno";
import { sendLicenseKeyEmail } from "../_shared/email.ts";
import { type ProductSlug } from "../_shared/products.ts";
import { checkoutOwner, rememberCheckoutOwner } from "../_shared/checkout-ownership.ts";

// Product this webhook issues licenses for. All Stripe products created
// for aidotnet.dev route through here; Harmonic Engine (and any future
// products) will get their own webhook + edge function path.
const WEBHOOK_PRODUCT: ProductSlug = "aidotnet";
// Key prefix that the AiDotNet client library parses for online-validated
// keys. See LicenseValidator.IsServerValidatedKeyFormat in src/Helpers/
// LicenseValidator.cs of this repo: keys must split into ≥4 dash-delimited
// segments starting with "AIDN" and end with ≥8 hex chars. The dotted
// `aidn.{id}.{sig}` form is also accepted by the library, but only with a
// real HMAC signature against a build key the webhook doesn't have access
// to — so the dash form is the only viable issuance path here.
const WEBHOOK_PREFIX = "AIDN-PROD" as const;

// Two separate tier vocabularies exist:
//   * `profiles.subscription_tier` (free text, UI checks `=== 'pro'`)
//   * `license_keys.tier` (enum `license_tier`: 'community' | 'professional' | 'enterprise')
// Stripe Payment Link metadata.tier can be configured to either word
// depending on dashboard history, so the webhook accepts both `pro` and
// `professional` as synonyms and writes the CANONICAL name to each
// column. Previous code wrote the same string to both columns, so one
// side was always wrong (UI saw 'professional' and rendered Free, or the
// license_keys insert failed the enum check).
type TierAlias = "pro" | "professional" | "enterprise";
interface CanonicalTier {
  // Goes into profiles.subscription_tier — matches the value the
  // billing/account UI checks for (`=== 'pro'`).
  profileTier: string;
  // Goes into license_keys.tier — must be a value of the
  // public.license_tier enum.
  licenseTier: "professional" | "enterprise";
  // Used by AIDN-PROD-{TIER}-{32hex} key generation.
  keySegment: string;
  maxActivations: number;
}
const TIER_CANONICAL: Record<TierAlias, CanonicalTier> = {
  pro:          { profileTier: "pro", licenseTier: "professional", keySegment: "PRO", maxActivations: 10 },
  professional: { profileTier: "pro", licenseTier: "professional", keySegment: "PRO", maxActivations: 10 },
  enterprise:   { profileTier: "enterprise", licenseTier: "enterprise", keySegment: "ENTERPRISE", maxActivations: 10 },
};
function resolveTier(raw: unknown): CanonicalTier | null {
  if (typeof raw !== "string") return null;
  const lower = raw.toLowerCase() as TierAlias;
  return TIER_CANONICAL[lower] ?? null;
}

// Secrets are read lazily inside the request handler so the function can
// be deployed BEFORE the project-level secrets are configured in the
// Supabase dashboard. A deploy-time throw would block the deploy workflow
// on a cold install.
let stripe: Stripe | null = null;
function getStripe(): Stripe {
  if (stripe) return stripe;
  const key = Deno.env.get("STRIPE_SECRET_KEY");
  if (!key) throw new Error("missing_stripe_secret_key");
  stripe = new Stripe(key, {
    apiVersion: "2023-10-16",
    httpClient: Stripe.createFetchHttpClient(),
  });
  return stripe;
}

serve(async (req: Request) => {
  if (req.method !== "POST") {
    return new Response(JSON.stringify({ error: "method_not_allowed" }), {
      status: 405, headers: { "Content-Type": "application/json" },
    });
  }

  const signature = req.headers.get("stripe-signature");
  if (!signature) {
    return new Response(JSON.stringify({ error: "missing_signature" }), {
      status: 400, headers: { "Content-Type": "application/json" },
    });
  }

  const endpointSecret = Deno.env.get("STRIPE_WEBHOOK_SECRET");
  if (!endpointSecret) {
    console.error("STRIPE_WEBHOOK_SECRET not configured on this Supabase project");
    return new Response(JSON.stringify({ error: "server_not_configured" }), {
      status: 500, headers: { "Content-Type": "application/json" },
    });
  }

  let event: Stripe.Event;
  try {
    const body = await req.text();
    event = await getStripe().webhooks.constructEventAsync(body, signature, endpointSecret);
  } catch (err) {
    console.error("Webhook signature verification failed:", err);
    return new Response(JSON.stringify({ error: "invalid_signature" }), {
      status: 400, headers: { "Content-Type": "application/json" },
    });
  }

  const supabaseUrl = Deno.env.get("SUPABASE_URL") ?? "";
  const supabaseServiceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
  const serviceClient = createClient(supabaseUrl, supabaseServiceKey);

  try {
    switch (event.type) {
      case "checkout.session.completed":
      case "checkout.session.async_payment_succeeded":
        await handleCheckoutCompleted(serviceClient, event.data.object as Stripe.Checkout.Session);
        break;

      case "invoice.paid":
        await handleInvoicePaid(serviceClient, event.data.object as Stripe.Invoice);
        break;

      case "customer.subscription.updated":
        await reconcileSubscription(serviceClient, (event.data.object as Stripe.Subscription).id);
        break;

      case "customer.subscription.deleted":
        await handleSubscriptionDeleted(serviceClient, event.data.object as Stripe.Subscription);
        break;

      default:
        console.log(`Unhandled event type: ${event.type}`);
    }
  } catch (err) {
    console.error(`Error handling ${event.type}:`, err);
    // Return 500 so Stripe retries. Do NOT return 200 on handler failure —
    // the subscriber already paid; a silently-swallowed error means they
    // never get their license and we never find out.
    return new Response(JSON.stringify({ error: "processing_error" }), {
      status: 500, headers: { "Content-Type": "application/json" },
    });
  }

  return new Response(JSON.stringify({ received: true }), {
    status: 200, headers: { "Content-Type": "application/json" },
  });
});

/**
 * Derives the profile mirror without discarding other active grants.
 */
async function syncBillingProfile(client: SupabaseClient, customer: string | null, tier: string, status: string, userId?: string) {
  const { error } = await client.rpc("sync_aidotnet_billing_profile", {
    p_customer_id: customer, p_fallback_tier: tier, p_fallback_status: status,
    ...(userId ? { p_user_id: userId } : {}),
  });
  if (error) throw error;
}

function subscriptionTier(subscription: Stripe.Subscription): CanonicalTier {
  const tiers = subscription.items.data.map(item => resolveTier(item.price.metadata?.tier));
  const tier = tiers.find(t => t?.licenseTier === "enterprise") ?? tiers.find(t => t !== null);
  if (!tier) throw new Error("Subscription has no recognized plan tier");
  return tier;
}

async function handleCheckoutCompleted(client: SupabaseClient, session: Stripe.Checkout.Session) {
  if (session.payment_status === "unpaid") return;
  if (session.payment_status !== "paid" && session.payment_status !== "no_payment_required") {
    throw new Error("Unknown Checkout payment status");
  }
  const userId = await checkoutOwner(client, session);
  if (!userId) throw new Error("cannot_resolve_user");
  let canonical = resolveTier(session.metadata?.tier);
  if (!canonical) throw new Error("Checkout has no recognized plan tier");
  const customer = typeof session.customer === "string" ? session.customer : session.customer?.id ?? null;
  const subscriptionId = typeof session.subscription === "string" ? session.subscription : session.subscription?.id ?? null;
  const identity = subscriptionId ? "stripe_subscription_id" : "stripe_checkout_session_id";
  const { data: existing, error: lookupError } = await client.from("license_keys")
    .select("id,status,user_id").eq(identity, subscriptionId ?? session.id).eq("product", WEBHOOK_PRODUCT).limit(1).maybeSingle();
  if (lookupError) throw lookupError;
  if (existing?.status === "revoked") return;

  if (existing && existing.user_id !== userId) throw new Error("Entitlement belongs to a different verified account");
  await rememberCheckoutOwner(client, userId, session);
  const repairProfileLink = async () => {
    if (!customer) return;
    // Replays may repair a missing link, never replace another billing account.
    const { error } = await client.from("profiles").update({ stripe_customer_id: customer })
      .eq("id", userId).is("stripe_customer_id", null);
    if (error) throw error;
  };
  if (existing) {
    await repairProfileLink();
    if (subscriptionId) await reconcileSubscription(client, subscriptionId);
    else await syncBillingProfile(client, customer, "free", "expired", userId);
    return;
  }

  // Persist before asking Stripe about mutable state. An invoice/deletion that
  // ran before this insert cannot be the last chance to reconcile the key.
  // Every subscription starts disabled, including a currently active one.
  const licenseKey = `${WEBHOOK_PREFIX}-${canonical.keySegment}-${crypto.randomUUID().replace(/-/g, "")}`;
  const { error: insertError } = await client.from("license_keys").insert({
    user_id: userId, license_key: licenseKey, product: WEBHOOK_PRODUCT,
    tier: canonical.licenseTier, status: subscriptionId ? "suspended" : "active",
    max_activations: canonical.maxActivations, stripe_subscription_id: subscriptionId,
    stripe_checkout_session_id: session.id, stripe_customer_id: customer,
    notes: `Issued by Stripe Checkout (session ${session.id})`,
  });
  if (insertError) throw insertError; // Unique conflicts retry against the existing durable row.
  // A failed profile repair must leave the verified paid entitlement recoverable.
  await repairProfileLink();
  if (subscriptionId) {
    const current = await reconcileSubscription(client, subscriptionId);
    if (current.status === "canceled" || current.status === "incomplete_expired") return;
    canonical = subscriptionTier(current);
  } else {
    await syncBillingProfile(client, customer, "free", "expired", userId);
  }
  if (session.customer_details?.email) {
    await sendLicenseKeyEmail({
      to: session.customer_details.email, licenseKey, tier: canonical.licenseTier,
      product: WEBHOOK_PRODUCT, isExisting: false,
    });
  }
}

/**
 * Handles invoice.paid — keeps the license active on renewal.
 *
 * A manually-revoked license should stay revoked even if Stripe keeps
 * renewing, so we explicitly exclude `revoked` from the update target.
 */
async function handleInvoicePaid(
  client: SupabaseClient,
  invoice: Stripe.Invoice,
) {
  // Basil moved the invoice's subscription reference under parent. Accept
  // either event payload shape; subsequent retrieval uses our pinned API.
  const related = invoice.subscription ?? (invoice as Stripe.Invoice & {
    parent?: { subscription_details?: { subscription?: string | { id: string } } };
  }).parent?.subscription_details?.subscription;
  const subscriptionId = typeof related === "string" ? related : related?.id;

  if (!subscriptionId) return;

  await reconcileSubscription(client, subscriptionId);
}

/** Reconcile a fresh Stripe snapshot rather than replaying an older event's state. */
async function reconcileSubscription(client: SupabaseClient, subscriptionId: string) {
  const fingerprint = (sub: Stripe.Subscription) => JSON.stringify({
    status: sub.status, customer: typeof sub.customer === "string" ? sub.customer : sub.customer?.id,
    tiers: sub.items?.data.map(item => item.price.metadata?.tier).sort(),
  });
  for (let attempt = 0; attempt < 3; attempt++) {
    const current = await getStripe().subscriptions.retrieve(subscriptionId);
    if (current.status === "canceled" || current.status === "incomplete_expired") await handleSubscriptionDeleted(client, current);
    else await handleSubscriptionUpdated(client, current);
    // Verify after the writes, not just before them. A concurrent activation or
    // deletion must not leave the last completed delivery applying an old state.
    const verified = await getStripe().subscriptions.retrieve(subscriptionId);
    if (fingerprint(current) === fingerprint(verified)) return verified;
  }
  throw new Error("Subscription changed repeatedly during reconciliation; retry required");
}

/**
 * Handles customer.subscription.updated — plan changes, cancellation
 * scheduling, past_due transitions, etc.
 */
async function handleSubscriptionUpdated(
  client: SupabaseClient,
  subscription: Stripe.Subscription,
) {
  const subscriptionId = subscription.id;

  // Map Stripe status → our license status.
  let licenseStatus: "active" | "suspended" | "expired";
  switch (subscription.status) {
    case "active":
    case "trialing":
      licenseStatus = "active";
      break;
    case "past_due":
    case "unpaid":
      licenseStatus = "suspended";
      break;
    case "canceled":
    case "incomplete_expired":
      licenseStatus = "expired";
      break;
    default:
      licenseStatus = "suspended";
  }

  // Determine tier from the subscription's price metadata. Today each
  // Stripe subscription has exactly one line item (one plan per sub),
  // but multi-item subscriptions — say, a main plan bundled with an
  // add-on — are allowed by Stripe and the webhook should degrade
  // gracefully if one lands.
  //
  // Strategy: walk every item, normalise each tier through resolveTier
  // (which accepts both 'pro' and 'professional' aliases — see
  // TIER_CANONICAL above), and pick the highest-ranked tier we know
  // about. Enterprise wins over professional; anything resolveTier
  // returns null for is ignored rather than trusted.
  const TIER_LICENSE_RANK: Record<CanonicalTier["licenseTier"], number> = {
    professional: 1,
    enterprise: 2,
  };
  const items = subscription.items?.data ?? [];
  const seenCanonical = items
    .map((item) => resolveTier(item.price?.metadata?.tier))
    .filter((c): c is CanonicalTier => c !== null);

  // If the subscription has items but NONE resolved to a known tier,
  // fail loud. Silently updating `status` without `tier`/`max_activations`
  // would leave the old entitlement in place after a plan change —
  // exactly the "license still active for the wrong tier" footgun
  // this handler is supposed to prevent. Stripe will retry the event
  // and the admin can investigate the metadata drift in the meantime.
  if (items.length > 0 && seenCanonical.length === 0) {
    const rawTiers = items.map((it) => it.price?.metadata?.tier ?? '(missing)');
    throw new Error(
      `customer.subscription.updated: subscription ${subscription.id} has ${items.length} ` +
      `item(s) but none of their price.metadata.tier values resolve to a canonical tier. ` +
      `Saw: [${rawTiers.join(', ')}]. ` +
      `Update Stripe Price metadata to one of: 'pro', 'professional', 'enterprise'.`
    );
  }

  const winner = seenCanonical.length > 0
    ? seenCanonical.reduce((a, b) =>
        TIER_LICENSE_RANK[a.licenseTier] >= TIER_LICENSE_RANK[b.licenseTier] ? a : b
      )
    : null;

  const updateData: Record<string, string | number> = { status: licenseStatus };
  if (winner) {
    updateData.tier = winner.licenseTier;
    updateData.max_activations = winner.maxActivations;
  }

  const { error } = await client
    .from("license_keys")
    .update(updateData)
    .eq("stripe_subscription_id", subscriptionId)
    .eq("product", WEBHOOK_PRODUCT)
    .neq("status", "revoked")
    .neq("status", "expired");

  if (error) {
    console.error("subscription.updated license update failed:", error);
    throw error;
  }

  // Mirror subscription state on the user profile so /account/billing
  // shows the right tier/status without doing its own Stripe call.
  const stripeCustomerId = typeof subscription.customer === "string"
    ? subscription.customer
    : subscription.customer?.id;

  if (stripeCustomerId) {
    await syncBillingProfile(client, stripeCustomerId, winner?.profileTier ?? "free", licenseStatus);
  }
}

/**
 * Handles customer.subscription.deleted — expires the license + downgrades
 * the profile back to free.
 */
async function handleSubscriptionDeleted(
  client: SupabaseClient,
  subscription: Stripe.Subscription,
) {
  const subscriptionId = subscription.id;

  const { error } = await client
    .from("license_keys")
    .update({ status: "expired" })
    .eq("stripe_subscription_id", subscriptionId)
    .eq("product", WEBHOOK_PRODUCT)
    .neq("status", "revoked");

  if (error) {
    console.error("subscription.deleted update failed:", error);
    throw error;
  }

  const stripeCustomerId = typeof subscription.customer === "string"
    ? subscription.customer
    : subscription.customer?.id;

  if (stripeCustomerId) {
    await syncBillingProfile(client, stripeCustomerId, "free", "expired");
  }

  console.log(`Expired licenses for subscription ${subscriptionId}`);
}
