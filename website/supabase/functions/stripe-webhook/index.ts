import { serve } from "https://deno.land/std@0.177.0/http/server.ts";
import { createClient, SupabaseClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";
import Stripe from "https://esm.sh/stripe@14.14.0?target=deno";

// Product this webhook issues licenses for. All Stripe products created
// for aidotnet.dev route through here; Harmonic Engine (and any future
// products) will get their own webhook + edge function path.
const WEBHOOK_PRODUCT = "aidotnet" as const;
// Key prefix must match what the client library parses; see
// website/src/pages/admin/licenses/index.astro's PRODUCTS list.
const WEBHOOK_PREFIX = "aidn" as const;

// Per-tier activation caps. Keeping them in one map so bumping a tier's
// cap doesn't require hunting through the handlers.
const TIER_MAX_ACTIVATIONS: Record<string, number> = {
  professional: 10,
  enterprise: 10,
};

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
        await handleCheckoutCompleted(serviceClient, event.data.object as Stripe.Checkout.Session);
        break;

      case "invoice.paid":
        await handleInvoicePaid(serviceClient, event.data.object as Stripe.Invoice);
        break;

      case "customer.subscription.updated":
        await handleSubscriptionUpdated(serviceClient, event.data.object as Stripe.Subscription);
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
 * Resolves a Supabase user ID from a Stripe Checkout Session.
 *
 * Preferred path is `client_reference_id`, which the pricing page sets to
 * session.user.id when the customer is signed in at checkout. When that's
 * missing (customer paid without signing in), we fall back to looking up
 * the user by the email address Stripe collected at checkout.
 *
 * Returns null when no matching user is found. Callers log + surface a
 * provisional license row so the admin can reconcile manually.
 */
async function resolveUserIdFromSession(
  client: SupabaseClient,
  session: Stripe.Checkout.Session,
): Promise<string | null> {
  if (session.client_reference_id) {
    return session.client_reference_id;
  }

  const email = session.customer_details?.email?.toLowerCase();
  if (!email) return null;

  const { data, error } = await client
    .from("profiles")
    .select("id")
    .eq("email", email)
    .maybeSingle();

  if (error) {
    console.error("Profile lookup by email failed:", error);
    return null;
  }

  return data?.id ?? null;
}

/**
 * Handles checkout.session.completed — provisions the customer's license.
 */
async function handleCheckoutCompleted(
  client: SupabaseClient,
  session: Stripe.Checkout.Session,
) {
  const userId = await resolveUserIdFromSession(client, session);
  if (!userId) {
    // The payment succeeded but we can't find a user. Log loudly — the
    // admin can reconcile manually via the admin licenses page using the
    // stripe_customer_id that's attached to every future event from this
    // customer.
    console.error(
      `checkout.session.completed: unable to resolve user for session ${session.id}. `
      + `client_reference_id=${session.client_reference_id} `
      + `customer_email=${session.customer_details?.email}`,
    );
    throw new Error("cannot_resolve_user");
  }

  // Derive the set of allowed tiers from TIER_MAX_ACTIVATIONS rather
  // than hardcoding a parallel list. Adding a new paid tier becomes a
  // one-line change to the map instead of needing to also update a
  // separate validator list that can drift out of sync.
  const tier = session.metadata?.tier;
  if (!tier || !(tier in TIER_MAX_ACTIVATIONS)) {
    throw new Error(`checkout.session.completed: invalid or missing tier '${tier}' in session metadata`);
  }

  const maxActivations = TIER_MAX_ACTIVATIONS[tier];

  const stripeCustomerId = typeof session.customer === "string"
    ? session.customer
    : session.customer?.id ?? null;

  const subscriptionId = typeof session.subscription === "string"
    ? session.subscription
    : session.subscription?.id ?? null;

  // Update profile with customer/sub info + subscription tier so the
  // billing page renders the right UI on next load. Log (but don't
  // throw) on failure — the license insert below is the critical path
  // for the customer; a stale profile row is a recoverable nit that
  // can be fixed manually via the admin panel.
  const profileUpdate: Record<string, string> = {
    subscription_tier: tier,
    subscription_status: "active",
  };
  if (stripeCustomerId) profileUpdate.stripe_customer_id = stripeCustomerId;
  const { error: profileError } = await client.from("profiles").update(profileUpdate).eq("id", userId);
  if (profileError) {
    console.warn(`Failed to update profile for user ${userId} (license will still be issued):`, profileError);
  }

  // At-most-one-active-per-(user, product, tier) is enforced in the DB by
  // idx_license_keys_one_active_per_user_product_tier (migration
  // 20260419000000). If the user already holds an active license at this
  // tier (e.g., Stripe retried the event), skip issuance and return.
  const { data: existing } = await client
    .from("license_keys")
    .select("id, license_key")
    .eq("user_id", userId)
    .eq("product", WEBHOOK_PRODUCT)
    .eq("tier", tier)
    .eq("status", "active")
    .maybeSingle();

  if (existing) {
    console.log(`User ${userId} already holds an active ${tier} license (${existing.id}). Stripe event was likely retried — no-op.`);
    return;
  }

  // Generate a fresh {prefix}.{12rand}.{16rand} key.
  const keyPart1 = crypto.randomUUID().replace(/-/g, "").substring(0, 12);
  const keyPart2 = crypto.randomUUID().replace(/-/g, "").substring(0, 16);
  const licenseKey = `${WEBHOOK_PREFIX}.${keyPart1}.${keyPart2}`;

  const { error: insertError } = await client
    .from("license_keys")
    .insert({
      user_id: userId,
      license_key: licenseKey,
      product: WEBHOOK_PRODUCT,
      tier,
      status: "active",
      max_activations: maxActivations,
      stripe_subscription_id: subscriptionId,
      stripe_customer_id: stripeCustomerId,
      notes: `Issued by Stripe Checkout (session ${session.id})`,
    });

  if (insertError) {
    console.error("Failed to insert license:", insertError);
    throw insertError;
  }

  console.log(`Issued ${tier} license ${licenseKey.substring(0, 12)}... for user ${userId}`);
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
  const subscriptionId = typeof invoice.subscription === "string"
    ? invoice.subscription
    : invoice.subscription?.id;

  if (!subscriptionId) return;

  const { error } = await client
    .from("license_keys")
    .update({ status: "active" })
    .eq("stripe_subscription_id", subscriptionId)
    .neq("status", "revoked");

  if (error) {
    console.error("invoice.paid update failed:", error);
    throw error;
  }
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
  // Strategy: walk every item, collect the tiers advertised in each
  // price's metadata, and pick the highest-ranked tier we know about.
  // Enterprise wins over professional; anything not in the known set
  // is ignored rather than trusted. That means adding a "premium"
  // tier later requires appending it to TIER_RANK below — which keeps
  // the precedence explicit, unlike a blind `first item wins` fallback.
  const TIER_RANK: Record<string, number> = {
    professional: 1,
    enterprise: 2,
  };
  const seenTiers = (subscription.items?.data ?? [])
    .map((item) => item.price?.metadata?.tier as string | undefined)
    .filter((t): t is string => typeof t === "string" && t in TIER_RANK);
  const tier = seenTiers.length > 0
    ? seenTiers.reduce((a, b) => (TIER_RANK[a] >= TIER_RANK[b] ? a : b))
    : undefined;

  const updateData: Record<string, string | number> = { status: licenseStatus };
  if (tier && TIER_MAX_ACTIVATIONS[tier] !== undefined) {
    updateData.tier = tier;
    updateData.max_activations = TIER_MAX_ACTIVATIONS[tier];
  }

  const { error } = await client
    .from("license_keys")
    .update(updateData)
    .eq("stripe_subscription_id", subscriptionId)
    .neq("status", "revoked");

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
    const profileUpdate: Record<string, string> = { subscription_status: licenseStatus };
    if (tier) profileUpdate.subscription_tier = tier;
    const { error: profileError } = await client
      .from("profiles")
      .update(profileUpdate)
      .eq("stripe_customer_id", stripeCustomerId);
    if (profileError) {
      // Don't throw — the license row already has the canonical status
      // and the admin panel can reconcile the profile. Swallowing silent
      // would hide future RLS / permission regressions.
      console.warn(`Failed to mirror subscription state to profile for customer ${stripeCustomerId}:`, profileError);
    }
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
    .neq("status", "revoked");

  if (error) {
    console.error("subscription.deleted update failed:", error);
    throw error;
  }

  const stripeCustomerId = typeof subscription.customer === "string"
    ? subscription.customer
    : subscription.customer?.id;

  if (stripeCustomerId) {
    const { error: profileError } = await client
      .from("profiles")
      .update({ subscription_tier: "free", subscription_status: "expired" })
      .eq("stripe_customer_id", stripeCustomerId);
    if (profileError) {
      // Non-fatal — the license row is already marked expired, which
      // is the source of truth for access control. A stale profile row
      // only affects the billing page's tier display.
      console.warn(`Failed to downgrade profile for customer ${stripeCustomerId}:`, profileError);
    }
  }

  console.log(`Expired licenses for subscription ${subscriptionId}`);
}
