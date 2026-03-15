import { serve } from "https://deno.land/std@0.177.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";
import Stripe from "https://esm.sh/stripe@14.14.0?target=deno";

const stripeSecretKey = Deno.env.get("STRIPE_SECRET_KEY");
const endpointSecret = Deno.env.get("STRIPE_WEBHOOK_SECRET");

if (!stripeSecretKey || !endpointSecret) {
  throw new Error("Missing required environment variables: STRIPE_SECRET_KEY or STRIPE_WEBHOOK_SECRET");
}

const stripe = new Stripe(stripeSecretKey, {
  apiVersion: "2023-10-16",
  httpClient: Stripe.createFetchHttpClient(),
});

serve(async (req: Request) => {
  if (req.method !== "POST") {
    return new Response(
      JSON.stringify({ error: "method_not_allowed" }),
      { status: 405, headers: { "Content-Type": "application/json" } }
    );
  }

  const signature = req.headers.get("stripe-signature");
  if (!signature) {
    return new Response(
      JSON.stringify({ error: "missing_signature" }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  let event: Stripe.Event;
  try {
    const body = await req.text();
    event = stripe.webhooks.constructEvent(body, signature, endpointSecret);
  } catch (err) {
    console.error("Webhook signature verification failed:", err);
    return new Response(
      JSON.stringify({ error: "invalid_signature" }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
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
    return new Response(
      JSON.stringify({ error: "processing_error" }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }

  return new Response(
    JSON.stringify({ received: true }),
    { status: 200, headers: { "Content-Type": "application/json" } }
  );
});

/**
 * Handles checkout.session.completed — creates a license key for the new subscriber.
 *
 * Expects the Stripe Checkout Session to have client_reference_id set to the Supabase user ID,
 * and metadata.tier set to "professional" or "enterprise".
 */
async function handleCheckoutCompleted(
  client: ReturnType<typeof createClient>,
  session: Stripe.Checkout.Session
) {
  const userId = session.client_reference_id;
  if (!userId) {
    throw new Error("checkout.session.completed: missing client_reference_id — cannot create license without a user ID");
  }

  const tier = session.metadata?.tier;
  if (!tier || !["professional", "enterprise"].includes(tier)) {
    throw new Error(`checkout.session.completed: invalid or missing tier '${tier}' in session metadata`);
  }

  const maxActivations = tier === "enterprise" ? 10 : 5;

  // Update user profile with Stripe customer ID and subscription tier
  const stripeCustomerId = typeof session.customer === "string"
    ? session.customer
    : session.customer?.id;

  if (stripeCustomerId) {
    await client
      .from("profiles")
      .update({
        stripe_customer_id: stripeCustomerId,
        subscription_tier: tier,
        subscription_status: "active",
      })
      .eq("id", userId);
  }

  // Check for existing active paid license for this user
  const { data: existingLicenses } = await client
    .from("license_keys")
    .select("id")
    .eq("user_id", userId)
    .eq("tier", tier)
    .eq("status", "active");

  if (existingLicenses && existingLicenses.length > 0) {
    console.log(`User ${userId} already has an active ${tier} license, skipping creation.`);
    return;
  }

  // Generate license key
  const keyPart1 = crypto.randomUUID().replace(/-/g, "").substring(0, 12);
  const keyPart2 = crypto.randomUUID().replace(/-/g, "").substring(0, 16);
  const licenseKey = `aidn.${keyPart1}.${keyPart2}`;

  const subscriptionId = typeof session.subscription === "string"
    ? session.subscription
    : session.subscription?.id || null;

  const { error: insertError } = await client
    .from("license_keys")
    .insert({
      user_id: userId,
      license_key: licenseKey,
      tier,
      status: "active",
      max_activations: maxActivations,
      stripe_subscription_id: subscriptionId,
      stripe_customer_id: stripeCustomerId || null,
      notes: `Created via Stripe Checkout (session: ${session.id})`,
    });

  if (insertError) {
    console.error("Failed to create license key:", insertError);
    throw insertError;
  }

  console.log(`Created ${tier} license for user ${userId}: ${licenseKey.substring(0, 12)}...`);
}

/**
 * Handles invoice.paid — keeps the license active and logs renewal.
 */
async function handleInvoicePaid(
  client: ReturnType<typeof createClient>,
  invoice: Stripe.Invoice
) {
  const subscriptionId = typeof invoice.subscription === "string"
    ? invoice.subscription
    : invoice.subscription?.id;

  if (!subscriptionId) return;

  // Ensure any license keys tied to this subscription are active
  const { error } = await client
    .from("license_keys")
    .update({ status: "active" })
    .eq("stripe_subscription_id", subscriptionId)
    .neq("status", "revoked"); // Don't reactivate manually revoked licenses

  if (error) {
    console.error("Failed to update license status on invoice.paid:", error);
  }
}

/**
 * Handles subscription updates — status changes, plan changes, cancellation scheduling.
 */
async function handleSubscriptionUpdated(
  client: ReturnType<typeof createClient>,
  subscription: Stripe.Subscription
) {
  const subscriptionId = subscription.id;

  // Map Stripe subscription status to our license status
  let licenseStatus: string;
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

  // Determine tier from the subscription's price metadata
  let tier: string | null = null;
  if (subscription.items?.data?.[0]?.price?.metadata?.tier) {
    tier = subscription.items.data[0].price.metadata.tier;
  }

  const updateData: Record<string, string | number> = { status: licenseStatus };
  if (tier) {
    updateData.tier = tier;
    // Update max_activations based on tier
    updateData.max_activations = tier === "enterprise" ? 10 : 5;
  }

  const { error } = await client
    .from("license_keys")
    .update(updateData)
    .eq("stripe_subscription_id", subscriptionId);

  if (error) {
    console.error("Failed to update license on subscription.updated:", error);
  }

  // Also update the user profile subscription status
  const stripeCustomerId = typeof subscription.customer === "string"
    ? subscription.customer
    : subscription.customer?.id;

  if (stripeCustomerId) {
    const profileUpdate: Record<string, string> = {
      subscription_status: licenseStatus,
    };
    if (tier) profileUpdate.subscription_tier = tier;

    await client
      .from("profiles")
      .update(profileUpdate)
      .eq("stripe_customer_id", stripeCustomerId);
  }
}

/**
 * Handles subscription deletion — expires the license key.
 */
async function handleSubscriptionDeleted(
  client: ReturnType<typeof createClient>,
  subscription: Stripe.Subscription
) {
  const subscriptionId = subscription.id;

  const { error } = await client
    .from("license_keys")
    .update({ status: "expired" })
    .eq("stripe_subscription_id", subscriptionId);

  if (error) {
    console.error("Failed to expire license on subscription.deleted:", error);
  }

  // Update user profile
  const stripeCustomerId = typeof subscription.customer === "string"
    ? subscription.customer
    : subscription.customer?.id;

  if (stripeCustomerId) {
    await client
      .from("profiles")
      .update({
        subscription_tier: "free",
        subscription_status: "expired",
      })
      .eq("stripe_customer_id", stripeCustomerId);
  }

  console.log(`Expired licenses for subscription ${subscriptionId}`);
}
