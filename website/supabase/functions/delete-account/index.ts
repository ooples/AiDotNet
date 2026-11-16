import { serve } from "https://deno.land/std@0.177.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";
import Stripe from "https://esm.sh/stripe@14.14.0?target=deno";

// Deletes the caller's own account: POST { confirm: "DELETE" }.
//
//   1. Refuses while a Stripe subscription would keep billing (the customer cancels first, from
//      /account/billing/, so nobody is charged for an account that no longer exists).
//   2. Revokes every license the account holds. The revoked-status trigger copies each into
//      public.revocations, whose rows outlive the license (ON DELETE SET NULL), so offline tokens
//      already issued to the account's machines stop validating.
//   3. Deletes the auth user; profiles, API keys, usage and licenses cascade.

const ALLOWED_ORIGIN = Deno.env.get("ALLOWED_ORIGIN") ?? "https://www.aidotnet.dev";
const corsHeaders = {
  "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};
const BILLING_STATUSES = new Set(["active", "trialing", "past_due", "unpaid", "incomplete"]);

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { ...corsHeaders, "Content-Type": "application/json" } });
}

serve(async (req: Request) => {
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: corsHeaders });
  if (req.method !== "POST") return json({ error: "method_not_allowed" }, 405);

  const authHeader = req.headers.get("Authorization");
  if (!authHeader) return json({ error: "unauthorized", message: "Sign in to delete your account." }, 401);
  const url = Deno.env.get("SUPABASE_URL"), anon = Deno.env.get("SUPABASE_ANON_KEY"), service = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  if (!url || !anon || !service) return json({ error: "server_configuration_error" }, 500);

  // Pass the token explicitly: see admin-resend-license-email for why argument-less getUser() fails.
  const jwt = authHeader.replace(/^Bearer\s+/i, "");
  const { data: { user }, error: authError } = await createClient(url, anon).auth.getUser(jwt);
  if (authError || !user) return json({ error: "unauthorized", message: "Invalid or expired session." }, 401);

  let body: { confirm?: unknown };
  try { body = await req.json(); } catch { return json({ error: "invalid_json" }, 400); }
  if (body.confirm !== "DELETE") return json({ error: "confirmation_required", message: "Type DELETE to confirm." }, 400);

  const admin = createClient(url, service);
  const { data: profile } = await admin.from("profiles").select("stripe_customer_id").eq("id", user.id).maybeSingle();
  const customer = profile?.stripe_customer_id as string | null | undefined;
  if (customer) {
    const key = Deno.env.get("STRIPE_SECRET_KEY");
    if (!key) return json({ error: "server_configuration_error" }, 500);
    const stripe = new Stripe(key, { apiVersion: "2023-10-16", httpClient: Stripe.createFetchHttpClient() });
    try {
      const subs = await stripe.subscriptions.list({ customer, status: "all", limit: 20 });
      if (subs.data.some((s) => BILLING_STATUSES.has(s.status) && !s.cancel_at_period_end)) {
        return json({
          error: "active_subscription",
          message: "Cancel your subscription on the Billing page first, so you are not charged again. Then delete your account.",
        }, 409);
      }
    } catch (err) {
      console.error(`delete-account: subscription check failed for ${user.id}:`, err);
      return json({ error: "stripe_error", message: "Could not check your subscription. Please try again." }, 502);
    }
  }

  const { error: revokeError } = await admin
    .from("license_keys")
    .update({ status: "revoked" })
    .eq("user_id", user.id)
    .neq("status", "revoked");
  if (revokeError) {
    console.error(`delete-account: license revoke failed for ${user.id}:`, revokeError);
    return json({ error: "revoke_failed", message: "Could not revoke your licenses. Nothing was deleted; please try again." }, 500);
  }

  const { error: deleteError } = await admin.auth.admin.deleteUser(user.id);
  if (deleteError) {
    console.error(`delete-account: deleteUser failed for ${user.id}:`, deleteError);
    return json({ error: "delete_failed", message: "Your licenses were revoked but the account could not be deleted. Email support@aidotnet.dev." }, 500);
  }
  return json({ deleted: true });
});
