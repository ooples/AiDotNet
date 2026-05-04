import { serve } from "https://deno.land/std@0.177.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";
import { sendLicenseKeyEmail } from "../_shared/email.ts";

// Admin-only endpoint that re-sends the license-key email for an existing
// license_keys row. Used by the Resend button on /admin/licenses when the
// original Stripe-checkout email never reached the customer (no Resend key
// configured at the time, recipient bounced, transport hiccupped).
//
// Authorization model:
//   1. Caller must present a valid Supabase JWT in the Authorization header.
//   2. The corresponding profile must satisfy public.is_admin() — same
//      function the existing /admin/* RLS policies key off.
//
// The license row itself is unchanged; only the email is re-sent.

const ALLOWED_ORIGIN = Deno.env.get("ALLOWED_ORIGIN") ?? "https://www.aidotnet.dev";
const corsHeaders = {
  "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: corsHeaders });
  }
  if (req.method !== "POST") {
    return json({ success: false, error: "method_not_allowed" }, 405);
  }

  const authHeader = req.headers.get("Authorization");
  if (!authHeader) {
    return json({ success: false, error: "unauthorized", message: "Authentication required." }, 401);
  }

  const supabaseUrl = Deno.env.get("SUPABASE_URL");
  const supabaseAnonKey = Deno.env.get("SUPABASE_ANON_KEY");
  const supabaseServiceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  if (!supabaseUrl || !supabaseAnonKey || !supabaseServiceKey) {
    console.error("admin-resend-license-email: missing supabase env (URL / ANON_KEY / SERVICE_ROLE_KEY)");
    return json({ success: false, error: "server_configuration_error" }, 500);
  }

  // User-scoped client to verify the caller's identity. We use the same
  // is_admin() SQL function that gates the RLS policies on /admin/* tables;
  // running it as the user (not service-role) means an attacker who steals
  // a non-admin JWT cannot bypass the check by hitting this endpoint.
  const userClient = createClient(supabaseUrl, supabaseAnonKey, {
    global: { headers: { Authorization: authHeader } },
  });

  const { data: { user }, error: authError } = await userClient.auth.getUser();
  if (authError || !user) {
    return json({ success: false, error: "unauthorized", message: "Invalid or expired session." }, 401);
  }

  const { data: isAdminResult, error: isAdminError } = await userClient.rpc("is_admin");
  if (isAdminError) {
    console.error("admin-resend-license-email: is_admin() rpc failed:", isAdminError);
    return json({ success: false, error: "authorization_check_failed" }, 500);
  }
  if (!isAdminResult) {
    return json({ success: false, error: "forbidden", message: "Admin role required." }, 403);
  }

  let body: { license_id?: string };
  try {
    body = await req.json();
  } catch {
    return json({ success: false, error: "invalid_json" }, 400);
  }
  const licenseId = body?.license_id?.trim();
  if (!licenseId) {
    return json({ success: false, error: "missing_license_id" }, 400);
  }

  // Service-role client for the cross-user license lookup. The is_admin()
  // gate above is what authorizes this; service-role here just bypasses
  // the per-user RLS so the row read works.
  const serviceClient = createClient(supabaseUrl, supabaseServiceKey);

  const { data: license, error: licenseError } = await serviceClient
    .from("license_keys")
    .select("id, user_id, license_key, product, tier, status, customer_email")
    .eq("id", licenseId)
    .maybeSingle();

  if (licenseError) {
    console.error("admin-resend-license-email: license lookup failed:", licenseError);
    return json({ success: false, error: "license_lookup_failed" }, 500);
  }
  if (!license) {
    return json({ success: false, error: "license_not_found" }, 404);
  }

  // Recipient resolution:
  //   1. license.customer_email (admin-typed at issuance, e.g. bulk training)
  //   2. profiles.email by license.user_id
  //   3. fail with explicit error so the admin sees which row is missing data
  let recipient: string | null = license.customer_email ?? null;
  if (!recipient && license.user_id) {
    const { data: profile, error: profileError } = await serviceClient
      .from("profiles")
      .select("email")
      .eq("id", license.user_id)
      .maybeSingle();
    if (profileError) {
      console.error("admin-resend-license-email: profile lookup failed:", profileError);
    } else if (profile?.email) {
      recipient = profile.email;
    }
  }

  if (!recipient) {
    return json({
      success: false,
      error: "no_recipient_email",
      message: "Neither the license row's customer_email nor the linked profile has an email address. "
        + "Edit the license in the admin UI to set customer_email before re-sending.",
    }, 422);
  }

  const result = await sendLicenseKeyEmail({
    to: recipient,
    licenseKey: license.license_key,
    tier: license.tier,
    product: license.product,
    isExisting: true,
  });

  if (!result.ok) {
    return json({
      success: false,
      error: result.reason ?? "send_failed",
      status: result.status,
      message: "Failed to dispatch email. Check the Resend dashboard and the function logs.",
    }, 502);
  }

  return json({
    success: true,
    recipient,
    status: result.status,
    message: `License-key email re-sent to ${recipient}.`,
  }, 200);
});

function json(body: unknown, status: number): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders, "Content-Type": "application/json" },
  });
}
