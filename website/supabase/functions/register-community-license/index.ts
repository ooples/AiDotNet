import { serve as serveHttp } from "https://deno.land/std@0.177.0/http/server.ts";
import { withCors } from "../_shared/cors.ts";
const serve = (handler: (req: Request) => Response | Promise<Response>) => serveHttp(withCors(handler));
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";
import { sendLicenseKeyEmail } from "../_shared/email.ts";
import { type ProductSlug } from "../_shared/products.ts";
import { capsForTier, signAidn2Token } from "../_shared/aidn2.ts";

// Short-lived aidn2 "bootstrap" token minted at community registration so the user can LOAD models offline
// immediately (before the SDK's first online validation derives a machine-bound token). It is NOT
// machine-bound (no machine at registration), but that's safe HERE and ONLY here: community caps are
// `tensors:load` — no save/persist — so a leaked bootstrap token grants nothing beyond offline read, which
// community already allows. Paid tiers deliberately do NOT get a non-bound token; their SDK derives a
// machine-bound one via issue-license. Kept short so it self-expires well inside the attestation window.
const BOOTSTRAP_EXP_DAYS = 7;

/** Best-effort: mints a non-bound, load-only community bootstrap token, or null if signing isn't configured
 *  or fails. NEVER throws — a signing hiccup must not fail an otherwise-successful registration. */
async function mintCommunityBootstrap(licenseId: string): Promise<string | null> {
  try {
    return await signAidn2Token({
      sub: licenseId,          // opaque license id, never the reusable AIDN-* key
      tier: COMMUNITY_TIER,
      seats: 1,
      caps: capsForTier(COMMUNITY_TIER), // ["tensors:load"] — load-only, so non-binding is safe
      jti: licenseId,
      expDays: BOOTSTRAP_EXP_DAYS,
      // no mach (no machine at registration), no scope
    });
  } catch (err) {
    console.warn("register-community-license: bootstrap token mint skipped:", (err as Error).message);
    return null;
  }
}

// Default matches the production origin the marketing site actually serves from.
// Override via the ALLOWED_ORIGIN function secret for staging / preview envs.
const ALLOWED_ORIGIN = Deno.env.get("ALLOWED_ORIGIN") ?? "https://www.aidotnet.dev";

const corsHeaders = {
  "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

// Product identity for the self-service community license. The marketing
// site's "Get Free License" button is AiDotNet-specific; other products
// have their own issuance paths. Keeping these as named constants avoids
// string drift between the lookup and insert paths and makes
// product-specific changes a one-liner.
const COMMUNITY_PRODUCT: ProductSlug = "aidotnet";
// Key prefix that the AiDotNet client library parses. See
// LicenseValidator.IsServerValidatedKeyFormat in src/Helpers/LicenseValidator.cs:
// ≥4 dash-delimited segments, segment[0]=="AIDN", last segment is ≥8 hex chars.
// The dotted `aidn.{id}.{sig}` form is also accepted but requires HMAC-SHA256
// signing against a build key this function does not have access to — so the
// dash form is the only viable issuance path here. See ooples/AiDotNet#1262.
const COMMUNITY_PREFIX = "AIDN-PROD" as const;
const COMMUNITY_TIER = "community" as const;
// Self-service issuance is serialized by a service-only database function.

serve(async (req: Request) => {
  // Handle CORS preflight
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: corsHeaders });
  }

  if (req.method !== "POST") {
    return new Response(
      JSON.stringify({ success: false, error: "method_not_allowed" }),
      { status: 405, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }

  try {
    // Verify the user is authenticated via the Authorization header
    const authHeader = req.headers.get("Authorization");
    if (!authHeader) {
      return new Response(
        JSON.stringify({ success: false, error: "unauthorized", message: "Authentication required." }),
        { status: 401, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const supabaseUrl = Deno.env.get("SUPABASE_URL");
    const supabaseAnonKey = Deno.env.get("SUPABASE_ANON_KEY");
    const supabaseServiceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");

    if (!supabaseUrl || !supabaseAnonKey || !supabaseServiceKey) {
      console.error("Missing required environment variables: SUPABASE_URL, SUPABASE_ANON_KEY, or SUPABASE_SERVICE_ROLE_KEY");
      return new Response(
        JSON.stringify({ success: false, error: "server_configuration_error" }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Create a client with the user's JWT to verify identity
    const userClient = createClient(supabaseUrl, supabaseAnonKey, {
      global: { headers: { Authorization: authHeader } },
    });

    // Pass the token explicitly. esm.sh resolves supabase-js@2.39.3's auth dependency to the
    // latest gotrue-js, whose argument-less getUser() returns AuthSessionMissingError without a
    // stored session unless the client sets hasCustomAuthorizationHeader, which 2.39.3 never does.
    const jwt = authHeader.replace(/^Bearer\s+/i, "");
    const { data: { user }, error: authError } = await userClient.auth.getUser(jwt);

    if (authError || !user) {
      return new Response(
        JSON.stringify({ success: false, error: "unauthorized", message: "Invalid or expired session." }),
        { status: 401, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Use service role to check existing licenses and create new ones
    const serviceClient = createClient(supabaseUrl, supabaseServiceKey);

    // Fast path for existing Community grants, including legacy duplicate rows.
    // The issuance RPC below rechecks under a lock before creating a new key.
    async function fetchExistingActiveLicense() {
      return await serviceClient
        .from("license_keys")
        .select("id, license_key, status, tier")
        .eq("user_id", user.id)
        .eq("product", COMMUNITY_PRODUCT)
        .eq("tier", COMMUNITY_TIER)
        .eq("status", "active")
        .order("id", { ascending: true })
        .limit(1)
        .maybeSingle();
    }

    // Fast-path check: if the user already has an active community license
    // for this product, return it without attempting an insert.
    const { data: existingLicense, error: queryError } = await fetchExistingActiveLicense();

    if (queryError) {
      console.error("Query error:", queryError);
      return new Response(
        JSON.stringify({ success: false, error: "server_error", message: "Failed to check existing licenses." }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    if (existingLicense) {
      const bootstrap_token = await mintCommunityBootstrap(existingLicense.id);
      return new Response(
        JSON.stringify({
          success: true,
          license_key: existingLicense.license_key,
          tier: COMMUNITY_TIER,
          is_existing: true,
          bootstrap_token,
          message: "You already have an active community license.",
        }),
        { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Generate a fresh AIDN-PROD-COMMUNITY-{32hex} key. Format must satisfy
    // LicenseValidator.IsServerValidatedKeyFormat in the AiDotNet client
    // library: ≥4 dash-delimited segments, segment[0]=="AIDN", last segment
    // is ≥8 hex chars. Empirically verified that `AIDN-PROD-{TIER}-{32hex}`
    // validates end-to-end through the online validate-license path.
    const keyHex = crypto.randomUUID().replace(/-/g, "");
    const licenseKey = `${COMMUNITY_PREFIX}-${COMMUNITY_TIER.toUpperCase()}-${keyHex}`;

    // Serialize concurrent self-service signups without restricting independent grants.
    const { data: issued, error: insertError } = await serviceClient.rpc("issue_aidotnet_community_license", {
      p_user_id: user.id,
      p_license_key: licenseKey,
    });
    if (insertError || !issued?.id || !issued?.license_key) {
      console.error("Community issuance error:", insertError);
      return new Response(
        JSON.stringify({ success: false, error: "server_error", message: "Failed to create license. Please try again." }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }
    if (issued.is_existing) {
      const bootstrap_token = await mintCommunityBootstrap(issued.id);
      return new Response(JSON.stringify({
        success: true, license_key: issued.license_key, tier: COMMUNITY_TIER,
        is_existing: true, bootstrap_token, message: "You already have an active community license.",
      }), { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } });
    }

    // Best-effort key delivery email. The license row above is the source
    // of truth; failure to email is logged but never thrown so a flaky
    // transport doesn't reject a successful issuance (the user already has
    // the key in the response body and on /account/licenses).
    if (user.email) {
      await sendLicenseKeyEmail({
        to: user.email,
        licenseKey,
        tier: COMMUNITY_TIER,
        product: COMMUNITY_PRODUCT,
        isExisting: false,
      });
    } else {
      console.warn(
        `Community license ${licenseKey.substring(0, 12)}... issued for user ${user.id} `
        + `but no email on profile — user must retrieve key from /account/licenses.`,
      );
    }

    const bootstrap_token = await mintCommunityBootstrap(issued.id);
    return new Response(
      JSON.stringify({
        success: true,
        license_key: licenseKey,
        tier: COMMUNITY_TIER,
        is_existing: false,
        bootstrap_token,
        message: "Community license created successfully! Set AIDOTNET_LICENSE_KEY or save to ~/.aidotnet/license.key",
      }),
      { status: 201, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (err) {
    console.error("Unexpected error:", err);
    return new Response(
      JSON.stringify({ success: false, error: "server_error", message: "An unexpected error occurred." }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
