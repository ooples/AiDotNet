import { serve } from "https://deno.land/std@0.177.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

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
const COMMUNITY_PRODUCT = "aidotnet" as const;
const COMMUNITY_PREFIX = "aidn" as const;
const COMMUNITY_TIER = "community" as const;
// Unique partial index name that enforces at-most-one active license per
// (user, product, tier). See migration
// 20260419000000_add_product_to_license_keys.sql. When a concurrent request
// races us and wins the insert, Postgres surfaces this constraint name in
// the error and we treat it as "the other request already issued the
// license — go read it back".
const UNIQUE_ACTIVE_INDEX = "idx_license_keys_one_active_per_user_product_tier";

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

    const { data: { user }, error: authError } = await userClient.auth.getUser();

    if (authError || !user) {
      return new Response(
        JSON.stringify({ success: false, error: "unauthorized", message: "Invalid or expired session." }),
        { status: 401, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Use service role to check existing licenses and create new ones
    const serviceClient = createClient(supabaseUrl, supabaseServiceKey);

    // Helper that fetches the user's existing active community license for
    // this product, if any. Used both as the fast path (skip insert if one
    // already exists) and as the race-recovery path (another concurrent
    // request inserted it while we were running — read back what they wrote).
    async function fetchExistingActiveLicense() {
      return await serviceClient
        .from("license_keys")
        .select("id, license_key, status, tier")
        .eq("user_id", user.id)
        .eq("product", COMMUNITY_PRODUCT)
        .eq("tier", COMMUNITY_TIER)
        .eq("status", "active")
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
      return new Response(
        JSON.stringify({
          success: true,
          license_key: existingLicense.license_key,
          tier: COMMUNITY_TIER,
          is_existing: true,
          message: "You already have an active community license.",
        }),
        { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Generate a new license key: {prefix}.{12-char-random}.{16-char-random}.
    // Prefix comes from COMMUNITY_PREFIX so the client library can classify
    // the key by string alone without a DB round-trip.
    const keyPart1 = crypto.randomUUID().replace(/-/g, "").substring(0, 12);
    const keyPart2 = crypto.randomUUID().replace(/-/g, "").substring(0, 16);
    const licenseKey = `${COMMUNITY_PREFIX}.${keyPart1}.${keyPart2}`;

    // Insert the new community license. The at-most-one-active partial
    // unique index (migration 20260419000000) guarantees atomicity: if a
    // concurrent request from the same user wins the insert race, this
    // insert fails with Postgres error code 23505 (unique_violation) and
    // we recover by reading back the winner's row instead of creating a
    // duplicate.
    const { error: insertError } = await serviceClient
      .from("license_keys")
      .insert({
        user_id: user.id,
        license_key: licenseKey,
        product: COMMUNITY_PRODUCT,
        tier: COMMUNITY_TIER,
        status: "active",
        max_activations: 3,
        notes: "Self-registered community license",
      });

    if (insertError) {
      // PostgREST surfaces the native Postgres SQLSTATE in `code`; 23505 is
      // unique_violation. We additionally match on the constraint name so
      // that other unique constraints (e.g., the license_key column's own
      // uniqueness) don't get silently swallowed.
      const isRaceViolation =
        insertError.code === "23505" &&
        typeof insertError.message === "string" &&
        insertError.message.includes(UNIQUE_ACTIVE_INDEX);

      if (isRaceViolation) {
        // A concurrent request already issued an active community license
        // for this user. Read it back and return that one.
        const { data: racedLicense, error: refetchError } = await fetchExistingActiveLicense();
        if (!refetchError && racedLicense) {
          return new Response(
            JSON.stringify({
              success: true,
              license_key: racedLicense.license_key,
              tier: COMMUNITY_TIER,
              is_existing: true,
              message: "You already have an active community license.",
            }),
            { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } }
          );
        }
        // The unique violation fired but we can't read the winner back.
        // Fall through to the generic 500 so the caller retries cleanly.
        console.error("Race-recovery refetch failed:", refetchError);
      }

      console.error("Insert error:", insertError);
      return new Response(
        JSON.stringify({ success: false, error: "server_error", message: "Failed to create license. Please try again." }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    return new Response(
      JSON.stringify({
        success: true,
        license_key: licenseKey,
        tier: COMMUNITY_TIER,
        is_existing: false,
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
