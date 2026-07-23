import { serve } from "https://deno.land/std@0.177.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";
import { signCrl } from "../_shared/aidn2.ts";

// get-revocations — serves the signed CRL the SDK's LicenseRevocationProvider fetches, verifies (against the
// embedded public key), caches, and enforces offline. Output shape:
//   { kid, payload:base64url({iat,exp,rkids,rjti}), sig:base64url(ed25519 over payload) }
//
// Public + cacheable: it contains only opaque token/key ids (no PII) and is signed, so it's safe to serve
// unauthenticated behind a CDN cache. The SDK fails OPEN if it can't fetch/verify (availability > strictness),
// so a short cache is fine. Deploy is BLOCKED until the signing-key secret exists (same key as issue-license).

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
};

// CRL validity. Short enough that a newly-revoked token is denied everywhere within a day, long enough that a
// briefly-offline client keeps a still-valid CRL. Must be <= the SDK's revocation cache TTL expectations.
const CRL_EXP_DAYS = Number(Deno.env.get("CRL_EXP_DAYS") ?? "1");

serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: corsHeaders });
  }
  if (req.method !== "GET") {
    return new Response(JSON.stringify({ error: "method_not_allowed" }), {
      status: 405,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }

  try {
    const supabaseUrl = Deno.env.get("SUPABASE_URL");
    const supabaseServiceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
    if (!supabaseUrl || !supabaseServiceKey) {
      console.error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY");
      return new Response(JSON.stringify({ error: "server_configuration_error" }), {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    const { data, error } = await supabase.from("revocations").select("jti, kid");
    if (error) {
      console.error("revocations query error:", error);
      return new Response(JSON.stringify({ error: "server_error" }), {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    // Dedupe and drop nulls; a row may carry a jti, a kid, or both.
    const rjti = [...new Set((data ?? []).map((r) => r.jti).filter((v): v is string => !!v))];
    const rkids = [...new Set((data ?? []).map((r) => r.kid).filter((v): v is string => !!v))];

    let crl: string;
    try {
      crl = await signCrl(rjti, rkids, CRL_EXP_DAYS);
    } catch (e) {
      console.error("CRL signing failed:", e);
      return new Response(JSON.stringify({ error: "signing_unavailable" }), {
        status: 503,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    return new Response(crl, {
      status: 200,
      headers: {
        ...corsHeaders,
        "Content-Type": "application/json",
        // Let a CDN / the client cache the CRL for a fraction of its validity so revocations still propagate
        // within the day while the endpoint isn't hit on every SDK start.
        "Cache-Control": "public, max-age=3600",
      },
    });
  } catch (err) {
    console.error("Unexpected error:", err);
    return new Response(JSON.stringify({ error: "server_error" }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
