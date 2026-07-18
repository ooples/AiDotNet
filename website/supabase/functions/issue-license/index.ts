import { serve } from "https://deno.land/std@0.177.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";
import { Aidn2Claims, capsForTier, signAidn2Token } from "../_shared/aidn2.ts";

// issue-license — the online→offline bridge of the v2 hybrid model.
//
// After the SDK validates online (validate-license), it calls this endpoint to obtain a SHORT-LIVED,
// machine-bound `aidn2` token it can cache and verify OFFLINE (against the public key embedded in the SDK)
// for the attestation window, without re-contacting the server on every run. Because the token is short-exp
// and machine-bound, a leaked cache file self-expires and won't validate on another machine; revocation is
// enforced by the CRL (get-revocations) via the token's jti (= license id). This is what lets an air-gapped
// or offline build keep working between online check-ins while staying revocable and expiring.
//
// Deploy is BLOCKED until the Ed25519 signing key exists as a function secret (the private key never ships
// to clients). Set it before deploying:
//   supabase secrets set AIDOTNET_LICENSE_SIGNING_KEY_PKCS8=<base64 pkcs8> AIDOTNET_LICENSE_KID=<kid>
// The matching PUBLIC key (JWK OKP) must be embedded in the released SDKs as AiDotNet.LicensePublicKey.

const ALLOWED_ORIGIN = Deno.env.get("ALLOWED_ORIGIN") ?? "https://www.aidotnet.dev";

const corsHeaders = {
  "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

// Offline token lifetime. Kept short (a token is only a cache of a fresh online check) so a leaked cache
// self-expires well inside the SDK's attestation window (AttestationValidity 30d + 7d grace). Override per
// environment with OFFLINE_TOKEN_EXP_DAYS.
const DEFAULT_EXP_DAYS = Number(Deno.env.get("OFFLINE_TOKEN_EXP_DAYS") ?? "14");

interface IssueRequest {
  license_key: string;
  /** The SDK's machine fingerprint hash — the token is node-locked to it (claims.mach). Must be the exact
   *  value the SDK's LicenseValidator.GetMachineIdHash() produces, or offline verification will reject it. */
  machine_id_hash: string;
  hostname?: string;
  os_description?: string;
  /** Optional audience binding (e.g. "ci") — the SDK must set AIDOTNET_LICENSE_SCOPE to the same value. */
  scope?: string;
}

serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: corsHeaders });
  }
  if (req.method !== "POST") {
    return new Response(JSON.stringify({ valid: false, error: "method_not_allowed" }), {
      status: 405,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }

  try {
    // The IssueRequest type is erased at runtime, so validate the shape explicitly: reject invalid JSON with
    // a 400 (not a 500), and require license_key/machine_id_hash to be non-blank STRINGS so arrays/objects/
    // whitespace can't slip through to the RPC as "present" fields.
    let raw: unknown;
    try {
      raw = await req.json();
    } catch {
      return new Response(
        JSON.stringify({ valid: false, error: "invalid_json", message: "Request body is not valid JSON." }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }
    const r = raw as Partial<IssueRequest>;
    if (
      typeof raw !== "object" || raw === null ||
      typeof r.license_key !== "string" || r.license_key.trim().length === 0 ||
      typeof r.machine_id_hash !== "string" || r.machine_id_hash.trim().length === 0 ||
      (r.scope !== undefined && typeof r.scope !== "string") ||
      (r.hostname !== undefined && typeof r.hostname !== "string") ||
      (r.os_description !== undefined && typeof r.os_description !== "string")
    ) {
      return new Response(
        JSON.stringify({
          valid: false,
          error: "missing_fields",
          message: "license_key and machine_id_hash are required non-empty strings.",
        }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }
    const body: IssueRequest = {
      license_key: r.license_key.trim(),
      machine_id_hash: r.machine_id_hash.trim(),
      hostname: r.hostname,
      os_description: r.os_description,
      scope: r.scope,
    };

    const supabaseUrl = Deno.env.get("SUPABASE_URL");
    const supabaseServiceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
    if (!supabaseUrl || !supabaseServiceKey) {
      console.error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY");
      return new Response(JSON.stringify({ valid: false, error: "server_configuration_error" }), {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    // Re-validate online through the SAME RPC the validate-license endpoint uses. This confirms the key is
    // active + unexpired, enforces the activation-seat limit for this machine, and yields the authoritative
    // tier + license_id — so an offline token is only ever minted off a fresh, seat-checked online decision.
    const { data, error } = await supabase.rpc("validate_license_key", {
      p_license_key: body.license_key,
      p_machine_id_hash: body.machine_id_hash,
      p_hostname: body.hostname ?? null,
      p_os_description: body.os_description ?? null,
      p_package: "offline-issue",
    });

    if (error) {
      console.error("RPC error:", error);
      return new Response(
        JSON.stringify({ valid: false, error: "server_error", message: "Validation failed. Try again." }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    const result = data as Record<string, unknown>;
    if (!result.valid) {
      // Pass the RPC's own rejection (invalid_key / license_revoked / license_expired / activation_limit)
      // straight through — no offline token for a key that can't validate online right now.
      return new Response(JSON.stringify(result), {
        status: 403,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const tier = String(result.tier ?? "community");
    const licenseId = String(result.license_id ?? "");
    // Prefer the caps the RPC actually returned (authoritative); fall back to the shared tier map only if an
    // older RPC build omitted them. capsForTier is kept identical to the RPC's mapping so both agree.
    const caps = Array.isArray(result.capabilities) ? (result.capabilities as string[]) : capsForTier(tier);

    const claims: Aidn2Claims = {
      // Opaque license id, NOT the reusable license_key: the token payload is only base64url-encoded (not
      // encrypted), so anyone who reads a leaked cached token must not be able to recover the permanent
      // online key and mint fresh tokens. license_id is the same non-secret value already returned to clients.
      sub: licenseId,
      tier,
      seats: 1,
      caps,
      jti: licenseId, // revocation target — matches license_keys.id, the CRL's rjti entries
      expDays: DEFAULT_EXP_DAYS,
      mach: body.machine_id_hash, // node-lock to the requesting machine
      scope: body.scope,
    };

    let token: string;
    try {
      token = await signAidn2Token(claims);
    } catch (e) {
      // Signing key not configured yet — surface a clear, actionable error rather than a generic 500.
      console.error("aidn2 signing failed:", e);
      return new Response(
        JSON.stringify({
          valid: false,
          error: "signing_unavailable",
          message: "Offline license issuance is not yet configured on the server.",
        }),
        { status: 503, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    return new Response(
      JSON.stringify({
        valid: true,
        tier,
        capabilities: caps,
        license_id: licenseId,
        offline_token: token,
        expires_in_days: DEFAULT_EXP_DAYS,
        message: "Offline license token issued.",
      }),
      { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } },
    );
  } catch (err) {
    console.error("Unexpected error:", err);
    return new Response(
      JSON.stringify({ valid: false, error: "server_error", message: "An unexpected error occurred." }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } },
    );
  }
});
