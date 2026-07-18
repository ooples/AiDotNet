// Shared aidn2 (v2) token signer + CRL signer for the license edge functions.
//
// aidn2 grammar (must match src/Helpers/AsymmetricLicenseVerifier.cs in ooples/AiDotNet and the
// AiDotNet.Tensors verifier): aidn2.<base64url(claims_json)>.<base64url(ed25519_signature)>, where the
// Ed25519 signature is over the RAW UTF-8 bytes of claims_json. Claims mirror LicenseClaims:
//   { sub, tier, seats, iat, exp, kid, alg:"EdDSA", jti, caps[], mach?, scope? }
//
// The Ed25519 PRIVATE signing key never ships to clients — it lives ONLY here as the
// AIDOTNET_LICENSE_SIGNING_KEY_PKCS8 function secret (base64 PKCS#8). The matching PUBLIC key is embedded
// in the SDKs (as JWK OKP) so they can verify offline. Set the secret with:
//   supabase secrets set AIDOTNET_LICENSE_SIGNING_KEY_PKCS8=<base64 pkcs8> AIDOTNET_LICENSE_KID=<kid>

function b64urlEncode(bytes: Uint8Array): string {
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function b64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

async function importSigningKey(): Promise<CryptoKey> {
  const pkcs8B64 = Deno.env.get("AIDOTNET_LICENSE_SIGNING_KEY_PKCS8");
  if (!pkcs8B64) {
    throw new Error(
      "AIDOTNET_LICENSE_SIGNING_KEY_PKCS8 is not configured — cannot sign aidn2 tokens. " +
        "Set it (base64 PKCS#8 Ed25519 private key) as a function secret.",
    );
  }
  return await crypto.subtle.importKey(
    "pkcs8",
    b64ToBytes(pkcs8B64),
    { name: "Ed25519" },
    false,
    ["sign"],
  );
}

export interface Aidn2Claims {
  sub: string;
  tier: string;
  seats: number;
  caps: string[];
  /** Token id — revocation target (use the license row id). */
  jti: string;
  /** Validity in days. Keep short so a leaked offline token self-expires; the client re-validates online. */
  expDays: number;
  /** Optional machine-binding hash (node-lock). */
  mach?: string;
  /** Optional audience/scope binding (e.g. "ci"). */
  scope?: string;
}

/** Mints a signed aidn2 token. Field order mirrors LicenseClaims for readability (verification is
 *  order-agnostic — the signature is over the exact bytes emitted here). */
export async function signAidn2Token(c: Aidn2Claims): Promise<string> {
  const kid = Deno.env.get("AIDOTNET_LICENSE_KID") ?? "prod-2026a";
  const now = Math.floor(Date.now() / 1000);
  const claims: Record<string, unknown> = {
    sub: c.sub,
    tier: c.tier,
    seats: c.seats,
    iat: now,
    exp: now + c.expDays * 86400,
    kid,
    alg: "EdDSA",
    jti: c.jti,
    caps: c.caps,
  };
  if (c.mach) claims.mach = c.mach;
  if (c.scope) claims.scope = c.scope;

  const claimsBytes = new TextEncoder().encode(JSON.stringify(claims));
  const key = await importSigningKey();
  const sig = new Uint8Array(await crypto.subtle.sign({ name: "Ed25519" }, key, claimsBytes));
  return `aidn2.${b64urlEncode(claimsBytes)}.${b64urlEncode(sig)}`;
}

/** Signs a revocation list (CRL) envelope the SDK's LicenseRevocationProvider verifies:
 *  { kid, payload:base64url({iat,exp,rkids,rjti}), sig:base64url(ed25519 over payload) }. */
export async function signCrl(revokedJti: string[], revokedKids: string[], expDays: number): Promise<string> {
  const kid = Deno.env.get("AIDOTNET_LICENSE_KID") ?? "prod-2026a";
  const now = Math.floor(Date.now() / 1000);
  const payload = { iat: now, exp: now + expDays * 86400, rkids: revokedKids, rjti: revokedJti };
  const payloadBytes = new TextEncoder().encode(JSON.stringify(payload));
  const key = await importSigningKey();
  const sig = new Uint8Array(await crypto.subtle.sign({ name: "Ed25519" }, key, payloadBytes));
  return JSON.stringify({ kid, payload: b64urlEncode(payloadBytes), sig: b64urlEncode(sig) });
}

/** Maps a tier to its capability grants.
 *
 *  MUST stay byte-for-byte identical to the tier→capabilities CASE in the validate_license_key RPC
 *  (website/supabase/migrations/…_fix_validate_license_advisory_lock_uuid.sql) so an offline aidn2 token
 *  and an online validation of the same license grant the exact same capabilities — any drift means a
 *  customer's save works online but not offline (or vice-versa). If you extend a tier (e.g. add
 *  "model:encrypt" / "offline"), change BOTH places in one coordinated migration. */
export function capsForTier(tier: string): string[] {
  switch (tier) {
    case "professional":
    case "pro":
      return ["tensors:save", "tensors:load", "model:save", "model:load"];
    case "enterprise":
      return ["tensors:save", "tensors:load", "model:save", "model:load"];
    case "community":
    default:
      return ["tensors:load"];
  }
}
