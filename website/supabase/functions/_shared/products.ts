// Single source of truth for the `license_product` enum's display
// metadata on the EDGE-FUNCTION side. The admin Astro page has its own
// PRODUCTS array in website/src/pages/admin/licenses/index.astro because
// it runs in a different runtime (browser/SSR) and can't import from
// supabase/functions/_shared. Both files MUST stay aligned with each
// other AND with the Postgres enum migration at
// website/supabase/migrations/20260419000000_add_product_to_license_keys.sql.
//
// Adding a product means three coordinated changes:
//   1. Add the slug to the enum (new migration: ALTER TYPE … ADD VALUE).
//   2. Add an entry here.
//   3. Add an entry in the admin Astro PRODUCTS array.
//
// Until the runtime gap can be closed (e.g., by emitting both files from
// a JSON manifest), the duplication is the cost of admin-page-vs-edge-
// function isolation. Centralizing AT LEAST the edge-function side here
// closes the previous third-source-of-truth — sendLicenseKeyEmail used
// to ship its own private slug→display map, which the reviewer flagged
// in PR #1268.

export interface ProductInfo {
  /** The `license_product` enum slug stored in the DB. */
  slug: string;
  /** Human-readable display name used in email subject/body/signature. */
  displayName: string;
}

/**
 * Edge-function product registry. Keep aligned with the admin UI's
 * PRODUCTS array and the `license_product` Postgres enum.
 */
export const PRODUCTS: ReadonlyArray<ProductInfo> = [
  { slug: "aidotnet", displayName: "AiDotNet" },
  { slug: "harmonic_engine", displayName: "Harmonic Engine" },
];

/**
 * Looks up the human-readable product name for a `license_product` slug.
 * Returns `undefined` for unknown slugs so the caller can decide whether
 * to fall back to a generic display name (preserving "ship something
 * sensible") or warn (preserving "fail loudly on unknown product").
 *
 * `sendLicenseKeyEmail` callers should resolve this and pass it as
 * `productDisplayName` so the entire email — subject, heading, body
 * paragraph, signature — uses the same brand name.
 */
export function lookupProductDisplayName(slug: string | undefined | null): string | undefined {
  const cleaned = (slug ?? "").trim().toLowerCase();
  if (cleaned.length === 0) return undefined;
  for (const p of PRODUCTS) {
    if (p.slug === cleaned) return p.displayName;
  }
  return undefined;
}
