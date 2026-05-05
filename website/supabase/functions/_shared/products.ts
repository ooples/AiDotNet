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
//   2. Add the slug + display name to PRODUCT_DISPLAY_NAMES below
//      (and the corresponding `ProductSlug` union literal).
//   3. Add an entry in the admin Astro PRODUCTS array.
//
// Until the runtime gap can be closed (e.g., by emitting both files from
// a JSON manifest), the duplication is the cost of admin-page-vs-edge-
// function isolation. Centralizing AT LEAST the edge-function side here
// closes the previous third-source-of-truth — sendLicenseKeyEmail used
// to ship its own private slug→display map, which the reviewer flagged
// in PR #1268.

/**
 * Compile-time-checked union of every supported `license_product` enum
 * slug. Adding a new product requires extending this union AND the
 * `PRODUCT_DISPLAY_NAMES` registry below — TypeScript will refuse to
 * compile if the two drift apart, which is the whole point of the union.
 *
 * Callers that hold a slug from a typed source (constant, env var,
 * stripe-webhook config) should use `ProductSlug` directly so a typo is
 * caught at build time instead of at runtime via
 * `resolveProductDisplayName` throwing on the unknown slug.
 */
export type ProductSlug = "aidotnet" | "harmonic_engine";

export interface ProductInfo {
  /** The `license_product` enum slug stored in the DB. */
  slug: ProductSlug;
  /** Human-readable display name used in email subject/body/signature. */
  displayName: string;
}

/**
 * Slug → display-name registry. The `Record<ProductSlug, string>` type
 * forces this object to have an entry for every member of `ProductSlug` —
 * forgetting one is a TypeScript error, not a silent runtime fallback to
 * a generic capitalized form.
 */
const PRODUCT_DISPLAY_NAMES: Readonly<Record<ProductSlug, string>> = {
  aidotnet: "AiDotNet",
  harmonic_engine: "Harmonic Engine",
};

/**
 * Edge-function product registry. Keep aligned with the admin UI's
 * PRODUCTS array and the `license_product` Postgres enum.
 */
export const PRODUCTS: ReadonlyArray<ProductInfo> =
  (Object.keys(PRODUCT_DISPLAY_NAMES) as ProductSlug[]).map((slug) => ({
    slug,
    displayName: PRODUCT_DISPLAY_NAMES[slug],
  }));

/**
 * Type guard — narrows an arbitrary string (e.g., a Postgres row value
 * read at runtime) to the `ProductSlug` union. Returns false for
 * unknown slugs so the caller can decide how to handle drift between
 * the DB enum and this registry.
 *
 * Implementation note: uses `Object.prototype.hasOwnProperty.call` rather
 * than the `in` operator. The `in` operator on a plain object also
 * matches inherited keys like `toString`, `constructor`, `__proto__`,
 * `hasOwnProperty`, `isPrototypeOf`, etc. — so `isProductSlug("toString")`
 * with the naive `in` check would incorrectly return true and any caller
 * that fed an attacker-controlled or accidentally-prototype-named string
 * into `resolveProductDisplayName` would either bypass the validation or
 * crash at the bracket access. Explicit own-property check fixes the
 * false positive (review-comment #1268.peG9 / #1268.pmPg).
 */
export function isProductSlug(value: unknown): value is ProductSlug {
  return typeof value === "string"
    && Object.prototype.hasOwnProperty.call(PRODUCT_DISPLAY_NAMES, value);
}

/**
 * Resolves a product slug to its display name. Throws on unknown slugs
 * because misbranding a customer-facing email is a programmer error
 * (the slug got into the DB enum without being added to this registry)
 * — failing fast surfaces it during local testing instead of shipping
 * a customer email titled "Welcome to Some_New_Product".
 *
 * For tolerant lookups (e.g., a logging path that should never throw)
 * use {@link lookupProductDisplayName}, which returns `undefined` on
 * unknown slugs instead of throwing. Both functions live in this module
 * — there is no exported direct-access surface for the underlying
 * registry (the registry is module-private to keep the resolver as the
 * single point of guarded lookup). Closes review-comment #1268.pmPy.
 */
export function resolveProductDisplayName(slug: string | null | undefined): string {
  const cleaned = (slug ?? "").trim().toLowerCase();
  if (cleaned.length === 0) {
    throw new Error("resolveProductDisplayName: product slug is required (got empty/missing).");
  }
  if (!isProductSlug(cleaned)) {
    throw new Error(
      `resolveProductDisplayName: '${cleaned}' is not a known product slug. ` +
      "Add it to PRODUCT_DISPLAY_NAMES in products.ts and ensure the " +
      "license_product enum migration covers it.",
    );
  }
  return PRODUCT_DISPLAY_NAMES[cleaned];
}

/**
 * Tolerant variant — returns `undefined` for unknown slugs instead of
 * throwing. Retained for callers that intentionally want a soft lookup
 * (e.g., a logging or analytics path where misbranding is preferable to
 * a 500). For email/subject building, use `resolveProductDisplayName`.
 */
export function lookupProductDisplayName(slug: string | undefined | null): string | undefined {
  const cleaned = (slug ?? "").trim().toLowerCase();
  if (cleaned.length === 0) return undefined;
  return isProductSlug(cleaned) ? PRODUCT_DISPLAY_NAMES[cleaned] : undefined;
}
