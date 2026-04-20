-- Migration: Add product classification to license_keys
--
-- Adds a `product` column so a single Supabase instance can issue and
-- administer licenses for multiple products (AiDotNet, Harmonic Engine, ...).
-- The product is also embedded in the license-key prefix (aidn.*, harm.*),
-- but we store it as a column so the admin panel, RLS, and issuance flow
-- don't have to parse the key string to classify it.
--
-- Accompanies:
--   - website/supabase/functions/register-community-license/index.ts
--     (selects/inserts with `.eq("product", "aidotnet")`)
--   - website/src/pages/admin/licenses/index.astro
--     (product filter dropdown, product column, product-scoped stats)

-- Product enum. Keep in sync with PRODUCT_META in the admin page.
do $$ begin
    create type public.license_product as enum ('aidotnet', 'harmonic_engine');
exception when duplicate_object then null;
end $$;

-- Add the column. DEFAULT backfills existing rows to 'aidotnet', then NOT NULL
-- locks the invariant so every row is product-classified going forward.
alter table public.license_keys
    add column if not exists product public.license_product not null default 'aidotnet';

-- Drop the default now that existing rows are backfilled. Keeping the default
-- in place would let any future insert path that forgets to populate `product`
-- silently create an AiDotNet license — in a multi-product schema that turns
-- a missing write into bad data. After this, every INSERT must name `product`
-- explicitly or the DB rejects it with a NOT NULL violation.
alter table public.license_keys
    alter column product drop default;

-- Index for product-scoped admin queries and the composite lookups used by
-- the community-license registration endpoint (user_id + product + tier + status).
create index if not exists idx_license_keys_product_tier_status
    on public.license_keys (product, tier, status);

-- Atomic-issuance guard: at most one active license per (user, product, tier).
-- This is the database-level companion to the check-then-insert logic in
-- register-community-license/index.ts — it closes the race where two
-- concurrent requests from the same user both pass the check and both insert.
-- Partial predicate (status = 'active') means suspended/revoked/expired
-- licenses don't block a user from being re-issued; only currently-active ones.
create unique index if not exists idx_license_keys_one_active_per_user_product_tier
    on public.license_keys (user_id, product, tier)
    where status = 'active';

comment on column public.license_keys.product is
    'Product this license is valid for (e.g., aidotnet, harmonic_engine). '
    'Matches the license-key prefix (aidn, harm, ...).';
comment on index public.idx_license_keys_one_active_per_user_product_tier is
    'Enforces at most one active license per (user, product, tier). '
    'Prevents duplicate issuance from concurrent check-then-insert races.';
