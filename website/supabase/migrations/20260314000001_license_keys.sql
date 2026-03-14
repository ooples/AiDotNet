-- Migration: Create license_keys table for BSL licensing
-- Stores license keys issued to users, linked to their Supabase auth profile.
-- Supports community (free), professional, and enterprise tiers.

-- License tier enum
do $$ begin
    create type public.license_tier as enum ('community', 'professional', 'enterprise');
exception when duplicate_object then null;
end $$;

-- License status enum
do $$ begin
    create type public.license_status as enum ('active', 'suspended', 'revoked', 'expired');
exception when duplicate_object then null;
end $$;

create table if not exists public.license_keys (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    license_key text not null unique,
    tier public.license_tier not null default 'community',
    status public.license_status not null default 'active',
    max_activations int not null default 3,
    stripe_subscription_id text,
    stripe_customer_id text,
    organization_name text,
    notes text,
    issued_at timestamptz not null default now(),
    expires_at timestamptz,
    revoked_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- Index for license key lookups (the primary validation path)
create unique index if not exists idx_license_keys_key
    on public.license_keys (license_key);

-- Index for user lookups (dashboard, account management)
create index if not exists idx_license_keys_user_id
    on public.license_keys (user_id);

-- Index for Stripe subscription management
create index if not exists idx_license_keys_stripe_sub
    on public.license_keys (stripe_subscription_id)
    where stripe_subscription_id is not null;

-- Index for admin queries by tier and status
create index if not exists idx_license_keys_tier_status
    on public.license_keys (tier, status);

-- Auto-update updated_at
create or replace function public.update_license_keys_updated_at()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

drop trigger if exists license_keys_updated_at on public.license_keys;
create trigger license_keys_updated_at
    before update on public.license_keys
    for each row
    execute function public.update_license_keys_updated_at();

-- RLS
alter table public.license_keys enable row level security;

-- Users can view their own license keys
create policy "license_keys_select_own"
    on public.license_keys
    for select
    to authenticated
    using (auth.uid() = user_id);

-- Only admins can insert/update/delete license keys
create policy "license_keys_insert_admin"
    on public.license_keys
    for insert
    to authenticated
    with check (public.is_admin());

create policy "license_keys_update_admin"
    on public.license_keys
    for update
    to authenticated
    using (public.is_admin());

create policy "license_keys_delete_admin"
    on public.license_keys
    for delete
    to authenticated
    using (public.is_admin());

-- Service role can also read for validation endpoint
create policy "license_keys_select_service"
    on public.license_keys
    for select
    to service_role
    using (true);

-- Comments
comment on table public.license_keys is 'BSL license keys issued to users. Supports community/professional/enterprise tiers.';
comment on column public.license_keys.license_key is 'Unique license key string presented by the client library for validation.';
comment on column public.license_keys.tier is 'License tier: community (free, <$1M rev), professional ($29/mo), enterprise ($99/mo).';
comment on column public.license_keys.max_activations is 'Maximum concurrent machine activations allowed for this license.';
comment on column public.license_keys.stripe_subscription_id is 'Stripe subscription ID for paid tiers. NULL for community licenses.';
