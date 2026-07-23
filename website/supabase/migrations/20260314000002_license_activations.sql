-- Migration: Create license_activations table for tracking machine activations
-- Each license key can be activated on a limited number of machines (max_activations).
-- The machine_id_hash is a one-way hash of the machine fingerprint — not PII.

create table if not exists public.license_activations (
    id uuid primary key default gen_random_uuid(),
    license_key_id uuid not null references public.license_keys(id) on delete cascade,
    machine_id_hash text not null,
    hostname text,
    os_description text,
    first_seen_at timestamptz not null default now(),
    last_seen_at timestamptz not null default now(),
    is_active boolean not null default true,
    deactivated_at timestamptz,
    created_at timestamptz not null default now()
);

-- Unique constraint: one activation per machine per license key
create unique index if not exists idx_license_activations_key_machine
    on public.license_activations (license_key_id, machine_id_hash)
    where is_active = true;

-- Index for license key lookups (count activations)
create index if not exists idx_license_activations_key
    on public.license_activations (license_key_id, is_active);

-- RLS
alter table public.license_activations enable row level security;

-- Users can view activations for their own license keys; admins can view all
create policy "license_activations_select_own"
    on public.license_activations
    for select
    to authenticated
    using (
        public.is_admin() or
        license_key_id in (
            select id from public.license_keys where user_id = auth.uid()
        )
    );

-- Only admins can manage activations directly
create policy "license_activations_insert_admin"
    on public.license_activations
    for insert
    to authenticated
    with check (public.is_admin());

create policy "license_activations_update_admin"
    on public.license_activations
    for update
    to authenticated
    using (public.is_admin());

create policy "license_activations_delete_admin"
    on public.license_activations
    for delete
    to authenticated
    using (public.is_admin());

-- Service role can manage activations for the validation endpoint
create policy "license_activations_all_service"
    on public.license_activations
    for all
    to service_role
    using (true)
    with check (true);

-- Function: validate and activate a license key
-- Called by the license validation endpoint
create or replace function public.validate_license_key(
    p_license_key text,
    p_machine_id_hash text,
    p_hostname text default null,
    p_os_description text default null
)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
    v_license record;
    v_activation_count int;
    v_existing_activation record;
begin
    -- Look up the license key
    select * into v_license
    from public.license_keys
    where license_key = p_license_key;

    if not found then
        return jsonb_build_object(
            'valid', false,
            'error', 'invalid_key',
            'message', 'License key not found.'
        );
    end if;

    -- Check license status
    if v_license.status != 'active' then
        return jsonb_build_object(
            'valid', false,
            'error', 'license_' || v_license.status::text,
            'message', 'License is ' || v_license.status::text || '.'
        );
    end if;

    -- Check expiration
    if v_license.expires_at is not null and v_license.expires_at < now() then
        return jsonb_build_object(
            'valid', false,
            'error', 'license_expired',
            'message', 'License has expired.'
        );
    end if;

    -- Check if this machine is already activated
    select * into v_existing_activation
    from public.license_activations
    where license_key_id = v_license.id
      and machine_id_hash = p_machine_id_hash
      and is_active = true;

    if found then
        -- Update last_seen_at for existing activation
        update public.license_activations
        set last_seen_at = now(),
            hostname = coalesce(p_hostname, hostname),
            os_description = coalesce(p_os_description, os_description)
        where id = v_existing_activation.id;

        return jsonb_build_object(
            'valid', true,
            'tier', v_license.tier::text,
            'license_id', v_license.id,
            'activation_id', v_existing_activation.id,
            'message', 'License validated (existing activation).'
        );
    end if;

    -- Advisory lock on the license key ID to prevent race conditions
    -- when two concurrent validations try to activate the same license
    perform pg_advisory_xact_lock(v_license.id);

    -- Re-count active activations after acquiring the lock
    select count(*) into v_activation_count
    from public.license_activations
    where license_key_id = v_license.id
      and is_active = true;

    if v_activation_count >= v_license.max_activations then
        return jsonb_build_object(
            'valid', false,
            'error', 'activation_limit',
            'message', 'Maximum activations (' || v_license.max_activations || ') reached. Deactivate a machine first.',
            'current_activations', v_activation_count,
            'max_activations', v_license.max_activations
        );
    end if;

    -- Create new activation
    insert into public.license_activations (license_key_id, machine_id_hash, hostname, os_description)
    values (v_license.id, p_machine_id_hash, p_hostname, p_os_description);

    return jsonb_build_object(
        'valid', true,
        'tier', v_license.tier::text,
        'license_id', v_license.id,
        'message', 'License validated and activated on this machine.'
    );
end;
$$;

-- Restrict execution to only the service_role and authenticated users (not anon)
revoke execute on function public.validate_license_key from anon;
grant execute on function public.validate_license_key to service_role;

-- Comments
comment on table public.license_activations is 'Tracks which machines have activated a given license key. Enforces max_activations limit.';
comment on column public.license_activations.machine_id_hash is 'One-way SHA-256 hash of machine fingerprint. Not reversible to PII.';
comment on function public.validate_license_key is 'Validates a license key and registers/updates a machine activation. Returns JSON with valid, tier, error fields.';
