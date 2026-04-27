-- Migration: Add per-tier `capabilities` array to validate_license_key responses
--            and persist the new `last_seen_package` analytics tag
--
-- Issue: ooples/AiDotNet#1195 — the AiDotNet.Tensors release ships its own
-- license-enforcement boundary that authorises by capability rather than by
-- tier name. The tensor guard reads `capabilities` from the validation
-- response and looks up specific strings (`tensors:save`, `tensors:load`,
-- `model:save`, `model:load`) by exact ordinal match.
--
-- Rollout: this is the first of three coordinated changes per #1195:
--   1. (this) Server returns capabilities. Old clients ignore the new
--      field — additive, no schema-version break.
--   2. Upstream AiDotNet release wires the key into the tensor layer and
--      tags its requests with package=AiDotNet.
--   3. AiDotNet.Tensors release with the capability-scoped guard ships
--      last (until then, real keys would be rejected by the tensor guard
--      because the response wouldn't include any capabilities).
--
-- Tier → capability mapping (aligned with current public.license_tier enum):
--   community    → tensors:load   (read pre-trained models — free for OSS)
--   professional → tensors:save, tensors:load, model:save, model:load
--   enterprise   → all of the above + room for future enterprise:* extensions
--
-- The issue's table also lists `tensors-only` and `trial` rows; neither is
-- present in the current `license_tier` enum, so they're omitted here. If
-- the pricing page introduces them later, add the corresponding ENUM value
-- and a CASE arm in this function in the same migration.

-- Add a column to record which client package most recently called the
-- validation endpoint for this activation. Issue #1195 §1 calls this an
-- "analytics hint"; keeping it on license_activations alongside hostname
-- and os_description means it inherits the same RLS policy and is
-- automatically retained for the activation's lifetime. Old clients that
-- don't send `package` simply leave it NULL.
alter table public.license_activations
    add column if not exists last_seen_package text;

comment on column public.license_activations.last_seen_package is
    'Optional analytics tag identifying the client SDK package that last validated this activation '
    '(e.g., "AiDotNet", "AiDotNet.Tensors"). NULL for clients on SDK versions that pre-date issue #1195. '
    'NOT used for authorisation — capability gating happens client-side.';

create or replace function public.validate_license_key(
    p_license_key text,
    p_machine_id_hash text,
    p_hostname text default null,
    p_os_description text default null,
    p_package text default null
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
    v_capabilities jsonb;
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

    -- Resolve the per-tier capability set. Build the array once so success
    -- paths (existing-activation update, new-activation insert) return the
    -- same shape — divergence between the two paths would mean tensor-side
    -- enforcement behaves differently for a returning machine vs a new one.
    v_capabilities := case v_license.tier
        when 'community' then
            jsonb_build_array('tensors:load')
        when 'professional' then
            jsonb_build_array('tensors:save', 'tensors:load', 'model:save', 'model:load')
        when 'enterprise' then
            jsonb_build_array('tensors:save', 'tensors:load', 'model:save', 'model:load')
        else
            -- Defensive default: a tier that doesn't appear in this CASE
            -- (e.g., a future enum value added without updating this
            -- function) gets an empty capability set, so the tensor guard
            -- denies operations until the mapping is filled in. Better than
            -- silently granting full capabilities to an unrecognised tier.
            jsonb_build_array()
    end;

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
            os_description = coalesce(p_os_description, os_description),
            -- Overwrite (not coalesce) so a switch from one package to the
            -- other shows up immediately. coalesce would freeze the value
            -- to whichever package validated first on this activation.
            last_seen_package = p_package
        where id = v_existing_activation.id;

        return jsonb_build_object(
            'valid', true,
            'tier', v_license.tier::text,
            'capabilities', v_capabilities,
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
    insert into public.license_activations (license_key_id, machine_id_hash, hostname, os_description, last_seen_package)
    values (v_license.id, p_machine_id_hash, p_hostname, p_os_description, p_package);

    return jsonb_build_object(
        'valid', true,
        'tier', v_license.tier::text,
        'capabilities', v_capabilities,
        'license_id', v_license.id,
        'message', 'License validated and activated on this machine.'
    );
end;
$$;

comment on function public.validate_license_key is
    'Validates a license key and registers/updates a machine activation. Returns JSON with valid, tier, capabilities, error fields. Capabilities array shape introduced for AiDotNet.Tensors capability-scoped enforcement (issue #1195).';
