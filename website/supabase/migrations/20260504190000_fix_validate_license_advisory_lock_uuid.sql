-- Fix: validate_license_key RPC threw `pg_advisory_xact_lock(uuid) does not
-- exist` on every new-machine activation, because the function takes only
-- bigint or two int args while v_license.id is uuid. Existing-activation
-- short-circuit hit the path before the lock, so the bug was invisible
-- on machines that already had a row in license_activations — but every
-- first-time activation on a fresh dev machine returned `server_error`
-- back to the AiDotNet client library, surfacing as
-- LicenseRequiredException to the customer.
--
-- See ooples/AiDotNet#1261 for the diagnostic trace.
--
-- Replace the lock argument with a deterministic bigint hash of the uuid:
-- hashtextextended(text, seed) returns bigint, is deterministic across
-- calls (same uuid always hashes to the same lock key), so concurrent
-- validations on the same license_key still serialize correctly. Only the
-- one broken line changes; the rest of the function body is unchanged.
--
-- This migration was applied to the production project (yfkqwpgjahoamlgckjib)
-- via the Management API on 2026-05-04 to unblock the AiDotNet Transformer
-- baseline runs in HarmonicEngine. This file commits the same fix so
-- environment refreshes / fresh installs pick it up without manual SQL.

CREATE OR REPLACE FUNCTION public.validate_license_key(
    p_license_key text,
    p_machine_id_hash text,
    p_hostname text DEFAULT NULL::text,
    p_os_description text DEFAULT NULL::text,
    p_package text DEFAULT NULL::text
)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $function$
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

    -- Advisory lock on the license key id to prevent race conditions
    -- when two concurrent validations try to activate the same license.
    --
    -- BUG FIX: pg_advisory_xact_lock takes bigint, but v_license.id is uuid.
    -- The previous version called it with the raw uuid which threw
    -- `function pg_advisory_xact_lock(uuid) does not exist` and broke
    -- every new-machine activation in production. hashtextextended(text,
    -- seed) returns bigint deterministically, so the same uuid always
    -- maps to the same lock key — the serialization invariant holds.
    perform pg_advisory_xact_lock(hashtextextended(v_license.id::text, 0));

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
$function$;
