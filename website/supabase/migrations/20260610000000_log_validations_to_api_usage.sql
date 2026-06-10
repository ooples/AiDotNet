-- Migration: validate_license_key writes a row to public.api_usage so the
--            /account/usage/ dashboard actually has data to render.
--
-- Issue: The Usage page reads from api_usage but nothing in the codebase
-- writes to it. Until this migration, the page rendered "No usage data
-- yet" for every authenticated user, even active license holders. This
-- migration plumbs the existing validate_license_key RPC (which fires
-- once per SDK validation call) into api_usage so each successful
-- validation logs a row.
--
-- Rollout: drop+recreate the 5-arg overload (the only one currently in
-- use after 20260427000000). The signature, return shape, and CASE arms
-- for v_capabilities are preserved verbatim — only the side-effect of
-- recording the call is added. Old callers continue to work.

drop function if exists public.validate_license_key(text, text, text, text, text);

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
    v_started timestamptz := clock_timestamp();
    v_status_code int;
    v_response jsonb;
    v_endpoint constant text := 'validate_license_key';
    -- Endpoint label kept short and grep-able so the /account/usage/
    -- "Calls by Endpoint" chart shows a clean row. If additional RPC
    -- entry points start logging too, follow the same naming scheme.
begin
    -- Look up the license key
    select * into v_license
    from public.license_keys
    where license_key = p_license_key;

    if not found then
        v_status_code := 404;
        v_response := jsonb_build_object(
            'valid', false,
            'error', 'invalid_key',
            'message', 'License key not found.'
        );
        -- No user_id known when the key is invalid — skip the api_usage
        -- write rather than logging against an arbitrary user. (The
        -- service-role validate-license edge function still logs the
        -- failure server-side for fraud / abuse monitoring.)
        return v_response;
    end if;

    -- Check license status
    if v_license.status != 'active' then
        v_status_code := 403;
        v_response := jsonb_build_object(
            'valid', false,
            'error', 'license_' || v_license.status::text,
            'message', 'License is ' || v_license.status::text || '.'
        );
        insert into public.api_usage (user_id, endpoint, status_code, latency_ms)
        values (
            v_license.user_id,
            v_endpoint,
            v_status_code,
            extract(milliseconds from clock_timestamp() - v_started)::int
        );
        return v_response;
    end if;

    -- Check expiration
    if v_license.expires_at is not null and v_license.expires_at < now() then
        v_status_code := 403;
        v_response := jsonb_build_object(
            'valid', false,
            'error', 'license_expired',
            'message', 'License has expired.'
        );
        insert into public.api_usage (user_id, endpoint, status_code, latency_ms)
        values (
            v_license.user_id,
            v_endpoint,
            v_status_code,
            extract(milliseconds from clock_timestamp() - v_started)::int
        );
        return v_response;
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
            jsonb_build_array()
    end;

    -- Check if this machine is already activated
    select * into v_existing_activation
    from public.license_activations
    where license_key_id = v_license.id
      and machine_id_hash = p_machine_id_hash
      and is_active = true;

    if found then
        update public.license_activations
        set last_seen_at = now(),
            hostname = coalesce(p_hostname, hostname),
            os_description = coalesce(p_os_description, os_description),
            last_seen_package = p_package
        where id = v_existing_activation.id;

        v_status_code := 200;
        v_response := jsonb_build_object(
            'valid', true,
            'tier', v_license.tier::text,
            'capabilities', v_capabilities,
            'license_id', v_license.id,
            'activation_id', v_existing_activation.id,
            'message', 'License validated (existing activation).'
        );
        -- model_id intentionally NULL: validate_license_key isn't a
        -- model invocation, and the /account/usage/ "Top Models Used"
        -- panel shouldn't list tier names as if they were ML models.
        -- Tier is already returned in the response and persisted on
        -- the license_keys row; api_usage only needs endpoint + result.
        insert into public.api_usage (user_id, endpoint, status_code, latency_ms)
        values (
            v_license.user_id,
            v_endpoint,
            v_status_code,
            extract(milliseconds from clock_timestamp() - v_started)::int
        );
        return v_response;
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
        v_status_code := 429;
        v_response := jsonb_build_object(
            'valid', false,
            'error', 'activation_limit',
            'message', 'Maximum activations (' || v_license.max_activations || ') reached. Deactivate a machine first.',
            'current_activations', v_activation_count,
            'max_activations', v_license.max_activations
        );
        insert into public.api_usage (user_id, endpoint, status_code, latency_ms)
        values (
            v_license.user_id,
            v_endpoint,
            v_status_code,
            extract(milliseconds from clock_timestamp() - v_started)::int
        );
        return v_response;
    end if;

    -- Create new activation
    insert into public.license_activations (license_key_id, machine_id_hash, hostname, os_description, last_seen_package)
    values (v_license.id, p_machine_id_hash, p_hostname, p_os_description, p_package);

    v_status_code := 200;
    v_response := jsonb_build_object(
        'valid', true,
        'tier', v_license.tier::text,
        'capabilities', v_capabilities,
        'license_id', v_license.id,
        'message', 'License validated and activated on this machine.'
    );
    -- See "model_id intentionally NULL" comment on the existing-activation
    -- branch above. Validate-license is not a model call; tier lives on
    -- license_keys + in the response payload, not in api_usage.model_id.
    insert into public.api_usage (user_id, endpoint, status_code, latency_ms)
    values (
        v_license.user_id,
        v_endpoint,
        v_status_code,
        extract(milliseconds from clock_timestamp() - v_started)::int
    );
    return v_response;
end;
$$;

revoke execute on function public.validate_license_key(text, text, text, text, text) from anon;
grant execute on function public.validate_license_key(text, text, text, text, text) to service_role;

comment on function public.validate_license_key(text, text, text, text, text) is
    'Validates a license key and registers/updates a machine activation. Returns JSON with valid, tier, capabilities, error fields. Logs every call to api_usage so the /account/usage/ dashboard reflects SDK activity.';
