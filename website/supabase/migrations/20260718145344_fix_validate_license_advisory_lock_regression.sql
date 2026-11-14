-- REGRESSION FIX (found by end-to-end testing on 2026-07-18): 20260610000000_log_validations_to_api_usage
-- rebuilt validate_license_key from the pre-fix (20260427) version and reintroduced
-- `pg_advisory_xact_lock(v_license.id)`. pg_advisory_xact_lock takes bigint, not uuid, so EVERY first-time
-- activation on a NEW machine has thrown `function pg_advisory_xact_lock(uuid) does not exist` and returned
-- 500 (the validate-license edge fn surfaces it as `server_error`) since 2026-06-10. Existing activations
-- short-circuit BEFORE the lock, which hid it — but every new CI runner / new customer machine hit it, a
-- direct contributor to the "invalid license in CI" symptom.
--
-- This restores the 20260504 fix (hashtextextended(uuid::text) -> deterministic bigint) at BOTH lock sites
-- while preserving the api_usage logging from 20260610. Idempotent (`create or replace`); a fresh env that
-- already has the corrected 20260610 applies this as a no-op.

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
begin
    select * into v_license from public.license_keys where license_key = p_license_key;
    if not found then
        return jsonb_build_object('valid', false, 'error', 'invalid_key', 'message', 'License key not found.');
    end if;

    if v_license.status != 'active' then
        v_status_code := 403;
        v_response := jsonb_build_object('valid', false, 'error', 'license_' || v_license.status::text,
            'message', 'License is ' || v_license.status::text || '.');
        insert into public.api_usage (user_id, endpoint, status_code, latency_ms)
        values (v_license.user_id, v_endpoint, v_status_code, extract(milliseconds from clock_timestamp() - v_started)::int);
        return v_response;
    end if;

    if v_license.expires_at is not null and v_license.expires_at < now() then
        v_status_code := 403;
        v_response := jsonb_build_object('valid', false, 'error', 'license_expired', 'message', 'License has expired.');
        insert into public.api_usage (user_id, endpoint, status_code, latency_ms)
        values (v_license.user_id, v_endpoint, v_status_code, extract(milliseconds from clock_timestamp() - v_started)::int);
        return v_response;
    end if;

    v_capabilities := case v_license.tier
        when 'community' then jsonb_build_array('tensors:load')
        when 'professional' then jsonb_build_array('tensors:save', 'tensors:load', 'model:save', 'model:load')
        when 'enterprise' then jsonb_build_array('tensors:save', 'tensors:load', 'model:save', 'model:load')
        else jsonb_build_array()
    end;

    select * into v_existing_activation from public.license_activations
    where license_key_id = v_license.id and machine_id_hash = p_machine_id_hash and is_active = true;

    if found then
        update public.license_activations
        set last_seen_at = now(), hostname = coalesce(p_hostname, hostname),
            os_description = coalesce(p_os_description, os_description),
            last_seen_package = coalesce(p_package, last_seen_package)
        where id = v_existing_activation.id;
        v_status_code := 200;
        v_response := jsonb_build_object('valid', true, 'tier', v_license.tier::text, 'capabilities', v_capabilities,
            'license_id', v_license.id, 'activation_id', v_existing_activation.id, 'message', 'License validated (existing activation).');
        insert into public.api_usage (user_id, endpoint, status_code, latency_ms)
        values (v_license.user_id, v_endpoint, v_status_code, extract(milliseconds from clock_timestamp() - v_started)::int);
        return v_response;
    end if;

    -- FIX: bigint advisory lock, not the uuid overload that doesn't exist.
    perform pg_advisory_xact_lock(hashtextextended(v_license.id::text, 0));

    select * into v_existing_activation from public.license_activations
    where license_key_id = v_license.id and machine_id_hash = p_machine_id_hash and is_active = true;
    if found then
        update public.license_activations
        set last_seen_at = now(), hostname = coalesce(p_hostname, hostname),
            os_description = coalesce(p_os_description, os_description),
            last_seen_package = coalesce(p_package, last_seen_package)
        where id = v_existing_activation.id;
        v_status_code := 200;
        v_response := jsonb_build_object('valid', true, 'tier', v_license.tier::text, 'capabilities', v_capabilities,
            'license_id', v_license.id, 'activation_id', v_existing_activation.id, 'message', 'License validated (existing activation, raced into lock).');
        insert into public.api_usage (user_id, endpoint, status_code, latency_ms)
        values (v_license.user_id, v_endpoint, v_status_code, extract(milliseconds from clock_timestamp() - v_started)::int);
        return v_response;
    end if;

    select count(*) into v_activation_count from public.license_activations
    where license_key_id = v_license.id and is_active = true;
    if v_activation_count >= v_license.max_activations then
        v_status_code := 429;
        v_response := jsonb_build_object('valid', false, 'error', 'activation_limit',
            'message', 'Maximum activations (' || v_license.max_activations || ') reached. Deactivate a machine first.',
            'current_activations', v_activation_count, 'max_activations', v_license.max_activations);
        insert into public.api_usage (user_id, endpoint, status_code, latency_ms)
        values (v_license.user_id, v_endpoint, v_status_code, extract(milliseconds from clock_timestamp() - v_started)::int);
        return v_response;
    end if;

    insert into public.license_activations (license_key_id, machine_id_hash, hostname, os_description, last_seen_package)
    values (v_license.id, p_machine_id_hash, p_hostname, p_os_description, p_package);
    v_status_code := 200;
    v_response := jsonb_build_object('valid', true, 'tier', v_license.tier::text, 'capabilities', v_capabilities,
        'license_id', v_license.id, 'message', 'License validated and activated on this machine.');
    insert into public.api_usage (user_id, endpoint, status_code, latency_ms)
    values (v_license.user_id, v_endpoint, v_status_code, extract(milliseconds from clock_timestamp() - v_started)::int);
    return v_response;
end;
$$;

revoke execute on function public.validate_license_key(text, text, text, text, text) from anon;
grant execute on function public.validate_license_key(text, text, text, text, text) to service_role;
