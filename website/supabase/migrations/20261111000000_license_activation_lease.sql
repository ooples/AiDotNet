-- Seats are LEASES (2026-11-11). validate_license_key counted every is_active activation ever seen and never
-- read last_seen_at, so a seat, once taken, was held forever. Every ephemeral CI runner has a fresh machine id, so
-- a CI key ran out of seats within days and builds failed with "maximum number of developer seats" (Ooples CI:
-- 19 LicenseRequiredException failures in one packaged-source run, and an intermittent model drop from search).
--
-- Now an activation holds a seat only while its machine has validated within license_keys.activation_lease
-- (default 7 days). Every validation renews it. A machine unseen for longer stops counting; if it returns, it
-- re-activates when a seat is free and is refused otherwise. This is the heartbeat/lease model of Keygen machine
-- heartbeats and FlexLM/JetBrains floating leases. AiDotNet.Serving's LicenseService enforces the same rule
-- (LicenseService.ActivationLease).
--
-- Idempotent: the column is added if missing; the function is replaced.

alter table public.license_keys
    add column if not exists activation_lease interval not null default interval '7 days';

comment on column public.license_keys.activation_lease is
    'How long an activation holds a seat after its machine last validated. Lapsed activations do not count toward max_activations.';

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
    v_now timestamptz;
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

    -- Every lease decision for this key is serialized, and made on a time read AFTER the lock. Renewing before
    -- the lock let a machine pass the lease check just before expiry, stall, and renew after another machine had
    -- already taken the freed seat, leaving two leases on one seat. now() would not do even after the lock: it is
    -- the transaction's start time, so a request that waited here would judge leases by a clock from before it
    -- waited. One clock_timestamp() taken after the lock is used for every lease check and renewal below.
    -- bigint advisory lock, not the uuid overload that doesn't exist.
    perform pg_advisory_xact_lock(hashtextextended(v_license.id::text, 0));
    v_now := clock_timestamp();

    select * into v_existing_activation from public.license_activations
    where license_key_id = v_license.id and machine_id_hash = p_machine_id_hash and is_active = true
      and last_seen_at >= v_now - v_license.activation_lease;
    if found then
        update public.license_activations
        set last_seen_at = v_now, hostname = coalesce(p_hostname, hostname),
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

    -- Seats held: active activations whose lease is current. This machine's own row, if its lease lapsed, is
    -- not among them, so it re-activates exactly when a seat is free.
    select count(*) into v_activation_count from public.license_activations
    where license_key_id = v_license.id and is_active = true
      and last_seen_at >= v_now - v_license.activation_lease;
    if v_activation_count >= v_license.max_activations then
        v_status_code := 429;
        v_response := jsonb_build_object('valid', false, 'error', 'activation_limit',
            'message', 'Maximum activations (' || v_license.max_activations || ') reached. Deactivate a machine first.',
            'current_activations', v_activation_count, 'max_activations', v_license.max_activations);
        insert into public.api_usage (user_id, endpoint, status_code, latency_ms)
        values (v_license.user_id, v_endpoint, v_status_code, extract(milliseconds from clock_timestamp() - v_started)::int);
        return v_response;
    end if;

    -- A lapsed machine renews its own row (the partial unique index allows one active row per machine);
    -- a new one is inserted.
    update public.license_activations
    set last_seen_at = v_now, hostname = coalesce(p_hostname, hostname),
        os_description = coalesce(p_os_description, os_description),
        last_seen_package = coalesce(p_package, last_seen_package)
    where license_key_id = v_license.id and machine_id_hash = p_machine_id_hash and is_active = true;
    if not found then
        insert into public.license_activations (license_key_id, machine_id_hash, hostname, os_description, last_seen_package, last_seen_at)
        values (v_license.id, p_machine_id_hash, p_hostname, p_os_description, p_package, v_now);
    end if;
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
