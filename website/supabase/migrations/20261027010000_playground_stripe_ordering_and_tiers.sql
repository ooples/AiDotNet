-- Correct the deployed RPCs without replacing tables, receipts, or grants.
create or replace function public.claim_playground_stripe_event(
    p_profile_id uuid, p_event_id text, p_event_created bigint, p_token uuid
) returns text language plpgsql security definer set search_path = '' as $$
declare
    s public.playground_stripe_state%rowtype;
begin
    if p_profile_id is null or p_token is null or coalesce(p_event_id, '') = ''
        or p_event_created is null or p_event_created <= 0 then
        raise exception 'Invalid Stripe event identity';
    end if;
    insert into public.playground_stripe_state(profile_id) values (p_profile_id)
        on conflict do nothing;
    select * into strict s from public.playground_stripe_state
        where profile_id = p_profile_id for update;
    if exists (select 1 from public.playground_stripe_events where event_id = p_event_id) then
        return 'duplicate';
    end if;
    if s.lease_token is not null and s.lease_expires_at > clock_timestamp() then
        return 'busy';
    end if;
    -- Event time is not subscription order: a replacement checkout can precede
    -- the old subscription's cancellation. Reconcile current Stripe state under
    -- this lease; completion rejects older subscriptions by their creation time.
    update public.playground_stripe_state
        set lease_token = p_token, lease_event_id = p_event_id,
            lease_event_created = p_event_created,
            lease_expires_at = clock_timestamp() + interval '2 minutes'
        where profile_id = p_profile_id;
    return 'claimed';
end;
$$;

create or replace function public.complete_playground_stripe_event(
    p_profile_id uuid, p_token uuid, p_customer_id text,
    p_subscription_id text, p_subscription_created bigint, p_status text
) returns text language plpgsql security definer set search_path = '' as $$
declare
    s public.playground_stripe_state%rowtype;
    result text := 'applied';
    profile_status text;
begin
    if coalesce(p_customer_id, '') = '' or coalesce(p_subscription_id, '') = ''
        or p_subscription_created is null or p_subscription_created <= 0
        or p_status is null or p_status not in
            ('active', 'trialing', 'past_due', 'canceled', 'unpaid', 'incomplete', 'incomplete_expired', 'paused') then
        raise exception 'Invalid Stripe subscription state';
    end if;
    select * into strict s from public.playground_stripe_state
        where profile_id = p_profile_id for update;
    if s.lease_token is distinct from p_token or p_token is null
        or s.lease_expires_at <= clock_timestamp() then
        raise exception 'Stripe event lease expired or replaced';
    end if;
    -- An old subscription's delayed cancellation must not cancel a replacement.
    -- A terminal event from a different subscription created in the same second
    -- likewise cannot revoke the currently selected subscription.
    if p_subscription_created < s.subscription_created
        or (p_subscription_created = s.subscription_created
            and s.subscription_id is distinct from p_subscription_id
            and p_status in ('canceled', 'incomplete_expired')) then
        result := 'stale';
    else
        profile_status := case when p_status = 'trialing' then 'active' else p_status end;
        update public.profiles
            set subscription_tier = case when p_status in ('active', 'trialing') then 'pro' else 'free' end,
                subscription_status = profile_status, stripe_customer_id = p_customer_id,
                updated_at = clock_timestamp()
            where id = p_profile_id;
        if not found then
            raise exception 'Stripe profile is missing';
        end if;
        update public.playground_stripe_state
            set last_event_created = greatest(last_event_created, s.lease_event_created),
                subscription_id = p_subscription_id, subscription_created = p_subscription_created
            where profile_id = p_profile_id;
    end if;
    insert into public.playground_stripe_events(event_id, profile_id, event_created, outcome)
        values (s.lease_event_id, p_profile_id, s.lease_event_created, result);
    update public.playground_stripe_state
        set lease_token = null, lease_event_id = null, lease_event_created = null, lease_expires_at = null
        where profile_id = p_profile_id;
    return result;
end;
$$;

