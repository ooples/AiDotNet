-- Paid subscriptions own separate entitlements even when an existing manual
-- grant or another subscription has the same tier. Retain every existing key.
drop index if exists public.idx_license_keys_one_active_per_user_product_tier;

-- One-time Checkout receipts also own distinct grants. Backfill known receipts
-- without changing or revoking any legacy key.
alter table public.license_keys add column if not exists stripe_checkout_session_id text;
update public.license_keys
  set stripe_checkout_session_id = substring(notes from '^Issued by Stripe Checkout \(session ([^)]+)\)$')
  where stripe_checkout_session_id is null and notes ~ '^Issued by Stripe Checkout \(session ([^)]+)\)$';
create unique index if not exists idx_license_keys_checkout_product
  on public.license_keys(stripe_checkout_session_id,product)
  where stripe_checkout_session_id is not null;

-- Derive the profile mirror from all active grants atomically. Locking the
-- profile makes overlapping cancellation/renewal callbacks converge without
-- removing access supplied by another subscription or a manual grant.
create or replace function public.sync_aidotnet_billing_profile(
  p_customer_id text, p_fallback_tier text, p_fallback_status text, p_user_id uuid default null
) returns void language plpgsql security definer
set search_path = pg_catalog, public
as $$
declare owner_id uuid; entitlement_rank integer; fallback_rank integer;
begin
  if p_fallback_tier not in ('free','pro','enterprise')
     or p_fallback_status not in ('active','suspended','expired') then
    raise exception 'Invalid billing mirror state' using errcode = '22023';
  end if;
  fallback_rank := case when p_fallback_status = 'active' then
    case p_fallback_tier when 'enterprise' then 2 when 'pro' then 1 else 0 end else -1 end;
  for owner_id in select id from public.profiles
    where (p_user_id is not null and id = p_user_id) or (p_user_id is null and stripe_customer_id = p_customer_id)
    order by id for update loop
    select greatest(coalesce(max(case tier::text when 'enterprise' then 2 when 'professional' then 1 else 0 end),-1),fallback_rank)
      into entitlement_rank from public.license_keys
      where user_id = owner_id and product = 'aidotnet' and status = 'active';
    update public.profiles set
      subscription_tier = case entitlement_rank when 2 then 'enterprise' when 1 then 'pro' else 'free' end,
      subscription_status = case when entitlement_rank >= 0 then 'active' else p_fallback_status end
      where id = owner_id;
  end loop;
end $$;
revoke all on function public.sync_aidotnet_billing_profile(text,text,text,uuid) from public, anon, authenticated;
grant execute on function public.sync_aidotnet_billing_profile(text,text,text,uuid) to service_role;
-- Preserve atomic Community self-service issuance while allowing independent grants.
create or replace function public.issue_aidotnet_community_license(p_user_id uuid, p_license_key text)
returns jsonb language plpgsql security definer set search_path = pg_catalog, public as $$
declare existing public.license_keys%rowtype;
begin
  if p_user_id is null or p_license_key !~ '^AIDN-PROD-COMMUNITY-[0-9a-f]{32}$' then
    raise exception 'Invalid Community issuance arguments';
  end if;
  perform pg_advisory_xact_lock(hashtextextended('aidotnet-community:' || p_user_id::text, 0));
  select * into existing from public.license_keys
    where user_id = p_user_id and product = 'aidotnet' and tier = 'community' and status = 'active'
    order by id limit 1;
  if found then
    return jsonb_build_object('id', existing.id, 'license_key', existing.license_key, 'is_existing', true);
  end if;
  insert into public.license_keys(user_id, license_key, product, tier, status, max_activations, notes)
    values(p_user_id, p_license_key, 'aidotnet', 'community', 'active', 3, 'Self-registered community license')
    returning * into existing;
  return jsonb_build_object('id', existing.id, 'license_key', existing.license_key, 'is_existing', false);
end $$;
revoke all on function public.issue_aidotnet_community_license(uuid,text) from public, anon, authenticated;
grant execute on function public.issue_aidotnet_community_license(uuid,text) to service_role;
