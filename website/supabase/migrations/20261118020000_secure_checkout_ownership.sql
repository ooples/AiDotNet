-- Verified Checkout ownership is separate from the profile's preferred billing link.
create table public.billing_customers (
  customer_id text primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  created_at timestamptz not null default now()
);
create index billing_customers_user_id_idx on public.billing_customers(user_id);
create table public.billing_checkout_intents (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  tier text not null check (tier in ('pro','enterprise')),
  interval text not null check (interval in ('month','year')),
  created_at timestamptz not null default now(),
  expires_at timestamptz not null default now() + interval '1 day'
);
create index billing_checkout_intents_user_id_idx on public.billing_checkout_intents(user_id);
create table public.billing_checkout_bindings (
  session_id text primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  customer_id text references public.billing_customers(customer_id),
  created_at timestamptz not null default now()
);
create index billing_checkout_bindings_user_id_idx on public.billing_checkout_bindings(user_id);
create index billing_checkout_bindings_customer_id_idx on public.billing_checkout_bindings(customer_id);
alter table public.billing_customers enable row level security;
alter table public.billing_checkout_intents enable row level security;
alter table public.billing_checkout_bindings enable row level security;
revoke all on public.billing_customers, public.billing_checkout_intents, public.billing_checkout_bindings from public, anon, authenticated;
grant all on public.billing_customers, public.billing_checkout_intents, public.billing_checkout_bindings to service_role;
create policy service_billing_customers on public.billing_customers for all to service_role using (true) with check (true);
create policy service_checkout_intents on public.billing_checkout_intents for all to service_role using (true) with check (true);
create policy service_checkout_bindings on public.billing_checkout_bindings for all to service_role using (true) with check (true);

-- Preserve consistent legacy ownership without choosing an owner for conflicting data.
insert into public.billing_customers(customer_id,user_id)
select customer_id,(array_agg(distinct user_id))[1] from (
  select stripe_customer_id customer_id,id user_id from public.profiles where stripe_customer_id is not null
  union all
  select stripe_customer_id,user_id from public.license_keys where stripe_customer_id is not null
) legacy group by customer_id having count(distinct user_id)=1;

create function public.bind_aidotnet_checkout(p_user_id uuid,p_session_id text,p_customer_id text)
returns void language plpgsql security definer set search_path=pg_catalog,public as $$
declare owner_id uuid; bound_customer text;
begin
  if p_user_id is null or p_session_id is null or p_session_id='' then raise exception 'Invalid Checkout binding'; end if;
  if p_customer_id is not null then
    insert into public.billing_customers(customer_id,user_id) values(p_customer_id,p_user_id) on conflict do nothing;
    select user_id into owner_id from public.billing_customers where customer_id=p_customer_id for update;
    if owner_id is distinct from p_user_id then raise exception 'Billing customer already belongs to another account'; end if;
  end if;
  insert into public.billing_checkout_bindings(session_id,user_id,customer_id) values(p_session_id,p_user_id,p_customer_id) on conflict do nothing;
  select user_id,customer_id into owner_id,bound_customer from public.billing_checkout_bindings where session_id=p_session_id for update;
  if owner_id is distinct from p_user_id or bound_customer is distinct from p_customer_id then raise exception 'Checkout already belongs to another account'; end if;
end $$;
revoke all on function public.bind_aidotnet_checkout(uuid,text,text) from public,anon,authenticated;
grant execute on function public.bind_aidotnet_checkout(uuid,text,text) to service_role;

-- Profile repair uses current Auth data, not a potentially stale JWT email.
create function public.current_auth_email()
returns text language sql stable security definer set search_path=pg_catalog,auth as $$
  select email::text from auth.users where id=auth.uid()
$$;
revoke all on function public.current_auth_email() from public,anon;
grant execute on function public.current_auth_email() to authenticated,service_role;

create or replace function public.guard_profile_protected_fields()
returns trigger language plpgsql security invoker
set search_path = pg_catalog, public
as $$
begin
  if current_user in ('postgres', 'supabase_admin', 'supabase_auth_admin', 'service_role')
     or auth.role() = 'service_role' or public.is_admin() then
    return new;
  end if;
  if tg_op = 'INSERT' then
    if new.id is distinct from auth.uid() or new.role is distinct from 'user'
       or new.subscription_tier is distinct from 'free'
       or new.subscription_status is distinct from 'active'
       or new.stripe_customer_id is not null then
      raise exception 'New customer profiles must use their own free account' using errcode = '42501';
    end if;
    -- Profile repair cannot forge an email used for Checkout reconciliation or an audit date.
    new.email := public.current_auth_email();
    new.created_at := now();
    return new;
  end if;
  if row(new.id, new.role, new.subscription_tier, new.subscription_status,
         new.stripe_customer_id, new.email, new.created_at)
     is distinct from
     row(old.id, old.role, old.subscription_tier, old.subscription_status,
         old.stripe_customer_id, old.email, old.created_at) then
    raise exception 'Profile role, billing, identity, and audit fields require an administrator'
      using errcode = '42501';
  end if;
  return new;
end $$;
revoke all on function public.guard_profile_protected_fields() from public, anon, authenticated;
drop trigger if exists guard_profile_protected_fields on public.profiles;
create trigger guard_profile_protected_fields before insert or update on public.profiles
for each row execute function public.guard_profile_protected_fields();


create or replace function public.guard_api_key_protected_fields()
returns trigger language plpgsql security invoker
set search_path = pg_catalog, public
as $$
begin
  if current_user in ('postgres', 'supabase_admin', 'service_role')
     or auth.role() = 'service_role' or public.is_admin() then
    return new;
  end if;
  if tg_op = 'INSERT' then
    new.created_at := now();
    new.last_used_at := null;
    return new;
  end if;
  if not old.is_active and new.is_active then
    raise exception 'Revoked API keys cannot be reactivated by their owner' using errcode = '42501';
  end if;
  if row(new.id, new.user_id, new.key_hash, new.key_prefix, new.scopes, new.created_at, new.last_used_at)
     is distinct from
     row(old.id, old.user_id, old.key_hash, old.key_prefix, old.scopes, old.created_at, old.last_used_at) then
    raise exception 'API key identity, credentials, scopes, and audit fields cannot be changed'
      using errcode = '42501';
  end if;
  return new;
end $$;
revoke all on function public.guard_api_key_protected_fields() from public, anon, authenticated;
drop trigger if exists guard_api_key_protected_fields on public.user_api_keys;
create trigger guard_api_key_protected_fields before insert or update on public.user_api_keys
for each row execute function public.guard_api_key_protected_fields();


-- Active grants determine access; Stripe fallback reports billing status only.
create or replace function public.sync_aidotnet_billing_profile(
  p_customer_id text, p_fallback_tier text, p_fallback_status text, p_user_id uuid default null
) returns void language plpgsql security definer
set search_path = pg_catalog, public
as $$
declare owner_id uuid; entitlement_rank integer;
begin
  if p_fallback_tier is null or p_fallback_status is null or p_fallback_tier not in ('free','pro','enterprise')
     or p_fallback_status not in ('active','suspended','expired') then
    raise exception 'Invalid billing mirror state' using errcode = '22023';
  end if;
  for owner_id in select id from public.profiles
    where (p_user_id is not null and id = p_user_id) or (p_user_id is null and (stripe_customer_id = p_customer_id or id in (select user_id from public.billing_customers where customer_id=p_customer_id)))
    order by id for update loop
    select coalesce(max(case tier::text when 'enterprise' then 2 when 'professional' then 1 else 0 end),-1)
      into entitlement_rank from public.license_keys
      where user_id = owner_id and product = 'aidotnet' and status = 'active' and (expires_at is null or expires_at > now());
    update public.profiles set
      stripe_customer_id = coalesce(stripe_customer_id, (select customer_id from public.billing_customers where user_id=owner_id and customer_id=p_customer_id)),
      subscription_tier = case entitlement_rank when 2 then 'enterprise' when 1 then 'pro' else 'free' end,
      subscription_status = case when entitlement_rank >= 0 then 'active' else p_fallback_status end
      where id = owner_id;
  end loop;
end $$;
revoke all on function public.sync_aidotnet_billing_profile(text,text,text,uuid) from public, anon, authenticated;
grant execute on function public.sync_aidotnet_billing_profile(text,text,text,uuid) to service_role;
