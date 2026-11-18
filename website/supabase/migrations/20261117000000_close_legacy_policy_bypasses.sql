-- PostgreSQL permissive policies are ORed. Legacy owner-or-admin policies
-- defeated the newer protected-field checks, and one usage policy allowed
-- every role to insert records. Replace all overlapping write policies.
do $$
declare p record;
begin
  for p in select tablename, policyname from pg_policies
    where schemaname = 'public' and permissive = 'PERMISSIVE'
      and ((tablename in ('profiles', 'user_api_keys') and cmd = 'UPDATE')
        or (tablename = 'profiles' and cmd = 'INSERT')
        or (tablename = 'api_usage' and cmd = 'INSERT'))
  loop
    execute format('drop policy %I on public.%I', p.policyname, p.tablename);
  end loop;
end $$;

-- Triggers compare OLD and NEW directly, avoiding recursive RLS subqueries
-- and continuing to protect fields even if another permissive policy is added.
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
    new.email := auth.email();
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

create policy profiles_update_own on public.profiles for update to authenticated
using (auth.uid() = id) with check (auth.uid() = id);
create policy profiles_update_admin on public.profiles for update to authenticated
using (public.is_admin()) with check (public.is_admin());
create policy profiles_insert_own on public.profiles for insert to authenticated
with check (auth.uid() = id and role = 'user' and subscription_tier = 'free'
  and subscription_status = 'active' and stripe_customer_id is null);
create policy profiles_insert_admin on public.profiles for insert to authenticated
with check (public.is_admin());

create or replace function public.guard_api_key_protected_fields()
returns trigger language plpgsql security invoker
set search_path = pg_catalog, public
as $$
begin
  if current_user in ('postgres', 'supabase_admin', 'service_role')
     or auth.role() = 'service_role' or public.is_admin() then
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
create trigger guard_api_key_protected_fields before update on public.user_api_keys
for each row execute function public.guard_api_key_protected_fields();

create policy api_keys_update_own on public.user_api_keys for update to authenticated
using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy api_keys_update_admin on public.user_api_keys for update to authenticated
using (public.is_admin()) with check (public.is_admin());
create policy api_usage_insert_service on public.api_usage for insert to service_role
with check (true);
revoke insert on public.api_usage from public, anon, authenticated;
grant insert on public.api_usage to service_role;
