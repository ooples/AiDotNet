-- ============================================================================
-- 1. Add email column to profiles for admin user lookup
-- ============================================================================

alter table public.profiles add column if not exists email text;

create index if not exists idx_profiles_email on public.profiles (email);

-- Backfill existing profiles with email from auth.users
update public.profiles p
set email = u.email
from auth.users u
where p.id = u.id
  and p.email is null;

-- Update trigger to include email on new signups
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = 'pg_catalog, public'
as $$
begin
  insert into public.profiles (id, full_name, email)
  values (
    new.id,
    coalesce(new.raw_user_meta_data ->> 'full_name', new.raw_user_meta_data ->> 'name', ''),
    new.email
  );
  return new;
end;
$$;

-- Sync email from auth.users to profiles when user updates their email
create or replace function public.handle_user_email_update()
returns trigger
language plpgsql
security definer
set search_path = 'pg_catalog, public'
as $$
begin
  if new.email is distinct from old.email then
    update public.profiles
    set email = new.email
    where id = new.id;
  end if;
  return new;
end;
$$;

-- Trigger: fire on auth.users email changes
drop trigger if exists on_auth_user_email_update on auth.users;
create trigger on_auth_user_email_update
  after update of email on auth.users
  for each row execute function public.handle_user_email_update();

-- Close the migration window for emails changed before the trigger existed.
update public.profiles p
set email = u.email
from auth.users u
where p.id = u.id
  and p.email is distinct from u.email;

-- ============================================================================
-- 2. Fix RLS role escalation vulnerability
--    The existing WITH CHECK uses a subquery on profiles which runs under RLS,
--    allowing users to change their own role/tier. Fix by using a SECURITY
--    DEFINER function that bypasses RLS to read the current row.
-- ============================================================================

-- Helper function to get the current user's protected fields (bypasses RLS)
-- Bound to auth.uid() internally to prevent callers from reading other users' data
create or replace function public.get_profile_protected_fields()
returns table(role text, subscription_tier text, subscription_status text, stripe_customer_id text, email text)
language sql
security definer
stable
set search_path = 'pg_catalog, public'
as $$
  select p.role, p.subscription_tier, p.subscription_status, p.stripe_customer_id, p.email
  from public.profiles p
  where p.id = auth.uid()
  limit 1;
$$;

-- Replace the broken user update policy
-- Email is treated as a protected field (synced from auth.users, not client-editable)
drop policy if exists "Users can update own profile" on public.profiles;
create policy "Users can update own profile"
  on public.profiles for update
  using (auth.uid() = id)
  with check (
    auth.uid() = id
    and email is not distinct from (select pf.email from public.get_profile_protected_fields() pf)
    and role = (select pf.role from public.get_profile_protected_fields() pf)
    and subscription_tier = (select pf.subscription_tier from public.get_profile_protected_fields() pf)
    and subscription_status = (select pf.subscription_status from public.get_profile_protected_fields() pf)
    and stripe_customer_id is not distinct from (select pf.stripe_customer_id from public.get_profile_protected_fields() pf)
  );

-- Also ensure admins have a proper update policy
drop policy if exists "Admins can update all profiles" on public.profiles;
create policy "Admins can update all profiles"
  on public.profiles for update
  using (public.is_admin());
