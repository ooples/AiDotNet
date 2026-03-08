-- Tighten RLS policies to prevent privilege escalation
-- 1. Restrict profile self-updates: users cannot change role/subscription fields
-- 2. Add WITH CHECK to user_api_keys update to prevent user_id reassignment
-- 3. Fix is_admin() search_path to include pg_catalog

-- ============================================================================
-- 1. Fix is_admin() search_path
-- ============================================================================

create or replace function public.is_admin()
returns boolean
language sql
security definer
set search_path = 'pg_catalog, public'
stable
as $$
  select exists (
    select 1
    from public.profiles
    where id = auth.uid()
      and role = 'admin'
  );
$$;

-- ============================================================================
-- 2. Restrict profile self-updates to non-privileged fields only
-- ============================================================================

-- Replace the permissive update policy with one that prevents role escalation.
-- Users can only update full_name and updated_at on their own profile.
-- Role, subscription_tier, subscription_status, and stripe_customer_id changes require admin.
drop policy if exists "Users can update own profile" on public.profiles;
create policy "Users can update own profile"
  on public.profiles for update
  using (auth.uid() = id)
  with check (
    auth.uid() = id
    and role = (select p.role from public.profiles p where p.id = auth.uid())
    and subscription_tier = (select p.subscription_tier from public.profiles p where p.id = auth.uid())
    and subscription_status = (select p.subscription_status from public.profiles p where p.id = auth.uid())
    and stripe_customer_id is not distinct from (select p.stripe_customer_id from public.profiles p where p.id = auth.uid())
  );

-- ============================================================================
-- 3. Restrict user_api_keys updates to prevent user_id reassignment
-- ============================================================================

-- Users can only revoke their own keys (set is_active = false).
-- They cannot change user_id, key_hash, key_prefix, or scopes.
drop policy if exists "Users can update own keys" on public.user_api_keys;
create policy "Users can update own keys"
  on public.user_api_keys for update
  using (auth.uid() = user_id)
  with check (
    auth.uid() = user_id
    and key_hash = (select k.key_hash from public.user_api_keys k where k.id = id)
    and key_prefix = (select k.key_prefix from public.user_api_keys k where k.id = id)
    and scopes = (select k.scopes from public.user_api_keys k where k.id = id)
    and user_id = (select k.user_id from public.user_api_keys k where k.id = id)
  );
