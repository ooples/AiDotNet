-- Initial schema for AiDotNet website
-- Tables: profiles, user_api_keys, api_usage
-- Functions: is_admin(), handle_new_user()
-- RLS policies for all tables

-- ============================================================================
-- 1. PROFILES TABLE
-- ============================================================================

create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  full_name text,
  role text not null default 'user',
  subscription_tier text not null default 'free',
  subscription_status text not null default 'active',
  stripe_customer_id text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

comment on table public.profiles is 'User profiles extending Supabase auth.users';

-- ============================================================================
-- 2. USER API KEYS TABLE
-- ============================================================================

create table if not exists public.user_api_keys (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  key_name text not null,
  key_prefix text not null,
  key_hash text not null,
  scopes text[] not null default '{}',
  is_active boolean not null default true,
  last_used_at timestamptz,
  created_at timestamptz not null default now()
);

comment on table public.user_api_keys is 'API keys for programmatic access';

create index if not exists idx_user_api_keys_user_id on public.user_api_keys(user_id);
create index if not exists idx_user_api_keys_active on public.user_api_keys(user_id, is_active);

-- ============================================================================
-- 3. API USAGE TABLE
-- ============================================================================

create table if not exists public.api_usage (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  endpoint text,
  model_id text,
  status_code integer,
  latency_ms integer,
  created_at timestamptz not null default now()
);

comment on table public.api_usage is 'API usage logs for analytics and billing';

create index if not exists idx_api_usage_user_id on public.api_usage(user_id);
create index if not exists idx_api_usage_created_at on public.api_usage(created_at);
create index if not exists idx_api_usage_user_created on public.api_usage(user_id, created_at);

-- ============================================================================
-- 4. HELPER FUNCTIONS
-- ============================================================================

-- is_admin(): SECURITY DEFINER to avoid infinite RLS recursion when checking
-- the caller's role inside RLS policies on the profiles table itself.
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

comment on function public.is_admin() is 'Check if the current user has admin role (SECURITY DEFINER to avoid RLS recursion)';

-- handle_new_user(): Trigger function to auto-create a profile row when a new
-- user signs up via Supabase Auth.
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = 'public'
as $$
begin
  insert into public.profiles (id, full_name)
  values (
    new.id,
    coalesce(new.raw_user_meta_data ->> 'full_name', new.raw_user_meta_data ->> 'name', '')
  );
  return new;
end;
$$;

-- Trigger on auth.users insert
drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row
  execute function public.handle_new_user();

-- ============================================================================
-- 5. ROW LEVEL SECURITY
-- ============================================================================

-- Enable RLS on all tables
alter table public.profiles enable row level security;
alter table public.user_api_keys enable row level security;
alter table public.api_usage enable row level security;

-- ---- PROFILES ----

drop policy if exists "Users can read own profile" on public.profiles;
create policy "Users can read own profile"
  on public.profiles for select
  using (auth.uid() = id);

drop policy if exists "Users can update own profile" on public.profiles;
create policy "Users can update own profile"
  on public.profiles for update
  using (auth.uid() = id)
  with check (
    auth.uid() = id
    and role = (select p.role from public.profiles p where p.id = auth.uid())
    and subscription_tier = (select p.subscription_tier from public.profiles p where p.id = auth.uid())
    and subscription_status = (select p.subscription_status from public.profiles p where p.id = auth.uid())
  );

drop policy if exists "Admins can read all profiles" on public.profiles;
create policy "Admins can read all profiles"
  on public.profiles for select
  using (public.is_admin());

drop policy if exists "Admins can update all profiles" on public.profiles;
create policy "Admins can update all profiles"
  on public.profiles for update
  using (public.is_admin());

-- ---- USER API KEYS ----

drop policy if exists "Users can read own keys" on public.user_api_keys;
create policy "Users can read own keys"
  on public.user_api_keys for select
  using (auth.uid() = user_id);

drop policy if exists "Users can create own keys" on public.user_api_keys;
create policy "Users can create own keys"
  on public.user_api_keys for insert
  with check (auth.uid() = user_id);

drop policy if exists "Users can update own keys" on public.user_api_keys;
create policy "Users can update own keys"
  on public.user_api_keys for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

drop policy if exists "Admins can read all keys" on public.user_api_keys;
create policy "Admins can read all keys"
  on public.user_api_keys for select
  using (public.is_admin());

drop policy if exists "Admins can update all keys" on public.user_api_keys;
create policy "Admins can update all keys"
  on public.user_api_keys for update
  using (public.is_admin());

-- ---- API USAGE ----

drop policy if exists "Users can read own usage" on public.api_usage;
create policy "Users can read own usage"
  on public.api_usage for select
  using (auth.uid() = user_id);

drop policy if exists "Admins can read all usage" on public.api_usage;
create policy "Admins can read all usage"
  on public.api_usage for select
  using (public.is_admin());
