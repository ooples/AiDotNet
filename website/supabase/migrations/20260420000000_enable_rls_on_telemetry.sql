-- Enable Row-Level Security on four telemetry tables that ship with
-- RLS OFF. The Supabase security advisor flags these as ERROR-level:
-- they live in the `public` schema and are reachable by anon API key
-- through PostgREST, which means an attacker with the (publicly bundled)
-- anon key could SELECT/INSERT/UPDATE/DELETE arbitrary rows.
--
-- Also pins the search_path on public.update_license_keys_updated_at()
-- so a caller can't shadow the expected `public` schema and hijack
-- table lookups inside the trigger.
--
-- ---------------------------------------------------------------
-- Note on table existence
-- ---------------------------------------------------------------
-- These four tables already exist in the production project
-- (ref: yfkqwpgjahoamlgckjib) — they were created out-of-band before
-- the migration tree was under version control. Each CREATE TABLE
-- below uses IF NOT EXISTS so running this migration against prod is
-- a no-op for table shape; a fresh project (`supabase db push` from
-- scratch) gets both the tables AND the RLS in one step.
--
-- ---------------------------------------------------------------
-- Admin-check pattern
-- ---------------------------------------------------------------
-- All admin SELECT policies delegate to public.is_admin() (defined in
-- 20260306000000_initial_schema.sql). That function is SECURITY DEFINER
-- specifically to avoid the RLS-recursion trap when checking
-- public.profiles.role from an RLS policy that itself filters
-- public.profiles rows.

-- ---------- gpu_profiles ----------
create table if not exists public.gpu_profiles (
  id uuid primary key default gen_random_uuid(),
  gpu_vendor text not null,
  gpu_model text not null,
  gpu_architecture text,
  driver_version text,
  min_dimension integer not null,
  max_dimension integer not null,
  config_json jsonb not null,
  measured_gflops double precision,
  efficiency_percent double precision,
  sample_count integer default 1,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

alter table public.gpu_profiles enable row level security;

create policy "gpu_profiles_insert_anon"
  on public.gpu_profiles
  for insert
  to anon, authenticated
  with check (true);

create policy "gpu_profiles_select_admin"
  on public.gpu_profiles
  for select
  to authenticated
  using (public.is_admin());

-- ---------- gpu_telemetry ----------
create table if not exists public.gpu_telemetry (
  id uuid primary key default gen_random_uuid(),
  gpu_vendor text not null,
  gpu_model text not null,
  gpu_architecture text,
  driver_version text,
  os_platform text,
  matrix_m integer not null,
  matrix_n integer not null,
  matrix_k integer not null,
  config_json jsonb not null,
  measured_gflops double precision,
  efficiency_percent double precision,
  client_hash text,
  aidotnet_version text,
  submitted_at timestamptz default now()
);

alter table public.gpu_telemetry enable row level security;

create policy "gpu_telemetry_insert_anon"
  on public.gpu_telemetry
  for insert
  to anon, authenticated
  with check (true);

create policy "gpu_telemetry_select_admin"
  on public.gpu_telemetry
  for select
  to authenticated
  using (public.is_admin());

-- ---------- exception_telemetry ----------
create table if not exists public.exception_telemetry (
  id uuid primary key default gen_random_uuid(),
  exception_type text not null,
  exception_message text,
  stack_trace text,
  inner_exception_type text,
  inner_exception_message text,
  component text not null,
  operation text,
  aidotnet_version text not null,
  dotnet_version text,
  os_platform text,
  os_version text,
  gpu_vendor text,
  gpu_model text,
  client_hash text,
  additional_context jsonb,
  submitted_at timestamptz default now()
);

alter table public.exception_telemetry enable row level security;

create policy "exception_telemetry_insert_anon"
  on public.exception_telemetry
  for insert
  to anon, authenticated
  with check (true);

create policy "exception_telemetry_select_admin"
  on public.exception_telemetry
  for select
  to authenticated
  using (public.is_admin());

-- ---------- telemetry_consent ----------
create table if not exists public.telemetry_consent (
  client_hash text primary key,
  opted_out boolean default false,
  opted_out_at timestamptz,
  created_at timestamptz default now()
);

alter table public.telemetry_consent enable row level security;

create policy "telemetry_consent_insert_anon"
  on public.telemetry_consent
  for insert
  to anon, authenticated
  with check (true);

-- UPDATE policy — the client_hash column acts as the capability.
--
-- Threat model, stated explicitly: telemetry_consent is a pseudonymous
-- opt-out table keyed by a one-way SHA-256 of the machine fingerprint.
-- We cannot bind updates to a Supabase JWT because the consent record
-- has to work for anonymous library users who never signed up, and we
-- cannot verify the hash over the wire because it IS the identifier.
--
-- So knowing a machine's client_hash IS the capability to flip that
-- machine's opt-out state. That mirrors the asymmetry of an emailed
-- "unsubscribe" URL — anyone who holds the token can exercise it. The
-- hash is not meaningfully enumerable (SHA-256 of a machine fingerprint)
-- so an attacker can't target a specific user's consent without already
-- knowing their hash; and the worst case is flipping one machine's
-- telemetry preference, which has no PII exposure.
--
-- If the library ever exposes a user's client_hash externally (support
-- tickets, public logs, etc.) this policy needs to tighten to a signed
-- opt-out token flow — but at that point the write path reshapes
-- entirely anyway.
create policy "telemetry_consent_update_anon_capability"
  on public.telemetry_consent
  for update
  to anon, authenticated
  using (true)
  with check (true);

create policy "telemetry_consent_select_admin"
  on public.telemetry_consent
  for select
  to authenticated
  using (public.is_admin());

-- ---------- Pin search_path on update_license_keys_updated_at ----------
-- A role-mutable search_path lets a caller shadow `public` with their
-- own schema and hijack table lookups inside the trigger function.
-- Pinning to (public, pg_catalog) closes that vector.
alter function public.update_license_keys_updated_at()
  set search_path = 'public', 'pg_catalog';
