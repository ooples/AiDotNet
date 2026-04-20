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

-- Indexes for admin analytics. The admin panel slices GPU profiles by
-- vendor/model (device-specific charts) and filters to recent samples
-- ("GPUs seen in the last 30 days"). Without indexes both queries are
-- full-scans — cheap at current volume but painful once the library
-- ships telemetry to real users.
create index if not exists gpu_profiles_vendor_model_idx
  on public.gpu_profiles (gpu_vendor, gpu_model);
create index if not exists gpu_profiles_created_at_idx
  on public.gpu_profiles (created_at desc);

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

-- Retention policy for exception_telemetry.
--
-- Stack traces + messages + additional_context can inadvertently carry
-- PII, file paths, or secrets even though the client library scrubs
-- aggressively. To bound the exposure window we time-box retention to
-- 90 days and delete older rows on a schedule.
--
-- The function below is idempotent, non-transactional in the sense
-- that every call deletes only what's older than 90 days from now,
-- and safe to invoke repeatedly. Callers (pg_cron / the scheduler
-- Edge Function / an admin-triggered run) do not need to coordinate.
-- Returns the delete count so the caller can log it.
--
-- Scheduling: this migration does NOT enable pg_cron or schedule the
-- function — doing so requires dashboard-level settings on Supabase
-- (the pg_cron extension has to be whitelisted at project level).
-- Until scheduled, run manually from SQL Editor when needed:
--     select public.cleanup_old_exception_telemetry(90);
-- Follow-up task tracked outside this migration: enable pg_cron and
-- add `select cron.schedule('exception_telemetry_daily_cleanup',
--   '0 3 * * *', $$select public.cleanup_old_exception_telemetry(90)$$);`
create or replace function public.cleanup_old_exception_telemetry(retention_days integer default 90)
returns integer
language plpgsql
security definer
set search_path = 'public', 'pg_catalog'
as $$
declare
  deleted_count integer;
begin
  delete from public.exception_telemetry
   where submitted_at < (now() - make_interval(days => retention_days));
  get diagnostics deleted_count = row_count;
  return deleted_count;
end;
$$;

comment on function public.cleanup_old_exception_telemetry(integer) is
  'Deletes exception_telemetry rows older than `retention_days` (default 90). '
  'Intended to be invoked by pg_cron or an admin-triggered maintenance job.';

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

-- UPDATE policy — row-level targeting enforced via X-Client-Hash header.
--
-- Threat model: telemetry_consent is a pseudonymous opt-out table keyed
-- by a one-way SHA-256 of the machine fingerprint. We cannot bind
-- updates to a Supabase JWT because the consent record has to work for
-- anonymous library users who never signed up.
--
-- A previous revision of this policy used `using (true)` on the theory
-- that knowing a machine's client_hash was already the capability to
-- flip that row. That was wrong: `using (true)` also permits bulk
-- updates without a WHERE filter, so a single PATCH with `{opted_out:
-- true}` could toggle every row in the table at once. The downside is
-- low (no PII leak, just an annoyance) but it's a trivial-to-exploit
-- amplification we don't need to accept.
--
-- Fix: require the caller to send an `X-Client-Hash` header whose
-- value matches the row's client_hash. PostgREST exposes request
-- headers via `request.headers`; if the header is missing,
-- `current_setting(..., true)` returns NULL and the match fails.
--
-- Client-side pattern (library code, once implemented):
--   1. Compute the machine's client_hash locally.
--   2. PATCH /telemetry_consent with:
--        header:  X-Client-Hash: <hash>
--        body:    { "opted_out": true, "opted_out_at": now() }
--        filter:  ?client_hash=eq.<hash>
--   3. Only the row whose client_hash matches the header updates.
--
-- Bulk updates are now structurally impossible: the RLS filter binds
-- every update to the single row named by the header.
create policy "telemetry_consent_update_own_row"
  on public.telemetry_consent
  for update
  to anon, authenticated
  using (
    client_hash = (current_setting('request.headers', true)::jsonb ->> 'x-client-hash')
  )
  with check (
    client_hash = (current_setting('request.headers', true)::jsonb ->> 'x-client-hash')
  );

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
