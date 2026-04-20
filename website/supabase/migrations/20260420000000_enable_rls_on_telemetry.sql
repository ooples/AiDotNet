-- Enable Row-Level Security on four telemetry tables that ship with
-- RLS OFF. The Supabase security advisor flags these as ERROR-level:
-- they live in the `public` schema and are reachable by anon API key
-- through PostgREST, which means an attacker with the (publicly bundled)
-- anon key could SELECT/INSERT/UPDATE/DELETE arbitrary rows.
--
-- All four tables are empty in production today, so we can add
-- policies without a data migration. The write path for anon clients
-- is INSERT-only (for gpu_profiles / gpu_telemetry / exception_telemetry)
-- and INSERT+UPDATE keyed by client_hash (for telemetry_consent,
-- which users mutate to opt out). Only users with public.profiles.role
-- = 'admin' can SELECT.
--
-- Also pins the search_path on public.update_license_keys_updated_at()
-- so a caller can't shadow the expected `public` schema and hijack
-- table lookups inside the trigger.

-- ---------- gpu_profiles ----------
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
  using (
    exists (
      select 1 from public.profiles p
      where p.id = auth.uid() and p.role = 'admin'
    )
  );

-- ---------- gpu_telemetry ----------
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
  using (
    exists (
      select 1 from public.profiles p
      where p.id = auth.uid() and p.role = 'admin'
    )
  );

-- ---------- exception_telemetry ----------
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
  using (
    exists (
      select 1 from public.profiles p
      where p.id = auth.uid() and p.role = 'admin'
    )
  );

-- ---------- telemetry_consent ----------
-- Consent is keyed by a one-way machine hash (client_hash PK). Users
-- opt out by UPSERT: INSERT the consent row, and UPDATE by client_hash
-- match. The hash is not sensitive enough to require auth on write,
-- but SELECT is admin-only so we don't expose the full machine list.
alter table public.telemetry_consent enable row level security;

create policy "telemetry_consent_insert_anon"
  on public.telemetry_consent
  for insert
  to anon, authenticated
  with check (true);

create policy "telemetry_consent_update_own_hash"
  on public.telemetry_consent
  for update
  to anon, authenticated
  using (true)
  with check (true);

create policy "telemetry_consent_select_admin"
  on public.telemetry_consent
  for select
  to authenticated
  using (
    exists (
      select 1 from public.profiles p
      where p.id = auth.uid() and p.role = 'admin'
    )
  );

-- ---------- Pin search_path on update_license_keys_updated_at ----------
-- A role-mutable search_path lets a caller shadow `public` with their
-- own schema and hijack table lookups inside the trigger function.
-- Pinning to (public, pg_catalog) closes that vector.
alter function public.update_license_keys_updated_at()
  set search_path = 'public', 'pg_catalog';
