-- Migration: Create telemetry_events table for anonymous library usage telemetry
-- This table receives fire-and-forget INSERT-only events from the AiDotNet library.
-- No PII is stored — machine_id_hash is a one-way SHA-256 truncation.

create table if not exists public.telemetry_events (
    id bigint generated always as identity primary key,
    event_type text not null check (char_length(event_type) between 1 and 100),
    machine_id_hash text not null check (char_length(machine_id_hash) between 1 and 128),
    library_version text not null default '0.0.0' check (char_length(library_version) <= 50),
    timestamp_utc timestamptz not null default now(),
    properties jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

-- Index for querying by event type (admin analytics)
create index if not exists idx_telemetry_events_event_type
    on public.telemetry_events (event_type);

-- Index for time-range queries (dashboards)
create index if not exists idx_telemetry_events_timestamp
    on public.telemetry_events (timestamp_utc desc);

-- Index for machine-level aggregation (unique users)
create index if not exists idx_telemetry_events_machine
    on public.telemetry_events (machine_id_hash, event_type);

-- RLS: Enable row-level security
alter table public.telemetry_events enable row level security;

-- RLS Policy: Anyone can INSERT (anon key), but only admins can SELECT.
-- This is intentional: the NuGet library sends anonymous telemetry using the
-- Supabase anon key. The with check (true) allows unrestricted inserts from
-- any client with the anon key. No PII is collected — only event type, version,
-- and machine_id_hash. The risk of spam is accepted as telemetry is append-only
-- and low-value to attackers.
create policy "telemetry_events_insert_anon"
    on public.telemetry_events
    for insert
    to anon
    with check (true);

-- Admins can read all telemetry events for analytics
create policy "telemetry_events_select_admin"
    on public.telemetry_events
    for select
    to authenticated
    using (public.is_admin());

-- Admins can delete old telemetry events for cleanup
create policy "telemetry_events_delete_admin"
    on public.telemetry_events
    for delete
    to authenticated
    using (public.is_admin());

-- Comment on table and columns for documentation
comment on table public.telemetry_events is 'Anonymous library usage telemetry. INSERT-only via anon key, read by admins.';
comment on column public.telemetry_events.event_type is 'Event category: trial_operation, trial_expired, licensed_operation, licensing_error';
comment on column public.telemetry_events.machine_id_hash is 'Truncated SHA-256 of machine fingerprint. Not reversible to PII.';
comment on column public.telemetry_events.library_version is 'AiDotNet library version that generated the event.';
comment on column public.telemetry_events.properties is 'Event-specific JSON properties (operation_count, reason, etc.).';
