-- Phase 2 community GPU-tuning exchange (opt-in).
-- One row = a locally-measured winning kernel config, keyed by the per-MODEL
-- hardware class (not the per-card UUID) so a config tuned on one card reaches
-- every card of that model on the same driver + architecture.
-- Trust model: measured_gflops is advisory. Clients treat a downloaded row as a
-- candidate that must win their own on-device sweep before it is used.
create table if not exists public.gpu_tuning_profiles (
    id               bigint generated always as identity primary key,
    model_key        text        not null,
    vendor           text        not null,
    model            text        not null,
    architecture     text        not null,
    driver_version   integer     not null,
    category         text        not null,
    kernel_name      text        not null,
    shape_key        text        not null,
    variant          text        not null,
    parameters       jsonb       not null default '{}'::jsonb,
    measured_gflops  double precision not null,
    client_hash      text,
    aidotnet_version text,
    recorded_at      timestamptz not null default now(),
    constraint gpu_tuning_profiles_gflops_nonneg check (measured_gflops >= 0)
);

create index if not exists gpu_tuning_profiles_lookup
    on public.gpu_tuning_profiles (model_key, category, kernel_name, shape_key, measured_gflops desc);

alter table public.gpu_tuning_profiles enable row level security;

drop policy if exists gpu_tuning_profiles_anon_insert on public.gpu_tuning_profiles;
create policy gpu_tuning_profiles_anon_insert
    on public.gpu_tuning_profiles for insert to anon with check (true);

drop policy if exists gpu_tuning_profiles_anon_select on public.gpu_tuning_profiles;
create policy gpu_tuning_profiles_anon_select
    on public.gpu_tuning_profiles for select to anon using (true);
