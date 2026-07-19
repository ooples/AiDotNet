-- Forward-fix for 20260718000000_revocations.sql, which was already applied to production before review.
-- Two issues (CodeRabbit, PR #1891):
--   * The license_key_id FK was ON DELETE CASCADE, so deleting a license deleted its CRL deny-list rows —
--     previously-issued offline tokens for that license would validate again until expiry. Must be SET NULL
--     so the jti deny-list entry outlives the license row.
--   * revoke_license's insert used a racy NOT EXISTS check-then-insert; two concurrent calls could both
--     pass and double-insert. A partial unique index + ON CONFLICT DO NOTHING makes it concurrency-safe.
-- Idempotent, so a fresh environment that already got the corrected 20260718000000 applies this as a no-op.

-- 1. FK: cascade -> set null. Postgres auto-named the inline FK revocations_license_key_id_fkey.
alter table public.revocations drop constraint if exists revocations_license_key_id_fkey;
alter table public.revocations
    add constraint revocations_license_key_id_fkey
    foreign key (license_key_id) references public.license_keys(id) on delete set null;

-- 2. Partial unique index backing the idempotent upsert.
create unique index if not exists revocations_license_jti_unique
    on public.revocations (license_key_id, jti)
    where license_key_id is not null and jti is not null;

-- 3. Concurrency-safe revoke_license.
create or replace function public.revoke_license(
    p_license_key_id uuid,
    p_reason text default null
)
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
    update public.license_keys
    set status = 'revoked', revoked_at = now(), updated_at = now()
    where id = p_license_key_id;

    insert into public.revocations (license_key_id, jti, reason)
    values (p_license_key_id, p_license_key_id::text, p_reason)
    on conflict (license_key_id, jti)
        where license_key_id is not null and jti is not null
        do nothing;
end;
$$;

revoke execute on function public.revoke_license(uuid, text) from public;
grant execute on function public.revoke_license(uuid, text) to service_role;
