-- Migration: offline revocation list (CRL) for v2 `aidn2` tokens.
--
-- Online (AIDN-*) keys are revoked instantly by flipping license_keys.status to 'revoked' — the next
-- validate_license_key call returns license_revoked. But an OFFLINE aidn2 token verifies purely against the
-- SDK-embedded public key and never contacts the server, so status alone can't stop it inside its validity
-- window. This table is the deny-list the SDK's LicenseRevocationProvider consults: get-revocations signs a
-- CRL listing the revoked token ids (jti = license_keys.id) and signing-key ids (kid), the SDK fetches +
-- verifies + caches it, and rejects any matching token even offline.
--
-- Additive and safe: no existing table or function is altered, so the 12 live paid customers are unaffected.

create table if not exists public.revocations (
    id             uuid primary key default gen_random_uuid(),
    -- The license whose offline tokens are being revoked. FK so deleting a license cascades its revocation
    -- rows; nullable so a kid-only (whole-signing-key) revocation can exist without a specific license.
    license_key_id uuid references public.license_keys(id) on delete cascade,
    -- Token id to deny. For customer tokens this equals license_keys.id (what issue-license stamps as jti and
    -- what validate_license_key returns as license_id). Stored as text to also allow ad-hoc/non-license jtis.
    jti            text,
    -- Optional: revoke an ENTIRE signing key id (mass-revoke every token signed with a compromised key).
    kid            text,
    reason         text,
    revoked_at     timestamptz not null default now(),
    created_at     timestamptz not null default now(),
    -- At least one of jti/kid must be present, else the row denies nothing.
    constraint revocations_target_present check (jti is not null or kid is not null)
);

comment on table public.revocations is
    'Offline revocation list (CRL) source for v2 aidn2 tokens. get-revocations signs the active rows into a '
    'CRL the SDK verifies + enforces offline. Online AIDN-* keys revoke via license_keys.status instead.';

create index if not exists revocations_jti_idx on public.revocations (jti) where jti is not null;
create index if not exists revocations_kid_idx on public.revocations (kid) where kid is not null;

-- RLS: server-only. The CRL is emitted by the service-role edge function; no anon/authenticated access to
-- the raw table (the signed CRL is the public surface, not this table).
alter table public.revocations enable row level security;
revoke all on public.revocations from anon, authenticated;

-- Atomic revoke helper for the admin dashboard / support tooling: flips the online status AND records the
-- offline deny-list entry in one transaction, so a revoked license is stopped on BOTH paths and the two can
-- never drift (online-revoked-but-still-offline-valid, or vice-versa).
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
    -- Online path: next validate_license_key returns license_revoked immediately.
    update public.license_keys
    set status = 'revoked', revoked_at = now(), updated_at = now()
    where id = p_license_key_id;

    -- Offline path: deny the license's aidn2 tokens (jti = license id) via the CRL. Idempotent — a repeat
    -- revoke of the same license doesn't stack duplicate rows.
    insert into public.revocations (license_key_id, jti, reason)
    select p_license_key_id, p_license_key_id::text, p_reason
    where not exists (
        select 1 from public.revocations
        where license_key_id = p_license_key_id and jti = p_license_key_id::text
    );
end;
$$;

revoke execute on function public.revoke_license(uuid, text) from anon, authenticated;
grant execute on function public.revoke_license(uuid, text) to service_role;

comment on function public.revoke_license(uuid, text) is
    'Revokes a license on BOTH the online (license_keys.status=revoked) and offline (revocations CRL) paths '
    'in one transaction. Call from admin tooling; enforced offline once get-revocations re-signs the CRL.';
