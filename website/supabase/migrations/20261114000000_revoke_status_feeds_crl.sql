-- Any license that becomes 'revoked' goes on the CRL (public.revocations), whoever revoked it.
--
-- Before this, only public.revoke_license() wrote the CRL row, and nothing called it: the admin
-- page's Revoke button and the Stripe webhook both set license_keys.status = 'revoked' directly.
-- Online validation refused the key at once, but the license's offline aidn2 tokens (jti = license
-- id) kept verifying until they expired, up to OFFLINE_TOKEN_EXP_DAYS (14) later.
--
-- A trigger covers every path that sets the status, including the ones written later. revoke_license()
-- keeps working: its own insert and the trigger's insert hit the same partial unique index, and both
-- use ON CONFLICT DO NOTHING.

create or replace function public.license_keys_revoked_to_crl()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
    insert into public.revocations (license_key_id, jti, reason)
    values (new.id, new.id::text, 'license_keys.status set to revoked')
    on conflict (license_key_id, jti)
        where license_key_id is not null and jti is not null
        do nothing;
    return new;
end;
$$;

revoke execute on function public.license_keys_revoked_to_crl() from public, anon, authenticated;

drop trigger if exists license_keys_revoked_to_crl on public.license_keys;
create trigger license_keys_revoked_to_crl
    after insert or update of status on public.license_keys
    for each row
    when (new.status = 'revoked')
    execute function public.license_keys_revoked_to_crl();

-- Backfill: licenses revoked before this migration whose offline tokens were never denied.
insert into public.revocations (license_key_id, jti, reason)
select id, id::text, 'backfill: revoked before the CRL trigger existed'
from public.license_keys
where status = 'revoked'
on conflict (license_key_id, jti)
    where license_key_id is not null and jti is not null
    do nothing;
