-- Security hardening for the revocations feature (follows 20260718000000_revocations.sql).
--
-- Caught by the Supabase security advisor immediately after the prior migration:
--   * revoke_license is a DESTRUCTIVE SECURITY DEFINER function, but Postgres grants EXECUTE to PUBLIC on
--     function creation. Revoking from anon/authenticated specifically does NOT strip that inherited PUBLIC
--     grant, so the function stayed reachable via /rest/v1/rpc/revoke_license by the anon role — anyone could
--     revoke a license. Revoke from PUBLIC so only service_role (admin tooling / edge functions) can call it.
--   * revocations had RLS enabled with no policy (rls_enabled_no_policy). Add a service-role-only policy so
--     get-revocations can read the deny-list while anon/authenticated stay fully denied.

revoke execute on function public.revoke_license(uuid, text) from public;
grant execute on function public.revoke_license(uuid, text) to service_role;

drop policy if exists revocations_service_all on public.revocations;
create policy revocations_service_all on public.revocations
    for all to service_role using (true) with check (true);
