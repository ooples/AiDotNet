-- Internal functions that should NOT be callable as public RPCs.
-- Functions are granted EXECUTE to PUBLIC by default; revoking closes the
-- /rest/v1/rpc/* exposure. Trigger functions still fire (triggers bypass the
-- EXECUTE privilege check), and the maintenance function stays callable by the
-- backend (service_role).

-- auth trigger helpers (fired by triggers on auth.users, never meant as RPC)
REVOKE EXECUTE ON FUNCTION public.handle_new_user() FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION public.handle_user_email_update() FROM PUBLIC, anon, authenticated;

-- maintenance/cleanup: backend only
REVOKE EXECUTE ON FUNCTION public.cleanup_old_exception_telemetry(integer) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.cleanup_old_exception_telemetry(integer) TO service_role;
