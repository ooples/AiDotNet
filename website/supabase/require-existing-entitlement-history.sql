-- These release workflows target the existing shared project, not arbitrary legacy databases.
-- The historical index/backfill migration is already applied there. Never replay its
-- lock-taking DDL against an unknown dataset or guess identities for duplicate receipts.
do $$ begin
  if (select count(*) from supabase_migrations.schema_migrations
      where version in ('20261117000000','20261118000000','20261118010000')) <> 3 then
    raise exception 'Existing entitlement migration history is required. Stop deployment: prepare a separate online rollout and inspect duplicate Checkout receipts without deleting or revoking historical keys.';
  end if;
end $$;
