import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync, execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { randomUUID } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { setTimeout } from 'node:timers/promises';

// A disposable PostgreSQL instance, with no host ports or network access.
// These checks execute SQL policies and triggers, rather than matching SQL text.
const container = `aidotnet-rls-${randomUUID()}`;
let created = false;
const user = '11111111-1111-4111-8111-111111111111';
const other = '22222222-2222-4222-8222-222222222222';
const admin = '33333333-3333-4333-8333-333333333333';
const key = '44444444-4444-4444-8444-444444444444';
function docker(args, input) {
  const result = spawnSync('docker', args, { input, encoding: 'utf8', timeout: 180_000 });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(result.stderr || result.stdout);
  return result.stdout.trim();
}
function sql(query) {
  return docker(['exec', '-i', container, 'psql', '-X', '-U', 'postgres', '-v', 'ON_ERROR_STOP=1', '-Atq'], query);
}
const migration = (name) => readFileSync(new URL(`../../supabase/migrations/${name}`, import.meta.url), 'utf8');
const asUser = (statement, id = user, role = 'authenticated') => sql(`begin;
  set local role ${role};
  select set_config('request.jwt.claim.sub', '${role === 'anon' ? '' : id}', true);
  select set_config('request.jwt.claim.role', '${role}', true);
  ${statement}
  rollback;`);
function expectDenied(statement, id = user, role = 'authenticated') {
  asUser(`do $test$ begin
    begin
      ${statement};
      raise exception 'Expected insufficient_privilege, but the write succeeded';
    exception when insufficient_privilege then null;
    end;
  end $test$;`, id, role);
}

before(async () => {
  docker(['run', '--rm', '--name', container, '--network', 'none', '-e', 'POSTGRES_HOST_AUTH_METHOD=trust', '-d', 'postgres:17-alpine']);
  created = true;
  let ready = false;
  for (let i = 0; i < 80; i++) {
    // The image starts a socket-only initialization server, then restarts it.
    // TCP readiness distinguishes the final server from that temporary process.
    const check = spawnSync('docker', ['exec', container, 'pg_isready', '-h', '127.0.0.1', '-U', 'postgres'], { encoding: 'utf8' });
    if (check.status === 0) { ready = true; break; }
    await setTimeout(250);
  }
  if (!ready) throw new Error('Test PostgreSQL did not become ready');
  sql(`create role anon; create role authenticated; create role service_role bypassrls;
    create schema auth;
    create function auth.uid() returns uuid language sql stable as $$ select nullif(current_setting('request.jwt.claim.sub', true), '')::uuid $$;
    create function auth.role() returns text language sql stable as $$ select nullif(current_setting('request.jwt.claim.role', true), '') $$;
    create function auth.email() returns text language sql stable as $$ select 'user@example.test'::text $$;
    create table auth.users(id uuid primary key, email text, raw_user_meta_data jsonb default '{}');
    grant usage on schema auth, public to anon, authenticated, service_role;`);
  sql(migration('20260306000000_initial_schema.sql'));
  sql(migration('20260307000000_tighten_rls_policies.sql'));
  sql(migration('20260329000000_add_email_to_profiles.sql'));
  sql(`grant select, insert, update, delete on all tables in schema public to anon, authenticated, service_role;
    grant execute on all functions in schema auth to anon, authenticated, service_role;
    insert into auth.users(id,email) values ('${user}','user@example.test'), ('${other}','other@example.test'), ('${admin}','admin@example.test');
    update public.profiles set role='admin' where id='${admin}';
    insert into public.user_api_keys(id,user_id,key_name,key_hash,key_prefix,scopes) values
      ('${key}','${user}','First','original-hash','original-prefix',array['read']),
      (gen_random_uuid(),'${user}','Second','second-hash','second-prefix',array['read']);
    create policy admin_update_all_profiles on public.profiles for update
      using (auth.uid()=id or public.is_admin());
    create policy "Users can update own API keys" on public.user_api_keys for update using (auth.uid()=user_id);
    create policy legacy_public_usage_insert on public.api_usage for insert to public with check (true);
    create policy existing_auth_requirement on public.profiles as restrictive for update to authenticated
      using (current_setting('request.jwt.claim.aal',true) is distinct from 'aal1');
  `);
  // Prove that the observed deployed-policy combination actually permits escalation.
  assert.match(asUser(`update public.profiles set role='admin' where id='${user}'; select role from public.profiles where id='${user}';`), /admin/);
  for (const role of ['anon', 'authenticated']) {
    assert.match(asUser(`insert into public.api_usage(user_id,endpoint) values ('${user}','/legacy-forged'); select 'inserted';`, user, role), /inserted/);
  }
  const fix = migration('20261117000000_close_legacy_policy_bypasses.sql');
  sql(fix);
  sql(fix); // Safe to reapply in local migration rehearsal.
  sql(`create table public.license_keys(id uuid default gen_random_uuid(), user_id uuid, stripe_subscription_id text, stripe_customer_id text, product text, tier text, status text, notes text, license_key text, max_activations int, expires_at timestamptz);
    create unique index idx_license_keys_one_active_per_user_product_tier on public.license_keys(user_id,product,tier) where status='active';`);
  sql(migration('20261118000000_unique_subscription_entitlement.sql'));
  sql(migration('20261118010000_independent_subscription_entitlements.sql'));
  sql(migration('20261118020000_secure_checkout_ownership.sql'));
});

after(() => { if (created) docker(['stop', container]); });

test('subscription identity rejects a second tier or suspended placeholder, while independent grants remain possible', () => {
  assert.equal(sql(`begin;
    insert into public.license_keys(stripe_subscription_id,product,tier,status) values ('sub_review','aidotnet','professional','suspended');
    do $$ begin
      begin
        insert into public.license_keys(stripe_subscription_id,product,tier,status) values ('sub_review','aidotnet','enterprise','active');
        raise exception 'duplicate subscription was allowed';
      exception when unique_violation then null; end;
    end $$;
    insert into public.license_keys(stripe_subscription_id,product,tier,status) values (null,'aidotnet','professional','active'),(null,'aidotnet','professional','active');
    insert into public.license_keys(stripe_subscription_id,product,tier,status) values ('sub_review','harmonic_engine','professional','active');
    select count(*) from public.license_keys; rollback;`), '4');
});

test('independent paid subscriptions recover alongside existing manual grants at the same tier', () => {
  assert.equal(sql(`begin;
    insert into public.license_keys(user_id,stripe_subscription_id,product,tier,status) values
      ('${user}','sub_a','aidotnet','professional','active'),
      ('${user}','sub_b','aidotnet','professional','suspended'),
      ('${user}',null,'aidotnet','professional','active'),
      ('${user}',null,'aidotnet','professional','active');
    update public.license_keys set status='active' where stripe_subscription_id='sub_b';
    select count(*) from public.license_keys where status='active'; rollback;`), '4');
});

test('profile mirroring retains access from other subscriptions and manual grants', () => {
  const output = sql(`begin;
    update public.profiles set stripe_customer_id='cus_review' where id='${user}';
    insert into public.license_keys(user_id,stripe_subscription_id,product,tier,status) values
      ('${user}','sub_a','aidotnet','professional','expired'),
      ('${user}','sub_b','aidotnet','professional','active'),
      ('${user}',null,'aidotnet','enterprise','suspended');
    select public.sync_aidotnet_billing_profile('cus_review','free','expired');
    select subscription_tier||':'||subscription_status from public.profiles where id='${user}';
    update public.license_keys set status='expired' where stripe_subscription_id='sub_b';
    update public.license_keys set status='active' where tier='enterprise';
    select public.sync_aidotnet_billing_profile('cus_review','free','expired');
    select subscription_tier||':'||subscription_status from public.profiles where id='${user}';
    update public.license_keys set status='expired' where tier='enterprise';
    select public.sync_aidotnet_billing_profile('cus_review','free','expired');
    select subscription_tier||':'||subscription_status from public.profiles where id='${user}'; rollback;`);
  assert.deepEqual(output.split('\n').filter(Boolean), ['pro:active','enterprise:active','free:expired']);
});

for (const role of ['anon','authenticated']) test(`${role} cannot call the privileged profile mirror`, () => {
  expectDenied("perform public.sync_aidotnet_billing_profile('cus_review','enterprise','active')", user, role);
  expectDenied(`perform public.issue_aidotnet_community_license('${user}','AIDN-PROD-COMMUNITY-${'a'.repeat(32)}')`, user, role);
});

test('Community self-service reuses existing grants and creates only one key', () => {
  assert.equal(sql(`begin;
    select public.issue_aidotnet_community_license('${user}','AIDN-PROD-COMMUNITY-${'a'.repeat(32)}')->>'is_existing';
    select public.issue_aidotnet_community_license('${user}','AIDN-PROD-COMMUNITY-${'b'.repeat(32)}')->>'is_existing';
    select count(*) from public.license_keys where user_id='${user}' and tier='community'; rollback;`), 'false\ntrue\n1');
});

test('concurrent Community requests serialize and return the same persisted key', async () => {
  const run = promisify(execFile);
  const issue = async (hex) => (await run('docker', ['exec', container, 'psql', '-X', '-U', 'postgres', '-v', 'ON_ERROR_STOP=1', '-Atq', '-c',
    `begin; select public.issue_aidotnet_community_license('${other}','AIDN-PROD-COMMUNITY-${hex.repeat(32)}')->>'license_key'; select pg_sleep(0.2); commit;`], { timeout: 30_000 })).stdout.trim();
  try {
    const results = await Promise.all([issue('a'), issue('b')]);
    assert.equal(results[0], results[1]);
    assert.equal(sql(`select count(*) from public.license_keys where user_id='${other}' and tier='community';`), '1');
  } finally { sql(`delete from public.license_keys where user_id='${other}' and tier='community';`); }
});

test('Checkout receipt uniqueness permits independent products but rejects duplicate receipts', () => {
  assert.equal(sql(`begin;
    insert into public.license_keys(stripe_checkout_session_id,product) values ('cs_review','aidotnet'),('cs_review','harmonic_engine');
    do $$ begin begin
      insert into public.license_keys(stripe_checkout_session_id,product) values ('cs_review','aidotnet');
      raise exception 'duplicate receipt was allowed';
    exception when unique_violation then null; end; end $$;
    select count(*) from public.license_keys; rollback;`), '2');
});

test('deployment prerequisite checks reject missing or nonunique entitlement indexes', () => {
  const guard = readFileSync(new URL('../../supabase/verify-entitlement-prerequisites.sql', import.meta.url), 'utf8');
  sql(guard);
  assert.throws(() => sql('begin; drop index public.idx_license_keys_subscription_product;\n'+guard), /Run the reviewed migrations/);
  assert.throws(() => sql('begin; drop index public.idx_license_keys_subscription_product; create index idx_license_keys_subscription_product on public.license_keys(stripe_subscription_id,product) where stripe_subscription_id is not null;\n'+guard), /Run the reviewed migrations/);
  assert.throws(() => sql("begin; drop index public.idx_license_keys_subscription_product; create unique index idx_license_keys_subscription_product on public.license_keys(stripe_subscription_id,product) where stripe_subscription_id is not null and tier='professional';\n"+guard), /Run the reviewed migrations/);
  assert.throws(() => sql('begin; drop index public.idx_license_keys_checkout_product;\n'+guard), /Run the reviewed migrations/);
  assert.throws(() => sql('begin; drop function public.issue_aidotnet_community_license(uuid,text);\n'+guard), /Run the reviewed migrations/);
  for (const definition of ['p_customer_id text,p_fallback_tier text,p_fallback_status text,p_user_id uuid','wrong_customer text,p_fallback_tier text,p_fallback_status text,p_user_id uuid default null',"p_customer_id text,p_fallback_tier text,p_fallback_status text,p_user_id uuid default '11111111-1111-4111-8111-111111111111'"]) {
    assert.throws(() => sql(`begin; drop function public.sync_aidotnet_billing_profile(text,text,text,uuid);
      create function public.sync_aidotnet_billing_profile(${definition}) returns void language sql as $$select$$;
      revoke all on function public.sync_aidotnet_billing_profile(text,text,text,uuid) from public,anon,authenticated;
      grant execute on function public.sync_aidotnet_billing_profile(text,text,text,uuid) to service_role;
      `+guard), /RPC argument contract/);
  }
  sql(guard); // Failed transactions rolled back; function-only redeployment stays available.
});

test('owners can save names and revoke one of multiple API keys', () => {
  assert.match(asUser(`update public.profiles set full_name='Saved' where id='${user}'; select full_name from public.profiles where id='${user}';`), /Saved/);
  assert.match(asUser(`update public.user_api_keys set is_active=false where id='${key}'; select is_active from public.user_api_keys where id='${key}';`), /\nf$/);
});

for (const assignment of ["role='admin'", "subscription_tier='enterprise'", "subscription_status='active', stripe_customer_id='cus_forged'", "email='forged@example.test'", "created_at='2000-01-01'"]) {
  test(`owners cannot change protected profile fields: ${assignment}`, () => expectDenied(`update public.profiles set ${assignment} where id='${user}'`));
}
for (const assignment of ["key_hash='forged'", "key_prefix='forged'", "scopes=array['admin']", `user_id='${other}'`, 'last_used_at=now()']) {
  test(`owners cannot change API key credentials/audit fields: ${assignment}`, () => expectDenied(`update public.user_api_keys set ${assignment} where id='${key}'`));
}

test('users cannot insert a privileged profile', () => {
  sql(`begin; delete from public.profiles where id='${user}'; set local role authenticated;
    select set_config('request.jwt.claim.sub','${user}',true);
    do $$ begin
      begin
        insert into public.profiles(id,role) values ('${user}','admin');
        raise exception 'Privileged profile insertion succeeded';
      exception when insufficient_privilege then null; end;
    end $$; rollback;`);
});

test('customer profile repair derives identity and audit fields from trusted state', () => {
  assert.match(sql(`begin; delete from public.profiles where id='${user}'; set local role authenticated;
    select set_config('request.jwt.claim.sub','${user}',true);
    insert into public.profiles(id,email,created_at) values ('${user}','victim@example.test','2000-01-01');
    select email || ':' || (created_at = now())::text from public.profiles where id='${user}'; rollback;`), /user@example.test:true/);
});

test('owners cannot reactivate a revoked API key', () => {
  expectDenied(`update public.user_api_keys set is_active=false where id='${key}'; update public.user_api_keys set is_active=true where id='${key}'`);
});
test('owners cannot update or see another customer profile', () => {
  assert.match(asUser(`with changed as (update public.profiles set full_name='Forbidden' where id='${other}' returning id) select count(*) from changed;`), /\n0$/);
  assert.match(asUser(`select count(*) from public.profiles where id='${other}';`), /\n0$/);
});

test('profile repair uses current Auth email even when the token email is stale', () => {
  assert.match(sql(`begin; update auth.users set email='current@example.test' where id='${user}';
    delete from public.profiles where id='${user}';
    set local role authenticated;
    select set_config('request.jwt.claim.sub','${user}',true);
    select set_config('request.jwt.claim.role','authenticated',true);
    insert into public.profiles(id,email) values('${user}','forged@example.test');
    select email from public.profiles where id='${user}'; rollback;`), /current@example.test$/);
});

test('owner key inserts cannot forge creation or usage history', () => {
  const output = asUser(`insert into public.user_api_keys(user_id,key_name,key_hash,key_prefix,scopes,created_at,last_used_at)
    values('${user}','Audit fixture','audit-hash','audit',array['read'],'2000-01-01','2000-01-02');
    select (created_at >= now()-interval '1 second') and last_used_at is null from public.user_api_keys where key_hash='audit-hash';`);
  assert.match(output,/\nt$/);
});

test('Checkout ownership is immutable across customers and accounts', () => {
  assert.equal(sql(`begin;
    select public.bind_aidotnet_checkout('${user}','cs_owned','cus_owned');
    select public.bind_aidotnet_checkout('${user}','cs_owned','cus_owned');
    do $$ begin begin
      perform public.bind_aidotnet_checkout('${other}','cs_other','cus_owned');
      raise exception 'foreign customer accepted';
    exception when raise_exception then if sqlerrm='foreign customer accepted' then raise; end if; end; end $$;
    select count(*) from public.billing_customers; rollback;`), '1');
});

test('a revoked Enterprise subscription cannot elevate an active Professional grant', () => {
  assert.equal(sql(`begin; update public.profiles set stripe_customer_id='cus_review' where id='${user}';
    insert into public.license_keys(user_id,product,tier,status) values('${user}','aidotnet','professional','active'),('${user}','aidotnet','enterprise','revoked');
    select public.sync_aidotnet_billing_profile('cus_review','enterprise','active');
    select subscription_tier from public.profiles where id='${user}'; rollback;`), 'pro');
});

for (const role of ['anon','authenticated']) test(`${role} cannot forge Checkout ownership or intents`, () => {
  expectDenied(`perform public.bind_aidotnet_checkout('${user}','cs_fake','cus_fake')`,user,role);
  expectDenied(`insert into public.billing_checkout_intents(user_id,tier,interval) values('${user}','enterprise','month')`,user,role);
});

test('hardening preserves existing restrictive authentication policies', () => {
  assert.equal(sql("select count(*) from pg_policies where policyname='existing_auth_requirement' and permissive='RESTRICTIVE'"), '1');
  assert.match(asUser(`select set_config('request.jwt.claim.aal','aal1',true);
    with changed as (update public.profiles set full_name='Blocked' where id='${user}' returning id) select count(*) from changed;`), /\n0$/);
});
test('admins can edit roles and plans', () => {
  assert.match(asUser(`update public.profiles set role='admin', subscription_tier='pro' where id='${other}'; select subscription_tier from public.profiles where id='${other}';`, admin), /\npro$/);
});
test('service role can mirror Stripe billing', () => {
  assert.match(asUser(`update public.profiles set subscription_tier='pro', stripe_customer_id='cus_fixture' where id='${user}'; select subscription_tier from public.profiles where id='${user}';`, user, 'service_role'), /\npro$/);
});
test('the trusted auth email-sync trigger still updates profile email', () => {
  assert.match(sql(`begin; update auth.users set email='updated@example.test' where id='${user}';
    select email from public.profiles where id='${user}'; rollback;`), /updated@example.test/);
});
for (const role of ['anon', 'authenticated']) {
  test(`${role} cannot forge usage records`, () => expectDenied(`insert into public.api_usage(user_id,endpoint) values ('${user}','/forged')`, user, role));
}
test('service role can record usage', () => {
  assert.match(asUser(`insert into public.api_usage(user_id,endpoint) values ('${user}','/fixture'); select count(*) from public.api_usage;`, user, 'service_role'), /\n1$/);
});
test('an accidentally reintroduced permissive policy does not bypass field guards', () => {
  sql(`create policy regression_owner_policy on public.profiles for update using (auth.uid()=id);`);
  expectDenied(`update public.profiles set role='admin' where id='${user}'`);
});

test('release DDL and its exact migration history entry commit or roll back together', () => {
  sql('create schema supabase_migrations; create table supabase_migrations.schema_migrations(version text primary key, name text, statements text[]);');
  const source = migration('20261117000000_close_legacy_policy_bypasses.sql');
  const args = ['exec', '-i', container, 'psql', '-X', '-U', 'postgres', '--single-transaction', '-v', 'ON_ERROR_STOP=1',
    '--set=migration_version=20261117000000', `--set=migration_source=${source}`, '-f', '-'];
  const history = "insert into supabase_migrations.schema_migrations(version,name,statements) values (:'migration_version','close_legacy_policy_bypasses',array[:'migration_source']);";
  docker(args, source + '\n' + history);
  assert.equal(sql('select version from supabase_migrations.schema_migrations'), '20261117000000');
  assert.throws(() => docker(args, 'create table public.failed_release_marker(id int);\n' + history), /duplicate key/);
  assert.equal(sql("select to_regclass('public.failed_release_marker') is null"), 't');
});
