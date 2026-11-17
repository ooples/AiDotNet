import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
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
  select set_config('request.jwt.claim.sub', '${id}', true);
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
    const check = spawnSync('docker', ['exec', container, 'pg_isready', '-U', 'postgres'], { encoding: 'utf8' });
    if (check.status === 0) { ready = true; break; }
    await setTimeout(250);
  }
  if (!ready) throw new Error('Test PostgreSQL did not become ready');
  sql(`create role anon; create role authenticated; create role service_role bypassrls;
    create schema auth;
    create function auth.uid() returns uuid language sql stable as $$ select nullif(current_setting('request.jwt.claim.sub', true), '')::uuid $$;
    create function auth.role() returns text language sql stable as $$ select nullif(current_setting('request.jwt.claim.role', true), '') $$;
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
  `);
  // Prove that the observed deployed-policy combination actually permits escalation.
  assert.match(asUser(`update public.profiles set role='admin' where id='${user}'; select role from public.profiles where id='${user}';`), /admin/);
  const fix = migration('20261117000000_close_legacy_policy_bypasses.sql');
  sql(fix);
  sql(fix); // Safe to reapply in local migration rehearsal.
});

after(() => { if (created) docker(['stop', container]); });

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
test('owners cannot update or see another customer profile', () => {
  assert.match(asUser(`with changed as (update public.profiles set full_name='Forbidden' where id='${other}' returning id) select count(*) from changed;`), /\n0$/);
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
