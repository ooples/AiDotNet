import { test, expect, type APIRequestContext } from '@playwright/test';
import { randomBytes } from 'node:crypto';
import { createConfirmedUser, deleteUser, journeySkipReason, randomPassword, restInsert, restSelect, testUserEmail } from '../helpers/backend';

test('real customer tokens cannot bypass profile, key or usage boundaries', async ({ request }) => {
  const skip = journeySkipReason();
  test.skip(!!skip, skip);
  test.setTimeout(120_000);
  const anon = process.env.PUBLIC_SUPABASE_ANON_KEY;
  if (!anon) throw new Error('PUBLIC_SUPABASE_ANON_KEY is required for direct customer-token RLS checks');
  const url = process.env.SUPABASE_URL!;
  const users: Array<{ id: string; email: string; token: string }> = [];
  try {
    for (const label of ['owner', 'other']) {
      const password = randomPassword(), email = testUserEmail(`rls-${label}`);
      const user = await createConfirmedUser(request, email, password, 'E2E RLS boundary');
      // Register cleanup before authentication so a failed login cannot leak the account.
      users.push({ ...user, token: '' });
      const response = await request.post(`${url}/auth/v1/token?grant_type=password`, {
        headers: { apikey: anon }, data: { email, password },
      });
      expect(response.status(), 'disposable customer can sign in').toBe(200);
      users.at(-1)!.token = (await response.json()).access_token;
    }
    const [owner, other] = users;
    const headers = (token: string) => ({ apikey: anon, Authorization: `Bearer ${token}`, Prefer: 'return=representation' });
    const patch = (table: string, id: string, data: object, token = owner.token) => request.patch(`${url}/rest/v1/${table}?id=eq.${id}`, { headers: headers(token), data });
    for (const payload of [{ role: 'admin' }, { subscription_tier: 'enterprise' }, { stripe_customer_id: 'cus_forged' }, { email: 'forged@example.test' }]) {
      expect((await patch('profiles', owner.id, payload)).status(), 'protected profile write is denied').toBe(403);
    }
    const saved = await patch('profiles', owner.id, { full_name: 'E2E saved by owner' });
    expect(saved.status()).toBe(200);
    expect((await saved.json())[0].full_name).toBe('E2E saved by owner');
    const foreign = await patch('profiles', owner.id, { full_name: 'Forbidden' }, other.token);
    expect(foreign.status()).toBe(200);
    expect(await foreign.json()).toEqual([]);
    const key = await restInsert<{ id: string }>(request, 'user_api_keys', {
      user_id: owner.id, key_name: 'E2E RLS key', key_hash: randomBytes(32).toString('hex'), key_prefix: 'e2e', scopes: ['inference'],
    });
    expect((await patch('user_api_keys', key.id, { scopes: ['admin'] })).status()).toBe(403);
    expect((await patch('user_api_keys', key.id, { is_active: false })).status()).toBe(200);
    expect((await patch('user_api_keys', key.id, { is_active: true })).status()).toBe(403);
    const keys = await restSelect<Array<{ is_active: boolean }>>(request, `user_api_keys?select=is_active&id=eq.${key.id}`);
    expect(keys[0].is_active).toBe(false);
    for (const token of [undefined, owner.token]) {
      const response = await request.post(`${url}/rest/v1/api_usage`, {
        headers: token ? headers(token) : { apikey: anon }, data: { user_id: owner.id, endpoint: '/e2e-forged' },
      });
      expect(response.status(), 'untrusted usage insertion is denied').toBe(403);
    }
  } finally {
    const cleanup = await Promise.allSettled(users.map(user => deleteUser(request as APIRequestContext, user.id)));
    const failure = cleanup.find(result => result.status === 'rejected');
    if (failure?.status === 'rejected') throw failure.reason;
  }
});
