import { createHmac, randomBytes } from 'node:crypto';
import type { APIRequestContext } from '@playwright/test';

// Server-side helpers for the real-backend journeys. The E2E workflow provides:
//   SUPABASE_URL               the project URL (PUBLIC_SUPABASE_URL secret)
//   SUPABASE_SERVICE_ROLE_KEY  fetched at run time with the Supabase CLI and masked
//   STRIPE_WEBHOOK_SECRET      the signing secret the stripe-webhook function verifies against
//
// In CI a missing variable is a failure, never a skip: a journey that silently skips is a
// gate that fails open. Locally, without the variables, the journeys skip with the reason.

export const JOURNEY_ENV = ['SUPABASE_URL', 'SUPABASE_SERVICE_ROLE_KEY', 'STRIPE_WEBHOOK_SECRET'] as const;

export function missingJourneyEnv(): string[] {
  return JOURNEY_ENV.filter((k) => !process.env[k]);
}

/** Throws in CI when a journey variable is missing; returns a skip reason locally. */
export function journeySkipReason(): string | undefined {
  const missing = missingJourneyEnv();
  if (missing.length === 0) return undefined;
  if (process.env.CI) throw new Error(`Journey env not set in CI: ${missing.join(', ')}`);
  return `set ${missing.join(', ')} to run the real-backend journeys`;
}

const env = (k: (typeof JOURNEY_ENV)[number]) => process.env[k] as string;

function adminHeaders() {
  const key = env('SUPABASE_SERVICE_ROLE_KEY');
  return { apikey: key, authorization: `Bearer ${key}`, 'content-type': 'application/json' };
}

/** Test users live at Resend's test inbox: mail to it is accepted and never delivered. The
 *  nightly cleanup deletes users matching this prefix once they are an hour old. */
export const TEST_USER_PREFIX = 'delivered+e2e-user-';
export function testUserEmail(tag: string): string {
  return `${TEST_USER_PREFIX}${tag}-${Date.now()}-${randomBytes(3).toString('hex')}@resend.dev`;
}

export function randomPassword(): string {
  return `E2e!${randomBytes(12).toString('base64url')}`;
}

export interface AdminUser { id: string; email: string }

export async function createConfirmedUser(request: APIRequestContext, email: string, password: string, fullName: string): Promise<AdminUser> {
  const res = await request.post(`${env('SUPABASE_URL')}/auth/v1/admin/users`, {
    headers: adminHeaders(),
    data: { email, password, email_confirm: true, user_metadata: { full_name: fullName } },
  });
  if (!res.ok()) throw new Error(`createUser: HTTP ${res.status()} ${await res.text()}`);
  const body = await res.json();
  return { id: body.id, email: body.email };
}

export async function findUserIdByEmail(request: APIRequestContext, email: string): Promise<string> {
  const res = await request.get(`${env('SUPABASE_URL')}/rest/v1/profiles?select=id&email=eq.${encodeURIComponent(email)}`, {
    headers: adminHeaders(),
  });
  if (!res.ok()) throw new Error(`profile lookup: HTTP ${res.status()} ${await res.text()}`);
  const rows = (await res.json()) as Array<{ id: string }>;
  if (rows.length !== 1) throw new Error(`profile lookup: expected 1 row for the test user, got ${rows.length}`);
  return rows[0].id;
}

export async function deleteUser(request: APIRequestContext, id: string): Promise<void> {
  const res = await request.delete(`${env('SUPABASE_URL')}/auth/v1/admin/users/${id}`, { headers: adminHeaders() });
  if (!res.ok() && res.status() !== 404) throw new Error(`deleteUser: HTTP ${res.status()} ${await res.text()}`);
}

/** The link Supabase would email (signup confirmation or password recovery), without the email. */
export async function generateLink(
  request: APIRequestContext,
  type: 'signup' | 'recovery' | 'magiclink',
  email: string,
  redirectTo: string,
  password?: string,
): Promise<string> {
  const res = await request.post(`${env('SUPABASE_URL')}/auth/v1/admin/generate_link`, {
    headers: adminHeaders(),
    data: { type, email, password, redirect_to: redirectTo },
  });
  if (!res.ok()) throw new Error(`generate_link(${type}): HTTP ${res.status()} ${await res.text()}`);
  const body = await res.json();
  const link = body.action_link ?? body.properties?.action_link;
  if (typeof link !== 'string') throw new Error(`generate_link(${type}): no action_link in response`);
  return link;
}

export async function restSelect<T>(request: APIRequestContext, pathAndQuery: string): Promise<T> {
  const res = await request.get(`${env('SUPABASE_URL')}/rest/v1/${pathAndQuery}`, { headers: adminHeaders() });
  if (!res.ok()) throw new Error(`select ${pathAndQuery}: HTTP ${res.status()} ${await res.text()}`);
  return (await res.json()) as T;
}

export async function restInsert<T>(request: APIRequestContext, table: string, row: Record<string, unknown>): Promise<T> {
  const res = await request.post(`${env('SUPABASE_URL')}/rest/v1/${table}`, {
    headers: { ...adminHeaders(), prefer: 'return=representation' },
    data: row,
  });
  if (!res.ok()) throw new Error(`insert ${table}: HTTP ${res.status()} ${await res.text()}`);
  return ((await res.json()) as T[])[0];
}

/** Posts a Stripe event to the stripe-webhook function, signed exactly as Stripe signs it. */
export async function postSignedStripeEvent(request: APIRequestContext, event: Record<string, unknown>) {
  const payload = JSON.stringify(event);
  const t = Math.floor(Date.now() / 1000);
  const v1 = createHmac('sha256', env('STRIPE_WEBHOOK_SECRET')).update(`${t}.${payload}`).digest('hex');
  return request.post(`${env('SUPABASE_URL')}/functions/v1/stripe-webhook`, {
    headers: { 'content-type': 'application/json', 'stripe-signature': `t=${t},v1=${v1}` },
    data: payload,
  });
}
