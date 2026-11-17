import { test as setup, expect } from '@playwright/test';

setup('authorize the protected preview deployment', async ({ request, baseURL }) => {
  const secret = process.env.VERCEL_AUTOMATION_BYPASS_SECRET;
  if (!secret) throw new Error('A preview automation credential is required');
  // Establish a same-origin cookie once. Never attach a Vercel credential
  // globally: that would send it to Stripe, Supabase, and external links.
  const first = await request.get(baseURL!, {
    headers: { 'x-vercel-protection-bypass': secret, 'x-vercel-set-bypass-cookie': 'true' },
    maxRedirects: 0,
  });
  expect([200, 302, 303, 307, 308]).toContain(first.status());
  const response = await request.get(baseURL!);
  expect(response.status(), 'preview home must exist before browser journeys run').toBe(200);
  expect(new URL(response.url()).origin).toBe(new URL(baseURL!).origin);
  await request.storageState({ path: 'tests/e2e/.auth/deployment.json' });
});
