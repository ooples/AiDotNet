import { test, expect, type Page } from '@playwright/test';

// Deterministic UI contracts, not a substitute for Stripe/Supabase integration.
// Only the isolated build produced by test:preview can run these sessions.
test.skip(process.env.PUBLIC_SUPABASE_URL !== 'https://preview-test.supabase.co', 'run with npm run test:preview');

const user = {
  id: '11111111-1111-4111-8111-111111111111', email: 'preview@example.test',
  role: 'authenticated', aud: 'authenticated', created_at: '2026-01-01T00:00:00Z',
  app_metadata: { provider: 'email', providers: ['email'] }, user_metadata: { full_name: 'Preview Customer' },
};
const subscription = {
  status: 'active', tier: 'pro', interval: 'month', unit_amount: 2900, currency: 'usd', quantity: 2,
  current_period_end: 1893456000, cancel_at_period_end: false, trial_end: null,
};

async function installBackend(page: Page, options: {
  tier?: string; customer?: boolean; sub?: Record<string, unknown> | null; profileFailure?: boolean; summaryFailure?: boolean;
} = {}) {
  const profile = {
    id: user.id, email: user.email, full_name: 'Preview Customer', role: 'user',
    subscription_tier: options.tier ?? 'free', subscription_status: 'active',
    stripe_customer_id: options.customer ? 'cus_preview' : null,
  };
  const jwt = `${Buffer.from(JSON.stringify({ alg: 'HS256', typ: 'JWT' })).toString('base64url')}.${Buffer.from(JSON.stringify({ sub: user.id, exp: 4102444800, role: 'authenticated' })).toString('base64url')}.fixture`;
  await page.addInitScript(({ user, jwt }) => {
    localStorage.setItem('sb-preview-test-auth-token', JSON.stringify({
      access_token: jwt, refresh_token: 'preview-refresh', expires_at: 4102444800, expires_in: 3600, token_type: 'bearer', user,
    }));
  }, { user, jwt });
  const calls: Array<Record<string, unknown>> = [];
  await page.route('https://preview-test.supabase.co/**', async (route) => {
    const path = new URL(route.request().url()).pathname;
    const reply = (body: unknown, status = 200) => route.fulfill({ status, json: body });
    if (path === '/auth/v1/user') return reply(user);
    if (path === '/rest/v1/profiles') {
      if (options.profileFailure) return reply({ message: 'profile unavailable' }, 503);
      const single = route.request().headers().accept?.includes('pgrst.object');
      return reply(single ? profile : [profile]);
    }
    if (path === '/functions/v1/billing-portal') {
      const body = route.request().postDataJSON();
      calls.push(body);
      if (body.action === 'summary') {
        return options.summaryFailure ? reply({ message: 'Stripe unavailable' }, 502) : reply({ subscription: options.sub ?? null });
      }
      return reply({ message: 'Portal temporarily unavailable' }, 502);
    }
    // An unexpected request must fail the test instead of reaching any backend.
    return reply({ message: `Unhandled preview endpoint: ${path}` }, 501);
  });
  return calls;
}

for (const state of [
  { name: 'free', tier: 'free', customer: false, sub: null, label: 'Free', price: '$0/month' },
  { name: 'team-granted Pro', tier: 'professional', customer: false, sub: null, label: 'Pro', price: '$29/month per seat' },
  { name: 'Pro subscription', tier: 'pro', customer: true, sub: subscription, label: 'Pro', price: '$58/month (2 seats)' },
  { name: 'Enterprise overrides a stale Pro profile', tier: 'pro', customer: true, sub: { ...subscription, tier: 'enterprise', unit_amount: 9900, quantity: 1 }, label: 'Enterprise', price: '$99/month' },
  { name: 'scheduled cancellation', tier: 'pro', customer: true, sub: { ...subscription, cancel_at_period_end: true }, label: 'Pro', price: '$58/month (2 seats)' },
  { name: 'past due', tier: 'pro', customer: true, sub: { ...subscription, status: 'past_due' }, label: 'Pro', price: '$58/month (2 seats)' },
]) {
  test(`billing displays ${state.name} and the correct available actions`, async ({ page }) => {
    await installBackend(page, state);
    await page.goto('/account/billing/');
    await expect(page.locator('#billing-root')).toHaveAttribute('data-billing-loaded', 'true');
    await expect(page.locator('#billing-error')).toBeHidden();
    await expect(page.locator('#billing-plan')).toHaveText(state.label);
    await expect(page.locator('#billing-price')).toHaveText(state.price);
    await expect(page.locator('#portal-section')).toBeVisible({ visible: state.customer });
    await expect(page.locator('#billing-no-account')).toBeVisible({ visible: !state.customer && state.tier !== 'free' });
    await expect(page.locator('#billing-cancel')).toBeVisible({ visible: !!state.sub && !state.sub.cancel_at_period_end });
    if (state.sub?.cancel_at_period_end) {
      await expect(page.locator('#billing-badge')).toHaveText('Cancels at period end');
      await expect(page.locator('#billing-renewal')).toHaveText(/^Your access ends on /);
    }
    if (state.sub?.status === 'past_due') await expect(page.locator('#billing-badge')).toHaveText('past due');
  });
}

for (const failure of ['profile', 'summary'] as const) {
  test(`pricing blocks a second paid checkout when ${failure} cannot be loaded`, async ({ page }) => {
    const calls = await installBackend(page, {
      tier: 'pro', customer: true, profileFailure: failure === 'profile', summaryFailure: failure === 'summary',
    });
    await page.goto('/pricing/');
    await expect(page.locator('body')).toHaveAttribute('data-plan-loaded', 'true');
    await page.locator('.checkout-btn[data-tier="enterprise"]').click();
    await expect(page).toHaveURL(/\/pricing\/$/);
    await expect(page.getByText('Could not load your current plan, so checkout is blocked', { exact: false })).toBeVisible();
    expect(calls.filter((c) => c.action !== 'summary')).toEqual([]);
  });
}

test('a failed portal request surfaces its error and allows retry', async ({ page }) => {
  const calls = await installBackend(page, { tier: 'pro', customer: true, sub: subscription });
  await page.goto('/account/billing/');
  const manage = page.locator('#manage-subscription-btn');
  await manage.click();
  await expect(page.locator('#billing-error')).toHaveText('Portal temporarily unavailable');
  await expect(manage).toBeEnabled();
  await manage.click();
  await expect.poll(() => calls.filter((c) => c.action === 'manage').length).toBe(2);
  expect(calls.filter((c) => c.action === 'manage')).toEqual([
    { action: 'manage', return_path: '/account/billing/' },
    { action: 'manage', return_path: '/account/billing/' },
  ]);
});

for (const redirect of ['https://attacker.example/phish', '/\\attacker.example/phish', '//attacker.example/phish']) {
  test(`signed-in login rejects an external redirect ${JSON.stringify(redirect)}`, async ({ page, baseURL }) => {
    await installBackend(page);
    await page.route('**/account/', (route) => route.fulfill({ status: 200, contentType: 'text/html', body: '<h1>Account</h1>' }));
    await page.route('**://attacker.example/**', (route) => route.fulfill({ status: 200, body: 'Unsafe redirect' }));
    await page.goto(`/login/?redirect=${encodeURIComponent(redirect)}`);
    await expect(page).toHaveURL(new URL('/account/', baseURL).href);
  });
}
