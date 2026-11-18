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
  id: 'sub_preview',
  status: 'active', tier: 'pro', interval: 'month', unit_amount: 2900, currency: 'usd', quantity: 2,
  current_period_end: 1893456000, cancel_at_period_end: false, cancel_at: null as number | null, trial_end: null,
};

async function installBackend(page: Page, options: {
  tier?: string; customer?: boolean; sub?: Record<string, unknown> | null; profileFailure?: boolean; summaryFailure?: boolean; rescheduleFailure?: boolean; rescheduleConflict?: boolean;
  summaryAfterAction?: () => Promise<void>; summaryFailureAfterAction?: boolean;
  rescheduleAppliedFailure?: boolean; ordinaryCancelConflict?: boolean;
  licenses?: Array<Record<string,unknown>>;
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
  let currentSub = options.sub ?? null;
  let rescheduleAttempts = 0;
  await page.route('https://billing.stripe.com/**', route => route.fulfill({status:200,contentType:'text/html',body:'<h1>Stripe cancellation review</h1>'}));
  await page.route('https://preview-test.supabase.co/**', async (route) => {
    const path = new URL(route.request().url()).pathname;
    const reply = (body: unknown, status = 200) => route.fulfill({ status, json: body });
    if (path === '/auth/v1/user') return reply(user);
    if (path === '/rest/v1/license_keys') return reply(options.licenses ?? []);
    if (path === '/rest/v1/license_activations') return reply([]);
    if (path === '/rest/v1/profiles') {
      if (options.profileFailure) return reply({ message: 'profile unavailable' }, 503);
      const single = route.request().headers().accept?.includes('pgrst.object');
      return reply(single ? profile : [profile]);
    }
    if (path === '/functions/v1/billing-portal') {
      const body = route.request().postDataJSON();
      calls.push(body);
      if (body.action === 'checkout_reference') return reply({reference:'22222222-2222-4222-8222-222222222222'});
      if (body.action === 'claim_checkout') return reply({message:'Checkout belongs to another account'},403);
      if (body.action === 'summary') {
        if (rescheduleAttempts) {
          await options.summaryAfterAction?.();
          if (options.summaryFailureAfterAction) return reply({ message: 'Refresh unavailable' }, 502);
        }
        return options.summaryFailure ? reply({ message: 'Stripe unavailable' }, 502) : reply({ subscription: currentSub });
      }
      if (body.action === 'cancel_before_renewal') {
        rescheduleAttempts++;
        if (options.rescheduleAppliedFailure) {
          currentSub={...currentSub,cancel_at:currentSub!.current_period_end};
          return reply({message:'Action response unavailable'},502);
        }
        if (options.rescheduleConflict && body.expected_renewal === subscription.current_period_end) {
          currentSub = { ...currentSub, current_period_end: subscription.current_period_end + 86400 };
          return reply({ message: 'Your renewal date changed. Confirm again.' }, 409);
        }
        if (options.rescheduleFailure) return reply({ message: 'Cancellation could not be changed' }, 502);
        return reply({url:'https://billing.stripe.com/review'});
      }
      if (body.action === 'cancel' && options.ordinaryCancelConflict) {
        rescheduleAttempts++;
        currentSub={...currentSub,cancel_at:subscription.current_period_end+172800};
        return reply({error:'already_cancelling',message:'A cancellation was scheduled in another session.'},409);
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
  { name: 'date-based cancellation', tier: 'pro', customer: true, sub: { ...subscription, cancel_at: subscription.current_period_end }, label: 'Pro', price: '$58/month (2 seats)' },
  { name: 'custom-date cancellation', tier: 'pro', customer: true, sub: { ...subscription, cancel_at: subscription.current_period_end - 86400 }, label: 'Pro', price: '$58/month (2 seats)' },
  { name: 'cancellation after another renewal', tier: 'pro', customer: true, sub: { ...subscription, cancel_at: subscription.current_period_end + 86400 }, label: 'Pro', price: '$58/month (2 seats)' },
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
    await expect(page.locator('#billing-cancel')).toBeVisible({ visible: !!state.sub && ((!state.sub.cancel_at_period_end && !state.sub.cancel_at) || !!state.sub.cancel_at && state.sub.cancel_at > state.sub.current_period_end) });
    if (state.sub?.cancel_at_period_end || state.sub?.cancel_at) {
      await expect(page.locator('#billing-badge')).toHaveText(state.sub.cancel_at && state.sub.cancel_at !== state.sub.current_period_end ? 'Cancellation scheduled' : 'Cancels at period end');
      await expect(page.locator('#billing-renewal')).toHaveText(/^Your access ends on /);
    }
    if (state.sub?.status === 'past_due') await expect(page.locator('#billing-badge')).toHaveText('past due');
  });
}

test('a stale renewal confirmation refreshes the date before a second confirmation', async ({ page }) => {
  let release!: () => void, started!: () => void;
  const refreshStarted = new Promise<void>(resolve => { started = resolve; });
  const refreshGate = new Promise<void>(resolve => { release = resolve; });
  const calls = await installBackend(page, { tier: 'pro', customer: true, sub: { ...subscription, cancel_at: subscription.current_period_end + 172800 }, rescheduleConflict: true,
    summaryAfterAction: async () => { started(); await refreshGate; },
  });
  const confirmations: string[] = [];
  page.on('dialog', async d => { confirmations.push(d.message()); await d.accept(); });
  await page.goto('/account/billing/');
  const cancel = page.getByRole('button', { name: 'Review cancellation in Stripe' });
  await cancel.click();
  try {
    await refreshStarted;
    await expect(cancel).toBeDisabled();
  } finally { release(); }
  await expect(page.locator('#billing-error')).toHaveText('Your renewal date changed. Confirm again.');
  await expect(cancel).toBeEnabled();
  await cancel.click();
  await expect(page).toHaveURL('https://billing.stripe.com/review');
  expect(calls.filter(c => c.action === 'cancel_before_renewal').map(c => c.expected_renewal)).toEqual([subscription.current_period_end, subscription.current_period_end + 86400]);
  const dates = await page.evaluate(seconds => seconds.map(s => new Date(s * 1000).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })), [subscription.current_period_end, subscription.current_period_end + 86400, subscription.current_period_end + 172800]);
  expect(confirmations[0]).toContain(`scheduled to end on ${dates[2]}`);
  expect(confirmations[1]).toContain(`scheduled to end on ${dates[2]}`);
  expect(confirmations[0]).toContain(`End it on ${dates[0]} instead`);
  expect(confirmations[1]).toContain(`End it on ${dates[1]} instead`);
  expect(calls.filter(c => c.action === 'cancel_before_renewal').every(c => c.expected_cancel_at === subscription.current_period_end + 172800)).toBe(true);
});

test('a lost cancellation response is successful when the refreshed subscription confirms it', async ({ page }) => {
  const calls = await installBackend(page, {tier:'pro',customer:true,sub:{...subscription,cancel_at:subscription.current_period_end+172800},rescheduleAppliedFailure:true});
  page.on('dialog', d => d.accept());
  await page.goto('/account/billing/');
  await page.getByRole('button',{name:'Review cancellation in Stripe'}).click();
  await expect(page.locator('#billing-badge')).toHaveText('Cancels at period end');
  await expect(page.locator('#billing-error')).toBeHidden();
  await expect(page.locator('#billing-cancel')).toBeHidden();
  expect(calls.filter(c=>c.action==='cancel_before_renewal')).toHaveLength(1);
});

test('an ordinary cancellation conflict refreshes the scheduled action before enabling it', async ({ page }) => {
  let release!:()=>void, started!:()=>void;
  const gate=new Promise<void>(r=>release=r), refresh=new Promise<void>(r=>started=r);
  const calls=await installBackend(page,{tier:'pro',customer:true,sub:subscription,ordinaryCancelConflict:true,summaryAfterAction:async()=>{started();await gate;}});
  page.on('dialog',d=>d.accept());
  await page.goto('/account/billing/');
  await page.locator('#billing-cancel').click();
  try { await refresh; await expect(page.locator('#billing-cancel')).toBeDisabled(); }
  finally { release(); }
  await expect(page.getByRole('button',{name:'Review cancellation in Stripe'})).toBeEnabled();
  await expect(page.locator('#billing-badge')).toHaveText('Cancellation scheduled');
  expect(calls.filter(c=>c.action==='cancel')).toHaveLength(1);
});

test('a failed refresh disables further stale cancellation confirmations', async ({ page }) => {
  const calls = await installBackend(page, { tier: 'pro', customer: true, sub: { ...subscription, cancel_at: subscription.current_period_end + 172800 }, rescheduleConflict: true, summaryFailureAfterAction: true });
  let dialogs = 0;
  page.on('dialog', async d => { dialogs++; await d.accept(); });
  await page.goto('/account/billing/');
  await page.getByRole('button', { name: 'Review cancellation in Stripe' }).click();
  await expect(page.locator('#billing-error')).toContainText('Reload the page');
  await expect(page.locator('#billing-cancel')).toBeDisabled();
  await page.locator('#billing-cancel').evaluate(button => (button as HTMLButtonElement).click());
  expect(dialogs).toBe(1);
  expect(calls.filter(c => c.action === 'cancel_before_renewal')).toHaveLength(1);
});

test('a later cancellation opens Stripe review only after explicit confirmation', async ({ page }) => {
  const calls = await installBackend(page, { tier: 'pro', customer: true, sub: { ...subscription, cancel_at: subscription.current_period_end + 86400 } });
  let accept = false;
  page.on('dialog', d => accept ? d.accept() : d.dismiss());
  await page.goto('/account/billing/');
  const cancel = page.getByRole('button', { name: 'Review cancellation in Stripe' });
  await cancel.click();
  expect(calls.filter(c => c.action === 'cancel_before_renewal')).toHaveLength(0);
  await expect(cancel).toBeEnabled();
  accept = true;
  await cancel.click();
  await expect(page).toHaveURL('https://billing.stripe.com/review');
  expect(calls.filter(c => c.action === 'cancel_before_renewal')).toHaveLength(1);
  expect(calls.find(c => c.action === 'cancel_before_renewal')).toMatchObject({ expected_subscription_id: subscription.id, expected_renewal: subscription.current_period_end, expected_cancel_at: subscription.current_period_end + 86400 });
});

test('a failed cancellation change preserves the existing end and permits retry', async ({ page }) => {
  const calls = await installBackend(page, { tier: 'pro', customer: true, sub: { ...subscription, cancel_at: subscription.current_period_end + 86400 }, rescheduleFailure: true });
  page.on('dialog', d => d.accept());
  await page.goto('/account/billing/');
  const cancel = page.getByRole('button', { name: 'Review cancellation in Stripe' });
  await cancel.click();
  await expect(page.locator('#billing-error')).toHaveText('Cancellation could not be changed');
  await expect(cancel).toBeEnabled();
  await cancel.click();
  await expect.poll(() => calls.filter(c => c.action === 'cancel_before_renewal').length).toBe(2);
});

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

for (const validOlder of [true,false]) {
  test(`license setup ${validOlder ? 'uses an older valid key' : 'is hidden'} when the newest active key has expired`, async ({page}) => {
    const license={id:'expired-key',license_key:'AIDN-EXPIRED-FIXTURE-KEY',tier:'professional',status:'active',max_activations:2,organization_name:null,issued_at:'2025-01-01',expires_at:'2025-02-01'};
    await installBackend(page,{licenses:[license,...(validOlder?[{...license,id:'valid-key',license_key:'AIDN-VALID-FIXTURE-KEY',expires_at:null}]:[])]});
    await page.goto('/account/licenses/');
    await expect(page.locator('[data-license-status]')).toHaveCount(validOlder?2:1);
    await expect(page.locator('#setup-section')).toBeVisible({visible:validOlder});
    await expect(page.locator('#env-var-code')).toHaveText(validOlder?'AIDOTNET_LICENSE_KEY=AIDN-VALID-FIXTURE-KEY':'');
    await expect(page.locator('[data-license-status]').first()).toHaveAttribute('data-usable','false');
  });
}

test('explicit Checkout claim validates the reference and surfaces ownership errors',async({page})=>{
  const calls=await installBackend(page);
  await page.goto('/account/licenses/');
  await page.getByText('Paid before signing in? Claim your checkout',{exact:true}).click();
  await page.locator('#claim-checkout-reference').fill('not-a-checkout');
  await page.locator('#claim-checkout-btn').click();
  await expect(page.locator('#claim-checkout-message')).toContainText('beginning with cs_');
  expect(calls.filter(c=>c.action==='claim_checkout')).toHaveLength(0);
  await page.locator('#claim-checkout-reference').fill('https://checkout.stripe.com/c/pay/cs_test_claim');
  await page.locator('#claim-checkout-btn').click();
  await expect(page.locator('#claim-checkout-message')).toHaveText('Checkout belongs to another account');
  await expect(page.locator('#claim-checkout-btn')).toBeEnabled();
  expect(calls.filter(c=>c.action==='claim_checkout')).toEqual([{action:'claim_checkout',checkout_session_id:'cs_test_claim'}]);
});

for (const redirect of ['https://attacker.example/phish', '/\\attacker.example/phish', '//attacker.example/phish', '/a/..//attacker.example/phish']) {
  test(`signed-in login rejects an external redirect ${JSON.stringify(redirect)}`, async ({ page, baseURL }) => {
    await installBackend(page);
    await page.route('**/account/', (route) => route.fulfill({ status: 200, contentType: 'text/html', body: '<h1>Account</h1>' }));
    await page.route('**://attacker.example/**', (route) => route.fulfill({ status: 200, body: 'Unsafe redirect' }));
    await page.goto(`/login/?redirect=${encodeURIComponent(redirect)}`);
    await expect(page).toHaveURL(new URL('/account/', baseURL).href);
  });
}
