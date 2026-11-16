import { test, expect, type Page } from '@playwright/test';
import { randomBytes } from 'node:crypto';
import { journeySkipReason, restInsert, restSelect } from '../helpers/backend';
import { createPersona, disposePersona, signIn, type Persona } from '../helpers/personas';
import { getSubscription, planPrice } from '../helpers/stripe';

// What a signed-in customer can do with their plan, asserted by outcome, against the real backend
// and real Stripe objects (a trial subscription, so nothing is ever charged):
//
//   Pro bought online:  dashboard -> billing shows the live subscription -> Manage Subscription and
//                       View Invoices open a real Stripe portal -> pricing marks Pro as current and
//                       offers Enterprise / Community -> upgrade to Enterprise is confirmed in
//                       Stripe's portal and the site follows -> deleting the account is refused while
//                       the subscription bills -> cancel in Stripe's portal -> the site shows it ending
//                       -> delete the account: licenses are revoked into the CRL and sign-in stops.
//   Pro set by the team: billing says how to change it (no dead portal), pricing marks Pro as current.
//   Free:               billing offers Upgrade, pricing sells every plan.

test.describe.configure({ mode: 'serial' });

const STRIPE_PORTAL = /^https:\/\/billing\.stripe\.com\//;

async function billingLoaded(page: Page) {
  await page.goto('/account/billing/');
  await expect(page.locator('#billing-root')).toHaveAttribute('data-billing-loaded', 'true', { timeout: 20_000 });
  await expect(page.locator('#billing-error')).toBeHidden();
}

async function pricingLoaded(page: Page) {
  await page.goto('/pricing/');
  await expect(page.locator('body')).toHaveAttribute('data-plan-loaded', 'true', { timeout: 20_000 });
}

const card = (page: Page, tier: string) => page.locator(`.checkout-btn[data-tier="${tier}"]`);

/** Clicks a control that opens Stripe's portal and waits for the real portal page. */
async function openStripe(page: Page, click: () => Promise<void>) {
  await Promise.all([page.waitForURL(STRIPE_PORTAL, { timeout: 30_000 }), click()]);
  await page.waitForLoadState('domcontentloaded');
  await expect(page.locator('body')).not.toContainText(/session has expired|something went wrong|page not found/i, { timeout: 15_000 });
}

test.describe('Pro bought online', () => {
  // Stripe's hosted portal plus webhook polls of up to 60 s each, in sequence: the default 30 s
  // test timeout would end a test before its own waits ran out.
  test.describe.configure({ timeout: 5 * 60_000 });
  let p: Persona;
  let licenseId: string;
  test.beforeAll(async ({ request }) => {
    const skip = journeySkipReason();
    test.skip(!!skip, skip);
    p = await createPersona(request, 'pro-stripe');
    // The license the webhook issues at checkout, so deletion can be checked to revoke it.
    const lic = await restInsert<{ id: string }>(request, 'license_keys', {
      user_id: p.id, license_key: `AIDN-PROD-PRO-E2E${randomBytes(14).toString('hex').toUpperCase()}`, tier: 'professional',
      status: 'active', max_activations: 10, stripe_subscription_id: p.stripe!.subscriptionId, stripe_customer_id: p.stripe!.customerId,
      notes: 'e2e persona',
    });
    licenseId = lic.id;
  });
  test.afterAll(async ({ request }) => { await disposePersona(request, p); });

  test('dashboard and billing show the live subscription, and the portal really opens', async ({ page, baseURL }) => {
    await signIn(page, p);
    await expect(page.locator('#plan-badge')).toHaveText('Pro');
    await expect(page.locator('#upgrade-btn')).toHaveText('Manage Plan');
    await page.locator('#upgrade-btn').click();
    await expect(page).toHaveURL(/\/account\/billing\/$/);

    await billingLoaded(page);
    await expect(page.locator('#billing-plan')).toHaveText('Pro');
    await expect(page.locator('#billing-badge')).toHaveText('Trial');
    const dollars = `$${p.stripe!.price.unit_amount / 100}`;
    await expect(page.locator('#billing-price')).toHaveText(`${dollars}/month`);
    await expect(page.locator('#billing-renewal')).toHaveText(/^Trial ends on \w+ \d+, \d{4}\.$/);
    await expect(page.locator('#plan-features')).toContainText('Commercial use authorized');
    await expect(page.locator('#billing-change-plan')).toHaveText('Change plan');
    await expect(page.locator('#billing-cancel')).toBeVisible();
    await expect(page.locator('#billing-no-account')).toBeHidden();

    for (const id of ['#manage-subscription-btn', '#view-invoices-btn']) {
      await openStripe(page, () => page.locator(id).click());
      await expect(page.locator('body')).toContainText(dollars, { timeout: 15_000 });
      await page.goto(`${baseURL}/account/billing/`);
      await expect(page.locator('#billing-root')).toHaveAttribute('data-billing-loaded', 'true', { timeout: 20_000 });
    }

    await page.locator('#billing-change-plan').click();
    await expect(page).toHaveURL(/\/pricing\/$/);
  });

  test('pricing knows the plan and offers upgrade, downgrade and billing-period changes', async ({ page, request }) => {
    await signIn(page, p);
    await pricingLoaded(page);
    await expect(page.locator('#current-plan-banner')).toBeVisible();
    await expect(page.locator('#current-plan-text')).toHaveText(/^You are on the Pro plan, billed monthly\. Renews on /);
    await expect(card(page, 'pro')).toHaveText('Current plan');
    await expect(card(page, 'pro')).toBeDisabled();
    await expect(card(page, 'enterprise')).toHaveText('Upgrade to Enterprise');
    await expect(card(page, 'community')).toHaveText('Downgrade to Community');
    await expect(page.locator('[data-plan-status="community"]')).toHaveText(/Cancels Professional/);

    const yearly = await planPrice(request, 'pro', 'year').catch(() => null);
    if (yearly) {
      await page.locator('#billing-toggle').click();
      await expect(card(page, 'pro')).toHaveText('Switch to yearly billing');
      await expect(card(page, 'pro')).toBeEnabled();
      await page.locator('#billing-toggle').click();
      await expect(card(page, 'pro')).toHaveText('Current plan');
    }
  });

  test('upgrading to Enterprise is confirmed in Stripe and the site follows', async ({ page, request, baseURL }) => {
    await signIn(page, p);
    await pricingLoaded(page);
    const enterprise = await planPrice(request, 'enterprise');
    await openStripe(page, () => card(page, 'enterprise').click());
    await expect(page.locator('body')).toContainText(`$${enterprise.unit_amount / 100}`, { timeout: 15_000 });
    await page.getByRole('button', { name: /^confirm/i }).click();
    await page.waitForURL((u) => u.href.startsWith(`${baseURL}/pricing/`), { timeout: 60_000 });

    await expect.poll(async () => (await getSubscription(request, p.stripe!.subscriptionId)).items.data[0].price.id, { timeout: 30_000 })
      .toBe(enterprise.id);
    // The webhook mirrors the plan onto the profile and the license.
    await expect.poll(async () => (await restSelect<Array<{ subscription_tier: string }>>(request, `profiles?select=subscription_tier&id=eq.${p.id}`))[0]?.subscription_tier, { timeout: 60_000 })
      .toBe('enterprise');
    await expect.poll(async () => (await restSelect<Array<{ tier: string }>>(request, `license_keys?select=tier&id=eq.${licenseId}`))[0]?.tier, { timeout: 60_000 })
      .toBe('enterprise');

    await pricingLoaded(page);
    await expect(card(page, 'enterprise')).toHaveText('Current plan');
    await expect(card(page, 'pro')).toHaveText('Downgrade to Professional');
    await billingLoaded(page);
    await expect(page.locator('#billing-plan')).toHaveText('Enterprise');
    await expect(page.locator('#billing-price')).toHaveText(`$${enterprise.unit_amount / 100}/month`);
  });

  test('account deletion is refused while the subscription still bills', async ({ page, request }) => {
    await signIn(page, p);
    await page.goto('/account/settings/');
    await page.locator('#delete-account-btn').click();
    await expect(page.locator('#delete-account-modal')).toBeVisible();
    // Keyboard stays inside the dialog, Escape closes it, and focus goes back to the opener.
    for (let i = 0; i < 6; i++) {
      await page.keyboard.press(i % 2 ? 'Tab' : 'Shift+Tab');
      expect(await page.evaluate(() => !!document.activeElement?.closest('#delete-account-modal')), 'focus left the dialog').toBe(true);
    }
    await page.keyboard.press('Escape');
    await expect(page.locator('#delete-account-modal')).toBeHidden();
    await expect(page.locator('#delete-account-btn')).toBeFocused();
    await page.locator('#delete-account-btn').click();
    await expect(page.locator('#delete-account-modal')).toBeVisible();
    await expect(page.locator('#delete-confirm-btn')).toBeDisabled();
    await page.locator('#delete-confirm-input').fill('delete');
    await expect(page.locator('#delete-confirm-btn')).toBeDisabled();
    await page.locator('#delete-confirm-input').fill('DELETE');
    await page.locator('#delete-confirm-btn').click();
    await expect(page.locator('#delete-account-error')).toHaveText(/Cancel your subscription on the Billing page first/);
    await page.locator('#delete-cancel-btn').click();
    await expect(page.locator('#delete-account-modal')).toBeHidden();
    expect(await restSelect<unknown[]>(request, `profiles?select=id&id=eq.${p.id}`)).toHaveLength(1);
  });

  test('cancelling in Stripe shows the plan ending, everywhere', async ({ page, request, baseURL }) => {
    await signIn(page, p);
    await billingLoaded(page);
    await openStripe(page, () => page.locator('#billing-cancel').click());
    await page.getByRole('button', { name: /cancel (subscription|plan)/i }).last().click();
    await page.waitForURL((u) => u.href.startsWith(`${baseURL}/account/billing/`), { timeout: 60_000 });
    await expect.poll(async () => (await getSubscription(request, p.stripe!.subscriptionId)).cancel_at_period_end, { timeout: 30_000 }).toBe(true);

    await billingLoaded(page);
    await expect(page.locator('#billing-badge')).toHaveText('Cancels at period end');
    await expect(page.locator('#billing-renewal')).toHaveText(/^Your access ends on /);
    await expect(page.locator('#billing-cancel')).toBeHidden();
    await pricingLoaded(page);
    await expect(page.locator('#current-plan-text')).toHaveText(/It ends on /);
    await expect(page.locator('[data-plan-status="enterprise"]')).toHaveText('Ends at the close of this billing period');
  });

  test('deleting the account revokes its licenses into the CRL and ends sign-in', async ({ page, request }) => {
    await signIn(page, p);
    await page.goto('/account/settings/');
    await page.locator('#delete-account-btn').click();
    await page.locator('#delete-confirm-input').fill('DELETE');
    await page.locator('#delete-confirm-btn').click();
    await expect(page).toHaveURL(/\/\?account_deleted=1$/, { timeout: 30_000 });

    expect(await restSelect<unknown[]>(request, `profiles?select=id&id=eq.${p.id}`)).toHaveLength(0);
    const crl = await restSelect<unknown[]>(request, `revocations?select=id&jti=eq.${licenseId}`);
    expect(crl.length, 'the deleted account\'s license is in the revocation list').toBeGreaterThan(0);

    await page.goto('/login/');
    await page.getByLabel('Email address').fill(p.email);
    await page.getByLabel('Password').fill(p.password);
    await page.getByRole('button', { name: 'Sign In' }).click();
    await expect(page.getByText(/invalid login credentials|invalid email or password/i)).toBeVisible({ timeout: 15_000 });
    p.id = undefined; // already deleted; the Stripe customer is still removed in afterAll
  });
});

test.describe('Pro set up by the team (no Stripe account)', () => {
  let p: Persona;
  test.beforeAll(async ({ request }) => {
    const skip = journeySkipReason();
    test.skip(!!skip, skip);
    p = await createPersona(request, 'pro-granted');
  });
  test.afterAll(async ({ request }) => { await disposePersona(request, p); });

  test('billing explains how to change the plan instead of offering dead portal links', async ({ page }) => {
    await signIn(page, p);
    await billingLoaded(page);
    await expect(page.locator('#billing-plan')).toHaveText('Pro');
    await expect(page.locator('#billing-badge')).toHaveText('Active');
    await expect(page.locator('#billing-price')).toHaveText('$29/month per seat');
    await expect(page.locator('#portal-section')).toBeHidden();
    await expect(page.locator('#billing-cancel')).toBeHidden();
    await expect(page.locator('#billing-no-account')).toBeVisible();
    await expect(page.locator('#billing-no-account a')).toHaveAttribute('href', 'mailto:support@aidotnet.dev');
  });

  test('pricing marks Pro current and sells the Enterprise upgrade through checkout', async ({ page }) => {
    await signIn(page, p);
    await pricingLoaded(page);
    await expect(page.locator('#current-plan-text')).toHaveText('You are on the Pro plan.');
    await expect(card(page, 'pro')).toHaveText('Current plan');
    await expect(card(page, 'pro')).toBeDisabled();
    await expect(card(page, 'enterprise')).toHaveText('Upgrade to Enterprise');
    let checkout: URL | undefined;
    await page.route(/buy\.stripe\.com|checkout\.stripe\.com/, async (route) => {
      checkout = new URL(route.request().url());
      await route.fulfill({ status: 200, contentType: 'text/html', body: '<p>checkout (stubbed)</p>' });
    });
    await card(page, 'enterprise').click();
    await expect.poll(() => checkout?.href, { timeout: 15_000 }).toBeTruthy();
    expect(checkout!.searchParams.get('client_reference_id')).toBe(p.id);
    expect(checkout!.searchParams.get('prefilled_email')).toBe(p.email);
  });
});

test.describe('Free account', () => {
  let p: Persona;
  test.beforeAll(async ({ request }) => {
    const skip = journeySkipReason();
    test.skip(!!skip, skip);
    p = await createPersona(request, 'free');
  });
  test.afterAll(async ({ request }) => { await disposePersona(request, p); });

  test('billing offers the upgrade and pricing sells every plan', async ({ page }) => {
    await signIn(page, p);
    await expect(page.locator('#upgrade-btn')).toHaveAttribute('href', '/pricing/');
    await billingLoaded(page);
    await expect(page.locator('#billing-plan')).toHaveText('Free');
    await expect(page.locator('#billing-price')).toHaveText('$0/month');
    await expect(page.locator('#portal-section')).toBeHidden();
    await expect(page.locator('#billing-no-account')).toBeHidden();
    await expect(page.locator('#billing-change-plan')).toHaveText('Upgrade');
    await page.locator('#billing-change-plan').click();
    await expect(page).toHaveURL(/\/pricing\/$/);
    await expect(page.locator('body')).toHaveAttribute('data-plan-loaded', 'true', { timeout: 20_000 });
    await expect(page.locator('#current-plan-banner')).toBeHidden();
    await expect(card(page, 'pro')).toHaveText('Subscribe');
    await expect(card(page, 'enterprise')).toHaveText('Subscribe');
    await expect(card(page, 'community')).toHaveText('Get Free License');
  });
});
