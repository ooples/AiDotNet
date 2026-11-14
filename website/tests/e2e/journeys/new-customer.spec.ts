import { test, expect, type Page } from '@playwright/test';
import {
  journeySkipReason, testUserEmail, randomPassword, findUserIdByEmail, deleteUser, generateLink,
  postSignedStripeEvent, restSelect,
} from '../helpers/backend';

// A brand-new customer's whole path through the site, against the real backend, in one browser:
//
//   sign up -> confirm the email link -> dashboard -> change password -> buy Pro (Stripe redirect
//   + a correctly signed checkout.session.completed) -> license shows up and copies -> billing
//   shows Pro -> sign out -> wrong password is refused -> sign in -> forgot password -> reset
//   link -> new password -> sign in.
//
// Emails: the address is Resend's test inbox, so Supabase and the webhook really send their mail
// and nobody receives it. The links in those emails come from the admin generate_link API, which
// produces the same link without reading a mailbox.
//
// Payment: production Stripe is live, so the test stops at the Checkout redirect (asserting the
// user id and email ride along) and then delivers the event Stripe would send after payment,
// signed with the real webhook secret. Everything after that (license issuance, profile tier,
// the license email) is the production code path.
//
// The user is deleted afterwards (licenses cascade); the nightly job deletes any leftovers.

const email = testUserEmail('journey');
const fullName = `E2E Journey ${Date.now()}`;
const passwords = [randomPassword(), randomPassword(), randomPassword()];
let userId: string | undefined;

test.describe.configure({ mode: 'serial' });

async function signIn(page: Page, password: string) {
  await page.goto('/login/');
  await page.getByLabel('Email address').fill(email);
  await page.getByLabel('Password').fill(password);
  await page.getByRole('button', { name: 'Sign In' }).click();
}

async function signOut(page: Page, via: 'desktop' | 'mobile' = 'desktop') {
  if (via === 'mobile') await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/account/');
  if (via === 'mobile') {
    await page.locator('#mobile-menu-btn').click();
    await page.locator('#mobile-sign-out').click();
  } else {
    await page.locator('#user-menu-btn').click();
    await page.locator('#nav-sign-out').click();
  }
  await page.waitForURL((u) => !u.pathname.startsWith('/account'));
  // A signed-out visitor is sent from /account/ to /login/.
  await page.goto('/account/');
  await expect(page).toHaveURL(/\/login\//, { timeout: 15_000 });
  if (via === 'mobile') await page.setViewportSize({ width: 1440, height: 900 });
}

test.afterAll(async ({ request }) => {
  if (userId) await deleteUser(request, userId);
});

test('a new customer signs up, buys Pro, manages their account and recovers their password', async ({ page, request, context, baseURL }) => {
  const skip = journeySkipReason();
  test.skip(!!skip, skip);
  test.setTimeout(240_000);
  const origin = new URL(baseURL!).origin;
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  const alerts: string[] = [];
  page.on('dialog', async (d) => { alerts.push(`${d.type()}: ${d.message()}`); await d.accept(); });

  await test.step('sign up', async () => {
    await page.goto('/signup/');
    await page.locator('#full-name').fill(fullName);
    await page.locator('#email').fill(email);
    await page.locator('#password').fill(passwords[0]);
    await page.locator('#signup-form button[type="submit"]').click();
    await expect(page.locator('#signup-success')).toBeVisible({ timeout: 15_000 });
    await expect(page.locator('#signup-error')).toBeHidden();
    userId = await findUserIdByEmail(request, email);
  });

  await test.step('confirm the email link and land on the dashboard', async () => {
    // The same link the signup confirmation email carries.
    const link = await generateLink(request, 'signup', email, `${origin}/auth/callback/`, passwords[0]);
    await page.goto(link);
    await expect(page).toHaveURL(/\/account\/?$/, { timeout: 20_000 });
    await expect(page.getByText(fullName.split(' ')[0], { exact: false }).first()).toBeVisible();
  });

  await test.step('change password in settings', async () => {
    await page.goto('/account/settings/');
    await expect(page.locator('#password-section')).toBeVisible({ timeout: 15_000 });
    await page.locator('#new-password').fill(passwords[1]);
    await page.locator('#confirm-password').fill(passwords[1]);
    await page.locator('#password-form button[type="submit"]').click();
    await expect(page.locator('#password-success')).toBeVisible({ timeout: 15_000 });
    await expect(page.locator('#password-error')).toBeHidden();
  });

  let licenseKey = '';
  await test.step('buy Pro: Checkout redirect carries the user, the signed webhook provisions a license', async () => {
    await page.goto('/pricing/');
    let checkoutUrl: URL | undefined;
    await page.route(/^https:\/\/(buy|checkout)\.stripe\.com\//, async (route) => {
      checkoutUrl = new URL(route.request().url());
      await route.fulfill({ status: 200, contentType: 'text/html', body: '<p>stripe checkout (stubbed by test)</p>' });
    });
    await page.locator('.checkout-btn[data-tier="pro"]').first().click();
    await expect.poll(() => checkoutUrl?.href, { timeout: 15_000 }).toBeTruthy();
    expect(checkoutUrl!.searchParams.get('client_reference_id')).toBe(userId);
    expect(checkoutUrl!.searchParams.get('prefilled_email')).toBe(email);
    await page.unroute(/^https:\/\/(buy|checkout)\.stripe\.com\//);

    const res = await postSignedStripeEvent(request, {
      id: `evt_e2e_${Date.now()}`,
      object: 'event',
      api_version: '2024-06-20',
      created: Math.floor(Date.now() / 1000),
      type: 'checkout.session.completed',
      livemode: true,
      data: {
        object: {
          id: `cs_e2e_${Date.now()}`,
          object: 'checkout.session',
          mode: 'subscription',
          status: 'complete',
          payment_status: 'paid',
          client_reference_id: userId,
          customer: null,
          subscription: null,
          customer_details: { email },
          metadata: { tier: 'pro' },
        },
      },
    });
    expect(res.status(), `stripe-webhook: ${await res.text()}`).toBe(200);

    const rows = await restSelect<Array<{ license_key: string; tier: string; status: string }>>(
      request, `license_keys?select=license_key,tier,status&user_id=eq.${userId}`);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ tier: 'professional', status: 'active' });
    licenseKey = rows[0].license_key;
  });

  await test.step('the license appears on /account/licenses and copies in full', async () => {
    await context.grantPermissions(['clipboard-read', 'clipboard-write'], { origin });
    await page.goto('/account/licenses/');
    const copy = page.locator('.copy-key-btn').first();
    await expect(copy).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(/professional/i).first()).toBeVisible();
    await copy.click();
    await expect.poll(() => page.evaluate(() => navigator.clipboard.readText())).toBe(licenseKey);
  });

  await test.step('billing shows the Pro plan', async () => {
    await page.goto('/account/billing/');
    await expect(page.locator('#billing-plan')).toHaveText('Pro', { timeout: 15_000 });
  });

  await test.step('sign out, refuse a wrong password, sign in with the changed one', async () => {
    await signOut(page);
    await signIn(page, passwords[0]); // the pre-change password must no longer work
    await expect(page.getByText(/invalid login credentials/i)).toBeVisible({ timeout: 15_000 });
    await expect(page).toHaveURL(/\/login\//);
    await signIn(page, passwords[1]);
    await expect(page).toHaveURL(/\/account\/?$/, { timeout: 15_000 });
  });

  await test.step('forgot password -> reset link -> new password -> sign in', async () => {
    await signOut(page, 'mobile');
    await page.goto('/login/forgot-password/');
    await page.locator('#email').fill(email);
    await page.locator('#reset-btn').click();
    await expect(page.locator('#reset-success')).toBeVisible({ timeout: 15_000 });
    await expect(page.locator('#reset-error')).toBeHidden();

    const link = await generateLink(request, 'recovery', email, `${origin}/login/reset-password/`);
    await page.goto(link);
    await expect(page).toHaveURL(/\/login\/reset-password\//, { timeout: 20_000 });
    await page.locator('#password').fill(passwords[2]);
    await page.locator('#confirm-password').fill(passwords[2]);
    await page.locator('#pw-btn').click();
    await expect(page.locator('#pw-success')).toBeVisible({ timeout: 15_000 });
    await expect(page).toHaveURL(/\/login\/?$/, { timeout: 15_000 });
    await signIn(page, passwords[2]);
    await expect(page).toHaveURL(/\/account\/?$/, { timeout: 15_000 });
  });

  expect(errors, 'no uncaught page errors along the journey').toEqual([]);
  expect(alerts, 'no alerts along the journey').toEqual([]);
});
