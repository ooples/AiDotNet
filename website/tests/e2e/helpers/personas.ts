import { expect, type APIRequestContext, type Page } from '@playwright/test';
import { createConfirmedUser, deleteUser, randomPassword, restPatch, restInsert, testUserEmail } from './backend';
import { randomUUID } from 'node:crypto';
import { createTrialSubscription, deleteCustomer, type StripeFixture } from './stripe';

// The kinds of signed-in customer the site has to serve, each a real throwaway account:
//
//   free         a new account, no plan
//   pro-stripe   Pro bought online: a real Stripe customer with a live (trial) Pro subscription
//   pro-granted  Pro set by the team, with no Stripe billing account
//   admin        the standing admin test account (PLAYWRIGHT_ADMIN_*); never created or deleted
//
// Profiles are written with the service role exactly as the stripe-webhook writes them.

export type PersonaKind = 'free' | 'pro-stripe' | 'pro-granted' | 'admin';

export interface Persona {
  kind: PersonaKind;
  email: string;
  password: string;
  id?: string;
  stripe?: StripeFixture;
  licenseId?: string;
}

export async function createPersona(request: APIRequestContext, kind: PersonaKind, options: { entitlement?: boolean } = {}): Promise<Persona> {
  if (kind === 'admin') {
    const email = process.env.PLAYWRIGHT_ADMIN_EMAIL, password = process.env.PLAYWRIGHT_ADMIN_PASSWORD;
    if (!email || !password) throw new Error('PLAYWRIGHT_ADMIN_EMAIL / PLAYWRIGHT_ADMIN_PASSWORD are not set');
    return { kind, email, password };
  }
  const email = testUserEmail(kind);
  const password = randomPassword();
  const user = await createConfirmedUser(request, email, password, `E2E ${kind}`);
  const persona: Persona = { kind, email, password, id: user.id };
  try {
    if (kind === 'pro-stripe') {
      persona.stripe = await createTrialSubscription(request, email, user.id, 'pro');
      await restInsert(request, 'billing_customers', { customer_id: persona.stripe.customerId, user_id: user.id });
      if (options.entitlement !== false) {
        const license = await restInsert<{id:string}>(request,'license_keys',{
          user_id:user.id,license_key:`AIDN-PROD-PRO-${randomUUID().replace(/-/g,'')}`,product:'aidotnet',tier:'professional',status:'active',
          stripe_subscription_id:persona.stripe.subscriptionId,stripe_customer_id:persona.stripe.customerId,max_activations:10,notes:'E2E paid persona entitlement',
        });
        persona.licenseId=license.id;
      }
      await restPatch(request, `profiles?id=eq.${user.id}`, {
        subscription_tier: 'pro', subscription_status: 'active', stripe_customer_id: persona.stripe.customerId,
      });
    } else if (kind === 'pro-granted') {
      await restPatch(request, `profiles?id=eq.${user.id}`, { subscription_tier: 'pro', subscription_status: 'active' });
    }
  } catch (e) {
    try { await disposePersona(request, persona); }
    catch (cleanup) { throw new AggregateError([e,cleanup],'Persona creation and cleanup failed'); }
    throw e;
  }
  return persona;
}

export async function disposePersona(request: APIRequestContext, p: Persona | undefined): Promise<void> {
  if (!p || p.kind === 'admin') return;
  const results=await Promise.allSettled([
    p.stripe ? deleteCustomer(request,p.stripe.customerId) : Promise.resolve(),
    p.id ? deleteUser(request,p.id) : Promise.resolve(),
  ]);
  const failures=results.filter(result=>result.status==='rejected').map(result=>result.reason);
  if(failures.length) throw new AggregateError(failures,'Persona cleanup failed');
}

/** Signs in through the real login form and waits for the dashboard. */
export async function signIn(page: Page, p: Persona): Promise<void> {
  await page.goto('/login/');
  await page.getByLabel('Email address').fill(p.email);
  await page.getByLabel('Password').fill(p.password);
  await page.getByRole('button', { name: 'Sign In' }).click();
  await expect(page).toHaveURL(/\/account\/?$/, { timeout: 20_000 });
  // The URL changes at document commit, before CSS has loaded. Enumerating at
  // that point can mistake unstyled desktop-only controls for visible phone UI.
  await page.waitForLoadState('load');
  // The crawler may immediately visit another route. Let dashboard Auth/profile
  // requests finish first, so that forced navigation does not abort the SDK fetch.
  // Retain console errors: request failures are not explained away by this wait.
  await page.waitForLoadState('networkidle', { timeout: 10_000 }).catch(() => {});
}

/** True while the page's Supabase session is present in localStorage. */
export async function hasSession(page: Page): Promise<boolean> {
  return page.evaluate(() => Object.keys(localStorage).some((k) => /^sb-.*-auth-token$/.test(k)));
}
