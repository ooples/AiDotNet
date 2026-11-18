import { test, expect } from '@playwright/test';
import { randomUUID } from 'node:crypto';
import { journeySkipReason, postSignedStripeEvent, restSelect } from '../helpers/backend';
import { createPersona, disposePersona, type Persona } from '../helpers/personas';
import { deleteCustomer } from '../helpers/stripe';

// Real deployed handler, database and current Stripe subscription. Events are
// signed injections against a disposable card-free trial, never a settled payment.
test('delayed payment fulfills once and a late success cannot undo cancellation', async ({ request }) => {
  const skip = journeySkipReason();
  test.skip(!!skip, skip);
  test.setTimeout(120_000);
  let persona: Persona | undefined;
  try {
    persona = await createPersona(request, 'pro-stripe');
    const { customerId, subscriptionId } = persona.stripe!;
    const session = {
      id: `cs_e2e_${randomUUID().replace(/-/g, '')}`, object: 'checkout.session',
      mode: 'subscription', status: 'complete', client_reference_id: persona.id,
      customer: customerId, subscription: subscriptionId,
      customer_details: { email: persona.email }, metadata: { tier: 'pro' },
    };
    const deliver = async (type: string, payment_status: string) => {
      const response = await postSignedStripeEvent(request, {
        id: `evt_e2e_${randomUUID()}`, object: 'event', type,
        data: { object: { ...session, payment_status } },
      });
      expect(response.status(), `${type}: ${await response.text()}`).toBe(200);
    };
    const licenses = () => restSelect<Array<{ id: string; status: string; stripe_subscription_id: string }>>(
      request, `license_keys?select=id,status,stripe_subscription_id&user_id=eq.${persona!.id}`);

    await deliver('checkout.session.completed', 'unpaid');
    expect(await licenses(), 'unpaid Checkout cannot issue a license').toEqual([]);
    await deliver('checkout.session.async_payment_succeeded', 'paid');
    const issued = await licenses();
    expect(issued).toHaveLength(1);
    expect(issued[0]).toMatchObject({ status: 'active', stripe_subscription_id: subscriptionId });
    await deliver('checkout.session.async_payment_succeeded', 'paid');
    expect(await licenses(), 'replayed success does not create another key').toEqual(issued);

    // Deleting this owned trial customer cancels its subscription without a charge.
    await deleteCustomer(request, customerId);
    persona.stripe = undefined; // Already removed; finally still deletes the Auth user.
    await deliver('checkout.session.async_payment_succeeded', 'paid');
    expect(await licenses(), 'current cancellation wins over the old paid snapshot').toEqual([
      { ...issued[0], status: 'expired' },
    ]);
  } finally {
    await disposePersona(request, persona);
  }
});
