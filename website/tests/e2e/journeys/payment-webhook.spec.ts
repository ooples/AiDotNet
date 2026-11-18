import { test, expect } from '@playwright/test';
import { randomUUID } from 'node:crypto';
import { journeySkipReason, postSignedStripeEvent, restPatch, restSelect, restInsert } from '../helpers/backend';
import { createPersona, disposePersona, type Persona } from '../helpers/personas';
import { changeSubscriptionPrice, deleteCustomer, planPrice } from '../helpers/stripe';

// Real deployed handler, database and current Stripe subscription. Events are
// signed injections against a disposable card-free trial, never a settled payment.
test('delayed payment fulfills once and a late success cannot undo cancellation', async ({ request }) => {
  const skip = journeySkipReason();
  test.skip(!!skip, skip);
  test.setTimeout(120_000);
  let persona: Persona | undefined;
  try {
    persona = await createPersona(request, 'pro-stripe', { entitlement:false });
    const { customerId, subscriptionId } = persona.stripe!;
    const intent = await restInsert<{id:string}>(request,'billing_checkout_intents',{user_id:persona.id,tier:'pro',interval:'month'});
    const session = {
      id: `cs_e2e_${randomUUID().replace(/-/g, '')}`, object: 'checkout.session',
      mode: 'subscription', status: 'complete', created:Math.floor(Date.now()/1000), client_reference_id: intent.id,
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
    const licenses = () => restSelect<Array<{ id: string; tier: string; status: string; stripe_subscription_id: string }>>(
      request, `license_keys?select=id,tier,status,stripe_subscription_id&user_id=eq.${persona!.id}`);

    await deliver('checkout.session.completed', 'unpaid');
    expect(await licenses(), 'unpaid Checkout cannot issue a license').toEqual([]);
    await deliver('checkout.session.async_payment_succeeded', 'paid');
    const issued = await licenses();
    expect(issued).toHaveLength(1);
    expect(issued[0]).toMatchObject({ status: 'active', stripe_subscription_id: subscriptionId });
    await deliver('checkout.session.async_payment_succeeded', 'paid');
    expect(await licenses(), 'replayed success does not create another key').toEqual(issued);
    expect(await restPatch(request, `profiles?id=eq.${persona.id}`, { stripe_customer_id: null, subscription_tier: 'free' })).toBe(1);
    await deliver('checkout.session.async_payment_succeeded', 'paid');
    const profiles = await restSelect<Array<{ stripe_customer_id: string; subscription_tier: string }>>(
      request, `profiles?select=stripe_customer_id,subscription_tier&id=eq.${persona.id}`);
    expect(profiles).toEqual([{ stripe_customer_id: customerId, subscription_tier: 'pro' }]);
    expect(await licenses(), 'repairing the profile does not replace the key').toEqual(issued);

    const enterprise = await planPrice(request, 'enterprise');
    await changeSubscriptionPrice(request, subscriptionId, enterprise.id);
    await deliver('checkout.session.async_payment_succeeded', 'paid');
    const upgraded = await licenses();
    expect(upgraded).toEqual([{ ...issued[0], tier: 'enterprise' }]);

    // Deleting this owned trial customer cancels its subscription without a charge.
    await deleteCustomer(request, customerId);
    persona.stripe = undefined; // Already removed; finally still deletes the Auth user.
    await deliver('checkout.session.async_payment_succeeded', 'paid');
    expect(await licenses(), 'current cancellation wins over the old paid snapshot').toEqual([
      { ...upgraded[0], status: 'expired' },
    ]);
  } finally {
    await disposePersona(request, persona);
  }
});

test('late paid Checkout creates exactly one expired entitlement when cancellation arrived before any key', async ({request})=>{
  const skip=journeySkipReason(); test.skip(!!skip,skip);
  test.setTimeout(120_000);
  let persona:Persona|undefined;
  try {
    persona=await createPersona(request,'pro-stripe',{entitlement:false});
    const {customerId,subscriptionId}=persona.stripe!;
    const intent=await restInsert<{id:string}>(request,'billing_checkout_intents',{user_id:persona.id,tier:'pro',interval:'month'});
    await deleteCustomer(request,customerId);
    persona.stripe=undefined;
    const licenses=()=>restSelect<Array<{status:string;stripe_subscription_id:string}>>(request,`license_keys?select=status,stripe_subscription_id&user_id=eq.${persona!.id}`);
    expect(await licenses()).toEqual([]);
    const response=await postSignedStripeEvent(request,{id:`evt_e2e_${randomUUID()}`,object:'event',type:'checkout.session.completed',data:{object:{
      id:`cs_e2e_${randomUUID().replace(/-/g,'')}`,object:'checkout.session',created:Math.floor(Date.now()/1000),
      client_reference_id:intent.id,customer:customerId,subscription:subscriptionId,payment_status:'paid',metadata:{tier:'pro'},customer_details:{email:persona.email},
    }}});
    expect(response.status(),await response.text()).toBe(200);
    expect(await licenses()).toEqual([{status:'expired',stripe_subscription_id:subscriptionId}]);
  } finally { await disposePersona(request,persona); }
});
