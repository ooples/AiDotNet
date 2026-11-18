import { test, expect, type Locator } from '@playwright/test';
import { journeySkipReason, restSelect } from '../helpers/backend';
import { createPersona, disposePersona, signIn, type Persona } from '../helpers/personas';
import { createTrialCheckout, deleteCustomer } from '../helpers/stripe';
import { withCleanup } from '../helpers/cleanup';

test('a verified customer claims a completed legacy checkout and replay keeps one key', async ({page,request,baseURL}) => {
  const skip=journeySkipReason(); test.skip(!!skip,skip);
  test.setTimeout(240_000);
  let persona:Persona|undefined, customerId:string|undefined;
  await withCleanup(async()=>{
    persona=await createPersona(request,'free');
    const checkout=await createTrialCheckout(request,persona.email,persona.id!,new URL(baseURL!).origin);
    customerId=checkout.customerId;
    await signIn(page,persona);
    await page.goto(checkout.url);
    // No card or other payment method is supplied; the trial cancels if one is still missing.
    // Hosted Checkout renders the contact/confirmation form in a Stripe iframe.
    // Locate the actual accessible action there, without bypassing provider UI.
    let startTrial:Locator|undefined;
    await expect.poll(async()=>{
      for(const frame of page.frames()) {
        if(!/^https:\/\/(?:[a-z0-9-]+\.)*stripe\.com\//i.test(frame.url())) continue;
        const button=frame.getByRole('button',{name:/^(start.*trial|subscribe)$/i});
        if(await button.count()===1&&await button.isVisible()) {startTrial=button;return true;}
      }
      return false;
    },{timeout:30_000,message:'Stripe exposes the real trial confirmation action'}).toBe(true);
    await startTrial!.click();
    await page.waitForURL(url=>url.origin===new URL(baseURL!).origin&&url.pathname==='/account/licenses/',{timeout:60_000});
    const bindings=()=>restSelect<Array<{user_id:string}>>(request,`billing_checkout_bindings?select=user_id&session_id=eq.${checkout.sessionId}`);
    expect(await bindings(),'legacy checkout has not yet established verified ownership').toEqual([]);
    const licenses=()=>restSelect<Array<{id:string}>>(request,`license_keys?select=id&user_id=eq.${persona!.id}`);
    for(let attempt=0;attempt<2;attempt++) {
      const details=page.locator('details').filter({has:page.locator('#claim-checkout-form')});
      await details.locator('summary').click();
      await page.locator('#claim-checkout-reference').fill(checkout.sessionId);
      await Promise.all([
        page.waitForEvent('framenavigated',{predicate:frame=>frame===page.mainFrame()&&new URL(frame.url()).pathname==='/account/licenses/'}),
        page.waitForResponse(response=>/\/functions\/v1\/billing-portal(?:-preview)?$/.test(new URL(response.url()).pathname)&&response.request().postDataJSON()?.action==='claim_checkout').then(async response=>expect(response.status(),await response.text()).toBe(200)),
        page.locator('#claim-checkout-btn').click(),
      ]);
      await page.waitForLoadState('load');
      await expect.poll(bindings,{timeout:30_000}).toEqual([{user_id:persona!.id}]);
      await expect(page.locator('#licenses-loading')).toBeHidden();
      await expect(page.locator('#setup-section')).toBeVisible();
    }
    expect(await licenses()).toHaveLength(1);
  },async()=>{
    const cleanup=await Promise.allSettled([customerId?deleteCustomer(request,customerId):Promise.resolve(),disposePersona(request,persona)]);
    const failed=cleanup.filter(r=>r.status==='rejected').map(r=>r.reason);
    if(failed.length) throw new AggregateError(failed,'Checkout claim fixtures could not be cleaned');
  });
});
