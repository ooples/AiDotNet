import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import ts from 'typescript';

// Execute the handlers and ownership helper; only network/database boundaries are doubled.
function fixture(functionName, {preferred='cus_ambiguous',owned=['cus_verified']}={}) {
  let handler;
  const accessed=[], portals=[], deleted=[];
  const client={
    auth:{getUser:async()=>({data:{user:{id:'owner',email:'owner@example.test'}},error:null}),
      admin:{deleteUser:async id=>{deleted.push(id);return {error:null};}}},
    from(table){
      const filters=[];
      const rows=table==='profiles'?[{id:'owner',stripe_customer_id:preferred,subscription_tier:'pro'}]
        :table==='billing_customers'?owned.map(customer_id=>({user_id:'owner',customer_id})):[];
      const q={select(){return q;},eq(k,v){filters.push([k,v]);return q;},
        single:async()=>({data:rows.find(r=>filters.every(([k,v])=>r[k]===v)),error:null}),
        maybeSingle:async()=>q.single(),
        then(resolve,reject){return Promise.resolve({data:rows.filter(r=>filters.every(([k,v])=>r[k]===v)),error:null}).then(resolve,reject);}};
      return q;
    },
  };
  class Stripe {
    static createFetchHttpClient(){return {};}
    subscriptions={list:({customer})=>{accessed.push(customer);return (async function*(){})();}};
    prices={list:()=> (async function*(){})()};
    billingPortal={configurations:{list:()=> (async function*(){})(),create:async()=>({id:'bpc_fixture'})},
      sessions:{create:async params=>{portals.push(params.customer);return {url:'https://billing.stripe.com/fixture'};}}};
  }
  const compile=source=>ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022,esModuleInterop:true}}).outputText;
  const ownership={};
  runInNewContext(compile(readFileSync(new URL('../../supabase/functions/_shared/checkout-ownership.ts',import.meta.url),'utf8')),{exports:ownership});
  runInNewContext(compile(readFileSync(new URL(`../../supabase/functions/${functionName}/index.ts`,import.meta.url),'utf8')),{
    exports:{},Request,Response,console:{error(){}},
    Deno:{env:{get:()=> 'fixture-config'}},
    require(id){
      if(id.includes('deno.land')) return {serve:fn=>{handler=fn;}};
      if(id.endsWith('/cors.ts')) return {withCors:fn=>fn,websiteOrigin:()=> 'https://test.aidotnet.dev'};
      if(id.includes('supabase-js')) return {createClient:()=>client};
      if(id.includes('stripe@')) return Stripe;
      if(id.endsWith('/checkout-ownership.ts')) return ownership;
      throw new Error('Unexpected import '+id);
    },
  });
  return {accessed,portals,deleted,send:body=>handler(new Request('https://fixture.invalid/'+functionName,{method:'POST',headers:{authorization:'Bearer fixture'},body:JSON.stringify(body)}))};
}

test('an ambiguous legacy preference does not block the verified customer portal',async()=>{
  const f=fixture('billing-portal');
  const summary=await f.send({action:'summary'});
  assert.equal(summary.status,200);
  const state=await summary.json();
  assert.equal(state.ownership_resolution_required,true);
  assert.equal(state.billing_account,true);
  assert.deepEqual(state.billing_accounts,[{id:'cus_verified',selected:true}]);
  assert.match(state.ownership_message,/Claim your completed checkout/);
  assert.equal((await f.send({action:'manage'})).status,200);
  assert.deepEqual(f.portals,['cus_verified']);
  assert.ok(f.accessed.every(id=>id==='cus_verified'));
  assert.equal((await f.send({action:'manage',customer_id:'cus_ambiguous'})).status,403);
});

test('a legacy link alone exposes recovery guidance without granting Stripe access or new checkout',async()=>{
  const f=fixture('billing-portal',{owned:[]});
  const summary=await f.send({action:'summary'});
  assert.equal(summary.status,200);
  const state=await summary.json();
  assert.equal(state.billing_account,false);
  assert.equal(state.ownership_resolution_required,true);
  for(const body of [{action:'manage'},{action:'cancel'},{action:'checkout_reference',tier:'pro',interval:'month'}]) {
    const response=await f.send(body);
    assert.equal(response.status,409);
    assert.equal((await response.json()).error,'billing_ownership_unverified');
  }
  assert.deepEqual(f.accessed,[]);
  assert.deepEqual(f.portals,[]);
});

test('account deletion stays blocked when a legacy link is unresolved, even with other verified customers',async()=>{
  const f=fixture('delete-account');
  const response=await f.send({confirm:'DELETE'});
  assert.equal(response.status,503);
  assert.equal((await response.json()).error,'billing_ownership_unverified');
  assert.deepEqual(f.deleted,[]);
  assert.deepEqual(f.accessed,[]);
});

test('verified profile preference remains usable without an ownership warning',async()=>{
  const f=fixture('billing-portal',{preferred:'cus_verified'});
  assert.equal((await (await f.send({action:'summary'})).json()).ownership_resolution_required,false);
  assert.equal((await f.send({action:'manage'})).status,200);
  assert.deepEqual(f.portals,['cus_verified']);
});
