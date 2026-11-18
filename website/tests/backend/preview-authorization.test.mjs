import { test } from 'node:test';
import assert from 'node:assert/strict';
import { approvedPreviewOrigin, authorizePreview } from '../../scripts/authorize-preview.mjs';
import { hasExecutedCheck } from '../../scripts/verification-result.mjs';
import { serverRequest } from '../../scripts/secure-backend-http.mjs';

const commit = 'a'.repeat(40), secret = 'fixture-private-bypass';
for (const origin of ['https://attacker.example','http://test.aidotnet.dev','https://test.aidotnet.dev.attacker.example','https://test.aidotnet.dev/path','https://user@test.aidotnet.dev']) {
  test(`reject an unsafe target before sending a credential: ${origin}`, async () => {
    let requests = 0;
    await assert.rejects(authorizePreview({baseURL:origin,secret,expectedCommit:commit,fetcher:async()=>{requests++;throw new Error('must not send');}}));
    assert.equal(requests,0);
  });
}
test('authorize only the intended preview revision and transfer a scoped cookie', async () => {
  assert.equal(approvedPreviewOrigin('https://test.aidotnet.dev/'),'https://test.aidotnet.dev');
  const requests=[];
  const state=await authorizePreview({baseURL:'https://test.aidotnet.dev',secret,expectedCommit:commit,expectedRun:'123',fetcher:async(url,options)=>{
    requests.push({url,options});
    if(requests.length===1) return new Response('',{status:200,headers:{'set-cookie':'__vercel_protection_bypass=cookie-value; Path=/; Secure; HttpOnly; SameSite=Lax'}});
    assert.equal(options.headers['x-vercel-protection-bypass'],undefined);
    assert.equal(options.headers.Cookie,'__vercel_protection_bypass=cookie-value');
    return Response.json({commit,run_id:'123'});
  }});
  assert.equal(requests.length,2);
  assert.equal(state.cookies[0].domain,'test.aidotnet.dev');
  assert.equal(state.cookies[0].httpOnly,true);
});
test('an alias moved to a different revision fails instead of verifying the wrong build', async () => {
  let requests=0;
  await assert.rejects(authorizePreview({baseURL:'https://test.aidotnet.dev',secret,expectedCommit:commit,fetcher:async()=>++requests===1?new Response(''):Response.json({commit:'b'.repeat(40)})}),/Rerun the deployment job/);
});
test('credential-bearing network failures are sanitized before reaching reporters', async () => {
  await assert.rejects(authorizePreview({baseURL:'https://test.aidotnet.dev',secret,expectedCommit:commit,fetcher:async()=>{throw new Error(`network error with ${secret}`);}}),error=>!String(error).includes(secret)&&error.cause===undefined);
});
test('setup and skipped journeys cannot satisfy verification, but a passed check can', () => {
  const report={suites:[{specs:[{file:'deployment.setup.ts',tests:[{projectName:'deployment.setup',results:[{status:'passed'}]}]},{file:'journey.spec.ts',tests:[{projectName:'journeys',results:[{status:'skipped'}]}]}]}]};
  assert.equal(hasExecutedCheck(report),false);
  report.suites[0].specs[1].tests[0].results[0].status='passed';
  assert.equal(hasExecutedCheck(report),true);
});

test('privileged API failures never include credential-bearing headers',async()=>{
  await assert.rejects(serverRequest('https://api.stripe.com/v1/customers',{headers:{authorization:secret}},async()=>{throw new Error(secret);}),error=>!String(error).includes(secret)&&error.cause===undefined);
  let sent=false;
  await assert.rejects(serverRequest('https://attacker.example/',{headers:{authorization:secret}},async()=>{sent=true;}));
  assert.equal(sent,false);
});
