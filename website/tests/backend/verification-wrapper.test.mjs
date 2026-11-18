import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, rmSync, unlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, relative, basename } from 'node:path';
import { pathToFileURL } from 'node:url';
import { randomUUID } from 'node:crypto';
import { spawnSync } from 'node:child_process';

test('the real verification wrapper rejects setup-only success and preserves list mode',()=>{
  const root=mkdtempSync(join(tmpdir(),'aidotnet-wrapper-test-'));
  const fixture=resolve('tests/e2e/journeys',`gate-${randomUUID()}.spec.ts`);
  try {
    writeFileSync(join(root,'settings.env'),'PUBLIC_SUPABASE_URL=https://preview-test.supabase.co\nPUBLIC_SUPABASE_ANON_KEY=fixture-anon\n');
    writeFileSync(join(root,'identities.json'),JSON.stringify([{role:'user',email:'user@example.test',password:'FixtureOnlyUser!1'},{role:'admin',email:'admin@example.test',password:'FixtureOnlyAdmin!1'}]));
    writeFileSync(join(root,'bypass.txt'),'fixture-private-bypass');
    writeFileSync(join(root,'fetch.mjs'),`globalThis.fetch=async value=>String(value).endsWith('/verification-build.json')?Response.json({commit:process.env.EXPECTED_PREVIEW_COMMIT}):new Response('',{headers:{'set-cookie':'bypass=fixture-cookie; Path=/; Secure; HttpOnly'}});`);
    writeFileSync(fixture,"import {test} from '@playwright/test';test.skip('fixture intentionally skipped',()=>{});\n");
    const env={...process.env,CI:'',PLAYWRIGHT_BASE_URL:'https://test.aidotnet.dev',EXPECTED_PREVIEW_COMMIT:'a'.repeat(40),EXPECTED_PREVIEW_RUN:'',
      AIDOTNET_PREVIEW_ENV_FILE:join(root,'settings.env'),AIDOTNET_TEST_IDENTITIES_FILE:join(root,'identities.json'),AIDOTNET_PREVIEW_BYPASS_FILE:join(root,'bypass.txt'),
      PLAYWRIGHT_HTML_OUTPUT_DIR:join(root,'report'),NODE_OPTIONS:`${process.env.NODE_OPTIONS??''} --import=${pathToFileURL(join(root,'fetch.mjs')).href}`};
    const args=['scripts/test-deployed.mjs',relative(process.cwd(),fixture).split('\\').join('/'),'--project=journeys','--retries=0'];
    const result=spawnSync(process.execPath,args,{env,encoding:'utf8',timeout:90_000});
    assert.equal(result.status,1,result.stdout+result.stderr);
    assert.match(result.stderr,/No selected non-setup verification test passed/);
    assert.equal((result.stdout+result.stderr).includes('fixture-private-bypass'),false);
    const listed=spawnSync(process.execPath,[...args,'--list'],{env,encoding:'utf8',timeout:30_000});
    assert.equal(listed.status,0,listed.stdout+listed.stderr);
    assert.match(listed.stdout,/fixture intentionally skipped/);
    writeFileSync(fixture,"import {test} from '@playwright/test';test('fixture redacts private failure',()=>{throw new Error('Authorization: '+process.env.VERCEL_AUTOMATION_BYPASS_SECRET+'\\nReset link https://fixture.invalid/verify?token=fixture-recovery-token&type=recovery')});\n");
    const failed=spawnSync(process.execPath,args,{env,encoding:'utf8',timeout:60_000});
    assert.equal(failed.status,1);
    assert.equal((failed.stdout+failed.stderr).includes('fixture-private-bypass'),false,'terminal reporters must redact credentials before displaying failures');
    assert.equal((failed.stdout+failed.stderr).includes('fixture-recovery-token'),false,'one-time Auth links are credentials too');
    const inspect=spawnSync(process.platform==='win32'?'python':'python3',['-c',
      "import base64,io,json,pathlib,re,sys,zipfile; h=pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'); a=zipfile.ZipFile(io.BytesIO(base64.b64decode(re.search(r'data:application/zip;base64,([A-Za-z0-9+/=]+)',h).group(1)))); assert not any(secret in a.read(n) for n in a.namelist() for secret in [b'fixture-private-bypass',b'fixture-recovery-token']); [json.loads(a.read(n)) for n in a.namelist() if n.endswith('.json')]; print('Report credentials redacted and JSON structure preserved')",
      join(root,'report','index.html')],{encoding:'utf8',timeout:30_000});
    assert.equal(inspect.status,0,inspect.stdout+inspect.stderr);
  } finally {
    unlinkSync(fixture);
    assert.equal(resolve(root).startsWith(resolve(tmpdir())+requireSeparator()),true);
    assert.ok(basename(root).startsWith('aidotnet-wrapper-test-'));
    rmSync(root,{recursive:true,force:true});
  }
});
function requireSeparator(){ return process.platform==='win32'?'\\':'/'; }
