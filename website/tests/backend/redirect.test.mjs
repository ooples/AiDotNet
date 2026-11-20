import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

const exports = {};
runInNewContext(ts.transpileModule(readFileSync(new URL('../../src/lib/redirect.ts', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS },
}).outputText, { exports, URL });
const origin = 'https://test.aidotnet.dev';
for (const raw of ['/a/..//attacker.example/phish', '//attacker.example/phish', '/\\attacker.example/phish', '/\t/attacker.example/phish', 'https://attacker.example']) {
  test(`redirect cannot escape the origin after canonicalization: ${JSON.stringify(raw)}`, () => {
    const result = exports.safeRedirect(raw, '/account/', origin);
    assert.equal(result, '/account/');
    assert.equal(new URL(result, origin).origin, origin);
  });
}
test('a normal local redirect preserves its query and fragment', () => {
  assert.equal(exports.safeRedirect('/account/?tab=keys#setup', '/account/', origin), '/account/?tab=keys#setup');
});
