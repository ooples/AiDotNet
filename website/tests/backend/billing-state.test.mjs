import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

const exports = {};
const source = readFileSync(new URL('../../src/lib/billing.ts', import.meta.url), 'utf8');
const code = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
runInNewContext(code, { exports, require: () => ({ supabase: {} }) });

for (const [name, subscription, date] of [
  ['no subscription', null, null],
  ['active subscription', { current_period_end: 200, cancel_at_period_end: false }, null],
  ['legacy period-end cancellation', { current_period_end: 200, cancel_at_period_end: true }, 200],
  ['portal cancellation with the legacy flag false', { current_period_end: 200, cancel_at_period_end: false, cancel_at: 200 }, 200],
  ['custom cancellation date', { current_period_end: 200, cancel_at_period_end: false, cancel_at: 150 }, 150],
]) {
  test(`billing cancellation date: ${name}`, () => {
    assert.equal(exports.cancellationDate(subscription), date);
  });
}
