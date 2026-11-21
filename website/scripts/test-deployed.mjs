import { readFileSync, mkdtempSync, unlinkSync, rmdirSync } from 'node:fs';
import { parseEnv } from 'node:util';
import { spawnSync, execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { approvedPreviewOrigin } from './authorize-preview.mjs';
import { hasExecutedCheck } from './verification-result.mjs';

// Paths identify local credential files; their values never appear in reports
// or child command arguments. The browser receives only a same-origin cookie.
const previewOrigin = approvedPreviewOrigin(process.env.PLAYWRIGHT_BASE_URL || 'https://test.aidotnet.dev');
const envFile = process.env.AIDOTNET_PREVIEW_ENV_FILE;
const fixtureFile = process.env.AIDOTNET_TEST_IDENTITIES_FILE;
const bypassFile = process.env.AIDOTNET_PREVIEW_BYPASS_FILE;
if (!envFile || !fixtureFile || !bypassFile) throw new Error('Set AIDOTNET_PREVIEW_ENV_FILE, AIDOTNET_TEST_IDENTITIES_FILE, and AIDOTNET_PREVIEW_BYPASS_FILE');
const settings = parseEnv(readFileSync(envFile, 'utf8'));
const identities = JSON.parse(readFileSync(fixtureFile, 'utf8'));
const customer = identities.find((user) => user.role === 'user');
const admin = identities.find((user) => user.role === 'admin');
if (!customer || !admin) throw new Error('Disposable customer and admin identities are required');
const directory = mkdtempSync(join(tmpdir(), 'aidotnet-e2e-auth-'));
const report = join(directory, 'result.json');
const listOnly = process.argv.includes('--list');
const env = {
  ...process.env,
  PLAYWRIGHT_BASE_URL: previewOrigin,
  AIDOTNET_AUTH_RUN_DIR: directory,
  EXPECTED_PREVIEW_COMMIT: process.env.EXPECTED_PREVIEW_COMMIT || execFileSync('git', ['rev-parse','HEAD'], { encoding:'utf8' }).trim(),
  PLAYWRIGHT_JSON_OUTPUT_FILE: report,
  PLAYWRIGHT_HTML_OUTPUT_DIR: process.env.PLAYWRIGHT_HTML_OUTPUT_DIR || fileURLToPath(new URL('../playwright-report/deployed',import.meta.url)),
  PUBLIC_SUPABASE_URL: settings.PUBLIC_SUPABASE_URL,
  PUBLIC_SUPABASE_ANON_KEY: settings.PUBLIC_SUPABASE_ANON_KEY,
  PREVIEW_BYPASS_SECRET: readFileSync(bypassFile, 'utf8').trim(),
  PLAYWRIGHT_USER_EMAIL: customer.email,
  PLAYWRIGHT_USER_PASSWORD: customer.password,
  PLAYWRIGHT_ADMIN_EMAIL: admin.email,
  PLAYWRIGHT_ADMIN_PASSWORD: admin.password,
};
const result = spawnSync(process.execPath, [fileURLToPath(new URL('../node_modules/@playwright/test/cli.js', import.meta.url)),
  'test', '--config', 'playwright.deployed.config.ts', '--reporter=./tests/e2e/helpers/redact-reporter.ts,list,json,html', ...process.argv.slice(2)], {
  cwd: fileURLToPath(new URL('../', import.meta.url)), env, stdio: 'inherit',
});
let status = result.status ?? 1;
try {
  if (result.error) throw result.error;
  if (!listOnly) {
    const sanitized=spawnSync(process.platform==='win32'?'python':'python3',[fileURLToPath(new URL('./sanitize-playwright-report.py',import.meta.url)),env.PLAYWRIGHT_HTML_OUTPUT_DIR],{env,stdio:'inherit'});
    if (sanitized.status !== 0) status=1;
  }
  if (!listOnly && status === 0 && !hasExecutedCheck(JSON.parse(readFileSync(report,'utf8')))) {
    console.error('No selected non-setup verification test passed. Supply the backend configuration required by the selected journeys.');
    status = 1;
  }
} finally {
  for (const name of ['user.json','admin.json','deployment.json','result.json']) {
    try { unlinkSync(join(directory,name)); } catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
  rmdirSync(directory);
}
process.exit(status);
