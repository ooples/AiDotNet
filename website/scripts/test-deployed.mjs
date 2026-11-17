import { readFileSync } from 'node:fs';
import { parseEnv } from 'node:util';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

// Paths identify local credential files; their values never appear in reports
// or child command arguments. The browser receives only a same-origin cookie.
const envFile = process.env.AIDOTNET_PREVIEW_ENV_FILE;
const fixtureFile = process.env.AIDOTNET_TEST_IDENTITIES_FILE;
const bypassFile = process.env.AIDOTNET_PREVIEW_BYPASS_FILE;
if (!envFile || !fixtureFile || !bypassFile) throw new Error('Set AIDOTNET_PREVIEW_ENV_FILE, AIDOTNET_TEST_IDENTITIES_FILE, and AIDOTNET_PREVIEW_BYPASS_FILE');
const settings = parseEnv(readFileSync(envFile, 'utf8'));
const identities = JSON.parse(readFileSync(fixtureFile, 'utf8'));
const customer = identities.find((user) => user.role === 'user');
const admin = identities.find((user) => user.role === 'admin');
if (!customer || !admin) throw new Error('Disposable customer and admin identities are required');
const env = {
  ...process.env,
  PLAYWRIGHT_BASE_URL: process.env.PLAYWRIGHT_BASE_URL || 'https://test.aidotnet.dev',
  PUBLIC_SUPABASE_URL: settings.PUBLIC_SUPABASE_URL,
  PUBLIC_SUPABASE_ANON_KEY: settings.PUBLIC_SUPABASE_ANON_KEY,
  VERCEL_AUTOMATION_BYPASS_SECRET: readFileSync(bypassFile, 'utf8').trim(),
  PLAYWRIGHT_USER_EMAIL: customer.email,
  PLAYWRIGHT_USER_PASSWORD: customer.password,
  PLAYWRIGHT_ADMIN_EMAIL: admin.email,
  PLAYWRIGHT_ADMIN_PASSWORD: admin.password,
};
const result = spawnSync(process.execPath, [fileURLToPath(new URL('../node_modules/@playwright/test/cli.js', import.meta.url)),
  'test', '--config', 'playwright.deployed.config.ts', ...process.argv.slice(2)], {
  cwd: fileURLToPath(new URL('../', import.meta.url)), env, stdio: 'inherit',
});
if (result.error) throw result.error;
process.exit(result.status ?? 1);
