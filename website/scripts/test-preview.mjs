import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const cwd = fileURLToPath(new URL('../', import.meta.url));
const env = {
  ...process.env,
  VERCEL: '1',
  PLAYWRIGHT_BASE_URL: 'http://127.0.0.1:4322',
  PUBLIC_SUPABASE_URL: 'https://preview-test.supabase.co',
  PUBLIC_SUPABASE_ANON_KEY: 'preview-test-anon-key',
  PUBLIC_STRIPE_PRO_MONTHLY_LINK: 'https://buy.stripe.com/test_preview_pro_month',
  PUBLIC_STRIPE_PRO_YEARLY_LINK: 'https://buy.stripe.com/test_preview_pro_year',
  PUBLIC_STRIPE_ENTERPRISE_MONTHLY_LINK: 'https://buy.stripe.com/test_preview_enterprise_month',
  PUBLIC_STRIPE_ENTERPRISE_YEARLY_LINK: 'https://buy.stripe.com/test_preview_enterprise_year',
};
function run(script, args = []) {
  const result = spawnSync(process.execPath, [script, ...args], { cwd, env, stdio: 'inherit' });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}

run(fileURLToPath(new URL('../node_modules/astro/bin/astro.mjs', import.meta.url)), ['build']);
run(fileURLToPath(new URL('../node_modules/pagefind/lib/runner/bin.cjs', import.meta.url)), ['--site', 'dist']);
run(fileURLToPath(new URL('../node_modules/@playwright/test/cli.js', import.meta.url)),
  ['test', '--config', 'playwright.preview.config.ts', ...process.argv.slice(2)]);
