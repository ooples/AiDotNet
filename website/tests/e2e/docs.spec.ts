import { test, expect } from '@playwright/test';
import { readdirSync, statSync } from 'node:fs';
import { join, relative, sep } from 'node:path';

// Every page in src/content/docs is served, has a heading, and throws no page errors. Pages
// are discovered from the content folder, so a new doc is covered the day it is added.

function docRoutes(): string[] {
  const root = join(process.cwd(), 'src', 'content', 'docs');
  const out: string[] = [];
  const walk = (dir: string) => {
    for (const name of readdirSync(dir)) {
      const full = join(dir, name);
      if (statSync(full).isDirectory()) { walk(full); continue; }
      if (!/\.mdx?$/.test(name)) continue;
      const slug = relative(root, full).split(sep).join('/').replace(/\.mdx?$/, '').replace(/(^|\/)index$/, '');
      out.push(slug ? `/docs/${slug}/` : '/docs/');
    }
  };
  walk(root);
  return [...new Set(['/docs/', ...out])].sort();
}

test('documentation search finds a page, opens its result, and clears an empty search', async ({ page, request, baseURL }) => {
  const bundle = await request.get('/pagefind/pagefind-ui.js');
  const built = bundle.ok() && (bundle.headers()['content-type'] ?? '').includes('javascript');
  await bundle.dispose();
  test.skip(!built && process.env.PLAYWRIGHT_SITE_KIND === 'dev' && !process.env.CI && /^(localhost|127\.0\.0\.1)$/.test(new URL(baseURL!).hostname), 'Explicit dev target: Pagefind requires npm run build and astro preview');
  expect(built, 'built targets must serve Pagefind').toBe(true);
  await page.goto('/docs/getting-started/installation/');
  const search = page.locator('#docs-search input');
  await expect(search).toBeVisible();
  await search.fill('neural network');
  const result = page.locator('#docs-search .pagefind-ui__result-link').filter({ hasNotText: /^Installation$/ }).first();
  await expect(result).toBeVisible({ timeout: 15_000 });
  const target = await result.getAttribute('href');
  expect(target).toBeTruthy();
  const destination = new URL(target!, page.url()).href;
  expect(destination).not.toBe(page.url());
  await result.click();
  await expect(page).toHaveURL(destination);
  await expect(page.locator('article h1').or(page.locator('main h1')).first()).toBeVisible();
  // Quoting asks Pagefind for an exact phrase; unquoted queries intentionally fall back to
  // shorter prefix matches and can find one-letter identifiers in the generated API docs.
  await page.locator('#docs-search input').fill('"zzqxxvvppllkknnzzqxxvv"');
  await expect(page.locator('#docs-search .pagefind-ui__message')).toContainText(/no results/i, { timeout: 15_000 });
  await page.locator('#docs-search .pagefind-ui__search-clear').click();
  await expect(page.locator('#docs-search input')).toHaveValue('');
  await expect(page.locator('#docs-search .pagefind-ui__result-link')).toHaveCount(0);
});

test('phone documentation navigation opens and follows a destination', async ({ page }, info) => {
  test.skip(info.project.name !== 'anon-mobile', 'phone navigation only');
  await page.goto('/docs/getting-started/installation/');
  const nav = page.locator('details').filter({ has: page.getByText('Documentation Navigation', { exact: true }) });
  await nav.locator('summary').click();
  const link = nav.getByRole('link', { name: 'Quick Start', exact: true });
  await expect(link).toBeVisible();
  await link.click();
  await expect(page).toHaveURL(/\/docs\/getting-started\/quickstart\//);
  await expect(page.locator('h1')).toBeVisible();
});

test('every docs page renders with a heading and no page errors', async ({ page }, info) => {
  test.skip(info.project.name !== 'anon-desktop', 'run once');
  test.setTimeout(240_000);
  // Generated API entries share the docs template and are checked in parallel
  // below. Browser interactions belong to the template and ordinary doc flows.
  const routes = docRoutes().filter((route) => !route.startsWith('/docs/reference/wiki/'));
  expect(routes.length, 'docs discovered from src/content/docs').toBeGreaterThan(20);
  const problems: string[] = [];
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  for (const route of routes) {
    errors.length = 0;
    const res = await page.goto(route);
    if (!res || res.status() !== 200) { problems.push(`${route}: HTTP ${res?.status()}`); continue; }
    if (await page.locator('h1').count() === 0) problems.push(`${route}: no <h1>`);
    if (errors.length) problems.push(`${route}: page error ${errors[0]}`);
  }
  expect(problems).toEqual([]);
});

test('every generated API reference route is served with a heading', async ({ request }, info) => {
  test.skip(info.project.name !== 'anon-desktop', 'run once');
  test.setTimeout(20 * 60_000);
  let routes = docRoutes().filter((route) => route.startsWith('/docs/reference/wiki/'));
  // A deployed preview supplies its complete build inventory even when the
  // test runner checkout does not contain the generated, uncommitted wiki.
  const inventory = await request.get('/verification-routes.json');
  test.skip(inventory.status() === 404 && process.env.PLAYWRIGHT_SITE_KIND === 'dev' && !process.env.CI, 'Explicit dev target has no build inventory');
  expect(inventory.ok(), 'built targets must provide verification-routes.json').toBe(true);
  const built = await inventory.json();
  expect(Array.isArray(built), 'build inventory must be an array').toBe(true);
  expect(built.every((route: unknown) => typeof route === 'string' && route.startsWith('/')), 'all inventory entries must be canonical paths').toBe(true);
  routes = built.filter((route: string) => route.startsWith('/docs/reference/wiki/'));
  await inventory.dispose();
  test.skip(routes.length === 0, 'API reference was not generated in this build');
  const problems: string[] = [];
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(16, routes.length) }, async () => {
    while (next < routes.length) {
      const route = routes[next++];
      let response;
      try {
        response = await request.get(route, { timeout: 30_000, maxRedirects: 0 });
        if (response.status() !== 200) problems.push(`${route}: HTTP ${response.status()}`);
        else if (!/<h1(?:\s|>)/i.test(await response.text())) problems.push(`${route}: no heading`);
      } catch (error) { problems.push(`${route}: ${String(error).split('\n')[0]}`); }
      finally { await response?.dispose(); }
    }
  }));
  await info.attach('api-reference-coverage.json', { body: JSON.stringify({ checked: routes.length, problems }, null, 2), contentType: 'application/json' });
  expect(problems).toEqual([]);
});
