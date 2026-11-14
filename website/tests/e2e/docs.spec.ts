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

test('every docs page renders with a heading and no page errors', async ({ page }, info) => {
  test.skip(info.project.name !== 'anon-desktop', 'run once');
  test.setTimeout(240_000);
  const routes = docRoutes();
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
