import { test, expect } from '@playwright/test';
import { createServer, type Server } from 'node:http';
import type { EventEmitter } from 'node:events';
import { crawl } from './helpers/crawler';
import { signIn } from './helpers/personas';

// Test the coverage gate itself against an isolated HTTP fixture. A passing
// production crawl is only meaningful if broken controls make the gate fail.
test.describe('crawler regression checks', () => {
  let server: Server;
  let origin: string;
  let coveredRowLoads = 0;

  test.beforeAll(async () => {
    server = createServer((req, res) => {
      if (req.url === '/slow-auth.css') {
        res.setHeader('Content-Type', 'text/css');
        setTimeout(() => res.end('#desktop-menu { display: none }'), 1500);
        return;
      }
      if (req.url === '/login/') {
        res.setHeader('Content-Type', 'text/html');
        res.end('<!doctype html><label>Email address<input id="email"></label><label>Password<input id="password" type="password"></label><button onclick="location.href=\'/account/\'">Sign In</button>');
        return;
      }
      if (req.url === '/account/') {
        res.setHeader('Content-Type', 'text/html');
        res.end('<!doctype html><link rel="stylesheet" href="/slow-auth.css"><h1>Account</h1><button id="desktop-menu">Desktop menu</button>');
        return;
      }
      if (req.url === '/covered-rows/') coveredRowLoads++;
      res.setHeader('Content-Type', 'text/html');
      const body = req.url === '/broken/'
        ? '<button id="noop">Do nothing</button><button id="request-only" onclick="fetch(\'/ping/\')">Send request without a result</button><a href="/missing/">Missing page</a>'
        : req.url === '/depth/'
          ? '<button id="reveal" onclick="document.getElementById(\'deep\').hidden=false">Reveal</button><button hidden id="deep" onclick="alert(\'Hello\')">Deep control</button>'
        : req.url === '/error/'
          ? '<script>console.error("Failed to fetch: fixture failure")</script><h1>Error fixture</h1>'
        : req.url === '/covered-rows/'
          ? '<div style="position:relative"><div style="position:absolute;inset:0;z-index:2;background:white"></div>'
            + '<button>Covered row action</button>'.repeat(12) + '</div>'
        : req.url === '/nested-cover/'
          ? '<button id="theme" onclick="document.body.classList.toggle(\'theme\')">Theme</button>'
            + '<button id="menu" onclick="document.getElementById(\'cover\').hidden=false">Menu</button>'
            + '<button id="copy" onclick="alert(\'Copied\')">Copy</button>'
            + '<div id="cover" hidden style="position:fixed;inset:0;z-index:2;background:white"></div>'
          : '<h1>Fixture</h1><button id="dialog" onclick="alert(\'Hello\')">Open dialog</button>';
      res.statusCode = req.url === '/missing/' ? 404 : 200;
      res.end(`<!doctype html><html><head><title>Fixture</title></head><body>${body}</body></html>`);
    });
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const address = server.address();
    if (!address || typeof address === 'string') throw new Error('Fixture did not bind a TCP port');
    origin = `http://127.0.0.1:${address.port}`;
  });

  test.afterAll(async () => {
    await new Promise<void>((resolve, reject) => server.close((err) => err ? reject(err) : resolve()));
  });

  const options = () => ({ origin, persona: null, inScope: () => false, dedicated: [] });

  test('sign-in waits for destination styles before the crawler observes controls', async ({ browser }) => {
    const context = await browser.newContext({ baseURL: origin });
    try {
      const page = await context.newPage();
      await signIn(page, { kind: 'free', email: 'fixture@example.test', password: 'fixture' });
      expect(await page.evaluate(() => document.readyState)).toBe('complete');
      await expect(page.locator('#desktop-menu')).toBeHidden();
    } finally { await context.close(); }
  });

  test('the depth limit reports revealed controls it did not exercise', async ({ page, request }) => {
    const report = await crawl({ ...options(), page, request, maxDepth: 0, seeds: [`${origin}/depth/`] });
    expect(report.findings).toEqual(expect.arrayContaining([
      expect.objectContaining({ control: expect.stringContaining('#deep'), problem: expect.stringContaining('not exercised') }),
    ]));
  });

  test('reports inert controls and broken links', async ({ page, request }) => {
    const report = await crawl({ ...options(), page, request, seeds: [`${origin}/broken/`] });
    expect(report.findings).toEqual(expect.arrayContaining([
      expect.objectContaining({ control: expect.stringContaining('#noop'), problem: expect.stringContaining('did nothing visible') }),
      expect.objectContaining({ control: expect.stringContaining('#request-only'), problem: expect.stringContaining('did nothing visible') }),
      expect.objectContaining({ control: 'link /missing/', problem: 'target answered HTTP 404' }),
    ]));
  });

  test('duplicate unreachable rows fail coverage without repeatedly reloading the page', async ({ page, request }) => {
    const report = await crawl({ ...options(), page, request, seeds: [`${origin}/covered-rows/`] });
    expect(report.findings).toEqual(expect.arrayContaining([
      expect.objectContaining({ control: expect.stringContaining('Covered row action'), problem: expect.stringContaining('not exercised') }),
    ]));
    expect(coveredRowLoads).toBe(2); // Initial visit and one attempt to restore the covered control.
  });

  test('a control blocked in a nested state is retried from its reachable parent state', async ({ page, request }) => {
    const report = await crawl({ ...options(), page, request, seeds: [`${origin}/nested-cover/`] });
    expect(report.findings).toEqual([]);
    expect(report.exercised).toContain('/nested-cover/ #copy');
  });

  test('two passes do not retain handlers or mutate the first report', async ({ page, request }) => {
    const events = ['dialog', 'console', 'pageerror', 'request', 'response', 'requestfailed'] as const;
    const emitter = page as unknown as EventEmitter;
    const counts = events.map((event) => emitter.listenerCount(event));
    const first = await crawl({ ...options(), page, request, seeds: [`${origin}/`] });
    expect(first.findings).toEqual([]);
    expect(events.map((event) => emitter.listenerCount(event))).toEqual(counts);
    const second = await crawl({ ...options(), page, request, seeds: [`${origin}/error/`] });
    expect(second.findings).toEqual(expect.arrayContaining([
      expect.objectContaining({ problem: expect.stringContaining('fixture failure') }),
    ]));
    expect(first.findings).toEqual([]);
    expect(events.map((event) => emitter.listenerCount(event))).toEqual(counts);
  });
});
