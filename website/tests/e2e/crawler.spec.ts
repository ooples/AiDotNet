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
  let bootstrapAborted = 0;

  test.beforeAll(async () => {
    server = createServer((req, res) => {
      if (req.url === '/slow-bootstrap/') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.flushHeaders();
        res.on('close', () => { if (!res.writableEnded) bootstrapAborted++; });
        setTimeout(() => res.end('{}'), 1500);
        return;
      }
      if (req.url === '/slow-body/') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.flushHeaders();
        setTimeout(() => res.end('{"message":"Updated"}'), 3500);
        return;
      }
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
        res.end('<!doctype html><link rel="stylesheet" href="/slow-auth.css"><h1>Account</h1><button id="desktop-menu">Desktop menu</button><script>fetch("/slow-body/").then(r=>r.json()).then(()=>document.body.dataset.authReady="true").catch(e=>console.error(e))</script>');
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
        : req.url === '/form-email/'
          ? '<form onsubmit="event.preventDefault();alert(\'Saved\')"><input type="email" value="keep@example.test"><input type="email" required><button>Save</button></form>'
        : req.url === '/pointer-disabled/'
          ? '<button id="unclickable" style="pointer-events:none" onclick="alert(\'Unreachable\')">Visible action</button>'
        : req.url === '/nested-cover/'
          ? '<button id="theme" onclick="document.body.classList.toggle(\'theme\')">Theme</button>'
            + '<button id="menu" onclick="document.getElementById(\'cover\').hidden=false">Menu</button>'
            + '<button id="copy" onclick="alert(\'Copied\')">Copy</button>'
            + '<div id="cover" hidden style="position:fixed;inset:0;z-index:2;background:white"></div>'
        : req.url === '/row-representatives/'
          ? '<div style="position:relative"><button onclick="alert(\'First\')">Row action</button><div style="position:absolute;inset:0;background:white"></div></div><button onclick="alert(\'Second\')">Row action</button>'
        : req.url === '/async-control/'
          ? '<button id="update" onclick="fetch(\'/slow-body/\').then(r=>r.json()).then(j=>document.getElementById(\'result\').textContent=j.message)">Update</button><span id="result"></span>'
        : req.url === '/unrelated-abort/'
          ? '<script>fetch(\'/aborted/\').catch(()=>{}); setTimeout(()=>console.error(\'Failed to fetch: genuine unrelated failure\'),200)</script><h1>Abort fixture</h1>'
          : '<h1>Fixture</h1><button id="dialog" onclick="alert(\'Hello\')">Open dialog</button><script>if(localStorage.getItem("sb-fixture-auth-token"))fetch("/slow-bootstrap/").catch(e=>console.error(e))</script>';
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
      expect(await page.locator('body').getAttribute('data-auth-ready')).toBe('true');
    } finally { await context.close(); }
  });

  test('restored sessions finish bootstrap requests before the first forced visit', async ({ browser, request }) => {
    const context = await browser.newContext({ baseURL: origin, storageState: {
      cookies: [], origins: [{ origin, localStorage: [{ name: 'sb-fixture-auth-token', value: 'fixture' }] }],
    } });
    try {
      const page = await context.newPage();
      const report = await crawl({ ...options(), page, request, seeds: [`${origin}/account/`],
        persona: { kind: 'free', email: 'fixture@example.test', password: 'fixture' },
      });
      expect(report.findings).toEqual([]);
      expect(bootstrapAborted, 'the crawler must not interrupt session bootstrap').toBe(0);
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

  test('cleanup tracking contains only newly minted form emails', async ({ page, request }) => {
    const generatedEmails = new Set<string>();
    const report = await crawl({ ...options(), page, request, seeds: [`${origin}/form-email/`], generatedEmails });
    expect(report.findings).toEqual([]);
    expect([...generatedEmails]).toEqual([expect.stringMatching(/^delivered\+e2e-user-crawl-.*@resend\.dev$/)]);
    expect(generatedEmails.has('keep@example.test')).toBe(false);
  });

  test('a visible pointer-disabled control fails coverage', async ({ page, request }) => {
    const report = await crawl({ ...options(), page, request, seeds: [`${origin}/pointer-disabled/`] });
    expect(report.findings).toEqual(expect.arrayContaining([
      expect.objectContaining({ control: expect.stringContaining('#unclickable'), problem: expect.stringContaining('not exercised') }),
    ]));
  });

  test('a control blocked in a nested state is retried from its reachable parent state', async ({ page, request }) => {
    const report = await crawl({ ...options(), page, request, seeds: [`${origin}/nested-cover/`] });
    expect(report.findings).toEqual([]);
    expect(report.exercised).toContain('/nested-cover/ #copy');
  });

  test('an unreachable row does not suppress another reachable representative', async ({ page, request }) => {
    const report = await crawl({ ...options(), page, request, seeds: [`${origin}/row-representatives/`] });
    expect(report.findings).toEqual([]);
    expect(report.exercised).toContain('/row-representatives/ button::Row action');
  });

  test('waits for a streamed response body and the visible control result', async ({ page, request }) => {
    const report = await crawl({ ...options(), page, request, seeds: [`${origin}/async-control/`] });
    expect(report.findings).toEqual([]);
    await expect(page.locator('#result')).toHaveText('Updated');
  });

  test('an unrelated aborted request cannot hide a genuine fetch console error', async ({ page, request }) => {
    await page.route('**/aborted/', route => route.abort('aborted'));
    const report = await crawl({ ...options(), page, request, seeds: [`${origin}/unrelated-abort/`] });
    expect(report.findings).toEqual(expect.arrayContaining([
      expect.objectContaining({ problem: expect.stringContaining('genuine unrelated failure') }),
    ]));
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
