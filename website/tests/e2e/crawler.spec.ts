import { test, expect } from '@playwright/test';
import { createServer, type Server } from 'node:http';
import type { EventEmitter } from 'node:events';
import { crawl } from './helpers/crawler';

// Test the coverage gate itself against an isolated HTTP fixture. A passing
// production crawl is only meaningful if broken controls make the gate fail.
test.describe('crawler regression checks', () => {
  let server: Server;
  let origin: string;

  test.beforeAll(async () => {
    server = createServer((req, res) => {
      res.setHeader('Content-Type', 'text/html');
      const body = req.url === '/broken/'
        ? '<button id="noop">Do nothing</button><button id="request-only" onclick="fetch(\'/ping/\')">Send request without a result</button><a href="/missing/">Missing page</a>'
        : req.url === '/depth/'
          ? '<button id="reveal" onclick="document.getElementById(\'deep\').hidden=false">Reveal</button><button hidden id="deep" onclick="alert(\'Hello\')">Deep control</button>'
        : req.url === '/error/'
          ? '<script>console.error("Failed to fetch: fixture failure")</script><h1>Error fixture</h1>'
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
