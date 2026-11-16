import { test, expect } from '@playwright/test';
import { journeySkipReason } from '../helpers/backend';
import { createPersona, disposePersona, type Persona, type PersonaKind } from '../helpers/personas';
import { crawl, formatFindings, staticRoutes, type CrawlOptions } from '../helpers/crawler';

// Every link and every control on the site, for every kind of visitor (see helpers/crawler.ts for
// the rules). One test per persona so a failure names who it broke for.
//
// What the crawler cannot judge by itself (that the right plan is shown, that a plan change
// really happens in Stripe) is asserted by account-billing.spec.ts; the controls it leaves to
// that spec, or to new-customer.spec.ts, are listed below with the test that covers them. The
// crawler fails if one of these entries stops matching a real control, so the list cannot rot.

const ROUTES = staticRoutes();
const isAccount = (p: string) => p.startsWith('/account');
const isAdmin = (p: string) => p.startsWith('/admin');
const isAuthCallback = (p: string) => p.startsWith('/auth/');

const SIGNED_IN_DEDICATED: Array<[RegExp, string]> = [
  [/^\/account\/settings\/ button::Update Password$/, 'new-customer.spec: change password, then sign in with it'],
  [/^\/account\/settings\/ #delete-confirm-btn$/, 'account-billing.spec: delete account (refused with a live subscription, then deleted)'],
];
const PRO_STRIPE_DEDICATED: Array<[RegExp, string]> = [
  ...SIGNED_IN_DEDICATED,
  [/^\/pricing\/ button::Current plan$/, 'account-billing.spec: the current plan is shown and cannot be bought again'],
];
const PRO_GRANTED_DEDICATED = PRO_STRIPE_DEDICATED;

interface PersonaRun {
  kind: PersonaKind | 'anon';
  seeds: string[];
  inScope: (p: string) => boolean;
  dedicated: Array<[RegExp, string]>;
  stubWrites?: boolean;
  expectedStatus?: CrawlOptions['expectedStatus'];
  minPages: number;
}

const RUNS: PersonaRun[] = [
  {
    kind: 'anon',
    seeds: ROUTES.filter((p) => !isAccount(p) && !isAdmin(p) && !isAuthCallback(p)),
    inScope: (p) => !isAccount(p) && !isAdmin(p) && !isAuthCallback(p),
    dedicated: [],
    minPages: 40,
  },
  {
    kind: 'free',
    seeds: ['/', '/pricing/', ...ROUTES.filter(isAccount)],
    inScope: (p) => isAccount(p) || p === '/pricing/' || p === '/',
    dedicated: SIGNED_IN_DEDICATED,
    minPages: 7,
  },
  {
    kind: 'pro-stripe',
    seeds: ['/', '/pricing/', ...ROUTES.filter(isAccount)],
    inScope: (p) => isAccount(p) || p === '/pricing/' || p === '/',
    dedicated: PRO_STRIPE_DEDICATED,
    minPages: 7,
  },
  {
    kind: 'pro-granted',
    seeds: ['/', '/pricing/', ...ROUTES.filter(isAccount)],
    inScope: (p) => isAccount(p) || p === '/pricing/' || p === '/',
    dedicated: PRO_GRANTED_DEDICATED,
    minPages: 7,
  },
  {
    // The admin account sees real customer data, so its writes are answered by a stub: the
    // crawler still proves each admin control sends its request. The real writes are covered by
    // admin/license-lifecycle.spec.ts and admin/user-management.spec.ts on throwaway records.
    kind: 'admin',
    seeds: ROUTES.filter((p) => isAdmin(p) || isAccount(p)),
    inScope: (p) => isAdmin(p) || isAccount(p),
    dedicated: SIGNED_IN_DEDICATED,
    stubWrites: true,
    minPages: 10,
  },
];

test.describe.configure({ mode: 'parallel' });

for (const run of RUNS) {
  test(`every link and control works for: ${run.kind}`, async ({ page, request, baseURL }) => {
    if (run.kind !== 'anon') {
      const skip = journeySkipReason();
      test.skip(!!skip, skip);
    }
    test.setTimeout(45 * 60_000);
    const origin = new URL(baseURL!).origin;
    let persona: Persona | null = null;
    try {
      if (run.kind !== 'anon') persona = await createPersona(request, run.kind);
      const desktop = await crawl({
        page, request, origin, persona, seeds: run.seeds, inScope: run.inScope,
        dedicated: run.dedicated, stubWrites: run.stubWrites, expectedStatus: run.expectedStatus,
      });
      // Phone layout: the same pages again, exercising only controls the desktop pass never saw
      // (mobile menu, collapsed navigation, anything shown only at small widths).
      await page.setViewportSize({ width: 390, height: 844 });
      const mobile = await crawl({
        page, request, origin, persona, seeds: desktop.pages, inScope: () => false,
        dedicated: run.dedicated, stubWrites: run.stubWrites, expectedStatus: run.expectedStatus,
        skipKeys: new Set(desktop.exercised),
      });
      const findings = [...desktop.findings, ...mobile.findings.filter((f) => f.page !== '(dedicated list)')];
      test.info().annotations.push({
        type: 'coverage',
        description: `${desktop.pages.length} pages, ${desktop.exercised.length + mobile.exercised.length} controls, ${desktop.links} links`,
      });
      await test.info().attach(`coverage-${run.kind}.json`, {
        body: JSON.stringify({ pages: desktop.pages, desktop: desktop.exercised, mobile: mobile.exercised }, null, 2),
        contentType: 'application/json',
      });
      // A crawl that found almost nothing is a broken crawl, not a clean site.
      expect(desktop.pages.length, 'pages crawled').toBeGreaterThanOrEqual(run.minPages);
      expect(desktop.exercised.length, 'controls exercised').toBeGreaterThan(run.minPages);
      expect(findings, `Broken for ${run.kind}:\n${formatFindings(findings)}`).toEqual([]);
    } finally {
      await disposePersona(request, persona ?? undefined);
    }
  });
}
