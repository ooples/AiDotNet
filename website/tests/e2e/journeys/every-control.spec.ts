import { test, expect, devices } from '@playwright/test';
import { withCleanup } from '../helpers/cleanup';
import { deleteUser, journeySkipReason, restSelect } from '../helpers/backend';
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
  [/^\/account\/api-keys\/ #copy-key-btn$/, 'user/api-keys.spec: create, copy the exact revealed key, dismiss, and revoke'],
  [/^\/account\/api-keys\/ #close-new-key$/, 'user/api-keys.spec: dismiss the revealed key and verify the masked list'],
];
const PRO_STRIPE_DEDICATED: CrawlOptions['dedicated'] = [
  ...SIGNED_IN_DEDICATED,
  [/^\/pricing\/ button::Current plan.*\[data-tier=pro\]/, 'account-billing.spec: the current plan is shown and cannot be bought again'],
  [/^\/pricing\/ button::Cancel paid subscription/, 'account-billing.spec: Stripe confirms paid cancellation separately from Community registration', { optional: true }],
];
const PRO_GRANTED_DEDICATED = PRO_STRIPE_DEDICATED;

interface PersonaRun {
  kind: PersonaKind | 'anon';
  seeds: string[];
  inScope: (p: string) => boolean;
  dedicated: CrawlOptions['dedicated'];
  stubWrites?: boolean;
  expectedStatus?: CrawlOptions['expectedStatus'];
  minPages: number;
  minControls?: number;
}

const RUNS: PersonaRun[] = [
  {
    kind: 'anon',
    seeds: ROUTES.filter((p) => !isAccount(p) && !isAdmin(p) && !isAuthCallback(p)),
    inScope: (p) => !isAccount(p) && !isAdmin(p) && !isAuthCallback(p),
    dedicated: [
      // The crawler submits the sign-up form; the button then stays disabled as the success state.
      [/^\/signup\/ button::Check your email$/, 'new-customer.spec: sign up and confirm the email link', { optional: true }],
    ],
    minPages: 40,
    minControls: 25,
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
    dedicated: [
      ...SIGNED_IN_DEDICATED,
      [/^\/admin\/licenses\/ #close-activations-btn$/, 'admin/license-lifecycle.spec: open activations, deactivate machines, close, and verify freed seats'],
      [/^\/admin\/users\/ #(prev|next)-btn$/, 'admin/user-management.spec: move forward and back through user pages'],
    ],
    stubWrites: true,
    minPages: 10,
  },
];

test.describe.configure({ mode: 'parallel' });

for (const run of RUNS) {
  test(`every link and control works for: ${run.kind}`, async ({ page, request, browser, baseURL }) => {
    if (run.kind === 'anon' && (!process.env.SUPABASE_URL || !process.env.SUPABASE_SERVICE_ROLE_KEY)) {
      if (process.env.CI) throw new Error('Anonymous signup exploration requires Supabase cleanup credentials');
      test.skip(true, 'Set Supabase URL and service key to clean up anonymous signup fixtures');
    }
    if (run.kind !== 'anon') {
      const skip = journeySkipReason();
      test.skip(!!skip, skip);
    }
    test.setTimeout(45 * 60_000);
    const origin = new URL(baseURL!).origin;
    let persona: Persona | null = null;
    const generatedEmails = new Set<string>();
    await withCleanup(async () => {
      if (run.kind !== 'anon') persona = await createPersona(request, run.kind);
      const desktop = await crawl({
        page, request, origin, persona, seeds: run.seeds, inScope: run.inScope,
        dedicated: run.dedicated, stubWrites: run.stubWrites, expectedStatus: run.expectedStatus,
        generatedEmails,
      });
      // Exercise shared controls again on phone layouts: a successful desktop click does not
      // establish that overlays, wrapping, and touch-sized navigation work on a narrow screen.
      const mobileContext = await browser.newContext({
        ...devices['Pixel 5'], baseURL, storageState: await page.context().storageState(),
      });
      let mobile;
      try {
        mobile = await crawl({
          page: await mobileContext.newPage(), request: mobileContext.request, origin, persona,
          seeds: desktop.pages, inScope: () => false,
          dedicated: run.dedicated, stubWrites: run.stubWrites, expectedStatus: run.expectedStatus,
          generatedEmails,
        });
      } finally { await mobileContext.close(); }
      const findings = [...desktop.findings, ...mobile.findings.filter((f) => f.page !== '(dedicated list)')];
      test.info().annotations.push({
        type: 'coverage',
        description: `${desktop.pages.length} pages in the browser, ${desktop.httpPages} more over HTTP, ${desktop.exercised.length + mobile.exercised.length} controls, ${desktop.links} links`,
      });
      await test.info().attach(`coverage-${run.kind}.json`, {
        body: JSON.stringify({ pages: desktop.pages, desktop: desktop.exercised, mobile: mobile.exercised }, null, 2),
        contentType: 'application/json',
      });
      // A crawl that found almost nothing is a broken crawl, not a clean site.
      expect(desktop.pages.length, 'pages crawled').toBeGreaterThanOrEqual(run.minPages);
      expect(desktop.exercised.length, 'controls exercised').toBeGreaterThanOrEqual(run.minControls ?? run.minPages);
      expect(mobile.exercised.length, 'mobile controls exercised').toBeGreaterThanOrEqual(run.minControls ?? run.minPages);
      expect(findings, `Broken for ${run.kind}:\n${formatFindings(findings)}`).toEqual([]);
    }, async () => {
      const tasks: Array<Promise<unknown>> = [];
      if (run.kind === 'anon') for (const email of generatedEmails) {
        tasks.push((async () => {
          const rows = await restSelect<Array<{ id: string; email: string }>>(request,
            `profiles?select=id,email&email=eq.${encodeURIComponent(email)}`);
          for (const row of rows) {
            if (row.email !== email) throw new Error('Signup fixture ownership mismatch');
            await deleteUser(request, row.id);
          }
        })());
      }
      tasks.push(disposePersona(request, persona ?? undefined));
      const cleanup = await Promise.allSettled(tasks);
      const failures = cleanup.filter(result => result.status === 'rejected').map(result => result.reason);
      if (failures.length) throw new AggregateError(failures, 'Persona/signup cleanup failed');
    });
  });
}
