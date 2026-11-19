import type { APIRequestContext, ConsoleMessage, Dialog, ElementHandle, Page, Request, Response, Route } from '@playwright/test';
import { readdirSync, statSync } from 'node:fs';
import { join, relative, sep } from 'node:path';
import { randomUUID } from 'node:crypto';
import type { EventEmitter } from 'node:events';
import { randomPassword, testUserEmail } from './backend';
import { hasSession, signIn, type Persona } from './personas';

// Exhaustive crawler for one persona: visits every page the persona can reach, checks every link,
// and clicks every control, failing on anything a user would experience as broken.
//
// Links:     no href / "#" / "javascript:" (unless an in-page anchor to an element that exists);
//            internal targets must answer < 400; external targets must not be 404/410/unreachable.
// Controls:  each visible, enabled control is clicked (forms are filled first) and must DO
//            something observable: navigate, open a popup or download, show a dialog,
//            write the clipboard, or change what is on screen. A request alone is not a result.
//            While it runs there must be
//            no page error, no console error, no error-looking alert, and no backend response
//            >= 400 (signed-in personas) unless the test lists that exact case as expected.
//            Controls revealed by a click (menus, modals) are explored too, depth-first.
// Coverage:  a control the crawler cannot exercise (disabled and never enabled, or listed as
//            dedicated) must be named in `dedicated` with the test that covers it, and every
//            dedicated entry must match a control that exists, so the list cannot rot.
//
// Pages are discovered from src/pages and from links followed during the crawl, so new pages and
// new controls are covered without editing this file.
//
// Templated page families (the ~12,000 API reference pages, docs, features, solutions) are opened
// in the browser only for the first few pages of each family, since every page of a family has the
// same controls. Every other page is still fetched, and every link on it checked, over plain HTTP
// in parallel. That keeps a full-site crawl to minutes instead of hours.

export interface Finding { page: string; control: string; problem: string }

type Visual = Array<[number, string, boolean]>;
interface CapturedAction {
  target: Element;
  visual?: Visual;
  activated?: boolean;
  pressBaselines: WeakMap<Event, Visual>;
  pressResults: Array<[number, string]>;
  cleanup: Array<() => void>;
}
interface CaptureWindow {
  __fx: { mut: number; clip: number; open: number };
  __visualIds?: WeakMap<Element, number>;
  __visualNext?: number;
  __crawlVisual?: () => Visual;
  __crawlAction?: CapturedAction;
  __crawlPressResult?: (event: Event) => void;
}

export interface CrawlOptions {
  page: Page;
  request: APIRequestContext;
  origin: string;
  persona: Persona | null;              // null = signed out
  seeds: string[];
  inScope: (path: string) => boolean;   // same-origin pages this persona should visit
  /** Answer backend writes with a stub instead of sending them (admin: real customer data). */
  stubWrites?: boolean;
  /** Controls covered by a dedicated test instead: key regex -> the test that covers it. */
  dedicated: Array<[RegExp, string] | [RegExp, string, { optional: true }]>;
  /** Backend responses >= 400 that are the correct outcome of that control. */
  expectedStatus?: Array<{ control: RegExp; url: RegExp; status: number }>;
  /** Only exercise controls whose key is new (used for the mobile pass after the desktop pass). */
  skipKeys?: Set<string>;
  maxDepth?: number;
  /** Browser visits per templated family; the rest are link-checked over HTTP. */
  browserPagesPerFamily?: number;
  /** Emails minted by form exploration, so its caller can clean up signup fixtures. */
  generatedEmails?: Set<string>;
}

export interface CrawlReport {
  findings: Finding[];
  pages: string[];
  httpPages: number;
  exercised: string[];
  links: number;
  keys: Set<string>;
}

export function staticRoutes(): string[] {
  const root = join(process.cwd(), 'src', 'pages');
  const out: string[] = [];
  const walk = (dir: string) => {
    for (const name of readdirSync(dir)) {
      const full = join(dir, name);
      if (statSync(full).isDirectory()) { walk(full); continue; }
      if (!name.endsWith('.astro') || name.includes('[')) continue;
      const rel = relative(root, full).split(sep).join('/').replace(/\.astro$/, '');
      out.push((rel === 'index' ? '/' : `/${rel.replace(/(^|\/)index$/, '')}/`).replace(/\/+/g, '/'));
    }
  };
  walk(root);
  return out.sort();
}

interface ControlInfo {
  key: string;
  recordId: string | null;
  selector: string;
  label: string;
  tag: string;
  type: string;
  inForm: boolean;
  disabled: boolean;
  visible: boolean;
}

interface LinkInfo { raw: string | null; href: string; label: string; key: string }

const SUPABASE_HOST = /\.supabase\.co$/;
const FAMILY_ROOTS = /^\/(docs|features|solutions)\//;
/** One template per family: /docs/reference/wiki/** -> /docs/reference/wiki/*, /docs/tutorials/x/ ->
 *  /docs/tutorials/*, /features/x/ -> /features/*. Index pages (/docs/, /features/) are not in one. */
export function familyOf(path: string): string | null {
  if (!FAMILY_ROOTS.test(path)) return null;
  const segs = path.split('/').filter(Boolean);
  const depth = segs[0] === 'docs' ? Math.min(segs.length - 1, segs[1] === 'reference' && segs[2] === 'wiki' ? 3 : 2) : 1;
  if (segs.length <= depth) return null;
  return `/${segs.slice(0, depth).join('/')}/*`;
}

async function pool<T>(items: T[], size: number, fn: (item: T) => Promise<void>) {
  let i = 0;
  await Promise.all(Array.from({ length: Math.min(size, items.length) }, async () => {
    while (i < items.length) await fn(items[i++]);
  }));
}
const BENIGN_EXTERNAL_STATUS = new Set([401, 403, 405, 429, 999]); // bot walls (GitHub, LinkedIn, X)

// Runs in the page: tags every control with a replayable selector and a stable dedupe key.
// The open modal, if any: a dialog, or a fixed layer covering most of the viewport. While one is
// open only its own controls can be reached; the page behind it is covered.
const ACTIVE_MODAL = () => {
  const vw = window.innerWidth, vh = window.innerHeight;
  const layers = Array.from(document.querySelectorAll('body *')).filter((el) => {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return false;
    if ((el as HTMLElement).checkVisibility?.({ checkVisibilityCSS: true }) === false) return false;
    if (el.matches('dialog[open], [role=dialog], [role=alertdialog], [aria-modal=true]')) return true;
    const cs = getComputedStyle(el);
    return cs.position === 'fixed' && r.width >= vw * 0.9 && r.height >= vh * 0.9 && !!el.querySelector('button, a, input, select');
  });
  return (layers[layers.length - 1] as Element | undefined) ?? null;
};
type ModalWindow = { __activeModal?: () => Element | null };

/** The control exists, is laid out, and is not covered by a modal it is not part of. */
const REACHABLE = (selector: string) => {
  const el = document.querySelector(selector);
  if (!el) return false;
  const r = el.getBoundingClientRect();
  if (r.width === 0 || r.height === 0 || (el as HTMLElement).checkVisibility?.({ checkVisibilityCSS: true }) === false) return false;
  const modal = (window as unknown as ModalWindow).__activeModal?.() ?? null;
  if (modal && !modal.contains(el)) return false;
  const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
  return !!hit && (hit === el || el.contains(hit));
};

const ENUMERATE = (input: string | { path: string; target: Element }) => {
  const path = typeof input === 'string' ? input : input.path;
  const target = typeof input === 'string' ? null : input.target;
  const CONTROL = 'button, [role=button], [role=tab], [role=switch], [role=menuitem], [role=checkbox], input[type=submit], input[type=button], input[type=checkbox], input[type=radio], select, summary, a[href^="#"], a[href=""], a:not([href]), a[href^="javascript:"], [onclick]';
  const LAYOUT = 'header, nav, footer, #account-sidebar, aside, [data-layout]';
  const cssPath = (el: Element): string => {
    if (el.id) return `#${CSS.escape(el.id)}`;
    const parts: string[] = [];
    let cur: Element | null = el;
    while (cur && cur !== document.body) {
      if (cur.id) { parts.unshift(`#${CSS.escape(cur.id)}`); break; }
      const tag = cur.tagName.toLowerCase();
      const parent: Element | null = cur.parentElement;
      const same = parent ? Array.from(parent.children).filter((c) => c.tagName === cur!.tagName) : [];
      parts.unshift(same.length > 1 ? `${tag}:nth-of-type(${same.indexOf(cur) + 1})` : tag);
      cur = parent;
    }
    // A body child such as "button" must not match an earlier nested button.
    return `${parts[0]?.startsWith('#') ? '' : 'body > '}${parts.join(' > ')}`;
  };
  const name = (el: Element) => ((el.getAttribute('aria-label') || (el as HTMLElement).innerText || el.getAttribute('title')
    || (el as HTMLInputElement).value || el.getAttribute('name') || '') as string).replace(/\s+/g, ' ').trim().slice(0, 60);
  const visible = (el: Element) => {
    const r = el.getBoundingClientRect();
    // Pagefind's empty-search Clear button is transparent. Keep visible click-through
    // controls in coverage so the reachability check can report them as broken.
    return r.width > 0 && r.height > 0
      && (el as HTMLElement).checkVisibility?.({ checkVisibilityCSS: true, opacityProperty: true }) !== false;
  };
  const modal = (window as unknown as ModalWindow).__activeModal?.() ?? null;
  const controls = Array.from(document.querySelectorAll(CONTROL)).filter((el) => !el.closest('[aria-hidden="true"]')
    && (!modal || modal.contains(el))
    && !(el.matches('input[type=checkbox][disabled]') && el.closest('li.task-list-item, .contains-task-list, li')));
  const links = Array.from(document.querySelectorAll('a')).map((a) => ({
    raw: a.getAttribute('href'),
    href: (a as HTMLAnchorElement).href,
    label: name(a),
    key: '',
  }));
  return {
    controls: controls.map((el) => {
      // Pages built from one template (docs, feature and solution pages) share their controls.
      const template = path.replace(/^\/(docs|features|solutions)\/.+/, '/$1/*');
      const scope = el.closest(LAYOUT) ? 'layout' : template;
      const label = name(el);
      // data-* values tell same-label controls apart (Subscribe on the Pro card vs the Enterprise card);
      // ids and counters in them are normalised so table rows still collapse to one control.
      const norm = (v: string) => v.replace(/[0-9a-f]{8,}(-[0-9a-f]{4,})*|\d+/gi, 'N');
      // Repeated CRUD actions keep their identity when another customer changes
      // plan/name/role during a crawl. Those values are record payload, not paths.
      // Keep plan selectors elsewhere distinct (Subscribe Pro vs Enterprise).
      const recordAction = !!el.closest('tbody tr') && el.hasAttribute('data-id');
      const data = Array.from(el.attributes).filter((a) => a.name.startsWith('data-') && !a.name.startsWith('data-crawl-') && a.value.length < 40)
        .filter((a) => !recordAction || !/^data-(?:name|email|role|tier|status)$/.test(a.name))
        .map((a) => `[${a.name}=${norm(a.value)}]`).sort().join('');
      const controlName = el.matches('tbody .copy-key-btn') ? 'Copy license key' : label;
      const generic = `${el.tagName.toLowerCase()}:${el.getAttribute('role') ?? ''}:${norm(controlName)}${data}`;
      return {
        key: `${scope} ${el.id ? `#${el.id}` : generic}`,
        recordId: recordAction ? el.getAttribute('data-id') : null,
        selector: cssPath(el),
        isTarget: el === target,
        label: label || cssPath(el),
        tag: el.tagName.toLowerCase(),
        type: (el.getAttribute('type') || '').toLowerCase(),
        inForm: !!(el as HTMLButtonElement).form && ((el as HTMLButtonElement).type === 'submit'),
        disabled: (el as HTMLButtonElement).disabled === true || el.getAttribute('aria-disabled') === 'true',
        visible: visible(el),
      };
    }),
    links,
    ids: Array.from(document.querySelectorAll('[id]')).map((e) => e.id),
  };
};

export async function crawl(o: CrawlOptions): Promise<CrawlReport> {
  const { page, request, origin } = o;
  await page.context().grantPermissions(['clipboard-read', 'clipboard-write'], { origin });
  const findings: Finding[] = [];
  const tag = o.persona?.kind ?? 'anon';
  const add = (pg: string, control: string, problem: string) => {
    findings.push({ page: pg, control, problem });
    console.log(`[crawl ${tag}] FINDING ${pg} | ${control} -> ${problem}`);
  };
  const visited: string[] = [];
  const queue = [...new Set(o.seeds.map((seed) => new URL(seed, origin).pathname))];
  const queued = new Set(queue);
  const exercised = new Set<string>();
  const keys = new Set<string>();
  const disabledSeen = new Map<string, string>();   // key -> page
  const enabledSeen = new Map<string, string>();
  const internalLinks = new Map<string, string>();  // url -> first page seen
  const externalLinks = new Map<string, string>();
  const maxDepth = o.maxDepth ?? 3;
  const perFamily = o.browserPagesPerFamily ?? 3;
  const familyVisits = new Map<string, number>();
  const httpOnly: string[] = [];
  const pageStatus = new Map<string, number>();   // url -> status, for pages fetched over HTTP

  // Desktop and mobile reuse this page. Keep instrumentation scoped to this
  // pass, including when navigation or fixture cleanup throws.
  const emitter = page as unknown as EventEmitter;
  const ownedListeners: Array<[string, (...args: unknown[]) => void]> = [];
  type Payload = { dialog: Dialog; pageerror: Error; console: ConsoleMessage; request: Request; requestfailed: Request; response: Response };
  function on<E extends keyof Payload>(event: E, callback: (payload: Payload[E]) => void) {
    const listener = callback as (...args: unknown[]) => void;
    ownedListeners.push([event, listener]);
    emitter.on(event, listener);
  }
  let intercept: ((route: Route) => Promise<void>) | undefined;
  try {

  // ---- instrumentation shared by every action ----
  const externalNavs: string[] = [];
  const stubbed: string[] = [];
  intercept = async (route) => {
    const req = route.request();
    const url = new URL(req.url());
    if (req.isNavigationRequest() && req.frame() === page.mainFrame() && url.origin !== origin) {
      externalNavs.push(req.url());
      return route.fulfill({ status: 200, contentType: 'text/html', body: '<title>external</title><p>external page (stubbed by crawler)</p>' });
    }
    if (o.stubWrites && SUPABASE_HOST.test(url.hostname) && req.method() !== 'GET' && req.method() !== 'HEAD'
        && req.method() !== 'OPTIONS' && !url.pathname.startsWith('/auth/v1/token')) {
      stubbed.push(`${req.method()} ${url.pathname}`);
      // supabase-js stores the answer to an auth user update as the signed-in user, so a placeholder
      // here leaves user.id undefined and every later query asks for id=eq.undefined (HTTP 400).
      // Answer with the real user, carrying the requested metadata change.
      if (url.pathname === '/auth/v1/user') {
        const real = await route.fetch({ method: 'GET', postData: '' });
        let user: Record<string, unknown> = {};
        try { user = await real.json(); } catch { /* keep {} */ }
        let change: { data?: Record<string, unknown> } = {};
        try { change = req.postDataJSON() ?? {}; } catch { /* no body */ }
        if (change.data) user.user_metadata = { ...(user.user_metadata as object ?? {}), ...change.data };
        return route.fulfill({ status: real.status(), contentType: 'application/json', body: JSON.stringify(user) });
      }
      // Answer like PostgREST would with return=representation: the written row(s) echoed back.
      let body = '{"success":true}';
      if (url.pathname.startsWith('/rest/')) {
        let rows: unknown = {};
        try { rows = req.postDataJSON() ?? {}; } catch { /* no body */ }
        const list = (Array.isArray(rows) ? rows : [rows]).map((r) => ({ id: randomUUID(), ...(r as object) }));
        body = JSON.stringify(/vnd\.pgrst\.object/.test(req.headers()['accept'] ?? '') ? list[0] : list);
      }
      return route.fulfill({ status: 200, contentType: 'application/json', body });
    }
    return route.fallback();
  };
  await page.route('**/*', intercept);
  await page.addInitScript({ content: `window.__activeModal = ${ACTIVE_MODAL.toString()};` });
  await page.addInitScript(() => {
    const w = window as unknown as CaptureWindow;
    w.__fx = { mut: 0, clip: 0, open: 0 };
    w.__crawlVisual = () => {
      w.__visualIds ??= new WeakMap();
      w.__visualNext ??= 0;
      const visual: Visual = [];
      for (const el of Array.from(document.querySelectorAll('body *'))) {
        if (!el.checkVisibility({ opacityProperty: true, visibilityProperty: true })) continue;
        let id = w.__visualIds.get(el);
        if (id === undefined) { id = ++w.__visualNext; w.__visualIds.set(el, id); }
        const s = getComputedStyle(el);
        // Pending/delayed transitions can look stable during the idle samples.
        const animated = el.getAnimations().some(a => a.pending || a.playState === 'running');
        visual.push([id, [s.backgroundColor,s.color,s.opacity,s.transform,s.borderColor,s.width,s.height].join('|'), animated]);
      }
      return visual;
    };
    const captureNativeInteraction = (event: Event) => {
      const action = w.__crawlAction;
      if (!action || action.activated || !(event.target instanceof Element) || !action.target.contains(event.target)) return;
      // Observe native hover/focus styles before application handlers run. Do not
      // move the pointer or blur anything: children and deliberate focus transfers
      // must remain available after the click. Focus elsewhere is the result.
      action.visual = w.__crawlVisual!();
    };
    // Press-handler output precedes native focus, which may replace visual.
    // Observe each propagation node after its existing handlers, including
    // handlers that stop propagation. Never change pointer or focus state.
    w.__crawlPressResult = event => {
      const action = w.__crawlAction;
      const baseline = action?.pressBaselines.get(event);
      if (!action || action.activated || !baseline) return;
      const before = new Map(baseline.map(([id,value,animated]) => [id,{value,animated}]));
      for (const [id,value,animated] of w.__crawlVisual!()) {
        const prior = before.get(id);
        if (prior && !prior.animated && !animated && prior.value !== value
          && !action.pressResults.some(([seen,result]) => seen === id && result === value)) action.pressResults.push([id,value]);
      }
    };
    const capturePress = (event: Event) => {
      const action = w.__crawlAction;
      if (!action || action.activated || !(event.target instanceof Element) || !action.target.contains(event.target)) return;
      action.pressBaselines.set(event, w.__crawlVisual!());
      for (const node of event.composedPath()) {
        if (node === window) continue;
        for (const capture of [true, false]) {
          const listener = w.__crawlPressResult!;
          node.addEventListener(event.type, listener, { capture, once: true });
          action.cleanup.push(() => node.removeEventListener(event.type, listener, capture));
        }
      }
    };
    window.addEventListener('pointerdown', capturePress, true);
    window.addEventListener('mousedown', capturePress, true);
    window.addEventListener('mouseover', captureNativeInteraction, true);
    window.addEventListener('focus', captureNativeInteraction, true);
    window.addEventListener('click', event => {
      const action = w.__crawlAction;
      if (action && event.target instanceof Element && action.target.contains(event.target)) action.activated = true;
    }, true); // A handler's later focus/mouse events are results, not new baselines.
    new MutationObserver((list) => {
      for (const m of list) if (!(m.type === 'attributes' && m.attributeName === 'style')) w.__fx.mut++;
    }).observe(document, { subtree: true, childList: true, attributes: true, characterData: true });
    const clip = navigator.clipboard;
    if (clip) {
      const orig = clip.writeText?.bind(clip);
      if (orig) clip.writeText = async (t: string) => { await orig(t); w.__fx.clip++; };
      const origWrite = clip.write?.bind(clip);
      if (origWrite) clip.write = async (d) => { await origWrite(d); w.__fx.clip++; };
    }
    const origExec = document.execCommand.bind(document);
    document.execCommand = (c: string, ...rest: unknown[]) => {
      const succeeded = origExec(c, ...(rest as [boolean, string]));
      if (c === 'copy' && succeeded) w.__fx.clip++;
      return succeeded;
    };
    const origOpen = window.open.bind(window);
    window.open = (...a: Parameters<typeof window.open>) => { w.__fx.open++; return origOpen(...a); };
  });

  let current = { page: '', control: '' };
  function resourceLabel(value: string): string {
    try {
      const url=new URL(value);
      if(!['http:','https:'].includes(url.protocol)) return `${url.protocol}(inline)`;
      // Keep endpoint attribution, without query tokens or external session paths.
      const path=url.origin===origin||SUPABASE_HOST.test(url.hostname)
        ? url.pathname : '/'+url.pathname.split('/').filter(Boolean).slice(0,2).join('/');
      return url.origin+path;
    } catch { return '(unknown source)'; }
  }
  on('requestfailed',request=>{
    const failure=request.failure()?.errorText??'unknown network failure';
    if(!/ERR_(ABORTED|BLOCKED)/.test(failure)) console.log(`[crawl ${tag}] request failure ${request.method()} ${resourceLabel(request.url())}: ${failure}`);
  });
  const dialogs: string[] = [];
  on('dialog', async (d) => {
    dialogs.push(`${d.type()}: ${d.message()}`);
    if (d.type() === 'alert' && /error|fail|unable|could not|couldn't|try again|being configured|not configured|unavailable/i.test(d.message())) {
      add(current.page, current.control, `alert shows a failure: "${d.message()}"`);
    }
    await (d.type() === 'alert' ? d.accept() : d.dismiss()).catch(() => {});
  });
  on('pageerror', (e) => add(current.page, current.control || '(page load)', `page error: ${e.message}`));
  on('console', (m) => {
    if (m.type() !== 'error') return;
    const t = m.text();
    if (/Failed to load resource: net::ERR_(ABORTED|BLOCKED)|favicon|stubbed by crawler/.test(t)) return;
    // Failed responses are reported once, with their URL, by the response handler below.
    if (/Failed to load resource: the server responded with a status of/.test(t)) return;
    // Console messages do not identify the failed request. An unrelated navigation
    // abort cannot safely explain this error, so retain it synchronously.
    add(current.page, current.control || '(page load)', `console error: ${t.slice(0, 200)} [source: ${resourceLabel(m.location().url)}]`);
  });

  const signedIn = !!o.persona;
  const pending: Request[] = [];
  const responses: Response[] = [];
  on('request', (r) => { if (['fetch', 'xhr', 'document', 'eventsource'].includes(r.resourceType())) pending.push(r); });
  on('response', (r) => responses.push(r));

  async function ensureSession() {
    if (!o.persona) return;
    // An external checkout/portal navigation cannot read this site's localStorage. Return to
    // this origin before deciding whether a login is necessary; /login redirects an existing
    // session to /account, where waiting for email fields would otherwise hang the crawl.
    if (!page.url().startsWith(origin)) await page.goto(origin, { timeout: 30_000 });
    if (page.url().startsWith(origin) && await hasSession(page).catch(() => false)) return;
    await signIn(page, o.persona);
  }

  async function load(path: string): Promise<boolean> {
    current = { page: path, control: '' };
    await ensureSession();
    // A context restored from storage can already be signed in: ensureSession
    // visits this origin to inspect it without calling signIn. Let its bootstrap
    // requests settle before our forced visit, just as after a fresh login.
    await page.waitForLoadState('networkidle', { timeout: 10_000 });
    responses.length = 0;
    const res = await page.goto(new URL(path, origin).href, { waitUntil: 'load', timeout: 30_000 }).catch((e) => { add(path, '(page load)', `navigation failed: ${e}`); return null; });
    if (!res) return false;
    if (res.status() >= 400) { add(path, '(page load)', `HTTP ${res.status()}`); return false; }
    await page.waitForLoadState('networkidle', { timeout: 10_000 }).catch(() => {});
    await revealAll();
    const landed = new URL(page.url()).pathname;
    if (signedIn && /^\/login\/?$/.test(landed) && path !== '/login/') {
      add(path, '(page load)', 'a signed-in user was sent to the login page');
      return false;
    }
    checkResponses(path, '(page load)');
    return true;
  }

  // Scroll-triggered reveals start at opacity 0; scroll through so every section is shown.
  async function revealAll() {
    await page.evaluate(async () => {
      const step = Math.max(200, Math.floor(window.innerHeight * 0.8));
      for (let y = 0; y < document.documentElement.scrollHeight; y += step) {
        // CSS scroll-smooth otherwise cancels each still-running scroll with
        // the next one, leaving most scroll-triggered controls unrevealed.
        window.scrollTo({top:y,left:0,behavior:'instant'});
        await new Promise((r) => setTimeout(r, 60));
      }
      window.scrollTo({top:0,left:0,behavior:'instant'});
    }).catch(() => {});
    await page.waitForTimeout(400);
  }

  function checkResponses(pg: string, control: string) {
    for (const r of responses.splice(0)) {
      const u = new URL(r.url());
      const ours = u.origin === origin || SUPABASE_HOST.test(u.hostname);
      if (!ours || r.status() < 400) continue;
      if (r.request().resourceType() === 'image' || r.request().resourceType() === 'font') { add(pg, control, `asset ${u.pathname} HTTP ${r.status()}`); continue; }
      const expected = o.expectedStatus?.some((e) => e.control.test(`${pg} ${control}`) && e.url.test(u.pathname) && e.status === r.status());
      // Signed out, an auth 4xx (wrong password, unconfirmed email) is the expected answer to a form.
      const anonAuth = !signedIn && u.pathname.startsWith('/auth/v1/') && r.status() < 500;
      if (!expected && !anonAuth) add(pg, control, `${r.request().method()} ${u.pathname} answered HTTP ${r.status()}`);
    }
  }

  function enqueue(p: string) {
    if (queued.has(p) || !o.inScope(p)) return;
    queued.add(p);
    const fam = familyOf(p);
    const n = fam ? familyVisits.get(fam) ?? 0 : 0;
    if (fam && n >= perFamily) { httpOnly.push(p); return; }
    if (fam) familyVisits.set(fam, n + 1);
    queue.push(p);
  }

  async function harvestLinks(pg: string, links: LinkInfo[], ids: string[]) {
    for (const l of links) {
      const raw = (l.raw ?? '').trim();
      if (raw === '' || raw === '#' || raw.startsWith('javascript:')) {
        add(pg, `link "${l.label}"`, `link goes nowhere (href="${l.raw ?? ''}")`);
        continue;
      }
      if (raw.startsWith('#')) {
        if (!ids.includes(decodeURIComponent(raw.slice(1)))) add(pg, `link "${l.label}"`, `in-page anchor ${raw} has no target element`);
        continue;
      }
      if (raw.startsWith('mailto:')) {
        if (!/^mailto:[^@\s]+@[^@\s]+\.[a-z]{2,}/i.test(raw)) add(pg, `link "${l.label}"`, `malformed ${raw}`);
        continue;
      }
      if (raw.startsWith('tel:')) continue;
      let u: URL;
      try { u = new URL(l.href); } catch { add(pg, `link "${l.label}"`, `unparseable href ${raw}`); continue; }
      u.hash = '';
      if (u.origin === origin) {
        if (!internalLinks.has(u.href)) internalLinks.set(u.href, pg);
        const p = u.pathname.endsWith('/') || /\.[a-z0-9]+$/i.test(u.pathname) ? u.pathname : `${u.pathname}/`;
        if (!/\.[a-z0-9]+$/i.test(p)) enqueue(p);
      } else if (/^https?:$/.test(u.protocol)) {
        if (!externalLinks.has(u.href)) externalLinks.set(u.href, pg);
      }
    }
  }

  async function snapshot() {
    return page.evaluate(() => {
      const w = window as unknown as CaptureWindow;
      let visibleCount = 0;
      for (const el of Array.from(document.querySelectorAll('body *'))) {
        const r = (el as HTMLElement).getBoundingClientRect();
        if (r.width > 0 && r.height > 0) visibleCount++;
      }
      const text = document.body.innerText;
      let h = 0;
      for (let i = 0; i < text.length; i++) h = (h * 31 + text.charCodeAt(i)) | 0;
      // Control state does not move on its own the way text and layout can while a signed-in page
      // settles: selected tabs, expanded or pressed controls, open details, shown panels, field values.
      const parts: string[] = [];
      for (const el of Array.from(document.querySelectorAll('[aria-selected],[aria-expanded],[aria-pressed],[aria-checked],details,[role="tabpanel"],select,input,textarea'))) {
        const e = el as HTMLInputElement;
        if (!e.checkVisibility({ opacityProperty: true, visibilityProperty: true })) continue;
        parts.push([e.getAttribute('aria-selected'), e.getAttribute('aria-expanded'), e.getAttribute('aria-pressed'), e.getAttribute('aria-checked'),
          (e as unknown as HTMLDetailsElement).open, e.hidden || e.classList.contains('hidden'), 'value' in e ? e.value : '', e.checked].join(','));
      }
      const visual = w.__crawlVisual?.() ?? [];
      const state = parts.join('|');
      let sh = 0;
      for (let i = 0; i < state.length; i++) sh = (sh * 31 + state.charCodeAt(i)) | 0;
      return {
        url: location.href, visibleCount, textHash: h, stateHash: sh, visual,
        interactionVisual: w.__crawlAction?.visual ?? null,
        pressResults: w.__crawlAction?.pressResults ?? [], htmlClass: document.documentElement.className,
        mut: w.__fx?.mut ?? 0, clip: w.__fx?.clip ?? 0, open: w.__fx?.open ?? 0,
      };
    }).catch(() => null);
  }

  async function fillForm(button: ElementHandle<HTMLElement | SVGElement>) {
    const formHandle = await button.evaluateHandle((el) => (el as HTMLButtonElement).form);
    const formToken = randomUUID();
    const fields = await formHandle.evaluate((f, token) => f ? Array.from((f as HTMLFormElement).elements).map((e, i) => {
      const el = e as HTMLInputElement;
      el.setAttribute('data-crawl-field', `${token}-${i}`);
      return { i, tag: el.tagName.toLowerCase(), type: (el.type || '').toLowerCase(), value: el.value, required: el.required, name: `${el.name} ${el.id}` };
    }) : [], formToken);
    const password = randomPassword();
    const email = testUserEmail('crawl');
    for (const f of fields) {
      const loc = page.locator(`[data-crawl-field="${formToken}-${f.i}"]`);
      if (!(await loc.isVisible().catch(() => false))) continue;
      if (f.tag === 'select') { await loc.selectOption({ index: 1 }).catch(() => {}); continue; }
      if (f.type === 'checkbox') { if (f.required) await loc.check().catch(() => {}); continue; }
      if (['submit', 'button', 'hidden', 'radio', 'file', 'reset'].includes(f.type) || f.tag === 'button' || f.tag === 'fieldset') continue;
      if (f.value && f.type !== 'password') continue;
      const v = f.type === 'email' ? email : f.type === 'password' ? password : f.type === 'number' ? '1'
        : f.type === 'url' ? 'https://example.com' : f.type === 'date' ? '2030-01-01' : /name/i.test(f.name) ? 'E2E Crawler' : 'e2e crawler';
      await loc.fill(v).catch(() => {});
      if (f.type === 'email' && await loc.inputValue().catch(() => '') === email) o.generatedEmails?.add(email);
    }
  }

  async function clearActionCapture() {
    await page.evaluate(() => {
      const w = window as unknown as CaptureWindow;
      for (const cleanup of w.__crawlAction?.cleanup ?? []) cleanup();
      delete w.__crawlAction;
    }).catch(() => {});
  }

  // Clicks one control and reports what it did. Returns 'nav' when the page changed.
  async function act(pg: string, c: ControlInfo): Promise<'nav' | 'same'> {
    const loc = await page.locator(c.selector).first().elementHandle();
    const verified = loc && (await page.evaluate(ENUMERATE, { path: pg, target: loc })).controls.find(control => control.isTarget);
    if (!loc || !verified || verified.key !== c.key || verified.recordId !== c.recordId) {
      add(pg, c.key, 'control identity changed before the action; coverage was not credited');
      return 'same';
    }
    // Pin the actual node: positional selectors must not silently click its sibling.
    current = { page: pg, control: c.key };
    if (c.inForm) await fillForm(loc);
    await loc.scrollIntoViewIfNeeded({ timeout: 3_000 }).catch(() => {});
    // Two idle snapshots: a signal that moves on its own (animation, clock) cannot count as an effect.
    const idle = await snapshot();
    await page.waitForTimeout(300);
    const before = await snapshot();
    if (!before || !idle) return 'nav';
    const finalIdentity = (await page.evaluate(ENUMERATE, { path: pg, target: loc })).controls.find(control => control.isTarget);
    if (!finalIdentity || finalIdentity.key !== c.key || finalIdentity.recordId !== c.recordId) {
      add(pg, c.key, 'control identity changed before the action; coverage was not credited');
      return 'same';
    }
    exercised.add(c.key);
    console.log(`[crawl ${tag}]   click ${c.key}`);
    const noisy = { mut: before.mut !== idle.mut, text: before.textHash !== idle.textHash, count: before.visibleCount !== idle.visibleCount, state: before.stateHash !== idle.stateHash };
    const idleVisual = new Map(idle.visual.map(([id,value,animated]) => [id,{value,animated}]));
    const stableVisual = new Map(before.visual
      .filter(([id,value,animated]) => !animated && !idleVisual.get(id)?.animated && idleVisual.get(id)?.value === value)
      .map(([id,value]) => [id,value]));
    pending.length = 0;
    responses.length = 0;
    const navsBefore = externalNavs.length, dialogsBefore = dialogs.length;
    const popup = page.context().waitForEvent('page', { timeout: 2_500 }).catch(() => null);
    const download = page.waitForEvent('download', { timeout: 2_500 }).catch(() => null);
    try {
      await loc.evaluate(el => {
        const w = window as unknown as CaptureWindow;
        const action: CapturedAction = { target: el, pressBaselines: new WeakMap(), pressResults: [], cleanup: [] };
        w.__crawlAction = action;
        // Registered after application window handlers; descendants are added
        // by capturePress before dispatch reaches them.
        for (const type of ['pointerdown', 'mousedown']) for (const capture of [true, false]) {
          const listener = w.__crawlPressResult!;
          window.addEventListener(type, listener, capture);
          action.cleanup.push(() => window.removeEventListener(type, listener, capture));
        }
      });
      if (c.tag === 'select') {
        // Clicking a native select changes nothing; choosing another option fires input and change.
        const next = await loc.evaluate((el) => {
          const sel = el as HTMLSelectElement;
          return Array.from(sel.options).findIndex((op, k) => k !== sel.selectedIndex && !op.disabled);
        });
        if (next < 0) { add(pg, c.key, 'select has no other enabled option to choose'); await clearActionCapture(); return 'same'; }
        await loc.selectOption({ index: next }, { timeout: 5_000 });
      } else {
        await loc.click({ timeout: 5_000 });
      }
    } catch (e) {
      const msg = String(e).split('\n').find((l) => /intercepts pointer|not visible|outside of the viewport|detached/.test(l)) ?? String(e).split('\n')[0];
      add(pg, c.key, `cannot be clicked: ${msg.trim().slice(0, 200)}`);
      await clearActionCapture();
      return 'nav';
    }
    // Give the handler time to do something, then let its requests finish.
    let after = before;
    for (let t = 0; t < 12; t++) {
      await page.waitForTimeout(250);
      const s = await snapshot();
      if (!s) { after = { ...before, url: 'navigated' }; break; }
      after = s;
      if (s.url !== before.url || (!noisy.mut && s.mut !== before.mut) || s.clip !== before.clip || pending.length || dialogs.length > dialogsBefore) break;
    }
    const opened = (await popup) ?? null;
    const dl = (await download) ?? null;
    const deadline = Date.now() + 10_000;
    let observed = 0;
    while (Date.now() < deadline) {
      const batch = pending.slice(observed);
      observed = pending.length;
      if (!batch.length) {
        await page.waitForTimeout(Math.min(350, Math.max(1, deadline - Date.now())));
        if (pending.length === observed) break;
        continue;
      }
      await Promise.race([
        Promise.all(batch.map(async r => {
          const response = await r.response().catch(() => null);
          await response?.finished().catch(() => null);
        })),
        page.waitForTimeout(Math.max(1, deadline - Date.now())),
      ]);
    }
    await page.waitForLoadState('load', { timeout: 10_000 }).catch(() => {});
    await page.waitForTimeout(100);
    if (opened) await opened.close().catch(() => {});
    // Read the resulting state without undoing a revealed panel or focused field.
    const settling = (await snapshot()) ?? after;
    await page.waitForTimeout(300);
    const final = (await snapshot()) ?? after;
    const settledVisual = new Map(settling.visual.map(([id,value,animated]) => [id,{value,animated}]));
    const interactionVisual = new Map((final.interactionVisual ?? before.visual)
      .map(([id,value,animated]) => [id,{value,animated}]));
    await clearActionCapture();
    const effect = final.url !== before.url || after.url !== before.url || externalNavs.length > navsBefore
      || !!opened || !!dl || dialogs.length > dialogsBefore
      || final.clip > before.clip
      || (!noisy.mut && final.mut > before.mut) || (!noisy.text && final.textHash !== before.textHash)
      || (!noisy.count && final.visibleCount !== before.visibleCount)
      || (!noisy.state && final.stateHash !== before.stateHash)
      || final.visual.some(([id,value,animated]) => stableVisual.has(id) && !animated
        && !settledVisual.get(id)?.animated && settledVisual.get(id)?.value === value
        && ((interactionVisual.has(id) && !interactionVisual.get(id)!.animated && interactionVisual.get(id)!.value !== value)
          || final.pressResults.some(([resultId,result]) => resultId === id && result === value)))
      || final.htmlClass !== before.htmlClass;
    if (!effect) add(pg, c.key, `clicking "${c.label}" did nothing visible (no navigation, dialog, clipboard or screen change)`);
    checkResponses(pg, c.key);
    for (const nav of externalNavs.slice(navsBefore)) {
      if (!externalLinks.has(nav)) externalLinks.set(nav, `${pg} via ${c.key}`);
    }
    const landedElsewhere = final.url !== before.url || externalNavs.length > navsBefore;
    if (landedElsewhere) {
      const u = new URL(final.url.startsWith('http') ? final.url : before.url);
      if (u.origin === origin) enqueue(u.pathname);
    }
    return landedElsewhere ? 'nav' : 'same';
  }

  async function reachable(selector: string) {
    try { await page.locator(selector).first().scrollIntoViewIfNeeded({ timeout: 3_000 }); }
    catch { return false; }
    return page.evaluate(REACHABLE, selector).catch(() => false);
  }

  async function explore(pg: string, path: string[], depth: number, doneOnPage: Set<string>) {
    // Auth or a replayed control can commit a new document without waiting for
    // its stylesheet. Visibility checks must observe the completed document.
    await page.waitForLoadState('load');
    const info = await page.evaluate(ENUMERATE, pg).catch(() => null);
    if (!info) return;
    await harvestLinks(pg, info.links, info.ids);
    const todo = info.controls.filter((c) => {
      keys.add(c.key);
      if (c.disabled) { if (c.visible && !enabledSeen.has(c.key)) disabledSeen.set(c.key, pg); return false; }
      if (!c.visible) return false;
      if (!enabledSeen.has(c.key)) enabledSeen.set(c.key, pg);
      disabledSeen.delete(c.key);
      if (doneOnPage.has(JSON.stringify([c.selector, c.key])) || exercised.has(c.key) || o.skipKeys?.has(c.key)) return false;
      if (o.dedicated.some(([re]) => re.test(c.key))) return false;
      return true;
    });
    if (!todo.length) return;
    const restoredKeys = new Set<string>();
    for (const c of todo) {
      const resolve = async () => {
        const controls = (await page.evaluate(ENUMERATE, pg)).controls;
        return controls.find(control => control.key === c.key && control.selector === c.selector)
          ?? controls.find(control => control.key === c.key && control.visible && !control.disabled);
      };
      let candidate = await resolve();
      // Repeated table rows can share a key. An unreachable row must not trigger
      // hundreds of identical reload/replay attempts from the same sibling list.
      const attempt = JSON.stringify([c.selector, c.key]);
      if (exercised.has(c.key) || doneOnPage.has(attempt)) continue;
      doneOnPage.add(attempt);
      // Return to the state this control was found in: fresh page plus the clicks that revealed it.
      // A sibling's click may have opened a modal over this control: start again from a fresh page.
      if (new URL(page.url()).pathname !== pg || !candidate || !(await reachable(candidate.selector))) {
        if (restoredKeys.has(c.key)) continue;
        restoredKeys.add(c.key);
        if (!(await load(pg))) return;
        for (const step of path) await page.locator(step).first().click({ timeout: 5_000 }).catch(() => {});
        await page.waitForTimeout(300);
        candidate = await resolve();
        if (!candidate || !(await reachable(candidate.selector))) continue;
      }
      const result = await act(pg, candidate);
      restoredKeys.clear();
      if (result === 'same' && depth < maxDepth) {
        // A blocked control in this child state can still be reachable from
        // the parent. Only successful actions are deduplicated across states.
        await explore(pg, [...path, candidate.selector], depth + 1, new Set());
      } else if (result === 'same') {
        // Reaching the exploration limit must not silently count newly
        // revealed controls as covered. Report them unless another path
        // exercises them or a dedicated test explicitly owns them.
        const revealed = await page.evaluate(ENUMERATE, pg);
        for (const control of revealed.controls) {
          keys.add(control.key);
          if (control.visible && !control.disabled) {
            if (!enabledSeen.has(control.key)) enabledSeen.set(control.key, pg);
            disabledSeen.delete(control.key);
          }
        }
      }
      if (o.persona && !(await hasSession(page).catch(() => false))) await ensureSession();
    }
    // Collapsing a section or finishing an asynchronous render can expose controls that were
    // absent from the original sibling list. Re-enumerate this state until no new controls
    // remain; doneOnPage prevents retries of unreachable controls and cycles.
    if (depth < maxDepth && new URL(page.url()).pathname === pg) await explore(pg, path, depth, doneOnPage);
  }

  for (const seed of queue) { const f = familyOf(seed); if (f) familyVisits.set(f, (familyVisits.get(f) ?? 0) + 1); }
  const ANCHOR = /<a\b[^>]*?\bhref\s*=\s*(?:"([^"]*)"|'([^']*)')[^>]*>([\s\S]*?)<\/a>/gi;
  const ID = /\bid\s*=\s*["']([^"']+)["']/gi;
  let fetched = 0;
  // A page fetched over HTTP can link to a new template family, which goes to the browser queue:
  // keep going until both queues are empty.
  while (queue.length || httpOnly.length) {
    while (queue.length) {
      const pg = queue.shift()!;
      const started = Date.now();
      if (!(await load(pg))) continue;
      visited.push(pg);
      const before = exercised.size;
      await explore(pg, [], 0, new Set());
      console.log(`[crawl ${tag}] ${pg}: ${exercised.size - before} controls in ${((Date.now() - started) / 1000).toFixed(1)}s (${queue.length} pages queued)`);
    }

    // ---- templated pages beyond the browser sample: fetch each, check every link on it ----
    if (httpOnly.length) {
      const batch = httpOnly.splice(0);
      await pool(batch, 16, async (pg) => {
        const url = new URL(pg, origin).href;
        const res = await request.get(url, { timeout: 30_000 }).catch(() => null);
        fetched++;
        pageStatus.set(url, res?.status() ?? 0);
        if (!res || res.status() >= 400) { add(pg, '(page load)', res ? `HTTP ${res.status()}` : 'no response'); await res?.dispose(); return; }
        // The request context keeps every body until it is disposed; 12K pages would pile up.
        let html: string;
        try { html = await res.text(); } finally { await res.dispose(); }
        const ids = [...html.matchAll(ID)].map((m) => m[1]);
        const links: LinkInfo[] = [...html.matchAll(ANCHOR)].map((m) => {
          const raw = m[1] ?? m[2] ?? '';
          let href = raw;
          try { href = new URL(raw.replace(/&amp;/g, '&'), url).href; } catch { /* reported below */ }
          return { raw, href, label: m[3].replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim().slice(0, 60), key: '' };
        });
        await harvestLinks(pg, links, ids);
      });
      console.log(`[crawl ${tag}] fetched ${fetched} templated pages over HTTP (${httpOnly.length} more queued)`);
    }
  }

  // ---- link targets ----
  console.log(`[crawl ${tag}] checking ${internalLinks.size} internal and ${externalLinks.size} external link targets`);
  current = { page: '', control: '' };
  await pool([...internalLinks], 16, async ([url, pg]) => {
    const known = pageStatus.get(url) ?? pageStatus.get(url.endsWith('/') ? url : `${url}/`);
    let status = known;
    if (status === undefined) {
      const res = await request.get(url, { maxRedirects: 5, timeout: 30_000 }).catch(() => null);
      status = res?.status() ?? 0;
      await res?.dispose();
    }
    if (status === 0 || status >= 400) add(pg, `link ${new URL(url).pathname}`, `target answered ${status ? `HTTP ${status}` : 'nothing'}`);
  });
  await pool([...externalLinks], 8, async ([url, pg]) => {
    let res = await request.head(url, { maxRedirects: 5, timeout: 20_000 }).catch(() => null);
    if (!res || res.status() === 405 || res.status() === 404) {
      await res?.dispose();
      res = await request.get(url, { maxRedirects: 5, timeout: 20_000 }).catch(() => null);
    }
    if (!res) add(pg, `link ${url}`, 'external target unreachable');
    else if (res.status() >= 400 && !BENIGN_EXTERNAL_STATUS.has(res.status())) add(pg, `link ${url}`, `external target answered HTTP ${res.status()}`);
    await res?.dispose();
  });

  // ---- coverage contract ----
  for (const [key, pg] of enabledSeen) {
    if (!exercised.has(key) && !o.skipKeys?.has(key) && !o.dedicated.some(([re]) => re.test(key))) {
      add(pg, key, `visible enabled control was not exercised (unreachable state or exploration depth limit; viewport ${page.viewportSize()?.width ?? 'unknown'}px)`);
    }
  }
  for (const [key, pg] of disabledSeen) {
    if (!o.dedicated.some(([re]) => re.test(key))) add(pg, key, 'control is disabled and nothing in the crawl enabled it; cover it with a dedicated test and list it');
  }
  for (const [re, why, opts] of o.dedicated) {
    if (!opts?.optional && ![...keys].some((k) => re.test(k))) add('(dedicated list)', String(re), `matches no control on any page (stale entry for: ${why})`);
  }

  return { findings, pages: visited, httpPages: fetched, exercised: [...exercised], links: internalLinks.size + externalLinks.size, keys };
  } finally {
    for (const [event, listener] of ownedListeners) emitter.removeListener(event, listener);
    if (intercept) await page.unroute('**/*', intercept);
  }
}

export function formatFindings(f: Finding[]): string {
  return f.map((x) => `  ${x.page} | ${x.control}\n      -> ${x.problem}`).join('\n');
}
