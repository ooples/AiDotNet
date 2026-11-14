import { test, expect } from '@playwright/test';
import { readdirSync, statSync } from 'node:fs';
import { join, relative, sep } from 'node:path';

// Every dialog on every page must be usable on a short laptop screen and on a phone: each
// control inside it has to be scrollable into view, inside the viewport, and not covered.
//
// The Issue License dialog shipped taller than a laptop viewport with no scroll container,
// so its submit button was unreachable and an admin had to zoom out to issue a key. That
// went unnoticed because no test opened a dialog at a realistic height.
//
// Pages and dialogs are discovered, not listed: every static route under src/pages, every
// overlay element in its HTML. Scripts are disabled so the auth-gated pages render their
// static markup without redirecting; the dialog is shown by removing its `hidden` class,
// which is exactly what the page scripts do.

const VIEWPORTS = [
  { name: 'laptop 1280x650', width: 1280, height: 650 },
  { name: 'phone 390x664', width: 390, height: 664 },
];

function staticRoutes(): string[] {
  const root = join(process.cwd(), 'src', 'pages');
  const out: string[] = [];
  const walk = (dir: string) => {
    for (const name of readdirSync(dir)) {
      const full = join(dir, name);
      if (statSync(full).isDirectory()) { walk(full); continue; }
      if (!name.endsWith('.astro') || name.includes('[')) continue;
      const rel = relative(root, full).split(sep).join('/').replace(/\.astro$/, '');
      const route = rel === 'index' ? '/' : `/${rel.replace(/(^|\/)index$/, '')}/`;
      out.push(route.replace(/\/+/g, '/'));
    }
  };
  walk(root);
  return out.sort();
}

const DIALOG_SELECTOR = '[role="dialog"], [id$="-modal"], .fixed.inset-0';

test.use({ javaScriptEnabled: false });

for (const vp of VIEWPORTS) {
  test(`every dialog's controls are reachable at ${vp.name}`, async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== 'anon-desktop', 'viewports are set explicitly; run once');
    test.setTimeout(180_000);
    await page.setViewportSize({ width: vp.width, height: vp.height });

    const problems: string[] = [];
    let dialogsChecked = 0;
    for (const route of staticRoutes()) {
      const res = await page.goto(route);
      if (!res || res.status() >= 400) continue;
      const ids: string[] = await page.$$eval(DIALOG_SELECTOR, (els) =>
        els.map((el, i) => {
          if (!el.id) el.id = `__dialog_${i}`;
          return el.id;
        }),
      );
      for (const id of [...new Set(ids)]) {
        const found = await page.evaluate((dialogId) => {
          const dialog = document.getElementById(dialogId)!;
          // Show this dialog and only this one, as the page scripts would.
          document.querySelectorAll<HTMLElement>('[id$="-modal"], [role="dialog"], .fixed.inset-0')
            .forEach((d) => { if (d !== dialog) d.classList.add('hidden'); });
          dialog.classList.remove('hidden');
          const issues: string[] = [];
          const controls = dialog.querySelectorAll<HTMLElement>('button, input:not([type="hidden"]), select, textarea, a[href]');
          for (const el of controls) {
            const style = getComputedStyle(el);
            if (el.offsetParent === null || style.visibility === 'hidden') continue; // not shown in this state
            el.scrollIntoView({ block: 'nearest', inline: 'nearest' });
            const r = el.getBoundingClientRect();
            const label = (el.getAttribute('aria-label') || el.textContent || el.id || el.tagName).trim().slice(0, 40);
            if (r.top < -1 || r.left < -1 || r.bottom > innerHeight + 1 || r.right > innerWidth + 1) {
              issues.push(`"${label}" cannot be scrolled into view (rect ${Math.round(r.left)},${Math.round(r.top)} to ${Math.round(r.right)},${Math.round(r.bottom)}; viewport ${innerWidth}x${innerHeight})`);
              continue;
            }
            const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
            if (hit && hit !== el && !el.contains(hit) && !(hit instanceof HTMLLabelElement && hit.control === el)) {
              issues.push(`"${label}" is covered by <${hit.tagName.toLowerCase()}${hit.id ? '#' + hit.id : ''}>`);
            }
          }
          dialog.classList.add('hidden');
          return { count: controls.length, issues };
        }, id);
        if (found.count === 0) continue;
        dialogsChecked++;
        for (const issue of found.issues) problems.push(`${route} #${id}: ${issue}`);
      }
    }

    expect(dialogsChecked, 'the scan must find the known dialogs; zero means the selector broke').toBeGreaterThanOrEqual(5);
    expect(problems).toEqual([]);
  });
}
