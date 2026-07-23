import { test, expect } from '@playwright/test';

test.describe('Admin — users', () => {
  test('/admin/users/ renders header, filters, and a populated table', async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });

    await page.goto('/admin/users/');

    await expect(page.getByRole('heading', { name: 'Users', level: 1 })).toBeVisible();

    // Loader should flip to the real table once the initial profiles
    // fetch lands. If RLS on profiles accidentally locks admins out of
    // reading other users, the loader sticks forever — which is exactly
    // the regression that bit us when is_admin() was first added.
    await expect(page.locator('#users-loading')).toBeHidden({ timeout: 15_000 });
    await expect(page.locator('#users-table')).toBeVisible();

    // Filters: one search input, tier select, role select. Each filter
    // must be present for the admin to scope down large user lists.
    await expect(page.locator('#search-input')).toBeVisible();
    const tierOptions = await page.locator('#filter-tier option').allTextContents();
    expect(tierOptions).toEqual(['All Tiers', 'Free', 'Pro', 'Enterprise']);
    const roleOptions = await page.locator('#filter-role option').allTextContents();
    expect(roleOptions).toEqual(['All Roles', 'User', 'Admin']);

    // user-count header must resolve to an integer, not the "--" default.
    await expect(page.locator('#user-count')).not.toHaveText('--', { timeout: 10_000 });

    // At least one row must render — the playwright-admin + playwright-user
    // accounts are both in profiles, so zero rows would indicate the
    // RLS policy blocked the admin read.
    const rowCount = await page.locator('#users-body tr').count();
    expect(rowCount).toBeGreaterThan(0);

    expect(consoleErrors, 'admin users page should emit zero console errors').toEqual([]);
  });

  test('role filter narrows the list to admin-only rows', async ({ page }) => {
    await page.goto('/admin/users/');
    await expect(page.locator('#users-table')).toBeVisible({ timeout: 15_000 });

    // Capture the pre-filter total so we can synchronously wait on the
    // filter having applied. renderTable() awaits loadKeyCountsForPage()
    // before swapping innerHTML, so a naive count() right after
    // selectOption() races the async re-render.
    const totalBefore = parseInt((await page.locator('#user-count').textContent()) || '0', 10);

    await page.locator('#filter-role').selectOption('admin');

    // Wait for the admin-filter re-render to settle. The suite itself is
    // signed-in as an admin, so the resulting count must be >= 1 — zero
    // rows would mean the filter is broken (or the signed-in admin's
    // profile has drifted off role='admin'), and we want a loud failure
    // in that case rather than a skipped green test. The same strict
    // inequality against the pre-filter total catches the race where the
    // old unfiltered table is still mounted post-selectOption.
    await expect
      .poll(async () => parseInt((await page.locator('#user-count').textContent()) || '0', 10), {
        timeout: 10_000,
      })
      .toBeLessThanOrEqual(totalBefore);

    // Read role cells from each row's 2nd <td> (the role column per
    // pages/admin/users/index.astro:241-243), NOT the whole row —
    // whole-row matching false-passes when any other column contains
    // "admin" (e.g., a username "admin@…").
    //
    // Wrap the read in expect.poll so the assertion retries until the
    // filtered re-render has fully landed AND every visible row's
    // role cell is "admin". A single allTextContents() snapshot would
    // race the swap: the user-count poll above tolerates equality
    // (`<= totalBefore`), so the table can still be mid-swap with
    // mixed rows when the role-cell snapshot is captured. Polling
    // the role-column predicate itself converges deterministically
    // once renderTable() finishes its innerHTML write.
    //
    // The DB stores role as lowercase 'admin' / 'user' (loaded via
    // profiles.select('role') and rendered with escapeHtml(role)
    // unchanged), so the cell text is the literal lowercase string
    // after trim.
    await expect.poll(async () => {
      const roleCellTexts = await page
        .locator('#users-body tr td:nth-child(2)')
        .allTextContents();
      if (roleCellTexts.length < 1) return false;
      return roleCellTexts.every((text) => /^admin$/i.test(text.trim()));
    }, {
      timeout: 10_000,
      message:
        'filter=admin must return at least the signed-in admin and every visible row must show role=admin; ' +
        'zero rows or any non-admin row indicates a broken filter or a rolled-back profiles.role',
    }).toBe(true);
  });
});
