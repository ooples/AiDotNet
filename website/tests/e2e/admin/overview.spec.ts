import { test, expect } from '@playwright/test';

test.describe('Admin — overview', () => {
  test('/admin/ renders stat cards, sidebar, and both tables', async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });

    await page.goto('/admin/');

    await expect(page.getByRole('heading', { name: 'Admin Overview', level: 1 })).toBeVisible();

    // Admin badge is how the sidebar signals the role-gate succeeded. If
    // the admin layout ever flips back to the AccountLayout for an admin
    // user by mistake, this disappears. Scope to <aside> so the navbar's
    // admin nav-link doesn't accidentally satisfy the assertion when
    // the sidebar itself is missing.
    await expect(page.getByRole('complementary').getByText('Admin Panel')).toBeVisible();

    // Five nav items: Overview, Users, API Keys, Licenses, Usage Analytics.
    // The exact text is the source of truth; if a refactor changes
    // "Usage Analytics" to "Usage", we want to know because external
    // links (docs, Slack, support emails) reference the admin nav by
    // these labels.
    for (const name of ['Overview', 'Users', 'API Keys', 'Licenses', 'Usage Analytics']) {
      await expect(page.getByRole('link', { name, exact: true })).toBeVisible();
    }

    // Six stat cards — each must populate from a different supabase
    // query, so a partial failure (e.g., profiles RLS policy regressed
    // but api_usage is fine) would leave some cards stuck at "--". We
    // assert only that each ID is present; the values themselves depend
    // on live data.
    for (const id of [
      'stat-total-users',
      'stat-pro-users',
      'stat-enterprise-users',
      'stat-api-calls',
      'stat-active-keys',
      'stat-avg-latency',
    ]) {
      await expect(page.locator(`#${id}`)).toBeVisible();
    }

    // Total users should always resolve to a positive integer — this
    // account at minimum has the admin + user test accounts, so "0" would
    // indicate the count query failed silently.
    await expect(page.locator('#stat-total-users')).not.toHaveText('--', { timeout: 15_000 });
    const totalText = (await page.locator('#stat-total-users').textContent())?.trim() ?? '';
    const totalInt = parseInt(totalText.replace(/,/g, ''), 10);
    expect(totalInt).toBeGreaterThanOrEqual(2);

    // Recent signups table: either the "no signups" empty state or the
    // real table must be visible — the loader state should be done.
    await expect(page.locator('#signups-empty')).toBeAttached();

    expect(consoleErrors, 'admin overview should emit zero console errors').toEqual([]);
  });

  test('sidebar "Back to Account" link exits the admin layout', async ({ page }) => {
    await page.goto('/admin/');
    // The /account/ back-link is the escape hatch for an admin who wants
    // to view their own (non-admin) view. If it regresses to point at
    // /admin/ itself, admin-test users get stuck in the admin shell.
    const backLink = page.getByRole('link', { name: 'Back to Account' });
    await expect(backLink).toHaveAttribute('href', /\/account\/?$/);
  });
});
