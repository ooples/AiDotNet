import { createClient, type SupabaseClient } from '@supabase/supabase-js';

const supabaseUrl = (import.meta.env.PUBLIC_SUPABASE_URL || '').trim();
const supabaseAnonKey = (import.meta.env.PUBLIC_SUPABASE_ANON_KEY || '').trim();

/**
 * True when PUBLIC_SUPABASE_URL and PUBLIC_SUPABASE_ANON_KEY were supplied at
 * build time. When false, every auth/DB call becomes a no-op that surfaces a
 * user-visible error banner instead of crashing the entire script module —
 * that way sign-in pages still render and users see a clear error instead of
 * buttons that silently do nothing.
 */
export const supabaseConfigured = !!supabaseUrl && !!supabaseAnonKey;

/**
 * User-facing message shown when Supabase isn't configured. Login / signup /
 * dashboard pages can surface this directly in the UI.
 */
export const supabaseMissingConfigMessage =
  'Authentication is temporarily unavailable. Please try again in a few minutes. ' +
  'If the problem persists, contact support@aidotnet.dev.';

function createDisabledClient(): SupabaseClient {
  const err = new Error(
    'Supabase client is not configured. Set PUBLIC_SUPABASE_URL and ' +
      'PUBLIC_SUPABASE_ANON_KEY at build time.',
  );
  // Log once at module load so deployment issues are visible in browser devtools
  // AND in server logs (Astro SSR paths).
  if (typeof console !== 'undefined' && typeof console.error === 'function') {
    console.error('[AiDotNet] Supabase env vars missing — auth disabled.', err.message);
  }

  // Return auth-shaped no-op results so unguarded destructuring patterns stay
  // safe. Multiple pages (Navbar, pricing, signup, login, licenses,
  // AdminLayout, AccountLayout) do
  //
  //     const { data: { session } } = await supabase.auth.getSession()
  //
  // which crashes with "Cannot read property 'session' of null" if we return
  // { data: null, ... }. Real Supabase returns:
  //   - getSession()          → { data: { session: Session | null }, error }
  //   - onAuthStateChange(cb) → { data: { subscription }, error: null }  (SYNC)
  // Match those shapes so pages degrade gracefully.
  const disabledSubscription = { unsubscribe() {} };
  const resolveDisabledCall = (method?: PropertyKey): unknown => {
    switch (method) {
      case 'getSession':
      case 'getUser':
        return Promise.resolve({ data: { session: null, user: null }, error: err });
      case 'onAuthStateChange':
        // onAuthStateChange is synchronous in the real client, not thenable.
        return { data: { subscription: disabledSubscription }, error: null };
      default:
        return Promise.resolve({ data: null, error: err });
    }
  };

  // Methods that END a call chain — everything else is treated as an
  // intermediate fluent builder (e.g. supabase.from('t').select('*').eq(...))
  // so `.select(...)` called on the result of `.from(...)` still returns a
  // proxy instead of crashing with "Cannot read property 'select' of Promise".
  const terminalMethods = new Set<string>([
    'getSession',
    'getUser',
    'onAuthStateChange',
    'signInWithOAuth',
    'signInWithPassword',
    'signUp',
    'signOut',
    'resetPasswordForEmail',
    'updateUser',
    'exchangeCodeForSession',
    'refreshSession',
    // Query terminals — awaiting a Supabase PostgrestFilterBuilder runs the
    // query, so anything the caller awaits must also resolve to a shape.
    'then',
    'single',
    'maybeSingle',
    'csv',
    'explain',
  ]);

  const createProxy = (path: PropertyKey[] = []): any =>
    new Proxy(() => undefined, {
      get(_target, prop) {
        if (prop === 'then') return undefined; // don't accidentally become awaitable
        // Walk the path (e.g. supabase.auth.getSession) so apply() knows which
        // method was called and can return the right result shape.
        return createProxy([...path, prop]);
      },
      apply() {
        const method = String(path[path.length - 1] ?? '');
        // Terminal methods resolve to their shape; everything else returns
        // another proxy so fluent chains keep working without TypeError.
        return terminalMethods.has(method)
          ? resolveDisabledCall(method)
          : createProxy(path);
      },
    });

  return createProxy() as SupabaseClient;
}

export const supabase: SupabaseClient = supabaseConfigured
  ? createClient(supabaseUrl, supabaseAnonKey)
  : createDisabledClient();
