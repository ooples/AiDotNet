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

  const rejectingPromise = () => Promise.reject(err);
  const handler: ProxyHandler<object> = {
    get(_target, prop) {
      if (prop === 'then') return undefined; // don't accidentally become awaitable
      // Nested access (e.g. supabase.auth.getSession) returns another proxy so the
      // chain keeps working syntactically; terminal calls return rejecting promises.
      return new Proxy(rejectingPromise, handler);
    },
    apply() {
      return rejectingPromise();
    },
  };
  return new Proxy(rejectingPromise, handler) as unknown as SupabaseClient;
}

export const supabase: SupabaseClient = supabaseConfigured
  ? createClient(supabaseUrl, supabaseAnonKey)
  : createDisabledClient();
