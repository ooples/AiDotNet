/** Browser URL parsing normalizes backslashes and strips control characters.
 * Validate the parsed destination as well as requiring a local absolute path. */
export function safeRedirect(raw: string | null, fallback: string, origin = window.location.origin): string {
  if (!raw || !raw.startsWith('/') || raw.startsWith('//') || /[\\\u0000-\u0020\u007f]/.test(raw)) return fallback;
  try {
    const destination = new URL(raw, origin);
    if (destination.origin !== origin || destination.pathname.startsWith('//')) return fallback;
    return `${destination.pathname}${destination.search}${destination.hash}`;
  } catch {
    return fallback;
  }
}
