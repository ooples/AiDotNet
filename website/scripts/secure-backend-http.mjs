/** Native fetch avoids Playwright recording privileged request headers in failed API steps. */
export async function serverRequest(value, options = {}, fetcher = fetch) {
  const url = new URL(value);
  if (!['https://yfkqwpgjahoamlgckjib.supabase.co','https://api.stripe.com'].includes(url.origin) || url.username || url.password) throw new Error('Unexpected privileged verification target');
  let response, bytes;
  try {
    const headers = { ...options.headers };
    const data = options.data;
    if (data !== undefined && typeof data !== 'string') headers['content-type'] ??= 'application/json';
    response = await fetcher(url.href, { method: options.method ?? 'GET', headers,
      body: data === undefined ? undefined : typeof data === 'string' ? data : JSON.stringify(data),
      redirect: 'manual', signal: AbortSignal.timeout(options.timeout ?? 30_000) });
    if (response.status >= 300 && response.status < 400) {
      await response.body?.cancel();
      throw new Error('Unexpected API redirect');
    }
    bytes = await response.arrayBuffer();
  } catch { throw new Error('Privileged verification request failed; request headers and credentials were omitted.'); }
  return {
    status: () => response.status, ok: () => response.ok, url: () => response.url,
    headers: () => Object.fromEntries(response.headers.entries()),
    text: async () => new TextDecoder().decode(bytes),
    json: async () => JSON.parse(new TextDecoder().decode(bytes)),
    dispose: async () => {},
  };
}
