export function approvedPreviewOrigin(value) {
  const url = new URL(value);
  if (url.protocol !== 'https:' || url.username || url.password || url.port || url.pathname !== '/' || url.search || url.hash
    || !(url.hostname === 'test.aidotnet.dev' || /^aidotnetwebsite-[a-z0-9]+-franklins-projects-02a0b5a0\.vercel\.app$/.test(url.hostname))) throw new Error('An approved HTTPS AiDotNet preview origin is required');
  return url.origin;
}

// Run credential-bearing fetches outside Playwright request/error logging.
export async function authorizePreview({ baseURL, secret, expectedCommit, expectedRun, fetcher = fetch }) {
  const origin = approvedPreviewOrigin(baseURL);
  if (!secret || !/^[0-9a-f]{40}$/.test(expectedCommit ?? '')) throw new Error('Preview credential and expected deployment commit are required');
  try {
    const first = await fetcher(origin, { redirect: 'manual', signal: AbortSignal.timeout(30_000),
      headers: { 'x-vercel-protection-bypass': secret, 'x-vercel-set-bypass-cookie': 'true' } });
    if (![200,302,303,307,308].includes(first.status)) throw new Error('authorization');
    const cookies = first.headers.getSetCookie().map(raw => {
      const [pair, ...attributes] = raw.split(';').map(s => s.trim());
      const equal = pair.indexOf('=');
      if (equal < 1) throw new Error('cookie');
      const attrs = new Map(attributes.map(a => { const [k,...v] = a.split('='); return [k.toLowerCase(),v.join('=')]; }));
      const maxAge = attrs.get('max-age');
      return { name: pair.slice(0,equal), value: pair.slice(equal+1), domain: new URL(origin).hostname,
        path: attrs.get('path') || '/', expires: maxAge ? Math.floor(Date.now()/1000)+Number(maxAge) : -1,
        httpOnly: attrs.has('httponly'), secure: true,
        sameSite: ['Strict','Lax','None'].find(s => s.toLowerCase() === attrs.get('samesite')?.toLowerCase()) || 'Lax' };
    });
    await first.body?.cancel();
    const headers = { Cookie: cookies.map(c => `${c.name}=${c.value}`).join(';') };
    const marker = await fetcher(`${origin}/verification-build.json`, { headers, redirect:'manual', signal:AbortSignal.timeout(30_000) });
    if (marker.status !== 200) throw new Error('marker');
    const identity = await marker.json();
    if (identity.commit !== expectedCommit || (expectedRun && identity.run_id !== expectedRun)) throw new Error('revision');
    return { cookies, origins: [] };
  } catch {
    throw new Error('Preview authorization or deployment identity check failed. Rerun the deployment job as well as browser tests; verify the approved preview target and automation credential.');
  }
}
