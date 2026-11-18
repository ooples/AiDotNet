import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';

/** An already synchronized key needs no secret-write permission after a manual rotation. */
export async function syncFunctionMailKey({ token, key, fetcher = fetch }) {
  if (!token || !key) throw new Error('Supabase management access and the existing mail key are required');
  const url = 'https://api.supabase.com/v1/projects/yfkqwpgjahoamlgckjib/secrets';
  const headers = { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' };
  const digest = createHash('sha256').update(key).digest('hex');
  async function list() {
    const response = await fetcher(url, { headers, signal: AbortSignal.timeout(30_000) });
    if (!response.ok) throw new Error(`Cannot verify the function mail key: HTTP ${response.status}`);
    return response.json();
  }
  const matches = rows => rows.some(row => row.name === 'RESEND_API_KEY' && (row.value ?? row.digest) === digest);
  if (matches(await list())) return { changed: false };
  const response = await fetcher(url, { method: 'POST', headers, body: JSON.stringify([{ name: 'RESEND_API_KEY', value: key }]), signal: AbortSignal.timeout(30_000) });
  if (!response.ok) throw new Error(`Function mail key differs and could not be synchronized: HTTP ${response.status}; edge_functions_secrets_write is required`);
  for (let attempt = 0; attempt < 5; attempt++) {
    if (matches(await list())) return { changed: true };
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  throw new Error('The function mail key could not be verified after synchronization');
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    const result = await syncFunctionMailKey({ token: process.env.SUPABASE_ACCESS_TOKEN, key: process.env.RESEND_API_KEY });
    console.log(result.changed ? 'Function mail key synchronized and verified.' : 'The existing function mail key already matches.');
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
