// Run deliberately from the Preview workflow. Only add the existing preview host to the
// redirect allow list; preserve the production site URL and every other Auth setting.
const token = process.env.SUPABASE_ACCESS_TOKEN;
if (!token) throw new Error('SUPABASE_ACCESS_TOKEN is required');
const endpoint = 'https://api.supabase.com/v1/projects/yfkqwpgjahoamlgckjib/config/auth';
const headers = { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' };
async function config(method = 'GET', body) {
  const response = await fetch(endpoint, { method, headers, body: body && JSON.stringify(body), signal: AbortSignal.timeout(30_000) });
  if (!response.ok) throw new Error(`Preview Auth configuration: HTTP ${response.status}`);
  return response.json();
}
const current = await config();
const entries = (current.uri_allow_list ?? '').split(',').map((value) => value.trim()).filter(Boolean);
const preview = 'https://test.aidotnet.dev/**';
console.log(JSON.stringify({ site_url: current.site_url, preview_already_allowed: entries.includes(preview), preview_entries: entries.filter((entry) => entry.includes('test.aidotnet.dev')) }));
// Management API list replacement has no compare-and-swap; preserve concurrent edits.
if (!entries.includes(preview)) throw new Error('Add https://test.aidotnet.dev/** in https://supabase.com/dashboard/project/yfkqwpgjahoamlgckjib/auth/url-configuration, preserving existing redirects, then rerun. No Auth configuration was changed.');
let updated = await config();
for (let attempt = 0; attempt < 5 && !(updated.uri_allow_list ?? '').split(',').map((value) => value.trim()).includes(preview); attempt++) {
  await new Promise((resolve) => setTimeout(resolve, 2000));
  updated = await config();
}
if (!(updated.uri_allow_list ?? '').split(',').map((value) => value.trim()).includes(preview)) throw new Error('Preview redirect allow-list update was not persisted');
if (updated.site_url !== current.site_url) throw new Error('Production site URL unexpectedly changed');
console.log('Preview callback and recovery redirects are allowed; production site URL preserved.');
