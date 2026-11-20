import { execFileSync } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const shaPattern = /^(?:[0-9a-f]{40}|[0-9a-f]{64})$/i;

export function ordinaryCommits(cwd, base, head) {
  if (!shaPattern.test(base) || !shaPattern.test(head)) {
    throw new Error('The base and head must be full commit SHAs.');
  }
  const git = (...args) => execFileSync('git', ['-c', `safe.directory=${cwd}`, ...args], {
    cwd, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024,
  });
  if (git('rev-parse', '--is-shallow-repository').trim() !== 'false') {
    throw new Error('Commit lint requires complete history; check out with fetch-depth: 0.');
  }
  const fields = git('log', '--no-merges', '--reverse', '--topo-order',
    '--format=%H%x00%B%x00', `${base}..${head}`).split('\0');
  const commits = [];
  for (let i = 0; i + 1 < fields.length; i += 2) {
    const hash = fields[i].trim();
    if (!shaPattern.test(hash)) throw new Error('Unexpected commit record from git log.');
    commits.push({ hash, message: fields[i + 1] });
  }
  return commits;
}

export async function lintCommits(cwd, commits) {
  const [{ default: load }, { default: lint }] = await Promise.all([
    import('@commitlint/load'), import('@commitlint/lint'),
  ]);
  const config = await load({}, { cwd, file: resolve(cwd, 'commitlint.config.mjs') });
  const options = {
    parserOpts: config.parserPreset?.parserOpts ?? {},
    plugins: config.plugins ?? {},
    ignores: config.ignores ?? [],
    defaultIgnores: config.defaultIgnores ?? true,
  };
  const results = [];
  for (const commit of commits) {
    const result = await lint(commit.message, config.rules, options);
    results.push({ ...commit, ...result });
  }
  return results;
}

async function main() {
  const [base, head, reportPath] = process.argv.slice(2);
  const commits = ordinaryCommits(process.cwd(), base, head);
  const results = await lintCommits(process.cwd(), commits);
  console.log(`Checked ${commits.length} ordinary commits; actual merge commits were excluded by parent count.`);
  const { format } = await import('@commitlint/format');
  const formatted = format({ results }, { color: false });
  if (formatted) console.log(formatted);
  if (reportPath) writeFileSync(reportPath, JSON.stringify(results, null, 2));
  if (results.some(result => !result.valid)) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(error => {
    console.error(error.message);
    process.exitCode = 1;
  });
}
