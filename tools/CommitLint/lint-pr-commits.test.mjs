import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { copyFileSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import test from 'node:test';
import config from '../../commitlint.config.mjs';
import { ordinaryCommits, lintCommits } from './lint-pr-commits.mjs';

const repository = fileURLToPath(new URL('../../', import.meta.url));

test('enumerates beyond 100 commits and includes merged side branches, excluding only real merges', () => {
  const cwd = mkdtempSync(join(tmpdir(), 'commitlint-history-'));
  const git = (...args) => execFileSync('git', args, { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
  const commit = message => {
    git('commit', '--allow-empty', '-m', message);
    return git('rev-parse', 'HEAD');
  };
  try {
    git('init', '-b', 'main');
    git('config', 'user.name', 'Commitlint Test');
    git('config', 'user.email', 'commitlint-test@example.invalid');
    const base = commit('chore: base');
    git('switch', '-c', 'side');
    const side = commit('fix: side branch change');
    git('switch', 'main');
    const ordinary = [];
    for (let i = 0; i < 105; i++) {
      ordinary.push(commit(i === 100 ? "Merge branch 'ordinary misleading subject'" : `fix: main change ${i}`));
    }
    git('merge', '--no-ff', 'side', '-m', 'merge descriptive subject');
    const head = git('rev-parse', 'HEAD');
    assert.equal(git('show', '-s', '--format=%P', head).split(' ').length, 2);
    const commits = ordinaryCommits(cwd, base, head);
    assert.equal(commits.length, 106);
    assert.deepEqual(new Set(commits.map(c => c.hash)), new Set([...ordinary, side]));
    assert.ok(commits.some(c => c.message.startsWith("Merge branch 'ordinary misleading subject'")));
    assert.ok(!commits.some(c => c.hash === base || c.hash === head));

    copyFileSync(join(repository, 'commitlint.config.mjs'), join(cwd, 'commitlint.config.mjs'));
    const script = join(repository, 'tools/CommitLint/lint-pr-commits.mjs');
    const passing = spawnSync(process.execPath, [script, base, ordinary[99]], { cwd, encoding: 'utf8' });
    assert.equal(passing.status, 0, passing.stderr);
    assert.match(passing.stdout, /Checked 100 ordinary commits/);
    const report = join(cwd, 'lint-results.json');
    const failing = spawnSync(process.execPath, [script, base, head, report], { cwd, encoding: 'utf8' });
    assert.equal(failing.status, 1, failing.stderr);
    assert.equal(failing.stderr, '');
    assert.match(failing.stdout, /Checked 106 ordinary commits/);
    const invalid = JSON.parse(readFileSync(report, 'utf8')).filter(result => !result.valid);
    assert.deepEqual(invalid.map(commit => commit.hash), [ordinary[100]]);

    const shallow = join(cwd, 'shallow');
    git('clone', '--depth', '1', pathToFileURL(cwd).href, shallow);
    assert.throws(() => ordinaryCommits(shallow, base, head), /complete history/);
    assert.throws(() => ordinaryCommits(cwd, '--all', head), /full commit SHAs/);
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
});

test('ordinary Merge subjects cannot use message-only merge exemptions', () => {
  assert.equal(config.defaultIgnores, false);
  for (const message of ["Merge branch 'topic'", 'Merge pull request #1', 'merge #2524 descriptive subject']) {
    assert.ok(config.ignores.every(ignore => !ignore(message)));
  }
  assert.ok(config.ignores.some(ignore => ignore('deps: Bump package\n\nSigned-off-by: dependabot[bot]')));
});

test('applies unchanged conventional rules to every enumerated ordinary commit', async () => {
  const messages = [
    'fix: valid change',
    "Merge branch 'ordinary misleading subject'",
    'merge #2524 descriptive subject',
    `fix: ${'x'.repeat(96)}`,
  ];
  const results = await lintCommits(repository, messages.map((message, index) => ({ hash: String(index).padStart(40, '0'), message })));
  assert.deepEqual(results.map(r => r.valid), [true, false, false, false]);
  assert.ok(results[3].errors.some(error => error.name === 'header-max-length'));
});
