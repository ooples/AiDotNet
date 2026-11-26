export default {
  extends: ['@commitlint/config-conventional'],
  // CI identifies real merge commits from their parents. Message-only defaults would also
  // exempt an ordinary commit whose subject happens to start with "Merge".
  defaultIgnores: false,
  rules: {
    // Allow longer body lines (default is 100, increase to 200 for detailed technical explanations)
    'body-max-line-length': [1, 'always', 200],
    // Match the body limit for footers. The conventional-commits parser classifies any
    // "word: value" line (e.g. a "L=2, per-epoch: ..." benchmark line, or a squashed PR's
    // "* fix(x): ..." bullet) as a footer, so detailed messages tripped the default 100.
    // Kept a warning at 200, consistent with body-max-line-length above.
    'footer-max-line-length': [1, 'always', 200],
    // Add 'deps' as valid type for dependabot commits
    'type-enum': [2, 'always', [
      'feat', 'fix', 'docs', 'refactor', 'perf', 'test', 'chore', 'ci', 'build', 'style', 'revert', 'merge',
      'deps'  // For dependency updates (dependabot)
    ]],
  },
  ignores: [
    // ===== GitHub Actions Bot Commits =====
    // Ignore commits made by GitHub Actions (auto-fixes, dependabot, etc.)
    (message) => /Co-Authored-By: github-actions\[bot\]/.test(message),
    // Dependabot signs off rather than co-authoring, so the rule above never matched it and the
    // ignore never covered the case its own comment names. Its subjects are "deps: Bump X from A to
    // B" — 'deps' is already an accepted type above, but the capitalised "Bump" trips subject-case,
    // and the wording is dependabot's to choose, not this repo's. Any PR whose range still contains
    // a dependabot commit therefore failed the job outright.
    (message) => /Signed-off-by: dependabot\[bot\]/.test(message),

    // NOTE: All legacy ignores have been REMOVED as of 2025-12-14.
    // The pr-title-lint.yml and commitlint-autofix.yml workflows now handle
    // auto-fixing non-compliant PR titles and commit messages.
    // All new commits MUST follow conventional commits format:
    //   type(scope)?: description
    // Valid types: feat, fix, docs, refactor, perf, test, chore, ci, build, style, revert, merge, deps
  ]
};
