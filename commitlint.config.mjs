export default {
  extends: ['@commitlint/config-conventional'],
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
      'feat', 'fix', 'docs', 'refactor', 'perf', 'test', 'chore', 'ci', 'style', 'revert',
      'deps'  // For dependency updates (dependabot)
    ]],
  },
  ignores: [
    // ===== Merge Commits ONLY =====
    // These are the ONLY legitimate ignores - merge commits cannot follow conventional format

    // Ignore GitHub auto-generated merge commits (PR merges)
    (message) => /^Merge pull request #\d+/.test(message),
    // Ignore merge commits when merging branches
    (message) => /^Merge branch '.+'/.test(message),
    // Ignore merge commits from remote
    (message) => /^Merge remote-tracking branch/.test(message),
    // Ignore general merge commits containing "Merge" followed by common patterns
    (message) => /^Merge (origin|upstream|master|main)/i.test(message),

    // ===== GitHub Actions Bot Commits =====
    // Ignore commits made by GitHub Actions (auto-fixes, dependabot, etc.)
    (message) => /Co-Authored-By: github-actions\[bot\]/.test(message),

    // NOTE: All legacy ignores have been REMOVED as of 2025-12-14.
    // The pr-title-lint.yml and commitlint-autofix.yml workflows now handle
    // auto-fixing non-compliant PR titles and commit messages.
    // All new commits MUST follow conventional commits format:
    //   type(scope)?: description
    // Valid types: feat, fix, docs, refactor, perf, test, chore, ci, style
  ]
};
