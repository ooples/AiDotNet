export default {
  extends: ['@commitlint/config-conventional'],
  rules: {
    // Allow longer body lines (default is 100, increase to 200 for detailed technical explanations)
    'body-max-line-length': [1, 'always', 200],
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
