export default {
  extends: ['@commitlint/config-conventional'],
  rules: {
    // Allow longer body lines (default is 100, increase to 200 for detailed technical explanations)
    'body-max-line-length': [1, 'always', 200],
  },
  ignores: [
    // ===== Merge Commits =====
    // Ignore GitHub auto-generated merge commits (PR merges)
    (message) => /^Merge pull request #\d+/.test(message),
    // Ignore merge commits when merging branches
    (message) => /^Merge branch '.+'/.test(message),
    // Ignore merge commits from remote
    (message) => /^Merge remote-tracking branch/.test(message),
    // Ignore manual merge commits (e.g., "merge: resolve conflicts...")
    (message) => /^merge:/i.test(message),
    // Ignore general merge commits containing "Merge" followed by common patterns
    (message) => /^Merge (origin|upstream|master|main)/i.test(message),

    // ===== Legacy Commits (before conventional commits enforced) =====
    // These are squashed PR commits that used generic PR titles

    // Ignore "Work Session Planning" and "Work on Issue" PRs
    (message) => /^Work (Session|on Issue)/i.test(message),

    // Ignore "Fix issue X in AiDotNet (#Y)" PRs
    (message) => /^Fix issue \d+ in AiDotNet/i.test(message),

    // Ignore legacy "Implement X" commits (not following feat: format)
    (message) => /^Implement\s+\w+/i.test(message),

    // Ignore legacy "Add X" commits (not following feat: format)
    (message) => /^Add\s+\w+/i.test(message),

    // Ignore legacy "Update X" commits (not following feat/fix: format)
    (message) => /^Update\s+\w+/i.test(message),

    // Ignore legacy "Create X" commits
    (message) => /^Create\s+\w+/i.test(message),

    // Ignore legacy "Remove X" commits
    (message) => /^Remove\s+\w+/i.test(message),

    // Ignore commits that are just PR titles (Title (#123))
    (message) => /^[A-Z][^:]+\s+\(#\d+\)$/.test(message.split('\n')[0]),
  ]
};
