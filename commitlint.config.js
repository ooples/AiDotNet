export default {
  extends: ['@commitlint/config-conventional'],
  rules: {
    // Allow longer body lines (default is 100, increase to 200 for detailed technical explanations)
    'body-max-line-length': [1, 'always', 200],
  },
  ignores: [
    // Only ignore GitHub auto-generated merge commits (PR merges)
    (message) => /^Merge pull request #\d+/.test(message),
    // Ignore merge commits when merging branches
    (message) => /^Merge branch '.+'/.test(message),
    // Ignore merge commits from remote
    (message) => /^Merge remote-tracking branch/.test(message),
    // Ignore legacy commit from Issue #373 implementation (before conventional commits)
    (message) => /^Implement comprehensive test coverage for RAG/.test(message),
    // Ignore legacy commit from Issue #371 implementation (before conventional commits were enforced)
    (message) => /^Implement comprehensive tests for RAG retrieval strategies/.test(message),
    // Ignore legacy commit from Issue #376 implementation (before conventional commits were enforced)
    (message) => /^Add comprehensive unit tests for specialized loss function/.test(message),
    // Ignore legacy commit from Issue #369 implementation (before conventional commits were enforced)
    (message) => /^Add comprehensive test coverage for RAG context compression/.test(message),
    // Ignore manual merge commits (e.g., "merge: resolve conflicts...")
    (message) => /^merge:/i.test(message),
    // Ignore general merge commits containing "Merge" followed by common patterns
    (message) => /^Merge (origin|upstream|master|main)/i.test(message)
  ]
};
