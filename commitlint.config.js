export default {
  extends: ['@commitlint/config-conventional'],
  ignores: [
    // Only ignore GitHub auto-generated merge commits (PR merges)
    (message) => /^Merge pull request #\d+/.test(message),
    // Ignore merge commits when merging branches
    (message) => /^Merge branch '.+'/.test(message),
    // Ignore merge commits from remote
    (message) => /^Merge remote-tracking branch/.test(message),
    // Ignore legacy commit from Issue #373 implementation (before conventional commits)
    (message) => /^Implement comprehensive test coverage for RAG/.test(message)
  ]
};
