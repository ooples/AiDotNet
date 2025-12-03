export default {
  extends: ['@commitlint/config-conventional'],
  ignores: [
    // Only ignore GitHub auto-generated merge commits (PR merges)
    (message) => /^Merge pull request #\d+/.test(message),
    // Ignore merge commits when merging branches
    (message) => /^Merge branch '.+'/.test(message),
    // Ignore merge commits from remote
    (message) => /^Merge remote-tracking branch/.test(message)
  ]
};
