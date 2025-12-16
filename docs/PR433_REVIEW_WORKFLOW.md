## PR#433 Review Workflow (Unresolved Comments)

This repo’s PR#433 work must be completed using the following loop, **in order**, until there are no unresolved comments left:

1) Use the **GitHub GraphQL API** (via `gh api graphql`) to fetch **unresolved review threads** for PR#433.
2) Take the next unresolved thread (oldest first unless specified otherwise).
3) Implement the required code/docs change in the repo.
4) **Before moving to the next thread**:
   - If it’s a normal review thread: resolve it via the GraphQL `resolveReviewThread` mutation.
   - If it’s a code-scanning / bot thread that cannot be manually resolved: add a reply comment to the thread describing what was fixed, so we can track progress.

Notes:
- Do not batch multiple threads in one pass; always follow the “fix → resolve/reply → next” sequence.
- Keep the public API surface minimal per the facade philosophy; prefer `internal` and nested session types.

