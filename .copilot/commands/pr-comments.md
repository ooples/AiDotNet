# PR Comment Resolution Workflow

**Command**: `/pr-comments <PR_NUMBER|all>`

**Version**: 2.2.0

**Purpose**: Systematically fix unresolved PR review comments using GraphQL API, implement production-ready fixes, commit changes, and resolve threads. Uses git worktrees for parallel execution.

**Usage**:
- `/pr-comments 326` - Work on PR #326
- `/pr-comments all` - Work on all open PRs with unresolved comments

---

## Instructions

You are tasked with systematically resolving unresolved code review comments on GitHub Pull Requests. Follow these steps precisely:

**IMPORTANT**: This command uses git worktrees to allow multiple AI instances to work in parallel without conflicts.

### Step 1: Extract PR Number(s) from Command Arguments

The PR number or "all" keyword is provided as the first argument:

**Option A: Single PR**
- If user runs `/pr-comments 326`, use PR number **326**
- Store as `PR_NUMBERS=(326)`

**Option B: All Open PRs**
- If user runs `/pr-comments all`, fetch all open PR numbers:
  ```bash
  PR_NUMBERS=$(gh pr list --state open --json number -q '.[].number')
  ```
- Store the list and process each PR sequentially

**Option C: No Argument**
- If user runs `/pr-comments` without arguments, ask which PR(s) to work on

**IMPORTANT**: For each PR in PR_NUMBERS, execute all subsequent steps.

### Step 2: For Each PR - Create Unique Git Worktree

**IMPORTANT**: If processing multiple PRs, repeat Steps 2-10 for each PR in the list.

Create a unique worktree for this PR to avoid conflicts with other AI instances or user's work:

```bash
# Store original directory for cleanup later
ORIGINAL_REPO_DIR=$(pwd)

# Get PR branch name
BRANCH_NAME=$(gh pr view $PR_NUMBER --json headRefName -q .headRefName)

# Create unique worktree directory using timestamp and PR number
WORKTREE_ID="pr-${PR_NUMBER}-$(date +%s)"
WORKTREE_PATH="../worktrees/${WORKTREE_ID}"

# Fetch the latest branch
git fetch origin $BRANCH_NAME

# Create worktree from the PR branch
mkdir -p ../worktrees
git worktree add "$WORKTREE_PATH" "$BRANCH_NAME" || git worktree add "$WORKTREE_PATH" -b "$BRANCH_NAME" "origin/$BRANCH_NAME"

# Change to worktree directory
cd "$WORKTREE_PATH"

echo "Created worktree at: $WORKTREE_PATH"
echo "Processing PR #$PR_NUMBER on branch $BRANCH_NAME"
```

**CRITICAL**: All subsequent file operations must occur within `$WORKTREE_PATH`. Store both `$WORKTREE_PATH` and `$ORIGINAL_REPO_DIR` for cleanup later.

### Step 3: Fetch Unresolved Review Threads via GraphQL

Use the GitHub GraphQL API to get all unresolved review threads for the specified PR number.

**CRITICAL**: Extract repo owner and name dynamically, then use GraphQL:

```bash
# Extract repository owner and name from git remote
REPO_URL=$(git remote get-url origin)
REPO_OWNER=$(echo $REPO_URL | sed -E 's|.*github\.com[:/]([^/]+)/.*|\1|')
REPO_NAME=$(echo $REPO_URL | sed -E 's|.*github\.com[:/][^/]+/([^/\.]+)(\.git)?$|\1|')

echo "Repository: $REPO_OWNER/$REPO_NAME"
echo "PR Number: $PR_NUMBER"

# Create GraphQL query to fetch ALL review threads with pagination
# CRITICAL: Use 'last: 100' to get MOST RECENT threads first (where unresolved comments usually are)
cat > /tmp/pr-threads-query.graphql << 'EOF'
query($owner: String!, $repo: String!, $prNumber: Int!) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $prNumber) {
      reviewThreads(last: 100) {
        nodes {
          id
          isResolved
          isOutdated
          path
          line
          originalLine
          comments(first: 10) {
            nodes {
              id
              author {
                login
              }
              body
              createdAt
              path
              line
              originalLine
              diffHunk
            }
          }
        }
      }
    }
  }
}
EOF

# Execute GraphQL query with dynamic owner/repo
# This fetches the LAST (most recent) 100 threads where unresolved comments typically are
gh api graphql -F owner="$REPO_OWNER" -F repo="$REPO_NAME" -F prNumber=$PR_NUMBER -f query="$(cat /tmp/pr-threads-query.graphql)" > /tmp/pr-threads-recent.json

# For PRs with >100 threads, also fetch first 100 to ensure we don't miss anything
cat > /tmp/pr-threads-query-old.graphql << 'EOF'
query($owner: String!, $repo: String!, $prNumber: Int!) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $prNumber) {
      reviewThreads(first: 100) {
        nodes {
          id
          isResolved
        }
      }
    }
  }
}
EOF

gh api graphql -F owner="$REPO_OWNER" -F repo="$REPO_NAME" -F prNumber=$PR_NUMBER -f query="$(cat /tmp/pr-threads-query-old.graphql)" > /tmp/pr-threads-old.json

# Merge both result sets and deduplicate by thread ID
# Priority: recent threads (last 100) are more likely to have unresolved comments
echo "Fetched review threads - prioritizing most recent comments"
```

**IMPORTANT**: The owner and repo are extracted automatically from `git remote get-url origin`.

### Step 4: Parse and Display Unresolved Comments

Parse the GraphQL response and display only UNRESOLVED comments (where `isResolved: false`).

For each unresolved thread, show:
- Thread ID
- File path and line number
- Comment author
- Comment body
- Code context (diffHunk)

Number each unresolved comment (1, 2, 3, etc.) and create a TODO list.

**Output format**:
```
Found X unresolved review comments on PR #$PR_NUMBER:

1. ThreadID: PRRT_xxx - file.cs:123
   Author: @reviewer
   Comment: [description]

2. ThreadID: PRRT_yyy - file.cs:456
   Author: @reviewer
   Comment: [description]
...
```

### Step 5: Read Project Instructions

Before making any fixes, read and understand the project requirements:

```bash
# Read project instructions
cat .github/PROJECT_INSTRUCTIONS.md
```

Follow all requirements specified in this file for:
- Code style and conventions
- Testing requirements
- Documentation standards
- Build and validation steps

### Step 6: Systematically Fix Each Comment

**For each unresolved comment:**

1. **Read the relevant code section** using the Read tool
2. **Analyze the issue** described in the comment
3. **Check project instructions** for relevant standards/requirements
4. **Implement production-ready fix** using the Edit tool
   - Follow all project instructions from .github/PROJECT_INSTRUCTIONS.md
   - Follow all coding standards
   - Ensure type safety
   - Add necessary validation
   - Handle edge cases
5. **Build and verify** the fix compiles successfully:
   ```bash
   dotnet build
   ```
   If build fails, fix the errors before committing
6. **Verify the fix** by reading the code again
7. **Commit the fix immediately** with a descriptive message:
   ```bash
   git add <file>
   git commit -m "fix: <concise description of what was fixed>

   Resolves review comment on line X of <file>
   - <brief explanation of the fix>

   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>"
   ```
8. **Mark TODO as completed**

### Step 6: Push All Commits

After fixing all comments, push all commits to the remote:

```bash
git push origin <current-branch>
```

### Step 7: Verify Build Passes

Ensure the complete solution builds successfully:

```bash
dotnet build --configuration Release
```

If there are test failures or build errors, fix them before proceeding.

### Step 8: Resolve Comment Threads via GraphQL

For each fixed comment, resolve the review thread using GraphQL mutation:

```bash
# For each thread ID that was fixed, resolve it
THREAD_ID="<THREAD_ID_FROM_STEP_4>"

gh api graphql -f query='mutation { resolveReviewThread(input: {threadId: "'"$THREAD_ID"'"}) { thread { id isResolved } } }'
```

**Alternative approach** (if mutation syntax has issues):

```bash
# Create mutation in temp file
cat > /tmp/resolve-mutation.graphql << EOF
mutation {
  resolveReviewThread(input: {threadId: "$THREAD_ID"}) {
    thread {
      id
      isResolved
    }
  }
}
EOF

gh api graphql -f query="$(cat /tmp/resolve-mutation.graphql)"
```

**IMPORTANT**: Only resolve threads after the fix is committed and pushed to remote.

### Step 9: Verify All Comments Resolved

Re-run the GraphQL query from Step 3 to verify all comments are now resolved for the current PR.

Report to the user for each PR:
- PR number
- Number of comments fixed
- Number of commits created
- Number of threads resolved
- Any remaining unresolved comments (if any)

### Step 10: Cleanup Worktree

**CRITICAL**: After completing work on the PR, clean up the worktree to free disk space:

```bash
# Return to original repository directory
cd "$ORIGINAL_REPO_DIR"

# Remove the worktree
git worktree remove "$WORKTREE_PATH" --force

echo "Cleaned up worktree: $WORKTREE_PATH"
```

**If processing multiple PRs ("all" mode)**:
- Repeat Steps 2-10 for each PR in PR_NUMBERS
- Clean up each worktree after completing that PR
- Provide a summary report at the end showing totals across all PRs

---

## Important Notes

### Error Handling

- If a fix fails to compile or has errors, do NOT commit
- Mark the TODO as blocked and move to the next comment
- Return to blocked items after completing others

### Commit Best Practices

- One commit per logical fix (can be one commit per comment, or group related comments)
- Use conventional commit format: `fix:`, `refactor:`, etc.
- Include the file name and line number in commit body
- Keep commits atomic and focused

### GraphQL Thread Resolution

- Thread IDs are in format `PRRT_*` (Pull Request Review Thread)
- Only resolve threads where you've actually fixed the issue
- If uncertain about a fix, ask the user before resolving

### Code Quality Standards

All fixes must be:
- Type-safe (no use of `default!` operator)
- Well-validated (check for null, negative values, etc.)
- Production-ready (handle edge cases)
- Properly documented (if adding new methods)
- Following project conventions

---

## Example Usage

### Example 1: Single PR

```
User: /pr-comments 256
Assistant:
I'll fetch and fix all unresolved review comments on PR #256.

[Checks out PR #256 branch]
[Runs GraphQL query]

Found 8 unresolved review comments on PR #256:

1. DVoRAAdapter.cs:169-181 - ParameterCount omits LoRA parameters
2. DVoRAAdapter.cs:293-330 - Missing parameter validation
3. DVoRAAdapter.cs:704-715 - Magnitude gradient calculation incorrect
4. DyLoRAAdapter.cs:557-603 - Gradients not applied to layer
5. LoRAXSAdapter.cs:246-305 - ParameterCount crashes base constructor
6. MoRAAdapter.cs:209-223 - Static Random.NextDouble() calls
7. MoRAAdapter.cs:534-552 - ParameterCount uninitialized buffer detection
8. MoRAAdapter.cs:291-333 - Wrong parameters packed

Creating TODO list and fixing each comment...

[Fixes comments 1-8, commits each fix]

All 8 comments fixed. Building to verify...
[Build succeeds]

Pushing commits...
[Pushes to remote]

Resolving review threads via GraphQL...
[Resolves each thread]

✅ PR #256 Summary:
- 8 comments fixed
- 8 commits created
- 8 threads resolved
- 0 remaining unresolved comments
```

### Example 2: All Open PRs

```
User: /pr-comments all
Assistant:
I'll fetch and fix all unresolved review comments on all open PRs.

[Fetches open PR list]

Found 3 open PRs: #256, #326, #328

Processing PR #256...
[Follows steps 2-9 for PR #256]
✅ PR #256: 8 comments fixed, 8 threads resolved

Processing PR #326...
[Follows steps 2-9 for PR #326]
✅ PR #326: 10 comments fixed, 10 threads resolved

Processing PR #328...
[Follows steps 2-9 for PR #328]
✅ PR #328: 0 unresolved comments (skipping)

✅ Overall Summary:
- Total PRs processed: 3
- Total PRs with fixes: 2
- Total comments fixed: 18
- Total commits created: 18
- Total threads resolved: 18
```

---

## Command Behavior

- **DO NOT** ask user for clarification on fixes unless absolutely necessary
- **DO** implement best-judgment fixes based on comment descriptions
- **DO** commit frequently (after each fix or small group of related fixes)
- **DO** resolve threads only after fixes are pushed
- **DO NOT** create any markdown report/analysis files
- **DO** use TODO list to track progress

---

## GraphQL API Reference

### Get Unresolved Threads (Most Recent First)
```graphql
query($owner: String!, $repo: String!, $prNumber: Int!) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $prNumber) {
      reviewThreads(last: 100) {
        nodes {
          id
          isResolved
          path
          line
          comments(first: 10) {
            nodes {
              body
              author { login }
              createdAt
            }
          }
        }
      }
    }
  }
}
```

**Note**: Using `last: 100` fetches the most recent 100 threads, where unresolved comments typically are. For PRs with >100 threads, also fetch `first: 100` and merge results.

### Resolve Thread
```graphql
mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) {
    thread {
      id
      isResolved
    }
  }
}
```

---

**Version**: 2.1.0
**Last Updated**: 2025-11-07
**Author**: Claude Code Configuration

**Changelog**:
- v2.2.0: **CRITICAL FIX** - Fixed pagination to fetch RECENT unresolved comments
  - Changed from `first: 100` to `last: 100` to get most recent threads
  - Most recent unresolved comments are typically at the end, not beginning
  - Fetches both recent (last 100) and old (first 100) threads for comprehensive coverage
  - Fixes issue where only oldest comments were processed, missing fresh reviews
- v2.1.0: **CRITICAL FIX** - Made command repo-agnostic
  - Dynamically extract REPO_OWNER and REPO_NAME from git remote
  - Fixed GraphQL query syntax with temp file approach
  - Works with any GitHub repository, not hardcoded to specific repo
- v2.0.0: **BREAKING** - Added git worktree support for parallel execution
  - Each invocation creates a unique worktree (timestamp-based)
  - Allows multiple AI instances to work on different PRs simultaneously
  - Prevents conflicts with user's working directory
  - Auto-cleanup after PR completion
- v1.2.0: Added support for "all" open PRs mode and PR number argument
- v1.1.0: Added build verification and project instructions compliance
- v1.0.0: Initial release

## Benefits of Worktree Architecture

**For Users:**
- Continue working in your main repo while AI fixes PR comments
- No branch switching disrupts your work
- Multiple AI instances can work in parallel

**For AI Instances:**
- Each instance works in isolation
- No conflicts between concurrent executions
- Clean environment for each PR

**Performance:**
- Worktrees share .git directory (fast creation)
- Minimal disk overhead
- Auto-cleanup prevents accumulation
