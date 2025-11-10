# Conventional Commits Guide for AiDotNet

## Quick Reference

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Type:** `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `chore`, `style`, `ci`
**Scope:** Optional context (e.g., `neural-net`, `matrix`, `tensor`)
**Description:** Short summary in imperative mood (e.g., "add feature" not "added feature")

## Why Conventional Commits?

Conventional Commits provide:

1. **Automated Versioning** - Version numbers are calculated from commit messages
2. **Generated Changelogs** - Release notes are created automatically
3. **Better Communication** - Clear commit history helps teams understand changes
4. **Consistent Format** - Standardized commits across the project

## Commit Types

### `feat:` - New Features

Use when adding new functionality or capabilities.

**Examples:**
```
feat: Add convolutional neural network layer
feat(tensor): Add support for GPU acceleration
feat: Implement dropout regularization
```

**Version Impact:** MINOR bump (0.0.5 → 0.1.0)

### `fix:` - Bug Fixes

Use when fixing a bug or correcting unexpected behavior.

**Examples:**
```
fix: Correct gradient calculation in backpropagation
fix(matrix): Fix dimension mismatch in multiplication
fix: Resolve memory leak in batch processing
```

**Version Impact:** MINOR bump (0.0.5 → 0.1.0)

### `refactor:` - Code Refactoring

Use when restructuring code without changing external behavior.

**Examples:**
```
refactor: Simplify layer initialization logic
refactor(activation): Extract sigmoid function
refactor: Use LINQ for data transformations
```

**Version Impact:** MINOR bump (0.0.5 → 0.1.0)

### `perf:` - Performance Improvements

Use when optimizing code for better performance.

**Examples:**
```
perf: Optimize matrix multiplication with SIMD
perf(batch): Reduce memory allocations
perf: Cache computed gradients
```

**Version Impact:** MINOR bump (0.0.5 → 0.1.0)

### `docs:` - Documentation

Use when updating documentation only (no code changes).

**Examples:**
```
docs: Add API reference for neural network module
docs(readme): Update installation instructions
docs: Fix typos in code comments
```

**Version Impact:** MINOR bump (0.0.5 → 0.1.0)

### `test:` - Tests

Use when adding or updating tests.

**Examples:**
```
test: Add unit tests for activation functions
test(integration): Test end-to-end training pipeline
test: Increase coverage for tensor operations
```

**Version Impact:** None (no release)

### `chore:` - Maintenance

Use for build process, dependency updates, or other maintenance.

**Examples:**
```
chore: Update NuGet dependencies
chore(deps): Bump BenchmarkDotNet from 0.15.5 to 0.15.6
chore: Update .gitignore
```

**Version Impact:** None (no release)

### `style:` - Code Style

Use for formatting, whitespace, or code style changes.

**Examples:**
```
style: Format code with consistent indentation
style: Add missing braces to if statements
style: Reorder using statements
```

**Version Impact:** None (no release)

### `ci:` - CI/CD

Use for changes to CI/CD configuration or scripts.

**Examples:**
```
ci: Add automated release workflow
ci: Update GitHub Actions to use .NET 8
ci: Add code coverage reporting
```

**Version Impact:** None (no release)

## Breaking Changes

### BREAKING CHANGE Footer

Use when making incompatible API changes.

**Format:**
```
<type>: <description>

BREAKING CHANGE: <explanation of breaking change>
```

**Example:**
```
feat: Redesign tensor API

BREAKING CHANGE: Tensor constructor now requires explicit shape parameter.
Previous code using Tensor(data) must be updated to Tensor(data, shape).
```

**Version Impact:** MAJOR bump (0.0.5 → 1.0.0)

### Alternative: `!` Notation

You can also use `!` after type/scope:

```
feat!: Redesign tensor API
fix(api)!: Change method signature
```

## Scopes

Scopes provide additional context about what part of the codebase changed.

### Common Scopes for AiDotNet

| Scope | Description |
|-------|-------------|
| `tensor` | Tensor operations and data structures |
| `matrix` | Matrix operations |
| `neural-net` | Neural network layers and models |
| `activation` | Activation functions |
| `loss` | Loss functions |
| `optimizer` | Optimization algorithms |
| `data` | Data loading and preprocessing |
| `test` | Test infrastructure |
| `deps` | Dependencies |
| `build` | Build system |

**Examples:**
```
feat(neural-net): Add LSTM layer implementation
fix(activation): Correct ReLU derivative
refactor(optimizer): Simplify Adam optimizer logic
```

## Commit Message Body

Provide additional context in the commit body:

```
feat(neural-net): Add batch normalization layer

Batch normalization improves training stability and allows higher
learning rates. This implementation follows the paper by Ioffe & Szegedy.

Includes:
- BatchNormLayer class with training/inference modes
- Automatic running mean/variance tracking
- Configurable momentum and epsilon parameters
```

## Commit Message Footer

Use footers for metadata:

```
feat(optimizer): Add AdaGrad optimizer

Adaptive Gradient Algorithm (AdaGrad) adjusts learning rates based on
historical gradient information.

Closes #123
Reviewed-by: @username
Refs: #456, #789
```

### Common Footer Keywords

| Keyword | Purpose |
|---------|---------|
| `Fixes #123` | Closes an issue |
| `Closes #123` | Closes an issue |
| `Resolves #123` | Closes an issue |
| `Refs #123` | References an issue |
| `Reviewed-by:` | Code reviewer |
| `Co-authored-by:` | Co-author |

## Examples by Scenario

### Adding a New Feature

```
feat(neural-net): Add ResNet block implementation

Residual blocks allow training of very deep networks by using
skip connections to preserve gradient flow.

Features:
- Configurable number of layers
- Optional batch normalization
- Support for both 1D and 2D convolutions

Refs: #234
```

### Fixing a Bug

```
fix(matrix): Correct transpose operation for non-square matrices

The previous implementation incorrectly handled rectangular matrices,
resulting in dimension mismatches during multiplication.

This fix properly transposes matrices of any shape.

Fixes #567
```

### Performance Optimization

```
perf(tensor): Use Span<T> for zero-allocation slicing

Replaced array allocations with Span<T> to reduce GC pressure during
tensor slicing operations. Benchmarks show 40% improvement in allocation
rate and 15% faster execution.

Closes #890
```

### Breaking Change

```
feat!: Change API to use async/await pattern

BREAKING CHANGE: All training methods now return Task<T> instead of T.
This enables better cancellation support and progress reporting.

Migration guide:
- Change `model.Train(data)` to `await model.TrainAsync(data)`
- Update method signatures to be async
- Add CancellationToken parameters where needed

Closes #1234
```

### Documentation Update

```
docs: Add comprehensive API reference for neural network module

Created detailed API documentation including:
- Class descriptions and usage examples
- Parameter explanations
- Return value specifications
- Code samples for common scenarios
```

### Dependency Update

```
chore(deps): Bump Microsoft.Data.Sqlite from 8.0.0 to 8.0.21

Updates Microsoft.Data.Sqlite to address security vulnerabilities
and improve compatibility with .NET 8.
```

## Best Practices

### DO ✅

- **Use imperative mood**: "add feature" not "added feature"
- **Keep first line under 72 characters**
- **Provide context in body for complex changes**
- **Reference related issues**
- **Use consistent scopes across the project**
- **Write clear, descriptive messages**

### DON'T ❌

- **Don't use past tense**: "Added" or "Adding"
- **Don't be vague**: "Fix bug" or "Update code"
- **Don't include period at end of subject line**
- **Don't start with lowercase letter**
- **Don't mix multiple changes in one commit**

## Good vs Bad Examples

### ❌ Bad
```
fixed bug
Update stuff
WIP
asdfasdf
quick fix
```

### ✅ Good
```
fix(matrix): Correct dimension validation in multiply operation
feat(neural-net): Add support for custom activation functions
refactor: Extract layer initialization to separate method
perf(optimizer): Cache gradient computations
docs: Update README with new installation instructions
```

## Git Configuration

Set up commit message template:

```bash
git config --global commit.template ~/.gitmessage
```

Create `~/.gitmessage`:
```
# <type>(<scope>): <subject>
# |<----  Using a Maximum Of 72 Characters  ---->|

# Explain why this change is being made
# |<----   Try To Limit Each Line to a Maximum Of 72 Characters   ---->|

# Provide links or keys to any relevant tickets, articles or other resources
# Example: Fixes #23

# --- COMMIT END ---
# Type can be:
#   feat     (new feature)
#   fix      (bug fix)
#   refactor (refactoring code)
#   perf     (performance improvement)
#   docs     (documentation)
#   test     (adding tests)
#   chore    (maintenance)
#   style    (formatting)
#   ci       (CI/CD)
# --------------------
# Remember:
#   - Use imperative mood in subject line
#   - Don't end subject line with period
#   - Separate subject from body with blank line
#   - Use body to explain what and why vs. how
#   - Reference issues and PRs in footer
```

## Tools

### Commitizen

Interactive commit message builder:

```bash
npm install -g commitizen
commitizen init cz-conventional-changelog --save-dev --save-exact
git cz
```

### Commitlint

Validate commit messages:

```bash
npm install -g @commitlint/cli @commitlint/config-conventional
echo "module.exports = {extends: ['@commitlint/config-conventional']}" > commitlint.config.js
```

### Git Hooks

Pre-commit hook to validate messages:

```bash
# .git/hooks/commit-msg
#!/bin/sh
npx --no -- commitlint --edit $1
```

## IDE Integration

### Visual Studio Code

Extensions:
- [Conventional Commits](https://marketplace.visualstudio.com/items?itemName=vivaxy.vscode-conventional-commits)
- [Commit Message Editor](https://marketplace.visualstudio.com/items?itemName=adam-bender.commit-message-editor)

### Visual Studio

Extensions:
- [Git Extensions](https://gitextensions.github.io/)
- Configure commit message template

## FAQ

### What if I forget to use conventional commits?

The release workflow will skip versioning if no conventional commits are found. Your changes won't trigger a release.

### Can I combine multiple types in one commit?

No. Each commit should have one clear purpose. If you have multiple changes, make multiple commits:

```bash
git add src/Feature.cs
git commit -m "feat: Add new feature"

git add tests/FeatureTests.cs
git commit -m "test: Add tests for new feature"
```

### What about merge commits?

Merge commits are ignored by the versioning system. Use squash merges or ensure merged branches use conventional commits.

### How do I handle WIP commits?

During development, you can use any commit message. Before merging to main, squash and rewrite commits to follow conventions:

```bash
git rebase -i HEAD~5  # Interactive rebase last 5 commits
# Squash and rewrite messages
```

## Resources

- [Conventional Commits Specification](https://www.conventionalcommits.org/)
- [Angular Commit Guidelines](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit)
- [Commitizen](http://commitizen.github.io/cz-cli/)
- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)

## Getting Help

If you have questions about conventional commits:

1. Check this guide and [VERSIONING.md](.github/VERSIONING.md)
2. Review examples in project commit history: `git log --oneline -20`
3. Ask in pull request reviews
4. Open a discussion on GitHub
