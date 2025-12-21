## Production-Ready PR Checklist

### Code Implementation
- [ ] Code follows project style guidelines
- [ ] Comprehensive XML documentation added for all public APIs
- [ ] No TODO/FIXME comments in final code
- [ ] Proper error handling implemented with meaningful messages
- [ ] Thread safety considered where applicable
- [ ] Security review completed (no hardcoded secrets, proper validation)
- [ ] Performance implications analyzed

### Testing
- [ ] Unit tests added with >90% code coverage
- [ ] Integration tests added for component interactions
- [ ] Performance benchmarks included where relevant
- [ ] All edge cases tested (null checks, boundary conditions)
- [ ] Exception paths tested
- [ ] Tests pass locally before PR submission
- [ ] Tests pass in CI/CD pipeline

### Documentation
- [ ] README.md updated if applicable
- [ ] Usage examples provided in documentation
- [ ] API documentation complete with parameters and return values
- [ ] Exception documentation included
- [ ] Migration guide for breaking changes
- [ ] CHANGELOG.md updated

### Validation
- [ ] Manual testing completed for key scenarios
- [ ] Performance requirements met (benchmarks pass)
- [ ] Memory usage within acceptable limits
- [ ] No memory leaks detected in profiling
- [ ] Backwards compatibility verified
- [ ] Works on all target frameworks (.NET Framework 4.7.1, .NET 8)
- [ ] Cross-platform compatibility tested (Windows/Linux/macOS)

### Review Process
- [ ] Self-review completed using checklist
- [ ] At least one peer review obtained
- [ ] All reviewer feedback addressed
- [ ] Code formatting passes (dotnet format)
- [ ] Static analysis passes (SonarQube warnings addressed)
- [ ] Security scan passes (no critical vulnerabilities)
## User Story / Context
- Reference: [US-XXX] (if applicable)
- Base branch: `master` (default) or feature branch if stacking PRs

## Summary
- What changed and why (scoped strictly to the user story / PR intent)

## Verification
- [ ] Builds succeed (scoped to changed projects)
- [ ] Unit tests pass locally
- [ ] Code coverage >= 90% for touched code
- [ ] Codecov upload succeeded (if token configured)
- [ ] TFM verification (net471, net8.0) passes (if packaging)
- [ ] No unresolved GitHub review comments on HEAD

## Copilot Review Loop (Outcome-Based)
Record counts before/after your last push:
- Comments on HEAD BEFORE: [N]
- Comments on HEAD AFTER (60s): [M]
- Final HEAD SHA: [sha]

## Files Modified
- [ ] List files changed (must align with scope)

## Performance Characteristics
- **Time Complexity**: <!-- O(n) notation -->
- **Space Complexity**: <!-- O(n) notation -->
- **Memory Usage**: <!-- MB/GB if relevant -->
- **Throughput**: <!-- Operations per second -->
- **Latency**: <!-- Average response time -->

## Breaking Changes
- <!-- List any breaking changes here -->
### Migration Guide
- <!-- Provide migration steps if breaking changes -->

## Security Considerations
- [ ] No hardcoded secrets or credentials
- [ ] Input validation implemented
- [ ] Output encoding where applicable
- [ ] Proper random number generation
- [ ] Safe file handling

## Additional Context
<!-- Any additional information reviewers should know -->

## Related Issues
- Closes #123
- Related to #456

## Screenshots (if applicable)
<!-- Add screenshots for UI changes -->

## Notes
- Any follow-ups, caveats, or migration details
