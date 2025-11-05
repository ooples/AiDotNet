# AiDotNet Project Rules

## Code Architecture Rules

### 1. Generic Types and Numeric Operations
- **ALWAYS** use generic types with `INumericOperations<T>` interface
- **NEVER** hardcode `double`, `float`, or specific numeric types
- **NEVER** use `default(T)` - Instead use:
  - `NumOps.Zero` for zero values
  - `NumOps.One` for one values
  - `NumOps.FromDouble(value)` to convert from double
- **NEVER** request `INumericOperations<T>` in public constructors - handled internally
- **Always include in base classes**:
  ```csharp
  protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
  ```
- Use custom data types: `Vector<T>`, `Matrix<T>`, `Tensor<T>` instead of arrays/collections

### 2. Class Organization
- **NEVER** use constraints like `where T : struct` or `where T : IComparable<T>`
- Each class, enum, interface must be in its own separate file
- All interfaces belong in root `Interfaces` folder, NOT in sub-folders
- Folder structure mirrors namespace structure (e.g., `RAG/DocumentStores`, `RAG/ChunkingStrategies`)
- **NO** "Base" folders - use namespace-based organization

### 3. Inheritance Pattern
- Every feature has: Interface → Base Class → Concrete Implementations
- Concrete classes inherit from Base Class, NOT directly from Interface
- Base classes contain common functionality using `INumericOperations<T>`

### 4. Integration with PredictionModelBuilder
- **ALL** new features MUST integrate with existing `PredictionModelBuilder.cs` pipeline
- **NO** standalone configuration builders - extend existing builder pattern
- New features must work within existing model building process

### 5. Code Quality
- Follow SOLID principles
- Follow DRY principles
- Use existing helpers (`StatisticsHelper`, `MathHelper`, etc.) - don't duplicate functionality
- Proper null checks with exceptions, NOT nullable operators (`!`) for .NET Framework compatibility
- **ALWAYS** ensure new code includes unit tests with a minimum of 80% code coverage.

### 6. Default Values
- All string properties: `= string.Empty;`
- All collection properties: `= new List<T>();` or appropriate empty collection
- **NEVER** leave properties without default values

### 7. Documentation
- **Follow existing documentation format exactly**
- **Include XML documentation for all public members**
- **Add beginner-friendly explanations in remarks**
- Document parameters, return values, exceptions, and usage examples
- Include performance characteristics and thread-safety considerations

## File Management Rules

### DO NOT Create Unless Explicitly Requested:
- ❌ README.md files (STATUS.md, REPORT.md, AGENTS.md, etc.)
- ❌ Status reports or analysis documents (.md files)
- ❌ One-off scripts (.ps1, .py unless requested)
- ❌ .json configuration files in repo root
- ❌ Instructions or temporary files (.txt)

### DO NOT Check In:
- Temporary analysis files
- Personal scripts
- Debug files
- Reports or summaries

### What TO Create:
- Source code files (.cs)
- Configuration files when needed for functionality
- Documentation only when explicitly requested by name/path

## Git Workflow

### Branch Management
- Create feature branches for new work: `feature/issue-XXX-description`
- Commit early and often with descriptive messages
- Create PR when work is complete
- **DO NOT close PRs** - fix them properly instead
- **NO** cleanup commits on unrelated branches
- Cherry-pick orphaned commits to correct branches if needed

### PR Comments
- Use GitHub GraphQL API to get unresolved comments
- Fix each comment with production-ready code
- Commit fixes
- Mark comments as resolved using GraphQL API
- Repeat until no unresolved comments remain

## Development Process

### Never Use Scripts For:
- File editing/refactoring (use tools directly)
- Bulk changes (they break things)

### Always Use:
- Direct tool calls (edit, create, view)
- Manual verification
- Incremental changes

### When Working on Features:
1. Use Google Gemini CLI for comprehensive codebase analysis (2M token context)
   - **ALWAYS** use model `gemini-2.5-flash` for analysis
2. Check existing implementations before duplicating
3. Integrate with existing architecture
4. Follow established patterns exactly
5. Complete ALL work without stopping when given a complete plan

## Null Handling

- **NO nullable suppress operator (!)** unless absolutely necessary
- Perform proper null checks
- Throw exceptions for invalid states
- Support older .NET Framework versions - don't use features that break compatibility

## Examples Project

- Path: `C:\Users\cheat\source\repos\AiDotNet\testconsole\AiDotNetTestConsole.csproj`
- Examples go here, NOT in main library

## Authentication

- GitHub Token: Use configured MCP authentication
- GraphQL API: Already authenticated through MCP server
- Don't re-authenticate when already working

## Remember

- Reference these rules EVERY time before making changes
- If unsure, check existing code for patterns
- Complete work without pausing when plan is clear
- Don't create files unless explicitly requested
