# AiDotNet Copilot Instructions - CRITICAL RULES

## Architecture Rules (DO NOT VIOLATE)

### 1. Generic Types and Numeric Operations
- **ALWAYS** use generic types with `INumericOperations<T>` interface
- **NEVER** hardcode `double`, `float`, or specific numeric types
- Use `NumOps.FromDouble()`, `NumOps.Zero`, `NumOps.One` instead of literals or `default(T)`
- **NEVER** request `INumericOperations<T>` in public constructors - this is handled internally
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
- Use existing helpers (`StatisticsHelper`, etc.) - don't duplicate functionality
- Proper null checks with exceptions, NOT nullable operators (`!`) for .NET Framework compatibility

### 6. Default Values
- All string properties: `= string.Empty;`
- All collection properties: `= new List<T>();` or appropriate empty collection
- **NEVER** leave properties without default values

## File Management Rules

### DO NOT Create Unless Explicitly Requested:
- ❌ README.md files
- ❌ Status reports (.md files)
- ❌ Analysis documents
- ❌ One-off scripts (.ps1, .py unless requested)
- ❌ .json configuration files in repo root
- ❌ Instructions files (.txt)

### DO NOT Check In:
- Temporary analysis files
- Personal scripts
- Debug files
- Reports or summaries

## Git Workflow

### Branch Management
- Create feature branches for new work: `feature/issue-XXX-description`
- Commit early and often with descriptive messages
- Create PR when work is complete
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
2. Check existing implementations before duplicating
3. Integrate with existing architecture
4. Follow established patterns exactly
5. Complete ALL work without stopping when given a complete plan

## Examples Project
- Path: `C:\Users\cheat\source\repos\AiDotNet\testconsole\AiDotNetTestConsole.csproj`
- Examples go here, NOT in main library

## Authentication
- GitHub Token: Use configured MCP authentication
- GraphQL API: Already authenticated through MCP server
- Don't re-authenticate when already working

## Remember
- Save these instructions and reference them EVERY time before making changes
- If unsure, check existing code for patterns
- Complete work without pausing when plan is clear
- Don't create files unless explicitly requested
