# AI Assistant Project Rules

## Code Architecture Rules

### Generic Types and INumericOperations
1. **NEVER use default(T)** - Instead use:
   - NumOps.Zero for zero values
   - NumOps.One for one values
   - NumOps.FromDouble(value) to convert from double

2. **Always include INumericOperations in base classes**:
   ```csharp
   protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
   ```

3. **Use generic types everywhere** - Avoid hardcoding double, float, int, etc.

4. **Use custom data types** - Prefer Vector<T>, Matrix<T>, Tensor<T> over arrays and collections

5. **NEVER use constraints like where T : struct** - Our architecture doesn't use these

### Inheritance Pattern
1. **Always create interfaces first**
2. **Then create base classes with common methods**
3. **Concrete implementations MUST inherit from base class, NOT directly from interface**

### Documentation
1. **Follow existing documentation format exactly**
2. **Include XML documentation for all public members**
3. **Add beginner-friendly explanations in remarks**

## File Management Rules

### DO NOT Create Unless Explicitly Requested
1. **NO README files** (AGENTS.md, STATUS.md, REPORT.md, etc.)
2. **NO one-off scripts** unless specifically asked
3. **NO temporary analysis files**
4. **NO status tracking markdown files**

### What TO Create
1. Source code files (.cs)
2. Configuration files when needed for functionality
3. Documentation only when explicitly requested by name/path

## Git Workflow

1. **Always work on feature branches**
2. **Commit early and often**
3. **Use descriptive commit messages**
4. **Create PR when complete**
5. **DO NOT close PRs**, fix them properly instead

## Code Organization

1. **Folders match namespaces** - No "Base" folder, use proper namespace hierarchy
2. **Examples go in separate test project** - `testconsole/AiDotNetTestConsole.csproj`
3. **Use StatisticsHelper for common calculations** - Don't duplicate code
4. **Integrate with existing architecture** - Must work with PredictionModelBuilder and pipeline builder

## SOLID & DRY Principles

1. **Always follow SOLID principles**
2. **Always follow DRY principles**
3. **Reuse existing helper classes** (StatisticsHelper, MathHelper, etc.)
4. **Don't duplicate functionality**

## Null Handling

1. **NO nullable suppress operator (!)** unless absolutely necessary
2. **Perform proper null checks**
3. **Throw exceptions for invalid states**
4. **Support older .NET Framework versions** - Don't use features that break compatibility
