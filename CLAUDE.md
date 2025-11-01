# AiDotNet Project Guidelines for Claude Code

## Critical Code Standards

### 1. Interface Usage - IFullModel vs IModel

**ALWAYS use `IFullModel` as the base interface, NEVER `IModel`**

- ❌ **WRONG**: `IModel<TInput, TOutput, TMetadata>`
- ✅ **CORRECT**: `IFullModel<T, TInput, TOutput>`

**Reason**: `IFullModel` includes all necessary capabilities (training, prediction, serialization, parameterization) while `IModel` is just the basic interface. All models in this codebase should use the full model interface.

**Example**:
```csharp
// WRONG - Do not use IModel
public void SetBaseModel<TInput, TOutput, TMetadata>(IModel<TInput, TOutput, TMetadata> model)

// CORRECT - Use IFullModel
public void SetBaseModel<TInput, TOutput>(IFullModel<T, TInput, TOutput> model)
```

### 2. Type Safety - Never Use `object` for Model Storage

**ALWAYS use strongly-typed interfaces, NEVER use `object`**

- ❌ **WRONG**: `protected object? _baseModel;`
- ✅ **CORRECT**: `protected IFullModel<T, TInput, TOutput>? _baseModel;`

**Reason**: Using `object` removes all type safety and requires runtime casting. Use the appropriate base interface (`IFullModel`) to maintain compile-time type checking.

### 3. .NET Framework Compatibility

**Target Frameworks**: This project targets multiple frameworks including `net462` (.NET Framework 4.6.2)

**Critical**: Do NOT use .NET 6+ or C# 11+ only features

#### Forbidden Features (Not Compatible with net462):

**❌ NEVER USE: `required` keyword** (C# 11/.NET 7+)
```csharp
// WRONG - Causes CS0656 error on net462
public class MyClass
{
    public required string Name { get; set; }
}

// CORRECT - Use constructor with parameters or make nullable
public class MyClass
{
    public string Name { get; set; }

    public MyClass(string name)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
    }
}
```

**❌ NEVER USE: `ArgumentNullException.ThrowIfNull()`** (.NET 6+)
```csharp
// WRONG
ArgumentNullException.ThrowIfNull(param);

// CORRECT
if (param == null) throw new ArgumentNullException(nameof(param));
```

**Other .NET 6+/C# 11+ features to avoid**:
- ❌ `required` keyword (C# 11) - **CRITICAL: Causes CS0656 on net462**
- ❌ File-scoped namespaces (use block-scoped: `namespace Foo { }`)
- ❌ Global using directives
- ❌ Raw string literals (`"""text"""`)
- ❌ List patterns in pattern matching
- ❌ UTF-8 string literals (`"text"u8`)
- ❌ Generic attributes without generic parameter
- ❌ Static abstract members in interfaces

### 4. Generic Type Flexibility

**Preserve generic type parameters for flexibility**

When updating interfaces, maintain generic `TInput` and `TOutput` parameters to allow users to choose different data types instead of forcing specific types like `Tensor<T>`.

- ❌ **WRONG**: Hardcoding to specific types
  ```csharp
  void SetBaseModel(IFullModel<T, Tensor<T>, Tensor<T>> model)
  ```
- ✅ **CORRECT**: Keeping flexibility with generics
  ```csharp
  void SetBaseModel<TInput, TOutput>(IFullModel<T, TInput, TOutput> model)
  ```

### 5. ArgumentException Best Practices

**Always include parameter name in ArgumentException**

```csharp
// CORRECT
throw new ArgumentException($"Expected {ExpectedParameterCount} parameters, but got {parameters.Length}", nameof(parameters));
```

### 6. UTF-8 Encoding and Character Corruption Prevention

**CRITICAL**: Encoding issues have been a serious recurring problem in this project. Follow these rules strictly:

#### File Encoding Standards

**✅ ALWAYS:**
- Save all C# source files as **UTF-8 without BOM** (Byte Order Mark)
- Use proper Unicode characters for mathematical symbols
- Verify file encoding before committing

**❌ NEVER:**
- Use UTF-8 with BOM (causes build issues and appears as `﻿` in some editors)
- Copy-paste from Word documents or web pages without checking encoding
- Use Windows-1252 or other legacy encodings
- Leave corrupted placeholder characters (�) in code

#### Mathematical Symbols - Use Correct Unicode

**Common mathematical symbols that get corrupted:**

| Symbol | Unicode | HTML | ASCII Fallback | Common Corruption |
|--------|---------|------|----------------|-------------------|
| Multiplication | × | `&times;` | * | � |
| Superscript ² | ² | `&sup2;` | ^2 | � |
| Em-dash | — | `&mdash;` | -- | � |
| Degree | ° | `&deg;` | deg | � |
| Plus-minus | ± | `&plusmn;` | +/- | � |

**Examples:**

```csharp
// ❌ WRONG - Corrupted characters
/// price = 50,000 � bedrooms + 100 � square_feet
/// Maximum conductance (mS/cm�)
/// Information moves in one direction�forward�from input

// ✅ CORRECT - Proper Unicode
/// price = 50,000 × bedrooms + 100 × square_feet
/// Maximum conductance (mS/cm²)
/// Information moves in one direction -- forward -- from input

// ✅ ACCEPTABLE - ASCII fallback if Unicode causes issues
/// price = 50,000 * bedrooms + 100 * square_feet
/// Maximum conductance (mS/cm^2)
/// Information moves in one direction -- forward -- from input
```

#### Pre-Commit Checks for Encoding

**Before committing, verify:**

1. **Check for BOM markers:**
   ```bash
   # Search for files with BOM
   grep -rlI $'^\xEF\xBB\xBF' src/
   ```

2. **Check for corrupted characters:**
   ```bash
   # Search for replacement character
   grep -rn "�" src/
   ```

3. **Verify file encoding:**
   - Visual Studio: File → Advanced Save Options → UTF-8 without BOM
   - VS Code: Bottom-right status bar → UTF-8
   - Command line: `file --mime-encoding filename.cs`

#### Common Corruption Sources

1. **Copy-paste from documentation**:
   - Always review pasted content for encoding issues
   - Use "Paste as Plain Text" when possible

2. **AI-generated content**:
   - AI sometimes generates Unicode that doesn't render properly
   - Always verify mathematical symbols after AI assistance

3. **Cross-platform development**:
   - Windows, Mac, and Linux handle encoding differently
   - Use `.editorconfig` to enforce UTF-8

4. **Git operations**:
   - Configure Git to not modify line endings for .cs files
   - Use `.gitattributes` to specify text handling

#### EditorConfig Settings (Already in project)

The project's `.editorconfig` should include:

```ini
[*.cs]
charset = utf-8
insert_final_newline = true
end_of_line = lf
```

#### Incident History

**2025-10-30**: Multiple PRs had encoding issues:
- PR #242: Documentation had � instead of *, ², --
- PR #252: ExtremeLearningMachine, NEAT, RestrictedBoltzmannMachine had �
- Impact: Build failures, incorrect documentation, CodeRabbit reviews blocked merging

**Prevention**: This section added to CLAUDE.md to prevent recurrence.

## Critical Files - NEVER DELETE OR EMPTY

**ABSOLUTE PROHIBITION**: The following base class files are critical to the project and protected by pre-commit hooks:

- `src/Regression/RegressionBase.cs` - Base class for all regression models (746 lines)
- `src/Optimizers/OptimizerBase.cs` - Base class for optimization algorithms (286 lines)
- `src/Models/NeuralNetworkModel.cs` - Core neural network implementation
- `src/TimeSeries/TimeSeriesModelBase.cs` - Base class for time series models
- `src/Regression/DecisionTreeRegressionBase.cs` - Base class for decision tree regression
- `src/Regression/DecisionTreeAsyncRegressionBase.cs` - Base class for async decision tree regression
- `src/Regression/NonLinearRegressionBase.cs` - Base class for non-linear regression

**If you need to modify these files**:
1. ✅ Adding new methods is OK
2. ✅ Fixing bugs in existing methods is OK
3. ✅ Improving documentation is OK
4. ❌ NEVER delete or empty these files
5. ❌ NEVER remove critical methods without team discussion
6. ⚠️ Refactoring requires creating new files first, then migrating

**Pre-commit hook protection**: Commits that delete or empty these files (< 100 bytes) will be automatically blocked.

**Incident history**: On 2025-10-23, RegressionBase.cs and OptimizerBase.cs were accidentally emptied. See `.claude/CRITICAL_FILE_SAFEGUARDS.md` for details.

## Common Mistakes to Avoid

1. **Using IModel instead of IFullModel** - Always use IFullModel for model references
2. **Using object for type erasure** - Use proper base interfaces instead
3. **Using `required` keyword** - **CRITICAL**: Causes CS0656 error on net462, use constructors instead
4. **Using .NET 6+ only APIs** - Check compatibility with net462 target framework
5. **Removing generic type parameters** - Maintain flexibility unless explicitly required
6. **Missing parameter names in exceptions** - Always use `nameof(param)` in ArgumentException
7. **Deleting or emptying critical base class files** - Protected by pre-commit hooks (see above)
8. **Using hardcoded primitive types** - Use generic types (TInput, TOutput, T) instead of double[][]

## Common Build Errors and Solutions

### CS0656: Missing compiler required member 'RequiredMemberAttribute'

**Error**:
```
error CS0656: Missing compiler required member 'System.Runtime.CompilerServices.RequiredMemberAttribute..ctor'
```

**Cause**: Using the `required` keyword which requires C# 11/.NET 7+

**Solution**: Remove `required` keyword and use constructor parameters instead:
```csharp
// BEFORE (causing CS0656):
public class FairnessMetrics<T>
{
    public required T DemographicParity { get; set; }
    public required T EqualOpportunity { get; set; }
}

// AFTER (compatible with net462):
public class FairnessMetrics<T>
{
    public T DemographicParity { get; set; }
    public T EqualOpportunity { get; set; }

    public FairnessMetrics(T demographicParity, T equalOpportunity)
    {
        DemographicParity = demographicParity;
        EqualOpportunity = equalOpportunity;
    }
}
```

### CS0535: Class does not implement interface member

**Error**:
```
error CS0535: 'MyClass' does not implement interface member 'IFullModel.SaveModel(string)'
```

**Cause**: Class implements `IFullModel` but is missing required methods

**Solution**: Implement all interface members (SaveModel, LoadModel, SetParameters, ParameterCount, GetFeatureImportance, SetActiveFeatureIndices, Clone)

## Build Targets

- net8.0
- net7.0
- net6.0
- net462 (minimum framework - all code must be compatible)

## Testing Changes

Always verify compatibility across all target frameworks:

```bash
dotnet build
```

This will build for all target frameworks and catch compatibility issues.
