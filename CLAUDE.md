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

## Common Mistakes to Avoid

1. **Using IModel instead of IFullModel** - Always use IFullModel for model references
2. **Using object for type erasure** - Use proper base interfaces instead
3. **Using `required` keyword** - **CRITICAL**: Causes CS0656 error on net462, use constructors instead
4. **Using .NET 6+ only APIs** - Check compatibility with net462 target framework
5. **Removing generic type parameters** - Maintain flexibility unless explicitly required
6. **Missing parameter names in exceptions** - Always use `nameof(param)` in ArgumentException

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
