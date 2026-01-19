---
name: Feature User Story
about: Create a new feature user story with complete architectural requirements
title: '[Feature] '
labels: enhancement
assignees: ''
---

### User Story

> As a [role], I want [feature], so that [benefit].

---

### Problem Statement

[Describe what's missing and why it's needed]

---

### Phase 1: [Feature Name]

**Goal:** [What this phase accomplishes]

#### AC 1.1: Create Interface and Base Class (X points)

**Requirement:** Implement interface and base class following AiDotNet architecture.

##### Class Organization (REQUIRED)

- [ ] **Interface:** Create `I{FeatureName}.cs` in `src/Interfaces/` (root level, NOT subfolders)
  - Define contract methods for the feature
  - Include XML documentation with beginner-friendly remarks

- [ ] **Base Class:** Create `{FeatureName}Base.cs` in `src/{FeatureArea}/`
  - Inherit from `I{FeatureName}<T>` (or appropriate interface)
  - Include `protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();`
  - Implement common functionality shared by all concrete classes
  - NO hardcoded `double`/`float` types - use `T` with `INumericOperations<T>`

##### INumericOperations<T> Requirements (CRITICAL)

- [ ] **Protected Field:** Include `protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();` in base class
- [ ] **Numeric Operations:**
  - NEVER hardcode `double`, `float`, or specific numeric types
  - NEVER use `default(T)` - use `NumOps.Zero` instead
  - Use `NumOps.Zero` for zero values
  - Use `NumOps.One` for one values
  - Use `NumOps.FromDouble(value)` to convert from double
  - Use `NumOps.Add`, `NumOps.Multiply`, `NumOps.Divide`, etc. for arithmetic
  - Use `NumOps.LessThan`, `NumOps.GreaterThan`, etc. for comparisons

##### Property Initialization (CRITICAL)

- [ ] **NO `default!` Operator:** Never use `default!` (not compatible with .NET Framework 4.6.2)
- [ ] **String Properties:** Initialize with `= string.Empty;`
- [ ] **Collection Properties:** Initialize with `= new List<T>();` or appropriate empty collection
- [ ] **Numeric Properties:** Initialize with appropriate default or `NumOps.Zero`
- [ ] **Optional Properties:** Use nullable types (`T?`) and initialize with `= null;` only if null is semantically correct

#### AC 1.2: Create Concrete Implementation (X points)

**Requirement:** Implement concrete class with industry-standard defaults.

- [ ] **File:** `src/{FeatureArea}/{ImplementationName}.cs`
- [ ] **Class:** `public class {ImplementationName}<T> : {FeatureName}Base<T>`
- [ ] **Namespace:** `namespace AiDotNet.{FeatureArea};` (must mirror folder structure)
- [ ] **Constructor with Defaults:** Use one of these patterns:

**Pattern A: Constructor Parameters** (for 3-7 parameters)
```csharp
public {ImplementationName}(
    param1Type param1,
    param2Type param2,
    double hyperParam1 = [DEFAULT],  // Default: [RESEARCH-BASED VALUE]
    int hyperParam2 = [DEFAULT])     // Default: [STANDARD VALUE]
{
    // Validation
    if (hyperParam1 < 0 || hyperParam1 > 1)
        throw new ArgumentException("hyperParam1 must be between 0 and 1", nameof(hyperParam1));

    // Initialization
    _hyperParam1 = NumOps.FromDouble(hyperParam1);
}
```

**Pattern B: Options Class** (for 8+ parameters)
```csharp
public class {ImplementationName}Options
{
    public double HyperParam1 { get; set; } = [DEFAULT];  // Cite: [Paper/Standard]
    public int HyperParam2 { get; set; } = [DEFAULT];     // Cite: [Research]
}

public {ImplementationName}({ImplementationName}Options? options = null)
{
    var opts = options ?? new {ImplementationName}Options();
    _hyperParam1 = NumOps.FromDouble(opts.HyperParam1);
}
```

- [ ] **Default Values:** Must be based on industry standards/research:
  - Document WHY each default was chosen (cite papers/standards)
  - Include in XML documentation

#### AC 1.3: AiModelBuilder Integration (REQUIRED)

**Requirement:** Integrate feature with existing builder pipeline.

- [ ] **Add Private Field:** Add to `AiModelBuilder.cs`:
  ```csharp
  private I{FeatureName}<T, TInput, TOutput>? _{featureName};
  ```

- [ ] **Add Configure Method:** Add to `AiModelBuilder.cs`:
  ```csharp
  /// <summary>
  /// Configures the {feature description}.
  /// </summary>
  /// <param name="{featureName}">The {feature} implementation to use.</param>
  /// <returns>This builder instance for method chaining.</returns>
  /// <remarks>
  /// <b>For Beginners:</b> [Explain what this feature does in simple terms]
  ///
  /// [Provide real-world analogy or example]
  /// </remarks>
  public IAiModelBuilder<T, TInput, TOutput> Configure{FeatureName}(
      I{FeatureName}<T, TInput, TOutput> {featureName})
  {
      _{featureName} = {featureName};
      return this;
  }
  ```

**CRITICAL Configure Method Pattern:**
- MUST take ONLY the interface (no additional parameters)
- MUST store in private field (no processing/validation)
- MUST return `this` for method chaining
- MUST include XML documentation with beginner-friendly remarks

- [ ] **Use in Build():** Modify `Build()` method in `AiModelBuilder.cs`:
  ```csharp
  // Create default if not configured
  var {featureName} = _{featureName} ?? new Default{FeatureName}<T, TInput, TOutput>();

  // Actually USE the feature in the pipeline
  var result = {featureName}.Process(...);
  // ... integrate result into training/prediction flow
  ```

- [ ] **Verify Actual Usage:** Feature must be ACTUALLY USED in execution flow, not just configured

#### AC 1.4: Documentation (REQUIRED)

**Requirement:** Complete XML documentation with beginner-friendly explanations.

- [ ] **XML Documentation:** All public members must have XML docs including:
  - Class/interface summary
  - `<typeparam name="T">` for all generic types
  - `<param>` tags for all parameters
  - `<returns>` tags for all return values
  - `<exception>` tags for all thrown exceptions

- [ ] **Beginner-Friendly Remarks:** All public APIs must include:
  ```xml
  /// <remarks>
  /// <b>For Beginners:</b> [Simple explanation]
  ///
  /// [Real-world analogy or example]
  ///
  /// [Common use cases]
  ///
  /// [Default behavior and why]
  /// </remarks>
  ```

- [ ] **Default Value Justification:** Document WHY defaults were chosen:
  ```csharp
  /// <param name="learningRate">The learning rate. Default: 0.001 (Adam optimizer standard from Kingma & Ba, 2014)</param>
  ```

- [ ] **Follow Existing Format:** Match documentation style of similar components (e.g., ActivationFunctionBase, AiModelBuilder)

#### AC 1.5: Unit Tests (REQUIRED)

**Requirement:** Comprehensive unit tests with 80%+ code coverage.

- [ ] **Test File:** Create `tests/UnitTests/{FeatureArea}/{ImplementationName}Tests.cs`
- [ ] **Namespace:** `namespace AiDotNet.Tests.{FeatureArea};`
- [ ] **Minimum 80% Code Coverage:** Verify with coverage tools

##### Required Test Cases:

- [ ] **Generic Type Testing:** Test with multiple numeric types
  ```csharp
  [Theory]
  [InlineData(typeof(double))]
  [InlineData(typeof(float))]
  public void Method_ShouldWork_ForDifferentNumericTypes(Type numericType)
  ```

- [ ] **Default Value Testing:**
  ```csharp
  [Fact]
  public void Constructor_WithNoParameters_ShouldUseStandardDefaults()
  {
      var instance = new {ImplementationName}<double>();
      Assert.Equal(EXPECTED_DEFAULT, instance.Property);
  }
  ```

- [ ] **Edge Case Testing:**
  - Null inputs (where applicable)
  - Empty collections
  - Zero values
  - Maximum/minimum values
  - Boundary conditions

- [ ] **Exception Testing:**
  ```csharp
  [Fact]
  public void Constructor_WithInvalidParameter_ShouldThrowArgumentException()
  {
      Assert.Throws<ArgumentException>(() => new {ImplementationName}<double>(invalidValue));
  }
  ```

- [ ] **Integration Testing** (if feature integrates with AiModelBuilder):
  ```csharp
  [Fact]
  public void Configure{FeatureName}_ShouldBeUsedInBuild()
  {
      var feature = new {ImplementationName}<double>();
      var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
          .Configure{FeatureName}(feature)
          .ConfigureModel(mockModel);

      var result = builder.Build(x, y);

      // Verify feature was actually used
      mockModel.Verify(m => m.{MethodUsingFeature}(...), Times.AtLeastOnce());
  }
  ```

- [ ] **INumericOperations Testing:**
  ```csharp
  [Fact]
  public void Activate_ShouldUseNumOps_NotHardcodedTypes()
  {
      // Verify no direct arithmetic operations, only NumOps usage
  }
  ```

---

### Definition of Done

- [ ] All acceptance criteria are complete
- [ ] All classes follow Interface → Base → Concrete pattern
- [ ] All numeric operations use INumericOperations<T> (no hardcoded double/float)
- [ ] All properties properly initialized (no `default!`)
- [ ] All interfaces in `src/Interfaces/` (root level)
- [ ] Integrated with AiModelBuilder (Configure method + usage in Build())
- [ ] All defaults based on research/industry standards and documented
- [ ] Complete XML documentation with beginner-friendly remarks
- [ ] All unit tests pass with 80%+ code coverage
- [ ] Code follows SOLID and DRY principles
- [ ] No duplicate functionality (checked against existing helpers)

---

### Additional Notes

[Any additional context, research papers to reference, or implementation details]

---

**Reference Documents:**
- **Architectural Requirements:** `.github/USER_STORY_ARCHITECTURAL_REQUIREMENTS.md`
- **Project Rules:** `.github/PROJECT_RULES.md`
- **Example Implementations:** See `src/ActivationFunctions/` for Interface → Base → Concrete pattern
