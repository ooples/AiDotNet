# AiDotNet User Story - Architectural Requirements Gap Analysis

**Last Updated:** 2025-01-06
**Purpose:** Document missing architectural requirements in GitHub issues to prevent repeated implementation mistakes

---

## Executive Summary

**Codebase Health:** ✅ Excellent
- 280+ uses of `INumericOperations<T>` - consistently applied
- 67 interfaces properly organized in `src/Interfaces/`
- 0 uses of `default!` operator - no violations
- Consistent Interface → Base → Concrete inheritance pattern
- 14 Configure methods in PredictionModelBuilder (mostly correct pattern)

**User Story Quality:** ❌ Critical Gaps
- Missing INumericOperations<T> enforcement
- Missing inheritance pattern requirements
- Missing PredictionModelBuilder integration requirements
- Missing beginner-friendly defaults specifications
- Missing property initialization guidelines
- Missing class organization rules
- Missing documentation standards
- Inadequate testing requirements

---

## Gap 1: INumericOperations<T> Requirements

### The Problem

User stories specify generic types (`public class SomeClass<T>`) but don't enforce that numeric operations must use `INumericOperations<T>`.

### Current Codebase Pattern (Correct)

```csharp
// From ActivationFunctionBase.cs:28
public abstract class ActivationFunctionBase<T> : IActivationFunction<T>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public virtual T Activate(T input)
    {
        return input; // Uses T, not double
    }
}
```

### What User Stories Currently Say ❌

**Example from Issue #335 (SHAP Explainer):**
```markdown
- [ ] **Class:** `public class SHAPExplainer<T>`
- [ ] **Method:** `public Vector<T> Explain(Vector<T> instance)`
```

### What User Stories MUST Say ✅

```markdown
- [ ] **Class:** `public class SHAPExplainer<T>` with `INumericOperations<T>` support
- [ ] **Protected Field:** Must include `protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();`
- [ ] **Numeric Operations:**
  - NEVER hardcode `double`, `float`, or specific numeric types
  - NEVER use `default(T)` - use `NumOps.Zero` instead
  - Use `NumOps.Zero` for zero values
  - Use `NumOps.One` for one values
  - Use `NumOps.FromDouble(value)` to convert from double
  - Use `NumOps.Add`, `NumOps.Multiply`, etc. for arithmetic
```

### Examples from Codebase

**✅ Correct Usage (280+ instances):**
```csharp
// From ReLUActivation.cs
T zero = NumOps.Zero;
if (NumOps.LessThan(input, zero))
    return zero;
return input;
```

**❌ Incorrect (would violate PROJECT_RULES):**
```csharp
// DON'T DO THIS:
double zero = 0.0;  // Hardcoded type
if (input < zero)   // Direct comparison
    return default(T);  // Using default
```

---

## Gap 2: Inheritance Pattern Requirements

### The Problem

User stories specify individual classes but don't require the Interface → Base → Concrete pattern.

### Required Pattern

Every feature must have three layers:

1. **Interface** in `src/Interfaces/{InterfaceName}.cs`
2. **Base Class** in `src/{FeatureArea}/{FeatureName}Base.cs`
3. **Concrete Classes** in `src/{FeatureArea}/{ImplementationName}.cs`

### Current Codebase Pattern (Correct)

```
src/Interfaces/IActivationFunction.cs          ← Interface (root Interfaces folder)
    ↓
src/ActivationFunctions/ActivationFunctionBase.cs  ← Base class with INumericOperations
    ↓
src/ActivationFunctions/ReLUActivation.cs      ← Concrete implementation
src/ActivationFunctions/SigmoidActivation.cs   ← Another concrete implementation
```

### What User Stories Currently Say ❌

**Example from Issue #332 (Dropout):**
```markdown
- [ ] **File:** `src/Regularization/DropoutRegularization.cs`
- [ ] **Class:** `public class DropoutRegularization<T> : RegularizationBase<T>`
```

### What User Stories MUST Say ✅

```markdown
### Inheritance Pattern (REQUIRED)

- [ ] **Interface:** Create `IRegularization.cs` in `src/Interfaces/` (if not exists)
  - Define contract methods for the feature
  - Include XML documentation

- [ ] **Base Class:** Create `RegularizationBase.cs` in `src/Regularization/` (if not exists)
  - Inherit from `IRegularization<T, TInput, TOutput>`
  - Include `protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();`
  - Implement common functionality shared by all concrete classes

- [ ] **Concrete Class:** Create `DropoutRegularization.cs` in `src/Regularization/`
  - Inherit from `RegularizationBase<T>` (NOT directly from interface)
  - Implement feature-specific logic
```

### Why This Matters

- **Consistency:** All features follow the same pattern
- **Reusability:** Common functionality in base class (NumOps, validation, etc.)
- **Maintainability:** Changes to common behavior only need to update base class
- **Testability:** Can test base class separately from concrete implementations

---

## Gap 3: PredictionModelBuilder Integration

### The Problem

**CRITICAL:** User stories don't mention integrating with `PredictionModelBuilder` at all, yet PROJECT_RULES.md states:

> **ALL** new features MUST integrate with existing `PredictionModelBuilder.cs` pipeline

### What Integration Means

Every new feature must:

1. Add a private nullable field to `PredictionModelBuilder`
2. Add a `Configure{Feature}` method following the exact pattern
3. Use the feature in `Build()` or `Predict()` methods
4. Provide sensible defaults if not configured

### Current Pattern from PredictionModelBuilder

```csharp
// From PredictionModelBuilder.cs:

// 1. Private nullable field
private IRegularization<T, TInput, TOutput>? _regularization;

// 2. Configure method (takes ONLY interface, returns builder)
public IPredictionModelBuilder<T, TInput, TOutput> ConfigureRegularization(
    IRegularization<T, TInput, TOutput> regularization)
{
    _regularization = regularization;  // Just store it
    return this;                       // Return for chaining
}

// 3. Use in Build() with default
var regularization = _regularization ?? new NoRegularization<T, TInput, TOutput>();
// ... actually use regularization during training
```

### What User Stories Currently Say ❌

**Example from Issue #333 (Cross-Validation):**
```markdown
- [ ] **File:** `src/Validation/Splitters/KFoldSplitter.cs`
- [ ] **Class:** `public class KFoldSplitter<T>`
```

*No mention of PredictionModelBuilder integration!*

### What User Stories MUST Say ✅

```markdown
### PredictionModelBuilder Integration (REQUIRED)

- [ ] **Add Private Field:** Add `private IKFoldSplitter<T>? _kfoldSplitter;` to `PredictionModelBuilder.cs`

- [ ] **Add Configure Method:** Add to `PredictionModelBuilder.cs`:
  ```csharp
  public IPredictionModelBuilder<T, TInput, TOutput> ConfigureKFoldSplitter(
      IKFoldSplitter<T> splitter)
  {
      _kfoldSplitter = splitter;
      return this;
  }
  ```
  **CRITICAL:** Configure methods MUST:
  - Take ONLY the interface (no additional parameters)
  - Store in private field (no processing)
  - Return `this` for method chaining

- [ ] **Use in Build():** Modify `Build()` method to actually USE the splitter:
  ```csharp
  var splitter = _kfoldSplitter ?? new StandardKFoldSplitter<T>(nSplits: 5);
  var splits = splitter.Split(XTrain, yTrain);
  // ... use splits for cross-validation
  ```

- [ ] **Verify Usage:** Feature must be ACTUALLY USED in execution flow, not just configured
```

### Anti-Pattern: ConfigureRetrievalAugmentedGeneration ⚠️

**Exception Found in Codebase (needs fixing):**

```csharp
// PredictionModelBuilder.cs:451 - VIOLATES pattern!
public IPredictionModelBuilder<T, TInput, TOutput> ConfigureRetrievalAugmentedGeneration(
    IRetriever<T>? retriever = null,      // ❌ Multiple parameters
    IReranker<T>? reranker = null,        // ❌ With defaults
    IGenerator<T>? generator = null,      // ❌ Wrong pattern
    IEnumerable<IQueryProcessor>? queryProcessors = null)
{
    // ... implementation
}
```

**Correct Pattern:**
```csharp
// Should be separate Configure methods:
public IPredictionModelBuilder<T, TInput, TOutput> ConfigureRetriever(IRetriever<T> retriever)
{
    _ragRetriever = retriever;
    return this;
}

public IPredictionModelBuilder<T, TInput, TOutput> ConfigureReranker(IReranker<T> reranker)
{
    _ragReranker = reranker;
    return this;
}

public IPredictionModelBuilder<T, TInput, TOutput> ConfigureGenerator(IGenerator<T> generator)
{
    _ragGenerator = generator;
    return this;
}
```

---

## Gap 4: Beginner-Friendly Defaults

### The Problem

User stories specify constructors but don't require **industry-standard default values** based on research.

### Required Default Values (from PROJECT_RULES.md)

```markdown
**Meta-Learning (N-way K-shot):**
- nWay: `5` (5-way is standard in meta-learning literature)
- kShot: `5` (balanced between 1-shot difficulty and 10-shot ease)
- queryShots: `15` (3x kShot is common practice)

**Neural Networks:**
- Learning rate: `0.001` (Adam optimizer standard)
- Batch size: `32` (good balance for most tasks)
- Epochs: `100` (sufficient for initial training)

**Regularization:**
- L1/L2 lambda: `0.01` (mild regularization)
- Dropout rate: `0.2` (20% is common)

**Data Splitting:**
- Train/val/test: `70/15/15` (standard split)

**Normalization:**
- Default: StandardScaler (zero mean, unit variance)
```

### Implementation Methods

**Method 1: Constructor Parameters with Defaults** (Preferred for 3-7 parameters)

```csharp
public UniformEpisodicDataLoader(
    Matrix<T> datasetX,
    Matrix<T> datasetY,
    int nWay = 5,        // Default: 5-way classification
    int kShot = 5,       // Default: 5-shot learning
    int queryShots = 15, // Default: 15 query samples
    int? seed = null)    // Default: random seed
{
    // ... implementation
}
```

**Method 2: Options Classes** (Preferred for 8+ parameters)

```csharp
public class AdamOptimizerOptions
{
    public double LearningRate { get; set; } = 0.001;  // Default from research
    public double Beta1 { get; set; } = 0.9;           // Adam paper standard
    public double Beta2 { get; set; } = 0.999;         // Adam paper standard
    public double Epsilon { get; set; } = 1e-8;        // Numerical stability
    public int MaxIterations { get; set; } = 100;      // Sufficient for initial training
}

public AdamOptimizer(AdamOptimizerOptions? options = null)
{
    var opts = options ?? new AdamOptimizerOptions();
    _learningRate = opts.LearningRate;
    _beta1 = opts.Beta1;
    // ... use other options
}
```

### What User Stories Currently Say ❌

**Example from Issue #332 (Dropout):**
```markdown
- [ ] **Constructor:** Takes `double dropoutRate`
```

*No default specified!*

### What User Stories MUST Say ✅

```markdown
- [ ] **Constructor:** Takes `double dropoutRate = 0.2` (20% is standard, cite: Srivastava et al. 2014)
- [ ] **Validation:** Throw `ArgumentException` if `dropoutRate < 0 || dropoutRate > 1`
- [ ] **Documentation:** XML doc must explain default choice and typical ranges
```

### Beginner Usage Example

```csharp
// Beginner - uses all defaults:
var model = new SomeModel<double>();
var result = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(model)
    .Build(x, y);  // Everything else uses defaults

// Advanced - custom configuration:
var customLoader = new UniformEpisodicDataLoader<double>(
    x, y,
    nWay: 10,      // Override default
    kShot: 1);     // Override default

var result = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(model)
    .ConfigureEpisodicDataLoader(customLoader)  // Pass configured instance
    .Build(x, y);
```

---

## Gap 5: Property Initialization

### The Problem

User stories don't warn against `default!` operator or specify initialization patterns.

### Required Property Initialization Patterns

```csharp
// ✅ String properties
public string Name { get; set; } = string.Empty;
public string Description { get; set; } = string.Empty;

// ✅ Collection properties
public List<T> Items { get; set; } = new List<T>();
public Vector<T> Values { get; set; } = new Vector<T>(0);
public Matrix<T> Data { get; set; } = new Matrix<T>(0, 0);

// ✅ Numeric properties (when zero is appropriate)
public int Count { get; set; } = 0;
public double LearningRate { get; set; } = 0.001;

// ✅ Nullable properties (when null is semantically correct)
public IModel<T>? BaseModel { get; set; } = null;
public int? OptionalValue { get; set; } = null;

// ✅ Generic numeric properties
public T MinValue { get; set; } = NumOps.Zero;  // Uses INumericOperations

// ❌ NEVER do this:
public string Name { get; set; } = default!;  // Violates .NET Framework compatibility
public Vector<T> Values { get; set; } = default!;  // Hides real issues
public T SomeValue { get; set; } = default!;  // Defeats nullable reference types
```

### Codebase Status ✅

**Excellent:** 0 occurrences of `default!` operator found in entire codebase.

### What User Stories MUST Say

```markdown
### Property Initialization Requirements

- [ ] **NO `default!` Operator:** Never use `default!` (not compatible with .NET Framework 4.6.2)
- [ ] **String Properties:** Initialize with `= string.Empty;`
- [ ] **Collection Properties:** Initialize with `= new List<T>();` or appropriate empty collection
- [ ] **Numeric Properties:** Initialize with appropriate default (0, 0.001, etc.) or `NumOps.Zero`
- [ ] **Optional Properties:** Use nullable types (`T?`) and initialize with `= null;` only if null is semantically correct
```

---

## Gap 6: Class Organization

### The Problem

User stories specify file names but not folder structure or organization rules.

### Required Organization Rules

#### Rule 1: One Class Per File

```
src/Regularization/
  ├── DropoutRegularization.cs     ← One class
  ├── L1Regularization.cs          ← One class
  └── L2Regularization.cs          ← One class

❌ DON'T create src/Regularization/AllRegularizers.cs with multiple classes
```

#### Rule 2: Interfaces in Root Interfaces Folder

```
src/Interfaces/
  ├── IActivationFunction.cs       ← ALL interfaces here
  ├── IRegularization.cs
  ├── IOptimizer.cs
  └── INormalizer.cs

❌ DON'T create src/Regularization/Interfaces/IRegularization.cs
❌ DON'T create src/ActivationFunctions/Interfaces/IActivationFunction.cs
```

#### Rule 3: Namespace Mirrors Folder Structure

```csharp
// File: src/Regularization/DropoutRegularization.cs
namespace AiDotNet.Regularization;  // ✅ Matches folder

// File: src/Interpretability/BiasDetection/DisparateImpactDetector.cs
namespace AiDotNet.Interpretability.BiasDetection;  // ✅ Matches folder path
```

#### Rule 4: NO "Base" Folders

```
✅ src/Regularization/RegularizationBase.cs
❌ src/Regularization/Base/RegularizationBase.cs
```

### What User Stories MUST Say

```markdown
### Class Organization (REQUIRED)

- [ ] **Interface Location:** Create `I{FeatureName}.cs` in `src/Interfaces/` (root level, NOT in subfolders)
- [ ] **Base Class Location:** Create `{FeatureName}Base.cs` in `src/{FeatureArea}/`
- [ ] **Concrete Class Location:** Create `{ImplementationName}.cs` in `src/{FeatureArea}/`
- [ ] **One Class Per File:** Each class, enum, and interface in separate file
- [ ] **Namespace Matching:** Namespace must mirror folder structure (e.g., `src/Regularization/` → `namespace AiDotNet.Regularization`)
```

---

## Gap 7: Documentation Standards

### The Problem

User stories mention "unit tests" but don't specify documentation requirements.

### Required Documentation Pattern

From `IActivationFunction.cs` (exemplar):

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines an interface for activation functions used in neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> An activation function is like a decision-maker in a neural network.
///
/// Imagine each neuron receives a number as input. The activation function decides
/// how strongly that neuron should "fire" based on that input.
///
/// Common activation functions include Sigmoid, ReLU, and Tanh.
///
/// This interface defines the standard methods that all activation functions must implement.
/// </remarks>
public interface IActivationFunction<T>
{
    /// <summary>
    /// Applies the activation function to the input value.
    /// </summary>
    /// <param name="input">The input value to the activation function.</param>
    /// <returns>The activated output value.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method takes a number and transforms it
    /// according to the activation function's rule.
    ///
    /// For example, with ReLU: negative inputs become 0, positive inputs stay the same.
    /// </remarks>
    T Activate(T input);
}
```

### Required Elements

1. **Class/Interface Summary:** What it is, what it does
2. **Type Parameter Documentation:** `<typeparam name="T">` explanations
3. **Beginner-Friendly Remarks:** `<b>For Beginners:</b>` sections with:
   - Analogies for complex concepts
   - Real-world examples
   - Common use cases
4. **Method Documentation:**
   - Summary
   - All `<param>` tags
   - `<returns>` tag
   - `<exception>` tags for thrown exceptions
   - Beginner remarks where appropriate
5. **Default Value Documentation:** Explain WHY default values were chosen (cite research if applicable)

### What User Stories MUST Say

```markdown
### Documentation Requirements

- [ ] **XML Documentation:** All public members (classes, interfaces, methods, properties) must have XML docs
- [ ] **Type Parameters:** Document all generic type parameters (e.g., `<typeparam name="T">`)
- [ ] **Beginner-Friendly Remarks:** Include `<b>For Beginners:</b>` sections with:
  - Simple analogies for complex concepts
  - Real-world examples
  - Common use cases
- [ ] **Parameter Documentation:** Document all parameters with `<param>` tags
- [ ] **Return Value Documentation:** Document return values with `<returns>` tags
- [ ] **Exception Documentation:** Document thrown exceptions with `<exception>` tags
- [ ] **Default Value Justification:** Explain WHY default values were chosen (cite research/standards)
- [ ] **Follow Existing Format:** Match documentation style of similar components in codebase
```

---

## Gap 8: Testing Requirements

### The Problem

User stories mention "unit tests" but don't specify coverage requirements or testing patterns.

### Required Testing Standards

From PROJECT_RULES.md:

> **ALWAYS** ensure new code includes unit tests with a minimum of **80% code coverage**.

### Testing Checklist

```markdown
### Testing Requirements (REQUIRED)

- [ ] **Minimum 80% Code Coverage:** All new code must achieve 80%+ coverage

- [ ] **Test Files:** Create `tests/UnitTests/{FeatureArea}/{ClassName}Tests.cs`
  - Example: `tests/UnitTests/Regularization/DropoutRegularizationTests.cs`

- [ ] **Generic Type Testing:** Test with multiple numeric types
  ```csharp
  [Theory]
  [InlineData(typeof(double))]
  [InlineData(typeof(float))]
  public void Activate_ShouldWork_ForDifferentNumericTypes(Type numericType)
  {
      // Test that INumericOperations<T> works correctly
  }
  ```

- [ ] **Default Value Testing:** Verify default values are applied correctly
  ```csharp
  [Fact]
  public void Constructor_WithNoParameters_ShouldUseStandardDefaults()
  {
      var dropout = new DropoutRegularization<double>();
      Assert.Equal(0.2, dropout.DropoutRate);  // Verify default
  }
  ```

- [ ] **Edge Case Testing:** Test boundary conditions
  - Null inputs (where applicable)
  - Empty collections
  - Zero values
  - Maximum values

- [ ] **Exception Testing:** Verify proper exceptions thrown for invalid inputs
  ```csharp
  [Fact]
  public void Constructor_WithInvalidDropoutRate_ShouldThrowArgumentException()
  {
      Assert.Throws<ArgumentException>(() => new DropoutRegularization<double>(1.5));
  }
  ```

- [ ] **Integration Testing:** If feature integrates with PredictionModelBuilder, test:
  - Configure method works
  - Feature is used in Build() process
  - Defaults are applied when not configured
```

---

## Complete Enhanced User Story Template

Use this template for ALL new user stories:

````markdown
### User Story

> As a [role], I want [feature], so that [benefit].

---

### Problem Statement

[Describe what's missing and why it's needed]

---

### Phase N: [Feature Name]

**Goal:** [What this phase accomplishes]

#### AC N.1: Create Interface and Base Class (X points)

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

#### AC N.2: Create Concrete Implementation (X points)

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

#### AC N.3: PredictionModelBuilder Integration (REQUIRED)

**Requirement:** Integrate feature with existing builder pipeline.

- [ ] **Add Private Field:** Add to `PredictionModelBuilder.cs`:
  ```csharp
  private I{FeatureName}<T, TInput, TOutput>? _{featureName};
  ```

- [ ] **Add Configure Method:** Add to `PredictionModelBuilder.cs`:
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
  public IPredictionModelBuilder<T, TInput, TOutput> Configure{FeatureName}(
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

- [ ] **Use in Build():** Modify `Build()` method in `PredictionModelBuilder.cs`:
  ```csharp
  // Create default if not configured
  var {featureName} = _{featureName} ?? new Default{FeatureName}<T, TInput, TOutput>();

  // Actually USE the feature in the pipeline
  var result = {featureName}.Process(...);
  // ... integrate result into training/prediction flow
  ```

- [ ] **Verify Actual Usage:** Feature must be ACTUALLY USED in execution flow, not just configured

#### AC N.4: Documentation (REQUIRED)

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

- [ ] **Follow Existing Format:** Match documentation style of similar components (e.g., ActivationFunctionBase, PredictionModelBuilder)

#### AC N.5: Unit Tests (REQUIRED)

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

- [ ] **Integration Testing** (if feature integrates with PredictionModelBuilder):
  ```csharp
  [Fact]
  public void Configure{FeatureName}_ShouldBeUsedInBuild()
  {
      var feature = new {ImplementationName}<double>();
      var builder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>()
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
- [ ] Integrated with PredictionModelBuilder (Configure method + usage in Build())
- [ ] All defaults based on research/industry standards and documented
- [ ] Complete XML documentation with beginner-friendly remarks
- [ ] All unit tests pass with 80%+ code coverage
- [ ] Code follows SOLID and DRY principles
- [ ] No duplicate functionality (checked against existing helpers)
````

---

## Recommended Next Steps

1. **Create `.github/ISSUE_TEMPLATE/feature_user_story.md`** with this enhanced template
2. **Update all open issues** to include architectural requirements
3. **Add pre-merge checklist** to PR template verifying these requirements
4. **Create GitHub Actions workflow** to verify:
   - No `default!` operator in new code
   - All new interfaces in `src/Interfaces/`
   - Test coverage meets 80% threshold
5. **Document exceptions** where architectural rules can't be followed (with justification)

---

## References

- **PROJECT_RULES.md:** `.github/PROJECT_RULES.md`
- **Example Interface:** `src/Interfaces/IActivationFunction.cs`
- **Example Base Class:** `src/ActivationFunctions/ActivationFunctionBase.cs`
- **Example Concrete:** `src/ActivationFunctions/ReLUActivation.cs`
- **Example Builder Integration:** `src/PredictionModelBuilder.cs`
- **Example Defaults:** `src/Data/Loaders/UniformEpisodicDataLoader.cs` (constructor parameters)
- **Example Options:** Search codebase for `Options` classes

---

**Version:** 1.0
**Last Review:** 2025-01-06
**Next Review:** After first 5 issues updated with this template
