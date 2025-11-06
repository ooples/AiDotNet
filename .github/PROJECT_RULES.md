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
- New features must work within the existing model-building process
- Features must be **ACTUALLY USED** in the Build() or Predict() process, not just configured
- Add Configure methods AND integrate them into the execution flow

### 5. Beginner-Friendly Defaults (CRITICAL)
- **PRIMARY GOAL**: Make the library usable by beginners with minimal configuration
- **Beginners should ONLY need to provide**: Input data (X) and Output data (Y)
- **EVERYTHING ELSE must have sensible defaults** based on industry standards and research
- Follow the pattern in PredictionModelBuilder.Build():
  ```csharp
  var normalizer = _normalizer ?? new NoNormalizer<T, TInput, TOutput>();
  var optimizer = _optimizer ?? new NormalOptimizer<T, TInput, TOutput>(_model);
  ```

#### Default Value Guidelines:
- **Meta-Learning (N-way K-shot)**:
  - Default nWay: `5` (5-way is standard in meta-learning literature)
  - Default kShot: `5` (balanced between 1-shot difficulty and 10-shot ease)
  - Default queryShots: `15` (3x kShot is common practice)
  - Default loader type: `EpisodicDataLoader` (uniform random sampling)

- **Neural Networks**:
  - Default learning rate: `0.001` (Adam optimizer standard)
  - Default batch size: `32` (good balance for most tasks)
  - Default epochs: `100` (sufficient for initial training)

- **Regularization**:
  - Default L1/L2 lambda: `0.01` (mild regularization)
  - Default dropout rate: `0.2` (20% is common)

- **Data Splitting**:
  - Default train/val/test: `70/15/15` (standard split)

- **Feature Selection**:
  - Default: No feature selection (use all features)

- **Normalization**:
  - Default: StandardScaler (zero mean, unit variance)

#### Configure Method Pattern (CRITICAL):

**ALL Configure methods MUST follow this exact pattern:**
```csharp
// In PredictionModelBuilder.cs:
private ISomeInterface<T>? _someComponent;  // Private nullable field

public IPredictionModelBuilder<T, TInput, TOutput> ConfigureSomeComponent(ISomeInterface<T> component)
{
    _someComponent = component;  // Just store the interface
    return this;                 // Return for method chaining
}

// In Build() method - create default if needed:
var someComponent = _someComponent ?? new DefaultSomeComponent<T>();
```

**NEVER:**
- ❌ Add parameters to Configure methods (takes ONLY the interface)
- ❌ Create factory functions in Configure methods
- ❌ Store configuration values in PredictionModelBuilder fields
- ❌ Make computed properties public that expose private nullable fields

**WHY:** This pattern ensures:
- Dependency injection compatibility
- Consistent API across all components
- Clean separation of concerns
- Testability

#### How to Implement Defaults:

**Method 1: Constructor Parameters with Defaults (PREFERRED for simple cases)**
- Add default values directly to constructor parameters
- Best for classes with 3-7 parameters
- Example from UniformEpisodicDataLoader:
  ```csharp
  public UniformEpisodicDataLoader(
      Matrix<T> datasetX,
      Vector<T> datasetY,
      int nWay = 5,        // Default: 5-way
      int kShot = 5,       // Default: 5-shot
      int queryShots = 15, // Default: 15 queries
      int? seed = null)
  ```

**Method 2: Options Classes with Property Initializers (PREFERRED for complex cases)**
- Use options classes for components with many configuration parameters (8+)
- Set defaults using property initializers
- Example from AdamOptimizerOptions:
  ```csharp
  public class AdamOptimizerOptions
  {
      public double LearningRate { get; set; } = 0.001;
      public double Beta1 { get; set; } = 0.9;
      public double Beta2 { get; set; } = 0.999;
      public double Epsilon { get; set; } = 1e-8;
      // ... more properties with defaults
  }
  ```
- Then use in constructor:
  ```csharp
  public AdamOptimizer(AdamOptimizerOptions? options = null)
  {
      var opts = options ?? new AdamOptimizerOptions();
      _learningRate = opts.LearningRate;
      // ... use other options
  }
  ```

**General Guidelines:**
1. **Configure methods take ONLY interfaces** - no parameters, no options
2. **Default values belong in concrete class constructors** - NOT in PredictionModelBuilder
3. **Check for null in Build()** and create defaults if needed (for required components)
4. **Document defaults** in XML comments and beginner remarks
5. **Test with minimal code**: Ensure `builder.Build(x, y)` works with just data

#### Example Pattern:
```csharp
// Beginner-friendly: Use defaults by creating instances with default constructor parameters
var model = new SomeModel<double>();
var result = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(model)
    .Build(x, y);  // Everything else uses defaults

// Advanced: Create instances with custom parameters, pass to Configure methods
var customLoader = new UniformEpisodicDataLoader<double>(x, y, nWay: 10, kShot: 1);
var result = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(model)
    .ConfigureEpisodicDataLoader(customLoader)  // Pass configured instance
    .Build(x, y);
```

### 6. Code Quality
- Follow SOLID principles
- Follow DRY principles
- Use existing helpers (`StatisticsHelper`, `MathHelper`, etc.) - don't duplicate functionality
- Proper null checks with exceptions, NOT nullable operators (`!`) for .NET Framework compatibility
- **ALWAYS** ensure new code includes unit tests with a minimum of 80% code coverage.

### 7. Property Default Values
- All string properties: `= string.Empty;`
- All collection properties: `= new List<T>();` or appropriate empty collection
- **NEVER** leave properties without default values

### 8. Documentation
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

- Located in the `testconsole` directory
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
