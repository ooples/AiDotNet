# Issue #389: Junior Developer Implementation Guide - Feature Engineering Pipelines

## Understanding Feature Engineering Pipelines

### What is a Pipeline?
A pipeline chains multiple preprocessing/transformation steps together, ensuring they execute in the correct order with consistent data flow. Pipelines solve three major problems:

1. **Reproducibility**: Same transformations applied consistently to train and test data
2. **Modularity**: Each step is independent and testable
3. **Convenience**: Fit/transform all steps with single commands

### Key Concepts

**Pipeline Pattern**:
```csharp
// Pipeline: Scaler → Feature Selection → Model
var pipeline = new Pipeline<double, Matrix<double>, Vector<double>>()
    .AddStep(new StandardScaler())
    .AddStep(new FeatureSelector(method: "variance"))
    .AddStep(new LogisticRegression());

// Fit entire pipeline
await pipeline.FitAsync(trainX, trainY);

// Transform new data through entire pipeline
var predictions = await pipeline.TransformAsync(testX);
```

**IPipelineStep Interface** (Already exists!):
- `FitAsync`: Learn from data
- `TransformAsync`: Apply transformations
- `FitTransformAsync`: Convenience method
- `GetParameters/SetParameters`: Persistence
- `ValidateInput`: Safety checks

---

## Phase 1: Core Pipeline Infrastructure

### AC 1.1: Implement Pipeline Class

**File**: `src/Preprocessing/Pipeline.cs`

```csharp
namespace AiDotNet.Preprocessing;

/// <summary>
/// Chains multiple preprocessing and modeling steps together into a unified workflow.
/// </summary>
/// <remarks>
/// <para>
/// A Pipeline allows you to:
/// - Chain multiple transformation steps (scalers, selectors, etc.)
/// - Fit all steps sequentially on training data
/// - Transform new data through all steps automatically
/// - Save/load entire pipeline configurations
/// - Ensure consistent preprocessing between training and prediction
/// </para>
/// <para><b>For Beginners:</b> Think of a pipeline like an assembly line.
///
/// Just like a car factory has stages (frame assembly → painting → engine installation),
/// a machine learning pipeline has stages (scaling → feature selection → model training).
///
/// Each stage processes the output from the previous stage. The pipeline ensures:
/// - Stages run in the correct order
/// - Each stage uses the right settings (learned during training)
/// - New data goes through the exact same process as training data
///
/// Example:
/// 1. StandardScaler: Normalizes features
/// 2. FeatureSelector: Picks best features
/// 3. LogisticRegression: Makes predictions
///
/// Without a pipeline, you'd have to manually track and apply each step. The pipeline
/// automates this and prevents mistakes like forgetting a step or using wrong parameters.
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric type for calculations.</typeparam>
/// <typeparam name="TInput">Input data type.</typeparam>
/// <typeparam name="TOutput">Output data type.</typeparam>
public class Pipeline<T, TInput, TOutput>
{
    private readonly List<IPipelineStep<T, TInput, TOutput>> _steps;
    private readonly INumericOperations<T> _numOps;
    private bool _isFitted;

    /// <summary>
    /// Gets whether the pipeline has been fitted.
    /// </summary>
    public bool IsFitted => _isFitted;

    /// <summary>
    /// Gets the number of steps in the pipeline.
    /// </summary>
    public int StepCount => _steps.Count;

    /// <summary>
    /// Initializes a new empty pipeline.
    /// </summary>
    public Pipeline()
    {
        _steps = new List<IPipelineStep<T, TInput, TOutput>>();
        _numOps = NumericOperations<T>.Instance;
        _isFitted = false;
    }

    /// <summary>
    /// Adds a step to the end of the pipeline.
    /// </summary>
    /// <param name="step">The pipeline step to add.</param>
    /// <returns>The pipeline (for method chaining).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This adds another stage to your assembly line.
    ///
    /// You can chain multiple AddStep calls:
    /// ```csharp
    /// var pipeline = new Pipeline()
    ///     .AddStep(scaler)
    ///     .AddStep(selector)
    ///     .AddStep(model);
    /// ```
    ///
    /// Steps execute in the order added. The output of step 1 becomes input to step 2, etc.
    /// </para>
    /// </remarks>
    public Pipeline<T, TInput, TOutput> AddStep(IPipelineStep<T, TInput, TOutput> step)
    {
        if (step == null)
            throw new ArgumentNullException(nameof(step));

        _steps.Add(step);
        _isFitted = false; // Adding a step invalidates fitting
        return this;
    }

    /// <summary>
    /// Removes a step at the specified index.
    /// </summary>
    /// <param name="index">Index of step to remove.</param>
    /// <returns>The pipeline (for method chaining).</returns>
    public Pipeline<T, TInput, TOutput> RemoveStep(int index)
    {
        if (index < 0 || index >= _steps.Count)
            throw new ArgumentOutOfRangeException(nameof(index));

        _steps.RemoveAt(index);
        _isFitted = false;
        return this;
    }

    /// <summary>
    /// Gets the step at the specified index.
    /// </summary>
    public IPipelineStep<T, TInput, TOutput> GetStep(int index)
    {
        if (index < 0 || index >= _steps.Count)
            throw new ArgumentOutOfRangeException(nameof(index));

        return _steps[index];
    }

    /// <summary>
    /// Fits the entire pipeline on training data.
    /// </summary>
    /// <param name="inputs">Training input data.</param>
    /// <param name="targets">Training target data (optional).</param>
    /// <returns>Task representing the async operation.</returns>
    /// <remarks>
    /// <para>
    /// Fitting the pipeline:
    /// 1. Step 1 fits on original inputs, transforms inputs
    /// 2. Step 2 fits on transformed inputs from step 1, transforms further
    /// 3. Step 3 fits on transformed inputs from step 2, etc.
    ///
    /// Each step learns from the output of the previous step.
    /// </para>
    /// <para><b>For Beginners:</b> This trains each stage of your pipeline.
    ///
    /// Think of it like training factory workers at each station:
    /// - Worker 1 (scaler) learns the min/max of raw materials
    /// - Worker 2 (selector) learns which features are important in scaled materials
    /// - Worker 3 (model) learns patterns in the selected features
    ///
    /// Each worker learns from what they see (output of previous worker).
    /// </para>
    /// </remarks>
    public async Task FitAsync(TInput inputs, TOutput? targets = default)
    {
        if (_steps.Count == 0)
            throw new InvalidOperationException("Pipeline has no steps. Add steps before fitting.");

        TInput currentInput = inputs;
        TOutput? currentTargets = targets;

        for (int i = 0; i < _steps.Count; i++)
        {
            var step = _steps[i];

            // Validate input before fitting
            if (!step.ValidateInput(currentInput))
            {
                throw new InvalidOperationException(
                    $"Step {i} ({step.GetType().Name}) validation failed. " +
                    $"Input data is not compatible with this step.");
            }

            // Fit and transform this step
            await step.FitAsync(currentInput, currentTargets);
            currentInput = await step.TransformAsync(currentInput);

            // For intermediate steps, targets remain unchanged
            // Final step may use targets for supervised learning
        }

        _isFitted = true;
    }

    /// <summary>
    /// Transforms input data through all fitted pipeline steps.
    /// </summary>
    /// <param name="inputs">Input data to transform.</param>
    /// <returns>Transformed output data.</returns>
    /// <remarks>
    /// <para>
    /// Must call FitAsync before TransformAsync.
    /// Applies each step's transform sequentially using the statistics learned during fitting.
    /// </para>
    /// <para><b>For Beginners:</b> This runs new data through your trained pipeline.
    ///
    /// Like a product going through the assembly line:
    /// - Raw materials enter station 1 (scaler uses learned min/max)
    /// - Scaled materials go to station 2 (selector uses learned feature list)
    /// - Selected features go to station 3 (model uses learned weights)
    /// - Final prediction comes out
    ///
    /// Each station uses what it learned during training (FitAsync).
    /// </para>
    /// </remarks>
    public async Task<TOutput> TransformAsync(TInput inputs)
    {
        if (!_isFitted)
            throw new InvalidOperationException("Pipeline must be fitted before transforming. Call FitAsync first.");

        if (_steps.Count == 0)
            throw new InvalidOperationException("Pipeline has no steps.");

        TInput currentInput = inputs;

        for (int i = 0; i < _steps.Count; i++)
        {
            var step = _steps[i];

            // Validate input
            if (!step.ValidateInput(currentInput))
            {
                throw new InvalidOperationException(
                    $"Step {i} ({step.GetType().Name}) validation failed during transform.");
            }

            // Transform through this step
            var output = await step.TransformAsync(currentInput);

            // If this is the last step, return the output
            if (i == _steps.Count - 1)
            {
                return output;
            }

            // Otherwise, use output as input for next step
            // This requires TOutput to be convertible to TInput
            // For most pipelines, TInput and TOutput are the same type
            if (output is TInput nextInput)
            {
                currentInput = nextInput;
            }
            else
            {
                throw new InvalidOperationException(
                    $"Step {i} output type {typeof(TOutput)} is not compatible " +
                    $"with next step input type {typeof(TInput)}.");
            }
        }

        throw new InvalidOperationException("Pipeline transform completed without returning output.");
    }

    /// <summary>
    /// Fits and transforms in one operation.
    /// </summary>
    /// <param name="inputs">Training data.</param>
    /// <param name="targets">Target data (optional).</param>
    /// <returns>Transformed output data.</returns>
    public async Task<TOutput> FitTransformAsync(TInput inputs, TOutput? targets = default)
    {
        await FitAsync(inputs, targets);
        return await TransformAsync(inputs);
    }

    /// <summary>
    /// Gets parameters for all steps in the pipeline.
    /// </summary>
    /// <returns>List of parameter dictionaries, one per step.</returns>
    public List<Dictionary<string, object>> GetAllParameters()
    {
        return _steps.Select(step => step.GetParameters()).ToList();
    }

    /// <summary>
    /// Sets parameters for all steps in the pipeline.
    /// </summary>
    /// <param name="allParameters">List of parameter dictionaries, one per step.</param>
    public void SetAllParameters(List<Dictionary<string, object>> allParameters)
    {
        if (allParameters.Count != _steps.Count)
        {
            throw new ArgumentException(
                $"Parameter count ({allParameters.Count}) doesn't match step count ({_steps.Count}).");
        }

        for (int i = 0; i < _steps.Count; i++)
        {
            _steps[i].SetParameters(allParameters[i]);
        }

        _isFitted = true; // Restored from saved parameters
    }

    /// <summary>
    /// Gets metadata for all steps.
    /// </summary>
    public List<Dictionary<string, string>> GetAllMetadata()
    {
        return _steps.Select(step => step.GetMetadata()).ToList();
    }

    /// <summary>
    /// Clears all steps from the pipeline.
    /// </summary>
    public void Clear()
    {
        _steps.Clear();
        _isFitted = false;
    }

    /// <summary>
    /// Creates a clone of this pipeline with the same steps.
    /// </summary>
    /// <remarks>
    /// The clone has the same steps but is not fitted.
    /// You must call FitAsync on the clone before using it.
    /// </remarks>
    public Pipeline<T, TInput, TOutput> Clone()
    {
        var clone = new Pipeline<T, TInput, TOutput>();

        // Clone each step (requires steps to be cloneable)
        foreach (var step in _steps)
        {
            // For now, just add the same step reference
            // TODO: Implement proper deep cloning if steps support ICloneable
            clone.AddStep(step);
        }

        return clone;
    }
}
```

### AC 1.2: Create PipelineStep Base Class

**File**: `src/Preprocessing/PipelineStepBase.cs`

```csharp
namespace AiDotNet.Preprocessing;

/// <summary>
/// Base class for pipeline steps with common functionality.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This provides common functionality for all pipeline steps.
///
/// When creating a new pipeline step (like a scaler or selector), inherit from this base class.
/// It handles:
/// - Parameter storage and retrieval
/// - Basic validation
/// - Metadata management
///
/// You only need to implement the specific fit/transform logic for your step.
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric type.</typeparam>
/// <typeparam name="TInput">Input type.</typeparam>
/// <typeparam name="TOutput">Output type.</typeparam>
public abstract class PipelineStepBase<T, TInput, TOutput> : IPipelineStep<T, TInput, TOutput>
{
    protected readonly INumericOperations<T> NumOps;
    protected Dictionary<string, object> Parameters;
    protected bool IsFitted;

    /// <summary>
    /// Initializes the base pipeline step.
    /// </summary>
    protected PipelineStepBase()
    {
        NumOps = NumericOperations<T>.Instance;
        Parameters = new Dictionary<string, object>();
        IsFitted = false;
    }

    /// <summary>
    /// Fits this step on training data.
    /// </summary>
    public abstract Task FitAsync(TInput inputs, TOutput? targets = default);

    /// <summary>
    /// Transforms input data using fitted parameters.
    /// </summary>
    public abstract Task<TOutput> TransformAsync(TInput inputs);

    /// <summary>
    /// Fits and transforms in one operation.
    /// </summary>
    public virtual async Task<TOutput> FitTransformAsync(TInput inputs, TOutput? targets = default)
    {
        await FitAsync(inputs, targets);
        return await TransformAsync(inputs);
    }

    /// <summary>
    /// Gets the parameters of this step.
    /// </summary>
    public virtual Dictionary<string, object> GetParameters()
    {
        return new Dictionary<string, object>(Parameters);
    }

    /// <summary>
    /// Sets the parameters of this step.
    /// </summary>
    public virtual void SetParameters(Dictionary<string, object> parameters)
    {
        Parameters = new Dictionary<string, object>(parameters);
        IsFitted = true;
    }

    /// <summary>
    /// Validates input data.
    /// </summary>
    public virtual bool ValidateInput(TInput inputs)
    {
        if (inputs == null)
            return false;

        // Type-specific validation
        if (inputs is Matrix<T> matrix)
        {
            return matrix.Rows > 0 && matrix.Columns > 0;
        }
        else if (inputs is Tensor<T> tensor)
        {
            return tensor.Shape.All(dim => dim > 0);
        }
        else if (inputs is Vector<T> vector)
        {
            return vector.Length > 0;
        }

        return true;
    }

    /// <summary>
    /// Gets metadata about this step.
    /// </summary>
    public virtual Dictionary<string, string> GetMetadata()
    {
        return new Dictionary<string, string>
        {
            { "Type", GetType().Name },
            { "IsFitted", IsFitted.ToString() },
            { "ParameterCount", Parameters.Count.ToString() }
        };
    }

    /// <summary>
    /// Ensures the step has been fitted before transformation.
    /// </summary>
    protected void EnsureFitted()
    {
        if (!IsFitted)
        {
            throw new InvalidOperationException(
                $"{GetType().Name} must be fitted before transformation. Call FitAsync first.");
        }
    }
}
```

---

## Phase 2: Scaler Pipeline Steps

### AC 2.1: StandardScalerStep Wrapper

**File**: `src/Preprocessing/StandardScalerStep.cs`

```csharp
namespace AiDotNet.Preprocessing;

/// <summary>
/// Pipeline step wrapper for StandardScaler.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This wraps StandardScaler so it can be used in pipelines.
///
/// StandardScaler normalizes data to mean=0, stddev=1.
/// This wrapper makes it compatible with the pipeline interface.
/// </para>
/// </remarks>
public class StandardScalerStep<T> : PipelineStepBase<T, Matrix<T>, Matrix<T>>
{
    private readonly StandardScaler<T, Matrix<T>, Vector<T>> _scaler;
    private List<NormalizationParameters<T>>? _fittedParameters;

    public StandardScalerStep()
    {
        _scaler = new StandardScaler<T, Matrix<T>, Vector<T>>();
    }

    public override async Task FitAsync(Matrix<T> inputs, Matrix<T>? targets = default)
    {
        _fittedParameters = _scaler.Fit(inputs);
        IsFitted = true;

        // Store parameters for serialization
        Parameters["Mean"] = _fittedParameters.Select(p => p.Mean).ToList();
        Parameters["StdDev"] = _fittedParameters.Select(p => p.StdDev).ToList();

        await Task.CompletedTask;
    }

    public override async Task<Matrix<T>> TransformAsync(Matrix<T> inputs)
    {
        EnsureFitted();

        if (_fittedParameters == null)
            throw new InvalidOperationException("Scaler parameters not available.");

        var result = _scaler.Transform(inputs, _fittedParameters);
        return await Task.FromResult(result);
    }

    public override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ScalerType"] = "StandardScaler";
        metadata["FeatureCount"] = _fittedParameters?.Count.ToString() ?? "0";
        return metadata;
    }
}
```

### AC 2.2: MinMaxScalerStep Wrapper

Similar implementation for MinMaxScaler.

---

## Phase 3: Feature Engineering Steps

### AC 3.1: PolynomialFeatures Step

**File**: `src/Preprocessing/PolynomialFeaturesStep.cs`

```csharp
namespace AiDotNet.Preprocessing;

/// <summary>
/// Generates polynomial and interaction features.
/// </summary>
/// <remarks>
/// <para>
/// Creates new features by computing polynomial combinations of input features.
/// For example, with degree=2 and features [x1, x2]:
/// - Original features: x1, x2
/// - Polynomial features: 1, x1, x2, x1², x1*x2, x2²
/// </para>
/// <para><b>For Beginners:</b> This creates new features by combining existing ones.
///
/// Think of it like creating new measurements from existing ones:
/// - Original features: height, weight
/// - Polynomial features: height², weight², height*weight
///
/// Why do this?
/// - Captures non-linear relationships (like BMI = weight/height²)
/// - Allows linear models to learn curved patterns
/// - Finds interactions between features
///
/// Example with degree=2:
/// - Input: [2, 3]
/// - Output: [1, 2, 3, 4, 6, 9]
/// - Meaning: [1, x1, x2, x1², x1*x2, x2²]
/// </para>
/// </remarks>
public class PolynomialFeaturesStep<T> : PipelineStepBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _degree;
    private readonly bool _includeBias;
    private readonly bool _interactionOnly;
    private int _inputFeatures;

    /// <summary>
    /// Initializes polynomial feature generator.
    /// </summary>
    /// <param name="degree">Maximum degree of polynomial features (default: 2).</param>
    /// <param name="includeBias">Include bias column (all 1s) (default: true).</param>
    /// <param name="interactionOnly">Only create interaction terms, not powers (default: false).</param>
    public PolynomialFeaturesStep(int degree = 2, bool includeBias = true, bool interactionOnly = false)
    {
        if (degree < 1)
            throw new ArgumentException("Degree must be at least 1.", nameof(degree));

        _degree = degree;
        _includeBias = includeBias;
        _interactionOnly = interactionOnly;
    }

    public override async Task FitAsync(Matrix<T> inputs, Matrix<T>? targets = default)
    {
        _inputFeatures = inputs.Columns;
        IsFitted = true;

        Parameters["Degree"] = _degree;
        Parameters["IncludeBias"] = _includeBias;
        Parameters["InteractionOnly"] = _interactionOnly;
        Parameters["InputFeatures"] = _inputFeatures;

        await Task.CompletedTask;
    }

    public override async Task<Matrix<T>> TransformAsync(Matrix<T> inputs)
    {
        EnsureFitted();

        if (inputs.Columns != _inputFeatures)
        {
            throw new ArgumentException(
                $"Expected {_inputFeatures} features, got {inputs.Columns}.");
        }

        var outputFeatures = CalculateOutputFeatureCount();
        var result = new Matrix<T>(inputs.Rows, outputFeatures);

        for (int row = 0; row < inputs.Rows; row++)
        {
            var inputRow = inputs.GetRow(row);
            var outputRow = GeneratePolynomialFeatures(inputRow);
            result.SetRow(row, outputRow);
        }

        return await Task.FromResult(result);
    }

    /// <summary>
    /// Generates polynomial features for a single sample.
    /// </summary>
    private Vector<T> GeneratePolynomialFeatures(Vector<T> input)
    {
        var features = new List<T>();

        // Add bias term if requested
        if (_includeBias)
        {
            features.Add(NumOps.One);
        }

        // Add original features
        for (int i = 0; i < input.Length; i++)
        {
            features.Add(input[i]);
        }

        // Add higher degree terms
        if (_degree >= 2)
        {
            for (int d = 2; d <= _degree; d++)
            {
                GenerateDegreeCombinations(input, d, features);
            }
        }

        return Vector<T>.FromEnumerable(features);
    }

    /// <summary>
    /// Generates all combinations of features for a given degree.
    /// </summary>
    private void GenerateDegreeCombinations(Vector<T> input, int degree, List<T> features)
    {
        // Generate all combinations of 'degree' features
        var indices = new int[degree];
        GenerateCombinationsRecursive(input, indices, 0, 0, degree, features);
    }

    /// <summary>
    /// Recursively generates feature combinations.
    /// </summary>
    private void GenerateCombinationsRecursive(
        Vector<T> input,
        int[] indices,
        int start,
        int depth,
        int maxDepth,
        List<T> features)
    {
        if (depth == maxDepth)
        {
            // Calculate product of selected features
            T product = NumOps.One;
            bool isInteraction = indices.Distinct().Count() > 1;

            // Skip if interaction_only=true and this is a power term
            if (_interactionOnly && !isInteraction)
            {
                return;
            }

            for (int i = 0; i < indices.Length; i++)
            {
                product = NumOps.Multiply(product, input[indices[i]]);
            }

            features.Add(product);
            return;
        }

        // Generate combinations
        for (int i = start; i < input.Length; i++)
        {
            indices[depth] = i;
            GenerateCombinationsRecursive(input, indices, i, depth + 1, maxDepth, features);
        }
    }

    /// <summary>
    /// Calculates the number of output features.
    /// </summary>
    private int CalculateOutputFeatureCount()
    {
        int count = 0;

        if (_includeBias)
            count++;

        // Original features
        count += _inputFeatures;

        // Higher degree combinations
        for (int d = 2; d <= _degree; d++)
        {
            if (_interactionOnly)
            {
                // Only interaction terms: C(n, d)
                count += Combinations(_inputFeatures, d);
            }
            else
            {
                // All combinations with replacement: C(n+d-1, d)
                count += Combinations(_inputFeatures + d - 1, d);
            }
        }

        return count;
    }

    /// <summary>
    /// Calculates binomial coefficient C(n, k).
    /// </summary>
    private int Combinations(int n, int k)
    {
        if (k > n) return 0;
        if (k == 0 || k == n) return 1;

        // Use the formula C(n,k) = n! / (k! * (n-k)!)
        // Optimized to avoid overflow
        int result = 1;
        for (int i = 1; i <= k; i++)
        {
            result = result * (n - i + 1) / i;
        }
        return result;
    }

    public override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Degree"] = _degree.ToString();
        metadata["IncludeBias"] = _includeBias.ToString();
        metadata["InteractionOnly"] = _interactionOnly.ToString();
        metadata["InputFeatures"] = _inputFeatures.ToString();
        metadata["OutputFeatures"] = IsFitted ? CalculateOutputFeatureCount().ToString() : "Unknown";
        return metadata;
    }
}
```

---

## Phase 4: Complete Pipeline Example

### AC 4.1: End-to-End Pipeline Usage

```csharp
// File: examples/PipelineExample.cs

public class PipelineExample
{
    public async Task RunExample()
    {
        // Create sample data
        var trainX = new Matrix<double>(new[,] {
            { 1.0, 2.0 },
            { 2.0, 4.0 },
            { 3.0, 6.0 },
            { 4.0, 8.0 }
        });
        var trainY = new Matrix<double>(new[,] {
            { 5.0 },
            { 10.0 },
            { 15.0 },
            { 20.0 }
        });

        // Build pipeline
        var pipeline = new Pipeline<double, Matrix<double>, Matrix<double>>()
            .AddStep(new StandardScalerStep<double>())           // Step 1: Standardize
            .AddStep(new PolynomialFeaturesStep<double>(degree: 2))  // Step 2: Add polynomial features
            .AddStep(new MinMaxScalerStep<double>());            // Step 3: Scale to [0,1]

        // Fit pipeline on training data
        await pipeline.FitAsync(trainX);

        // Transform training data
        var trainTransformed = await pipeline.TransformAsync(trainX);

        // Transform test data (uses statistics from training)
        var testX = new Matrix<double>(new[,] {
            { 2.5, 5.0 },
            { 3.5, 7.0 }
        });
        var testTransformed = await pipeline.TransformAsync(testX);

        // Save pipeline parameters
        var parameters = pipeline.GetAllParameters();
        // ... serialize parameters to JSON/file

        // Load pipeline parameters later
        var newPipeline = new Pipeline<double, Matrix<double>, Matrix<double>>()
            .AddStep(new StandardScalerStep<double>())
            .AddStep(new PolynomialFeaturesStep<double>(degree: 2))
            .AddStep(new MinMaxScalerStep<double>());

        newPipeline.SetAllParameters(parameters);

        // Use loaded pipeline
        var predictions = await newPipeline.TransformAsync(testX);
    }
}
```

---

## Common Pitfalls to Avoid

1. **Wrong Step Order**: Order matters!
   ```csharp
   // WRONG - Polynomial after scaling makes tiny values
   .AddStep(new StandardScaler())
   .AddStep(new PolynomialFeatures(degree: 3))  // Creates tiny^3 values

   // CORRECT - Polynomial before final scaling
   .AddStep(new PolynomialFeatures(degree: 3))
   .AddStep(new StandardScaler())
   ```

2. **Forgetting to Fit**: Must fit before transform
   ```csharp
   // WRONG
   var result = await pipeline.TransformAsync(data);  // Error!

   // CORRECT
   await pipeline.FitAsync(trainData);
   var result = await pipeline.TransformAsync(testData);
   ```

3. **Type Mismatches**: Pipeline steps must have compatible types
   ```csharp
   // Step output type must match next step input type
   // Most steps: Matrix<T> → Matrix<T>
   ```

4. **Data Leakage**: Never fit on test data
   ```csharp
   // WRONG
   await pipeline.FitAsync(testData);

   // CORRECT
   await pipeline.FitAsync(trainData);
   await pipeline.TransformAsync(testData);
   ```

---

## Testing Strategy

### Unit Tests
```csharp
[Fact]
public async Task Pipeline_FitTransform_ExecutesStepsInOrder()
{
    // Test that steps execute sequentially
    // Test that each step receives correct input
}

[Fact]
public async Task Pipeline_SaveLoad_PreservesParameters()
{
    // Fit pipeline, save parameters
    // Create new pipeline, load parameters
    // Verify identical transforms
}

[Fact]
public async Task PolynomialFeatures_Degree2_GeneratesCorrectFeatures()
{
    // Input: [2, 3]
    // Expected: [1, 2, 3, 4, 6, 9]
}
```

### Integration Tests
```csharp
[Fact]
public async Task EndToEndPipeline_TrainTest_ProducesConsistentResults()
{
    // Build full pipeline
    // Fit on train, transform both train and test
    // Verify no data leakage
}
```

---

## Next Steps

1. Implement Pipeline class
2. Create PipelineStepBase
3. Implement StandardScalerStep, MinMaxScalerStep, RobustScalerStep
4. Implement PolynomialFeaturesStep
5. Write comprehensive tests
6. Add pipeline serialization (JSON)
7. Create integration examples

**Estimated Effort**: 4-5 days for a junior developer

**Files to Create**:
- `src/Preprocessing/Pipeline.cs`
- `src/Preprocessing/PipelineStepBase.cs`
- `src/Preprocessing/StandardScalerStep.cs`
- `src/Preprocessing/MinMaxScalerStep.cs`
- `src/Preprocessing/RobustScalerStep.cs`
- `src/Preprocessing/PolynomialFeaturesStep.cs`
- `tests/UnitTests/Preprocessing/PipelineTests.cs`
- `tests/UnitTests/Preprocessing/PolynomialFeaturesTests.cs`
