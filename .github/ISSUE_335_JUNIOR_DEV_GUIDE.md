# Issue #335: Junior Developer Implementation Guide

## 🚨 CRITICAL BUGS FOUND: Infinite Loop Errors

**This issue is about FIXING CRITICAL BUGS, not implementing new features from scratch.**

### What's Broken:

1. **SHAP has infinite loop bug**: `VectorModel.GetShapValuesAsync()` → `InterpretableModelHelper.GetShapValuesAsync()` → `model.GetShapValuesAsync()` → **INFINITE LOOP**

2. **LIME has infinite loop bug**: Same circular reference pattern as SHAP

### What Exists But Is Hidden:

1. **SHAP Algorithm**: Working implementation in `src/FitDetectors/ShapleyValueFitDetector.cs` (lines 220-247)
2. **Permutation Feature Importance**: Working implementation in `src/FitDetectors/FeatureImportanceFitDetector.cs` (lines 280-330)
3. **LIME Data Structure**: `src/Interpretability/LimeExplanation.cs` (structure only, no algorithm)

### What's Needed:

1. Fix InterpretableModelHelper circular references (SHAP and LIME)
2. Extract SHAP algorithm from FitDetector into standalone explainer
3. Implement actual LIME algorithm
4. Extract Permutation Feature Importance from FitDetector

---

## Understanding the Circular Reference Bug

**File**: `src/Models/VectorModel.cs` (lines 1347-1358)

### The Broken Code:

```csharp
// VectorModel.cs - Line 1347
public virtual Task<Matrix<T>> GetShapValuesAsync(Tensor<T> inputs)
{
    return InterpretableModelHelper.GetShapValuesAsync(this, _enabledMethods, inputs);
}

// InterpretableModelHelper.cs - Line 62
public static Task<Matrix<T>> GetShapValuesAsync<T>(
    IInterpretableModel<T> model,
    HashSet<InterpretationMethod> enabledMethods,
    Tensor<T> inputs)
{
    if (!enabledMethods.Contains(InterpretationMethod.SHAP))
        throw new InvalidOperationException("SHAP method is not enabled.");

    return model.GetShapValuesAsync(inputs);  // CALLS BACK TO MODEL!
}
```

### The Problem:

```
User calls: model.GetShapValuesAsync(inputs)
    ↓
VectorModel.GetShapValuesAsync() calls Helper.GetShapValuesAsync(this, ...)
    ↓
Helper.GetShapValuesAsync() calls model.GetShapValuesAsync(inputs)  ← Back to start!
    ↓
INFINITE LOOP - Stack Overflow Exception
```

### Why This Happened:

The helper was supposed to **implement** the algorithm, but instead it **delegates back to the model**, creating a circular reference.

---

## Phase 1: Fix SHAP Circular Reference Bug

### Step 1: Understand the SHAP Algorithm

**SHAP (SHapley Additive exPlanations)** calculates feature importance using game theory.

**Algorithm** (from ShapleyValueFitDetector.cs lines 220-247):

```csharp
private Dictionary<string, T> CalculateShapleyValues(
    ModelEvaluationData<T, TInput, TOutput> evaluationData,
    List<string> features)
{
    var shapleyValues = new Dictionary<string, T>();
    var n = features.Count;

    foreach (var feature in features)
    {
        T shapleyValue = NumOps.Zero;

        // Monte Carlo sampling
        for (int i = 0; i < _options.MonteCarloSamples; i++)
        {
            // Random permutation of features
            var permutation = features.OrderBy(x => _random.Next()).ToList();
            var index = permutation.IndexOf(feature);

            // Coalition with and without this feature
            var withFeature = new HashSet<string>(permutation.Take(index + 1));
            var withoutFeature = new HashSet<string>(permutation.Take(index));

            // Marginal contribution = Performance(with) - Performance(without)
            var marginalContribution = NumOps.Subtract(
                CalculatePerformance(evaluationData, withFeature),
                CalculatePerformance(evaluationData, withoutFeature));

            shapleyValue = NumOps.Add(shapleyValue, marginalContribution);
        }

        // Average over all samples
        shapleyValues[feature] = NumOps.Divide(shapleyValue, NumOps.FromDouble(_options.MonteCarloSamples));
    }

    return shapleyValues;
}
```

**Key Concepts**:
1. **Permutation**: Random ordering of features
2. **Coalition**: Subset of features used for prediction
3. **Marginal Contribution**: How much performance changes when adding this feature
4. **Monte Carlo**: Average over many random permutations

### Step 2: Fix InterpretableModelHelper.GetShapValuesAsync()

**File**: `src/Interpretability/InterpretableModelHelper.cs`

**REPLACE lines 54-73** with actual implementation:

```csharp
/// <summary>
/// Gets SHAP values for the given inputs.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <param name="model">The model to analyze.</param>
/// <param name="enabledMethods">The set of enabled interpretation methods.</param>
/// <param name="inputs">The inputs to analyze (shape: [num_samples, num_features]).</param>
/// <param name="backgroundData">Optional background dataset for baseline (if null, uses zero baseline).</param>
/// <param name="numSamples">Number of Monte Carlo samples (default: 100).</param>
/// <returns>A matrix containing SHAP values (shape: [num_samples, num_features]).</returns>
/// <remarks>
/// <para><b>For Beginners:</b> SHAP values explain how much each feature contributed to a prediction.
///
/// Think of SHAP like giving credit to team members:
/// - Positive SHAP value = feature pushed prediction higher
/// - Negative SHAP value = feature pushed prediction lower
/// - Zero SHAP value = feature had no effect
///
/// The algorithm uses game theory to fairly distribute credit among features.
/// </para>
/// </remarks>
public static async Task<Matrix<T>> GetShapValuesAsync<T>(
    IInterpretableModel<T> model,
    HashSet<InterpretationMethod> enabledMethods,
    Tensor<T> inputs,
    Matrix<T>? backgroundData = null,
    int numSamples = 100)
{
    if (!enabledMethods.Contains(InterpretationMethod.SHAP))
    {
        throw new InvalidOperationException("SHAP method is not enabled.");
    }

    var numOps = MathHelper.GetNumericOperations<T>();

    // Convert tensor to matrix for easier processing
    var inputMatrix = ConvertTensorToMatrix(inputs);
    int numInstances = inputMatrix.Rows;
    int numFeatures = inputMatrix.Columns;

    // Use background data mean as baseline, or zeros if not provided
    var baseline = backgroundData != null
        ? CalculateColumnMeans(backgroundData)
        : new Vector<T>(numFeatures); // All zeros

    // Initialize SHAP values matrix
    var shapValues = new Matrix<T>(numInstances, numFeatures);

    // Calculate SHAP values for each instance
    for (int instanceIdx = 0; instanceIdx < numInstances; instanceIdx++)
    {
        var instance = inputMatrix.GetRow(instanceIdx);

        // For each feature
        for (int featureIdx = 0; featureIdx < numFeatures; featureIdx++)
        {
            T shapValue = numOps.Zero;

            // Monte Carlo sampling
            for (int sample = 0; sample < numSamples; sample++)
            {
                // Random permutation of feature indices
                var permutation = Enumerable.Range(0, numFeatures)
                    .OrderBy(x => Guid.NewGuid())
                    .ToList();

                int position = permutation.IndexOf(featureIdx);

                // Create coalition with and without this feature
                var withFeature = CreateCoalitionInstance(instance, baseline, permutation.Take(position + 1).ToList());
                var withoutFeature = CreateCoalitionInstance(instance, baseline, permutation.Take(position).ToList());

                // Predict with both coalitions
                var predWith = await PredictSingleAsync(model, withFeature);
                var predWithout = await PredictSingleAsync(model, withoutFeature);

                // Marginal contribution
                var contribution = numOps.Subtract(predWith, predWithout);
                shapValue = numOps.Add(shapValue, contribution);
            }

            // Average over all samples
            shapValues[instanceIdx, featureIdx] = numOps.Divide(shapValue, numOps.FromDouble(numSamples));
        }
    }

    return shapValues;
}

/// <summary>
/// Creates a coalition instance by using actual feature values for coalition members and baseline for others.
/// </summary>
private static Vector<T> CreateCoalitionInstance<T>(Vector<T> instance, Vector<T> baseline, List<int> coalitionIndices)
{
    var result = new Vector<T>(instance.Length);

    for (int i = 0; i < instance.Length; i++)
    {
        // Use actual value if feature is in coalition, otherwise use baseline
        result[i] = coalitionIndices.Contains(i) ? instance[i] : baseline[i];
    }

    return result;
}

/// <summary>
/// Makes a prediction for a single instance vector.
/// </summary>
private static async Task<T> PredictSingleAsync<T>(IInterpretableModel<T> model, Vector<T> instance)
{
    // Convert vector to matrix (1 row)
    var instanceMatrix = new Matrix<T>(1, instance.Length);
    for (int i = 0; i < instance.Length; i++)
    {
        instanceMatrix[0, i] = instance[i];
    }

    // Predict (model should accept Matrix<T> and return Vector<T>)
    var prediction = await Task.Run(() =>
    {
        if (model is IModel<T, Matrix<T>, Vector<T>> predictiveModel)
        {
            return predictiveModel.Predict(instanceMatrix);
        }
        throw new InvalidOperationException("Model does not support Matrix<T> predictions.");
    });

    return prediction[0]; // Return first (and only) prediction
}

/// <summary>
/// Converts Tensor<T> to Matrix<T> assuming 2D tensor [rows, cols].
/// </summary>
private static Matrix<T> ConvertTensorToMatrix<T>(Tensor<T> tensor)
{
    if (tensor.Dimensions.Length != 2)
        throw new ArgumentException("Tensor must be 2D for SHAP calculations.");

    int rows = tensor.Dimensions[0];
    int cols = tensor.Dimensions[1];
    var matrix = new Matrix<T>(rows, cols);

    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            matrix[r, c] = tensor[r, c];

    return matrix;
}

/// <summary>
/// Calculates column means for background data.
/// </summary>
private static Vector<T> CalculateColumnMeans<T>(Matrix<T> data)
{
    var numOps = MathHelper.GetNumericOperations<T>();
    var means = new Vector<T>(data.Columns);

    for (int col = 0; col < data.Columns; col++)
    {
        T sum = numOps.Zero;
        for (int row = 0; row < data.Rows; row++)
        {
            sum = numOps.Add(sum, data[row, col]);
        }
        means[col] = numOps.Divide(sum, numOps.FromDouble(data.Rows));
    }

    return means;
}
```

**What Changed**:
- ❌ **BEFORE**: `return model.GetShapValuesAsync(inputs);` (circular reference)
- ✅ **AFTER**: Actual Monte Carlo Shapley value calculation

---

## Phase 2: Fix LIME Circular Reference Bug

### Step 1: Understand the LIME Algorithm

**LIME (Local Interpretable Model-agnostic Explanations)** creates a simple linear model that approximates the complex model locally around a specific instance.

**Algorithm**:
1. Generate perturbed samples around the instance
2. Get predictions from the black-box model for perturbed samples
3. Weight samples by distance from original instance
4. Train a simple linear model on weighted samples
5. Extract feature importances from linear model coefficients

### Step 2: Implement LIME Algorithm in InterpretableModelHelper

**File**: `src/Interpretability/InterpretableModelHelper.cs`

**REPLACE lines 75-96** with actual implementation:

```csharp
/// <summary>
/// Gets LIME explanation for a specific input.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <param name="model">The model to analyze.</param>
/// <param name="enabledMethods">The set of enabled interpretation methods.</param>
/// <param name="input">The input to explain (shape: [num_features]).</param>
/// <param name="numFeatures">The number of top features to include in explanation.</param>
/// <param name="numSamples">Number of perturbed samples to generate (default: 5000).</param>
/// <returns>A LIME explanation.</returns>
/// <remarks>
/// <para><b>For Beginners:</b> LIME explains a single prediction by creating a simple model nearby.
///
/// Think of LIME like learning to ride a bike:
/// - The black-box model is like a complex bike path with hills and turns
/// - LIME draws a straight line that approximates the path near where you are
/// - This simple line helps you understand what direction you're going locally
///
/// LIME creates synthetic examples near your input and fits a simple linear model to them.
/// </para>
/// </remarks>
public static async Task<LimeExplanation<T>> GetLimeExplanationAsync<T>(
    IInterpretableModel<T> model,
    HashSet<InterpretationMethod> enabledMethods,
    Tensor<T> input,
    int numFeatures = 10,
    int numSamples = 5000)
{
    if (!enabledMethods.Contains(InterpretationMethod.LIME))
    {
        throw new InvalidOperationException("LIME method is not enabled.");
    }

    var numOps = MathHelper.GetNumericOperations<T>();
    var random = new Random();

    // Convert input tensor to vector
    var inputVector = ConvertTensorToVector(input);
    int totalFeatures = inputVector.Length;

    // Get prediction for original instance
    var originalPrediction = await PredictSingleAsync(model, inputVector);

    // Generate perturbed samples
    var samples = new Matrix<T>(numSamples, totalFeatures);
    var predictions = new Vector<T>(numSamples);
    var weights = new Vector<T>(numSamples);

    for (int i = 0; i < numSamples; i++)
    {
        // Create perturbed sample by adding Gaussian noise
        var perturbedSample = new Vector<T>(totalFeatures);
        T distance = numOps.Zero;

        for (int j = 0; j < totalFeatures; j++)
        {
            // Add Gaussian noise: N(0, 0.5 * feature_std)
            double noise = SampleGaussian(random) * 0.5;
            T perturbation = numOps.FromDouble(noise);
            perturbedSample[j] = numOps.Add(inputVector[j], perturbation);

            // Track distance from original
            T diff = numOps.Subtract(perturbedSample[j], inputVector[j]);
            distance = numOps.Add(distance, numOps.Multiply(diff, diff));

            samples[i, j] = perturbedSample[j];
        }

        // Get prediction for perturbed sample
        predictions[i] = await PredictSingleAsync(model, perturbedSample);

        // Calculate weight: exp(-distance^2 / kernel_width^2)
        // Higher weight for samples closer to original instance
        T kernelWidth = numOps.FromDouble(0.75);
        T expArg = numOps.Negate(numOps.Divide(distance, numOps.Multiply(kernelWidth, kernelWidth)));
        weights[i] = numOps.Exp(expArg);
    }

    // Fit weighted linear regression
    var (coefficients, intercept, r2) = FitWeightedLinearRegression(samples, predictions, weights);

    // Select top features by absolute coefficient magnitude
    var featureImportance = new Dictionary<int, T>();
    var sortedFeatures = coefficients.ToArray()
        .Select((coef, idx) => new { Index = idx, AbsValue = Math.Abs(numOps.ToDouble(coef)) })
        .OrderByDescending(x => x.AbsValue)
        .Take(numFeatures);

    foreach (var feature in sortedFeatures)
    {
        featureImportance[feature.Index] = coefficients[feature.Index];
    }

    return new LimeExplanation<T>
    {
        FeatureImportance = featureImportance,
        Intercept = intercept,
        PredictedValue = originalPrediction,
        LocalModelScore = r2,
        NumFeatures = numFeatures
    };
}

/// <summary>
/// Samples from a standard Gaussian distribution using Box-Muller transform.
/// </summary>
private static double SampleGaussian(Random random)
{
    double u1 = 1.0 - random.NextDouble(); // Uniform(0,1]
    double u2 = 1.0 - random.NextDouble();
    return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
}

/// <summary>
/// Fits a weighted linear regression model.
/// </summary>
/// <returns>Tuple of (coefficients, intercept, R-squared)</returns>
private static (Vector<T> coefficients, T intercept, T r2) FitWeightedLinearRegression<T>(
    Matrix<T> X,
    Vector<T> y,
    Vector<T> weights)
{
    var numOps = MathHelper.GetNumericOperations<T>();
    int n = X.Rows;
    int p = X.Columns;

    // Weighted least squares: (X^T * W * X)^-1 * X^T * W * y
    // Where W is diagonal matrix of weights

    // Create weighted X and y
    var XWeighted = new Matrix<T>(n, p);
    var yWeighted = new Vector<T>(n);

    for (int i = 0; i < n; i++)
    {
        T sqrtWeight = numOps.Sqrt(weights[i]);
        for (int j = 0; j < p; j++)
        {
            XWeighted[i, j] = numOps.Multiply(X[i, j], sqrtWeight);
        }
        yWeighted[i] = numOps.Multiply(y[i], sqrtWeight);
    }

    // Add intercept column (all ones, weighted)
    var XWithIntercept = new Matrix<T>(n, p + 1);
    for (int i = 0; i < n; i++)
    {
        XWithIntercept[i, 0] = numOps.Sqrt(weights[i]); // Intercept column
        for (int j = 0; j < p; j++)
        {
            XWithIntercept[i, j + 1] = XWeighted[i, j];
        }
    }

    // Solve: (X^T * X)^-1 * X^T * y
    var XTX = XWithIntercept.Transpose().Multiply(XWithIntercept);
    var XTy = XWithIntercept.Transpose().MultiplyVector(yWeighted);

    // Solve linear system (may need to add regularization for stability)
    var beta = XTX.Solve(XTy);

    T intercept = beta[0];
    var coefficients = new Vector<T>(p);
    for (int i = 0; i < p; i++)
    {
        coefficients[i] = beta[i + 1];
    }

    // Calculate R-squared
    T ssTot = numOps.Zero;
    T ssRes = numOps.Zero;
    T yMean = numOps.Zero;

    for (int i = 0; i < n; i++)
    {
        yMean = numOps.Add(yMean, y[i]);
    }
    yMean = numOps.Divide(yMean, numOps.FromDouble(n));

    for (int i = 0; i < n; i++)
    {
        T yPred = intercept;
        for (int j = 0; j < p; j++)
        {
            yPred = numOps.Add(yPred, numOps.Multiply(coefficients[j], X[i, j]));
        }

        T residual = numOps.Subtract(y[i], yPred);
        ssRes = numOps.Add(ssRes, numOps.Multiply(residual, residual));

        T deviation = numOps.Subtract(y[i], yMean);
        ssTot = numOps.Add(ssTot, numOps.Multiply(deviation, deviation));
    }

    T r2 = numOps.Subtract(numOps.One, numOps.Divide(ssRes, ssTot));

    return (coefficients, intercept, r2);
}

/// <summary>
/// Converts Tensor<T> to Vector<T> assuming 1D tensor.
/// </summary>
private static Vector<T> ConvertTensorToVector<T>(Tensor<T> tensor)
{
    if (tensor.Dimensions.Length == 1)
    {
        int length = tensor.Dimensions[0];
        var vector = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            vector[i] = tensor[i];
        }
        return vector;
    }
    else if (tensor.Dimensions.Length == 2 && tensor.Dimensions[0] == 1)
    {
        // [1, features] tensor - treat as single vector
        int length = tensor.Dimensions[1];
        var vector = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            vector[i] = tensor[0, i];
        }
        return vector;
    }

    throw new ArgumentException("Tensor must be 1D or [1, features] for LIME.");
}
```

---

## Common Pitfalls to Avoid:

1. **DON'T call model methods from helper that call back to helper** - This causes infinite loops
2. **DON'T forget to add helper methods** - `CreateCoalitionInstance()`, `PredictSingleAsync()`, etc.
3. **DON'T use small numSamples** - SHAP/LIME need hundreds of samples for stable estimates
4. **DON'T forget to handle Matrix.Solve() failures** - May need regularization for numerical stability
5. **DO validate tensor shapes** - SHAP needs 2D [samples, features], LIME needs 1D [features]
6. **DO use proper kernel width** - LIME kernel width controls locality (0.75 is a good default)
7. **DO weight LIME samples** - Closer samples should have higher weight
8. **DO test with known models** - Test on linear models where SHAP/LIME should match coefficients

---

## Testing Strategy:

1. **Unit Tests**: Test helper methods work correctly (no infinite loops!)
2. **Algorithm Tests**: Verify SHAP/LIME produce reasonable explanations
3. **Known Model Tests**: Test on linear models where explanations are known
4. **Numerical Stability**: Test with ill-conditioned matrices
5. **Performance**: Measure speed with different numSamples values

**Next Steps**:
1. Fix InterpretableModelHelper.GetShapValuesAsync() (remove circular reference, add algorithm)
2. Fix InterpretableModelHelper.GetLimeExplanationAsync() (remove circular reference, add algorithm)
3. Add helper methods (CreateCoalitionInstance, PredictSingleAsync, etc.)
4. Test thoroughly to ensure no more infinite loops
5. Extract to standalone SHAPExplainer and LIMEExplainer classes (optional but recommended)
