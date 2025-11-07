## Problem Statement

**Cross-validation infrastructure EXISTS (90% complete)** but uses an **incompatible training workflow** that prevents integration with `PredictionModelBuilder`.

### Current State (What EXISTS)

**All cross-validation infrastructure is already implemented:**

**Interfaces**: ✅ COMPLETE
- `src/Interfaces/ICrossValidator.cs` (lines 24-55)

**Base Classes**: ✅ COMPLETE
- `src/CrossValidators/CrossValidatorBase.cs`

**Concrete Implementations**: ✅ COMPLETE (8 validators)
- `src/CrossValidators/KFoldCrossValidator.cs`
- `src/CrossValidators/StratifiedKFoldCrossValidator.cs`
- `src/CrossValidators/StandardCrossValidator.cs`
- `src/CrossValidators/TimeSeriesCrossValidator.cs`
- `src/CrossValidators/LeaveOneOutCrossValidator.cs`
- `src/CrossValidators/MonteCarloValidator.cs`
- `src/CrossValidators/GroupKFoldCrossValidator.cs`
- `src/CrossValidators/NestedCrossValidator.cs`

**Result Classes**: ✅ COMPLETE
- `src/Models/Results/CrossValidationResult.cs`
- `src/Models/Results/FoldResult.cs`

**Configuration**: ✅ COMPLETE
- `src/Models/Options/CrossValidationOptions.cs`

### The Problem: Architectural Mismatch

**Existing Cross-Validation Workflow** (src/CrossValidators/CrossValidatorBase.cs:131):
```csharp
// CrossValidatorBase calls model.Train() directly
model.Train(XTrain, yTrain);
```

**PredictionModelBuilder Workflow** (src/PredictionModelBuilder.cs:276):
```csharp
// PredictionModelBuilder uses optimizer.Optimize()
var optimizer = _optimizer ?? new NormalOptimizer<T, TInput, TOutput>(_model);
var optimizationResult = optimizer.Optimize(
    OptimizerHelper<T, TInput, TOutput>.CreateOptimizationInputData(
        XTrain, yTrain, XVal, yVal, XTest, yTest));
```

**Why This Matters**:
- Users configure optimizers via `.ConfigureOptimizer()` in the builder
- Cross-validation bypasses the optimizer entirely
- Users lose control over training hyperparameters during CV
- Cannot use advanced optimizers (genetic algorithms, Bayesian optimization, etc.) with CV

**Additional Issue**: Model State Leakage
- CrossValidatorBase.cs:131 reuses the same model instance across all folds
- Should call `model.DeepCopy()` for each fold to ensure independence
- Note: IFullModel already has DeepCopy() via ICloneable inheritance (src/Interfaces/ICloneable.cs:12)

## Required Changes

### Change 1: Modify `ICrossValidator<T>` Interface

**File**: `src/Interfaces/ICrossValidator.cs`

**Current signature** (line 51):
```csharp
CrossValidationResult<T> Validate(
    IFullModel<T, Matrix<T>, Vector<T>> model,
    Matrix<T> X,
    Vector<T> y);
```

**New signature**:
```csharp
CrossValidationResult<T> Validate(
    IFullModel<T, Matrix<T>, Vector<T>> model,
    IOptimizer<T, Matrix<T>, Vector<T>> optimizer,  // NEW
    Matrix<T> X,
    Vector<T> y);
```

**Rationale**: Cross-validation must accept optimizer to match builder workflow.

### Change 2: Modify `CrossValidatorBase<T>` Base Class

**File**: `src/CrossValidators/CrossValidatorBase.cs`

**A. Update PerformCrossValidation method signature** (line 116-117):

**BEFORE**:
```csharp
protected CrossValidationResult<T> PerformCrossValidation(
    IFullModel<T, Matrix<T>, Vector<T>> model,
    Matrix<T> X,
    Vector<T> y,
    IEnumerable<(int[] trainIndices, int[] validationIndices)> folds)
```

**AFTER**:
```csharp
protected CrossValidationResult<T> PerformCrossValidation(
    IFullModel<T, Matrix<T>, Vector<T>> model,
    IOptimizer<T, Matrix<T>, Vector<T>> optimizer,  // NEW
    Matrix<T> X,
    Vector<T> y,
    IEnumerable<(int[] trainIndices, int[] validationIndices)> folds)
```

**B. Replace model.Train() with optimizer.Optimize()** (lines 127-135):

**BEFORE**:
```csharp
foreach (var (trainIndices, validationIndices) in folds)
{
    var sw = System.Diagnostics.Stopwatch.StartNew();
    foldNumber++;

    // Extract training and validation data for this fold
    var XTrain = X[trainIndices, Enumerable.Range(0, X.Cols).ToArray()];
    var yTrain = y[trainIndices];
    var XVal = X[validationIndices, Enumerable.Range(0, X.Cols).ToArray()];
    var yVal = y[validationIndices];

    // Train the model on this fold
    model.Train(XTrain, yTrain);  // ← PROBLEM: Bypasses optimizer
```

**AFTER**:
```csharp
foreach (var (trainIndices, validationIndices) in folds)
{
    var sw = System.Diagnostics.Stopwatch.StartNew();
    foldNumber++;

    // Extract training and validation data for this fold
    var XTrain = X[trainIndices, Enumerable.Range(0, X.Cols).ToArray()];
    var yTrain = y[trainIndices];
    var XVal = X[validationIndices, Enumerable.Range(0, X.Cols).ToArray()];
    var yVal = y[validationIndices];

    // Create independent model copy for this fold (prevent state leakage)
    var foldModel = model.DeepCopy();  // ← FIX: Use DeepCopy() from ICloneable

    // Create empty test sets (CV doesn't use test data)
    var emptyMatrix = new Matrix<T>(0, X.Cols);
    var emptyVector = new Vector<T>(0);

    // Use optimizer workflow like PredictionModelBuilder does
    var optimizationInput = OptimizerHelper<T, Matrix<T>, Vector<T>>.CreateOptimizationInputData(
        XTrain, yTrain,
        XVal, yVal,
        emptyMatrix, emptyVector);  // CV doesn't have test set

    var optimizationResult = optimizer.Optimize(optimizationInput);  // ← FIX: Use optimizer
    var trainedModel = optimizationResult.BestSolution;  // ← Get trained model from result
```

**C. Update FoldResult construction** (around line 140):

**BEFORE**:
```csharp
// Generate predictions and calculate errors
var trainPredictions = model.Predict(XTrain);
var valPredictions = model.Predict(XVal);
```

**AFTER**:
```csharp
// Generate predictions and calculate errors using the trained model from optimization
var trainPredictions = trainedModel.Predict(XTrain);
var valPredictions = trainedModel.Predict(XVal);
```

### Change 3: Update All 8 Concrete Implementations

**Files to modify**:
- `src/CrossValidators/KFoldCrossValidator.cs`
- `src/CrossValidators/StratifiedKFoldCrossValidator.cs`
- `src/CrossValidators/StandardCrossValidator.cs`
- `src/CrossValidators/TimeSeriesCrossValidator.cs`
- `src/CrossValidators/LeaveOneOutCrossValidator.cs`
- `src/CrossValidators/MonteCarloValidator.cs`
- `src/CrossValidators/GroupKFoldCrossValidator.cs`
- `src/CrossValidators/NestedCrossValidator.cs`

**Pattern** (example from KFoldCrossValidator.cs:76):

**BEFORE**:
```csharp
public override CrossValidationResult<T> Validate(
    IFullModel<T, Matrix<T>, Vector<T>> model,
    Matrix<T> X,
    Vector<T> y)
{
    var folds = CreateFolds(X, y);
    return PerformCrossValidation(model, X, y, folds);
}
```

**AFTER**:
```csharp
public override CrossValidationResult<T> Validate(
    IFullModel<T, Matrix<T>, Vector<T>> model,
    IOptimizer<T, Matrix<T>, Vector<T>> optimizer,  // NEW
    Matrix<T> X,
    Vector<T> y)
{
    var folds = CreateFolds(X, y);
    return PerformCrossValidation(model, optimizer, X, y, folds);  // Pass optimizer
}
```

**Apply this pattern to all 8 validators**.

### Change 4: Integrate with `PredictionModelBuilder`

**File**: `src/PredictionModelBuilder.cs`

**A. Add private field** (around line 48):
```csharp
private ICrossValidator<T>? _crossValidator;
```

**B. Add configuration method** (after line 473):
```csharp
/// <summary>
/// Configures the cross-validation strategy for model evaluation.
/// </summary>
/// <param name="validator">The cross-validation strategy to use.</param>
/// <returns>This builder instance for method chaining.</returns>
/// <remarks>
/// <b>For Beginners:</b> Cross-validation helps you get a more reliable estimate of how well
/// your model will perform on new data by testing it on multiple train/test splits instead of
/// just one. This gives you a better understanding of your model's true performance.
/// </remarks>
public IPredictionModelBuilder<T, TInput, TOutput> ConfigureCrossValidator(
    ICrossValidator<T> validator)
{
    _crossValidator = validator;
    return this;
}
```

**C. Modify Build() method** (around lines 563-596):

**BEFORE**:
```csharp
// Preprocess the data
var (preprocessedX, preprocessedY, normInfo) = dataPreprocessor.PreprocessData(x, y);

// Split data and optimize
var (XTrain, yTrain, XVal, yVal, XTest, yTest) = dataPreprocessor.SplitData(preprocessedX, preprocessedY);
optimizationResult = optimizer.Optimize(
    OptimizerHelper<T, TInput, TOutput>.CreateOptimizationInputData(
        XTrain, yTrain, XVal, yVal, XTest, yTest));

return new PredictionModelResult<T, TInput, TOutput>(
    optimizationResult,
    normInfo,
    _biasDetector,
    _fairnessEvaluator,
    _ragRetriever,
    _ragReranker,
    _ragGenerator,
    _queryProcessors,
    _loraConfiguration);
```

**AFTER**:
```csharp
// Preprocess the data
var (preprocessedX, preprocessedY, normInfo) = dataPreprocessor.PreprocessData(x, y);

OptimizationResult<T, TInput, TOutput> optimizationResult;
CrossValidationResult<T>? crossValidationResult = null;

if (_crossValidator != null)
{
    // Convert TInput/TOutput to Matrix/Vector for cross-validation
    var xMatrix = ConversionsHelper.ConvertToMatrix<T, TInput>(preprocessedX);
    var yVector = ConversionsHelper.ConvertToVector<T, TOutput>(preprocessedY);

    // Create optimizer for CV to use (same as would be used for normal training)
    var cvOptimizer = new NormalOptimizer<T, Matrix<T>, Vector<T>>(_model);

    // Perform cross-validation using the optimizer
    crossValidationResult = _crossValidator.Validate(_model, cvOptimizer, xMatrix, yVector);

    // Use the best fold's model as the final model
    var bestFold = crossValidationResult.FoldResults
        .OrderByDescending(f => f.ValidationPredictionStats.R2)
        .FirstOrDefault();

    if (bestFold?.Model != null)
    {
        // Create synthetic OptimizationResult from CV results
        optimizationResult = CreateOptimizationResultFromCV(crossValidationResult, bestFold.Model);
    }
    else
    {
        // Fallback: single train/val/test split
        var (XTrain, yTrain, XVal, yVal, XTest, yTest) = dataPreprocessor.SplitData(preprocessedX, preprocessedY);
        optimizationResult = optimizer.Optimize(
            OptimizerHelper<T, TInput, TOutput>.CreateOptimizationInputData(
                XTrain, yTrain, XVal, yVal, XTest, yTest));
    }
}
else
{
    // Normal workflow: single train/val/test split
    var (XTrain, yTrain, XVal, yVal, XTest, yTest) = dataPreprocessor.SplitData(preprocessedX, preprocessedY);
    optimizationResult = optimizer.Optimize(
        OptimizerHelper<T, TInput, TOutput>.CreateOptimizationInputData(
            XTrain, yTrain, XVal, yVal, XTest, yTest));
}

return new PredictionModelResult<T, TInput, TOutput>(
    optimizationResult,
    normInfo,
    _biasDetector,
    _fairnessEvaluator,
    _ragRetriever,
    _ragReranker,
    _ragGenerator,
    _queryProcessors,
    _loraConfiguration,
    crossValidationResult);  // NEW: Pass CV result to constructor
```

**D. Add helper method**:
```csharp
/// <summary>
/// Creates a synthetic OptimizationResult from CrossValidationResult for API compatibility.
/// </summary>
private OptimizationResult<T, TInput, TOutput> CreateOptimizationResultFromCV(
    CrossValidationResult<T> cvResult,
    IFullModel<T, Matrix<T>, Vector<T>> bestModel)
{
    var bestFold = cvResult.FoldResults
        .OrderByDescending(f => f.ValidationPredictionStats.R2)
        .FirstOrDefault();

    if (bestFold == null)
        throw new InvalidOperationException("Cross-validation produced no valid folds.");

    return new OptimizationResult<T, TInput, TOutput>
    {
        BestSolution = (IFullModel<T, TInput, TOutput>)bestModel,
        BestFitnessScore = bestFold.ValidationPredictionStats.R2,
        TrainingResult = new OptimizationResult<T, TInput, TOutput>.DatasetResult
        {
            ErrorStats = bestFold.TrainingErrors,
            PredictionStats = bestFold.TrainingPredictionStats
        },
        ValidationResult = new OptimizationResult<T, TInput, TOutput>.DatasetResult
        {
            ErrorStats = bestFold.ValidationErrors,
            PredictionStats = bestFold.ValidationPredictionStats
        },
        TestResult = new OptimizationResult<T, TInput, TOutput>.DatasetResult
        {
            ErrorStats = bestFold.ValidationErrors,
            PredictionStats = bestFold.ValidationPredictionStats
        }
    };
}
```

### Change 5: Update `PredictionModelResult`

**File**: `src/PredictionModelResult.cs`

**A. Add property**:
```csharp
/// <summary>
/// Gets the cross-validation results, if cross-validation was performed.
/// </summary>
public CrossValidationResult<T>? CrossValidationResult { get; }
```

**B. Update regular build constructor** (add parameter):
```csharp
public PredictionModelResult(
    OptimizationResult<T, TInput, TOutput> optimizationResult,
    NormalizationInfo<T, TInput, TOutput> normalizationInfo,
    IBiasDetector<T>? biasDetector,
    IFairnessEvaluator<T>? fairnessEvaluator,
    IRetriever<T>? ragRetriever,
    IReranker<T>? ragReranker,
    IGenerator<T>? ragGenerator,
    IEnumerable<IQueryProcessor>? queryProcessors,
    ILoRAConfiguration<T>? loraConfiguration,
    CrossValidationResult<T>? crossValidationResult = null)  // NEW
{
    // ... existing assignments ...
    CrossValidationResult = crossValidationResult;
}
```

**C. Update meta-learning constructor**:
```csharp
public PredictionModelResult(
    IMetaLearner<T, TInput, TOutput> metaLearner,
    MetaLearningResult<T, TInput, TOutput> metaResult,
    ILoRAConfiguration<T>? loraConfiguration,
    IBiasDetector<T>? biasDetector,
    IFairnessEvaluator<T>? fairnessEvaluator,
    IRetriever<T>? ragRetriever,
    IReranker<T>? ragReranker,
    IGenerator<T>? ragGenerator,
    IEnumerable<IQueryProcessor>? queryProcessors)
{
    // ... existing assignments ...
    CrossValidationResult = null;  // Not applicable for meta-learning
}
```

### Change 6: Verify FoldResult Stores Model

**File**: `src/Models/Results/FoldResult.cs`

**Verify this property exists**:
```csharp
/// <summary>
/// Gets the trained model for this fold.
/// </summary>
public IFullModel<T, Matrix<T>, Vector<T>>? Model { get; init; }
```

**If missing, add it**. Each fold must store its trained model for best-fold selection.

## Story Points Calculation

Based on actual changes required:

- **ICrossValidator interface change**: 1 file, add 1 parameter = 2 points
- **CrossValidatorBase refactoring**: 1 file, 3 methods modified (signature + optimizer logic + predictions) = 8 points
- **8 concrete implementations**: 8 files, 1 method each = 8 points
- **PredictionModelBuilder integration**: 1 file, 3 additions (field + ConfigureCrossValidator + Build logic) = 8 points
- **CreateOptimizationResultFromCV helper**: 1 new method = 3 points
- **PredictionModelResult changes**: 1 file, 2 constructor updates + 1 property = 5 points
- **FoldResult verification**: 1 file check = 1 point
- **Testing**: Integration tests for CV workflow = 5 points
- **Documentation**: XML docs update = 2 points

**Total**: 42 story points

## Acceptance Criteria

1. ✅ User can call `.ConfigureCrossValidator(new KFoldCrossValidator<T>(options))` on builder
2. ✅ Cross-validation uses the optimizer configured via `.ConfigureOptimizer()` (or NormalOptimizer default)
3. ✅ Each fold gets independent model copy via `model.DeepCopy()` (no state leakage)
4. ✅ When CV configured, `Build(X, y)` performs cross-validation instead of single split
5. ✅ Results accessible via `result.CrossValidationResult.R2Stats.Mean`, `.StandardDeviation`, `.FoldResults`
6. ✅ When CV not configured, existing single-split behavior unchanged (backward compatible)
7. ✅ Best fold's model used for predictions via `result.Predict(newData)`
8. ✅ All numeric types (float, double, decimal) supported

## Usage Example

**Without Cross-Validation (Current Default)**:
```csharp
var builder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new LinearRegressionModel<double, Matrix<double>, Vector<double>>())
    .ConfigureOptimizer(new GeneticAlgorithmOptimizer<double, Matrix<double>, Vector<double>>());

var result = builder.Build(X, y);
// Single train/val/test split, one optimization run
```

**With Cross-Validation (New Feature)**:
```csharp
var cvOptions = new CrossValidationOptions
{
    NumberOfFolds = 5,
    RandomSeed = 42,
    Shuffle = true
};

var builder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new LinearRegressionModel<double, Matrix<double>, Vector<double>>())
    .ConfigureOptimizer(new GeneticAlgorithmOptimizer<double, Matrix<double>, Vector<double>>())
    .ConfigureCrossValidator(new KFoldCrossValidator<double>(cvOptions));  // NEW

var result = builder.Build(X, y);
// 5-fold CV, each fold optimized with GeneticAlgorithmOptimizer

// Access aggregated CV statistics
Console.WriteLine($"Average R²: {result.CrossValidationResult.R2Stats.Mean:F4}");
Console.WriteLine($"Std Dev: {result.CrossValidationResult.R2Stats.StandardDeviation:F4}");
Console.WriteLine($"RMSE: {result.CrossValidationResult.RMSEStats.Mean:F4}");

// Access individual fold results
foreach (var fold in result.CrossValidationResult.FoldResults)
{
    Console.WriteLine($"Fold {fold.FoldIndex}: R² = {fold.ValidationPredictionStats.R2:F4}");
}

// Generate report
Console.WriteLine(result.CrossValidationResult.GenerateReport());

// Make predictions using best fold's model
var predictions = result.Predict(newData);
```

---

# Part 2: Clustering Metrics for Validation

## Problem Statement

**Clustering metrics exist** but are **not automatically utilized** in cross-validation and model evaluation workflows.

### Current State (Clustering Metrics)

**Silhouette Score**: ✅ EXISTS but ❌ NOT INTEGRATED
- `MetricType.SilhouetteScore` exists in enum (src/Enums/MetricType.cs:522)
- `StatisticsHelper.CalculateSilhouetteScore()` exists (src/Helpers/StatisticsHelper.cs:5838)
- `ModelStats.SilhouetteScore` property exists and is calculated (src/Statistics/ModelStats.cs:297, 528)
- **BUT**: Not included in `FoldResult` for cross-validation
- **BUT**: Not automatically calculated for clustering models during validation

**Adjusted Rand Index**: ❌ DOES NOT EXIST
- Not in `MetricType` enum
- No implementation in `StatisticsHelper`
- Not available in `ModelStats`
- Needs to be created from scratch

**Other Clustering Metrics** (exist but same integration issue):
- `CalinskiHarabaszIndex` - exists in ModelStats but not in FoldResult
- `DaviesBouldinIndex` - exists in ModelStats but not in FoldResult

### The Problem

**For clustering models, cross-validation should automatically calculate clustering-specific metrics:**

```csharp
// ❌ Current: FoldResult only has ErrorStats and PredictionStats
var fold = cvResult.FoldResults[0];
// No access to fold.SilhouetteScore
// No access to fold.AdjustedRandIndex
// No access to fold.CalinskiHarabaszIndex
```

**Desired:**
```csharp
// ✅ Desired: FoldResult includes clustering metrics for clustering models
var fold = cvResult.FoldResults[0];
Console.WriteLine($"Silhouette: {fold.ClusteringMetrics.SilhouetteScore:F4}");
Console.WriteLine($"Adjusted Rand Index: {fold.ClusteringMetrics.AdjustedRandIndex:F4}");
Console.WriteLine($"Calinski-Harabasz: {fold.ClusteringMetrics.CalinskiHarabaszIndex:F4}");
```

## Required Changes (Clustering Metrics)

### Change 7: Add Adjusted Rand Index Metric

**File**: `src/Enums/MetricType.cs`

**Add after line 522 (after SilhouetteScore)**:
```csharp
/// <summary>
/// Measures the similarity between two clusterings adjusted for chance.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Adjusted Rand Index compares two different ways of grouping the same data
/// and tells you how similar they are. It ranges from -1 to 1, where 1 means perfect agreement,
/// 0 means random agreement, and negative values mean worse than random. It's useful for comparing
/// your model's clusters against known ground truth labels.
/// </para>
/// </remarks>
AdjustedRandIndex,
```

### Change 8: Implement Adjusted Rand Index Calculation

**File**: `src/Helpers/StatisticsHelper.cs`

**Add new method after CalculateSilhouetteScore (after line ~5900)**:
```csharp
/// <summary>
/// Calculates the Adjusted Rand Index between two clusterings.
/// </summary>
/// <param name="labels1">The first set of cluster labels.</param>
/// <param name="labels2">The second set of cluster labels.</param>
/// <returns>The Adjusted Rand Index value.</returns>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This method calculates how similar two different ways of grouping the same
/// data are. The result ranges from -1 to 1:
/// - 1.0: Perfect agreement (identical groupings)
/// - 0.0: Agreement no better than random chance
/// - Negative: Agreement worse than random
/// It's adjusted for chance, meaning it accounts for the possibility of random agreement.
/// </para>
/// </remarks>
public static T CalculateAdjustedRandIndex(Vector<T> labels1, Vector<T> labels2)
{
    if (labels1.Length != labels2.Length)
        throw new ArgumentException("Label vectors must have the same length.");

    var numOps = NumericOperations<T>.Instance;
    int n = labels1.Length;

    // Build contingency table
    var uniqueLabels1 = labels1.Distinct().ToList();
    var uniqueLabels2 = labels2.Distinct().ToList();

    var contingencyTable = new Dictionary<(T, T), int>();
    foreach (var label1 in uniqueLabels1)
    {
        foreach (var label2 in uniqueLabels2)
        {
            contingencyTable[(label1, label2)] = 0;
        }
    }

    for (int i = 0; i < n; i++)
    {
        var key = (labels1[i], labels2[i]);
        if (contingencyTable.ContainsKey(key))
            contingencyTable[key]++;
        else
            contingencyTable[key] = 1;
    }

    // Calculate sums
    var nij = contingencyTable.Values.Select(v => v).ToList();
    var ai = uniqueLabels1.Select(label1 =>
        Enumerable.Range(0, n).Count(i => numOps.AreEqual(labels1[i], label1))).ToList();
    var bj = uniqueLabels2.Select(label2 =>
        Enumerable.Range(0, n).Count(i => numOps.AreEqual(labels2[i], label2))).ToList();

    // Calculate index
    T sumCombNij = numOps.Zero;
    foreach (var count in nij)
    {
        if (count >= 2)
            sumCombNij = numOps.Add(sumCombNij, numOps.FromDouble(count * (count - 1) / 2.0));
    }

    T sumCombAi = numOps.Zero;
    foreach (var count in ai)
    {
        if (count >= 2)
            sumCombAi = numOps.Add(sumCombAi, numOps.FromDouble(count * (count - 1) / 2.0));
    }

    T sumCombBj = numOps.Zero;
    foreach (var count in bj)
    {
        if (count >= 2)
            sumCombBj = numOps.Add(sumCombBj, numOps.FromDouble(count * (count - 1) / 2.0));
    }

    T combN = numOps.FromDouble(n * (n - 1) / 2.0);
    T expectedIndex = numOps.Divide(numOps.Multiply(sumCombAi, sumCombBj), combN);
    T maxIndex = numOps.Divide(numOps.Add(sumCombAi, sumCombBj), numOps.FromDouble(2.0));

    T numerator = numOps.Subtract(sumCombNij, expectedIndex);
    T denominator = numOps.Subtract(maxIndex, expectedIndex);

    if (numOps.AreEqual(denominator, numOps.Zero))
        return numOps.One; // Perfect agreement

    return numOps.Divide(numerator, denominator);
}
```

### Change 9: Add Adjusted Rand Index to ModelStats

**File**: `src/Statistics/ModelStats.cs`

**A. Add property after SilhouetteScore (after line 297)**:
```csharp
/// <summary>
/// Gets the Adjusted Rand Index, measuring agreement between predicted and actual clusterings.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This index tells you how well your clustering matches the true groups.
/// It ranges from -1 to 1, where:
/// - 1.0 means perfect agreement
/// - 0.0 means no better than random
/// - Negative means worse than random
/// </para>
/// </remarks>
public T AdjustedRandIndex { get; private set; }
```

**B. Initialize in constructor (around line 415)**:
```csharp
AdjustedRandIndex = _numOps.Zero;
```

**C. Calculate in Calculate method (after line 528)**:
```csharp
AdjustedRandIndex = StatisticsHelper<T>.CalculateAdjustedRandIndex(actual, predicted);
```

**D. Add to GetMetric switch (after line 591)**:
```csharp
MetricType.AdjustedRandIndex => AdjustedRandIndex,
```

**E. Add to HasMetric switch (after line 647)**:
```csharp
MetricType.AdjustedRandIndex => true,
```

### Change 10: Add ClusteringMetrics to FoldResult

**File**: `src/Models/Results/FoldResult.cs`

**A. Add property after TrainingPredictionStats (around line 34)**:
```csharp
/// <summary>
/// Gets the clustering-specific metrics for this fold (null for non-clustering models).
/// </summary>
public ClusteringMetrics<T>? ClusteringMetrics { get; }
```

**B. Update constructor to accept clustering metrics (add parameter around line 88)**:
```csharp
public FoldResult(
    int foldIndex,
    Vector<T> trainingActual,
    Vector<T> trainingPredicted,
    Vector<T> validationActual,
    Vector<T> validationPredicted,
    Dictionary<string, T>? featureImportance = null,
    TimeSpan? trainingTime = null,
    TimeSpan? evaluationTime = null,
    int featureCount = 0,
    ClusteringMetrics<T>? clusteringMetrics = null)  // NEW
{
    // ... existing code ...
    ClusteringMetrics = clusteringMetrics;  // NEW
}
```

### Change 11: Create ClusteringMetrics Result Class

**File**: `src/Models/Results/ClusteringMetrics.cs` (NEW FILE)

```csharp
namespace AiDotNet.Models.Results;

/// <summary>
/// Contains clustering-specific evaluation metrics.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class holds metrics that are specifically useful for evaluating
/// clustering models, which group similar data points together. These metrics help you understand
/// how well your model separates different groups and how similar items are within each group.
/// </para>
/// </remarks>
public class ClusteringMetrics<T>
{
    /// <summary>
    /// Gets the Silhouette Score for the clustering.
    /// </summary>
    public T SilhouetteScore { get; }

    /// <summary>
    /// Gets the Adjusted Rand Index comparing predicted clusters to true labels.
    /// </summary>
    public T AdjustedRandIndex { get; }

    /// <summary>
    /// Gets the Calinski-Harabasz Index measuring cluster separation.
    /// </summary>
    public T CalinskiHarabaszIndex { get; }

    /// <summary>
    /// Gets the Davies-Bouldin Index measuring cluster similarity.
    /// </summary>
    public T DaviesBouldinIndex { get; }

    /// <summary>
    /// Creates a new instance of ClusteringMetrics.
    /// </summary>
    public ClusteringMetrics(
        T silhouetteScore,
        T adjustedRandIndex,
        T calinskiHarabaszIndex,
        T daviesBouldinIndex)
    {
        SilhouetteScore = silhouetteScore;
        AdjustedRandIndex = adjustedRandIndex;
        CalinskiHarabaszIndex = calinskiHarabaszIndex;
        DaviesBouldinIndex = daviesBouldinIndex;
    }
}
```

### Change 12: Add Clustering Metrics to CrossValidationResult

**File**: `src/Models/Results/CrossValidationResult.cs`

**Add properties after MAEStats (around line 40)**:
```csharp
/// <summary>
/// Gets basic statistics for Silhouette Score across folds (null if not a clustering model).
/// </summary>
public BasicStats<T>? SilhouetteStats { get; }

/// <summary>
/// Gets basic statistics for Adjusted Rand Index across folds (null if not a clustering model).
/// </summary>
public BasicStats<T>? AdjustedRandIndexStats { get; }
```

**Update constructor to calculate clustering stats (around line 70-92)**:
```csharp
public CrossValidationResult(List<FoldResult<T>> foldResults, TimeSpan totalTime)
{
    FoldResults = foldResults;
    TotalTime = totalTime;

    // ... existing calculations ...

    // Calculate clustering metrics if available
    if (foldResults.Any(f => f.ClusteringMetrics != null))
    {
        var silhouetteValues = new Vector<T>([.. foldResults
            .Where(f => f.ClusteringMetrics != null)
            .Select(f => f.ClusteringMetrics!.SilhouetteScore)]);
        SilhouetteStats = new BasicStats<T>(new BasicStatsInputs<T> { Values = silhouetteValues });

        var ariValues = new Vector<T>([.. foldResults
            .Where(f => f.ClusteringMetrics != null)
            .Select(f => f.ClusteringMetrics!.AdjustedRandIndex)]);
        AdjustedRandIndexStats = new BasicStats<T>(new BasicStatsInputs<T> { Values = ariValues });
    }
    else
    {
        SilhouetteStats = null;
        AdjustedRandIndexStats = null;
    }
}
```

### Change 13: Update CrossValidatorBase to Calculate Clustering Metrics

**File**: `src/CrossValidators/CrossValidatorBase.cs`

**Modify fold result creation (around line 140-160)** to include clustering metrics when applicable:

```csharp
// After calculating predictions...
var trainPredictions = trainedModel.Predict(XTrain);
var valPredictions = trainedModel.Predict(XVal);

// Calculate clustering metrics if this is a clustering model
ClusteringMetrics<T>? clusteringMetrics = null;
if (IsClusteringModel(model))
{
    clusteringMetrics = new ClusteringMetrics<T>(
        silhouetteScore: StatisticsHelper<T>.CalculateSilhouetteScore(XVal, yVal),
        adjustedRandIndex: StatisticsHelper<T>.CalculateAdjustedRandIndex(yVal, valPredictions),
        calinskiHarabaszIndex: StatisticsHelper<T>.CalculateCalinskiHarabaszIndex(XVal, yVal),
        daviesBouldinIndex: StatisticsHelper<T>.CalculateDaviesBouldinIndex(XVal, yVal)
    );
}

foldResults.Add(new FoldResult<T>(
    foldNumber,
    yTrain,
    trainPredictions,
    yVal,
    valPredictions,
    featureImportance,
    sw.Elapsed,
    TimeSpan.Zero,
    X.Cols,
    clusteringMetrics));  // NEW
```

**Add helper method to CrossValidatorBase**:
```csharp
/// <summary>
/// Determines if the model is a clustering model based on its type or interfaces.
/// </summary>
private bool IsClusteringModel(IFullModel<T, Matrix<T>, Vector<T>> model)
{
    // Check if model implements clustering-specific interfaces or is a known clustering type
    var modelType = model.GetType();
    return modelType.Name.Contains("Cluster") ||
           modelType.Name.Contains("KMeans") ||
           modelType.Name.Contains("DBSCAN") ||
           modelType.Name.Contains("Hierarchical");
}
```

## Story Points Calculation (Clustering Metrics)

- **Add AdjustedRandIndex to MetricType enum**: 1 point
- **Implement CalculateAdjustedRandIndex method**: 5 points
- **Add AdjustedRandIndex to ModelStats**: 3 points
- **Create ClusteringMetrics.cs class**: 3 points
- **Update FoldResult with ClusteringMetrics**: 2 points
- **Update CrossValidationResult with clustering stats**: 3 points
- **Modify CrossValidatorBase to calculate clustering metrics**: 5 points
- **Testing clustering metrics**: 5 points

**Clustering Metrics Subtotal**: 27 story points

## Combined Total Story Points

- **Cross-Validation Integration**: 42 points
- **Clustering Metrics**: 27 points
- **Total**: 69 story points

## Definition of Done

**Cross-Validation (Part 1):**
- [ ] `ICrossValidator<T>` interface updated with optimizer parameter
- [ ] `CrossValidatorBase.PerformCrossValidation()` modified to use optimizer
- [ ] `model.DeepCopy()` added before each fold in CrossValidatorBase
- [ ] All 8 concrete validators updated with new signature
- [ ] `_crossValidator` field added to PredictionModelBuilder
- [ ] `ConfigureCrossValidator()` method added to PredictionModelBuilder
- [ ] `Build()` method modified to conditionally use cross-validation
- [ ] `CreateOptimizationResultFromCV()` helper method implemented
- [ ] `CrossValidationResult` property added to PredictionModelResult
- [ ] Both PredictionModelResult constructors updated
- [ ] `FoldResult.Model` property verified to exist
- [ ] Integration tests pass for CV workflow
- [ ] Backward compatibility verified (non-CV path unchanged)

**Clustering Metrics (Part 2):**
- [ ] `AdjustedRandIndex` added to MetricType enum
- [ ] `CalculateAdjustedRandIndex()` implemented in StatisticsHelper
- [ ] `AdjustedRandIndex` property added to ModelStats
- [ ] `ClusteringMetrics.cs` class created
- [ ] `ClusteringMetrics` property added to FoldResult
- [ ] Constructor updated to accept clustering metrics
- [ ] `SilhouetteStats` and `AdjustedRandIndexStats` added to CrossValidationResult
- [ ] CrossValidatorBase calculates clustering metrics for clustering models
- [ ] `IsClusteringModel()` helper method added
- [ ] Clustering metrics automatically included in CV reports
- [ ] Integration tests pass for clustering models with CV
- [ ] All numeric types tested (double, float)
- [ ] Documentation complete with examples
