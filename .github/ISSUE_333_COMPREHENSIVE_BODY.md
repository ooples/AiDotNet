## Problem Statement

Cross-validation functionality exists in AiDotNet (`DefaultModelEvaluator.PerformCrossValidation()` at line 236), but it is **NOT integrated with the automatic `PredictionModelBuilder.Build()` workflow**. Users cannot configure cross-validation as part of the builder pattern, and CV results are not automatically calculated during model building.

**Current State:**
- `DefaultModelEvaluator.PerformCrossValidation()` exists at `src/Evaluation/DefaultModelEvaluator.cs:236`
- 8 cross-validators exist: `KFoldCrossValidator`, `StratifiedKFoldCrossValidator`, etc.
- These are **manual-only** - users must call them explicitly after building

**Required State:**
- Cross-validation configurable via `PredictionModelBuilder.ConfigureCrossValidator()`
- CV executes automatically during `Build(X, y)` when configured
- CV results stored in `PredictionModelResult.CrossValidationResult`
- CV follows industry standard pattern (sklearn workflow)

## Architecture Decisions

Based on deep analysis of `OptimizationAlgorithmOptions.cs`, `OptimizerBase.cs`, and `PredictionModelBuilder.cs`, and following sklearn's cross-validation workflow:

### 1. WHERE: Configuration Location

**Decision:** `ICrossValidator` configuration lives in `PredictionModelBuilder` as a private field.

**WHY:**
- Cross-validation is a **high-level strategy** for evaluating the overall model building process
- CV often involves **multiple training runs** (one per fold)
- `PredictionModelBuilder` orchestrates the entire pipeline (preprocessing, model selection, optimization)
- `OptimizationAlgorithmOptions` is suited for **single optimization run** parameters
- Follows builder pattern: builder-level configuration for pipeline strategies

**Code Location:** `src/PredictionModelBuilder.cs` line ~48 (add new field)

```csharp
private ICrossValidator<T, TInput, TOutput>? _crossValidator;
```

**Configuration Method:** After line ~473

```csharp
/// <summary>
/// Configures the cross-validation strategy to use for model evaluation.
/// </summary>
/// <param name="validator">The cross-validation strategy to use.</param>
/// <returns>This builder instance for method chaining.</returns>
/// <remarks>
/// <b>For Beginners:</b> Cross-validation helps you get a more reliable estimate of how well your model
/// will perform on new, unseen data. Instead of just one train/test split, it performs multiple splits
/// and averages the results. This gives you a better idea of your model's true performance and helps
/// prevent overfitting to a single data split.
/// </remarks>
public IPredictionModelBuilder<T, TInput, TOutput> ConfigureCrossValidator(ICrossValidator<T, TInput, TOutput> validator)
{
    _crossValidator = validator;
    return this;
}
```

### 2. WHEN: Execution Timing

**Decision:** Cross-validation executes **instead of** the single train/validation/test split within `PredictionModelBuilder.Build(x, y)` method, if configured.

**WHY:**
- **Industry standard (sklearn):** CV is used **during model development**, not after
- CV inherently **replaces the need** for a single fixed train/val/test split
- Standard CV: split entire dataset into folds, train on subset, evaluate on remaining fold, repeat for each fold
- Each fold gets its own train/validation split
- Final model evaluation uses aggregated CV results

**Code Location:** `src/PredictionModelBuilder.cs` Build() method around line 276-287

**Existing Flow (NO CV):**
```csharp
// Line ~276: Preprocess data first
var (preprocessedX, preprocessedY, normInfo) = dataPreprocessor.PreprocessData(x, y);

// Line ~287: Split into train/val/test
var (XTrain, yTrain, XVal, yVal, XTest, yTest) = dataPreprocessor.SplitData(preprocessedX, preprocessedY);

// Optimize once
optimizationResult = optimizer.Optimize(OptimizerHelper<T, TInput, TOutput>.CreateOptimizationInputData(XTrain, yTrain, XVal, yVal, XTest, yTest));
```

**New Flow (WITH CV):**
```csharp
// Line ~276: Preprocess data first (SAME)
var (preprocessedX, preprocessedY, normInfo) = dataPreprocessor.PreprocessData(x, y);

OptimizationResult<T, TInput, TOutput> optimizationResult;
CrossValidationResult<T, TInput, TOutput>? crossValidationResult = null;
IFullModel<T, TInput, TOutput> finalModel;

if (_crossValidator != null)
{
    // NEW: Perform cross-validation
    crossValidationResult = _crossValidator.PerformCrossValidation(
        _model,                // Base model to copy for each fold
        optimizer,             // Base optimizer to copy for each fold
        dataPreprocessor,
        preprocessedX,         // ENTIRE preprocessed dataset
        preprocessedY);

    // Use best model from CV
    finalModel = crossValidationResult.BestModel ?? _model.DeepCopy();

    // Take best fold's result as primary optimization result
    optimizationResult = crossValidationResult.FoldResults.OrderByDescending(r => r.BestFitnessScore).FirstOrDefault()
                         ?? new OptimizationResult<T, TInput, TOutput>();
}
else
{
    // EXISTING: Split data and optimize once
    var (XTrain, yTrain, XVal, yVal, XTest, yTest) = dataPreprocessor.SplitData(preprocessedX, preprocessedY);
    optimizationResult = optimizer.Optimize(OptimizerHelper<T, TInput, TOutput>.CreateOptimizationInputData(XTrain, yTrain, XVal, yVal, XTest, yTest));
    finalModel = optimizationResult.BestSolution;
}
```

### 3. HOW: Interaction with Train/Val/Test Split

**Decision:** When `ICrossValidator` is configured, it takes the **entire preprocessed dataset** and internally manages its own splitting into folds. The existing `dataPreprocessor.SplitData()` call is **bypassed**.

**WHY:**
- CV **needs access to full dataset** to perform internal folding
- Each fold internally creates its own train/validation split
- CV manages data splitting independently
- No conflict with existing split logic - they're mutually exclusive

**Data Flow:**

**Without CV:**
```
Raw Data (X, y)
  → dataPreprocessor.PreprocessData() → (preprocessedX, preprocessedY)
  → dataPreprocessor.SplitData() → (XTrain, yTrain, XVal, yVal, XTest, yTest)
  → optimizer.Optimize() → OptimizationResult
```

**With CV:**
```
Raw Data (X, y)
  → dataPreprocessor.PreprocessData() → (preprocessedX, preprocessedY)
  → _crossValidator.PerformCrossValidation(preprocessedX, preprocessedY)
      → For each fold:
          → Create train/test indices for this fold
          → foldXTrain, foldYTrain, foldXVal, foldYVal from indices
          → foldModel = model.DeepCopy()
          → foldOptimizer = optimizer.DeepCopy(foldModel)
          → foldOptimizer.Optimize() → foldOptimizationResult
      → Aggregate all fold results
  → CrossValidationResult (AverageFitness, StandardDeviation, FoldResults, BestModel)
```

### 4. WHAT: Complete Implementation Specification

## Files to Create

### File 1: `src/CrossValidators/ICrossValidator.cs`

**Purpose:** Interface for cross-validation strategies

```csharp
using AiDotNet.DataProcessor;
using AiDotNet.Optimizers;

namespace AiDotNet.CrossValidators;

/// <summary>
/// Defines the interface for cross-validation strategies.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input data for the model.</typeparam>
/// <typeparam name="TOutput">The type of output data for the model.</typeparam>
public interface ICrossValidator<T, TInput, TOutput>
{
    /// <summary>
    /// Performs cross-validation on the given model and data.
    /// </summary>
    /// <param name="model">The base model to be used for each fold (a deep copy will be made for each fold).</param>
    /// <param name="optimizer">The base optimizer to be used for each fold (a deep copy will be made for each fold).</param>
    /// <param name="dataPreprocessor">The data preprocessor to use for splitting data into folds.</param>
    /// <param name="preprocessedX">The preprocessed input features for the entire dataset.</param>
    /// <param name="preprocessedY">The preprocessed output values for the entire dataset.</param>
    /// <returns>A <see cref="CrossValidationResult{T, TInput, TOutput}"/> containing the aggregated results from all folds.</returns>
    CrossValidationResult<T, TInput, TOutput> PerformCrossValidation(
        IFullModel<T, TInput, TOutput> model,
        IOptimizer<T, TInput, TOutput> optimizer,
        IDataPreprocessor<T, TInput, TOutput> dataPreprocessor,
        TInput preprocessedX,
        TOutput preprocessedY);
}
```

**WHY:**
- Interface allows multiple CV strategies (K-Fold, Stratified, Time-Series)
- Follows AiDotNet pattern: interface → base class → concrete implementations
- Generic `<T, TInput, TOutput>` maintains type safety across numeric types

### File 2: `src/CrossValidators/CrossValidationResult.cs`

**Purpose:** Data class to hold aggregated CV results

```csharp
using AiDotNet.Optimizers;

namespace AiDotNet.CrossValidators;

/// <summary>
/// Represents the aggregated results of a cross-validation process.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input data for the model.</typeparam>
/// <typeparam name="TOutput">The type of output data for the model.</typeparam>
public class CrossValidationResult<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the average fitness score across all cross-validation folds.
    /// </summary>
    public T AverageFitness { get; set; }

    /// <summary>
    /// Gets or sets the standard deviation of fitness scores across all cross-validation folds.
    /// </summary>
    public T StandardDeviation { get; set; }

    /// <summary>
    /// Gets the list of individual optimization results for each cross-validation fold.
    /// </summary>
    public List<OptimizationResult<T, TInput, TOutput>> FoldResults { get; } = new List<OptimizationResult<T, TInput, TOutput>>();

    /// <summary>
    /// Gets or sets the best model found across all cross-validation folds.
    /// This model can be used for final predictions.
    /// </summary>
    public IFullModel<T, TInput, TOutput>? BestModel { get; set; }
}
```

**WHY:**
- Stores aggregated metrics (AverageFitness, StandardDeviation) for overall performance assessment
- Preserves individual `FoldResults` for detailed analysis (per-fold performance)
- `BestModel` provides ready-to-use model for predictions
- Follows existing pattern: `OptimizationResult` stores optimization results, `CrossValidationResult` stores CV results

### File 3: `src/Extensions/ListExtensions.cs`

**Purpose:** Utility for shuffling lists (needed for random fold creation)

```csharp
namespace AiDotNet.Extensions;

/// <summary>
/// Provides extension methods for List<T>.
/// </summary>
public static class ListExtensions
{
    /// <summary>
    /// Shuffles the elements of a list randomly.
    /// </summary>
    /// <typeparam name="T">The type of elements in the list.</typeparam>
    /// <param name="list">The list to shuffle.</param>
    /// <param name="random">The random number generator to use.</param>
    public static void Shuffle<T>(this IList<T> list, Random random)
    {
        int n = list.Count;
        while (n > 1)
        {
            n--;
            int k = random.Next(n + 1);
            T value = list[k];
            list[k] = list[n];
            list[n] = value;
        }
    }
}
```

**WHY:**
- Fisher-Yates shuffle algorithm ensures unbiased random fold assignment
- Prevents fold bias (ensures each sample has equal probability of being in any fold)
- Reusable extension method for other random sampling needs

### File 4: `src/CrossValidators/StandardCrossValidator.cs`

**Purpose:** Concrete K-Fold cross-validation implementation

```csharp
using AiDotNet.DataProcessor;
using AiDotNet.Extensions;
using AiDotNet.Optimizers;
using AiDotNet.Models.Options;
using AiDotNet.NumericOperations;

namespace AiDotNet.CrossValidators;

/// <summary>
/// Implements a standard K-Fold cross-validation strategy.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input data for the model.</typeparam>
/// <typeparam name="TOutput">The type of output data for the model.</typeparam>
public class StandardCrossValidator<T, TInput, TOutput> : ICrossValidator<T, TInput, TOutput>
{
    private readonly int _numberOfFolds;
    private readonly Random _random;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="StandardCrossValidator{T, TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="numberOfFolds">The number of folds to use for cross-validation. Must be greater than 1.</param>
    public StandardCrossValidator(int numberOfFolds = 5)
    {
        if (numberOfFolds <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numberOfFolds), "Number of folds must be greater than 1.");
        }
        _numberOfFolds = numberOfFolds;
        _random = new Random();
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc/>
    public CrossValidationResult<T, TInput, TOutput> PerformCrossValidation(
        IFullModel<T, TInput, TOutput> model,
        IOptimizer<T, TInput, TOutput> optimizer,
        IDataPreprocessor<T, TInput, TOutput> dataPreprocessor,
        TInput preprocessedX,
        TOutput preprocessedY)
    {
        var foldResults = new List<OptimizationResult<T, TInput, TOutput>>();
        var fitnessScores = new List<T>();

        // Convert TInput/TOutput to Matrix/Vector for easier indexing
        var xMatrix = ConversionsHelper.ConvertToMatrix<T, TInput>(preprocessedX);
        var yVector = ConversionsHelper.ConvertToVector<T, TOutput>(preprocessedY);

        int totalSamples = xMatrix.Rows;
        if (totalSamples < _numberOfFolds)
        {
            throw new ArgumentException($"Cannot perform {_numberOfFolds}-fold cross-validation with only {totalSamples} samples. Reduce the number of folds or provide more data.");
        }

        var indices = Enumerable.Range(0, totalSamples).ToList();
        indices.Shuffle(_random); // Shuffle indices for random folds

        int foldSize = totalSamples / _numberOfFolds;

        IFullModel<T, TInput, TOutput>? bestOverallModel = null;
        T bestOverallFitness = _numOps.FromDouble(double.MinValue); // Assuming higher fitness is better

        for (int i = 0; i < _numberOfFolds; i++)
        {
            var testIndices = indices.Skip(i * foldSize).Take(foldSize).ToList();
            var trainIndices = indices.Except(testIndices).ToList();

            // Create fold-specific training and validation data
            var foldXTrain = (TInput)(object)xMatrix.GetRows(trainIndices);
            var foldYTrain = (TOutput)(object)yVector.GetItems(trainIndices);
            var foldXVal = (TInput)(object)xMatrix.GetRows(testIndices);
            var foldYVal = (TOutput)(object)yVector.GetItems(testIndices);

            // Create a fresh model and optimizer for each fold to ensure independence
            var foldModel = model.DeepCopy();
            var foldOptimizer = optimizer.DeepCopy(foldModel);

            // Perform optimization for this fold
            var optimizationInputData = OptimizerHelper<T, TInput, TOutput>.CreateOptimizationInputData(
                foldXTrain, foldYTrain,
                foldXVal, foldYVal,
                (TInput)(object)new Matrix<T>(0,0), (TOutput)(object)new Vector<T>(0) // No separate test set for folds
            );
            var foldOptimizationResult = foldOptimizer.Optimize(optimizationInputData);
            foldResults.Add(foldOptimizationResult);
            fitnessScores.Add(foldOptimizationResult.BestFitnessScore);

            // Track the best model across all folds
            if (bestOverallModel == null || foldOptimizer.FitnessCalculator.IsBetterFitness(foldOptimizationResult.BestFitnessScore, bestOverallFitness))
            {
                bestOverallFitness = foldOptimizationResult.BestFitnessScore;
                bestOverallModel = foldOptimizationResult.BestSolution.DeepCopy();
            }
        }

        // Calculate average fitness and standard deviation
        T averageFitness = _numOps.Divide(fitnessScores.Aggregate(_numOps.Add), _numOps.FromInt(fitnessScores.Count));
        T sumOfSquaredDifferences = fitnessScores.Select(f => _numOps.Multiply(_numOps.Subtract(f, averageFitness), _numOps.Subtract(f, averageFitness))).Aggregate(_numOps.Add);
        T variance = _numOps.Divide(sumOfSquaredDifferences, _numOps.FromInt(fitnessScores.Count - 1)); // Sample variance
        T standardDeviation = _numOps.Sqrt(variance);

        return new CrossValidationResult<T, TInput, TOutput>
        {
            AverageFitness = averageFitness,
            StandardDeviation = standardDeviation,
            FoldResults = foldResults,
            BestModel = bestOverallModel
        };
    }
}
```

**WHY - Algorithm Explanation:**
1. **Shuffle indices:** Randomize sample order to prevent fold bias
2. **Create folds:** Divide shuffled indices into K equal-sized groups
3. **For each fold:**
   - Use fold as validation set
   - Use remaining folds as training set
   - Create fresh model/optimizer copies (prevents state leakage between folds)
   - Optimize model on this fold's train/val split
   - Store fold's optimization result
4. **Aggregate results:**
   - Calculate average fitness across all folds
   - Calculate standard deviation (measures model stability)
   - Identify best model (highest fitness score)

**Critical Implementation Details:**
- `DeepCopy()` models and optimizers for each fold ensures independence
- Uses `INumericOperations<T>` for generic numeric type support
- Validates `totalSamples >= numberOfFolds` to prevent invalid CV
- Tracks best model for final predictions

## Files to Modify

### File 5: `src/Interfaces/IFullModel.cs`

**Change:** Add `DeepCopy()` method to interface

**Location:** After existing methods

```csharp
/// <summary>
/// Creates a deep copy of the current model instance.
/// </summary>
/// <returns>A new instance that is a deep copy of this model.</returns>
IFullModel<T, TInput, TOutput> DeepCopy();
```

**WHY:**
- Each CV fold needs **independent model instance** to prevent state leakage
- Shared model would accumulate training state across folds (WRONG)
- Deep copy ensures each fold starts with fresh, untrained model
- All concrete model classes must implement this (e.g., `NeuralNetworkModel`, `LinearRegressionModel`)

**Implementation Note for Concrete Models:**
```csharp
// Example: LinearRegressionModel.cs
public IFullModel<T, TInput, TOutput> DeepCopy()
{
    // Deep copy all internal state (weights, biases, configuration)
    return new LinearRegressionModel<T, TInput, TOutput>
    {
        Weights = this.Weights.DeepCopy(),
        Bias = this.Bias,
        // ... copy all fields
    };
}
```

### File 6: `src/Optimizers/IOptimizer.cs`

**Change:** Add `DeepCopy(IFullModel<T, TInput, TOutput> model)` method to interface

**Location:** After existing methods

```csharp
/// <summary>
/// Creates a deep copy of the current optimizer instance, associated with a new model.
/// </summary>
/// <param name="model">The new model instance that the copied optimizer will operate on.</param>
/// <returns>A new instance that is a deep copy of this optimizer.</returns>
IOptimizer<T, TInput, TOutput> DeepCopy(IFullModel<T, TInput, TOutput> model);
```

**WHY:**
- Each fold needs **independent optimizer** with its own state tracking
- Shared optimizer would mix fitness histories, iteration counts across folds (WRONG)
- New model parameter: optimizer must reference the fold's specific model copy
- Preserves optimizer configuration (options, fitness calculator, fit detector)

### File 7: `src/Optimizers/OptimizerBase.cs`

**Change:** Implement `DeepCopy()` method

**Location:** Add new method after existing methods

```csharp
/// <inheritdoc/>
public IOptimizer<T, TInput, TOutput> DeepCopy(IFullModel<T, TInput, TOutput> model)
{
    // Create a new instance of the derived optimizer type using reflection
    var constructor = GetType().GetConstructor(new[] { typeof(IFullModel<T, TInput, TOutput>), typeof(OptimizationAlgorithmOptions<T, TInput, TOutput>) });
    if (constructor == null)
    {
        throw new InvalidOperationException($"Derived optimizer type {GetType().Name} must have a constructor with parameters (IFullModel<T, TInput, TOutput> model, OptimizationAlgorithmOptions<T, TInput, TOutput> options) to support DeepCopy.");
    }

    // Create a deep copy of the options as well
    var optionsCopy = Options.DeepCopy();

    var newOptimizer = (IOptimizer<T, TInput, TOutput>)constructor.Invoke(new object[] { model, optionsCopy });

    return newOptimizer;
}
```

**WHY - Implementation Details:**
- **Reflection:** Automatically works for all derived optimizer types (NormalOptimizer, GeneticAlgorithm, ParticleSwarm, etc.)
- **Options deep copy:** Each fold gets independent fitness calculator, fit detector, model evaluator
- **Fresh state:** New optimizer instance has empty iteration history, no accumulated state from previous folds
- **Constructor pattern:** Relies on standard `(IFullModel, OptimizationAlgorithmOptions)` constructor pattern

### File 8: `src/Models/Options/OptimizationAlgorithmOptions.cs`

**Change:** Add `DeepCopy()` method

**Location:** Add new method

```csharp
/// <summary>
/// Creates a deep copy of the current options instance.
/// </summary>
/// <returns>A new instance that is a deep copy of this options object.</returns>
public OptimizationAlgorithmOptions<T, TInput, TOutput> DeepCopy()
{
    var copy = (OptimizationAlgorithmOptions<T, TInput, TOutput>)MemberwiseClone();

    // Deep copy PredictionOptions and ModelStatsOptions
    copy.PredictionOptions = PredictionOptions.DeepCopy();
    copy.ModelStatsOptions = ModelStatsOptions.DeepCopy();

    // Create new instances of interfaces to avoid shared state
    copy.ModelEvaluator = (IModelEvaluator<T, TInput, TOutput>)Activator.CreateInstance(ModelEvaluator.GetType())!;
    copy.FitDetector = (IFitDetector<T, TInput, TOutput>)Activator.CreateInstance(FitDetector.GetType())!;
    copy.FitnessCalculator = (IFitnessCalculator<T, TInput, TOutput>)Activator.CreateInstance(FitnessCalculator.GetType())!;
    copy.ModelCache = (IModelCache<T, TInput, TOutput>)Activator.CreateInstance(ModelCache.GetType())!;

    return copy;
}
```

**WHY:**
- **MemberwiseClone:** Copies value types and immutable reference types efficiently
- **Deep copy nested options:** `PredictionOptions` and `ModelStatsOptions` need independent copies
- **New interface instances:** ModelEvaluator, FitDetector, FitnessCalculator, ModelCache must be fresh instances
- **Prevents shared state bugs:** Without this, all CV folds would share same ModelCache (WRONG - causes cross-fold contamination)

### File 9: `src/Models/Options/PredictionStatsOptions.cs`

**Change:** Add `DeepCopy()` method

```csharp
public PredictionStatsOptions DeepCopy()
{
    return (PredictionStatsOptions)MemberwiseClone();
}
```

**WHY:** Simple value-type options, MemberwiseClone sufficient

### File 10: `src/Models/Options/ModelStatsOptions.cs`

**Change:** Add `DeepCopy()` method

```csharp
public ModelStatsOptions DeepCopy()
{
    return (ModelStatsOptions)MemberwiseClone();
}
```

**WHY:** Simple value-type options, MemberwiseClone sufficient

### File 11: `src/PredictionModelBuilder.cs`

**Change 1:** Add global using statements at top

```csharp
global using AiDotNet.CrossValidators;
global using AiDotNet.Extensions;
```

**Change 2:** Add private field around line ~48

```csharp
private ICrossValidator<T, TInput, TOutput>? _crossValidator;
```

**Change 3:** Add configuration method after line ~473

```csharp
/// <summary>
/// Configures the cross-validation strategy to use for model evaluation.
/// </summary>
/// <param name="validator">The cross-validation strategy to use.</param>
/// <returns>This builder instance for method chaining.</returns>
/// <remarks>
/// <b>For Beginners:</b> Cross-validation helps you get a more reliable estimate of how well your model
/// will perform on new, unseen data. Instead of just one train/test split, it performs multiple splits
/// and averages the results. This gives you a better idea of your model's true performance and helps
/// prevent overfitting to a single data split.
/// </remarks>
public IPredictionModelBuilder<T, TInput, TOutput> ConfigureCrossValidator(ICrossValidator<T, TInput, TOutput> validator)
{
    _crossValidator = validator;
    return this;
}
```

**Change 4:** Modify `Build(TInput x, TOutput y)` method around line 276-287

**Replace this:**
```csharp
// Preprocess the data
var (preprocessedX, preprocessedY, normInfo) = dataPreprocessor.PreprocessData(x, y);

// Split data and optimize
var (XTrain, yTrain, XVal, yVal, XTest, yTest) = dataPreprocessor.SplitData(preprocessedX, preprocessedY);
optimizationResult = optimizer.Optimize(OptimizerHelper<T, TInput, TOutput>.CreateOptimizationInputData(XTrain, yTrain, XVal, yVal, XTest, yTest));
```

**With this:**
```csharp
// Preprocess the data
var (preprocessedX, preprocessedY, normInfo) = dataPreprocessor.PreprocessData(x, y);

OptimizationResult<T, TInput, TOutput> optimizationResult;
CrossValidationResult<T, TInput, TOutput>? crossValidationResult = null;
IFullModel<T, TInput, TOutput> finalModel;

if (_crossValidator != null)
{
    // Perform cross-validation
    crossValidationResult = _crossValidator.PerformCrossValidation(
        _model,
        optimizer,
        dataPreprocessor,
        preprocessedX,
        preprocessedY);

    finalModel = crossValidationResult.BestModel ?? _model.DeepCopy();

    // Take best fold's result as primary optimization result
    optimizationResult = crossValidationResult.FoldResults.OrderByDescending(r => r.BestFitnessScore).FirstOrDefault()
                         ?? new OptimizationResult<T, TInput, TOutput>();
}
else
{
    // Existing flow: Split data and optimize once
    var (XTrain, yTrain, XVal, yVal, XTest, yTest) = dataPreprocessor.SplitData(preprocessedX, preprocessedY);
    optimizationResult = optimizer.Optimize(OptimizerHelper<T, TInput, TOutput>.CreateOptimizationInputData(XTrain, yTrain, XVal, yVal, XTest, yTest));
    finalModel = optimizationResult.BestSolution;
}
```

**Change 5:** Update `PredictionModelResult` constructor call at end of Build()

**Add parameter:**
```csharp
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
    crossValidationResult  // NEW PARAMETER
);
```

### File 12: `src/PredictionModelResult.cs`

**Change 1:** Add property for cross-validation results

```csharp
/// <summary>
/// Gets the results of the cross-validation process, if performed.
/// </summary>
public CrossValidationResult<T, TInput, TOutput>? CrossValidationResult { get; }
```

**Change 2:** Update meta-learning constructor

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
    CrossValidationResult = null; // Not applicable for meta-learning build
}
```

**Change 3:** Update regular build constructor

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
    CrossValidationResult<T, TInput, TOutput>? crossValidationResult = null) // NEW PARAMETER
{
    // ... existing assignments ...
    CrossValidationResult = crossValidationResult;
}
```

**Change 4:** Update `Predict()` method to use CV best model

```csharp
public TOutput Predict(TInput newData)
{
    // Use the best model from CV if available, otherwise use the model from the optimization result
    var modelToUse = CrossValidationResult?.BestModel ?? OptimizationResult.BestSolution;

    if (modelToUse == null)
    {
        throw new InvalidOperationException("No trained model available for prediction.");
    }

    // ... rest of Predict method using modelToUse ...
}
```

**WHY:** Predictions use the best model found during CV, not just the last fold's model

## Usage Example (Junior-Dev-Level)

**Without Cross-Validation (Current Default):**
```csharp
var builder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new LinearRegressionModel<double, Matrix<double>, Vector<double>>())
    .ConfigureOptimizer(new NormalOptimizer<double, Matrix<double>, Vector<double>>());

var result = builder.Build(X, y);
// Single train/val/test split, one optimization run
// result.OptimizationResult.TrainingResult.PredictionStats.Accuracy
```

**With Cross-Validation (New Feature):**
```csharp
var builder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new LinearRegressionModel<double, Matrix<double>, Vector<double>>())
    .ConfigureOptimizer(new NormalOptimizer<double, Matrix<double>, Vector<double>>())
    .ConfigureCrossValidator(new StandardCrossValidator<double, Matrix<double>, Vector<double>>(numberOfFolds: 5)); // NEW

var result = builder.Build(X, y);
// 5-fold cross-validation, 5 optimization runs (one per fold)
// result.CrossValidationResult.AverageFitness
// result.CrossValidationResult.StandardDeviation
// result.CrossValidationResult.FoldResults[0].TrainingResult.PredictionStats.Accuracy
// result.CrossValidationResult.BestModel // Use this for predictions
```

## Story Points Calculation

Based on complete architectural specification:

- **New Files:** 4 files (ICrossValidator, CrossValidationResult, ListExtensions, StandardCrossValidator) = 13 points
- **Interface Changes:** 2 files (IFullModel, IOptimizer) = 5 points
- **DeepCopy Implementations:** 5 files (OptimizerBase, OptimizationAlgorithmOptions, PredictionStatsOptions, ModelStatsOptions, concrete models) = 13 points
- **PredictionModelBuilder Integration:** 1 file with complex conditional logic = 8 points
- **PredictionModelResult Updates:** 1 file with constructor changes = 5 points
- **Testing:** Unit tests for CV logic, integration tests for Build() flow = 8 points

**Total:** 52 story points

## Acceptance Criteria

1. User can call `.ConfigureCrossValidator(new StandardCrossValidator<T, TInput, TOutput>(5))` on builder
2. When CV configured, `Build(X, y)` automatically performs K-fold CV
3. Results accessible via `result.CrossValidationResult.AverageFitness`, `.StandardDeviation`, `.FoldResults`, `.BestModel`
4. When CV not configured, existing single-split behavior unchanged
5. All metrics (ErrorStats, PredictionStats) automatically calculated for each fold
6. Predictions use best model from CV when available
7. Each fold uses independent model/optimizer copies (no state leakage)
8. All numeric types (float, double, decimal) supported via `INumericOperations<T>`
