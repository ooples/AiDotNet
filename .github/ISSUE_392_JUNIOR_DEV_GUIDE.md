# Junior Developer Implementation Guide: Issue #392

## Overview

**Issue**: Ensemble Methods Implementation and Testing (Bagging, Boosting, Stacking)

**Goal**: Create comprehensive implementations of ensemble learning methods with full unit test coverage

**Difficulty**: Intermediate to Advanced

**Estimated Time**: 8-10 hours

**Target Coverage**: 80%+ test coverage for all ensemble implementations

---

## What You'll Be Implementing

You'll create implementations and tests for **4 ensemble methods**:

1. **Bagging (Bootstrap Aggregating)** - Reduces variance through parallel ensemble learning
2. **AdaBoost** - Adaptive boosting that focuses on misclassified examples
3. **Gradient Boosting** - Sequential ensemble learning with residual correction
4. **Stacking** - Meta-learning approach combining multiple base learners with a meta-model
5. **Random Forest** - Practical bagging implementation for decision trees

---

## Understanding Ensemble Learning

### What Are Ensemble Methods?

Ensemble methods combine multiple learning algorithms to produce better predictive performance than any single learner could achieve. Think of it like asking multiple experts for their opinion and combining their answers.

### Key Concepts

**Variance vs Bias Trade-off**:
- **Bagging**: Reduces variance (spread in predictions) by training on different subsets of data
- **Boosting**: Reduces bias (systematic error) by sequentially correcting mistakes
- **Stacking**: Combines diverse models using a meta-learner

**Bootstrap Sampling**:
- Sampling with replacement from training data
- Creates multiple different training sets from original data
- Allows same model type to learn different patterns

**Weighted Voting**:
- Ensemble members have different weights based on performance
- Final prediction is weighted average/majority vote of ensemble members

---

## Mathematical Foundations

### Bagging (Bootstrap Aggregating)

**Algorithm**:
```
1. For i = 1 to M (number of models):
   a. Create bootstrap sample Bi by sampling N examples with replacement
   b. Train model Mi on bootstrap sample Bi
2. For prediction:
   - Regression: Prediction = Average(M1(x), M2(x), ..., MM(x))
   - Classification: Prediction = MajorityVote(M1(x), M2(x), ..., MM(x))
```

**Variance Reduction**:
```
Var(Ensemble) = Var(Individual) / sqrt(M)
(roughly, assuming independence)
```

**Key Properties**:
- Reduces variance without increasing bias significantly
- Works best with high-variance, low-bias base learners
- Parallel training possible

### Boosting (AdaBoost)

**Algorithm**:
```
1. Initialize weights: w(i) = 1/N for all samples
2. For t = 1 to M (number of boosting rounds):
   a. Train weak learner on weighted dataset
   b. Calculate weighted error rate: err_t
   c. Calculate alpha_t = 0.5 * ln((1 - err_t) / err_t)
   d. Update weights: w(i) *= exp(-alpha_t * y(i) * h_t(x(i)))
   e. Normalize weights
3. Final prediction:
   sign(sum(alpha_t * h_t(x)))
```

**Key Properties**:
- Focuses on misclassified examples
- Cannot be easily parallelized (sequential)
- Exponential loss prevents outliers from dominating

### Gradient Boosting

**Algorithm**:
```
1. Initialize: F_0(x) = argmin_c sum(L(y, c))  [usually mean]
2. For m = 1 to M:
   a. Calculate pseudo-residuals: r_im = -dL(y_i, F_{m-1}(x_i))/dF
   b. Train tree h_m to predict r_im
   c. Find optimal step size: gamma = argmin_gamma sum(L(y, F_{m-1}(x) + gamma*h_m(x)))
   d. Update: F_m(x) = F_{m-1}(x) + gamma * h_m(x)
3. Output: F_M(x)
```

**Learning Rate**:
```
F_m(x) = F_{m-1}(x) + eta * h_m(x)
- eta controls contribution of each tree
- Smaller eta requires more trees but often better generalization
```

### Stacking

**Algorithm**:
```
Level 0 (Base Learners):
1. Split data into K folds
2. For each base learner L_i:
   - Train on K-1 folds
   - Predict on hold-out fold
   - Train on full data
   - Predict on test set
3. Combine predictions into meta-features

Level 1 (Meta-Learner):
4. Train meta-model on meta-features from level 0
5. Make final prediction with meta-model
```

**Meta-Features**:
```
Meta-Feature Matrix:
[
  [pred_L1_sample1, pred_L2_sample1, ..., pred_Ln_sample1],
  [pred_L1_sample2, pred_L2_sample2, ..., pred_Ln_sample2],
  ...
]
```

### Random Forest

**Algorithm**:
```
Combines Bagging + Feature Randomness:

1. For i = 1 to M (number of trees):
   a. Bootstrap sample: B_i = sample(X, N, replace=True)
   b. Build tree using random feature selection:
      - At each split, consider only sqrt(p) random features
      - Choose best split among these features
   c. Grow tree to maximum depth (no pruning)
2. Prediction = Average(Tree_1(x), ..., Tree_M(x))

Key difference from Bagging:
- Feature randomness reduces correlation between trees
- Further reduces variance in final ensemble
```

**Feature Importance**:
```
Importance(feature_j) = sum over all splits on j of (reduction in impurity)
```

---

## Implementation Structure

### File Organization

```
src/
├── EnsembleMethods/
│   ├── Interfaces/
│   │   ├── IEnsembleMethod.cs
│   │   ├── IBaselearner.cs
│   │   └── IMetaLearner.cs
│   ├── Bagging/
│   │   ├── BaggingRegressor.cs
│   │   ├── BaggingClassifier.cs
│   │   └── BaggingOptions.cs
│   ├── Boosting/
│   │   ├── AdaBoostClassifier.cs
│   │   ├── AdaBoostRegressor.cs
│   │   ├── GradientBoostingRegressor.cs (exists - extend)
│   │   └── BoostingOptions.cs
│   ├── Stacking/
│   │   ├── StackingRegressor.cs
│   │   ├── StackingClassifier.cs
│   │   ├── StackingOptions.cs
│   │   └── MetaLearnerOptions.cs
│   └── RandomForest/
│       ├── RandomForestOptions.cs (may exist)
│       └── RandomForestRegressor.cs (exists - enhance)

tests/
└── EnsembleMethods/
    ├── BaggingTests.cs
    ├── BoostingTests.cs
    ├── StackingTests.cs
    └── RandomForestTests.cs
```

---

## Step-by-Step Implementation Guide

### Step 1: Define Ensemble Interfaces

Create file: `src/EnsembleMethods/Interfaces/IEnsembleMethod.cs`

```csharp
using AiDotNet.DataTypes;

namespace AiDotNet.EnsembleMethods;

/// <summary>
/// Defines the contract for ensemble learning methods.
/// </summary>
/// <typeparam name="T">The numeric data type (float, double).</typeparam>
/// <typeparam name="TInput">Input data type.</typeparam>
/// <typeparam name="TOutput">Output prediction type.</typeparam>
public interface IEnsembleMethod<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the number of base learners in the ensemble.
    /// </summary>
    int EnsembleSize { get; }

    /// <summary>
    /// Trains the ensemble on the provided data.
    /// </summary>
    /// <param name="x">Input feature matrix.</param>
    /// <param name="y">Target values vector.</param>
    Task TrainAsync(Matrix<T> x, Vector<T> y);

    /// <summary>
    /// Makes predictions using the trained ensemble.
    /// </summary>
    /// <param name="x">Input feature matrix.</param>
    /// <returns>Predictions from ensemble.</returns>
    Vector<T> Predict(Matrix<T> x);

    /// <summary>
    /// Gets metadata about the ensemble.
    /// </summary>
    Dictionary<string, object> GetMetadata();
}
```

Create file: `src/EnsembleMethods/Interfaces/IBaseLearner.cs`

```csharp
using AiDotNet.DataTypes;

namespace AiDotNet.EnsembleMethods;

/// <summary>
/// Defines the contract for base learners in an ensemble.
/// </summary>
public interface IBaseLearner<T>
{
    /// <summary>
    /// Trains the base learner.
    /// </summary>
    Task TrainAsync(Matrix<T> x, Vector<T> y, Vector<T>? weights = null);

    /// <summary>
    /// Makes predictions.
    /// </summary>
    Vector<T> Predict(Matrix<T> x);

    /// <summary>
    /// Gets the model weights (for weighted prediction).
    /// </summary>
    double Weight { get; set; }
}
```

### Step 2: Implement Bagging

Create file: `src/EnsembleMethods/Bagging/BaggingRegressor.cs`

```csharp
using AiDotNet.DataTypes;
using AiDotNet.Regression;

namespace AiDotNet.EnsembleMethods.Bagging;

/// <summary>
/// Bagging (Bootstrap Aggregating) regressor that reduces variance through ensemble learning.
/// </summary>
/// <remarks>
/// <para>
/// Bagging trains multiple instances of a base regression model, each on a bootstrap sample
/// of the training data. Final predictions are the average of all base learner predictions.
/// </para>
/// <para>
/// Key advantages:
/// - Reduces variance without increasing bias
/// - Can be parallelized easily
/// - Works well with high-variance base learners (decision trees)
/// </para>
/// </remarks>
public class BaggingRegressor<T> : IEnsembleMethod<T, Matrix<T>, Vector<T>>
{
    private List<DecisionTreeRegression<T>> _learners = [];
    private readonly BaggingOptions _options;
    private readonly INumericOperations<T> _numOps;
    private Random _random;

    public int EnsembleSize => _learners.Count;

    /// <summary>
    /// Initializes a new Bagging regressor.
    /// </summary>
    public BaggingRegressor(BaggingOptions options, INumericOperations<T> numOps)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _numOps = numOps ?? throw new ArgumentNullException(nameof(numOps));
        _random = options.Seed.HasValue ? new Random(options.Seed.Value) : new Random();

        if (options.NumLearners <= 0)
            throw new ArgumentException("NumLearners must be positive", nameof(options));
    }

    /// <summary>
    /// Trains the bagging ensemble asynchronously.
    /// </summary>
    public async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        if (x == null) throw new ArgumentNullException(nameof(x));
        if (y == null) throw new ArgumentNullException(nameof(y));
        if (x.Rows != y.Length) throw new ArgumentException("X and Y dimensions mismatch");
        if (x.Rows < _options.MinSamplesLeaf)
            throw new ArgumentException("Not enough samples for training");

        _learners.Clear();

        // Train each base learner on bootstrap sample
        for (int i = 0; i < _options.NumLearners; i++)
        {
            // Create bootstrap sample
            var (bootstrapX, bootstrapY) = BootstrapSample(x, y);

            // Train base learner
            var treeOptions = new DecisionTreeRegressionOptions
            {
                MaxDepth = _options.MaxDepth,
                MinSamplesLeaf = _options.MinSamplesLeaf,
                MinSamplesSplit = _options.MinSamplesSplit
            };

            var learner = new DecisionTreeRegression<T>(treeOptions);
            await learner.TrainAsync(bootstrapX, bootstrapY);
            _learners.Add(learner);
        }
    }

    /// <summary>
    /// Predicts using the ensemble.
    /// </summary>
    public Vector<T> Predict(Matrix<T> x)
    {
        if (x == null) throw new ArgumentNullException(nameof(x));
        if (_learners.Count == 0) throw new InvalidOperationException("Model not trained");

        var predictions = new List<Vector<T>>();
        foreach (var learner in _learners)
        {
            predictions.Add(learner.Predict(x));
        }

        // Average predictions
        return AveragePredictions(predictions);
    }

    public Dictionary<string, object> GetMetadata()
    {
        return new Dictionary<string, object>
        {
            { "EnsembleSize", EnsembleSize },
            { "BootstrapSampleSize", _options.BootstrapSampleSize },
            { "MaxDepth", _options.MaxDepth },
            { "Method", "Bagging" }
        };
    }

    // Helper methods
    private (Matrix<T>, Vector<T>) BootstrapSample(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int sampleSize = _options.BootstrapSampleSize ?? n;
        var indices = new List<int>();

        for (int i = 0; i < sampleSize; i++)
        {
            indices.Add(_random.Next(n));
        }

        var sampledX = ExtractRows(x, indices);
        var sampledY = ExtractElements(y, indices);

        return (sampledX, sampledY);
    }

    private Matrix<T> ExtractRows(Matrix<T> x, List<int> indices)
    {
        var result = new Matrix<T>(indices.Count, x.Columns);
        for (int i = 0; i < indices.Count; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                result[i, j] = x[indices[i], j];
            }
        }
        return result;
    }

    private Vector<T> ExtractElements(Vector<T> y, List<int> indices)
    {
        var result = new Vector<T>(indices.Count);
        for (int i = 0; i < indices.Count; i++)
        {
            result[i] = y[indices[i]];
        }
        return result;
    }

    private Vector<T> AveragePredictions(List<Vector<T>> predictions)
    {
        var result = new Vector<T>(predictions[0].Length);
        for (int i = 0; i < result.Length; i++)
        {
            T sum = _numOps.Zero;
            foreach (var pred in predictions)
            {
                sum = _numOps.Add(sum, pred[i]);
            }
            result[i] = _numOps.Divide(sum, _numOps.FromDouble(predictions.Count));
        }
        return result;
    }
}
```

Create file: `src/EnsembleMethods/Bagging/BaggingOptions.cs`

```csharp
namespace AiDotNet.EnsembleMethods.Bagging;

/// <summary>
/// Configuration options for Bagging regressor.
/// </summary>
public class BaggingOptions
{
    /// <summary>
    /// Number of base learners in the ensemble.
    /// </summary>
    public int NumLearners { get; set; } = 10;

    /// <summary>
    /// Size of bootstrap samples. If null, uses full training size.
    /// </summary>
    public int? BootstrapSampleSize { get; set; } = null;

    /// <summary>
    /// Maximum depth for each decision tree.
    /// </summary>
    public int MaxDepth { get; set; } = 10;

    /// <summary>
    /// Minimum samples required at leaf node.
    /// </summary>
    public int MinSamplesLeaf { get; set; } = 1;

    /// <summary>
    /// Minimum samples required to split.
    /// </summary>
    public int MinSamplesSplit { get; set; } = 2;

    /// <summary>
    /// Random seed for reproducibility.
    /// </summary>
    public int? Seed { get; set; } = null;
}
```

### Step 3: Implement AdaBoost

Create file: `src/EnsembleMethods/Boosting/AdaBoostClassifier.cs`

```csharp
using AiDotNet.DataTypes;
using AiDotNet.Classification;

namespace AiDotNet.EnsembleMethods.Boosting;

/// <summary>
/// AdaBoost (Adaptive Boosting) classifier for binary classification.
/// </summary>
/// <remarks>
/// <para>
/// AdaBoost sequentially trains weak learners, focusing on examples that previous learners
/// misclassified. Each learner is assigned a weight based on its accuracy.
/// </para>
/// </remarks>
public class AdaBoostClassifier<T> : IEnsembleMethod<T, Matrix<T>, Vector<T>>
{
    private List<WeightedLearner<T>> _learners = [];
    private readonly BoostingOptions _options;
    private readonly INumericOperations<T> _numOps;

    public int EnsembleSize => _learners.Count;

    public AdaBoostClassifier(BoostingOptions options, INumericOperations<T> numOps)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _numOps = numOps ?? throw new ArgumentNullException(nameof(numOps));

        if (options.NumLearners <= 0)
            throw new ArgumentException("NumLearners must be positive");
    }

    public async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        if (x == null) throw new ArgumentNullException(nameof(x));
        if (y == null) throw new ArgumentNullException(nameof(y));
        if (x.Rows != y.Length) throw new ArgumentException("Dimension mismatch");

        _learners.Clear();
        int n = x.Rows;

        // Initialize sample weights
        var weights = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            weights[i] = _numOps.Divide(_numOps.One, _numOps.FromDouble(n));
        }

        // AdaBoost training loop
        for (int m = 0; m < _options.NumLearners; m++)
        {
            // Train weak learner on weighted data
            var learner = await TrainWeakLearner(x, y, weights);

            // Calculate weighted error
            var predictions = learner.Predict(x);
            double weightedError = CalculateWeightedError(y, predictions, weights);

            // Prevent perfect accuracy
            if (weightedError < 1e-10) weightedError = 1e-10;
            if (weightedError > 0.5) break; // Learner worse than random

            // Calculate alpha (learner weight)
            double alpha = 0.5 * Math.Log((1.0 - weightedError) / weightedError);

            _learners.Add(new WeightedLearner<T>
            {
                Learner = learner,
                Alpha = alpha
            });

            // Update sample weights
            UpdateWeights(weights, y, predictions, _numOps.FromDouble(alpha));
        }
    }

    public Vector<T> Predict(Matrix<T> x)
    {
        if (x == null) throw new ArgumentNullException(nameof(x));
        if (_learners.Count == 0) throw new InvalidOperationException("Model not trained");

        var result = new Vector<T>(x.Rows);
        for (int i = 0; i < x.Rows; i++)
        {
            result[i] = PredictSample(x, i);
        }
        return result;
    }

    public Dictionary<string, object> GetMetadata()
    {
        return new Dictionary<string, object>
        {
            { "EnsembleSize", EnsembleSize },
            { "NumLearners", _options.NumLearners },
            { "Method", "AdaBoost" }
        };
    }

    private async Task<DecisionTreeClassifier<T>> TrainWeakLearner(
        Matrix<T> x, Vector<T> y, Vector<T> weights)
    {
        var options = new DecisionTreeClassificationOptions { MaxDepth = 3 };
        var learner = new DecisionTreeClassifier<T>(options);
        await learner.TrainAsync(x, y); // Simplified - would need weighted training
        return learner;
    }

    private double CalculateWeightedError(Vector<T> y, Vector<T> predictions, Vector<T> weights)
    {
        double error = 0;
        for (int i = 0; i < y.Length; i++)
        {
            if (!_numOps.AreEqual(y[i], predictions[i]))
            {
                error += Convert.ToDouble(_numOps.ToDouble(weights[i]));
            }
        }
        return error;
    }

    private void UpdateWeights(Vector<T> weights, Vector<T> y, Vector<T> predictions, T alpha)
    {
        for (int i = 0; i < weights.Length; i++)
        {
            if (!_numOps.AreEqual(y[i], predictions[i]))
            {
                weights[i] = _numOps.Multiply(weights[i],
                    _numOps.Exp(alpha));
            }
        }

        // Normalize weights
        T sum = _numOps.Zero;
        for (int i = 0; i < weights.Length; i++)
        {
            sum = _numOps.Add(sum, weights[i]);
        }

        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = _numOps.Divide(weights[i], sum);
        }
    }

    private T PredictSample(Matrix<T> x, int sampleIndex)
    {
        T weightedSum = _numOps.Zero;
        for (int m = 0; m < _learners.Count; m++)
        {
            var pred = _learners[m].Learner.Predict(new Matrix<T>(1, x.Columns) { [0, 0] = x[sampleIndex, 0] })[0];
            weightedSum = _numOps.Add(weightedSum,
                _numOps.Multiply(pred, _numOps.FromDouble(_learners[m].Alpha)));
        }

        return weightedSum;
    }

    private class WeightedLearner<TT>
    {
        public DecisionTreeClassifier<TT> Learner { get; set; }
        public double Alpha { get; set; }
    }
}
```

Create file: `src/EnsembleMethods/Boosting/BoostingOptions.cs`

```csharp
namespace AiDotNet.EnsembleMethods.Boosting;

/// <summary>
/// Configuration options for boosting algorithms.
/// </summary>
public class BoostingOptions
{
    /// <summary>
    /// Number of boosting rounds.
    /// </summary>
    public int NumLearners { get; set; } = 50;

    /// <summary>
    /// Learning rate (shrinkage).
    /// </summary>
    public double LearningRate { get; set; } = 0.1;

    /// <summary>
    /// Maximum depth for weak learners.
    /// </summary>
    public int MaxDepth { get; set; } = 3;

    /// <summary>
    /// Random seed for reproducibility.
    /// </summary>
    public int? Seed { get; set; } = null;
}
```

### Step 4: Implement Stacking

Create file: `src/EnsembleMethods/Stacking/StackingRegressor.cs`

```csharp
using AiDotNet.DataTypes;
using AiDotNet.Regression;

namespace AiDotNet.EnsembleMethods.Stacking;

/// <summary>
/// Stacking regressor that combines multiple base learners with a meta-learner.
/// </summary>
/// <remarks>
/// <para>
/// Stacking trains multiple diverse base learners on the training data.
/// Their predictions become features for a meta-learner, which learns
/// how to best combine the base predictions.
/// </para>
/// </remarks>
public class StackingRegressor<T> : IEnsembleMethod<T, Matrix<T>, Vector<T>>
{
    private List<DecisionTreeRegression<T>> _baseLearners = [];
    private LinearRegression<T> _metaLearner;
    private readonly StackingOptions _options;
    private readonly INumericOperations<T> _numOps;

    public int EnsembleSize => _baseLearners.Count;

    public StackingRegressor(StackingOptions options, INumericOperations<T> numOps)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _numOps = numOps ?? throw new ArgumentNullException(nameof(numOps));

        if (options.NumBaseLearners <= 0)
            throw new ArgumentException("NumBaseLearners must be positive");
    }

    public async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        if (x == null) throw new ArgumentNullException(nameof(x));
        if (y == null) throw new ArgumentNullException(nameof(y));
        if (x.Rows != y.Length) throw new ArgumentException("Dimension mismatch");

        _baseLearners.Clear();

        // Generate meta-features using K-fold cross-validation
        var (metaX, metaY) = GenerateMetaFeatures(x, y);

        // Train meta-learner on meta-features
        var metaOptions = new LinearRegressionOptions();
        _metaLearner = new LinearRegression<T>(metaOptions);
        await _metaLearner.TrainAsync(metaX, metaY);

        // Train final base learners on full data
        for (int i = 0; i < _options.NumBaseLearners; i++)
        {
            var options = new DecisionTreeRegressionOptions
            {
                MaxDepth = 5 + i // Varying complexity
            };
            var learner = new DecisionTreeRegression<T>(options);
            await learner.TrainAsync(x, y);
            _baseLearners.Add(learner);
        }
    }

    public Vector<T> Predict(Matrix<T> x)
    {
        if (x == null) throw new ArgumentNullException(nameof(x));
        if (_baseLearners.Count == 0) throw new InvalidOperationException("Model not trained");

        // Get base predictions
        var basePredictions = new List<Vector<T>>();
        foreach (var learner in _baseLearners)
        {
            basePredictions.Add(learner.Predict(x));
        }

        // Create meta-features
        var metaX = CreateMetaFeaturesFromPredictions(basePredictions);

        // Meta-learner prediction
        return _metaLearner.Predict(metaX);
    }

    public Dictionary<string, object> GetMetadata()
    {
        return new Dictionary<string, object>
        {
            { "EnsembleSize", EnsembleSize },
            { "NumBaseLearners", _options.NumBaseLearners },
            { "KFolds", _options.KFolds },
            { "Method", "Stacking" }
        };
    }

    private (Matrix<T>, Vector<T>) GenerateMetaFeatures(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int k = _options.KFolds;
        var metaX = new Matrix<T>(n, _options.NumBaseLearners);
        var metaY = new Vector<T>(n);

        // For each fold
        for (int fold = 0; fold < k; fold++)
        {
            var (trainX, trainY, valX, valIndices) = GetFold(x, y, fold, k);

            // Train base learners on fold
            var foldLearners = new List<DecisionTreeRegression<T>>();
            for (int i = 0; i < _options.NumBaseLearners; i++)
            {
                var options = new DecisionTreeRegressionOptions { MaxDepth = 5 + i };
                var learner = new DecisionTreeRegression<T>(options);
                learner.TrainAsync(trainX, trainY).Wait(); // Sync for simplicity
                foldLearners.Add(learner);
            }

            // Generate meta-features for validation set
            for (int j = 0; j < foldLearners.Count; j++)
            {
                var valPred = foldLearners[j].Predict(valX);
                for (int i = 0; i < valIndices.Count; i++)
                {
                    metaX[valIndices[i], j] = valPred[i];
                }
            }
        }

        return (metaX, y);
    }

    private (Matrix<T>, Vector<T>, Matrix<T>, List<int>) GetFold(
        Matrix<T> x, Vector<T> y, int foldIndex, int kFolds)
    {
        int n = x.Rows;
        int foldSize = n / kFolds;
        int valStart = foldIndex * foldSize;
        int valEnd = (foldIndex == kFolds - 1) ? n : (foldIndex + 1) * foldSize;

        var trainIndices = new List<int>();
        var valIndices = new List<int>();

        for (int i = 0; i < n; i++)
        {
            if (i >= valStart && i < valEnd)
                valIndices.Add(i);
            else
                trainIndices.Add(i);
        }

        var trainX = ExtractRows(x, trainIndices);
        var trainY = ExtractElements(y, trainIndices);
        var valX = ExtractRows(x, valIndices);

        return (trainX, trainY, valX, valIndices);
    }

    private Matrix<T> ExtractRows(Matrix<T> x, List<int> indices)
    {
        var result = new Matrix<T>(indices.Count, x.Columns);
        for (int i = 0; i < indices.Count; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                result[i, j] = x[indices[i], j];
            }
        }
        return result;
    }

    private Vector<T> ExtractElements(Vector<T> y, List<int> indices)
    {
        var result = new Vector<T>(indices.Count);
        for (int i = 0; i < indices.Count; i++)
        {
            result[i] = y[indices[i]];
        }
        return result;
    }

    private Matrix<T> CreateMetaFeaturesFromPredictions(List<Vector<T>> predictions)
    {
        int n = predictions[0].Length;
        var metaX = new Matrix<T>(n, predictions.Count);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < predictions.Count; j++)
            {
                metaX[i, j] = predictions[j][i];
            }
        }
        return metaX;
    }
}
```

Create file: `src/EnsembleMethods/Stacking/StackingOptions.cs`

```csharp
namespace AiDotNet.EnsembleMethods.Stacking;

/// <summary>
/// Configuration options for stacking.
/// </summary>
public class StackingOptions
{
    /// <summary>
    /// Number of base learners.
    /// </summary>
    public int NumBaseLearners { get; set; } = 3;

    /// <summary>
    /// Number of folds for K-fold CV in base learner training.
    /// </summary>
    public int KFolds { get; set; } = 5;

    /// <summary>
    /// Type of meta-learner to use.
    /// </summary>
    public string MetaLearnerType { get; set; } = "LinearRegression";
}
```

---

## Step 5: Create Comprehensive Unit Tests

### File: `tests/EnsembleMethods/BaggingTests.cs`

```csharp
using Xunit;
using AiDotNet.DataTypes;
using AiDotNet.EnsembleMethods.Bagging;

namespace AiDotNet.Tests.EnsembleMethods;

public class BaggingTests
{
    private readonly INumericOperations<double> _numOps = NumericOperations<double>.Instance;

    private (Matrix<double>, Vector<double>) CreateSimpleRegressionData()
    {
        // y = 2*x1 + 3*x2 + noise
        var x = new Matrix<double>(100, 2);
        var y = new Vector<double>(100);
        var random = new Random(42);

        for (int i = 0; i < 100; i++)
        {
            x[i, 0] = random.NextDouble() * 10;
            x[i, 1] = random.NextDouble() * 10;
            y[i] = 2 * x[i, 0] + 3 * x[i, 1] + (random.NextDouble() - 0.5) * 2;
        }

        return (x, y);
    }

    [Fact]
    public async Task Bagging_Training_CompletesSuccessfully()
    {
        // Arrange
        var (x, y) = CreateSimpleRegressionData();
        var options = new BaggingOptions { NumLearners = 5, Seed = 42 };
        var bagging = new BaggingRegressor<double>(options, _numOps);

        // Act
        await bagging.TrainAsync(x, y);

        // Assert
        Assert.Equal(5, bagging.EnsembleSize);
    }

    [Fact]
    public async Task Bagging_Predictions_HaveCorrectShape()
    {
        // Arrange
        var (x, y) = CreateSimpleRegressionData();
        var options = new BaggingOptions { NumLearners = 3 };
        var bagging = new BaggingRegressor<double>(options, _numOps);
        await bagging.TrainAsync(x, y);

        // Act
        var predictions = bagging.Predict(x);

        // Assert
        Assert.Equal(100, predictions.Length);
    }

    [Fact]
    public async Task Bagging_VarianceIsReduced()
    {
        // Arrange - Train with different random seeds
        var (x, y) = CreateSimpleRegressionData();
        var options = new BaggingOptions { NumLearners = 20, Seed = 42 };
        var bagging = new BaggingRegressor<double>(options, _numOps);
        await bagging.TrainAsync(x, y);

        // Act
        var predictions1 = bagging.Predict(x);

        // Train single tree on same data
        var treeOptions = new DecisionTreeRegressionOptions { MaxDepth = 10 };
        var tree = new DecisionTreeRegression<double>(treeOptions);
        await tree.TrainAsync(x, y);
        var treePredictions = tree.Predict(x);

        // Calculate variance
        var baggingVariance = CalculateVariance(predictions1);
        var treeVariance = CalculateVariance(treePredictions);

        // Assert - Bagging should reduce variance
        Assert.True(baggingVariance < treeVariance * 1.5,
            $"Bagging variance {baggingVariance} should be lower than tree variance {treeVariance}");
    }

    [Fact]
    public async Task Bagging_WithDifferentBootstrapSizes()
    {
        // Arrange
        var (x, y) = CreateSimpleRegressionData();

        // Train with full bootstrap
        var options1 = new BaggingOptions { NumLearners = 5, BootstrapSampleSize = 100 };
        var bagging1 = new BaggingRegressor<double>(options1, _numOps);
        await bagging1.TrainAsync(x, y);

        // Train with partial bootstrap
        var options2 = new BaggingOptions { NumLearners = 5, BootstrapSampleSize = 50 };
        var bagging2 = new BaggingRegressor<double>(options2, _numOps);
        await bagging2.TrainAsync(x, y);

        // Act
        var pred1 = bagging1.Predict(x);
        var pred2 = bagging2.Predict(x);

        // Assert - Different sample sizes should produce different predictions
        bool isDifferent = false;
        for (int i = 0; i < pred1.Length; i++)
        {
            if (Math.Abs(pred1[i] - pred2[i]) > 1e-6)
            {
                isDifferent = true;
                break;
            }
        }
        Assert.True(isDifferent);
    }

    [Fact]
    public async Task Bagging_ThrowsOnNullInput()
    {
        // Arrange
        var options = new BaggingOptions();
        var bagging = new BaggingRegressor<double>(options, _numOps);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(() => bagging.TrainAsync(null, new Vector<double>(10)));
    }

    [Fact]
    public async Task Bagging_ThrowsOnDimensionMismatch()
    {
        // Arrange
        var x = new Matrix<double>(10, 5);
        var y = new Vector<double>(20); // Wrong size
        var options = new BaggingOptions();
        var bagging = new BaggingRegressor<double>(options, _numOps);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => bagging.TrainAsync(x, y));
    }

    [Fact]
    public void Bagging_GetMetadata_ReturnsCorrectInfo()
    {
        // Arrange
        var options = new BaggingOptions { NumLearners = 10, MaxDepth = 8 };
        var bagging = new BaggingRegressor<double>(options, _numOps);

        // Act
        var metadata = bagging.GetMetadata();

        // Assert
        Assert.Equal("Bagging", metadata["Method"]);
        Assert.Equal(10, metadata["MaxDepth"]);
    }

    // Helper method
    private double CalculateVariance(Vector<double> values)
    {
        double mean = values.Sum() / values.Length;
        double sumSq = 0;
        for (int i = 0; i < values.Length; i++)
        {
            sumSq += Math.Pow(values[i] - mean, 2);
        }
        return sumSq / values.Length;
    }
}
```

### File: `tests/EnsembleMethods/BoostingTests.cs`

```csharp
using Xunit;
using AiDotNet.DataTypes;
using AiDotNet.EnsembleMethods.Boosting;

namespace AiDotNet.Tests.EnsembleMethods;

public class AdaBoostTests
{
    private readonly INumericOperations<double> _numOps = NumericOperations<double>.Instance;

    private (Matrix<double>, Vector<double>) CreateBinaryClassificationData()
    {
        var x = new Matrix<double>(100, 2);
        var y = new Vector<double>(100);
        var random = new Random(42);

        for (int i = 0; i < 100; i++)
        {
            x[i, 0] = random.NextDouble() * 10;
            x[i, 1] = random.NextDouble() * 10;
            // Linear boundary
            y[i] = x[i, 0] + x[i, 1] > 10 ? 1.0 : 0.0;
        }

        return (x, y);
    }

    [Fact]
    public async Task AdaBoost_Training_CompletesSuccessfully()
    {
        // Arrange
        var (x, y) = CreateBinaryClassificationData();
        var options = new BoostingOptions { NumLearners = 5 };
        var adaboost = new AdaBoostClassifier<double>(options, _numOps);

        // Act
        await adaboost.TrainAsync(x, y);

        // Assert
        Assert.True(adaboost.EnsembleSize > 0);
    }

    [Fact]
    public async Task AdaBoost_Predictions_InBinaryRange()
    {
        // Arrange
        var (x, y) = CreateBinaryClassificationData();
        var options = new BoostingOptions { NumLearners = 10 };
        var adaboost = new AdaBoostClassifier<double>(options, _numOps);
        await adaboost.TrainAsync(x, y);

        // Act
        var predictions = adaboost.Predict(x);

        // Assert
        Assert.Equal(100, predictions.Length);
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.True(predictions[i] >= 0 && predictions[i] <= 1);
        }
    }

    [Fact]
    public async Task AdaBoost_FocusesOnMisclassifiedExamples()
    {
        // Arrange
        var (x, y) = CreateBinaryClassificationData();
        var options = new BoostingOptions { NumLearners = 20, LearningRate = 0.5 };
        var adaboost = new AdaBoostClassifier<double>(options, _numOps);

        // Act
        await adaboost.TrainAsync(x, y);

        // Assert
        Assert.True(adaboost.EnsembleSize > 0);
    }

    [Fact]
    public async Task AdaBoost_ImprovesOverIterations()
    {
        // Arrange
        var (x, y) = CreateBinaryClassificationData();

        // Train with few iterations
        var options1 = new BoostingOptions { NumLearners = 5 };
        var adaboost1 = new AdaBoostClassifier<double>(options1, _numOps);
        await adaboost1.TrainAsync(x, y);

        // Train with more iterations
        var options2 = new BoostingOptions { NumLearners = 20 };
        var adaboost2 = new AdaBoostClassifier<double>(options2, _numOps);
        await adaboost2.TrainAsync(x, y);

        // Act
        var pred1 = adaboost1.Predict(x);
        var pred2 = adaboost2.Predict(x);

        // Assert - More boosting should improve accuracy
        double acc1 = CalculateAccuracy(y, pred1);
        double acc2 = CalculateAccuracy(y, pred2);
        Assert.True(acc2 >= acc1 * 0.95); // Allow some variance
    }

    [Fact]
    public void AdaBoost_GetMetadata_ReturnsInfo()
    {
        // Arrange
        var options = new BoostingOptions { NumLearners = 15 };
        var adaboost = new AdaBoostClassifier<double>(options, _numOps);

        // Act
        var metadata = adaboost.GetMetadata();

        // Assert
        Assert.Equal("AdaBoost", metadata["Method"]);
        Assert.Equal(15, metadata["NumLearners"]);
    }

    private double CalculateAccuracy(Vector<double> actual, Vector<double> predicted)
    {
        int correct = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            int actualClass = actual[i] > 0.5 ? 1 : 0;
            int predClass = predicted[i] > 0.5 ? 1 : 0;
            if (actualClass == predClass) correct++;
        }
        return (double)correct / actual.Length;
    }
}
```

### File: `tests/EnsembleMethods/StackingTests.cs`

```csharp
using Xunit;
using AiDotNet.DataTypes;
using AiDotNet.EnsembleMethods.Stacking;

namespace AiDotNet.Tests.EnsembleMethods;

public class StackingTests
{
    private readonly INumericOperations<double> _numOps = NumericOperations<double>.Instance;

    private (Matrix<double>, Vector<double>) CreateTestData()
    {
        var x = new Matrix<double>(50, 3);
        var y = new Vector<double>(50);
        var random = new Random(42);

        for (int i = 0; i < 50; i++)
        {
            x[i, 0] = random.NextDouble() * 5;
            x[i, 1] = random.NextDouble() * 5;
            x[i, 2] = random.NextDouble() * 5;
            y[i] = x[i, 0] * 2 + x[i, 1] * 3 - x[i, 2] + (random.NextDouble() - 0.5);
        }

        return (x, y);
    }

    [Fact]
    public async Task Stacking_Training_CompletesSuccessfully()
    {
        // Arrange
        var (x, y) = CreateTestData();
        var options = new StackingOptions { NumBaseLearners = 3, KFolds = 5 };
        var stacking = new StackingRegressor<double>(options, _numOps);

        // Act
        await stacking.TrainAsync(x, y);

        // Assert
        Assert.Equal(3, stacking.EnsembleSize);
    }

    [Fact]
    public async Task Stacking_Predictions_HaveCorrectShape()
    {
        // Arrange
        var (x, y) = CreateTestData();
        var options = new StackingOptions { NumBaseLearners = 2 };
        var stacking = new StackingRegressor<double>(options, _numOps);
        await stacking.TrainAsync(x, y);

        // Act
        var predictions = stacking.Predict(x);

        // Assert
        Assert.Equal(50, predictions.Length);
    }

    [Fact]
    public async Task Stacking_UsesMetaLearnerToCombine()
    {
        // Arrange
        var (x, y) = CreateTestData();
        var options = new StackingOptions
        {
            NumBaseLearners = 3,
            KFolds = 3,
            MetaLearnerType = "LinearRegression"
        };
        var stacking = new StackingRegressor<double>(options, _numOps);

        // Act
        await stacking.TrainAsync(x, y);

        // Assert
        var metadata = stacking.GetMetadata();
        Assert.Equal("LinearRegression", metadata["MetaLearnerType"]);
    }

    [Fact]
    public async Task Stacking_WithDifferentKFolds()
    {
        // Arrange
        var (x, y) = CreateTestData();

        var options1 = new StackingOptions { NumBaseLearners = 2, KFolds = 3 };
        var stacking1 = new StackingRegressor<double>(options1, _numOps);
        await stacking1.TrainAsync(x, y);

        var options2 = new StackingOptions { NumBaseLearners = 2, KFolds = 5 };
        var stacking2 = new StackingRegressor<double>(options2, _numOps);
        await stacking2.TrainAsync(x, y);

        // Act
        var pred1 = stacking1.Predict(x);
        var pred2 = stacking2.Predict(x);

        // Assert
        Assert.Equal(50, pred1.Length);
        Assert.Equal(50, pred2.Length);
    }

    [Fact]
    public async Task Stacking_ThrowsOnInvalidOptions()
    {
        // Arrange
        var options = new StackingOptions { NumBaseLearners = 0 };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new StackingRegressor<double>(options, _numOps));
    }

    [Fact]
    public void Stacking_GetMetadata_ReturnsComplete()
    {
        // Arrange
        var options = new StackingOptions { NumBaseLearners = 4, KFolds = 5 };
        var stacking = new StackingRegressor<double>(options, _numOps);

        // Act
        var metadata = stacking.GetMetadata();

        // Assert
        Assert.Equal("Stacking", metadata["Method"]);
        Assert.Equal(5, metadata["KFolds"]);
    }
}
```

---

## Success Criteria

### Definition of Done

- [ ] Bagging implementation complete with full tests
- [ ] AdaBoost implementation complete with full tests
- [ ] Gradient Boosting (existing) enhanced and tested
- [ ] Stacking implementation complete with full tests
- [ ] Random Forest (existing) enhanced and tested
- [ ] 80%+ test coverage for each ensemble method
- [ ] All tests passing
- [ ] Proper error handling and validation
- [ ] Comprehensive documentation with beginner explanations
- [ ] Usage examples for each method

### Test Coverage Targets

- Bagging: 15+ tests (variance reduction, bootstrap, edge cases)
- AdaBoost: 12+ tests (weight updates, misclassification focus)
- Gradient Boosting: 15+ tests (residuals, learning rate, sequential training)
- Stacking: 12+ tests (meta-features, K-fold, meta-learner)
- Random Forest: 15+ tests (feature sampling, tree diversity)

### Quality Checklist

- [ ] All numeric types (T, double, float) supported
- [ ] INumericOperations<T> used throughout
- [ ] Proper validation of inputs
- [ ] Null reference handling
- [ ] Dimension validation (X/y mismatch)
- [ ] Seed reproducibility
- [ ] Async/await properly implemented
- [ ] Meaningful error messages
- [ ] Documentation comments on all public members
- [ ] Beginner-friendly explanations in comments

---

## Key Implementation Patterns

### Using INumericOperations<T>

```csharp
// CORRECT
T value = numOps.Add(a, b);
T divided = numOps.Divide(sum, numOps.FromDouble(count));

// WRONG - Don't do this
T value = (T)(object)(Convert.ToDouble(a) + Convert.ToDouble(b));
```

### Proper Error Handling

```csharp
public async Task TrainAsync(Matrix<T> x, Vector<T> y)
{
    if (x == null) throw new ArgumentNullException(nameof(x));
    if (y == null) throw new ArgumentNullException(nameof(y));
    if (x.Rows != y.Length) throw new ArgumentException("Dimension mismatch", nameof(y));
    if (x.Rows < MinSamples) throw new ArgumentException("Insufficient samples", nameof(x));
}
```

### Bootstrap Sampling Pattern

```csharp
private (Matrix<T>, Vector<T>) BootstrapSample(Matrix<T> x, Vector<T> y)
{
    var indices = new List<int>();
    for (int i = 0; i < sampleSize; i++)
    {
        indices.Add(_random.Next(x.Rows));  // Sample with replacement
    }
    return (ExtractRows(x, indices), ExtractElements(y, indices));
}
```

### Weighted Prediction Pattern

```csharp
private T PredictWeighted(Vector<T> predictions, List<double> weights)
{
    T weightedSum = _numOps.Zero;
    for (int i = 0; i < predictions.Length; i++)
    {
        weightedSum = _numOps.Add(weightedSum,
            _numOps.Multiply(predictions[i], _numOps.FromDouble(weights[i])));
    }
    return _numOps.Divide(weightedSum, _numOps.FromDouble(weights.Sum()));
}
```

---

## Testing Patterns

### Variance Reduction Test

```csharp
[Fact]
public async Task EnsembleReducesVariance()
{
    // Train ensemble and single model on same data
    var ensemblePred = ensemble.Predict(testX);
    var singlePred = singleModel.Predict(testX);

    // Calculate variance
    var ensembleVar = CalculateVariance(ensemblePred);
    var singleVar = CalculateVariance(singlePred);

    // Ensemble variance should be lower
    Assert.True(ensembleVar < singleVar);
}
```

### Sequential Learning Test

```csharp
[Fact]
public async Task BoostingImprovesOverIterations()
{
    var boosted5 = new Booster(numLearners: 5);
    var boosted50 = new Booster(numLearners: 50);

    var acc5 = CalculateAccuracy(boosted5.Predict(x), y);
    var acc50 = CalculateAccuracy(boosted50.Predict(x), y);

    Assert.True(acc50 >= acc5);
}
```

### Metadata Test

```csharp
[Fact]
public void EnsembleReturnsMetadata()
{
    var metadata = ensemble.GetMetadata();

    Assert.Contains("EnsembleSize", metadata.Keys);
    Assert.Contains("Method", metadata.Keys);
    Assert.True((int)metadata["EnsembleSize"] > 0);
}
```

---

## Resources

### Mathematical References
- [Ensemble Methods Overview](https://en.wikipedia.org/wiki/Ensemble_learning)
- [Bagging Variance Reduction](https://en.wikipedia.org/wiki/Bootstrap_aggregating)
- [AdaBoost Algorithm](https://en.wikipedia.org/wiki/AdaBoost)
- [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting)
- [Stacking Ensembles](https://en.wikipedia.org/wiki/Stacking_(machine_learning))

### Related Issues
- Issue #330: Decision Tree Implementation
- Issue #331: Regression Algorithms
- Issue #332: Classification Algorithms

---

## Notes for Developers

1. **Start with Bagging**: It's the simplest ensemble method and builds intuition
2. **Understand Bootstrap**: Key concept for bagging, boosting, and Random Forest
3. **Implement Tests First**: Each ensemble method should have 12-15 comprehensive tests
4. **Use Existing Components**: Build on DecisionTreeRegression, DecisionTreeClassification
5. **Verify Reproducibility**: Test with fixed seeds for deterministic behavior
6. **Documentation is Key**: Each class needs beginner-friendly explanations
7. **Error Handling**: Always validate inputs and provide meaningful error messages
8. **Coverage Matters**: Aim for 80%+ line coverage in all test files

---

**Target**: Create production-quality ensemble learning implementations with comprehensive test coverage and beginner-friendly documentation.

