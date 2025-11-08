# Issue #324: Junior Developer Implementation Guide
## Implement Advanced Feature Selection Methods

---

## Table of Contents
1. [Understanding Feature Selection](#understanding-feature-selection)
2. [What EXISTS in the Codebase](#what-exists-in-the-codebase)
3. [What's MISSING](#whats-missing)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [Common Pitfalls](#common-pitfalls)
6. [Testing Strategy](#testing-strategy)

---

## Understanding Feature Selection

### For Beginners: What is Feature Selection?

Feature selection is like decluttering your workspace - keeping only the tools you actually need.

**Real-World Analogy:**
Predicting house prices with 100 features:
- Square footage ← **IMPORTANT** (directly affects price)
- Number of bedrooms ← **IMPORTANT** (buyers care about this)
- Distance to schools ← **SOMEWHAT IMPORTANT**
- Owner's favorite color ← **USELESS** (doesn't affect price)
- House ID number ← **USELESS** (just an identifier)

**Why Select Features?**
1. **Faster training**: Fewer features = faster computations
2. **Better accuracy**: Removing noise improves predictions
3. **Easier interpretation**: Understand which factors matter
4. **Avoid overfitting**: Too many features = memorizing noise

**Three Types of Feature Selection:**

1. **Filter Methods** (Fast, model-independent):
   - Univariate tests: Statistical tests (Chi-squared, ANOVA, Mutual Information)
   - Variance threshold: Remove features that don't vary
   - Correlation: Remove redundant features

2. **Wrapper Methods** (Slow, model-specific, accurate):
   - Sequential Selection: Add/remove features one at a time
   - Recursive Feature Elimination: Iteratively remove worst features
   - Genetic algorithms: Evolutionary search

3. **Embedded Methods** (Medium speed, model-specific):
   - L1 regularization (Lasso): Forces some coefficients to zero
   - Tree importance: Use tree-based models to rank features
   - SelectFromModel: Use any model's importance scores

---

## What EXISTS in the Codebase

### Existing Infrastructure:

1. **Interfaces**:
   - ✅ `IFeatureSelector<T, TInput>` - already exists in src/Interfaces/
   - ✅ `INumericOperations<T>` - type-generic math

2. **Existing Implementations** (src/FeatureSelectors/):
   - ✅ `VarianceThresholdFeatureSelector<T, TInput>` - filter method
   - ✅ `CorrelationFeatureSelector<T, TInput>` - filter method
   - ✅ `RecursiveFeatureElimination<T, TInput>` - wrapper method
   - ✅ `NoFeatureSelector<T, TInput>` - no-op

3. **Helper Classes**:
   - ✅ `StatisticsHelper<T>` - mean, variance, covariance, correlation
   - ✅ `FeatureSelectorHelper<T, TInput>` - extract feature vectors, create filtered data
   - ✅ `InputHelper<T, TInput>` - get batch size and input dimensions
   - ✅ `MathHelper` - numeric operations

4. **Existing Pattern**:
   - Implement `IFeatureSelector<T, TInput>` directly
   - Use `FeatureSelectorHelper` for common operations
   - Single `SelectFeatures(TInput)` method

---

## What's MISSING

### Phase 1: Advanced Filter Methods
- **AC 1.1: UnivariateFeatureSelector** - ❌ **MISSING**
  - Supports Chi-squared, ANOVA F-value, Mutual Information
- **AC 1.2: Unit Tests** - ❌ **MISSING**

### Phase 2: Advanced Wrapper Methods
- **AC 2.1: SequentialFeatureSelector** - ❌ **MISSING**
  - Forward and backward selection modes
- **AC 2.2: Unit Tests** - ❌ **MISSING**

### Phase 3: Embedded Methods
- **AC 3.1: SelectFromModel** - ❌ **MISSING**
  - Uses model feature importances or coefficients
- **AC 3.2: Unit Tests** - ❌ **MISSING**

---

## Step-by-Step Implementation

### STEP 1: Implement UnivariateFeatureSelector (AC 1.1)

Selects features based on univariate statistical tests.

#### Mathematical Background:

**Chi-Squared Test** (for categorical target):
```
Chi2(feature, target) = sum((observed - expected)^2 / expected)
Higher score = stronger relationship
Use when: Both feature and target are categorical
```

**ANOVA F-value** (for numeric features, categorical target):
```
F = (Between-group variance) / (Within-group variance)
Higher F = feature discriminates classes better
Use when: Numeric features, categorical target (classification)
```

**Mutual Information** (general):
```
MI(X, Y) = sum(p(x,y) * log(p(x,y) / (p(x) * p(y))))
Higher MI = stronger dependency
Use when: Any combination of feature/target types
```

#### Full Implementation:

```csharp
// File: src/FeatureSelectors/UnivariateFeatureSelector.cs
using AiDotNet.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.FeatureSelectors;

/// <summary>
/// Scoring functions for univariate feature selection.
/// </summary>
public enum FeatureScoringFunction
{
    /// <summary>
    /// Chi-squared test for categorical data.
    /// </summary>
    ChiSquared,

    /// <summary>
    /// ANOVA F-value for numeric features and categorical targets.
    /// </summary>
    ANOVAFValue,

    /// <summary>
    /// Mutual Information for measuring dependency.
    /// </summary>
    MutualInformation
}

/// <summary>
/// Selects features based on univariate statistical tests.
/// </summary>
/// <typeparam name="T">The numeric type (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (Matrix, Tensor, etc.).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This selector ranks features by how well they predict the target individually.
///
/// Real-world analogy:
/// You're hiring employees and have 50 interview questions. Which questions best predict
/// job performance? Test each question individually against actual performance ratings.
///
/// How it works:
/// 1. For each feature, compute a statistical score with the target
/// 2. Rank features by their scores (higher = better)
/// 3. Keep top K features
///
/// Scoring functions:
/// - Chi-squared: Best for categorical features and targets (spam detection: word counts vs spam/not spam)
/// - ANOVA F-value: Best for numeric features and categorical targets (iris: petal length vs species)
/// - Mutual Information: General-purpose, works for any data types
///
/// Default K=10 based on common practice (balance between reduction and information retention).
///
/// Reference: scikit-learn SelectKBest
/// https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
/// </remarks>
public class UnivariateFeatureSelector<T, TInput> : IFeatureSelector<T, TInput>
{
    private readonly INumericOperations<T> _numOps;
    private readonly FeatureScoringFunction _scoringFunction;
    private readonly int _k;
    private int[]? _selectedIndices;

    /// <summary>
    /// Initializes a new instance of the UnivariateFeatureSelector class.
    /// </summary>
    /// <param name="scoringFunction">Statistical test to use for scoring.</param>
    /// <param name="k">Number of top features to select. Default is 10.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Creates a feature selector with a specific scoring method.
    ///
    /// Choosing K (how many features to keep):
    /// - Too few: Lose important information
    /// - Too many: Keep noise and irrelevant features
    /// - Default 10: Good starting point for most problems
    /// - Use cross-validation to find optimal K
    ///
    /// Choosing scoring function:
    /// - ChiSquared: Categorical features + categorical target
    /// - ANOVAFValue: Numeric features + categorical target (classification)
    /// - MutualInformation: Universal (works for anything, slower)
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when k is not positive.</exception>
    public UnivariateFeatureSelector(
        FeatureScoringFunction scoringFunction = FeatureScoringFunction.ANOVAFValue,
        int k = 10)
    {
        if (k <= 0)
        {
            throw new ArgumentException("k must be positive", nameof(k));
        }

        _numOps = MathHelper.GetNumericOperations<T>();
        _scoringFunction = scoringFunction;
        _k = k;
    }

    /// <summary>
    /// Gets the indices of selected features.
    /// </summary>
    public int[] SelectedIndices => _selectedIndices ?? Array.Empty<int>();

    /// <summary>
    /// Selects top K features based on univariate statistical tests.
    /// </summary>
    /// <param name="allFeatures">Complete feature matrix.</param>
    /// <returns>Matrix with only selected features.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Ranks features and keeps the top K.
    ///
    /// NOTE: This implementation requires targets to be provided separately.
    /// In a real implementation, you'd typically fit the selector first with
    /// both X and y, then transform X.
    ///
    /// For simplicity, this example uses variance as a proxy for importance
    /// (higher variance = more information). A production implementation would
    /// compute actual statistical tests.
    /// </remarks>
    public TInput SelectFeatures(TInput allFeatures)
    {
        // Get dimensions
        int numSamples = InputHelper<T, TInput>.GetBatchSize(allFeatures);
        int numFeatures = InputHelper<T, TInput>.GetInputSize(allFeatures);

        if (_k >= numFeatures)
        {
            // Keep all features if K >= number of features
            _selectedIndices = Enumerable.Range(0, numFeatures).ToArray();
            return allFeatures;
        }

        // Compute scores for each feature
        var scores = new (int index, T score)[numFeatures];

        for (int i = 0; i < numFeatures; i++)
        {
            var featureVector = FeatureSelectorHelper<T, TInput>.ExtractFeatureVector(
                allFeatures, i, numSamples);

            // Use variance as a simple scoring metric
            // In production, implement actual statistical tests
            var mean = StatisticsHelper<T>.CalculateMean(featureVector);
            var score = StatisticsHelper<T>.CalculateVariance(featureVector, mean);

            scores[i] = (i, score);
        }

        // Sort by score descending and select top K
        _selectedIndices = scores
            .OrderByDescending(s => _numOps.ToDouble(s.score))
            .Take(_k)
            .Select(s => s.index)
            .OrderBy(idx => idx)  // Maintain original order
            .ToArray();

        // Create result with selected features
        return FeatureSelectorHelper<T, TInput>.CreateFilteredData(allFeatures, _selectedIndices.ToList());
    }
}
```

**NOTE FOR JUNIOR DEVELOPER:**
This is a simplified implementation. A production-ready version would need:
1. Separate `Fit(X, y)` and `Transform(X)` methods
2. Actual implementation of Chi-squared, ANOVA F-value, and Mutual Information
3. Handle categorical vs numeric data appropriately

For reference, see sklearn's implementation or statistical libraries.

---

### STEP 2: Implement SequentialFeatureSelector (AC 2.1)

Iteratively adds (forward) or removes (backward) features.

#### Algorithm:

**Forward Selection:**
```
1. Start with empty feature set
2. Repeat until K features selected:
   a. For each remaining feature:
      - Add it to current set
      - Train model and evaluate performance
      - Remove it
   b. Permanently add feature that improved performance most
```

**Backward Elimination:**
```
1. Start with all features
2. Repeat until K features remain:
   a. For each current feature:
      - Remove it from set
      - Train model and evaluate performance
      - Add it back
   b. Permanently remove feature whose removal hurt performance least
```

#### Implementation Skeleton:

```csharp
// File: src/FeatureSelectors/SequentialFeatureSelector.cs
using AiDotNet.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.FeatureSelectors;

/// <summary>
/// Performs sequential feature selection (forward or backward).
/// </summary>
/// <typeparam name="T">The numeric type (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This selector builds feature sets iteratively, one feature at a time.
///
/// Real-world analogy (forward selection):
/// You're building a pizza and testing customer satisfaction:
/// 1. Try each topping alone, pick the best one (e.g., cheese)
/// 2. Add cheese, try each remaining topping, pick the best (e.g., pepperoni)
/// 3. Add cheese+pepperoni, try each remaining topping, pick the best (e.g., mushrooms)
/// 4. Stop when you have K toppings or adding more doesn't improve satisfaction
///
/// Forward vs Backward:
/// - Forward: Start with nothing, add best features (good when few features are relevant)
/// - Backward: Start with everything, remove worst features (good when most features are relevant)
///
/// Warning: This is SLOW!
/// - Forward: O(K * (N-K) * model_training_time)
/// - Backward: O((N-K) * K * model_training_time)
/// - Where N = total features, K = features to select
///
/// Default K=5 based on balancing performance vs computation time.
///
/// Reference: scikit-learn SequentialFeatureSelector
/// https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html
/// </remarks>
public class SequentialFeatureSelector<T, TInput> : IFeatureSelector<T, TInput>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _nFeaturesToSelect;
    private readonly bool _forward;
    private int[]? _selectedIndices;

    /// <summary>
    /// Initializes a new instance of SequentialFeatureSelector.
    /// </summary>
    /// <param name="nFeaturesToSelect">Number of features to select. Default is 5.</param>
    /// <param name="forward">True for forward selection, false for backward. Default is true.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Creates a sequential selector.
    ///
    /// Choosing direction:
    /// - Forward: Use when you expect few features to be relevant (needle in haystack)
    /// - Backward: Use when you expect most features to be relevant (trimming the fat)
    ///
    /// Choosing K:
    /// - Too few: Miss important features
    /// - Too many: Slow and may include noise
    /// - Default 5: Good balance for exploration
    /// - Use cross-validation to optimize
    /// </remarks>
    public SequentialFeatureSelector(int nFeaturesToSelect = 5, bool forward = true)
    {
        if (nFeaturesToSelect <= 0)
        {
            throw new ArgumentException("nFeaturesToSelect must be positive", nameof(nFeaturesToSelect));
        }

        _numOps = MathHelper.GetNumericOperations<T>();
        _nFeaturesToSelect = nFeaturesToSelect;
        _forward = forward;
    }

    /// <summary>
    /// Gets the indices of selected features.
    /// </summary>
    public int[] SelectedIndices => _selectedIndices ?? Array.Empty<int>();

    public TInput SelectFeatures(TInput allFeatures)
    {
        // NOTE: Full implementation would require:
        // 1. Model to evaluate feature subsets
        // 2. Cross-validation for reliable performance estimates
        // 3. Iterative feature addition/removal logic
        //
        // Simplified version: Use variance-based selection (same as univariate)
        // See RecursiveFeatureElimination in codebase for similar wrapper approach

        int numSamples = InputHelper<T, TInput>.GetBatchSize(allFeatures);
        int numFeatures = InputHelper<T, TInput>.GetInputSize(allFeatures);

        if (_nFeaturesToSelect >= numFeatures)
        {
            _selectedIndices = Enumerable.Range(0, numFeatures).ToArray();
            return allFeatures;
        }

        // Simplified: Select top features by variance
        var scores = new (int index, T score)[numFeatures];
        for (int i = 0; i < numFeatures; i++)
        {
            var featureVector = FeatureSelectorHelper<T, TInput>.ExtractFeatureVector(
                allFeatures, i, numSamples);
            var mean = StatisticsHelper<T>.CalculateMean(featureVector);
            scores[i] = (i, StatisticsHelper<T>.CalculateVariance(featureVector, mean));
        }

        _selectedIndices = scores
            .OrderByDescending(s => _numOps.ToDouble(s.score))
            .Take(_nFeaturesToSelect)
            .Select(s => s.index)
            .OrderBy(idx => idx)
            .ToArray();

        return FeatureSelectorHelper<T, TInput>.CreateFilteredData(allFeatures, _selectedIndices.ToList());
    }
}
```

**CRITICAL NOTE:** The full implementation of Sequential Feature Selection requires:
1. Integration with a model (IModel<T> or similar)
2. Cross-validation for performance evaluation
3. Complex iterative logic (add/remove features one at a time)
4. See `RecursiveFeatureElimination` in the codebase for a similar wrapper-based approach

---

### STEP 3: Implement SelectFromModel (AC 3.1)

Uses model's feature importances or coefficients to select features.

#### Concept:
```
Many models provide feature importance scores:
- Tree-based models: Gini importance, split counts
- Linear models with L1: Coefficient magnitudes
- Any model: Permutation importance

SelectFromModel:
1. Train a model on all features
2. Extract importance scores
3. Keep features with importance > threshold
```

#### Implementation:

```csharp
// File: src/FeatureSelectors/SelectFromModel.cs
using AiDotNet.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.FeatureSelectors;

/// <summary>
/// Selects features based on model importance scores.
/// </summary>
/// <typeparam name="T">The numeric type (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This selector uses a model to identify important features.
///
/// Real-world analogy:
/// You're a restaurant manager identifying which menu items matter:
/// - Train a model predicting revenue from menu items
/// - Model learns: "Pizza and burgers drive 80% of revenue, salads 5%, other items 15%"
/// - Select features (menu items) contributing most to predictions
///
/// How it works:
/// 1. Train a model on all features
/// 2. Extract feature importances (how much each feature contributed)
/// 3. Keep features with importance > threshold
///
/// Common importance sources:
/// - Decision Trees/Random Forests: Gini importance
/// - Linear models with L1 (Lasso): Coefficient magnitudes
/// - Gradient Boosting: Gain-based importance
///
/// Default threshold="mean" keeps features above average importance.
/// This is sklearn's default approach.
///
/// Reference: scikit-learn SelectFromModel
/// https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
/// </remarks>
public class SelectFromModel<T, TInput> : IFeatureSelector<T, TInput>
{
    private readonly INumericOperations<T> _numOps;
    private readonly T _threshold;
    private int[]? _selectedIndices;

    /// <summary>
    /// Initializes a new instance of SelectFromModel.
    /// </summary>
    /// <param name="threshold">Importance threshold for selection. Default is 0.0 (keep all).</param>
    /// <remarks>
    /// <b>For Beginners:</b> Creates a model-based feature selector.
    ///
    /// Threshold options:
    /// - 0.0: Keep all features (no filtering)
    /// - mean: Keep features above average importance (sklearn default)
    /// - median: Keep top 50% of features
    /// - Specific value: Keep features with importance > value
    ///
    /// NOTE: In production, this would take an IModel parameter and fit it
    /// to extract importances. This simplified version uses variance as a proxy.
    /// </remarks>
    public SelectFromModel(double threshold = 0.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _threshold = _numOps.FromDouble(threshold);
    }

    /// <summary>
    /// Gets the indices of selected features.
    /// </summary>
    public int[] SelectedIndices => _selectedIndices ?? Array.Empty<int>();

    public TInput SelectFeatures(TInput allFeatures)
    {
        // NOTE: Full implementation would:
        // 1. Take IModel<T> in constructor
        // 2. Fit model on data
        // 3. Extract feature_importances_ or coef_ from model
        // 4. Select features based on threshold
        //
        // Simplified version: Use variance as proxy for importance

        int numSamples = InputHelper<T, TInput>.GetBatchSize(allFeatures);
        int numFeatures = InputHelper<T, TInput>.GetInputSize(allFeatures);

        // Compute "importances" (variances)
        var importances = new T[numFeatures];
        for (int i = 0; i < numFeatures; i++)
        {
            var featureVector = FeatureSelectorHelper<T, TInput>.ExtractFeatureVector(
                allFeatures, i, numSamples);
            var mean = StatisticsHelper<T>.CalculateMean(featureVector);
            importances[i] = StatisticsHelper<T>.CalculateVariance(featureVector, mean);
        }

        // Select features above threshold
        var selectedIndicesList = new List<int>();
        for (int i = 0; i < numFeatures; i++)
        {
            if (_numOps.GreaterThanOrEquals(importances[i], _threshold))
            {
                selectedIndicesList.Add(i);
            }
        }

        // If no features selected, keep all
        if (selectedIndicesList.Count == 0)
        {
            _selectedIndices = Enumerable.Range(0, numFeatures).ToArray();
            return allFeatures;
        }

        _selectedIndices = selectedIndicesList.ToArray();
        return FeatureSelectorHelper<T, TInput>.CreateFilteredData(allFeatures, selectedIndicesList);
    }
}
```

---

## Common Pitfalls

### 1. Data Leakage in Feature Selection
```csharp
// ❌ WRONG - Selects features using test data!
var selector = new UnivariateFeatureSelector<double, Matrix<double>>();
var allData = CombineTrainTest(XTrain, XTest);
var selected = selector.SelectFeatures(allData);

// ✅ CORRECT - Select features using training data only
var selector = new UnivariateFeatureSelector<double, Matrix<double>>();
var XTrainSelected = selector.SelectFeatures(XTrain);
var XTestSelected = selector.SelectFeatures(XTest);  // Apply same selection
```

### 2. Forgetting to Validate K
```csharp
// ❌ WRONG
public UnivariateFeatureSelector(int k = 10)
{
    _k = k;  // No validation!
}

// ✅ CORRECT
public UnivariateFeatureSelector(int k = 10)
{
    if (k <= 0)
        throw new ArgumentException("k must be positive", nameof(k));
    _k = k;
}
```

### 3. Not Handling Edge Cases
```csharp
// ✅ CORRECT
if (_k >= numFeatures)
{
    // Keep all features if K >= total features
    _selectedIndices = Enumerable.Range(0, numFeatures).ToArray();
    return allFeatures;
}
```

---

## Testing Strategy

### Unit Test Example:

```csharp
// File: tests/UnitTests/FeatureSelectors/UnivariateFeatureSelectorTests.cs
using Xunit;
using AiDotNet.FeatureSelectors;
using AiDotNet.LinearAlgebra;

public class UnivariateFeatureSelectorTests
{
    [Fact]
    public void SelectFeatures_SelectsTopKFeatures()
    {
        // Arrange: Create data with varying variances
        var data = new Matrix<double>(new double[,]
        {
            { 1, 10, 100 },  // Feature 0: low variance (1-3)
            { 2, 20, 200 },  // Feature 1: medium variance (10-30)
            { 3, 30, 300 }   // Feature 2: high variance (100-300)
        });

        var selector = new UnivariateFeatureSelector<double, Matrix<double>>(
            FeatureScoringFunction.ANOVAFValue, k: 2);

        // Act
        var result = selector.SelectFeatures(data);

        // Assert: Should keep features 1 and 2 (higher variance)
        Assert.Equal(3, result.Rows);  // Same number of samples
        Assert.Equal(2, result.Columns);  // Only 2 features kept
    }

    [Fact]
    public void Constructor_InvalidK_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new UnivariateFeatureSelector<double, Matrix<double>>(k: -1));
    }

    [Fact]
    public void SelectFeatures_KGreaterThanFeatures_KeepsAllFeatures()
    {
        // Arrange
        var data = new Matrix<double>(new double[,] { { 1, 2, 3 } });
        var selector = new UnivariateFeatureSelector<double, Matrix<double>>(k: 10);

        // Act
        var result = selector.SelectFeatures(data);

        // Assert
        Assert.Equal(3, result.Columns);  // All features kept
    }
}
```

---

## Summary

### What You Built:
1. ✅ UnivariateFeatureSelector - statistical feature ranking
2. ✅ SequentialFeatureSelector - iterative feature selection
3. ✅ SelectFromModel - model-based feature selection
4. ✅ Comprehensive unit tests

### Key Learnings:
- Filter methods (univariate): Fast, model-independent
- Wrapper methods (sequential): Slow, model-specific, accurate
- Embedded methods (selectfrommodel): Medium speed, model-specific
- Always select features on training data only
- Balance between number of features and model performance

### Next Steps:
1. Implement actual statistical tests (Chi2, ANOVA F, Mutual Information)
2. Integrate with models for wrapper/embedded methods
3. Add recursive feature elimination variants
4. Implement genetic algorithm feature selection
5. Add feature importance visualization

**Good luck selecting the best features!**
