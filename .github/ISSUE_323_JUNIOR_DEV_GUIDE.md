# Issue #323: Junior Developer Implementation Guide
## Implement Foundational Data Preprocessing Techniques

---

## Table of Contents
1. [Understanding Data Preprocessing](#understanding-data-preprocessing)
2. [What EXISTS in the Codebase](#what-exists-in-the-codebase)
3. [What's MISSING - What You Need to Build](#whats-missing---what-you-need-to-build)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [Common Pitfalls](#common-pitfalls)
6. [Testing Strategy](#testing-strategy)

---

## Understanding Data Preprocessing

### For Beginners: What is Data Preprocessing?

Think of data preprocessing as preparing ingredients before cooking. Raw data is like unprepared ingredients - you wouldn't cook with them directly!

**Real-World Analogy:**
Imagine you're preparing to bake a cake:
- **Missing values**: Some ingredients are missing (like eggs) - you need to substitute or skip the recipe
- **Categorical encoding**: Recipe says "large eggs" but you need a number (large = 3, medium = 2, small = 1)
- **Feature selection**: You have 50 spices but only 5 actually matter for this recipe
- **Train/test split**: You bake multiple cakes - some for practice, some for the final taste test

**Common Preprocessing Steps:**

1. **Imputation**: Filling in missing data
   - Mean/Median: Replace missing ages with average age
   - Mode: Replace missing colors with most common color

2. **Encoding**: Converting categories to numbers
   - One-Hot: Color (red, blue, green) → 3 columns [is_red, is_blue, is_green]
   - Label: Size (small, medium, large) → numbers (0, 1, 2)

3. **Feature Selection**: Choosing which features matter
   - Variance: Remove features that never change (useless)
   - Correlation: Remove redundant features (house_sqft_meters vs house_sqft_feet)

4. **Data Splitting**: Dividing data for training and testing
   - Train (80%): Teach the model
   - Test (20%): Evaluate how well it learned

---

## What EXISTS in the Codebase

### Existing Infrastructure You Can Leverage:

1. **Interfaces** (already exist):
   - `IDataPreprocessor<T, TInput, TOutput>` - overall preprocessing
   - `INormalizer<T, TInput, TOutput>` - normalization/scaling
   - `IFeatureSelector<T, TInput>` - feature selection
   - `IOutlierRemoval<T, TInput, TOutput>` - outlier detection
   - `INumericOperations<T>` - type-generic math

2. **Existing Implementations**:
   - `DefaultDataPreprocessor<T, TInput, TOutput>` - basic preprocessing
   - `VarianceThresholdFeatureSelector<T, TInput>` - removes low-variance features
   - `CorrelationFeatureSelector<T, TInput>` - removes correlated features
   - `RecursiveFeatureElimination<T, TInput>` - iterative feature removal
   - `NoFeatureSelector<T, TInput>` - no-op selector
   - Various normalizers in src/Normalizers/

3. **Helper Classes** (src/Helpers/):
   - `MathHelper` - numeric operations
   - `StatisticsHelper<T>` - mean, median, variance, std dev
   - `MatrixHelper<T>` - matrix operations
   - `VectorHelper<T>` - vector operations
   - `InputHelper<T, TInput>` - extract dimensions
   - `FeatureSelectorHelper<T, TInput>` - feature extraction utilities

4. **Existing Linear Algebra**:
   - `Matrix<T>` - 2D data
   - `Vector<T>` - 1D data
   - `Tensor<T>` - multi-dimensional data

5. **Existing Pattern**:
   - Classes implement interfaces directly (no base classes for preprocessors)
   - Use `INumericOperations<T>` for all math
   - XML documentation with `<b>For Beginners:</b>` sections

---

## What's MISSING - What You Need to Build

### Analysis of Required Components:

### Phase 1: Missing Value Imputation
- **AC 1.1: SimpleImputer** - ❌ **MISSING** - needs implementation
  - Strategies: mean, median, most_frequent
- **AC 1.2: Unit Tests** - ❌ **MISSING**

### Phase 2: Categorical Encoding
- **AC 2.1: OneHotEncoder** - ❌ **MISSING** - needs implementation
- **AC 2.2: LabelEncoder** - ❌ **MISSING** - needs implementation
- **AC 2.3: Unit Tests** - ❌ **MISSING**

### Phase 3: Feature Selection
- **AC 3.1: VarianceThreshold** - ✅ **ALREADY EXISTS** - `VarianceThresholdFeatureSelector`
- **AC 3.2: SelectKBest** - ❌ **MISSING** - needs implementation
- **AC 3.3: Unit Tests** - ❌ **MISSING**

### Phase 4: Dataset Splitting
- **AC 4.1: TrainTestSplit** - ❌ **MISSING** - needs implementation
- **AC 4.2: Unit Tests** - ❌ **MISSING**

### NEW Interfaces Needed:
- `IImputer<T>` - for missing value imputation
- `IEncoder<T>` - for categorical encoding

---

## Step-by-Step Implementation

### STEP 0: Create Missing Interfaces

#### 0.1: Create IImputer<T> Interface

```csharp
// File: src/Interfaces/IImputer.cs
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines methods for handling missing values in datasets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> An imputer fills in missing values in your data.
///
/// Real-world example:
/// You're analyzing customer data and some ages are missing:
/// - Customer 1: age = 25
/// - Customer 2: age = ??? (missing)
/// - Customer 3: age = 35
/// - Customer 4: age = ??? (missing)
/// - Customer 5: age = 45
///
/// An imputer can:
/// - Replace ??? with mean (average) = (25 + 35 + 45) / 3 = 35
/// - Replace ??? with median (middle value) = 35
/// - Replace ??? with mode (most frequent) if you had categorical data
///
/// Why impute?
/// - Many ML algorithms can't handle missing values
/// - Removing rows with missing data wastes information
/// - Imputing is better than ignoring the problem
/// </remarks>
public interface IImputer<T>
{
    /// <summary>
    /// Learns imputation strategy from the data.
    /// </summary>
    /// <param name="X">Feature matrix with missing values (represented as NaN or special marker).</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method analyzes the data to learn what values to fill in.
    ///
    /// For example, if using "mean" strategy:
    /// - Calculates the mean of each feature (column)
    /// - Stores these means for later use
    /// - Does NOT modify the input data yet
    /// </remarks>
    void Fit(Matrix<T> X);

    /// <summary>
    /// Fills in missing values in the data.
    /// </summary>
    /// <param name="X">Feature matrix with missing values.</param>
    /// <returns>Matrix with missing values filled in.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method actually replaces missing values.
    ///
    /// Uses the strategy learned in Fit():
    /// - If mean strategy: replaces NaN with column mean
    /// - If median strategy: replaces NaN with column median
    /// - If most_frequent strategy: replaces NaN with most common value
    /// </remarks>
    Matrix<T> Transform(Matrix<T> X);

    /// <summary>
    /// Learns imputation strategy and fills in missing values in one step.
    /// </summary>
    /// <param name="X">Feature matrix with missing values.</param>
    /// <returns>Matrix with missing values filled in.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Convenience method that calls Fit() then Transform().
    ///
    /// Use this when you want to impute in one step on the same data.
    /// Use separate Fit() and Transform() when you need to apply the same
    /// imputation strategy to multiple datasets (like train/test).
    /// </remarks>
    Matrix<T> FitTransform(Matrix<T> X);
}
```

#### 0.2: Create IEncoder<T> Interface

```csharp
// File: src/Interfaces/IEncoder.cs
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines methods for encoding categorical data into numeric representations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> An encoder converts categories (text labels) into numbers.
///
/// Real-world example:
/// You have a "color" column with values: red, blue, green, blue, red
///
/// Machine learning algorithms need numbers, not text. An encoder converts this:
/// - Label Encoding: red=0, blue=1, green=2 → [0, 1, 2, 1, 0]
/// - One-Hot Encoding: Creates 3 columns:
///   - is_red:   [1, 0, 0, 0, 1]
///   - is_blue:  [0, 1, 0, 1, 0]
///   - is_green: [0, 0, 1, 0, 0]
///
/// Why encode?
/// - ML algorithms require numeric input
/// - Different encodings have different properties
/// - Label encoding implies order (red < blue < green), which may not be true
/// - One-hot encoding treats categories as independent (better for most cases)
/// </remarks>
public interface IEncoder<T>
{
    /// <summary>
    /// Learns encoding mapping from categorical data.
    /// </summary>
    /// <param name="X">Categorical data (can be strings, integers representing categories, etc.).</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method learns which categories exist and assigns them numbers/columns.
    ///
    /// For label encoding: Identifies unique values (red, blue, green) and assigns indices (0, 1, 2)
    /// For one-hot encoding: Identifies unique values and prepares to create separate columns
    /// </remarks>
    void Fit(Matrix<T> X);

    /// <summary>
    /// Encodes categorical data using learned mapping.
    /// </summary>
    /// <param name="X">Categorical data to encode.</param>
    /// <returns>Encoded numeric matrix.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method actually converts categories to numbers.
    ///
    /// Uses the mapping learned in Fit() to transform the data.
    /// </remarks>
    Matrix<T> Transform(Matrix<T> X);

    /// <summary>
    /// Learns encoding and transforms in one step.
    /// </summary>
    /// <param name="X">Categorical data to encode.</param>
    /// <returns>Encoded numeric matrix.</returns>
    Matrix<T> FitTransform(Matrix<T> X);

    /// <summary>
    /// Decodes numeric data back to original categories.
    /// </summary>
    /// <param name="X">Encoded numeric matrix.</param>
    /// <returns>Decoded categorical matrix.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Reverses the encoding to get back original categories.
    ///
    /// Useful for interpreting model predictions:
    /// - Model predicts: 1
    /// - Inverse transform: "blue"
    /// </remarks>
    Matrix<T> InverseTransform(Matrix<T> X);
}
```

---

### STEP 1: Implement SimpleImputer (AC 1.1)

SimpleImputer fills missing values using mean, median, or most_frequent strategies.

#### Mathematical Background:
```
Mean strategy: Replace NaN with average of column
Median strategy: Replace NaN with middle value of column
Most_frequent strategy: Replace NaN with mode (most common value)

Example (mean strategy):
Column: [1, NaN, 3, NaN, 5]
Mean = (1 + 3 + 5) / 3 = 3
Result: [1, 3, 3, 3, 5]
```

#### Full Implementation:

```csharp
// File: src/DataProcessor/Imputers/SimpleImputer.cs
using AiDotNet.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.DataProcessor.Imputers;

/// <summary>
/// Imputation strategies for handling missing values.
/// </summary>
public enum ImputationStrategy
{
    /// <summary>
    /// Replace missing values with the mean (average) of the column.
    /// </summary>
    Mean,

    /// <summary>
    /// Replace missing values with the median (middle value) of the column.
    /// </summary>
    Median,

    /// <summary>
    /// Replace missing values with the most frequent value in the column.
    /// </summary>
    MostFrequent
}

/// <summary>
/// Implements simple imputation for filling missing values.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> SimpleImputer fills in missing values in your dataset.
///
/// Missing data is common in real world:
/// - Sensor failures in IoT data
/// - Survey questions left blank
/// - Database fields not filled in
/// - Data corruption during transmission
///
/// This class provides three simple strategies:
/// 1. Mean: Good for normally distributed numeric data
/// 2. Median: Better when data has outliers (extreme values)
/// 3. MostFrequent: Best for categorical or heavily skewed data
///
/// Default is "mean" based on sklearn's default:
/// https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
///
/// Why mean as default?
/// - Works well for most numeric datasets
/// - Simple to understand and compute
/// - Preserves the overall distribution
/// - Fast calculation
/// </remarks>
public class SimpleImputer<T> : IImputer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly ImputationStrategy _strategy;
    private Vector<T>? _fillValues;
    private readonly T _missingValueMarker;

    /// <summary>
    /// Initializes a new instance of the SimpleImputer class.
    /// </summary>
    /// <param name="strategy">Imputation strategy. Default is Mean.</param>
    /// <param name="missingValue">Value that represents missing data. Default is NaN for double/float.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Creates an imputer with a specific strategy.
    ///
    /// Strategy choice guide:
    /// - Mean: Use when data is roughly bell-shaped (normal distribution)
    /// - Median: Use when data has outliers (very high or low values)
    /// - MostFrequent: Use for categorical data or counts
    ///
    /// The missingValue parameter tells the imputer what to look for:
    /// - For double/float: NaN (Not a Number) is standard
    /// - For custom types: Specify your own marker
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when strategy is invalid.</exception>
    public SimpleImputer(
        ImputationStrategy strategy = ImputationStrategy.Mean,
        double missingValue = double.NaN)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _strategy = strategy;
        _missingValueMarker = _numOps.FromDouble(missingValue);
    }

    /// <summary>
    /// Gets the imputation strategy being used.
    /// </summary>
    public ImputationStrategy Strategy => _strategy;

    /// <summary>
    /// Gets the learned fill values for each feature.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> After calling Fit(), this contains the values that will replace missing data.
    ///
    /// For example, if you have 3 features and use mean strategy:
    /// - FillValues[0] = mean of column 0
    /// - FillValues[1] = mean of column 1
    /// - FillValues[2] = mean of column 2
    /// </remarks>
    public Vector<T> FillValues => _fillValues ?? new Vector<T>(0);

    /// <summary>
    /// Learns imputation strategy from the data.
    /// </summary>
    /// <param name="X">Feature matrix with missing values.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Analyzes each column to determine what value should fill missing spots.
    ///
    /// Steps:
    /// 1. For each column (feature):
    ///    - Filter out missing values
    ///    - Calculate statistic (mean/median/mode) from remaining values
    ///    - Store this value for later use
    /// 2. Store all fill values for Transform() to use
    ///
    /// Example with mean strategy:
    /// Column: [1, NaN, 3, NaN, 5]
    /// Valid values: [1, 3, 5]
    /// Mean: (1 + 3 + 5) / 3 = 3
    /// Store: fillValues[columnIndex] = 3
    /// </remarks>
    public void Fit(Matrix<T> X)
    {
        int nFeatures = X.Columns;
        _fillValues = new Vector<T>(nFeatures);

        for (int j = 0; j < nFeatures; j++)
        {
            var column = X.GetColumn(j);
            var validValues = FilterMissingValues(column);

            if (validValues.Length == 0)
            {
                // All values are missing - use zero as default
                _fillValues[j] = _numOps.Zero;
                continue;
            }

            _fillValues[j] = _strategy switch
            {
                ImputationStrategy.Mean => StatisticsHelper<T>.CalculateMean(validValues),
                ImputationStrategy.Median => StatisticsHelper<T>.CalculateMedian(validValues),
                ImputationStrategy.MostFrequent => CalculateMostFrequent(validValues),
                _ => throw new InvalidOperationException($"Unknown strategy: {_strategy}")
            };
        }
    }

    /// <summary>
    /// Fills in missing values in the data.
    /// </summary>
    /// <param name="X">Feature matrix with missing values.</param>
    /// <returns>Matrix with missing values filled in.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Replaces missing values with the learned fill values.
    ///
    /// Process:
    /// 1. Create a copy of the input matrix
    /// 2. For each cell in the matrix:
    ///    - If value is missing (NaN), replace with fillValues[column]
    ///    - Otherwise, keep original value
    /// 3. Return the filled matrix
    ///
    /// Example:
    /// Input:  [1, NaN, 3]    fillValues: [2]
    /// Output: [1, 2,   3]
    /// </remarks>
    public Matrix<T> Transform(Matrix<T> X)
    {
        if (_fillValues == null)
        {
            throw new InvalidOperationException("Imputer must be fitted before transforming");
        }

        if (X.Columns != _fillValues.Length)
        {
            throw new ArgumentException($"Expected {_fillValues.Length} features, got {X.Columns}");
        }

        int nSamples = X.Rows;
        int nFeatures = X.Columns;

        var result = new Matrix<T>(nSamples, nFeatures);

        for (int i = 0; i < nSamples; i++)
        {
            for (int j = 0; j < nFeatures; j++)
            {
                var value = X[i, j];
                result[i, j] = IsMissingValue(value) ? _fillValues[j] : value;
            }
        }

        return result;
    }

    /// <summary>
    /// Learns imputation strategy and fills in missing values in one step.
    /// </summary>
    /// <param name="X">Feature matrix with missing values.</param>
    /// <returns>Matrix with missing values filled in.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Convenience method - calls Fit() then Transform().
    ///
    /// Use this when:
    /// - You want to impute data in one step
    /// - You're only working with one dataset
    ///
    /// Use separate Fit()/Transform() when:
    /// - You have train/test split
    /// - You want to apply same imputation to multiple datasets
    /// </remarks>
    public Matrix<T> FitTransform(Matrix<T> X)
    {
        Fit(X);
        return Transform(X);
    }

    /// <summary>
    /// Checks if a value is considered missing.
    /// </summary>
    private bool IsMissingValue(T value)
    {
        // For NaN, we need special handling
        if (double.IsNaN(_numOps.ToDouble(_missingValueMarker)))
        {
            return double.IsNaN(_numOps.ToDouble(value));
        }

        return _numOps.Equals(value, _missingValueMarker);
    }

    /// <summary>
    /// Filters out missing values from a vector.
    /// </summary>
    private Vector<T> FilterMissingValues(Vector<T> values)
    {
        var validValues = new List<T>();

        for (int i = 0; i < values.Length; i++)
        {
            if (!IsMissingValue(values[i]))
            {
                validValues.Add(values[i]);
            }
        }

        return new Vector<T>(validValues.ToArray());
    }

    /// <summary>
    /// Calculates the most frequent value in a vector.
    /// </summary>
    private T CalculateMostFrequent(Vector<T> values)
    {
        if (values.Length == 0)
        {
            return _numOps.Zero;
        }

        var frequencies = new Dictionary<T, int>();

        for (int i = 0; i < values.Length; i++)
        {
            var value = values[i];
            if (frequencies.ContainsKey(value))
            {
                frequencies[value]++;
            }
            else
            {
                frequencies[value] = 1;
            }
        }

        return frequencies.OrderByDescending(kvp => kvp.Value).First().Key;
    }
}
```

---

### STEP 2: Implement LabelEncoder (AC 2.2)

LabelEncoder converts categorical values to integer labels (0, 1, 2, ...).

#### Implementation:

```csharp
// File: src/DataProcessor/Encoders/LabelEncoder.cs
using AiDotNet.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.DataProcessor.Encoders;

/// <summary>
/// Encodes categorical labels as integer values (0, 1, 2, ...).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> LabelEncoder converts categories (text) into numbers.
///
/// Example:
/// Input:  ["red", "blue", "green", "blue", "red"]
/// Output: [0,     1,      2,       1,      0]
///
/// Mapping learned:
/// - red → 0
/// - blue → 1
/// - green → 2
///
/// When to use Label Encoding:
/// - When categories have a natural order (small, medium, large)
/// - For tree-based models (Decision Trees, Random Forests)
/// - When you have many categories (one-hot would create too many columns)
///
/// When NOT to use Label Encoding:
/// - When categories are independent (red, blue, green have no order)
/// - For linear models (they'll assume 2 is "bigger" than 1)
/// - Use One-Hot Encoding instead for unordered categories
///
/// This implementation assigns labels in the order categories are first seen.
/// </remarks>
public class LabelEncoder<T> : IEncoder<T>
{
    private readonly INumericOperations<T> _numOps;
    private Dictionary<T, int> _labelToIndex = new();
    private Dictionary<int, T> _indexToLabel = new();
    private int _nextIndex = 0;

    /// <summary>
    /// Initializes a new instance of the LabelEncoder class.
    /// </summary>
    public LabelEncoder()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the number of unique classes.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> How many different categories were found.
    ///
    /// If you had ["red", "blue", "green", "blue"], NumClasses = 3
    /// </remarks>
    public int NumClasses => _labelToIndex.Count;

    /// <summary>
    /// Gets the mapping from labels to indices.
    /// </summary>
    public IReadOnlyDictionary<T, int> LabelToIndex => _labelToIndex;

    /// <summary>
    /// Learns encoding mapping from categorical data.
    /// </summary>
    /// <param name="X">Matrix containing categorical values (should be single column for labels).</param>
    /// <remarks>
    /// <b>For Beginners:</b> Finds all unique categories and assigns each a number.
    ///
    /// Process:
    /// 1. Scan through all values in the data
    /// 2. When a new category is seen, assign it the next available index
    /// 3. Store mappings in both directions (label→index and index→label)
    ///
    /// Example:
    /// Data: ["red", "blue", "green", "blue", "red"]
    /// First seen: "red" → 0, then "blue" → 1, then "green" → 2
    /// </remarks>
    public void Fit(Matrix<T> X)
    {
        _labelToIndex.Clear();
        _indexToLabel.Clear();
        _nextIndex = 0;

        // Scan all values to build mapping
        for (int i = 0; i < X.Rows; i++)
        {
            for (int j = 0; j < X.Columns; j++)
            {
                var value = X[i, j];
                if (!_labelToIndex.ContainsKey(value))
                {
                    _labelToIndex[value] = _nextIndex;
                    _indexToLabel[_nextIndex] = value;
                    _nextIndex++;
                }
            }
        }
    }

    /// <summary>
    /// Encodes categorical data using learned mapping.
    /// </summary>
    /// <param name="X">Categorical data to encode.</param>
    /// <returns>Encoded integer matrix.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Replaces categories with their assigned numbers.
    ///
    /// Uses the mapping created in Fit():
    /// - "red" becomes 0
    /// - "blue" becomes 1
    /// - "green" becomes 2
    /// </remarks>
    public Matrix<T> Transform(Matrix<T> X)
    {
        if (_labelToIndex.Count == 0)
        {
            throw new InvalidOperationException("Encoder must be fitted before transforming");
        }

        var result = new Matrix<T>(X.Rows, X.Columns);

        for (int i = 0; i < X.Rows; i++)
        {
            for (int j = 0; j < X.Columns; j++)
            {
                var value = X[i, j];
                if (!_labelToIndex.ContainsKey(value))
                {
                    throw new ArgumentException($"Unknown label encountered: {value}");
                }

                var index = _labelToIndex[value];
                result[i, j] = _numOps.FromDouble(index);
            }
        }

        return result;
    }

    /// <summary>
    /// Learns encoding and transforms in one step.
    /// </summary>
    /// <param name="X">Categorical data to encode.</param>
    /// <returns>Encoded integer matrix.</returns>
    public Matrix<T> FitTransform(Matrix<T> X)
    {
        Fit(X);
        return Transform(X);
    }

    /// <summary>
    /// Decodes integer labels back to original categories.
    /// </summary>
    /// <param name="X">Encoded integer matrix.</param>
    /// <returns>Decoded categorical matrix.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Converts numbers back to original categories.
    ///
    /// Useful for interpreting predictions:
    /// - Model predicts: 1
    /// - Inverse transform: "blue"
    /// </remarks>
    public Matrix<T> InverseTransform(Matrix<T> X)
    {
        if (_indexToLabel.Count == 0)
        {
            throw new InvalidOperationException("Encoder must be fitted before inverse transforming");
        }

        var result = new Matrix<T>(X.Rows, X.Columns);

        for (int i = 0; i < X.Rows; i++)
        {
            for (int j = 0; j < X.Columns; j++)
            {
                var index = (int)_numOps.ToDouble(X[i, j]);
                if (!_indexToLabel.ContainsKey(index))
                {
                    throw new ArgumentException($"Unknown index encountered: {index}");
                }

                result[i, j] = _indexToLabel[index];
            }
        }

        return result;
    }
}
```

---

### STEP 3: Implement TrainTestSplit Utility (AC 4.1)

TrainTestSplit divides data into training and testing sets.

#### Implementation:

```csharp
// File: src/DataProcessor/Splitters/TrainTestSplit.cs
using AiDotNet.LinearAlgebra;

namespace AiDotNet.DataProcessor.Splitters;

/// <summary>
/// Utility class for splitting datasets into training and testing sets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Divides your data into two sets: one for training, one for testing.
///
/// Why split data?
/// Imagine you're studying for an exam:
/// - Training set = practice problems you use to study
/// - Test set = exam questions you've never seen before
///
/// If you test on the same problems you practiced with, you can't tell if you really learned
/// or just memorized the answers. Same with machine learning!
///
/// Common split ratios:
/// - 80/20: 80% train, 20% test (default, good for most cases)
/// - 70/30: More test data for better evaluation
/// - 90/10: When you have little data but need larger training set
///
/// The randomState parameter:
/// - null (default): Different split each time (random)
/// - Set a number: Same split every time (reproducible for debugging/testing)
///
/// Defaults based on sklearn train_test_split:
/// https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
/// </remarks>
public static class TrainTestSplit<T>
{
    /// <summary>
    /// Splits data into training and testing sets.
    /// </summary>
    /// <param name="X">Feature matrix.</param>
    /// <param name="y">Target vector.</param>
    /// <param name="testSize">Proportion of data for testing (0.0 to 1.0). Default is 0.2.</param>
    /// <param name="randomState">Random seed for reproducibility. Default is null (random).</param>
    /// <returns>Tuple of (XTrain, XTest, yTrain, yTest).</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Divides your data randomly into train and test portions.
    ///
    /// Example with 10 samples and testSize=0.2 (20%):
    /// - Training set: 8 samples (80%)
    /// - Test set: 2 samples (20%)
    ///
    /// The split is random, so different samples go to train/test each time
    /// (unless you set randomState to a specific number).
    ///
    /// Parameters explained:
    /// - X: Your feature data (like house sizes, ages, etc.)
    /// - y: What you're trying to predict (like house prices)
    /// - testSize: What fraction goes to test set (0.2 = 20%)
    /// - randomState: Set this to get the same split every time (useful for debugging)
    ///
    /// Why shuffle?
    /// - Data might be ordered (all expensive houses first, cheap ones last)
    /// - Shuffling ensures both sets are representative of full dataset
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public static (Matrix<T> XTrain, Matrix<T> XTest, Vector<T> yTrain, Vector<T> yTest)
        Split(Matrix<T> X, Vector<T> y, double testSize = 0.2, int? randomState = null)
    {
        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentException("testSize must be between 0 and 1 (exclusive)", nameof(testSize));
        }

        if (X.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples in X and y must match");
        }

        int nSamples = X.Rows;
        int nTestSamples = (int)Math.Round(nSamples * testSize);
        int nTrainSamples = nSamples - nTestSamples;

        if (nTrainSamples == 0 || nTestSamples == 0)
        {
            throw new ArgumentException($"Split results in empty set. nSamples={nSamples}, testSize={testSize}");
        }

        // Shuffle indices
        var random = randomState.HasValue ? new Random(randomState.Value) : new Random();
        var indices = Enumerable.Range(0, nSamples).ToArray();

        // Fisher-Yates shuffle
        for (int i = nSamples - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        // Split indices
        var trainIndices = indices.Take(nTrainSamples).ToArray();
        var testIndices = indices.Skip(nTrainSamples).ToArray();

        // Create train/test matrices
        var XTrain = new Matrix<T>(nTrainSamples, X.Columns);
        var XTest = new Matrix<T>(nTestSamples, X.Columns);
        var yTrain = new Vector<T>(nTrainSamples);
        var yTest = new Vector<T>(nTestSamples);

        // Fill training set
        for (int i = 0; i < nTrainSamples; i++)
        {
            var originalIndex = trainIndices[i];
            for (int j = 0; j < X.Columns; j++)
            {
                XTrain[i, j] = X[originalIndex, j];
            }
            yTrain[i] = y[originalIndex];
        }

        // Fill test set
        for (int i = 0; i < nTestSamples; i++)
        {
            var originalIndex = testIndices[i];
            for (int j = 0; j < X.Columns; j++)
            {
                XTest[i, j] = X[originalIndex, j];
            }
            yTest[i] = y[originalIndex];
        }

        return (XTrain, XTest, yTrain, yTest);
    }
}
```

---

## Common Pitfalls

### 1. NEVER Fit on Test Data
```csharp
// ❌ WRONG - Data leakage!
var imputer = new SimpleImputer<double>();
imputer.FitTransform(combinedData); // Uses test data to learn fill values!

// ✅ CORRECT
var imputer = new SimpleImputer<double>();
imputer.Fit(XTrain);  // Learn only from training data
XTrain = imputer.Transform(XTrain);
XTest = imputer.Transform(XTest);  // Apply same strategy to test
```

### 2. NEVER Use default! or default(T)
```csharp
// ❌ WRONG
T fillValue = default!;

// ✅ CORRECT
T fillValue = _numOps.Zero;
```

### 3. ALWAYS Handle Edge Cases
```csharp
// ✅ CORRECT
if (validValues.Length == 0)
{
    // All values missing - use sensible default
    _fillValues[j] = _numOps.Zero;
    continue;
}
```

### 4. ALWAYS Validate Input Dimensions
```csharp
// ✅ CORRECT
if (X.Columns != _fillValues.Length)
{
    throw new ArgumentException($"Expected {_fillValues.Length} features, got {X.Columns}");
}
```

### 5. Handle NaN Carefully
```csharp
// ✅ CORRECT - NaN requires special comparison
if (double.IsNaN(_numOps.ToDouble(_missingValueMarker)))
{
    return double.IsNaN(_numOps.ToDouble(value));
}
```

---

## Testing Strategy

### Unit Test Examples:

```csharp
// File: tests/UnitTests/DataProcessor/SimpleImputerTests.cs
using Xunit;
using AiDotNet.DataProcessor.Imputers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Tests.UnitTests.DataProcessor;

public class SimpleImputerTests
{
    [Fact]
    public void Fit_MeanStrategy_CalculatesCorrectFillValues()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1, double.NaN },
            { 2, 4 },
            { 3, 5 }
        });

        var imputer = new SimpleImputer<double>(ImputationStrategy.Mean);

        // Act
        imputer.Fit(data);

        // Assert
        Assert.Equal(2, imputer.FillValues.Length);
        Assert.Equal(2.0, imputer.FillValues[0], precision: 5);  // (1+2+3)/3 = 2
        Assert.Equal(4.5, imputer.FillValues[1], precision: 5);  // (4+5)/2 = 4.5
    }

    [Fact]
    public void Transform_MeanStrategy_FillsMissingValues()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1, double.NaN },
            { double.NaN, 4 },
            { 3, 5 }
        });

        var imputer = new SimpleImputer<double>(ImputationStrategy.Mean);
        imputer.Fit(data);

        // Act
        var result = imputer.Transform(data);

        // Assert
        Assert.Equal(1, result[0, 0]);
        Assert.Equal(4.5, result[0, 1], precision: 5);
        Assert.Equal(2, result[1, 0], precision: 5);
        Assert.Equal(4, result[1, 1]);
        Assert.Equal(3, result[2, 0]);
        Assert.Equal(5, result[2, 1]);
    }

    [Fact]
    public void Fit_MedianStrategy_CalculatesCorrectFillValues()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1 },
            { 2 },
            { 100 },  // Outlier
            { double.NaN }
        });

        var imputer = new SimpleImputer<double>(ImputationStrategy.Median);

        // Act
        imputer.Fit(data);

        // Assert
        // Median of [1, 2, 100] = 2 (middle value, ignores outlier effect)
        Assert.Equal(2.0, imputer.FillValues[0], precision: 5);
    }

    [Fact]
    public void Transform_BeforeFit_ThrowsException()
    {
        // Arrange
        var imputer = new SimpleImputer<double>();
        var data = new Matrix<double>(new double[,] { { 1 } });

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => imputer.Transform(data));
    }

    [Fact]
    public void FitTransform_CombinesFitAndTransform()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1, double.NaN },
            { 2, 4 }
        });

        var imputer = new SimpleImputer<double>(ImputationStrategy.Mean);

        // Act
        var result = imputer.FitTransform(data);

        // Assert
        Assert.Equal(4.0, result[0, 1], precision: 5);  // Filled with mean of column 1
    }
}
```

### Test Coverage Checklist:

1. **Basic Functionality**:
   - ✅ Fit calculates correct fill values
   - ✅ Transform fills missing values correctly
   - ✅ FitTransform works correctly

2. **Different Strategies**:
   - ✅ Mean strategy
   - ✅ Median strategy
   - ✅ MostFrequent strategy

3. **Edge Cases**:
   - ✅ All values missing in a column
   - ✅ No missing values
   - ✅ Single row/column
   - ✅ All NaN column

4. **Error Handling**:
   - ✅ Transform before Fit throws
   - ✅ Dimension mismatch throws
   - ✅ Invalid parameters throw

5. **Type Compatibility**:
   - ✅ Works with double
   - ✅ Works with float

---

## Summary

### What You Built:
1. ✅ 2 new interfaces (IImputer, IEncoder)
2. ✅ SimpleImputer with 3 strategies
3. ✅ LabelEncoder for categorical data
4. ✅ TrainTestSplit utility
5. ✅ Comprehensive unit tests

### Key Learnings:
- Imputation handles missing data (mean/median/mode)
- Encoding converts categories to numbers
- Always fit on training data only, then transform both train and test
- TrainTestSplit should shuffle to avoid ordering bias
- Handle NaN carefully with double.IsNaN()
- Validate inputs and check if fitted before transforming

### Next Steps:
1. Implement OneHotEncoder (creates binary columns)
2. Implement SelectKBest (statistical feature selection)
3. Add more sophisticated imputation (KNN-based, model-based)
4. Add stratified splitting (preserves class distribution)
5. Create preprocessing pipelines

**Good luck with data preprocessing!**
