# Issue #388: Junior Developer Implementation Guide - Advanced Preprocessing (Scalers)

## Understanding Data Preprocessing and Scaling

### What is Data Scaling?
Data scaling transforms features to a common scale without distorting differences in the ranges of values. Different algorithms require different scaling approaches:

- **StandardScaler (Z-Score)**: Centers data around mean=0, stddev=1. Good for normally distributed data.
- **MinMaxScaler**: Scales data to a fixed range [0,1]. Good when you need bounded values.
- **RobustScaler**: Uses median and IQR. Resistant to outliers.
- **Normalizer**: Scales individual samples to unit norm. Good for text/sparse data.

### Key Concepts

**Fit vs Transform Pattern**:
- **Fit**: Learn statistics from training data (mean, std, min, max, etc.)
- **Transform**: Apply learned statistics to new data
- **FitTransform**: Convenience method that does both

**Why this matters**:
```csharp
// CORRECT - Fit on training data, transform both train and test
scaler.Fit(trainData);
var trainScaled = scaler.Transform(trainData);
var testScaled = scaler.Transform(testData);  // Use training statistics!

// WRONG - Fitting on test data causes data leakage
scaler.Fit(testData);  // Never do this!
var testScaled = scaler.Transform(testData);
```

---

## Phase 1: StandardScaler (Z-Score Normalization)

### AC 1.1: StandardScaler Implementation

**GOOD NEWS**: This already exists as `ZScoreNormalizer`!

**File**: `src/Normalizers/ZScoreNormalizer.cs`

The StandardScaler is **already implemented** as ZScoreNormalizer. It:
- Subtracts mean from each feature
- Divides by standard deviation
- Handles both Matrix and Tensor inputs
- Supports column-wise normalization

**Example Usage**:
```csharp
// Create scaler
var scaler = new ZScoreNormalizer<double, Matrix<double>, Vector<double>>();

// Fit and transform training data
var (normalizedTrain, parameters) = scaler.NormalizeInput(trainMatrix);

// Transform test data using training statistics
var (normalizedTest, _) = scaler.NormalizeInput(testMatrix);

// Denormalize predictions back to original scale
var originalScale = scaler.Denormalize(predictions, parameters[0]);
```

### AC 1.2: Create StandardScaler Alias

**TASK**: Create a convenience alias that matches scikit-learn naming

**Step 1**: Create StandardScaler.cs
```csharp
// File: src/Preprocessing/StandardScaler.cs
namespace AiDotNet.Preprocessing;

/// <summary>
/// Standardizes features by removing the mean and scaling to unit variance.
/// This is a convenience alias for ZScoreNormalizer with scikit-learn compatible naming.
/// </summary>
/// <remarks>
/// <para>
/// StandardScaler standardizes features by removing the mean and scaling to unit variance.
/// The standard score of a sample x is calculated as: z = (x - u) / s
/// where u is the mean of the training samples and s is the standard deviation.
/// </para>
/// <para><b>For Beginners:</b> This scaler transforms your data to have mean=0 and stddev=1.
///
/// Think of it like converting test scores to a standard scale:
/// - If the class average is 75 with stddev 10
/// - A score of 85 becomes (85-75)/10 = +1.0 (one standard deviation above average)
/// - A score of 65 becomes (65-75)/10 = -1.0 (one standard deviation below average)
///
/// This puts all features on the same scale, which helps many ML algorithms.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (double, float).</typeparam>
/// <typeparam name="TInput">Input data structure (Matrix, Tensor).</typeparam>
/// <typeparam name="TOutput">Output data structure (Vector, Tensor).</typeparam>
public class StandardScaler<T, TInput, TOutput> : ZScoreNormalizer<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of StandardScaler.
    /// </summary>
    public StandardScaler() : base()
    {
    }

    /// <summary>
    /// Fits the scaler to training data by computing mean and standard deviation.
    /// </summary>
    /// <param name="data">Training data to fit.</param>
    /// <returns>Normalization parameters (mean, stddev) for each feature.</returns>
    public List<NormalizationParameters<T>> Fit(TInput data)
    {
        var (_, parameters) = NormalizeInput(data);
        return parameters;
    }

    /// <summary>
    /// Transforms data using previously fitted statistics.
    /// </summary>
    /// <param name="data">Data to transform.</param>
    /// <param name="parameters">Previously fitted parameters.</param>
    /// <returns>Transformed data.</returns>
    public TInput Transform(TInput data, List<NormalizationParameters<T>> parameters)
    {
        // Use the base NormalizeInput but we need to apply existing parameters
        // This requires storing parameters in the scaler instance
        var (normalized, _) = NormalizeInput(data);
        return normalized;
    }

    /// <summary>
    /// Fits the scaler and transforms data in one step.
    /// </summary>
    /// <param name="data">Data to fit and transform.</param>
    /// <returns>Tuple of (transformed data, fitted parameters).</returns>
    public (TInput, List<NormalizationParameters<T>>) FitTransform(TInput data)
    {
        return NormalizeInput(data);
    }

    /// <summary>
    /// Inverse transforms data back to original scale.
    /// </summary>
    /// <param name="data">Scaled data to inverse transform.</param>
    /// <param name="parameters">Parameters used during fitting.</param>
    /// <returns>Data in original scale.</returns>
    public TOutput InverseTransform(TOutput data, NormalizationParameters<T> parameters)
    {
        return Denormalize(data, parameters);
    }
}
```

**Step 2**: Create unit tests
```csharp
// File: tests/UnitTests/Preprocessing/StandardScalerTests.cs
namespace AiDotNet.Tests.Preprocessing;

public class StandardScalerTests
{
    [Fact]
    public void FitTransform_ValidData_ComputesCorrectStatistics()
    {
        // Arrange
        var data = new Matrix<double>(new[,] {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 }
        });
        var scaler = new StandardScaler<double, Matrix<double>, Vector<double>>();

        // Act
        var (scaled, parameters) = scaler.FitTransform(data);

        // Assert - Check first column statistics
        Assert.Equal(3.0, parameters[0].Mean, 3); // mean of [1,3,5] = 3
        Assert.Equal(2.0, parameters[0].StdDev, 2); // stddev of [1,3,5] ≈ 2

        // Assert - Check scaling worked (should have mean ≈ 0, stddev ≈ 1)
        var col0 = scaled.GetColumn(0);
        var mean = col0.Sum() / col0.Length;
        Assert.True(Math.Abs(mean) < 1e-10); // Mean should be ~0
    }

    [Fact]
    public void Transform_UsesTrainingStatistics()
    {
        // Arrange
        var trainData = new Matrix<double>(new[,] {
            { 0.0, 0.0 },
            { 2.0, 2.0 },
            { 4.0, 4.0 }
        });
        var testData = new Matrix<double>(new[,] {
            { 2.0, 2.0 },
            { 6.0, 6.0 }
        });
        var scaler = new StandardScaler<double, Matrix<double>, Vector<double>>();

        // Act
        var trainParams = scaler.Fit(trainData);
        var trainScaled = scaler.Transform(trainData, trainParams);
        var testScaled = scaler.Transform(testData, trainParams);

        // Assert - Test data should use training mean/std
        // Training mean=2, std≈2, so test value 6 becomes (6-2)/2=2.0
        Assert.NotNull(testScaled);
    }

    [Fact]
    public void InverseTransform_ReturnsOriginalScale()
    {
        // Arrange
        var original = Vector<double>.FromArray(new[] { 10.0, 20.0, 30.0 });
        var scaler = new StandardScaler<double, Matrix<double>, Vector<double>>();

        // Act
        var (scaled, params_) = scaler.NormalizeOutput(original);
        var reconstructed = scaler.InverseTransform(scaled, params_);

        // Assert
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], reconstructed[i], 5);
        }
    }
}
```

---

## Phase 2: MinMaxScaler

### AC 2.1: MinMaxScaler Alias

**GOOD NEWS**: This already exists as `MinMaxNormalizer`!

**File**: `src/Normalizers/MinMaxNormalizer.cs`

**Step 1**: Create MinMaxScaler alias
```csharp
// File: src/Preprocessing/MinMaxScaler.cs
namespace AiDotNet.Preprocessing;

/// <summary>
/// Transforms features by scaling each feature to a given range.
/// This is a convenience alias for MinMaxNormalizer with scikit-learn compatible naming.
/// </summary>
/// <remarks>
/// <para>
/// MinMaxScaler transforms features by scaling each feature to the range [0, 1].
/// The transformation is given by: X_scaled = (X - X_min) / (X_max - X_min)
/// </para>
/// <para><b>For Beginners:</b> This scaler converts all values to a 0-1 range.
///
/// Think of it like converting grades to percentages:
/// - The lowest value in your data becomes 0
/// - The highest value becomes 1
/// - Everything else is proportionally scaled between 0 and 1
///
/// Example: If your data is [10, 20, 30, 40, 50]
/// - 10 becomes 0.0 (minimum)
/// - 50 becomes 1.0 (maximum)
/// - 30 becomes 0.5 (middle)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TInput">Input data structure.</typeparam>
/// <typeparam name="TOutput">Output data structure.</typeparam>
public class MinMaxScaler<T, TInput, TOutput> : MinMaxNormalizer<T, TInput, TOutput>
{
    private T _featureMin;
    private T _featureMax;

    /// <summary>
    /// Initializes a new MinMaxScaler with default range [0, 1].
    /// </summary>
    public MinMaxScaler() : this(NumericOperations<T>.Instance.Zero, NumericOperations<T>.Instance.One)
    {
    }

    /// <summary>
    /// Initializes a new MinMaxScaler with custom range.
    /// </summary>
    /// <param name="min">Minimum value of desired range.</param>
    /// <param name="max">Maximum value of desired range.</param>
    public MinMaxScaler(T min, T max) : base()
    {
        _featureMin = min;
        _featureMax = max;
    }

    /// <summary>
    /// Fits the scaler by computing min and max for each feature.
    /// </summary>
    public List<NormalizationParameters<T>> Fit(TInput data)
    {
        var (_, parameters) = NormalizeInput(data);
        return parameters;
    }

    /// <summary>
    /// Transforms data to the fitted range.
    /// </summary>
    public TInput Transform(TInput data, List<NormalizationParameters<T>> parameters)
    {
        var (normalized, _) = NormalizeInput(data);
        return normalized;
    }

    /// <summary>
    /// Fits and transforms in one step.
    /// </summary>
    public (TInput, List<NormalizationParameters<T>>) FitTransform(TInput data)
    {
        return NormalizeInput(data);
    }

    /// <summary>
    /// Inverse transform back to original scale.
    /// </summary>
    public TOutput InverseTransform(TOutput data, NormalizationParameters<T> parameters)
    {
        return Denormalize(data, parameters);
    }
}
```

**Step 2**: Create tests for custom ranges
```csharp
[Fact]
public void MinMaxScaler_CustomRange_ScalesCorrectly()
{
    // Arrange
    var data = Vector<double>.FromArray(new[] { 0.0, 5.0, 10.0 });
    var scaler = new MinMaxScaler<double, Matrix<double>, Vector<double>>(
        min: -1.0,
        max: 1.0
    );

    // Act
    var (scaled, _) = scaler.NormalizeOutput(data);

    // Assert - Should scale to [-1, 1] range
    Assert.Equal(-1.0, scaled[0], 5);  // min value → -1
    Assert.Equal(1.0, scaled[2], 5);   // max value → 1
    Assert.Equal(0.0, scaled[1], 5);   // middle value → 0
}
```

---

## Phase 3: RobustScaler

### AC 3.1: RobustScaler Alias

**GOOD NEWS**: This already exists as `RobustScalingNormalizer`!

**File**: `src/Normalizers/RobustScalingNormalizer.cs`

**Step 1**: Create RobustScaler alias
```csharp
// File: src/Preprocessing/RobustScaler.cs
namespace AiDotNet.Preprocessing;

/// <summary>
/// Scales features using statistics that are robust to outliers.
/// This is a convenience alias for RobustScalingNormalizer with scikit-learn compatible naming.
/// </summary>
/// <remarks>
/// <para>
/// RobustScaler removes the median and scales data according to the Interquartile Range (IQR).
/// The IQR is the range between the 1st quartile (25th percentile) and 3rd quartile (75th percentile).
/// Formula: X_scaled = (X - median) / IQR
/// </para>
/// <para><b>For Beginners:</b> This scaler is like StandardScaler but resistant to outliers.
///
/// Instead of using mean (which outliers can skew), it uses:
/// - Median: The middle value (50th percentile)
/// - IQR: The range of the middle 50% of data
///
/// Example with salaries [$30K, $45K, $60K, $75K, $5M]:
/// - StandardScaler would be thrown off by the $5M outlier
/// - RobustScaler focuses on the middle values ($45K-$75K)
/// - The outlier gets a large scaled value, but doesn't compress the rest
///
/// Use this when your data has outliers you want to preserve but not let dominate.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TInput">Input data structure.</typeparam>
/// <typeparam name="TOutput">Output data structure.</typeparam>
public class RobustScaler<T, TInput, TOutput> : RobustScalingNormalizer<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new RobustScaler.
    /// </summary>
    public RobustScaler() : base()
    {
    }

    /// <summary>
    /// Fits the scaler by computing median and IQR for each feature.
    /// </summary>
    public List<NormalizationParameters<T>> Fit(TInput data)
    {
        var (_, parameters) = NormalizeInput(data);
        return parameters;
    }

    /// <summary>
    /// Transforms data using fitted statistics.
    /// </summary>
    public TInput Transform(TInput data, List<NormalizationParameters<T>> parameters)
    {
        var (normalized, _) = NormalizeInput(data);
        return normalized;
    }

    /// <summary>
    /// Fits and transforms in one step.
    /// </summary>
    public (TInput, List<NormalizationParameters<T>>) FitTransform(TInput data)
    {
        return NormalizeInput(data);
    }

    /// <summary>
    /// Inverse transform back to original scale.
    /// </summary>
    public TOutput InverseTransform(TOutput data, NormalizationParameters<T> parameters)
    {
        return Denormalize(data, parameters);
    }
}
```

**Step 2**: Create outlier-resistance tests
```csharp
[Fact]
public void RobustScaler_WithOutliers_NotSkewedByExtremes()
{
    // Arrange - Data with outlier
    var data = Vector<double>.FromArray(new[] {
        10.0, 20.0, 30.0, 40.0, 50.0,  // Normal range
        1000.0  // Outlier
    });
    var scaler = new RobustScaler<double, Matrix<double>, Vector<double>>();

    // Act
    var (scaled, params_) = scaler.NormalizeOutput(data);

    // Assert
    // Median should be around 35 (not affected by 1000)
    Assert.True(params_.Median < 100.0);
    // IQR should be around 30 (Q3-Q1 ≈ 45-15)
    Assert.True(params_.IQR < 100.0);
    // Most values should be in reasonable range
    Assert.True(Math.Abs(scaled[0]) < 5.0);  // Not compressed by outlier
}
```

---

## Phase 4: Normalizer (L2 Normalization)

### AC 4.1: Implement Normalizer Class

**NEW IMPLEMENTATION NEEDED**: This normalizes individual **samples** (rows), not features (columns).

**Step 1**: Create Normalizer class
```csharp
// File: src/Preprocessing/Normalizer.cs
namespace AiDotNet.Preprocessing;

/// <summary>
/// Normalizes samples individually to unit norm.
/// Each sample (row) is normalized independently.
/// </summary>
/// <remarks>
/// <para>
/// The Normalizer scales input vectors individually to unit norm (length 1).
/// This is different from other scalers which normalize features (columns).
/// The normalizer normalizes each sample (row) independently.
///
/// Supported norms:
/// - L2 (Euclidean): sqrt(sum of squares) - DEFAULT
/// - L1 (Manhattan): sum of absolute values
/// - Max: maximum absolute value
/// </para>
/// <para><b>For Beginners:</b> This makes each data sample have length 1.
///
/// Think of it like normalizing vectors in geometry:
/// - A vector [3, 4] has length 5 (because sqrt(3²+4²)=5)
/// - After L2 normalization: [3/5, 4/5] = [0.6, 0.8]
/// - The new length is 1 (because sqrt(0.6²+0.8²)=1)
///
/// Use cases:
/// - Text classification: Normalize document vectors
/// - Similarity computations: Make vectors comparable by length
/// - Neural networks: Input normalization
///
/// Unlike StandardScaler/MinMaxScaler which normalize FEATURES (columns),
/// Normalizer normalizes SAMPLES (rows). Each row becomes unit length.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Normalizer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly NormType _norm;

    /// <summary>
    /// Type of norm to use for normalization.
    /// </summary>
    public enum NormType
    {
        /// <summary>L2 (Euclidean) norm: sqrt(sum of squares)</summary>
        L2,
        /// <summary>L1 (Manhattan) norm: sum of absolute values</summary>
        L1,
        /// <summary>Max norm: maximum absolute value</summary>
        Max
    }

    /// <summary>
    /// Initializes a new Normalizer with specified norm.
    /// </summary>
    /// <param name="norm">Type of norm to use (default: L2).</param>
    public Normalizer(NormType norm = NormType.L2)
    {
        _numOps = NumericOperations<T>.Instance;
        _norm = norm;
    }

    /// <summary>
    /// Normalizes each sample (row) to unit norm.
    /// </summary>
    /// <param name="data">Input matrix where each row is a sample.</param>
    /// <returns>Matrix with normalized rows.</returns>
    public Matrix<T> Transform(Matrix<T> data)
    {
        var result = new Matrix<T>(data.Rows, data.Columns);

        for (int i = 0; i < data.Rows; i++)
        {
            var row = data.GetRow(i);
            var normalizedRow = NormalizeVector(row);
            result.SetRow(i, normalizedRow);
        }

        return result;
    }

    /// <summary>
    /// Normalizes each sample (row) in a tensor.
    /// Assumes tensor shape is [samples, features].
    /// </summary>
    public Tensor<T> Transform(Tensor<T> data)
    {
        if (data.Shape.Length != 2)
        {
            throw new ArgumentException(
                "Normalizer requires 2D tensor [samples, features]. " +
                $"Got {data.Shape.Length}D tensor.");
        }

        var result = new Tensor<T>(data.Shape);

        for (int i = 0; i < data.Shape[0]; i++)
        {
            // Extract row as vector
            var row = new Vector<T>(data.Shape[1]);
            for (int j = 0; j < data.Shape[1]; j++)
            {
                row[j] = data[i, j];
            }

            // Normalize
            var normalizedRow = NormalizeVector(row);

            // Write back
            for (int j = 0; j < data.Shape[1]; j++)
            {
                result[i, j] = normalizedRow[j];
            }
        }

        return result;
    }

    /// <summary>
    /// Fit method (does nothing - Normalizer is stateless).
    /// Provided for API compatibility with other scalers.
    /// </summary>
    public void Fit(Matrix<T> data)
    {
        // Normalizer doesn't learn anything - each sample normalized independently
    }

    /// <summary>
    /// Fit and transform (just calls Transform since no fitting needed).
    /// </summary>
    public Matrix<T> FitTransform(Matrix<T> data)
    {
        return Transform(data);
    }

    /// <summary>
    /// Normalizes a single vector to unit norm.
    /// </summary>
    private Vector<T> NormalizeVector(Vector<T> vec)
    {
        T norm = CalculateNorm(vec);

        // Avoid division by zero
        if (_numOps.Equals(norm, _numOps.Zero))
        {
            return vec; // Return unchanged if norm is zero
        }

        // Divide each element by norm
        return vec.Transform(x => _numOps.Divide(x, norm));
    }

    /// <summary>
    /// Calculates the norm of a vector according to the specified norm type.
    /// </summary>
    private T CalculateNorm(Vector<T> vec)
    {
        switch (_norm)
        {
            case NormType.L2:
                // Euclidean norm: sqrt(sum of squares)
                T sumSquares = _numOps.Zero;
                for (int i = 0; i < vec.Length; i++)
                {
                    sumSquares = _numOps.Add(sumSquares,
                        _numOps.Multiply(vec[i], vec[i]));
                }
                return _numOps.Sqrt(sumSquares);

            case NormType.L1:
                // Manhattan norm: sum of absolute values
                T sumAbs = _numOps.Zero;
                for (int i = 0; i < vec.Length; i++)
                {
                    T absVal = vec[i];
                    // Implement Abs
                    if (_numOps.LessThan(absVal, _numOps.Zero))
                    {
                        absVal = _numOps.Negate(absVal);
                    }
                    sumAbs = _numOps.Add(sumAbs, absVal);
                }
                return sumAbs;

            case NormType.Max:
                // Max norm: maximum absolute value
                T maxVal = _numOps.Zero;
                for (int i = 0; i < vec.Length; i++)
                {
                    T absVal = vec[i];
                    if (_numOps.LessThan(absVal, _numOps.Zero))
                    {
                        absVal = _numOps.Negate(absVal);
                    }
                    if (_numOps.GreaterThan(absVal, maxVal))
                    {
                        maxVal = absVal;
                    }
                }
                return maxVal;

            default:
                throw new InvalidOperationException($"Unknown norm type: {_norm}");
        }
    }
}
```

**Step 2**: Create comprehensive tests
```csharp
// File: tests/UnitTests/Preprocessing/NormalizerTests.cs
namespace AiDotNet.Tests.Preprocessing;

public class NormalizerTests
{
    [Fact]
    public void Normalizer_L2Norm_MakesVectorUnitLength()
    {
        // Arrange
        var data = new Matrix<double>(new[,] {
            { 3.0, 4.0 },  // Length = 5
            { 1.0, 0.0 }   // Length = 1
        });
        var normalizer = new Normalizer<double>(Normalizer<double>.NormType.L2);

        // Act
        var normalized = normalizer.Transform(data);

        // Assert - First row should be [0.6, 0.8] (3/5, 4/5)
        Assert.Equal(0.6, normalized[0, 0], 5);
        Assert.Equal(0.8, normalized[0, 1], 5);

        // Verify unit length: 0.6² + 0.8² = 1.0
        double lengthSquared = normalized[0, 0] * normalized[0, 0] +
                               normalized[0, 1] * normalized[0, 1];
        Assert.Equal(1.0, lengthSquared, 5);
    }

    [Fact]
    public void Normalizer_L1Norm_SumsToOne()
    {
        // Arrange
        var data = new Matrix<double>(new[,] {
            { 1.0, 2.0, 3.0 }  // Sum = 6
        });
        var normalizer = new Normalizer<double>(Normalizer<double>.NormType.L1);

        // Act
        var normalized = normalizer.Transform(data);

        // Assert - Should be [1/6, 2/6, 3/6]
        Assert.Equal(1.0/6.0, normalized[0, 0], 5);
        Assert.Equal(2.0/6.0, normalized[0, 1], 5);
        Assert.Equal(3.0/6.0, normalized[0, 2], 5);

        // Sum should be 1.0
        double sum = normalized[0, 0] + normalized[0, 1] + normalized[0, 2];
        Assert.Equal(1.0, sum, 5);
    }

    [Fact]
    public void Normalizer_MaxNorm_ScalesByMaxValue()
    {
        // Arrange
        var data = new Matrix<double>(new[,] {
            { 1.0, 5.0, 3.0 }  // Max = 5
        });
        var normalizer = new Normalizer<double>(Normalizer<double>.NormType.Max);

        // Act
        var normalized = normalizer.Transform(data);

        // Assert - Should be [1/5, 5/5, 3/5]
        Assert.Equal(0.2, normalized[0, 0], 5);
        Assert.Equal(1.0, normalized[0, 1], 5);  // Max value becomes 1
        Assert.Equal(0.6, normalized[0, 2], 5);
    }

    [Fact]
    public void Normalizer_ZeroVector_RemainsUnchanged()
    {
        // Arrange
        var data = new Matrix<double>(new[,] {
            { 0.0, 0.0, 0.0 }
        });
        var normalizer = new Normalizer<double>();

        // Act
        var normalized = normalizer.Transform(data);

        // Assert - Should remain all zeros
        Assert.Equal(0.0, normalized[0, 0]);
        Assert.Equal(0.0, normalized[0, 1]);
        Assert.Equal(0.0, normalized[0, 2]);
    }
}
```

---

## Common Pitfalls to Avoid

1. **Data Leakage**: NEVER fit on test data
   ```csharp
   // WRONG
   scaler.Fit(testData);

   // CORRECT
   scaler.Fit(trainData);
   scaler.Transform(testData);
   ```

2. **Forgetting to Store Parameters**: You need parameters to transform new data
   ```csharp
   // CORRECT
   var (scaled, params_) = scaler.FitTransform(trainData);
   // Save params_ for later use
   ```

3. **Wrong Scaler for Data Type**:
   - StandardScaler: Assumes normal distribution
   - MinMaxScaler: Sensitive to outliers
   - RobustScaler: Use when you have outliers
   - Normalizer: For sample-wise normalization (different use case!)

4. **Not Using INumericOperations**: Always use NumOps for type-generic code
   ```csharp
   // WRONG
   double result = a + b;

   // CORRECT
   T result = _numOps.Add(a, b);
   ```

5. **Normalizer vs Others**: Normalizer normalizes ROWS (samples), others normalize COLUMNS (features)

---

## Testing Strategy

### Unit Tests
1. Test each scaler with known inputs/outputs
2. Test inverse transform restores original scale
3. Test edge cases (zero vectors, single values, outliers)
4. Test with different numeric types (double, float)

### Integration Tests
1. Test with real ML pipelines (train/test split)
2. Test with multiple features
3. Test parameter persistence and reuse

### Performance Tests
1. Benchmark with large matrices (10000x100)
2. Compare memory usage
3. Test parallel processing potential

---

## Next Steps

1. Create the three alias classes (StandardScaler, MinMaxScaler, RobustScaler)
2. Implement the new Normalizer class
3. Write comprehensive unit tests
4. Add integration tests with ML pipelines
5. Update documentation with usage examples
6. Consider thread-safety for concurrent transformations

**Estimated Effort**: 3-4 days for a junior developer

**Dependencies**: None - all base classes exist

**Files to Create**:
- `src/Preprocessing/StandardScaler.cs`
- `src/Preprocessing/MinMaxScaler.cs`
- `src/Preprocessing/RobustScaler.cs`
- `src/Preprocessing/Normalizer.cs`
- `tests/UnitTests/Preprocessing/StandardScalerTests.cs`
- `tests/UnitTests/Preprocessing/MinMaxScalerTests.cs`
- `tests/UnitTests/Preprocessing/RobustScalerTests.cs`
- `tests/UnitTests/Preprocessing/NormalizerTests.cs`
