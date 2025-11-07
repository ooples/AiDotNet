# Issue #317: Junior Developer Implementation Guide
## Implement MaxAbsScaler and QuantileTransformer Normalizers

## Understanding Normalization in AiDotNet

### What is Normalization?

**Normalization** (also called scaling) transforms data into a standard range, making it easier for machine learning algorithms to process. Different features often have vastly different scales (e.g., age: 0-100 vs income: $0-$1,000,000), which can cause problems for many algorithms.

### Existing Normalizers in AiDotNet

**Location**: `src/Normalizers/`

**Current implementations**:
1. `MinMaxNormalizer.cs` - Scales to [0, 1] or [min, max] range
2. `ZScoreNormalizer.cs` - Standardizes to mean=0, std=1
3. `RobustScalingNormalizer.cs` - Uses median and IQR (robust to outliers)
4. `LpNormNormalizer.cs` - Normalizes to unit norm (L1, L2, etc.)
5. `DecimalNormalizer.cs` - Decimal scaling normalization
6. `GlobalContrastNormalizer.cs` - For image contrast normalization
7. `BinningNormalizer.cs` - Discretizes into bins
8. `LogNormalizer.cs` - Logarithmic transformation
9. `MeanVarianceNormalizer.cs` - Mean-variance normalization
10. `LogMeanVarianceNormalizer.cs` - Log transform + mean-variance

### What's Missing?

**Issue #317** identifies two important normalizers:

1. **MaxAbsScaler** - Scales by maximum absolute value to [-1, 1]
   - Useful for sparse data
   - Preserves sparsity (zeros remain zeros)
   - Common in scikit-learn, TensorFlow

2. **QuantileTransformer** - Non-linear transformation to uniform/normal distribution
   - Robust to outliers
   - Transforms to desired distribution
   - Handles non-Gaussian data well

---

## Understanding NormalizerBase Architecture

### Base Class Structure

**File**: `src/Normalizers/NormalizerBase.cs`

```csharp
public abstract class NormalizerBase<T, TInput, TOutput> : INormalizer<T, TInput, TOutput>
{
    // Automatically initialized NumOps for type-safe arithmetic
    protected readonly INumericOperations<T> NumOps;

    protected NormalizerBase()
    {
        NumOps = MathHelper.GetNumericOperations<T>();
    }

    // Five abstract methods that concrete normalizers must implement:

    // 1. Normalize output (typically a Vector<T>)
    public abstract (TOutput, NormalizationParameters<T>) NormalizeOutput(TOutput data);

    // 2. Normalize input (typically a Matrix<T>)
    public abstract (TInput, List<NormalizationParameters<T>>) NormalizeInput(TInput data);

    // 3. Denormalize output
    public abstract TOutput Denormalize(TOutput data, NormalizationParameters<T> parameters);

    // 4. Denormalize coefficients
    public abstract TOutput Denormalize(TOutput coefficients,
        List<NormalizationParameters<T>> xParams,
        NormalizationParameters<T> yParams);

    // 5. Calculate denormalized intercept
    public abstract T Denormalize(TInput xMatrix, TOutput y, TOutput coefficients,
        List<NormalizationParameters<T>> xParams,
        NormalizationParameters<T> yParams);
}
```

### NormalizationParameters Class

```csharp
public class NormalizationParameters<T>
{
    public T Min { get; set; }           // Minimum value
    public T Max { get; set; }           // Maximum value
    public T Mean { get; set; }          // Mean value
    public T StandardDeviation { get; set; }  // Standard deviation
    public T Median { get; set; }        // Median value
    public T InterQuartileRange { get; set; }  // IQR (Q3 - Q1)

    // For quantile-based methods
    public List<T>? Quantiles { get; set; }  // Quantile values

    // Generic storage for custom parameters
    public Dictionary<string, object>? CustomParameters { get; set; }
}
```

---

## Phase 1: Implement MaxAbsScaler

### Conceptual Understanding

**MaxAbsScaler** scales each feature by dividing by its maximum absolute value, resulting in values in the range [-1, 1].

**Formula**: `scaled_value = value / max(|values|)`

**Key Properties**:
- Preserves sparsity (zeros remain zeros)
- No shifting of center (unlike MinMax scaling)
- Scales to [-1, 1] range
- Robust for features with different scales
- Works well with sparse data

**Example**:
```
Feature values: [-6, -3, 0, 3, 6]
Max absolute value: 6
Scaled values: [-1, -0.5, 0, 0.5, 1]
```

---

### AC 1.1: Create MaxAbsScaler.cs

**File**: `src/Normalizers/MaxAbsScaler.cs`

```csharp
namespace AiDotNet.Normalizers;

/// <summary>
/// Scales features by their maximum absolute value, preserving sparsity.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// MaxAbsScaler scales each feature by dividing it by its maximum absolute value,
/// resulting in values in the range [-1, 1]. Unlike MinMax scaling, this method:
/// - Does not shift/center the data (preserves zero)
/// - Preserves the sign of values
/// - Works well with sparse data (zeros remain zeros)
/// </para>
/// <para><b>For Beginners:</b> This normalizer is like finding the "largest magnitude" in your data.
///
/// Think of it like volume control:
/// - If the loudest sound in a recording is at 60% volume
/// - MaxAbsScaler "amplifies" everything so the loudest is at 100%
/// - But it keeps the proportions and doesn't shift the baseline
///
/// Example with temperatures (in Fahrenheit):
/// - Original: [-20, -10, 0, 10, 20]
/// - Max absolute value: 20
/// - Scaled: [-1.0, -0.5, 0.0, 0.5, 1.0]
///
/// Notice how:
/// - Zero stays zero (important for sparse data!)
/// - Negative values stay negative
/// - Everything scales proportionally
/// </para>
/// <para><b>When to Use MaxAbsScaler:</b>
/// - When you have sparse data (lots of zeros)
/// - When you want to preserve the sign of values
/// - When you don't want to shift the center of your data
/// - For algorithms that benefit from [-1, 1] scaling (like SVMs, neural networks)
///
/// When NOT to use:
/// - When features have very different outliers (outliers affect max absolute value)
/// - When you need strict [0, 1] range (use MinMaxNormalizer instead)
/// - When you want mean=0 and std=1 (use ZScoreNormalizer instead)
/// </para>
/// <para><b>Default Parameters:</b>
/// - No parameters needed! MaxAbsScaler just finds the maximum absolute value per feature.
/// </para>
/// <para><b>Comparison with Other Normalizers:</b>
/// - vs MinMaxScaler: MaxAbsScaler preserves zeros and signs, MinMax shifts to [0,1]
/// - vs StandardScaler: MaxAbsScaler scales to [-1,1], StandardScaler standardizes to mean=0, std=1
/// - vs RobustScaler: Both handle outliers differently; RobustScaler uses median/IQR
/// </para>
/// </remarks>
public class MaxAbsScaler<T> : NormalizerBase<T, Matrix<T>, Vector<T>>
{
    /// <summary>
    /// Initializes a new instance of the MaxAbsScaler class.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creating a MaxAbsScaler is simple - no configuration needed!
    ///
    /// Example usage:
    /// ```csharp
    /// var scaler = new MaxAbsScaler<double>();
    ///
    /// // Normalize your data
    /// var (normalizedData, parameters) = scaler.NormalizeInput(myMatrix);
    ///
    /// // Later, denormalize results
    /// var originalScale = scaler.Denormalize(prediction, parameters[0]);
    /// ```
    /// </para>
    /// </remarks>
    public MaxAbsScaler() : base()
    {
    }

    /// <summary>
    /// Normalizes a vector by dividing by its maximum absolute value.
    /// </summary>
    /// <param name="data">The vector to normalize.</param>
    /// <returns>
    /// A tuple containing:
    /// - The normalized vector (scaled to [-1, 1])
    /// - Normalization parameters (containing the max absolute value)
    /// </returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method scales a single vector (like a column of data).
    ///
    /// Step-by-step process:
    /// 1. Find the maximum absolute value in the vector
    ///    - Example: [-5, -2, 0, 3, 10] has max abs value of 10
    /// 2. Divide every value by this maximum
    ///    - Result: [-0.5, -0.2, 0, 0.3, 1.0]
    /// 3. Store the max abs value for later denormalization
    ///
    /// Special cases:
    /// - If max abs value is 0 (all zeros), return the original data unchanged
    /// - This prevents division by zero
    /// </para>
    /// </remarks>
    public override (Vector<T>, NormalizationParameters<T>) NormalizeOutput(Vector<T> data)
    {
        // Find maximum absolute value
        T maxAbsValue = NumOps.Zero;

        for (int i = 0; i < data.Length; i++)
        {
            T absValue = NumOps.Abs(data[i]);
            if (NumOps.GreaterThan(absValue, maxAbsValue))
            {
                maxAbsValue = absValue;
            }
        }

        // Handle edge case: all zeros
        if (NumOps.Equals(maxAbsValue, NumOps.Zero))
        {
            return (data, new NormalizationParameters<T>
            {
                Max = NumOps.Zero,
                CustomParameters = new Dictionary<string, object>
                {
                    { "MaxAbsValue", maxAbsValue }
                }
            });
        }

        // Scale by dividing by max absolute value
        var normalized = new Vector<T>(data.Length);
        for (int i = 0; i < data.Length; i++)
        {
            normalized[i] = NumOps.Divide(data[i], maxAbsValue);
        }

        var parameters = new NormalizationParameters<T>
        {
            Max = maxAbsValue,
            CustomParameters = new Dictionary<string, object>
            {
                { "MaxAbsValue", maxAbsValue }
            }
        };

        return (normalized, parameters);
    }

    /// <summary>
    /// Normalizes a matrix by scaling each column (feature) independently.
    /// </summary>
    /// <param name="data">The matrix to normalize (rows=samples, columns=features).</param>
    /// <returns>
    /// A tuple containing:
    /// - The normalized matrix (each column scaled to [-1, 1])
    /// - List of normalization parameters (one per feature/column)
    /// </returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method scales multiple features at once.
    ///
    /// Process:
    /// 1. For each column (feature) in the matrix:
    ///    - Find its maximum absolute value
    ///    - Divide all values in that column by this maximum
    /// 2. Each column is scaled independently
    ///
    /// Example with 3 samples, 2 features:
    /// ```
    /// Original:          After MaxAbsScaler:
    /// [[-4,  100]        [[-1.0,  0.5]
    ///  [ 0,  200]   -->   [ 0.0,  1.0]
    ///  [ 4, -100]]        [ 1.0, -0.5]]
    ///
    /// Max abs per column: [4, 200]
    /// ```
    ///
    /// Notice:
    /// - Column 1 was divided by 4 (its max abs value)
    /// - Column 2 was divided by 200 (its max abs value)
    /// - Each column is now in [-1, 1] range
    /// - Zeros and signs are preserved
    /// </para>
    /// </remarks>
    public override (Matrix<T>, List<NormalizationParameters<T>>) NormalizeInput(Matrix<T> data)
    {
        int rows = data.Rows;
        int cols = data.Columns;
        var normalized = new Matrix<T>(rows, cols);
        var parametersList = new List<NormalizationParameters<T>>();

        // Process each column (feature) independently
        for (int col = 0; col < cols; col++)
        {
            // Find maximum absolute value for this column
            T maxAbsValue = NumOps.Zero;

            for (int row = 0; row < rows; row++)
            {
                T absValue = NumOps.Abs(data[row, col]);
                if (NumOps.GreaterThan(absValue, maxAbsValue))
                {
                    maxAbsValue = absValue;
                }
            }

            // Store parameters for this column
            var parameters = new NormalizationParameters<T>
            {
                Max = maxAbsValue,
                CustomParameters = new Dictionary<string, object>
                {
                    { "MaxAbsValue", maxAbsValue }
                }
            };
            parametersList.Add(parameters);

            // Scale this column
            if (NumOps.Equals(maxAbsValue, NumOps.Zero))
            {
                // All zeros - copy unchanged
                for (int row = 0; row < rows; row++)
                {
                    normalized[row, col] = data[row, col];
                }
            }
            else
            {
                // Divide by max absolute value
                for (int row = 0; row < rows; row++)
                {
                    normalized[row, col] = NumOps.Divide(data[row, col], maxAbsValue);
                }
            }
        }

        return (normalized, parametersList);
    }

    /// <summary>
    /// Denormalizes a vector by multiplying by the original maximum absolute value.
    /// </summary>
    /// <param name="data">The normalized vector to denormalize.</param>
    /// <param name="parameters">The normalization parameters from the original normalization.</param>
    /// <returns>The vector in its original scale.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This reverses the normalization to get back to original values.
    ///
    /// Process:
    /// 1. Retrieve the max absolute value that was used for normalization
    /// 2. Multiply each value by this maximum
    ///
    /// Example:
    /// - Normalized values: [-1.0, -0.5, 0.0, 0.5, 1.0]
    /// - Max abs value was: 20
    /// - Denormalized: [-20, -10, 0, 10, 20]
    ///
    /// This is the exact reverse of normalization!
    /// </para>
    /// </remarks>
    public override Vector<T> Denormalize(Vector<T> data, NormalizationParameters<T> parameters)
    {
        T maxAbsValue = parameters.Max;

        // If max abs was zero, return unchanged
        if (NumOps.Equals(maxAbsValue, NumOps.Zero))
        {
            return data;
        }

        // Multiply by max absolute value
        var denormalized = new Vector<T>(data.Length);
        for (int i = 0; i < data.Length; i++)
        {
            denormalized[i] = NumOps.Multiply(data[i], maxAbsValue);
        }

        return denormalized;
    }

    /// <summary>
    /// Denormalizes model coefficients from normalized space to original space.
    /// </summary>
    /// <param name="coefficients">The model coefficients trained on normalized data.</param>
    /// <param name="xParams">The normalization parameters for input features.</param>
    /// <param name="yParams">The normalization parameters for the target variable.</param>
    /// <returns>Denormalized coefficients for use with original, unnormalized data.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> When you train a model on normalized data, its coefficients
    /// are in "normalized space". To use the model on original data, you need to adjust
    /// the coefficients.
    ///
    /// For MaxAbsScaler, the relationship is:
    /// - denormalized_coefficient = normalized_coefficient * (y_max_abs / x_max_abs)
    ///
    /// This accounts for the different scaling factors applied to inputs vs outputs.
    ///
    /// Example:
    /// - Input feature scaled by 100 (max abs value)
    /// - Output scaled by 50 (max abs value)
    /// - If normalized coefficient is 0.5
    /// - Denormalized coefficient = 0.5 * (50 / 100) = 0.25
    /// </para>
    /// </remarks>
    public override Vector<T> Denormalize(
        Vector<T> coefficients,
        List<NormalizationParameters<T>> xParams,
        NormalizationParameters<T> yParams)
    {
        var denormalized = new Vector<T>(coefficients.Length);
        T yMaxAbs = yParams.Max;

        for (int i = 0; i < coefficients.Length; i++)
        {
            T xMaxAbs = xParams[i].Max;

            // Handle edge cases
            if (NumOps.Equals(xMaxAbs, NumOps.Zero))
            {
                denormalized[i] = NumOps.Zero;
            }
            else if (NumOps.Equals(yMaxAbs, NumOps.Zero))
            {
                denormalized[i] = NumOps.Zero;
            }
            else
            {
                // coefficient_denorm = coefficient_norm * (y_max_abs / x_max_abs)
                T scale = NumOps.Divide(yMaxAbs, xMaxAbs);
                denormalized[i] = NumOps.Multiply(coefficients[i], scale);
            }
        }

        return denormalized;
    }

    /// <summary>
    /// Calculates the denormalized Y-intercept for a linear model.
    /// </summary>
    /// <param name="xMatrix">The original (unnormalized) input matrix.</param>
    /// <param name="y">The original (unnormalized) target vector.</param>
    /// <param name="coefficients">The normalized model coefficients.</param>
    /// <param name="xParams">The normalization parameters for input features.</param>
    /// <param name="yParams">The normalization parameters for the target variable.</param>
    /// <returns>The denormalized Y-intercept.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> For linear models (y = mx + b), the intercept (b) needs
    /// to be adjusted when denormalizing coefficients.
    ///
    /// MaxAbsScaler doesn't shift the mean (unlike MinMaxScaler), so the intercept
    /// calculation is simpler:
    /// - intercept_denorm = mean(y) - sum(denormalized_coefficients * mean(x))
    ///
    /// This ensures predictions on original data match the model trained on normalized data.
    /// </para>
    /// </remarks>
    public override T Denormalize(
        Matrix<T> xMatrix,
        Vector<T> y,
        Vector<T> coefficients,
        List<NormalizationParameters<T>> xParams,
        NormalizationParameters<T> yParams)
    {
        // Calculate means
        T yMean = CalculateMean(y);
        var xMeans = new Vector<T>(xMatrix.Columns);

        for (int col = 0; col < xMatrix.Columns; col++)
        {
            T sum = NumOps.Zero;
            for (int row = 0; row < xMatrix.Rows; row++)
            {
                sum = NumOps.Add(sum, xMatrix[row, col]);
            }
            xMeans[col] = NumOps.Divide(sum, NumOps.FromDouble(xMatrix.Rows));
        }

        // Denormalize coefficients
        var denormalizedCoefficients = Denormalize(coefficients, xParams, yParams);

        // Calculate intercept
        T intercept = yMean;
        for (int i = 0; i < denormalizedCoefficients.Length; i++)
        {
            T term = NumOps.Multiply(denormalizedCoefficients[i], xMeans[i]);
            intercept = NumOps.Subtract(intercept, term);
        }

        return intercept;
    }

    /// <summary>
    /// Helper method to calculate the mean of a vector.
    /// </summary>
    private T CalculateMean(Vector<T> data)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < data.Length; i++)
        {
            sum = NumOps.Add(sum, data[i]);
        }
        return NumOps.Divide(sum, NumOps.FromDouble(data.Length));
    }
}
```

---

### AC 1.2: Unit Tests for MaxAbsScaler

**File**: `tests/UnitTests/Normalizers/MaxAbsScalerTests.cs`

```csharp
namespace AiDotNet.Tests.Normalizers;

public class MaxAbsScalerTests
{
    [Fact]
    public void NormalizeOutput_SimpleVector_ScalesToMaxAbs()
    {
        // Arrange
        var scaler = new MaxAbsScaler<double>();
        var data = new Vector<double>(new[] { -10.0, -5.0, 0.0, 5.0, 10.0 });

        // Act
        var (normalized, parameters) = scaler.NormalizeOutput(data);

        // Assert
        Assert.Equal(10.0, parameters.Max);  // Max abs value
        Assert.Equal(-1.0, normalized[0]);    // -10 / 10
        Assert.Equal(-0.5, normalized[1]);    // -5 / 10
        Assert.Equal(0.0, normalized[2]);     // 0 / 10
        Assert.Equal(0.5, normalized[3]);     // 5 / 10
        Assert.Equal(1.0, normalized[4]);     // 10 / 10
    }

    [Fact]
    public void NormalizeOutput_NegativeMaxAbs_ScalesCorrectly()
    {
        // Arrange
        var scaler = new MaxAbsScaler<double>();
        var data = new Vector<double>(new[] { -20.0, -10.0, -5.0, 0.0, 5.0 });

        // Act
        var (normalized, parameters) = scaler.NormalizeOutput(data);

        // Assert
        Assert.Equal(20.0, parameters.Max);   // Max abs value is 20 (from -20)
        Assert.Equal(-1.0, normalized[0]);    // -20 / 20
        Assert.Equal(-0.5, normalized[1]);    // -10 / 20
        Assert.Equal(-0.25, normalized[2]);   // -5 / 20
        Assert.Equal(0.0, normalized[3]);     // 0 / 20
        Assert.Equal(0.25, normalized[4]);    // 5 / 20
    }

    [Fact]
    public void NormalizeOutput_AllZeros_ReturnsUnchanged()
    {
        // Arrange
        var scaler = new MaxAbsScaler<double>();
        var data = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0 });

        // Act
        var (normalized, parameters) = scaler.NormalizeOutput(data);

        // Assert
        Assert.Equal(0.0, parameters.Max);
        Assert.All(normalized.ToArray(), val => Assert.Equal(0.0, val));
    }

    [Fact]
    public void NormalizeInput_Matrix_ScalesEachColumnIndependently()
    {
        // Arrange
        var scaler = new MaxAbsScaler<double>();
        var data = new Matrix<double>(new double[,]
        {
            { -4.0, 100.0 },
            {  0.0, 200.0 },
            {  4.0, -100.0 }
        });

        // Act
        var (normalized, parameters) = scaler.NormalizeInput(data);

        // Assert - Column 1 (max abs = 4)
        Assert.Equal(4.0, parameters[0].Max);
        Assert.Equal(-1.0, normalized[0, 0]);  // -4 / 4
        Assert.Equal(0.0, normalized[1, 0]);   // 0 / 4
        Assert.Equal(1.0, normalized[2, 0]);   // 4 / 4

        // Assert - Column 2 (max abs = 200)
        Assert.Equal(200.0, parameters[1].Max);
        Assert.Equal(0.5, normalized[0, 1]);   // 100 / 200
        Assert.Equal(1.0, normalized[1, 1]);   // 200 / 200
        Assert.Equal(-0.5, normalized[2, 1]);  // -100 / 200
    }

    [Fact]
    public void NormalizeInput_PreservesSparsity()
    {
        // Arrange
        var scaler = new MaxAbsScaler<double>();
        var data = new Matrix<double>(new double[,]
        {
            { 10.0, 0.0, 0.0 },
            { 0.0, 20.0, 0.0 },
            { 0.0, 0.0, 30.0 }
        });

        // Act
        var (normalized, parameters) = scaler.NormalizeInput(data);

        // Assert - Zeros should remain zeros
        Assert.Equal(0.0, normalized[0, 1]);
        Assert.Equal(0.0, normalized[0, 2]);
        Assert.Equal(0.0, normalized[1, 0]);
        Assert.Equal(0.0, normalized[1, 2]);
        Assert.Equal(0.0, normalized[2, 0]);
        Assert.Equal(0.0, normalized[2, 1]);

        // Non-zero values should be scaled
        Assert.Equal(1.0, normalized[0, 0]);  // 10 / 10
        Assert.Equal(1.0, normalized[1, 1]);  // 20 / 20
        Assert.Equal(1.0, normalized[2, 2]);  // 30 / 30
    }

    [Fact]
    public void Denormalize_RevertsToOriginalScale()
    {
        // Arrange
        var scaler = new MaxAbsScaler<double>();
        var original = new Vector<double>(new[] { -20.0, -10.0, 0.0, 10.0, 20.0 });

        // Act
        var (normalized, parameters) = scaler.NormalizeOutput(original);
        var denormalized = scaler.Denormalize(normalized, parameters);

        // Assert
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], denormalized[i], precision: 10);
        }
    }

    [Fact]
    public void Denormalize_Coefficients_AdjustsForScaling()
    {
        // Arrange
        var scaler = new MaxAbsScaler<double>();
        var xParams = new List<NormalizationParameters<double>>
        {
            new NormalizationParameters<double> { Max = 10.0 },
            new NormalizationParameters<double> { Max = 20.0 }
        };
        var yParams = new NormalizationParameters<double> { Max = 100.0 };
        var normalizedCoeffs = new Vector<double>(new[] { 0.5, 0.3 });

        // Act
        var denormalizedCoeffs = scaler.Denormalize(normalizedCoeffs, xParams, yParams);

        // Assert
        // coefficient_denorm = coefficient_norm * (y_max_abs / x_max_abs)
        Assert.Equal(0.5 * (100.0 / 10.0), denormalizedCoeffs[0], precision: 10);  // 5.0
        Assert.Equal(0.3 * (100.0 / 20.0), denormalizedCoeffs[1], precision: 10);  // 1.5
    }

    [Theory]
    [InlineData(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 }, 5.0)]
    [InlineData(new[] { -5.0, -4.0, -3.0, -2.0, -1.0 }, 5.0)]
    [InlineData(new[] { -100.0, 50.0 }, 100.0)]
    [InlineData(new[] { 0.1, 0.2, 0.3 }, 0.3)]
    public void NormalizeOutput_VariousInputs_CorrectMaxAbs(double[] inputData, double expectedMaxAbs)
    {
        // Arrange
        var scaler = new MaxAbsScaler<double>();
        var data = new Vector<double>(inputData);

        // Act
        var (normalized, parameters) = scaler.NormalizeOutput(data);

        // Assert
        Assert.Equal(expectedMaxAbs, parameters.Max, precision: 10);
        Assert.All(normalized.ToArray(), val => Assert.True(Math.Abs(val) <= 1.0));
    }

    [Fact]
    public void NormalizeInput_WithMultipleTypes_WorksWithFloat()
    {
        // Arrange
        var scaler = new MaxAbsScaler<float>();
        var data = new Matrix<float>(new float[,]
        {
            { -2.0f, 4.0f },
            { 2.0f, -4.0f }
        });

        // Act
        var (normalized, parameters) = scaler.NormalizeInput(data);

        // Assert
        Assert.Equal(2.0f, parameters[0].Max);
        Assert.Equal(4.0f, parameters[1].Max);
        Assert.Equal(-1.0f, normalized[0, 0]);
        Assert.Equal(1.0f, normalized[0, 1]);
    }
}
```

---

## Phase 2: Implement QuantileTransformer

### Conceptual Understanding

**QuantileTransformer** applies a non-linear transformation that maps data to follow a specified distribution (uniform or normal).

**How it works**:
1. **Fit**: Compute quantiles from training data
2. **Transform**: Map each value to its quantile, then map quantile to target distribution
3. **Result**: Data follows uniform [0, 1] or normal (mean=0, std=1) distribution

**Key Properties**:
- Robust to outliers (uses rank-based transformation)
- Non-linear transformation
- Reduces impact of extreme values
- Can transform to uniform or normal distribution

**Example** (transforming to uniform distribution):
```
Original values (sorted): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Quantiles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
Transformed: Each value mapped to its corresponding quantile

Value 1 (10th percentile) -> 0.1
Value 5 (50th percentile) -> 0.5
Value 10 (100th percentile) -> 1.0
```

### AC 2.1: Create QuantileTransformer.cs

**File**: `src/Normalizers/QuantileTransformer.cs`

```csharp
namespace AiDotNet.Normalizers;

/// <summary>
/// Transforms features using quantile information to follow a uniform or normal distribution.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// QuantileTransformer performs a rank-based transformation that maps data to a specified
/// distribution (uniform or normal). This makes it robust to outliers and suitable for
/// non-Gaussian data.
/// </para>
/// <para><b>For Beginners:</b> This normalizer is like "redistributing" your data.
///
/// Think of students' test scores:
/// - Original scores: [40, 50, 60, 70, 80, 90, 95, 100, 100, 100]
/// - Some students cluster at the top (grade inflation)
///
/// QuantileTransformer:
/// 1. Ranks all values (40 is lowest, 100s are highest)
/// 2. Converts ranks to percentiles (40 is 10th percentile, 100 is 100th)
/// 3. Maps percentiles to target distribution (uniform [0,1] or normal)
///
/// Result: Data spread evenly or following a bell curve, regardless of original shape.
/// </para>
/// <para><b>When to Use QuantileTransformer:</b>
/// - When data has outliers (robust transformation)
/// - When data is highly skewed (non-Gaussian)
/// - When you want uniform or normal distribution
/// - For algorithms that assume specific distributions (like neural networks)
///
/// When NOT to use:
/// - When you need to preserve exact values (this is non-linear)
/// - When data already follows desired distribution
/// - For sparse data (quantile estimation needs sufficient samples)
/// </para>
/// <para><b>Default Parameters:</b>
/// - Output distribution: Uniform (range [0, 1])
/// - Number of quantiles: 1000 (provides fine-grained transformation)
/// - Source: scikit-learn default parameters
/// - Rationale: 1000 quantiles balance accuracy vs computational cost
/// </para>
/// </remarks>
public class QuantileTransformer<T> : NormalizerBase<T, Matrix<T>, Vector<T>>
{
    /// <summary>
    /// The target output distribution.
    /// </summary>
    public enum OutputDistribution
    {
        /// <summary>
        /// Transform to uniform distribution [0, 1].
        /// </summary>
        Uniform,

        /// <summary>
        /// Transform to normal distribution (mean=0, std=1).
        /// </summary>
        Normal
    }

    private readonly OutputDistribution _outputDistribution;
    private readonly int _nQuantiles;

    /// <summary>
    /// Initializes a new instance of the QuantileTransformer class.
    /// </summary>
    /// <param name="outputDistribution">
    /// The target distribution for transformed data.
    /// Default: Uniform. Use Normal for Gaussian distribution.
    /// </param>
    /// <param name="nQuantiles">
    /// Number of quantiles to compute for transformation.
    /// Default: 1000. Higher values = more accurate but slower.
    /// Must be less than or equal to number of samples.
    /// </param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Configure how you want your data transformed.
    ///
    /// Output Distribution:
    /// - Uniform: Spreads data evenly across [0, 1] range
    ///   * Good for: Ensuring all values are equally likely
    ///   * Example: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ///
    /// - Normal: Transforms to bell curve (mean=0, std=1)
    ///   * Good for: Algorithms that assume Gaussian data
    ///   * Example: [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    ///
    /// Number of Quantiles:
    /// - 1000 (default): Good balance for most datasets
    /// - 100: Faster, less accurate
    /// - 10000: More accurate, slower
    /// - Rule: Use at least 100, no more than number of samples
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">
    /// Thrown when nQuantiles is less than 2.
    /// </exception>
    public QuantileTransformer(
        OutputDistribution outputDistribution = OutputDistribution.Uniform,
        int nQuantiles = 1000)
        : base()
    {
        if (nQuantiles < 2)
            throw new ArgumentException("Number of quantiles must be at least 2.", nameof(nQuantiles));

        _outputDistribution = outputDistribution;
        _nQuantiles = nQuantiles;
    }

    /// <summary>
    /// Gets the configured output distribution.
    /// </summary>
    public OutputDistribution Distribution => _outputDistribution;

    /// <summary>
    /// Gets the number of quantiles used for transformation.
    /// </summary>
    public int NQuantiles => _nQuantiles;

    public override (Vector<T>, NormalizationParameters<T>) NormalizeOutput(Vector<T> data)
    {
        // Compute quantiles
        var quantiles = ComputeQuantiles(data);

        // Transform data using quantiles
        var normalized = Transform(data, quantiles);

        var parameters = new NormalizationParameters<T>
        {
            CustomParameters = new Dictionary<string, object>
            {
                { "Quantiles", quantiles },
                { "OutputDistribution", _outputDistribution },
                { "NQuantiles", _nQuantiles }
            }
        };

        return (normalized, parameters);
    }

    public override (Matrix<T>, List<NormalizationParameters<T>>) NormalizeInput(Matrix<T> data)
    {
        int rows = data.Rows;
        int cols = data.Columns;
        var normalized = new Matrix<T>(rows, cols);
        var parametersList = new List<NormalizationParameters<T>>();

        // Process each column independently
        for (int col = 0; col < cols; col++)
        {
            // Extract column as vector
            var columnData = new Vector<T>(rows);
            for (int row = 0; row < rows; row++)
            {
                columnData[row] = data[row, col];
            }

            // Normalize this column
            var (normalizedColumn, parameters) = NormalizeOutput(columnData);

            // Store in matrix
            for (int row = 0; row < rows; row++)
            {
                normalized[row, col] = normalizedColumn[row];
            }

            parametersList.Add(parameters);
        }

        return (normalized, parametersList);
    }

    public override Vector<T> Denormalize(Vector<T> data, NormalizationParameters<T> parameters)
    {
        var quantiles = (List<T>)parameters.CustomParameters["Quantiles"];
        var denormalized = new Vector<T>(data.Length);

        for (int i = 0; i < data.Length; i++)
        {
            denormalized[i] = InverseTransform(data[i], quantiles);
        }

        return denormalized;
    }

    public override Vector<T> Denormalize(
        Vector<T> coefficients,
        List<NormalizationParameters<T>> xParams,
        NormalizationParameters<T> yParams)
    {
        // Quantile transformation is non-linear, so coefficient denormalization
        // is not straightforward. Return coefficients as-is with a warning.
        // In practice, models should be trained on transformed data and
        // predictions should be inverse-transformed.
        return coefficients;
    }

    public override T Denormalize(
        Matrix<T> xMatrix,
        Vector<T> y,
        Vector<T> coefficients,
        List<NormalizationParameters<T>> xParams,
        NormalizationParameters<T> yParams)
    {
        // For quantile-transformed data, intercept calculation is complex
        // due to non-linearity. Return zero as a placeholder.
        return NumOps.Zero;
    }

    /// <summary>
    /// Computes quantiles from data.
    /// </summary>
    private List<T> ComputeQuantiles(Vector<T> data)
    {
        // Sort data
        var sorted = data.ToArray();
        Array.Sort(sorted, (a, b) => NumOps.Compare(a, b));

        // Determine number of quantiles (limited by data size)
        int effectiveNQuantiles = Math.Min(_nQuantiles, data.Length);

        var quantiles = new List<T>();

        // Compute quantile values
        for (int i = 0; i < effectiveNQuantiles; i++)
        {
            double percentile = (double)i / (effectiveNQuantiles - 1);
            int index = (int)(percentile * (data.Length - 1));
            quantiles.Add(sorted[index]);
        }

        return quantiles;
    }

    /// <summary>
    /// Transforms a single value using quantile mapping.
    /// </summary>
    private Vector<T> Transform(Vector<T> data, List<T> quantiles)
    {
        var transformed = new Vector<T>(data.Length);

        for (int i = 0; i < data.Length; i++)
        {
            // Find quantile rank
            double quantileRank = FindQuantileRank(data[i], quantiles);

            // Map to target distribution
            T mappedValue = _outputDistribution == OutputDistribution.Uniform
                ? NumOps.FromDouble(quantileRank)
                : MapToNormal(quantileRank);

            transformed[i] = mappedValue;
        }

        return transformed;
    }

    /// <summary>
    /// Finds the quantile rank (0.0 to 1.0) for a given value.
    /// </summary>
    private double FindQuantileRank(T value, List<T> quantiles)
    {
        // Binary search to find position in quantiles
        int left = 0;
        int right = quantiles.Count - 1;

        while (left < right)
        {
            int mid = (left + right) / 2;
            if (NumOps.LessThan(quantiles[mid], value))
            {
                left = mid + 1;
            }
            else
            {
                right = mid;
            }
        }

        return (double)left / (quantiles.Count - 1);
    }

    /// <summary>
    /// Maps a uniform quantile rank [0,1] to normal distribution value.
    /// </summary>
    private T MapToNormal(double uniformValue)
    {
        // Use inverse CDF (probit function) to map to normal
        // This is a simplified approximation
        // For production, use proper inverse normal CDF

        // Clip to avoid numerical issues
        uniformValue = Math.Max(0.001, Math.Min(0.999, uniformValue));

        // Approximation of probit function
        double normalValue = ApproximateProbit(uniformValue);

        return NumOps.FromDouble(normalValue);
    }

    /// <summary>
    /// Approximates the probit function (inverse normal CDF).
    /// </summary>
    private double ApproximateProbit(double p)
    {
        // Simplified approximation
        // For production use, implement proper algorithm or use library

        if (p <= 0.0 || p >= 1.0)
            throw new ArgumentException("Probability must be in (0, 1)");

        // Beasley-Springer-Moro approximation
        double q = p - 0.5;

        if (Math.Abs(q) <= 0.42)
        {
            double r = q * q;
            return q * ((((-25.44106049637 * r + 41.39119773534) * r - 18.61500062529) * r + 2.50662823884) * r - 1.0) /
                   ((((3.13082909833 * r - 21.06224101826) * r + 23.08336743743) * r - 8.47351093090) * r + 1.0);
        }
        else
        {
            double r = q < 0 ? p : 1 - p;
            r = Math.Log(-Math.Log(r));

            double normalValue = (((1.641345311 * r + 3.429567803) * r - 1.624446268) * r - 0.1405284734) /
                                ((0.5370614648 * r + 1.182874816) * r + 1.0);

            return q < 0 ? -normalValue : normalValue;
        }
    }

    /// <summary>
    /// Inverse transforms a value back to original space.
    /// </summary>
    private T InverseTransform(T value, List<T> quantiles)
    {
        double quantileRank;

        if (_outputDistribution == OutputDistribution.Uniform)
        {
            quantileRank = Convert.ToDouble(value);
        }
        else
        {
            // Map from normal to uniform using CDF
            quantileRank = ApproximateNormalCDF(Convert.ToDouble(value));
        }

        // Find corresponding quantile value
        int index = (int)(quantileRank * (quantiles.Count - 1));
        index = Math.Max(0, Math.Min(quantiles.Count - 1, index));

        return quantiles[index];
    }

    /// <summary>
    /// Approximates the normal CDF.
    /// </summary>
    private double ApproximateNormalCDF(double x)
    {
        // Approximation of cumulative distribution function
        // For production, use proper implementation

        return 0.5 * (1.0 + Math.Tanh(x * Math.Sqrt(2.0 / Math.PI)));
    }
}
```

**NOTE**: The QuantileTransformer implementation above is simplified. For production use:
1. Use proper quantile interpolation
2. Implement accurate inverse normal CDF (probit function)
3. Handle edge cases (constant features, outliers)
4. Add subsample parameter for large datasets
5. Consider scipy.stats or MathNet.Numerics for distributions

### AC 2.2: Unit Tests for QuantileTransformer

**File**: `tests/UnitTests/Normalizers/QuantileTransformerTests.cs`

```csharp
namespace AiDotNet.Tests.Normalizers;

public class QuantileTransformerTests
{
    [Fact]
    public void Constructor_DefaultParameters_UsesStandardDefaults()
    {
        // Arrange & Act
        var transformer = new QuantileTransformer<double>();

        // Assert
        Assert.Equal(QuantileTransformer<double>.OutputDistribution.Uniform, transformer.Distribution);
        Assert.Equal(1000, transformer.NQuantiles);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(0)]
    [InlineData(-10)]
    public void Constructor_InvalidNQuantiles_ThrowsArgumentException(int invalidN)
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new QuantileTransformer<double>(nQuantiles: invalidN));
    }

    [Fact]
    public void NormalizeOutput_UniformDistribution_ProducesUniformValues()
    {
        // Arrange
        var transformer = new QuantileTransformer<double>(
            outputDistribution: QuantileTransformer<double>.OutputDistribution.Uniform,
            nQuantiles: 10
        );
        var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 });

        // Act
        var (normalized, parameters) = transformer.NormalizeOutput(data);

        // Assert
        Assert.All(normalized.ToArray(), val =>
        {
            Assert.True(val >= 0.0 && val <= 1.0, $"Value {val} not in [0,1]");
        });
    }

    [Fact]
    public void NormalizeOutput_NormalDistribution_ProducesBellCurve()
    {
        // Arrange
        var transformer = new QuantileTransformer<double>(
            outputDistribution: QuantileTransformer<double>.OutputDistribution.Normal,
            nQuantiles: 10
        );
        var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 });

        // Act
        var (normalized, parameters) = transformer.NormalizeOutput(data);

        // Assert - values should be centered around 0
        double mean = normalized.ToArray().Average();
        Assert.InRange(mean, -0.5, 0.5);  // Approximate mean=0
    }

    [Fact]
    public void NormalizeInput_Matrix_TransformsEachColumnIndependently()
    {
        // Arrange
        var transformer = new QuantileTransformer<double>(
            outputDistribution: QuantileTransformer<double>.OutputDistribution.Uniform,
            nQuantiles: 5
        );
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 100.0 },
            { 2.0, 200.0 },
            { 3.0, 300.0 },
            { 4.0, 400.0 },
            { 5.0, 500.0 }
        });

        // Act
        var (normalized, parameters) = transformer.NormalizeInput(data);

        // Assert
        Assert.Equal(2, parameters.Count);  // One per column

        // Each column should be in [0, 1] for uniform distribution
        for (int col = 0; col < 2; col++)
        {
            for (int row = 0; row < 5; row++)
            {
                Assert.InRange(normalized[row, col], 0.0, 1.0);
            }
        }
    }

    [Fact]
    public void Denormalize_RevertsApproximately()
    {
        // Arrange
        var transformer = new QuantileTransformer<double>(
            outputDistribution: QuantileTransformer<double>.OutputDistribution.Uniform,
            nQuantiles: 10
        );
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 });

        // Act
        var (normalized, parameters) = transformer.NormalizeOutput(original);
        var denormalized = transformer.Denormalize(normalized, parameters);

        // Assert - values should be close to original (quantization may cause small differences)
        for (int i = 0; i < original.Length; i++)
        {
            Assert.InRange(denormalized[i], original[i] - 1.0, original[i] + 1.0);
        }
    }

    [Fact]
    public void NormalizeOutput_SkewedData_BecomesMoreUniform()
    {
        // Arrange
        var transformer = new QuantileTransformer<double>(
            outputDistribution: QuantileTransformer<double>.OutputDistribution.Uniform,
            nQuantiles: 20
        );
        // Highly skewed data (exponential-like)
        var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0 });

        // Act
        var (normalized, parameters) = transformer.NormalizeOutput(data);

        // Assert - transformed data should be more evenly distributed
        var sortedNormalized = normalized.ToArray();
        Array.Sort(sortedNormalized);

        // Check that values are roughly evenly spaced
        for (int i = 1; i < sortedNormalized.Length; i++)
        {
            double gap = sortedNormalized[i] - sortedNormalized[i - 1];
            Assert.InRange(gap, 0.05, 0.2);  // Gaps should be similar
        }
    }

    [Fact]
    public void NormalizeInput_WithDifferentTypes_WorksWithFloat()
    {
        // Arrange
        var transformer = new QuantileTransformer<float>(
            outputDistribution: QuantileTransformer<float>.OutputDistribution.Uniform,
            nQuantiles: 5
        );
        var data = new Matrix<float>(new float[,]
        {
            { 1.0f, 10.0f },
            { 2.0f, 20.0f },
            { 3.0f, 30.0f },
            { 4.0f, 40.0f },
            { 5.0f, 50.0f }
        });

        // Act
        var (normalized, parameters) = transformer.NormalizeInput(data);

        // Assert
        Assert.All(normalized.ToArray(), val => Assert.InRange(val, 0.0f, 1.0f));
    }
}
```

---

## Common Pitfalls to Avoid:

1. **DON'T use `default(T)` or `default!`**
   - Use `NumOps.Zero`, `NumOps.One`, etc.
   - Initialize collections properly: `new Vector<T>(size)`

2. **DO use INumericOperations<T> for all arithmetic**
   - Use `NumOps.Add()`, `NumOps.Divide()`, etc.
   - Use `NumOps.GreaterThan()`, `NumOps.LessThan()` for comparisons
   - Use `NumOps.FromDouble()` to convert constants

3. **DO handle edge cases**
   - Division by zero (max abs value = 0)
   - All-zero features
   - Single-value features
   - Empty data

4. **DO implement all five abstract methods**
   - NormalizeOutput
   - NormalizeInput
   - Denormalize (vector)
   - Denormalize (coefficients)
   - Denormalize (intercept)

5. **DO document defaults with research citations**
   - MaxAbsScaler: No parameters, standard ML practice
   - QuantileTransformer: 1000 quantiles (scikit-learn default)

6. **DO test with multiple numeric types**
   - Test with `double` and `float`
   - Ensure generic type safety

7. **DO preserve sparsity in MaxAbsScaler**
   - Zeros must remain zeros
   - No shifting of data

8. **DO use proper quantile algorithms**
   - For production, use established libraries
   - Current implementation is simplified

---

## Integration with PredictionModelBuilder

Both normalizers should work automatically with `PredictionModelBuilder` through the `INormalizer` interface:

```csharp
// Using MaxAbsScaler
var model = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>()
    .UseNormalizer(new MaxAbsScaler<double>())
    .Build();

// Using QuantileTransformer
var model = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>()
    .UseNormalizer(new QuantileTransformer<double>(
        outputDistribution: QuantileTransformer<double>.OutputDistribution.Normal,
        nQuantiles: 500
    ))
    .Build();
```

---

## Testing Strategy:

### Unit Tests (Required - 80%+ coverage):
- Constructor validation
- Edge cases (zeros, constant features)
- Normalization correctness
- Denormalization reversibility
- Multiple numeric types
- Sparsity preservation (MaxAbsScaler)
- Distribution shape (QuantileTransformer)

### Integration Tests:
- Use with PredictionModelBuilder
- Train model with normalized data
- Make predictions on original scale
- Compare with other normalizers

### Performance Tests:
- Large matrices
- Many features
- QuantileTransformer with many quantiles

---

## Summary:

**What You're Building**:
- MaxAbsScaler: Scales to [-1, 1] by maximum absolute value
- QuantileTransformer: Non-linear transformation to uniform/normal distribution
- Both inherit from NormalizerBase and work with existing infrastructure

**Key Architecture Insights**:
- Use `INumericOperations<T>` for all arithmetic (generic type safety)
- Implement all five abstract methods from NormalizerBase
- Store parameters for denormalization
- Handle edge cases gracefully

**Implementation Checklist**:
- [ ] Create MaxAbsScaler.cs with all methods
- [ ] Create QuantileTransformer.cs with all methods
- [ ] Write comprehensive unit tests (80%+ coverage)
- [ ] Test with multiple numeric types (double, float)
- [ ] Test edge cases (zeros, constant features, outliers)
- [ ] Document with research citations
- [ ] Ensure sparsity preservation in MaxAbsScaler
- [ ] Verify distribution shape in QuantileTransformer

**Success Criteria**:
- All unit tests pass
- 80%+ code coverage
- Works with PredictionModelBuilder
- Reversible transformations (normalize + denormalize = original)
- Clear beginner-friendly documentation
