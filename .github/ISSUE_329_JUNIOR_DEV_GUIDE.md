# Issue #329: Junior Developer Implementation Guide

## Implement Time-Series Moving Window Operations

**This issue creates NEW utilities for time-series feature engineering.**

###What You're Building:
1. **RollingWindow**: Compute statistics over sliding windows (mean, sum, min, max, std dev)
2. **LagLeadFeatures**: Shift time series data forward (lead) or backward (lag)

---

## Understanding Time-Series Concepts

### What is a Rolling Window?

**Rolling Window** (also called "sliding window" or "moving window") computes statistics over a fixed-size window that slides through the data.

**Example - Rolling Mean with window size 3:**
```
Data:     [10, 20, 30, 40, 50, 60]
Window 1: [10, 20, 30]  → Mean = 20
Window 2:     [20, 30, 40]  → Mean = 30
Window 3:         [30, 40, 50]  → Mean = 40
Window 4:             [40, 50, 60]  → Mean = 50

Result:   [20, 30, 40, 50]  (4 values for 6 inputs with window size 3)
```

**Use Cases**:
- Smoothing noisy data (rolling mean)
- Detecting volatility (rolling standard deviation)
- Finding local patterns (rolling min/max)
- Technical indicators in finance (moving averages)

### What are Lag/Lead Features?

**Lag Features**: Previous values in the time series (shift backward)
**Lead Features**: Future values in the time series (shift forward)

**Example - Lag 1 (previous value):**
```
Original: [10, 20, 30, 40, 50]
Lag 1:    [?,  10, 20, 30, 40]  (fill ? with default value)
```

**Example - Lead 1 (next value):**
```
Original: [10, 20, 30, 40, 50]
Lead 1:   [20, 30, 40, 50, ?]  (fill ? with default value)
```

**Use Cases**:
- Predicting next value (use previous values as features)
- Autocorrelation analysis
- Sequence-to-sequence models
- Creating time-lagged variables for regression

---

## Phase 1: Step-by-Step Implementation

### AC 1.1: Create RollingWindow.cs

**File**: `src/TimeSeries/RollingWindow.cs` (NEW FILE - create `src/TimeSeries` directory)

```csharp
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Provides utilities for computing rolling (moving window) statistics on time series data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Rolling window operations compute statistics over a sliding window of fixed size that moves
/// through the data. This is essential for time series analysis, smoothing, and feature engineering.
/// </para>
/// <para><b>For Beginners:</b> Think of rolling windows like looking at your data through a moving frame.
///
/// Imagine tracking daily temperatures:
/// - Raw data: [68°, 72°, 75°, 71°, 69°, 73°]
/// - 3-day rolling average: [71.67°, 72.67°, 71.67°, 71.0°]
///
/// This smooths out daily fluctuations and shows the overall trend.
///
/// Rolling statistics help you:
/// - Smooth noisy data
/// - Detect patterns over time
/// - Create features for machine learning
/// - Calculate technical indicators (like moving averages in stock trading)
/// </para>
/// </remarks>
public static class RollingWindow<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Computes the rolling mean (average) over a sliding window.
    /// </summary>
    /// <param name="data">The input time series data.</param>
    /// <param name="windowSize">The size of the rolling window (must be &gt;= 1 and &lt;= data length).</param>
    /// <returns>A vector containing rolling means, with length = data.Length - windowSize + 1.</returns>
    /// <exception cref="ArgumentException">Thrown when windowSize is invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Rolling mean calculates the average of values within each window.
    ///
    /// Example with windowSize=3:
    /// - Input: [10, 20, 30, 40, 50]
    /// - Output: [(10+20+30)/3, (20+30+40)/3, (30+40+50)/3] = [20, 30, 40]
    ///
    /// Use rolling mean to:
    /// - Smooth out short-term fluctuations
    /// - Highlight longer-term trends
    /// - Reduce noise in sensor data
    /// </para>
    /// </remarks>
    public static Vector<T> Mean(Vector<T> data, int windowSize)
    {
        ValidateWindowSize(data, windowSize);

        int outputLength = data.Length - windowSize + 1;
        var result = new Vector<T>(outputLength);

        // Calculate first window sum
        T currentSum = NumOps.Zero;
        for (int i = 0; i < windowSize; i++)
        {
            currentSum = NumOps.Add(currentSum, data[i]);
        }

        // Store first mean
        result[0] = NumOps.Divide(currentSum, NumOps.FromInt(windowSize));

        // Slide window: subtract leaving element, add entering element
        for (int i = 1; i < outputLength; i++)
        {
            currentSum = NumOps.Subtract(currentSum, data[i - 1]);
            currentSum = NumOps.Add(currentSum, data[i + windowSize - 1]);
            result[i] = NumOps.Divide(currentSum, NumOps.FromInt(windowSize));
        }

        return result;
    }

    /// <summary>
    /// Computes the rolling standard deviation over a sliding window.
    /// </summary>
    /// <param name="data">The input time series data.</param>
    /// <param name="windowSize">The size of the rolling window.</param>
    /// <param name="ddof">Delta degrees of freedom (default: 1 for sample std dev).</param>
    /// <returns>A vector containing rolling standard deviations.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Standard deviation measures how spread out values are from the mean.
    ///
    /// High standard deviation = values are widely spread (high volatility)
    /// Low standard deviation = values are clustered together (low volatility)
    ///
    /// Rolling standard deviation shows how volatility changes over time.
    ///
    /// Use rolling std dev to:
    /// - Detect periods of high/low volatility
    /// - Identify anomalies (values far from rolling mean)
    /// - Measure consistency over time
    /// </para>
    /// <para>
    /// Formula: sqrt( sum((x - mean)²) / (n - ddof) )
    /// ddof=1 gives unbiased sample standard deviation (default)
    /// ddof=0 gives population standard deviation
    /// </para>
    /// </remarks>
    public static Vector<T> StandardDeviation(Vector<T> data, int windowSize, int ddof = 1)
    {
        ValidateWindowSize(data, windowSize);

        if (ddof < 0 || ddof >= windowSize)
            throw new ArgumentException($"ddof must be in range [0, {windowSize})", nameof(ddof));

        int outputLength = data.Length - windowSize + 1;
        var result = new Vector<T>(outputLength);

        for (int i = 0; i < outputLength; i++)
        {
            // Calculate mean for this window
            T sum = NumOps.Zero;
            for (int j = i; j < i + windowSize; j++)
            {
                sum = NumOps.Add(sum, data[j]);
            }
            T mean = NumOps.Divide(sum, NumOps.FromInt(windowSize));

            // Calculate sum of squared deviations
            T sumSquaredDev = NumOps.Zero;
            for (int j = i; j < i + windowSize; j++)
            {
                T deviation = NumOps.Subtract(data[j], mean);
                sumSquaredDev = NumOps.Add(sumSquaredDev, NumOps.Multiply(deviation, deviation));
            }

            // Calculate variance and standard deviation
            T variance = NumOps.Divide(sumSquaredDev, NumOps.FromInt(windowSize - ddof));
            result[i] = NumOps.Sqrt(variance);
        }

        return result;
    }

    /// <summary>
    /// Computes the rolling sum over a sliding window.
    /// </summary>
    /// <param name="data">The input time series data.</param>
    /// <param name="windowSize">The size of the rolling window.</param>
    /// <returns>A vector containing rolling sums.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Rolling sum adds up all values within each window.
    ///
    /// Example with windowSize=3:
    /// - Input: [10, 20, 30, 40, 50]
    /// - Output: [10+20+30, 20+30+40, 30+40+50] = [60, 90, 120]
    ///
    /// Use rolling sum to:
    /// - Calculate cumulative totals over time periods
    /// - Aggregate values within windows
    /// - Detect changes in total activity
    /// </para>
    /// </remarks>
    public static Vector<T> Sum(Vector<T> data, int windowSize)
    {
        ValidateWindowSize(data, windowSize);

        int outputLength = data.Length - windowSize + 1;
        var result = new Vector<T>(outputLength);

        // Calculate first window sum
        T currentSum = NumOps.Zero;
        for (int i = 0; i < windowSize; i++)
        {
            currentSum = NumOps.Add(currentSum, data[i]);
        }
        result[0] = currentSum;

        // Slide window: subtract leaving element, add entering element
        for (int i = 1; i < outputLength; i++)
        {
            currentSum = NumOps.Subtract(currentSum, data[i - 1]);
            currentSum = NumOps.Add(currentSum, data[i + windowSize - 1]);
            result[i] = currentSum;
        }

        return result;
    }

    /// <summary>
    /// Computes the rolling minimum over a sliding window.
    /// </summary>
    /// <param name="data">The input time series data.</param>
    /// <param name="windowSize">The size of the rolling window.</param>
    /// <returns>A vector containing rolling minimums.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Rolling minimum finds the smallest value within each window.
    ///
    /// Example with windowSize=3:
    /// - Input: [50, 30, 70, 20, 90]
    /// - Output: [min(50,30,70), min(30,70,20), min(70,20,90)] = [30, 20, 20]
    ///
    /// Use rolling minimum to:
    /// - Find lowest values in recent history
    /// - Detect support levels in financial data
    /// - Track minimum thresholds over time
    /// </para>
    /// </remarks>
    public static Vector<T> Min(Vector<T> data, int windowSize)
    {
        ValidateWindowSize(data, windowSize);

        int outputLength = data.Length - windowSize + 1;
        var result = new Vector<T>(outputLength);

        for (int i = 0; i < outputLength; i++)
        {
            T min = data[i];
            for (int j = i + 1; j < i + windowSize; j++)
            {
                if (NumOps.LessThan(data[j], min))
                {
                    min = data[j];
                }
            }
            result[i] = min;
        }

        return result;
    }

    /// <summary>
    /// Computes the rolling maximum over a sliding window.
    /// </summary>
    /// <param name="data">The input time series data.</param>
    /// <param name="windowSize">The size of the rolling window.</param>
    /// <returns>A vector containing rolling maximums.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Rolling maximum finds the largest value within each window.
    ///
    /// Example with windowSize=3:
    /// - Input: [50, 30, 70, 20, 90]
    /// - Output: [max(50,30,70), max(30,70,20), max(70,20,90)] = [70, 70, 90]
    ///
    /// Use rolling maximum to:
    /// - Find highest values in recent history
    /// - Detect resistance levels in financial data
    /// - Track peak values over time
    /// </para>
    /// </remarks>
    public static Vector<T> Max(Vector<T> data, int windowSize)
    {
        ValidateWindowSize(data, windowSize);

        int outputLength = data.Length - windowSize + 1;
        var result = new Vector<T>(outputLength);

        for (int i = 0; i < outputLength; i++)
        {
            T max = data[i];
            for (int j = i + 1; j < i + windowSize; j++)
            {
                if (NumOps.GreaterThan(data[j], max))
                {
                    max = data[j];
                }
            }
            result[i] = max;
        }

        return result;
    }

    /// <summary>
    /// Validates that the window size is appropriate for the given data.
    /// </summary>
    private static void ValidateWindowSize(Vector<T> data, int windowSize)
    {
        if (data == null)
            throw new ArgumentNullException(nameof(data));

        if (windowSize < 1)
            throw new ArgumentException("Window size must be at least 1.", nameof(windowSize));

        if (windowSize > data.Length)
            throw new ArgumentException($"Window size ({windowSize}) cannot be larger than data length ({data.Length}).", nameof(windowSize));
    }
}
```

### AC 1.2: Unit Tests for Rolling Statistics

**File**: `tests/UnitTests/TimeSeries/RollingWindowTests.cs` (NEW FILE)

```csharp
using AiDotNet.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.TimeSeries;

public class RollingWindowTests
{
    [Fact]
    public void Mean_SimpleData_ReturnsCorrectMeans()
    {
        // Arrange
        var data = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });
        int windowSize = 3;

        // Expected: [(10+20+30)/3, (20+30+40)/3, (30+40+50)/3] = [20, 30, 40]

        // Act
        var result = RollingWindow<double>.Mean(data, windowSize);

        // Assert
        Assert.Equal(3, result.Length);
        Assert.Equal(20.0, result[0], precision: 10);
        Assert.Equal(30.0, result[1], precision: 10);
        Assert.Equal(40.0, result[2], precision: 10);
    }

    [Fact]
    public void Sum_SimpleData_ReturnsCorrectSums()
    {
        // Arrange
        var data = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0 });
        int windowSize = 2;

        // Expected: [10+20, 20+30, 30+40] = [30, 50, 70]

        // Act
        var result = RollingWindow<double>.Sum(data, windowSize);

        // Assert
        Assert.Equal(3, result.Length);
        Assert.Equal(30.0, result[0]);
        Assert.Equal(50.0, result[1]);
        Assert.Equal(70.0, result[2]);
    }

    [Fact]
    public void Min_SimpleData_ReturnsCorrectMinimums()
    {
        // Arrange
        var data = new Vector<double>(new[] { 50.0, 30.0, 70.0, 20.0, 90.0 });
        int windowSize = 3;

        // Expected: [min(50,30,70), min(30,70,20), min(70,20,90)] = [30, 20, 20]

        // Act
        var result = RollingWindow<double>.Min(data, windowSize);

        // Assert
        Assert.Equal(3, result.Length);
        Assert.Equal(30.0, result[0]);
        Assert.Equal(20.0, result[1]);
        Assert.Equal(20.0, result[2]);
    }

    [Fact]
    public void Max_SimpleData_ReturnsCorrectMaximums()
    {
        // Arrange
        var data = new Vector<double>(new[] { 50.0, 30.0, 70.0, 20.0, 90.0 });
        int windowSize = 3;

        // Expected: [max(50,30,70), max(30,70,20), max(70,20,90)] = [70, 70, 90]

        // Act
        var result = RollingWindow<double>.Max(data, windowSize);

        // Assert
        Assert.Equal(3, result.Length);
        Assert.Equal(70.0, result[0]);
        Assert.Equal(70.0, result[1]);
        Assert.Equal(90.0, result[2]);
    }

    [Fact]
    public void StandardDeviation_SimpleData_ReturnsCorrectStdDev()
    {
        // Arrange
        var data = new Vector<double>(new[] { 2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0 });
        int windowSize = 4;
        int ddof = 1; // Sample standard deviation

        // Expected: Manual calculation for first window [2,4,4,4]:
        // Mean = 3.5, Deviations = [-1.5, 0.5, 0.5, 0.5], Squared = [2.25, 0.25, 0.25, 0.25]
        // Sum = 3.0, Variance = 3.0/3 = 1.0, Std Dev = 1.0

        // Act
        var result = RollingWindow<double>.StandardDeviation(data, windowSize, ddof);

        // Assert
        Assert.Equal(5, result.Length);
        Assert.Equal(1.0, result[0], precision: 10);
    }

    [Fact]
    public void Mean_WindowSizeTooLarge_ThrowsArgumentException()
    {
        // Arrange
        var data = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
        int windowSize = 5; // Larger than data length

        // Act & Assert
        Assert.Throws<ArgumentException>(() => RollingWindow<double>.Mean(data, windowSize));
    }

    [Fact]
    public void Mean_WindowSizeZero_ThrowsArgumentException()
    {
        // Arrange
        var data = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
        int windowSize = 0;

        // Act & Assert
        Assert.Throws<ArgumentException>(() => RollingWindow<double>.Mean(data, windowSize));
    }
}
```

---

## Phase 2: Step-by-Step Implementation

### AC 2.1: Create LagLeadFeatures.cs

**File**: `src/TimeSeries/LagLeadFeatures.cs` (NEW FILE)

```csharp
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Provides utilities for creating lagged and leading features from time series data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Lagged features represent previous values in a time series, while leading features
/// represent future values. These are crucial for creating features for time series
/// forecasting and sequential modeling.
/// </para>
/// <para><b>For Beginners:</b> Lag and lead features shift your data in time.
///
/// Imagine predicting tomorrow's temperature:
/// - Original: [Mon:68°, Tue:72°, Wed:75°, Thu:71°]
/// - Lag 1 (yesterday's temp): [?, 68°, 72°, 75°]
/// - Lag 2 (2 days ago): [?, ?, 68°, 72°]
///
/// You can use yesterday's temperature (lag 1) as a feature to predict today's temperature.
///
/// Lead features work in reverse - they shift data forward to represent future values.
///
/// Use lag/lead features to:
/// - Create autoregressive features (use past to predict future)
/// - Capture temporal dependencies
/// - Build sequence-to-sequence models
/// - Analyze autocorrelation
/// </para>
/// </remarks>
public static class LagLeadFeatures<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Creates a lagged version of the time series by shifting data backward.
    /// </summary>
    /// <param name="data">The input time series data.</param>
    /// <param name="lagSteps">Number of steps to lag (shift backward). Must be positive.</param>
    /// <param name="fillValue">Value to fill in the new positions created by lagging.</param>
    /// <returns>A lagged vector of the same length as input.</returns>
    /// <exception cref="ArgumentException">Thrown when lagSteps is invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Lagging shifts data backward in time.
    ///
    /// Example with lagSteps=2:
    /// - Original: [10, 20, 30, 40, 50]
    /// - Lag 2:    [fill, fill, 10, 20, 30]
    ///
    /// The first 2 positions have no historical data, so they're filled with fillValue.
    ///
    /// Common fill strategies:
    /// - 0.0: Zero-fill (simple, but may bias results)
    /// - NaN: Mark missing values (for analysis)
    /// - First value: Forward-fill (assumes continuity)
    ///
    /// Use lag features when:
    /// - Predicting future values from past values
    /// - Building autoregressive models (AR, ARIMA)
    /// - Capturing temporal dependencies
    /// </para>
    /// </remarks>
    public static Vector<T> Lag(Vector<T> data, int lagSteps, T fillValue)
    {
        if (data == null)
            throw new ArgumentNullException(nameof(data));

        if (lagSteps < 0)
            throw new ArgumentException("Lag steps must be non-negative.", nameof(lagSteps));

        if (lagSteps >= data.Length)
            throw new ArgumentException($"Lag steps ({lagSteps}) must be less than data length ({data.Length}).", nameof(lagSteps));

        var result = new Vector<T>(data.Length);

        // Fill first lagSteps positions with fillValue
        for (int i = 0; i < lagSteps; i++)
        {
            result[i] = fillValue;
        }

        // Shift data backward by lagSteps
        for (int i = lagSteps; i < data.Length; i++)
        {
            result[i] = data[i - lagSteps];
        }

        return result;
    }

    /// <summary>
    /// Creates a leading version of the time series by shifting data forward.
    /// </summary>
    /// <param name="data">The input time series data.</param>
    /// <param name="leadSteps">Number of steps to lead (shift forward). Must be positive.</param>
    /// <param name="fillValue">Value to fill in the new positions created by leading.</param>
    /// <returns>A leading vector of the same length as input.</returns>
    /// <exception cref="ArgumentException">Thrown when leadSteps is invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Leading shifts data forward in time.
    ///
    /// Example with leadSteps=2:
    /// - Original: [10, 20, 30, 40, 50]
    /// - Lead 2:   [30, 40, 50, fill, fill]
    ///
    /// The last 2 positions have no future data, so they're filled with fillValue.
    ///
    /// Use lead features when:
    /// - You have future information available during training
    /// - Building models that predict past from future (rare)
    /// - Creating target variables for forecasting (shift y forward to create labels)
    ///
    /// Example for forecasting:
    /// - Features (X): Today's temperature
    /// - Target (y): Lead(temperature, 1) = Tomorrow's temperature
    /// </para>
    /// </remarks>
    public static Vector<T> Lead(Vector<T> data, int leadSteps, T fillValue)
    {
        if (data == null)
            throw new ArgumentNullException(nameof(data));

        if (leadSteps < 0)
            throw new ArgumentException("Lead steps must be non-negative.", nameof(leadSteps));

        if (leadSteps >= data.Length)
            throw new ArgumentException($"Lead steps ({leadSteps}) must be less than data length ({data.Length}).", nameof(leadSteps));

        var result = new Vector<T>(data.Length);

        // Shift data forward by leadSteps
        for (int i = 0; i < data.Length - leadSteps; i++)
        {
            result[i] = data[i + leadSteps];
        }

        // Fill last leadSteps positions with fillValue
        for (int i = data.Length - leadSteps; i < data.Length; i++)
        {
            result[i] = fillValue;
        }

        return result;
    }

    /// <summary>
    /// Creates multiple lagged features at once.
    /// </summary>
    /// <param name="data">The input time series data.</param>
    /// <param name="maxLag">Maximum number of lags to create (creates lags 1, 2, ..., maxLag).</param>
    /// <param name="fillValue">Value to fill missing positions.</param>
    /// <returns>A matrix where each column is a lagged version (column 0 = lag 1, column 1 = lag 2, etc.).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates multiple lagged features at once.
    ///
    /// Example with maxLag=3:
    /// - Original: [10, 20, 30, 40, 50]
    /// - Lag 1:    [?,  10, 20, 30, 40]
    /// - Lag 2:    [?,  ?,  10, 20, 30]
    /// - Lag 3:    [?,  ?,  ?,  10, 20]
    ///
    /// Result is a matrix with 3 columns (one for each lag).
    ///
    /// This is useful for autoregressive models that use multiple past values:
    /// - Predict value[t] using value[t-1], value[t-2], value[t-3]
    /// </para>
    /// </remarks>
    public static Matrix<T> CreateMultipleLags(Vector<T> data, int maxLag, T fillValue)
    {
        if (maxLag < 1)
            throw new ArgumentException("maxLag must be at least 1.", nameof(maxLag));

        var result = new Matrix<T>(data.Length, maxLag);

        for (int lagStep = 1; lagStep <= maxLag; lagStep++)
        {
            var laggedVector = Lag(data, lagStep, fillValue);
            for (int row = 0; row < data.Length; row++)
            {
                result[row, lagStep - 1] = laggedVector[row];
            }
        }

        return result;
    }
}
```

### AC 2.2: Unit Tests for Lag/Lead Features

**File**: `tests/UnitTests/TimeSeries/LagLeadFeaturesTests.cs` (NEW FILE)

```csharp
using AiDotNet.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.TimeSeries;

public class LagLeadFeaturesTests
{
    [Fact]
    public void Lag_SimpleData_ReturnsCorrectLaggedVector()
    {
        // Arrange
        var data = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });
        int lagSteps = 2;
        double fillValue = 0.0;

        // Expected: [0, 0, 10, 20, 30]

        // Act
        var result = LagLeadFeatures<double>.Lag(data, lagSteps, fillValue);

        // Assert
        Assert.Equal(5, result.Length);
        Assert.Equal(0.0, result[0]);
        Assert.Equal(0.0, result[1]);
        Assert.Equal(10.0, result[2]);
        Assert.Equal(20.0, result[3]);
        Assert.Equal(30.0, result[4]);
    }

    [Fact]
    public void Lead_SimpleData_ReturnsCorrectLeadingVector()
    {
        // Arrange
        var data = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });
        int leadSteps = 2;
        double fillValue = 0.0;

        // Expected: [30, 40, 50, 0, 0]

        // Act
        var result = LagLeadFeatures<double>.Lead(data, leadSteps, fillValue);

        // Assert
        Assert.Equal(5, result.Length);
        Assert.Equal(30.0, result[0]);
        Assert.Equal(40.0, result[1]);
        Assert.Equal(50.0, result[2]);
        Assert.Equal(0.0, result[3]);
        Assert.Equal(0.0, result[4]);
    }

    [Fact]
    public void Lag_ZeroSteps_ReturnsCopy()
    {
        // Arrange
        var data = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
        int lagSteps = 0;
        double fillValue = -1.0;

        // Expected: [10, 20, 30] (no shift)

        // Act
        var result = LagLeadFeatures<double>.Lag(data, lagSteps, fillValue);

        // Assert
        Assert.Equal(3, result.Length);
        Assert.Equal(10.0, result[0]);
        Assert.Equal(20.0, result[1]);
        Assert.Equal(30.0, result[2]);
    }

    [Fact]
    public void CreateMultipleLags_SimpleData_ReturnsCorrectMatrix()
    {
        // Arrange
        var data = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });
        int maxLag = 3;
        double fillValue = 0.0;

        // Expected matrix (5 rows x 3 columns):
        // Lag 1: [0,  10, 20, 30, 40]
        // Lag 2: [0,  0,  10, 20, 30]
        // Lag 3: [0,  0,  0,  10, 20]

        // Act
        var result = LagLeadFeatures<double>.CreateMultipleLags(data, maxLag, fillValue);

        // Assert
        Assert.Equal(5, result.Rows);
        Assert.Equal(3, result.Columns);

        // Check lag 1 (column 0)
        Assert.Equal(0.0, result[0, 0]);
        Assert.Equal(10.0, result[1, 0]);
        Assert.Equal(20.0, result[2, 0]);

        // Check lag 2 (column 1)
        Assert.Equal(0.0, result[0, 1]);
        Assert.Equal(0.0, result[1, 1]);
        Assert.Equal(10.0, result[2, 1]);

        // Check lag 3 (column 2)
        Assert.Equal(0.0, result[0, 2]);
        Assert.Equal(0.0, result[1, 2]);
        Assert.Equal(0.0, result[2, 2]);
        Assert.Equal(10.0, result[3, 2]);
    }

    [Fact]
    public void Lag_NegativeSteps_ThrowsArgumentException()
    {
        // Arrange
        var data = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
        int lagSteps = -1;

        // Act & Assert
        Assert.Throws<ArgumentException>(() => LagLeadFeatures<double>.Lag(data, lagSteps, 0.0));
    }

    [Fact]
    public void Lead_StepsTooLarge_ThrowsArgumentException()
    {
        // Arrange
        var data = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
        int leadSteps = 5; // Larger than data length

        // Act & Assert
        Assert.Throws<ArgumentException>(() => LagLeadFeatures<double>.Lead(data, leadSteps, 0.0));
    }
}
```

---

## Common Pitfalls to Avoid:

1. **DON'T use hardcoded numeric types** - Always use `NumOps.FromDouble()`, `NumOps.Zero`, etc.
2. **DON'T forget edge cases** - Window size = 1, window size = data length, lag = 0
3. **DON'T use inefficient algorithms** - Use sliding window optimization for Sum/Mean (O(n) not O(n*w))
4. **DO validate inputs** - Check window size, lag steps, data length
5. **DO handle fill values properly** - Consider zero-fill, NaN-fill, or forward-fill strategies
6. **DO document output length** - Rolling window outputs length = data.Length - windowSize + 1
7. **DO test with multiple numeric types** - Test with double and float generics
8. **DO explain ddof parameter** - Degrees of freedom affects standard deviation calculation

---

## Testing Strategy:

1. **Unit Tests**: Test each method with simple known inputs
2. **Edge Cases**: Test window size = 1, window size = data length, lag = 0, lead = 0
3. **Numeric Stability**: Test with very small/large values
4. **Type Tests**: Test with double and float generic types
5. **Performance Tests**: Verify O(n) complexity for Sum/Mean (not O(n*w))

**Next Steps**:
1. Create `src/TimeSeries` directory
2. Implement `RollingWindow.cs` with all statistics methods
3. Implement `LagLeadFeatures.cs` with Lag/Lead/CreateMultipleLags
4. Write comprehensive unit tests
5. Test with real time-series data
