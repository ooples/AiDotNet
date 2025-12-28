using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Statistics;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Statistics;

/// <summary>
/// Edge case tests for statistics classes.
/// These tests ensure proper handling of boundary conditions, special values, and unusual inputs.
///
/// Categories tested:
/// - Empty data
/// - Single value data
/// - Very large datasets
/// - Very large values
/// - Very small values
/// - Mixed positive/negative values
/// </summary>
public class StatisticsEdgeCaseTests
{
    private const double Tolerance = 1e-6;

    #region BasicStats Edge Cases

    private static BasicStats<T> CreateBasicStats<T>(T[] values)
    {
        return new BasicStats<T>(new BasicStatsInputs<T>
        {
            Values = new Vector<T>(values)
        });
    }

    /// <summary>
    /// Single value: Mean, Min, Max should all equal that value.
    /// </summary>
    [Fact]
    public void BasicStats_SingleValue_AllStatsEqualValue()
    {
        // Arrange
        var stats = CreateBasicStats(new[] { 42.0 });

        // Assert
        Assert.Equal(42.0, stats.Mean, Tolerance);
        Assert.Equal(42.0, stats.Min, Tolerance);
        Assert.Equal(42.0, stats.Max, Tolerance);
        Assert.Equal(42.0, stats.Median, Tolerance);
    }

    /// <summary>
    /// Two values: Median should be the average of both.
    /// </summary>
    [Fact]
    public void BasicStats_TwoValues_MedianIsAverage()
    {
        // Arrange
        var stats = CreateBasicStats(new[] { 10.0, 20.0 });

        // Assert
        Assert.Equal(15.0, stats.Mean, Tolerance);
        Assert.Equal(15.0, stats.Median, Tolerance);
        Assert.Equal(10.0, stats.Min, Tolerance);
        Assert.Equal(20.0, stats.Max, Tolerance);
    }

    /// <summary>
    /// All same values: Variance and StdDev should be 0.
    /// </summary>
    [Fact]
    public void BasicStats_AllSameValues_ZeroVariance()
    {
        // Arrange
        var stats = CreateBasicStats(new[] { 5.0, 5.0, 5.0, 5.0, 5.0 });

        // Assert
        Assert.Equal(5.0, stats.Mean, Tolerance);
        Assert.Equal(5.0, stats.Median, Tolerance);
        Assert.Equal(0.0, stats.Variance, Tolerance);
        Assert.Equal(0.0, stats.StandardDeviation, Tolerance);
    }

    /// <summary>
    /// Large dataset (1000 values) should compute correctly.
    /// </summary>
    [Fact]
    public void BasicStats_LargeDataset_ComputesCorrectly()
    {
        // Arrange - 1000 values from 1 to 1000
        var values = new double[1000];
        for (int i = 0; i < 1000; i++)
        {
            values[i] = i + 1.0;
        }
        var stats = CreateBasicStats(values);

        // Assert - Sum is n(n+1)/2 = 500500, Mean = 500.5
        Assert.Equal(500.5, stats.Mean, Tolerance);
        Assert.Equal(1.0, stats.Min, Tolerance);
        Assert.Equal(1000.0, stats.Max, Tolerance);
        Assert.Equal(500.5, stats.Median, Tolerance); // Median of 1-1000
    }

    /// <summary>
    /// Very large values should be handled without overflow.
    /// </summary>
    [Fact]
    public void BasicStats_VeryLargeValues_HandledCorrectly()
    {
        // Arrange - Values near max double
        var values = new[] { 1e100, 2e100, 3e100 };
        var stats = CreateBasicStats(values);

        // Assert
        Assert.Equal(2e100, stats.Mean, 1e94); // Tolerance proportional to scale
        Assert.Equal(1e100, stats.Min, 1e94);
        Assert.Equal(3e100, stats.Max, 1e94);
    }

    /// <summary>
    /// Very small values should be handled without underflow.
    /// </summary>
    [Fact]
    public void BasicStats_VerySmallValues_HandledCorrectly()
    {
        // Arrange - Values near machine epsilon
        var values = new[] { 1e-100, 2e-100, 3e-100 };
        var stats = CreateBasicStats(values);

        // Assert
        Assert.Equal(2e-100, stats.Mean, 1e-106);
        Assert.Equal(1e-100, stats.Min, 1e-106);
        Assert.Equal(3e-100, stats.Max, 1e-106);
    }

    /// <summary>
    /// Negative values should be handled correctly.
    /// </summary>
    [Fact]
    public void BasicStats_AllNegativeValues_HandledCorrectly()
    {
        // Arrange
        var values = new[] { -5.0, -3.0, -1.0, -7.0, -9.0 };
        var stats = CreateBasicStats(values);

        // Assert
        Assert.Equal(-5.0, stats.Mean, Tolerance); // (-5-3-1-7-9)/5 = -25/5 = -5
        Assert.Equal(-9.0, stats.Min, Tolerance);
        Assert.Equal(-1.0, stats.Max, Tolerance);
        Assert.Equal(-5.0, stats.Median, Tolerance);
    }

    /// <summary>
    /// Mixed positive and negative values.
    /// </summary>
    [Fact]
    public void BasicStats_MixedPositiveNegative_HandledCorrectly()
    {
        // Arrange
        var values = new[] { -10.0, -5.0, 0.0, 5.0, 10.0 };
        var stats = CreateBasicStats(values);

        // Assert
        Assert.Equal(0.0, stats.Mean, Tolerance);
        Assert.Equal(-10.0, stats.Min, Tolerance);
        Assert.Equal(10.0, stats.Max, Tolerance);
        Assert.Equal(0.0, stats.Median, Tolerance);
    }

    #endregion

    #region Quartile Edge Cases

    /// <summary>
    /// Two values: Q1, Q2, Q3 should be interpolated.
    /// Verified with NumPy: np.percentile([1, 5], [25, 50, 75]) = [2.0, 3.0, 4.0]
    /// </summary>
    [Fact]
    public void Quartile_TwoValues_InterpolatesCorrectly()
    {
        // Arrange
        var data = new Vector<double>(new[] { 1.0, 5.0 });

        // Act
        var quartile = new Quartile<double>(data);

        // Assert - NumPy verified
        Assert.Equal(2.0, quartile.Q1, Tolerance);
        Assert.Equal(3.0, quartile.Q2, Tolerance);
        Assert.Equal(4.0, quartile.Q3, Tolerance);
    }

    /// <summary>
    /// All same values: All quartiles should equal that value.
    /// </summary>
    [Fact]
    public void Quartile_AllSameValues_AllQuartilesEqual()
    {
        // Arrange
        var data = new Vector<double>(new[] { 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0 });

        // Act
        var quartile = new Quartile<double>(data);

        // Assert
        Assert.Equal(7.0, quartile.Q1, Tolerance);
        Assert.Equal(7.0, quartile.Q2, Tolerance);
        Assert.Equal(7.0, quartile.Q3, Tolerance);
    }

    #endregion

    #region Distance Metrics Edge Cases

    /// <summary>
    /// Single dimension: Euclidean distance equals absolute difference.
    /// </summary>
    [Fact]
    public void EuclideanDistance_SingleDimension_EqualsAbsDiff()
    {
        // Arrange
        var a = new Vector<double>(new[] { -5.0 });
        var b = new Vector<double>(new[] { 3.0 });
        var metric = new EuclideanDistance<double>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert - |3 - (-5)| = 8
        Assert.Equal(8.0, distance, Tolerance);
    }

    /// <summary>
    /// High-dimensional sparse vectors.
    /// </summary>
    [Fact]
    public void EuclideanDistance_SparseVectors_ComputesCorrectly()
    {
        // Arrange - Mostly zeros with one non-zero
        var a = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 });
        var b = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 });
        var metric = new EuclideanDistance<double>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert - sqrt(1 + 1) = sqrt(2)
        Assert.Equal(Math.Sqrt(2.0), distance, Tolerance);
    }

    /// <summary>
    /// Very small differences.
    /// </summary>
    [Fact]
    public void EuclideanDistance_VerySmallDifferences_HandledCorrectly()
    {
        // Arrange
        var a = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var b = new Vector<double>(new[] { 1.0 + 1e-10, 2.0 + 1e-10, 3.0 + 1e-10 });
        var metric = new EuclideanDistance<double>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert - Should be very small but non-zero
        Assert.True(distance > 0, "Distance should be positive");
        Assert.True(distance < 1e-9, "Distance should be very small");
    }

    /// <summary>
    /// Cosine distance for unit vectors.
    /// </summary>
    [Fact]
    public void CosineDistance_UnitVectors_ComputesCorrectly()
    {
        // Arrange - Unit vectors at 60 degrees
        var a = new Vector<double>(new[] { 1.0, 0.0 });
        var b = new Vector<double>(new[] { 0.5, Math.Sqrt(3.0) / 2.0 }); // 60 degrees
        var metric = new CosineDistance<double>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert - cos(60°) = 0.5, distance = 1 - 0.5 = 0.5
        Assert.Equal(0.5, distance, Tolerance);
    }

    #endregion

    #region PredictionStats Edge Cases

    private static PredictionStats<T> CreatePredictionStats<T>(T[] actual, T[] predicted)
    {
        return new PredictionStats<T>(new PredictionStatsInputs<T>
        {
            Actual = new Vector<T>(actual),
            Predicted = new Vector<T>(predicted),
            NumberOfParameters = 1,
            ConfidenceLevel = 0.95,
            LearningCurveSteps = 5,
            PredictionType = AiDotNet.Enums.PredictionType.Regression
        });
    }

    /// <summary>
    /// Two data points: Metrics should still compute.
    /// </summary>
    [Fact]
    public void PredictionStats_TwoDataPoints_ComputesMetrics()
    {
        // Arrange
        var actual = new[] { 1.0, 5.0 };
        var predicted = new[] { 1.0, 5.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert - Perfect prediction
        Assert.Equal(1.0, stats.R2, Tolerance);
        Assert.Equal(1.0, stats.PearsonCorrelation, Tolerance);
    }

    /// <summary>
    /// Very large errors should still compute finite metrics.
    /// </summary>
    [Fact]
    public void PredictionStats_LargeErrors_ComputesFiniteMetrics()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 1000.0, 2000.0, 3000.0, 4000.0, 5000.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert - Metrics should be finite
        Assert.True(!double.IsNaN(stats.R2) && !double.IsInfinity(stats.R2),
            $"R2 should be finite, got {stats.R2}");
        Assert.True(!double.IsNaN(stats.PearsonCorrelation) && !double.IsInfinity(stats.PearsonCorrelation),
            $"Pearson should be finite, got {stats.PearsonCorrelation}");
    }

    /// <summary>
    /// Perfect negative correlation (predictions reversed).
    /// </summary>
    [Fact]
    public void PredictionStats_PerfectNegativeCorrelation_ReturnsNegativeOne()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 5.0, 4.0, 3.0, 2.0, 1.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert
        Assert.Equal(-1.0, stats.PearsonCorrelation, Tolerance);
        Assert.Equal(-1.0, stats.SpearmanCorrelation, Tolerance);
    }

    /// <summary>
    /// Very similar values with small differences.
    /// </summary>
    [Fact]
    public void PredictionStats_SmallDifferences_HighR2()
    {
        // Arrange
        var actual = new[] { 100.0, 200.0, 300.0, 400.0, 500.0 };
        var predicted = new[] { 100.001, 200.001, 300.001, 400.001, 500.001 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert - Should have very high R2
        Assert.True(stats.R2 > 0.999999,
            $"R2 should be very close to 1, got {stats.R2}");
    }

    #endregion

    #region ErrorStats Edge Cases

    private static ErrorStats<T> CreateErrorStats<T>(T[] actual, T[] predicted)
    {
        return new ErrorStats<T>(new ErrorStatsInputs<T>
        {
            Actual = new Vector<T>(actual),
            Predicted = new Vector<T>(predicted)
        });
    }

    /// <summary>
    /// Two data points: Error metrics should compute correctly.
    /// </summary>
    [Fact]
    public void ErrorStats_TwoDataPoints_ComputesCorrectly()
    {
        // Arrange
        var actual = new[] { 1.0, 5.0 };
        var predicted = new[] { 2.0, 4.0 };

        // Act
        var stats = CreateErrorStats(actual, predicted);

        // Assert
        // MAE = (|2-1| + |4-5|) / 2 = (1 + 1) / 2 = 1
        Assert.Equal(1.0, stats.MAE, Tolerance);
        // MSE = ((2-1)² + (4-5)²) / 2 = (1 + 1) / 2 = 1
        Assert.Equal(1.0, stats.MSE, Tolerance);
    }

    /// <summary>
    /// All predictions exact: All error metrics should be zero.
    /// </summary>
    [Fact]
    public void ErrorStats_PerfectPredictions_AllZeroErrors()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act
        var stats = CreateErrorStats(actual, predicted);

        // Assert
        Assert.Equal(0.0, stats.MAE, Tolerance);
        Assert.Equal(0.0, stats.MSE, Tolerance);
        Assert.Equal(0.0, stats.RMSE, Tolerance);
        Assert.Equal(0.0, stats.MeanBiasError, Tolerance);
    }

    /// <summary>
    /// Negative actual values: Error metrics should still compute.
    /// </summary>
    [Fact]
    public void ErrorStats_NegativeActualValues_ComputesCorrectly()
    {
        // Arrange
        var actual = new[] { -5.0, -3.0, -1.0, 1.0, 3.0 };
        var predicted = new[] { -4.0, -2.0, 0.0, 2.0, 4.0 };

        // Act
        var stats = CreateErrorStats(actual, predicted);

        // Assert
        // All predictions are 1 above actual, so MAE = 1
        Assert.Equal(1.0, stats.MAE, Tolerance);
        // All errors are +1, so MBE = 1
        Assert.Equal(1.0, stats.MeanBiasError, Tolerance);
    }

    #endregion

    #region Mathematical Property Tests

    /// <summary>
    /// Variance should always be non-negative.
    /// </summary>
    [Fact]
    public void BasicStats_Variance_AlwaysNonNegative()
    {
        // Test with various datasets
        var datasets = new[]
        {
            new[] { 1.0 },
            new[] { 1.0, 1.0 },
            new[] { -5.0, -3.0, -1.0, 1.0, 3.0, 5.0 },
            new[] { 1e-10, 2e-10, 3e-10 },
            new[] { 1e10, 2e10, 3e10 }
        };

        foreach (var data in datasets)
        {
            var stats = CreateBasicStats(data);
            Assert.True(stats.Variance >= 0,
                $"Variance should be non-negative, got {stats.Variance}");
            Assert.True(stats.StandardDeviation >= 0,
                $"StandardDeviation should be non-negative, got {stats.StandardDeviation}");
        }
    }

    /// <summary>
    /// Mean should be between Min and Max.
    /// </summary>
    [Fact]
    public void BasicStats_Mean_BetweenMinAndMax()
    {
        // Test with various datasets
        var datasets = new[]
        {
            new[] { 1.0, 5.0, 10.0 },
            new[] { -100.0, 0.0, 100.0 },
            new[] { 0.1, 0.2, 0.3, 0.4, 0.5 }
        };

        foreach (var data in datasets)
        {
            var stats = CreateBasicStats(data);
            Assert.True(stats.Mean >= stats.Min && stats.Mean <= stats.Max,
                $"Mean ({stats.Mean}) should be between Min ({stats.Min}) and Max ({stats.Max})");
        }
    }

    /// <summary>
    /// Pearson correlation should be in [-1, 1].
    /// </summary>
    [Fact]
    public void PredictionStats_PearsonCorrelation_InValidRange()
    {
        // Test with various datasets
        var testCases = new[]
        {
            (new[] { 1.0, 2.0, 3.0 }, new[] { 1.0, 2.0, 3.0 }),        // Perfect positive
            (new[] { 1.0, 2.0, 3.0 }, new[] { 3.0, 2.0, 1.0 }),        // Perfect negative
            (new[] { 1.0, 2.0, 3.0 }, new[] { 2.0, 2.5, 2.5 }),        // Weak correlation
            (new[] { 1.0, 2.0, 3.0, 4.0, 5.0 }, new[] { 1.1, 1.9, 3.2, 3.8, 5.1 }) // Noisy
        };

        foreach (var (actual, predicted) in testCases)
        {
            var stats = CreatePredictionStats(actual, predicted);
            Assert.True(stats.PearsonCorrelation >= -1.0 && stats.PearsonCorrelation <= 1.0,
                $"Pearson correlation ({stats.PearsonCorrelation}) should be in [-1, 1]");
        }
    }

    /// <summary>
    /// MSE should equal MAE² when all errors have the same magnitude.
    /// </summary>
    [Fact]
    public void ErrorStats_ConstantError_MSEEqualsMAESquared()
    {
        // Arrange - All errors are exactly 2
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 3.0, 4.0, 5.0, 6.0, 7.0 };

        // Act
        var stats = CreateErrorStats(actual, predicted);

        // Assert - MAE = 2, MSE = 4
        Assert.Equal(2.0, stats.MAE, Tolerance);
        Assert.Equal(4.0, stats.MSE, Tolerance);
        Assert.Equal(stats.MAE * stats.MAE, stats.MSE, Tolerance);
    }

    /// <summary>
    /// RMSE should equal sqrt(MSE).
    /// </summary>
    [Fact]
    public void ErrorStats_RMSE_EqualsSqrtMSE()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 1.5, 2.5, 2.5, 4.5, 4.5 };

        // Act
        var stats = CreateErrorStats(actual, predicted);

        // Assert
        Assert.Equal(Math.Sqrt(stats.MSE), stats.RMSE, Tolerance);
    }

    #endregion
}
