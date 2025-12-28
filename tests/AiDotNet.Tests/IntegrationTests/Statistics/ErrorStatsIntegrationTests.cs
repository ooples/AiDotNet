using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Statistics;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Statistics;

/// <summary>
/// Integration tests for ErrorStats with mathematically verified ground truth values.
/// All expected values verified against NumPy/SciPy/sklearn as authoritative sources.
///
/// These tests ensure the mathematical correctness of error metric calculations.
/// If any test fails, the CODE must be fixed - never adjust the expected values.
/// </summary>
public class ErrorStatsIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region MAE (Mean Absolute Error) Tests

    /// <summary>
    /// Verified with sklearn: sklearn.metrics.mean_absolute_error([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]) = 0.5
    /// </summary>
    [Fact]
    public void MAE_StandardDataset_ReturnsExactValue()
    {
        // Arrange
        var actual = new Vector<double>(new[] { 3.0, -0.5, 2.0, 7.0 });
        var predicted = new Vector<double>(new[] { 2.5, 0.0, 2.0, 8.0 });

        // Act
        var inputs = new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1,
            PredictionType = PredictionType.Regression
        };
        var stats = new ErrorStats<double>(inputs);

        // Assert - sklearn verified: mean_absolute_error = 0.5
        Assert.Equal(0.5, stats.MAE, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy: np.mean(np.abs([1, 2, 3] - [1, 2, 3])) = 0.0
    /// </summary>
    [Fact]
    public void MAE_PerfectPrediction_ReturnsZero()
    {
        // Arrange
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        // Act
        var inputs = new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1,
            PredictionType = PredictionType.Regression
        };
        var stats = new ErrorStats<double>(inputs);

        // Assert
        Assert.Equal(0.0, stats.MAE, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy: np.mean(np.abs([0, 0, 0, 0] - [1, 1, 1, 1])) = 1.0
    /// </summary>
    [Fact]
    public void MAE_ConstantError_ReturnsConstant()
    {
        // Arrange - Each prediction is off by 1
        var actual = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0 });
        var predicted = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0 });

        // Act
        var inputs = new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1,
            PredictionType = PredictionType.Regression
        };
        var stats = new ErrorStats<double>(inputs);

        // Assert
        Assert.Equal(1.0, stats.MAE, Tolerance);
    }

    #endregion

    #region MSE (Mean Squared Error) Tests

    /// <summary>
    /// Verified with sklearn: sklearn.metrics.mean_squared_error([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]) = 0.375
    /// </summary>
    [Fact]
    public void MSE_StandardDataset_ReturnsExactValue()
    {
        // Arrange
        var actual = new Vector<double>(new[] { 3.0, -0.5, 2.0, 7.0 });
        var predicted = new Vector<double>(new[] { 2.5, 0.0, 2.0, 8.0 });

        // Act
        var inputs = new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1,
            PredictionType = PredictionType.Regression
        };
        var stats = new ErrorStats<double>(inputs);

        // Assert - sklearn verified: mean_squared_error = 0.375
        // (0.5^2 + 0.5^2 + 0^2 + 1^2) / 4 = (0.25 + 0.25 + 0 + 1) / 4 = 1.5 / 4 = 0.375
        Assert.Equal(0.375, stats.MSE, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy: np.mean(([1, 2, 3, 4, 5] - [1, 2, 3, 4, 5])**2) = 0.0
    /// </summary>
    [Fact]
    public void MSE_PerfectPrediction_ReturnsZero()
    {
        // Arrange
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        // Act
        var inputs = new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1,
            PredictionType = PredictionType.Regression
        };
        var stats = new ErrorStats<double>(inputs);

        // Assert
        Assert.Equal(0.0, stats.MSE, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy: np.mean(([0, 0, 0, 0] - [2, 2, 2, 2])**2) = 4.0
    /// </summary>
    [Fact]
    public void MSE_ConstantError_ReturnsSquaredError()
    {
        // Arrange - Each prediction is off by 2
        var actual = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0 });
        var predicted = new Vector<double>(new[] { 2.0, 2.0, 2.0, 2.0 });

        // Act
        var inputs = new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1,
            PredictionType = PredictionType.Regression
        };
        var stats = new ErrorStats<double>(inputs);

        // Assert - 2^2 = 4
        Assert.Equal(4.0, stats.MSE, Tolerance);
    }

    #endregion

    #region RMSE (Root Mean Squared Error) Tests

    /// <summary>
    /// Verified with sklearn: np.sqrt(sklearn.metrics.mean_squared_error([3, -0.5, 2, 7], [2.5, 0.0, 2, 8])) = 0.6123724356957945
    /// </summary>
    [Fact]
    public void RMSE_StandardDataset_ReturnsExactValue()
    {
        // Arrange
        var actual = new Vector<double>(new[] { 3.0, -0.5, 2.0, 7.0 });
        var predicted = new Vector<double>(new[] { 2.5, 0.0, 2.0, 8.0 });

        // Act
        var inputs = new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1,
            PredictionType = PredictionType.Regression
        };
        var stats = new ErrorStats<double>(inputs);

        // Assert - sqrt(0.375) = 0.6123724356957945
        Assert.Equal(0.6123724356957945, stats.RMSE, Tolerance);
    }

    /// <summary>
    /// RMSE = sqrt(MSE), so RMSE of 2.0 error should be 2.0
    /// </summary>
    [Fact]
    public void RMSE_ConstantError_ReturnsSqrtOfMSE()
    {
        // Arrange
        var actual = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0 });
        var predicted = new Vector<double>(new[] { 2.0, 2.0, 2.0, 2.0 });

        // Act
        var inputs = new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1,
            PredictionType = PredictionType.Regression
        };
        var stats = new ErrorStats<double>(inputs);

        // Assert - sqrt(4) = 2
        Assert.Equal(2.0, stats.RMSE, Tolerance);
    }

    #endregion

    #region MAPE (Mean Absolute Percentage Error) Tests

    /// <summary>
    /// MAPE = mean(|actual - predicted| / |actual|) * 100
    /// For [100, 200, 300] vs [90, 190, 310]:
    /// = mean([10/100, 10/200, 10/300]) * 100
    /// = mean([0.1, 0.05, 0.0333...]) * 100
    /// = 0.0611... * 100 = 6.11...
    /// </summary>
    [Fact]
    public void MAPE_StandardDataset_ReturnsExactValue()
    {
        // Arrange
        var actual = new Vector<double>(new[] { 100.0, 200.0, 300.0 });
        var predicted = new Vector<double>(new[] { 90.0, 190.0, 310.0 });

        // Act
        var inputs = new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1,
            PredictionType = PredictionType.Regression
        };
        var stats = new ErrorStats<double>(inputs);

        // Assert - (10/100 + 10/200 + 10/300) / 3 * 100 = 6.111...
        double expectedMAPE = ((10.0/100.0) + (10.0/200.0) + (10.0/300.0)) / 3.0 * 100.0;
        Assert.Equal(expectedMAPE, stats.MAPE, 0.01); // Slightly relaxed tolerance for percentage
    }

    #endregion

    #region MeanBiasError Tests

    /// <summary>
    /// MeanBiasError = mean(predicted - actual)
    /// Positive bias means model over-predicts on average.
    /// </summary>
    [Fact]
    public void MeanBiasError_OverPrediction_ReturnsPositive()
    {
        // Arrange - Predictions are consistently higher
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var predicted = new Vector<double>(new[] { 2.0, 3.0, 4.0, 5.0, 6.0 });

        // Act
        var inputs = new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1,
            PredictionType = PredictionType.Regression
        };
        var stats = new ErrorStats<double>(inputs);

        // Assert - Each prediction is 1 higher, so bias = 1.0
        Assert.Equal(1.0, stats.MeanBiasError, Tolerance);
    }

    /// <summary>
    /// MeanBiasError = mean(predicted - actual)
    /// Negative bias means model under-predicts on average.
    /// </summary>
    [Fact]
    public void MeanBiasError_UnderPrediction_ReturnsNegative()
    {
        // Arrange - Predictions are consistently lower
        var actual = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });
        var predicted = new Vector<double>(new[] { 8.0, 18.0, 28.0, 38.0, 48.0 });

        // Act
        var inputs = new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1,
            PredictionType = PredictionType.Regression
        };
        var stats = new ErrorStats<double>(inputs);

        // Assert - Each prediction is 2 lower, so bias = -2.0
        Assert.Equal(-2.0, stats.MeanBiasError, Tolerance);
    }

    /// <summary>
    /// MeanBiasError = 0 when errors cancel out.
    /// </summary>
    [Fact]
    public void MeanBiasError_BalancedErrors_ReturnsZero()
    {
        // Arrange - Errors cancel out: +1, -1, +1, -1
        var actual = new Vector<double>(new[] { 5.0, 5.0, 5.0, 5.0 });
        var predicted = new Vector<double>(new[] { 6.0, 4.0, 6.0, 4.0 });

        // Act
        var inputs = new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1,
            PredictionType = PredictionType.Regression
        };
        var stats = new ErrorStats<double>(inputs);

        // Assert
        Assert.Equal(0.0, stats.MeanBiasError, Tolerance);
    }

    #endregion

    #region MaxError Tests

    /// <summary>
    /// MaxError = max(|actual - predicted|)
    /// </summary>
    [Fact]
    public void MaxError_ReturnsLargestAbsoluteError()
    {
        // Arrange - Errors: 0.5, 0.5, 0, 1 -> Max = 1
        var actual = new Vector<double>(new[] { 3.0, -0.5, 2.0, 7.0 });
        var predicted = new Vector<double>(new[] { 2.5, 0.0, 2.0, 8.0 });

        // Act
        var inputs = new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1,
            PredictionType = PredictionType.Regression
        };
        var stats = new ErrorStats<double>(inputs);

        // Assert - Max absolute error is 1.0 (|7 - 8| = 1)
        Assert.Equal(1.0, stats.MaxError, Tolerance);
    }

    /// <summary>
    /// MaxError should find maximum regardless of sign.
    /// </summary>
    [Fact]
    public void MaxError_NegativeError_ReturnsAbsoluteMax()
    {
        // Arrange - Errors: 1, -5, 2 -> Max = 5
        var actual = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
        var predicted = new Vector<double>(new[] { 9.0, 25.0, 28.0 });

        // Act
        var inputs = new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1,
            PredictionType = PredictionType.Regression
        };
        var stats = new ErrorStats<double>(inputs);

        // Assert - Max absolute error is 5.0 (|20 - 25| = 5)
        Assert.Equal(5.0, stats.MaxError, Tolerance);
    }

    #endregion

    #region MedianAbsoluteError Tests

    /// <summary>
    /// MedianAbsoluteError = median of |actual - predicted|
    /// More robust to outliers than MAE.
    /// </summary>
    [Fact]
    public void MedianAbsoluteError_StandardDataset_ReturnsExactValue()
    {
        // Arrange - Errors: |0.5|, |0.5|, |0|, |1| = [0, 0.5, 0.5, 1]
        // Sorted: [0, 0.5, 0.5, 1] -> Median = (0.5 + 0.5) / 2 = 0.5
        var actual = new Vector<double>(new[] { 3.0, -0.5, 2.0, 7.0 });
        var predicted = new Vector<double>(new[] { 2.5, 0.0, 2.0, 8.0 });

        // Act
        var inputs = new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1,
            PredictionType = PredictionType.Regression
        };
        var stats = new ErrorStats<double>(inputs);

        // Assert
        Assert.Equal(0.5, stats.MedianAbsoluteError, Tolerance);
    }

    #endregion

    #region SMAPE (Symmetric Mean Absolute Percentage Error) Tests

    /// <summary>
    /// SMAPE = mean(2 * |actual - predicted| / (|actual| + |predicted|)) * 100
    /// Bounded between 0% and 200%.
    /// </summary>
    [Fact]
    public void SMAPE_StandardDataset_ReturnsBoundedValue()
    {
        // Arrange
        var actual = new Vector<double>(new[] { 100.0, 200.0, 300.0 });
        var predicted = new Vector<double>(new[] { 90.0, 190.0, 310.0 });

        // Act
        var inputs = new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1,
            PredictionType = PredictionType.Regression
        };
        var stats = new ErrorStats<double>(inputs);

        // Assert - SMAPE should be between 0 and 200
        Assert.True(stats.SMAPE >= 0.0 && stats.SMAPE <= 200.0,
            $"SMAPE should be between 0 and 200, was {stats.SMAPE}");
    }

    /// <summary>
    /// SMAPE = 0 for perfect predictions.
    /// </summary>
    [Fact]
    public void SMAPE_PerfectPrediction_ReturnsZero()
    {
        // Arrange
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        // Act
        var inputs = new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1,
            PredictionType = PredictionType.Regression
        };
        var stats = new ErrorStats<double>(inputs);

        // Assert
        Assert.Equal(0.0, stats.SMAPE, Tolerance);
    }

    #endregion

    #region RSS (Residual Sum of Squares) Tests

    /// <summary>
    /// RSS = sum((actual - predicted)^2)
    /// </summary>
    [Fact]
    public void RSS_StandardDataset_ReturnsExactValue()
    {
        // Arrange
        var actual = new Vector<double>(new[] { 3.0, -0.5, 2.0, 7.0 });
        var predicted = new Vector<double>(new[] { 2.5, 0.0, 2.0, 8.0 });

        // Act
        var inputs = new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1,
            PredictionType = PredictionType.Regression
        };
        var stats = new ErrorStats<double>(inputs);

        // Assert - RSS = 0.5^2 + 0.5^2 + 0^2 + 1^2 = 0.25 + 0.25 + 0 + 1 = 1.5
        Assert.Equal(1.5, stats.RSS, Tolerance);
    }

    #endregion

    #region GetMetric Tests

    [Fact]
    public void GetMetric_MAE_ReturnsCorrectValue()
    {
        // Arrange
        var actual = new Vector<double>(new[] { 3.0, -0.5, 2.0, 7.0 });
        var predicted = new Vector<double>(new[] { 2.5, 0.0, 2.0, 8.0 });
        var inputs = new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1,
            PredictionType = PredictionType.Regression
        };
        var stats = new ErrorStats<double>(inputs);

        // Act
        var mae = stats.GetMetric(MetricType.MAE);

        // Assert
        Assert.Equal(stats.MAE, mae, Tolerance);
    }

    [Fact]
    public void GetMetric_AllAliases_MatchProperties()
    {
        // Arrange
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var predicted = new Vector<double>(new[] { 1.1, 2.2, 3.3 });
        var inputs = new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1,
            PredictionType = PredictionType.Regression
        };
        var stats = new ErrorStats<double>(inputs);

        // Act & Assert - Verify aliases match properties
        Assert.Equal(stats.MAE, stats.MeanAbsoluteError, Tolerance);
        Assert.Equal(stats.MSE, stats.MeanSquaredError, Tolerance);
        Assert.Equal(stats.RMSE, stats.RootMeanSquaredError, Tolerance);
        Assert.Equal(stats.AUCROC, stats.AUC, Tolerance);
    }

    #endregion

    #region ErrorList Tests

    [Fact]
    public void ErrorList_ContainsResiduals()
    {
        // Arrange
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var predicted = new Vector<double>(new[] { 1.5, 2.0, 2.5 });
        var inputs = new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1,
            PredictionType = PredictionType.Regression
        };

        // Act
        var stats = new ErrorStats<double>(inputs);

        // Assert - ErrorList should contain residuals (actual - predicted)
        Assert.Equal(3, stats.ErrorList.Count);
        Assert.Equal(-0.5, stats.ErrorList[0], Tolerance); // 1.0 - 1.5 = -0.5
        Assert.Equal(0.0, stats.ErrorList[1], Tolerance);  // 2.0 - 2.0 = 0.0
        Assert.Equal(0.5, stats.ErrorList[2], Tolerance);  // 3.0 - 2.5 = 0.5
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void MAE_FloatType_ReturnsCorrectValue()
    {
        // Arrange
        var actual = new Vector<float>(new[] { 3.0f, -0.5f, 2.0f, 7.0f });
        var predicted = new Vector<float>(new[] { 2.5f, 0.0f, 2.0f, 8.0f });
        var inputs = new ErrorStatsInputs<float>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1,
            PredictionType = PredictionType.Regression
        };

        // Act
        var stats = new ErrorStats<float>(inputs);

        // Assert
        Assert.Equal(0.5f, stats.MAE, 1e-5f);
    }

    #endregion
}
