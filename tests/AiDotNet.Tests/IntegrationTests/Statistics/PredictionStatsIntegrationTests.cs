using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Statistics;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Statistics;

/// <summary>
/// Integration tests for PredictionStats with mathematically verified ground truth values.
/// All expected values verified against NumPy/SciPy as authoritative sources.
///
/// These tests ensure the mathematical correctness of prediction statistics calculations.
/// If any test fails, the CODE must be fixed - never adjust the expected values.
/// </summary>
public class PredictionStatsIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Helper Methods

    private static PredictionStats<T> CreatePredictionStats<T>(
        T[] actual,
        T[] predicted,
        int numberOfParameters = 1,
        double confidenceLevel = 0.95,
        PredictionType predictionType = PredictionType.Regression)
    {
        return new PredictionStats<T>(new PredictionStatsInputs<T>
        {
            Actual = new Vector<T>(actual),
            Predicted = new Vector<T>(predicted),
            NumberOfParameters = numberOfParameters,
            ConfidenceLevel = confidenceLevel,
            LearningCurveSteps = 5,
            PredictionType = predictionType
        });
    }

    #endregion

    #region R-Squared (Coefficient of Determination) Tests

    /// <summary>
    /// Perfect prediction: R2 = 1.0
    /// When actual == predicted, all variance is explained.
    /// </summary>
    [Fact]
    public void R2_PerfectPrediction_ReturnsOne()
    {
        // Arrange - Perfect predictions
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert - R2 should be exactly 1.0 for perfect prediction
        Assert.Equal(1.0, stats.R2, Tolerance);
    }

    /// <summary>
    /// Verified with sklearn:
    /// from sklearn.metrics import r2_score
    /// r2_score([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]) = 0.9486081370449679
    /// </summary>
    [Fact]
    public void R2_SklearnExample_ReturnsExactValue()
    {
        // Arrange - sklearn documentation example
        var actual = new[] { 3.0, -0.5, 2.0, 7.0 };
        var predicted = new[] { 2.5, 0.0, 2.0, 8.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert - sklearn verified: r2_score([3, -0.5, 2, 7], [2.5, 0.0, 2, 8])
        Assert.Equal(0.9486081370449679, stats.R2, Tolerance);
    }

    /// <summary>
    /// Predicting the mean for all values gives R2 = 0.
    /// R2 = 1 - SS_res/SS_tot, when predicting mean, SS_res = SS_tot.
    /// </summary>
    [Fact]
    public void R2_PredictMean_ReturnsZero()
    {
        // Arrange - Predict mean for all values
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var mean = 3.0; // Mean of actual
        var predicted = new[] { mean, mean, mean, mean, mean };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert - R2 should be 0 when predicting mean
        Assert.Equal(0.0, stats.R2, Tolerance);
    }

    /// <summary>
    /// When predictions are worse than mean, R2 can be negative.
    /// </summary>
    [Fact]
    public void R2_WorseThanMean_ReturnsNegative()
    {
        // Arrange - Predictions that are worse than predicting mean
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 5.0, 4.0, 3.0, 2.0, 1.0 }; // Reversed

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert - R2 should be negative when predictions are worse than mean
        Assert.True(stats.R2 < 0, $"R2 should be negative for predictions worse than mean, got {stats.R2}");
    }

    #endregion

    #region Adjusted R-Squared Tests

    /// <summary>
    /// Adjusted R2 should be less than or equal to R2.
    /// Adjusted R2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)
    /// </summary>
    [Fact]
    public void AdjustedR2_IsLessThanOrEqualToR2()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        var predicted = new[] { 1.1, 1.9, 3.1, 4.0, 4.9, 6.1, 6.9, 8.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted, numberOfParameters: 2);

        // Assert - Adjusted R2 should be <= R2
        Assert.True(stats.AdjustedR2 <= stats.R2,
            $"Adjusted R2 ({stats.AdjustedR2}) should be <= R2 ({stats.R2})");
    }

    /// <summary>
    /// Perfect predictions should give Adjusted R2 = 1.0.
    /// </summary>
    [Fact]
    public void AdjustedR2_PerfectPrediction_ReturnsOne()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        var predicted = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted, numberOfParameters: 1);

        // Assert
        Assert.Equal(1.0, stats.AdjustedR2, Tolerance);
    }

    #endregion

    #region Pearson Correlation Tests

    /// <summary>
    /// Verified with NumPy: np.corrcoef([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])[0, 1] = 1.0
    /// </summary>
    [Fact]
    public void PearsonCorrelation_PerfectCorrelation_ReturnsOne()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert
        Assert.Equal(1.0, stats.PearsonCorrelation, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy: np.corrcoef([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])[0, 1] = -1.0
    /// </summary>
    [Fact]
    public void PearsonCorrelation_PerfectNegativeCorrelation_ReturnsNegativeOne()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 5.0, 4.0, 3.0, 2.0, 1.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert
        Assert.Equal(-1.0, stats.PearsonCorrelation, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy:
    /// np.corrcoef([10, 20, 30, 40, 50], [12, 25, 28, 45, 48])[0, 1] = 0.9756366365628593
    ///
    /// Manual verification:
    /// x̄ = 30, ȳ = 31.6
    /// Σ(xi - x̄)(yi - ȳ) = 920
    /// Σ(xi - x̄)² = 1000
    /// Σ(yi - ȳ)² = 889.2
    /// r = 920 / sqrt(1000 * 889.2) = 0.97563663656285926
    /// </summary>
    [Fact]
    public void PearsonCorrelation_HighCorrelation_ReturnsExactValue()
    {
        // Arrange
        var actual = new[] { 10.0, 20.0, 30.0, 40.0, 50.0 };
        var predicted = new[] { 12.0, 25.0, 28.0, 45.0, 48.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert - NumPy verified
        Assert.Equal(0.9756366365628593, stats.PearsonCorrelation, Tolerance);
    }

    /// <summary>
    /// Linear transformation should preserve correlation.
    /// y' = a*y + b should have same correlation as y with x.
    /// </summary>
    [Fact]
    public void PearsonCorrelation_ScaledPredictions_SameCorrelation()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 10.0, 20.0, 30.0, 40.0, 50.0 }; // Scaled by 10
        var predictedShifted = new[] { 11.0, 21.0, 31.0, 41.0, 51.0 }; // Scaled and shifted

        // Act
        var statsScaled = CreatePredictionStats(actual, predicted);
        var statsShifted = CreatePredictionStats(actual, predictedShifted);

        // Assert - Both should have correlation = 1.0
        Assert.Equal(1.0, statsScaled.PearsonCorrelation, Tolerance);
        Assert.Equal(1.0, statsShifted.PearsonCorrelation, Tolerance);
    }

    #endregion

    #region Spearman Correlation Tests

    /// <summary>
    /// Verified with SciPy: scipy.stats.spearmanr([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])[0] = 1.0
    /// </summary>
    [Fact]
    public void SpearmanCorrelation_PerfectRankCorrelation_ReturnsOne()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert
        Assert.Equal(1.0, stats.SpearmanCorrelation, Tolerance);
    }

    /// <summary>
    /// Verified with SciPy: scipy.stats.spearmanr([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])[0] = -1.0
    /// </summary>
    [Fact]
    public void SpearmanCorrelation_PerfectNegativeRankCorrelation_ReturnsNegativeOne()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 5.0, 4.0, 3.0, 2.0, 1.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert
        Assert.Equal(-1.0, stats.SpearmanCorrelation, Tolerance);
    }

    /// <summary>
    /// Spearman correlation should be robust to monotonic transformations.
    /// If y increases monotonically with x, correlation = 1.
    /// </summary>
    [Fact]
    public void SpearmanCorrelation_MonotonicTransformation_ReturnsOne()
    {
        // Arrange - Monotonic but non-linear transformation
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 1.0, 4.0, 9.0, 16.0, 25.0 }; // Squared

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert - Spearman should be 1.0 for perfect monotonic relationship
        Assert.Equal(1.0, stats.SpearmanCorrelation, Tolerance);
    }

    #endregion

    #region Kendall Tau Tests

    /// <summary>
    /// Verified with SciPy: scipy.stats.kendalltau([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])[0] = 1.0
    /// </summary>
    [Fact]
    public void KendallTau_PerfectConcordance_ReturnsOne()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert
        Assert.Equal(1.0, stats.KendallTau, Tolerance);
    }

    /// <summary>
    /// Verified with SciPy: scipy.stats.kendalltau([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])[0] = -1.0
    /// </summary>
    [Fact]
    public void KendallTau_PerfectDiscordance_ReturnsNegativeOne()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 5.0, 4.0, 3.0, 2.0, 1.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert
        Assert.Equal(-1.0, stats.KendallTau, Tolerance);
    }

    #endregion

    #region Explained Variance Score Tests

    /// <summary>
    /// Perfect predictions should give ExplainedVarianceScore = 1.0.
    /// </summary>
    [Fact]
    public void ExplainedVarianceScore_PerfectPrediction_ReturnsOne()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert
        Assert.Equal(1.0, stats.ExplainedVarianceScore, Tolerance);
    }

    /// <summary>
    /// Explained variance score should be 1 even with constant bias.
    /// y_pred = y_actual + constant should have EVS = 1.
    /// </summary>
    [Fact]
    public void ExplainedVarianceScore_WithConstantBias_ReturnsOne()
    {
        // Arrange - Predictions with constant bias
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 6.0, 7.0, 8.0, 9.0, 10.0 }; // All shifted by 5

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert - EVS should be 1.0 despite constant bias
        Assert.Equal(1.0, stats.ExplainedVarianceScore, Tolerance);
    }

    #endregion

    #region Mean Prediction Error Tests

    /// <summary>
    /// Mean prediction error should be zero for perfect predictions.
    /// </summary>
    [Fact]
    public void MeanPredictionError_PerfectPrediction_ReturnsZero()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert
        Assert.Equal(0.0, stats.MeanPredictionError, Tolerance);
    }

    /// <summary>
    /// Mean prediction error should be positive for overprediction.
    /// MeanPredictionError = mean(predicted - actual)
    /// </summary>
    [Fact]
    public void MeanPredictionError_Overprediction_ReturnsPositive()
    {
        // Arrange - Predictions systematically higher
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 2.0, 3.0, 4.0, 5.0, 6.0 }; // All +1

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert - Mean error should be 1.0
        Assert.Equal(1.0, stats.MeanPredictionError, Tolerance);
    }

    #endregion

    #region Median Prediction Error Tests

    /// <summary>
    /// Median prediction error should be zero for perfect predictions.
    /// </summary>
    [Fact]
    public void MedianPredictionError_PerfectPrediction_ReturnsZero()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert
        Assert.Equal(0.0, stats.MedianPredictionError, Tolerance);
    }

    /// <summary>
    /// Median prediction error for constant overprediction.
    /// </summary>
    [Fact]
    public void MedianPredictionError_ConstantOverprediction_ReturnsConstant()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 3.0, 4.0, 5.0, 6.0, 7.0 }; // All +2

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert - Median of [2, 2, 2, 2, 2] = 2
        Assert.Equal(2.0, stats.MedianPredictionError, Tolerance);
    }

    #endregion

    #region Dynamic Time Warping Tests

    /// <summary>
    /// DTW distance should be zero for identical sequences.
    /// </summary>
    [Fact]
    public void DynamicTimeWarping_IdenticalSequences_ReturnsZero()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert - DTW should be 0 for identical sequences
        Assert.Equal(0.0, stats.DynamicTimeWarping, Tolerance);
    }

    /// <summary>
    /// DTW distance should be positive for different sequences.
    /// </summary>
    [Fact]
    public void DynamicTimeWarping_DifferentSequences_ReturnsPositive()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 2.0, 3.0, 4.0, 5.0, 6.0 }; // Shifted by 1

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert - DTW should be > 0 for different sequences
        Assert.True(stats.DynamicTimeWarping > 0,
            $"DTW should be positive for different sequences, got {stats.DynamicTimeWarping}");
    }

    #endregion

    #region Prediction Interval Tests

    /// <summary>
    /// Prediction interval should have lower < upper.
    /// </summary>
    [Fact]
    public void PredictionInterval_HasValidBounds()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        var predicted = new[] { 1.1, 2.1, 2.9, 4.0, 5.1, 5.9, 7.0, 8.1 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert - Lower should be less than Upper
        Assert.True(stats.PredictionInterval.Lower < stats.PredictionInterval.Upper,
            $"Prediction interval lower ({stats.PredictionInterval.Lower}) should be < upper ({stats.PredictionInterval.Upper})");
    }

    /// <summary>
    /// Perfect predictions should have narrow prediction interval.
    /// </summary>
    [Fact]
    public void PredictionInterval_PerfectPredictions_NarrowInterval()
    {
        // Arrange - Perfect predictions
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        var predicted = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert - For perfect predictions, the interval width should be minimal
        var width = stats.PredictionInterval.Upper - stats.PredictionInterval.Lower;
        Assert.True(width < 1.0, $"Prediction interval width ({width}) should be narrow for perfect predictions");
    }

    #endregion

    #region Confidence Interval Tests

    /// <summary>
    /// Confidence interval should have lower < upper.
    /// </summary>
    [Fact]
    public void ConfidenceInterval_HasValidBounds()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        var predicted = new[] { 1.1, 2.0, 3.1, 4.0, 5.1, 6.0, 7.1, 8.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted, confidenceLevel: 0.95);

        // Assert
        Assert.True(stats.ConfidenceInterval.Lower < stats.ConfidenceInterval.Upper,
            $"Confidence interval lower ({stats.ConfidenceInterval.Lower}) should be < upper ({stats.ConfidenceInterval.Upper})");
    }

    #endregion

    #region Credible Interval Tests

    /// <summary>
    /// Credible interval should have valid bounds.
    /// </summary>
    [Fact]
    public void CredibleInterval_HasValidBounds()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        var predicted = new[] { 1.1, 2.0, 3.1, 4.0, 5.1, 6.0, 7.1, 8.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert - For non-degenerate data, lower should be less than upper
        Assert.True(stats.CredibleInterval.Lower <= stats.CredibleInterval.Upper,
            $"Credible interval lower ({stats.CredibleInterval.Lower}) should be <= upper ({stats.CredibleInterval.Upper})");
    }

    #endregion

    #region Tolerance Interval Tests

    /// <summary>
    /// Tolerance interval should have valid bounds.
    /// </summary>
    [Fact]
    public void ToleranceInterval_HasValidBounds()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        var predicted = new[] { 1.1, 2.0, 3.1, 4.0, 5.1, 6.0, 7.1, 8.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert
        Assert.True(stats.ToleranceInterval.Lower <= stats.ToleranceInterval.Upper,
            $"Tolerance interval lower ({stats.ToleranceInterval.Lower}) should be <= upper ({stats.ToleranceInterval.Upper})");
    }

    #endregion

    #region Bootstrap Interval Tests

    /// <summary>
    /// Bootstrap interval should have valid bounds.
    /// </summary>
    [Fact]
    public void BootstrapInterval_HasValidBounds()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        var predicted = new[] { 1.1, 2.0, 3.1, 4.0, 5.1, 6.0, 7.1, 8.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert
        Assert.True(stats.BootstrapInterval.Lower <= stats.BootstrapInterval.Upper,
            $"Bootstrap interval lower ({stats.BootstrapInterval.Lower}) should be <= upper ({stats.BootstrapInterval.Upper})");
    }

    #endregion

    #region Jackknife Interval Tests

    /// <summary>
    /// Jackknife interval should have valid bounds.
    /// </summary>
    [Fact]
    public void JackknifeInterval_HasValidBounds()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        var predicted = new[] { 1.1, 2.0, 3.1, 4.0, 5.1, 6.0, 7.1, 8.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert
        Assert.True(stats.JackknifeInterval.Lower <= stats.JackknifeInterval.Upper,
            $"Jackknife interval lower ({stats.JackknifeInterval.Lower}) should be <= upper ({stats.JackknifeInterval.Upper})");
    }

    #endregion

    #region Forecast Interval Tests

    /// <summary>
    /// Forecast interval should have valid bounds.
    /// </summary>
    [Fact]
    public void ForecastInterval_HasValidBounds()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        var predicted = new[] { 1.1, 2.0, 3.1, 4.0, 5.1, 6.0, 7.1, 8.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert
        Assert.True(stats.ForecastInterval.Lower <= stats.ForecastInterval.Upper,
            $"Forecast interval lower ({stats.ForecastInterval.Lower}) should be <= upper ({stats.ForecastInterval.Upper})");
    }

    #endregion

    #region Percentile Interval Tests

    /// <summary>
    /// Percentile interval should have valid bounds.
    /// </summary>
    [Fact]
    public void PercentileInterval_HasValidBounds()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        var predicted = new[] { 1.1, 2.0, 3.1, 4.0, 5.1, 6.0, 7.1, 8.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert
        Assert.True(stats.PercentileInterval.Lower <= stats.PercentileInterval.Upper,
            $"Percentile interval lower ({stats.PercentileInterval.Lower}) should be <= upper ({stats.PercentileInterval.Upper})");
    }

    #endregion

    #region Simultaneous Prediction Interval Tests

    /// <summary>
    /// Simultaneous prediction interval should have valid bounds.
    /// </summary>
    [Fact]
    public void SimultaneousPredictionInterval_HasValidBounds()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        var predicted = new[] { 1.1, 2.0, 3.1, 4.0, 5.1, 6.0, 7.1, 8.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert
        Assert.True(stats.SimultaneousPredictionInterval.Lower <= stats.SimultaneousPredictionInterval.Upper,
            $"Simultaneous prediction interval lower ({stats.SimultaneousPredictionInterval.Lower}) should be <= upper ({stats.SimultaneousPredictionInterval.Upper})");
    }

    #endregion

    #region Prediction Interval Coverage Tests

    /// <summary>
    /// Prediction interval coverage should be between 0 and 1.
    /// </summary>
    [Fact]
    public void PredictionIntervalCoverage_IsValidProportion()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        var predicted = new[] { 1.1, 2.0, 3.1, 4.0, 5.1, 6.0, 7.1, 8.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert - Coverage should be between 0 and 1
        Assert.True(stats.PredictionIntervalCoverage >= 0.0 && stats.PredictionIntervalCoverage <= 1.0,
            $"Coverage ({stats.PredictionIntervalCoverage}) should be between 0 and 1");
    }

    #endregion

    #region Quantile Intervals Tests

    /// <summary>
    /// Quantile intervals list should be initialized.
    /// </summary>
    [Fact]
    public void QuantileIntervals_IsInitialized()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert
        Assert.NotNull(stats.QuantileIntervals);
    }

    #endregion

    #region Learning Curve Tests

    /// <summary>
    /// Learning curve should be initialized as a list.
    /// </summary>
    [Fact]
    public void LearningCurve_IsInitialized()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert
        Assert.NotNull(stats.LearningCurve);
    }

    #endregion

    #region Classification Metrics Tests (Accuracy, Precision, Recall, F1Score)

    /// <summary>
    /// Accuracy should be 1.0 for perfect classification.
    /// </summary>
    [Fact]
    public void Accuracy_PerfectClassification_ReturnsOne()
    {
        // Arrange - Binary classification (0 or 1)
        var actual = new[] { 0.0, 0.0, 1.0, 1.0, 1.0 };
        var predicted = new[] { 0.0, 0.0, 1.0, 1.0, 1.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted, predictionType: PredictionType.Binary);

        // Assert
        Assert.Equal(1.0, stats.Accuracy, Tolerance);
    }

    /// <summary>
    /// Precision should be 1.0 when all positive predictions are correct.
    /// </summary>
    [Fact]
    public void Precision_AllCorrectPositives_ReturnsOne()
    {
        // Arrange - All predicted positives are true positives
        var actual = new[] { 1.0, 1.0, 0.0, 0.0, 1.0 };
        var predicted = new[] { 1.0, 1.0, 0.0, 0.0, 1.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted, predictionType: PredictionType.Binary);

        // Assert
        Assert.Equal(1.0, stats.Precision, Tolerance);
    }

    /// <summary>
    /// Recall should be 1.0 when all actual positives are detected.
    /// </summary>
    [Fact]
    public void Recall_AllPositivesDetected_ReturnsOne()
    {
        // Arrange - All actual positives are predicted
        var actual = new[] { 1.0, 1.0, 0.0, 0.0, 1.0 };
        var predicted = new[] { 1.0, 1.0, 0.0, 0.0, 1.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted, predictionType: PredictionType.Binary);

        // Assert
        Assert.Equal(1.0, stats.Recall, Tolerance);
    }

    /// <summary>
    /// F1Score should be 1.0 for perfect classification.
    /// F1 = 2 * (precision * recall) / (precision + recall)
    /// </summary>
    [Fact]
    public void F1Score_PerfectClassification_ReturnsOne()
    {
        // Arrange
        var actual = new[] { 1.0, 1.0, 0.0, 0.0, 1.0 };
        var predicted = new[] { 1.0, 1.0, 0.0, 0.0, 1.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted, predictionType: PredictionType.Binary);

        // Assert
        Assert.Equal(1.0, stats.F1Score, Tolerance);
    }

    /// <summary>
    /// F1Score should be the harmonic mean of precision and recall.
    /// </summary>
    [Fact]
    public void F1Score_IsHarmonicMeanOfPrecisionAndRecall()
    {
        // Arrange
        var actual = new[] { 1.0, 1.0, 0.0, 0.0, 1.0 };
        var predicted = new[] { 1.0, 1.0, 0.0, 0.0, 1.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted, predictionType: PredictionType.Binary);

        // Assert - F1 = 2 * (P * R) / (P + R)
        var expectedF1 = 2.0 * (stats.Precision * stats.Recall) / (stats.Precision + stats.Recall);
        Assert.Equal(expectedF1, stats.F1Score, Tolerance);
    }

    #endregion

    #region RSquared Alias Test

    /// <summary>
    /// RSquared property should be an alias for R2.
    /// </summary>
    [Fact]
    public void RSquared_IsAliasForR2()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 1.1, 2.0, 3.0, 4.1, 4.9 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert - RSquared should equal R2
        Assert.Equal(stats.R2, stats.RSquared, Tolerance);
    }

    #endregion

    #region Best Distribution Fit Tests

    /// <summary>
    /// BestDistributionFit should be initialized.
    /// </summary>
    [Fact]
    public void BestDistributionFit_IsInitialized()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        var predicted = new[] { 1.1, 2.0, 3.1, 4.0, 5.1, 6.0, 7.1, 8.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert
        Assert.NotNull(stats.BestDistributionFit);
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void AllMetrics_FloatType_ReturnsCorrectValues()
    {
        // Arrange
        var actual = new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        var predicted = new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert
        Assert.Equal(1.0f, stats.R2, 1e-5f);
        Assert.Equal(1.0f, stats.PearsonCorrelation, 1e-5f);
        Assert.Equal(1.0f, stats.SpearmanCorrelation, 1e-5f);
        Assert.Equal(1.0f, stats.KendallTau, 1e-5f);
        Assert.Equal(1.0f, stats.ExplainedVarianceScore, 1e-5f);
        Assert.Equal(0.0f, stats.MeanPredictionError, 1e-5f);
        Assert.Equal(0.0f, stats.DynamicTimeWarping, 1e-5f);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Empty_ReturnsZeroValues()
    {
        // Arrange & Act
        var stats = PredictionStats<double>.Empty();

        // Assert - All main metrics should be zero
        Assert.Equal(0.0, stats.R2, Tolerance);
        Assert.Equal(0.0, stats.AdjustedR2, Tolerance);
        Assert.Equal(0.0, stats.PearsonCorrelation, Tolerance);
        Assert.Equal(0.0, stats.SpearmanCorrelation, Tolerance);
        Assert.Equal(0.0, stats.KendallTau, Tolerance);
        Assert.Equal(0.0, stats.ExplainedVarianceScore, Tolerance);
        Assert.Equal(0.0, stats.MeanPredictionError, Tolerance);
        Assert.Equal(0.0, stats.MedianPredictionError, Tolerance);
        Assert.Equal(0.0, stats.DynamicTimeWarping, Tolerance);
        Assert.Equal(0.0, stats.Accuracy, Tolerance);
        Assert.Equal(0.0, stats.Precision, Tolerance);
        Assert.Equal(0.0, stats.Recall, Tolerance);
        Assert.Equal(0.0, stats.F1Score, Tolerance);
    }

    [Fact]
    public void TwoValues_CalculatesCorrectly()
    {
        // Arrange
        var actual = new[] { 1.0, 5.0 };
        var predicted = new[] { 1.0, 5.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert - Perfect prediction
        Assert.Equal(1.0, stats.R2, Tolerance);
        Assert.Equal(1.0, stats.PearsonCorrelation, Tolerance);
        Assert.Equal(1.0, stats.SpearmanCorrelation, Tolerance);
    }

    [Fact]
    public void ConstantValues_HandledGracefully()
    {
        // Arrange - All same values (variance = 0)
        var actual = new[] { 5.0, 5.0, 5.0, 5.0, 5.0 };
        var predicted = new[] { 5.0, 5.0, 5.0, 5.0, 5.0 };

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert - Metrics should be finite (not NaN or Inf)
        Assert.True(!double.IsNaN(stats.R2) && !double.IsInfinity(stats.R2),
            $"R2 should be finite for constant values, got {stats.R2}");
    }

    [Fact]
    public void LargeDataset_CalculatesCorrectly()
    {
        // Arrange - 100 data points
        var actual = new double[100];
        var predicted = new double[100];
        for (int i = 0; i < 100; i++)
        {
            actual[i] = i + 1.0;
            predicted[i] = i + 1.0;
        }

        // Act
        var stats = CreatePredictionStats(actual, predicted);

        // Assert - Perfect prediction
        Assert.Equal(1.0, stats.R2, Tolerance);
        Assert.Equal(1.0, stats.PearsonCorrelation, Tolerance);
    }

    #endregion
}
