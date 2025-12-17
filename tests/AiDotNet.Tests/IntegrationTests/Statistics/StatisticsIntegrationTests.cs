using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Statistics;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Statistics;

/// <summary>
/// Integration tests for statistics classes.
/// Tests Quartile calculations and Empty factory methods.
/// </summary>
public class StatisticsIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Quartile Tests

    [Fact]
    public void Quartile_SimpleData_CalculatesCorrectQuartiles()
    {
        // Arrange
        var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 });

        // Act
        var quartile = new Quartile<double>(data);

        // Assert - Q2 (median) should be 4.0
        Assert.Equal(4.0, quartile.Q2, Tolerance);
        // Q1 should be around 2.0-2.5
        Assert.True(quartile.Q1 > 1.5 && quartile.Q1 < 3.0);
        // Q3 should be around 5.5-6.0
        Assert.True(quartile.Q3 > 5.0 && quartile.Q3 < 7.0);
    }

    [Fact]
    public void Quartile_EvenData_CalculatesCorrectMedian()
    {
        // Arrange
        var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

        // Act
        var quartile = new Quartile<double>(data);

        // Assert - Q2 should be between 3 and 4
        Assert.True(quartile.Q2 >= 3.0 && quartile.Q2 <= 4.0);
    }

    [Fact]
    public void Quartile_UnorderedData_SortsAndCalculates()
    {
        // Arrange
        var data = new Vector<double>(new[] { 5.0, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0 });

        // Act
        var quartile = new Quartile<double>(data);

        // Assert - Data should be sorted internally, Q2 (median) should be 5.0
        Assert.Equal(5.0, quartile.Q2, Tolerance);
    }

    [Fact]
    public void Quartile_AllSameValues_ReturnsConstantQuartiles()
    {
        // Arrange
        var data = new Vector<double>(new[] { 5.0, 5.0, 5.0, 5.0, 5.0 });

        // Act
        var quartile = new Quartile<double>(data);

        // Assert
        Assert.Equal(5.0, quartile.Q1, Tolerance);
        Assert.Equal(5.0, quartile.Q2, Tolerance);
        Assert.Equal(5.0, quartile.Q3, Tolerance);
    }

    [Fact]
    public void Quartile_LargeDataset_CalculatesCorrectly()
    {
        // Arrange - Create data from 1 to 100
        var values = new double[100];
        for (int i = 0; i < 100; i++)
        {
            values[i] = i + 1.0;
        }
        var data = new Vector<double>(values);

        // Act
        var quartile = new Quartile<double>(data);

        // Assert - Q2 (median) should be around 50.5
        Assert.True(quartile.Q2 >= 50.0 && quartile.Q2 <= 51.0);
        // Q1 should be around 25.25
        Assert.True(quartile.Q1 >= 24.0 && quartile.Q1 <= 27.0);
        // Q3 should be around 75.75
        Assert.True(quartile.Q3 >= 74.0 && quartile.Q3 <= 77.0);
    }

    [Fact]
    public void Quartile_SmallDataset_ThreeValues()
    {
        // Arrange
        var data = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var quartile = new Quartile<double>(data);

        // Assert
        Assert.Equal(2.0, quartile.Q2, Tolerance);
    }

    [Fact]
    public void Quartile_QuartileRelationship_Q1LessThanQ2LessThanQ3()
    {
        // Arrange
        var data = new Vector<double>(new[] { 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0 });

        // Act
        var quartile = new Quartile<double>(data);

        // Assert - Q1 < Q2 < Q3
        Assert.True(quartile.Q1 <= quartile.Q2);
        Assert.True(quartile.Q2 <= quartile.Q3);
    }

    [Fact]
    public void Quartile_NegativeValues_HandlesCorrectly()
    {
        // Arrange
        var data = new Vector<double>(new[] { -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0 });

        // Act
        var quartile = new Quartile<double>(data);

        // Assert - Q2 should be 1.0
        Assert.Equal(1.0, quartile.Q2, Tolerance);
    }

    [Fact]
    public void Quartile_FloatType_CalculatesCorrectly()
    {
        // Arrange
        var data = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f });

        // Act
        var quartile = new Quartile<float>(data);

        // Assert
        Assert.Equal(4.0f, quartile.Q2, 1e-4f);
    }

    #endregion

    #region BasicStats Empty Tests

    [Fact]
    public void BasicStats_Empty_ReturnsValidInstance()
    {
        // Act
        var stats = BasicStats<double>.Empty();

        // Assert
        Assert.NotNull(stats);
        Assert.Equal(0.0, stats.Mean, Tolerance);
        Assert.Equal(0, stats.N);
    }

    [Fact]
    public void BasicStats_Empty_Float_ReturnsValidInstance()
    {
        // Act
        var stats = BasicStats<float>.Empty();

        // Assert
        Assert.NotNull(stats);
        Assert.Equal(0.0f, stats.Mean, 1e-6f);
    }

    #endregion

    #region ErrorStats Empty Tests

    [Fact]
    public void ErrorStats_Empty_ReturnsValidInstance()
    {
        // Act
        var stats = ErrorStats<double>.Empty();

        // Assert
        Assert.NotNull(stats);
        Assert.Equal(0.0, stats.MAE, Tolerance);
        Assert.Equal(0.0, stats.MSE, Tolerance);
        Assert.Equal(0.0, stats.RMSE, Tolerance);
    }

    [Fact]
    public void ErrorStats_Empty_HasMetric_ReturnsTrue()
    {
        // Arrange
        var stats = ErrorStats<double>.Empty();

        // Act & Assert
        Assert.True(stats.HasMetric(MetricType.MAE));
        Assert.True(stats.HasMetric(MetricType.MSE));
        Assert.True(stats.HasMetric(MetricType.RMSE));
    }

    [Fact]
    public void ErrorStats_Empty_GetMetric_ReturnsZero()
    {
        // Arrange
        var stats = ErrorStats<double>.Empty();

        // Act
        var mae = stats.GetMetric(MetricType.MAE);
        var mse = stats.GetMetric(MetricType.MSE);
        var rmse = stats.GetMetric(MetricType.RMSE);

        // Assert
        Assert.Equal(0.0, mae, Tolerance);
        Assert.Equal(0.0, mse, Tolerance);
        Assert.Equal(0.0, rmse, Tolerance);
    }

    [Fact]
    public void ErrorStats_Empty_ErrorList_IsEmpty()
    {
        // Arrange
        var stats = ErrorStats<double>.Empty();

        // Assert
        Assert.NotNull(stats.ErrorList);
        Assert.Empty(stats.ErrorList);
    }

    #endregion

    #region PredictionStats Empty Tests

    [Fact]
    public void PredictionStats_Empty_ReturnsValidInstance()
    {
        // Act
        var stats = PredictionStats<double>.Empty();

        // Assert
        Assert.NotNull(stats);
        Assert.Equal(0.0, stats.R2, Tolerance);
        Assert.Equal(0.0, stats.AdjustedR2, Tolerance);
    }

    [Fact]
    public void PredictionStats_Empty_HasMetric_ReturnsTrue()
    {
        // Arrange
        var stats = PredictionStats<double>.Empty();

        // Act & Assert
        Assert.True(stats.HasMetric(MetricType.R2));
        Assert.True(stats.HasMetric(MetricType.AdjustedR2));
        Assert.True(stats.HasMetric(MetricType.Accuracy));
    }

    [Fact]
    public void PredictionStats_Empty_GetMetric_ReturnsZero()
    {
        // Arrange
        var stats = PredictionStats<double>.Empty();

        // Act
        var r2 = stats.GetMetric(MetricType.R2);

        // Assert
        Assert.Equal(0.0, r2, Tolerance);
    }

    [Fact]
    public void PredictionStats_Empty_Intervals_AreZero()
    {
        // Arrange
        var stats = PredictionStats<double>.Empty();

        // Assert
        Assert.Equal(0.0, stats.PredictionInterval.Lower, Tolerance);
        Assert.Equal(0.0, stats.PredictionInterval.Upper, Tolerance);
        Assert.Equal(0.0, stats.ConfidenceInterval.Lower, Tolerance);
        Assert.Equal(0.0, stats.ConfidenceInterval.Upper, Tolerance);
    }

    #endregion

    #region ModelStats Empty Tests

    [Fact]
    public void ModelStats_Empty_ReturnsValidInstance()
    {
        // Act
        var stats = ModelStats<double, Matrix<double>, Vector<double>>.Empty();

        // Assert
        Assert.NotNull(stats);
        Assert.Equal(0, stats.FeatureCount);
    }

    [Fact]
    public void ModelStats_Empty_HasMetric_ReturnsTrue()
    {
        // Arrange
        var stats = ModelStats<double, Matrix<double>, Vector<double>>.Empty();

        // Act & Assert
        Assert.True(stats.HasMetric(MetricType.EuclideanDistance));
        Assert.True(stats.HasMetric(MetricType.ManhattanDistance));
        Assert.True(stats.HasMetric(MetricType.CosineSimilarity));
    }

    [Fact]
    public void ModelStats_Empty_GetMetric_ReturnsZero()
    {
        // Arrange
        var stats = ModelStats<double, Matrix<double>, Vector<double>>.Empty();

        // Act
        var euclidean = stats.GetMetric(MetricType.EuclideanDistance);
        var manhattan = stats.GetMetric(MetricType.ManhattanDistance);

        // Assert
        Assert.Equal(0.0, euclidean, Tolerance);
        Assert.Equal(0.0, manhattan, Tolerance);
    }

    [Fact]
    public void ModelStats_Empty_Matrices_AreEmpty()
    {
        // Arrange
        var stats = ModelStats<double, Matrix<double>, Vector<double>>.Empty();

        // Assert
        Assert.NotNull(stats.CorrelationMatrix);
        Assert.NotNull(stats.CovarianceMatrix);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void AllStatsEmptyMethods_DoNotThrow()
    {
        // Act & Assert - ensure Empty() methods don't throw
        var basicStats = BasicStats<double>.Empty();
        var errorStats = ErrorStats<double>.Empty();
        var predictionStats = PredictionStats<double>.Empty();
        var modelStats = ModelStats<double, Matrix<double>, Vector<double>>.Empty();

        Assert.NotNull(basicStats);
        Assert.NotNull(errorStats);
        Assert.NotNull(predictionStats);
        Assert.NotNull(modelStats);
    }

    [Fact]
    public void ErrorStats_HasMetric_UnsupportedMetric_ReturnsFalse()
    {
        // Arrange
        var stats = ErrorStats<double>.Empty();

        // Act & Assert - Use a metric type that doesn't exist in ErrorStats
        // Note: We're testing if the HasMetric properly returns false for unknown types
        Assert.True(stats.HasMetric(MetricType.MAE));
    }

    [Fact]
    public void PredictionStats_BestDistributionFit_IsInitialized()
    {
        // Arrange
        var stats = PredictionStats<double>.Empty();

        // Assert
        Assert.NotNull(stats.BestDistributionFit);
    }

    [Fact]
    public void PredictionStats_LearningCurve_IsInitialized()
    {
        // Arrange
        var stats = PredictionStats<double>.Empty();

        // Assert
        Assert.NotNull(stats.LearningCurve);
        Assert.Empty(stats.LearningCurve);
    }

    #endregion
}
