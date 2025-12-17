using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Statistics;

public class ClassificationMetricsIntegrationTests
{
    private const double Tolerance = 1e-9;

    [Fact]
    public void CalculateAccuracy_Binary_WithProbabilities_UsesThresholding()
    {
        // Arrange
        var actual = new Vector<double>(new[] { 0.0, 1.0, 1.0, 0.0 });
        var predicted = new Vector<double>(new[] { 0.1, 0.9, 0.7, 0.2 });

        // Act
        var accuracy = StatisticsHelper<double>.CalculateAccuracy(actual, predicted, PredictionType.Binary);

        // Assert
        Assert.Equal(1.0, accuracy, Tolerance);
    }

    [Fact]
    public void CalculatePrecisionRecallF1_Binary_WithProbabilities_UsesThresholding()
    {
        // Arrange
        var actual = new Vector<double>(new[] { 0.0, 1.0, 1.0, 0.0 });
        var predicted = new Vector<double>(new[] { 0.6, 0.7, 0.4, 0.1 }); // labels -> [1,1,0,0]

        // Act
        var (precision, recall, f1) = StatisticsHelper<double>.CalculatePrecisionRecallF1(actual, predicted, PredictionType.Binary);

        // Assert
        Assert.Equal(0.5, precision, Tolerance);
        Assert.Equal(0.5, recall, Tolerance);
        Assert.Equal(0.5, f1, Tolerance);
    }

    [Fact]
    public void CalculateMacroPrecisionRecallF1_MultiClass_ReturnsMacroAverages()
    {
        // Arrange
        var actual = new Vector<double>(new[] { 0.0, 1.0, 2.0, 2.0 });
        var predicted = new Vector<double>(new[] { 0.0, 2.0, 2.0, 1.0 });

        // Act
        var accuracy = StatisticsHelper<double>.CalculateAccuracy(actual, predicted, PredictionType.MultiClass);
        var (precision, recall, f1) = StatisticsHelper<double>.CalculatePrecisionRecallF1(actual, predicted, PredictionType.MultiClass);

        // Assert
        Assert.Equal(0.5, accuracy, Tolerance);
        Assert.Equal(0.5, precision, Tolerance);
        Assert.Equal(0.5, recall, Tolerance);
        Assert.Equal(0.5, f1, Tolerance);
    }
}
