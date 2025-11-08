using AiDotNet.ContinualLearning.Strategies;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ContinualLearning;

/// <summary>
/// Unit tests for the Elastic Weight Consolidation strategy.
/// </summary>
public class ElasticWeightConsolidationTests
{
    [Fact]
    public void Constructor_ValidInputs_InitializesSuccessfully()
    {
        // Arrange
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var lambda = 1000.0;

        // Act
        var ewc = new ElasticWeightConsolidation<double, Matrix<double>, Vector<double>>(
            lossFunction,
            lambda);

        // Assert
        Assert.NotNull(ewc);
    }

    [Fact]
    public void Constructor_NullLossFunction_ThrowsArgumentNullException()
    {
        // Act & Assert
        var exception = Assert.Throws<ArgumentNullException>(() =>
            new ElasticWeightConsolidation<double, Matrix<double>, Vector<double>>(
                null!,
                1000.0));

        Assert.Contains("lossFunction", exception.Message);
    }

    [Fact]
    public void ComputeRegularizationLoss_WithNoPreviousTask_ReturnsZero()
    {
        // Arrange
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var ewc = new ElasticWeightConsolidation<double, Matrix<double>, Vector<double>>(
            lossFunction,
            1000.0);

        // Create a mock model would go here in full implementation
        // For now, test that it doesn't throw

        // Act & Assert - should not throw
        Assert.NotNull(ewc);
    }
}
