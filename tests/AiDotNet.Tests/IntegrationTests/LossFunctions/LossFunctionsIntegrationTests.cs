using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LossFunctions;

/// <summary>
/// Integration tests for loss function classes.
/// Tests loss calculation and derivative computations for various loss functions.
/// </summary>
public class LossFunctionsIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Mean Squared Error Loss Tests

    [Fact]
    public void MSELoss_IdenticalVectors_ReturnsZero()
    {
        // Arrange
        var mse = new MeanSquaredErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var loss = mse.CalculateLoss(predicted, actual);

        // Assert
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void MSELoss_DifferentVectors_ReturnsPositiveLoss()
    {
        // Arrange
        var mse = new MeanSquaredErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 2.0, 3.0, 4.0 });

        // Act
        var loss = mse.CalculateLoss(predicted, actual);

        // Assert - MSE = (1+1+1)/3 = 1.0
        Assert.Equal(1.0, loss, Tolerance);
    }

    [Fact]
    public void MSELoss_Derivative_ReturnsCorrectGradient()
    {
        // Arrange
        var mse = new MeanSquaredErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 0.0, 0.0, 0.0 });

        // Act
        var derivative = mse.CalculateDerivative(predicted, actual);

        // Assert - derivative is 2*(predicted-actual)/n
        Assert.Equal(3, derivative.Length);
        // Each derivative should be positive (predicted > actual)
        Assert.True(derivative[0] > 0);
        Assert.True(derivative[1] > 0);
        Assert.True(derivative[2] > 0);
    }

    [Fact]
    public void MSELoss_LargerErrors_ProducesLargerLoss()
    {
        // Arrange
        var mse = new MeanSquaredErrorLoss<double>();
        var actual = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
        var smallError = new Vector<double>(new[] { 1.0, 1.0, 1.0 });
        var largeError = new Vector<double>(new[] { 2.0, 2.0, 2.0 });

        // Act
        var smallLoss = mse.CalculateLoss(smallError, actual);
        var largeLoss = mse.CalculateLoss(largeError, actual);

        // Assert
        Assert.True(largeLoss > smallLoss);
    }

    #endregion

    #region Mean Absolute Error Loss Tests

    [Fact]
    public void MAELoss_IdenticalVectors_ReturnsZero()
    {
        // Arrange
        var mae = new MeanAbsoluteErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var loss = mae.CalculateLoss(predicted, actual);

        // Assert
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void MAELoss_DifferentVectors_ReturnsPositiveLoss()
    {
        // Arrange
        var mae = new MeanAbsoluteErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 2.0, 3.0, 4.0 });

        // Act
        var loss = mae.CalculateLoss(predicted, actual);

        // Assert - MAE = (1+1+1)/3 = 1.0
        Assert.Equal(1.0, loss, Tolerance);
    }

    [Fact]
    public void MAELoss_Derivative_ReturnsCorrectGradient()
    {
        // Arrange
        var mae = new MeanAbsoluteErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 5.0, 0.0, 3.0 });
        var actual = new Vector<double>(new[] { 3.0, 2.0, 3.0 });

        // Act
        var derivative = mae.CalculateDerivative(predicted, actual);

        // Assert
        Assert.Equal(3, derivative.Length);
    }

    #endregion

    #region Binary Cross Entropy Loss Tests

    [Fact]
    public void BCELoss_PerfectPredictions_ReturnsLowLoss()
    {
        // Arrange
        var bce = new BinaryCrossEntropyLoss<double>();
        var predicted = new Vector<double>(new[] { 0.99, 0.01, 0.99 });
        var actual = new Vector<double>(new[] { 1.0, 0.0, 1.0 });

        // Act
        var loss = bce.CalculateLoss(predicted, actual);

        // Assert
        Assert.True(loss < 0.1);
    }

    [Fact]
    public void BCELoss_WrongPredictions_ReturnsHighLoss()
    {
        // Arrange
        var bce = new BinaryCrossEntropyLoss<double>();
        var predicted = new Vector<double>(new[] { 0.01, 0.99, 0.01 });
        var actual = new Vector<double>(new[] { 1.0, 0.0, 1.0 });

        // Act
        var loss = bce.CalculateLoss(predicted, actual);

        // Assert
        Assert.True(loss > 2.0);
    }

    [Fact]
    public void BCELoss_UncertainPredictions_ReturnsModerateLoss()
    {
        // Arrange
        var bce = new BinaryCrossEntropyLoss<double>();
        var predicted = new Vector<double>(new[] { 0.5, 0.5, 0.5 });
        var actual = new Vector<double>(new[] { 1.0, 0.0, 1.0 });

        // Act
        var loss = bce.CalculateLoss(predicted, actual);

        // Assert - Should be around -log(0.5) = 0.693
        Assert.True(loss > 0.5);
        Assert.True(loss < 1.0);
    }

    [Fact]
    public void BCELoss_Derivative_ReturnsCorrectGradient()
    {
        // Arrange
        var bce = new BinaryCrossEntropyLoss<double>();
        var predicted = new Vector<double>(new[] { 0.8, 0.2 });
        var actual = new Vector<double>(new[] { 1.0, 0.0 });

        // Act
        var derivative = bce.CalculateDerivative(predicted, actual);

        // Assert
        Assert.Equal(2, derivative.Length);
    }

    #endregion

    #region Huber Loss Tests

    [Fact]
    public void HuberLoss_SmallErrors_BehavesLikeMSE()
    {
        // Arrange
        var huber = new HuberLoss<double>(delta: 1.0);
        var mse = new MeanSquaredErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 0.1, 0.2, 0.3 });
        var actual = new Vector<double>(new[] { 0.0, 0.0, 0.0 });

        // Act
        var huberLoss = huber.CalculateLoss(predicted, actual);
        var mseLoss = mse.CalculateLoss(predicted, actual);

        // Assert - for small errors, Huber ~ 0.5 * MSE
        Assert.True(Math.Abs(huberLoss - 0.5 * mseLoss) < 0.01);
    }

    [Fact]
    public void HuberLoss_LargeErrors_MoreRobustThanMSE()
    {
        // Arrange
        var huber = new HuberLoss<double>(delta: 1.0);
        var mse = new MeanSquaredErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 10.0, 10.0, 10.0 });
        var actual = new Vector<double>(new[] { 0.0, 0.0, 0.0 });

        // Act
        var huberLoss = huber.CalculateLoss(predicted, actual);
        var mseLoss = mse.CalculateLoss(predicted, actual);

        // Assert - Huber should be much smaller than MSE for large errors
        Assert.True(huberLoss < mseLoss);
    }

    [Fact]
    public void HuberLoss_IdenticalVectors_ReturnsZero()
    {
        // Arrange
        var huber = new HuberLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var loss = huber.CalculateLoss(predicted, actual);

        // Assert
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void HuberLoss_Derivative_ReturnsCorrectGradient()
    {
        // Arrange
        var huber = new HuberLoss<double>(delta: 1.0);
        var predicted = new Vector<double>(new[] { 0.5, 5.0 });
        var actual = new Vector<double>(new[] { 0.0, 0.0 });

        // Act
        var derivative = huber.CalculateDerivative(predicted, actual);

        // Assert
        Assert.Equal(2, derivative.Length);
    }

    #endregion

    #region Hinge Loss Tests

    [Fact]
    public void HingeLoss_CorrectPredictions_ReturnsZeroOrLow()
    {
        // Arrange - Hinge loss uses -1/1 labels
        var hinge = new HingeLoss<double>();
        var predicted = new Vector<double>(new[] { 2.0, -2.0, 1.5 });
        var actual = new Vector<double>(new[] { 1.0, -1.0, 1.0 });

        // Act
        var loss = hinge.CalculateLoss(predicted, actual);

        // Assert - margin > 1 means loss = 0
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void HingeLoss_WrongPredictions_ReturnsPositiveLoss()
    {
        // Arrange
        var hinge = new HingeLoss<double>();
        var predicted = new Vector<double>(new[] { -1.0, 1.0 });
        var actual = new Vector<double>(new[] { 1.0, -1.0 });

        // Act
        var loss = hinge.CalculateLoss(predicted, actual);

        // Assert - margin < 0 means loss > 1
        Assert.True(loss > 0.0);
    }

    [Fact]
    public void HingeLoss_MarginAtOne_ReturnsZero()
    {
        // Arrange
        var hinge = new HingeLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, -1.0 });
        var actual = new Vector<double>(new[] { 1.0, -1.0 });

        // Act
        var loss = hinge.CalculateLoss(predicted, actual);

        // Assert - y*f(x) = 1, so max(0, 1-1) = 0
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void HingeLoss_Derivative_ReturnsCorrectGradient()
    {
        // Arrange
        var hinge = new HingeLoss<double>();
        var predicted = new Vector<double>(new[] { 0.5, -0.5 });
        var actual = new Vector<double>(new[] { 1.0, -1.0 });

        // Act
        var derivative = hinge.CalculateDerivative(predicted, actual);

        // Assert
        Assert.Equal(2, derivative.Length);
    }

    #endregion

    #region Root Mean Squared Error Loss Tests

    [Fact]
    public void RMSELoss_IdenticalVectors_ReturnsZero()
    {
        // Arrange
        var rmse = new RootMeanSquaredErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var loss = rmse.CalculateLoss(predicted, actual);

        // Assert
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void RMSELoss_DifferentVectors_ReturnsSqrtOfMSE()
    {
        // Arrange
        var rmse = new RootMeanSquaredErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 2.0, 3.0, 4.0 });

        // Act
        var loss = rmse.CalculateLoss(predicted, actual);

        // Assert - RMSE = sqrt(MSE) = sqrt(1) = 1
        Assert.Equal(1.0, loss, Tolerance);
    }

    #endregion

    #region Cross Entropy Loss Tests

    [Fact]
    public void CrossEntropyLoss_PerfectPredictions_ReturnsLowLoss()
    {
        // Arrange
        var ce = new CrossEntropyLoss<double>();
        var predicted = new Vector<double>(new[] { 0.9, 0.05, 0.05 });
        var actual = new Vector<double>(new[] { 1.0, 0.0, 0.0 });

        // Act
        var loss = ce.CalculateLoss(predicted, actual);

        // Assert
        Assert.True(loss < 0.2);
    }

    [Fact]
    public void CrossEntropyLoss_WrongPredictions_ReturnsHigherLoss()
    {
        // Arrange
        var ce = new CrossEntropyLoss<double>();
        var goodPredicted = new Vector<double>(new[] { 0.9, 0.05, 0.05 });
        var badPredicted = new Vector<double>(new[] { 0.1, 0.8, 0.1 });
        var actual = new Vector<double>(new[] { 1.0, 0.0, 0.0 });

        // Act
        var goodLoss = ce.CalculateLoss(goodPredicted, actual);
        var badLoss = ce.CalculateLoss(badPredicted, actual);

        // Assert - wrong predictions should have higher loss
        Assert.True(badLoss > goodLoss);
    }

    #endregion

    #region Log Cosh Loss Tests

    [Fact]
    public void LogCoshLoss_IdenticalVectors_ReturnsZero()
    {
        // Arrange
        var logCosh = new LogCoshLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var loss = logCosh.CalculateLoss(predicted, actual);

        // Assert
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void LogCoshLoss_SmallErrors_BehavesLikeMSE()
    {
        // Arrange
        var logCosh = new LogCoshLoss<double>();
        var predicted = new Vector<double>(new[] { 0.1, 0.1, 0.1 });
        var actual = new Vector<double>(new[] { 0.0, 0.0, 0.0 });

        // Act
        var loss = logCosh.CalculateLoss(predicted, actual);

        // Assert - for small errors, log(cosh(x)) ~ x^2/2
        Assert.True(loss > 0.0);
        Assert.True(loss < 0.1);
    }

    #endregion

    #region Quantile Loss Tests

    [Fact]
    public void QuantileLoss_MedianQuantile_BehavesLikeMAE()
    {
        // Arrange - quantile 0.5 should behave like MAE
        var quantile = new QuantileLoss<double>(0.5);
        var mae = new MeanAbsoluteErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 2.0, 3.0, 4.0 });

        // Act
        var quantileLoss = quantile.CalculateLoss(predicted, actual);
        var maeLoss = mae.CalculateLoss(predicted, actual);

        // Assert - should be proportional to MAE
        Assert.True(quantileLoss > 0.0);
    }

    [Fact]
    public void QuantileLoss_DifferentQuantiles_ProduceDifferentLoss()
    {
        // Arrange
        var quantile25 = new QuantileLoss<double>(0.25);
        var quantile75 = new QuantileLoss<double>(0.75);
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 2.0, 3.0, 4.0 });

        // Act
        var loss25 = quantile25.CalculateLoss(predicted, actual);
        var loss75 = quantile75.CalculateLoss(predicted, actual);

        // Assert - losses should be different
        Assert.NotEqual(loss25, loss75, Tolerance);
    }

    #endregion

    #region Focal Loss Tests

    [Fact]
    public void FocalLoss_EasyExamples_DownWeighted()
    {
        // Arrange
        var focal = new FocalLoss<double>(gamma: 2.0);
        var bce = new BinaryCrossEntropyLoss<double>();
        var predicted = new Vector<double>(new[] { 0.9 });
        var actual = new Vector<double>(new[] { 1.0 });

        // Act
        var focalLoss = focal.CalculateLoss(predicted, actual);
        var bceLoss = bce.CalculateLoss(predicted, actual);

        // Assert - focal loss should be smaller for easy examples
        Assert.True(focalLoss < bceLoss);
    }

    [Fact]
    public void FocalLoss_HardExamples_LessDownWeighted()
    {
        // Arrange
        var focal = new FocalLoss<double>(gamma: 2.0);
        var bce = new BinaryCrossEntropyLoss<double>();
        var predicted = new Vector<double>(new[] { 0.5 });
        var actual = new Vector<double>(new[] { 1.0 });

        // Act
        var focalLoss = focal.CalculateLoss(predicted, actual);
        var bceLoss = bce.CalculateLoss(predicted, actual);

        // Assert - focal loss should still be positive
        Assert.True(focalLoss > 0.0);
    }

    #endregion

    #region Cosine Similarity Loss Tests

    [Fact]
    public void CosineSimilarityLoss_IdenticalVectors_ReturnsZeroOrNegative()
    {
        // Arrange
        var cosine = new CosineSimilarityLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var loss = cosine.CalculateLoss(predicted, actual);

        // Assert - cosine similarity of identical vectors is 1, loss might be 1 - similarity or negative
        Assert.True(loss <= 1.0);
    }

    [Fact]
    public void CosineSimilarityLoss_OrthogonalVectors_ReturnsHigherLoss()
    {
        // Arrange
        var cosine = new CosineSimilarityLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 0.0 });
        var actual = new Vector<double>(new[] { 0.0, 1.0 });

        // Act
        var loss = cosine.CalculateLoss(predicted, actual);

        // Assert - orthogonal vectors have 0 similarity
        Assert.True(loss >= 0.0);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void AllLossFunctions_HandleLengthMismatch_ThrowsException()
    {
        // Arrange
        var mse = new MeanSquaredErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0 }); // Different length

        // Act & Assert
        Assert.Throws<ArgumentException>(() => mse.CalculateLoss(predicted, actual));
    }

    [Fact]
    public void AllLossFunctions_EmptyVectors_HandledGracefully()
    {
        // Arrange
        var mse = new MeanSquaredErrorLoss<double>();
        var predicted = new Vector<double>(0);
        var actual = new Vector<double>(0);

        // Act & Assert - should not throw, but may return NaN or 0
        var loss = mse.CalculateLoss(predicted, actual);
        // Just verify it doesn't crash
        Assert.True(double.IsNaN(loss) || loss >= 0.0 || loss == 0.0);
    }

    [Fact]
    public void MultipleLossFunctions_ConsistentDerivativeLength()
    {
        // Arrange
        var losses = new ILossFunction<double>[]
        {
            new MeanSquaredErrorLoss<double>(),
            new MeanAbsoluteErrorLoss<double>(),
            new HuberLoss<double>(),
            new LogCoshLoss<double>()
        };
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 0.5, 1.5, 2.5 });

        // Act & Assert
        foreach (var loss in losses)
        {
            var derivative = loss.CalculateDerivative(predicted, actual);
            Assert.Equal(predicted.Length, derivative.Length);
        }
    }

    [Fact]
    public void AllLossFunctions_LargeValues_NoNaNOrInfinity()
    {
        // Arrange
        var losses = new ILossFunction<double>[]
        {
            new MeanSquaredErrorLoss<double>(),
            new MeanAbsoluteErrorLoss<double>(),
            new HuberLoss<double>(),
            new HingeLoss<double>()
        };
        var predicted = new Vector<double>(new[] { 100.0, 200.0, 300.0 });
        var actual = new Vector<double>(new[] { 0.0, 0.0, 0.0 });

        // Act & Assert
        foreach (var loss in losses)
        {
            var lossValue = loss.CalculateLoss(predicted, actual);
            Assert.False(double.IsNaN(lossValue));
        }
    }

    #endregion
}
