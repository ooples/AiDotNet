using AiDotNet.KnowledgeDistillation.Strategies;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.KnowledgeDistillation;

/// <summary>
/// Unit tests for the HybridDistillationStrategy class.
/// </summary>
public class HybridDistillationStrategyTests
{
    [Fact]
    public void Constructor_WithValidStrategies_InitializesCorrectly()
    {
        // Arrange
        var strategy1 = new DistillationLoss<double>(temperature: 3.0, alpha: 0.3);
        var strategy2 = new FeatureDistillationStrategy<double>(featureWeight: 0.5);
        var strategies = new[] { strategy1, strategy2 };
        var weights = new[] { 0.6, 0.4 };

        // Act
        var hybridStrategy = new HybridDistillationStrategy<double>(
            strategies,
            weights,
            temperature: 3.0,
            alpha: 0.3);

        // Assert
        Assert.NotNull(hybridStrategy);
        Assert.Equal(3.0, hybridStrategy.Temperature);
        Assert.Equal(0.3, hybridStrategy.Alpha);
    }

    [Fact]
    public void Constructor_WithNonNormalizedWeights_ThrowsArgumentException()
    {
        // Arrange
        var strategy1 = new DistillationLoss<double>(temperature: 3.0, alpha: 0.3);
        var strategy2 = new FeatureDistillationStrategy<double>(featureWeight: 0.5);
        var strategies = new[] { strategy1, strategy2 };
        var weights = new[] { 0.5, 0.3 }; // Sum = 0.8, not 1.0

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new HybridDistillationStrategy<double>(strategies, weights));
    }

    [Fact]
    public void Constructor_WithMismatchedArrayLengths_ThrowsArgumentException()
    {
        // Arrange
        var strategy1 = new DistillationLoss<double>(temperature: 3.0, alpha: 0.3);
        var strategy2 = new FeatureDistillationStrategy<double>(featureWeight: 0.5);
        var strategies = new[] { strategy1, strategy2 };
        var weights = new[] { 1.0 }; // Only one weight for two strategies

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new HybridDistillationStrategy<double>(strategies, weights));
    }

    [Fact]
    public void Constructor_WithNullStrategies_ThrowsArgumentNullException()
    {
        // Arrange
        var weights = new[] { 1.0 };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new HybridDistillationStrategy<double>(null!, weights));
    }

    [Fact]
    public void ComputeLoss_WithIdenticalOutputs_ReturnsZero()
    {
        // Arrange
        var strategy1 = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);
        var strategies = new[] { strategy1 };
        var weights = new[] { 1.0 };
        var hybridStrategy = new HybridDistillationStrategy<double>(strategies, weights);

        var studentOutput = new Vector<double>(new[] { 2.0, 1.0, 0.5 });
        var teacherOutput = new Vector<double>(new[] { 2.0, 1.0, 0.5 });

        // Act
        var loss = hybridStrategy.ComputeLoss(studentOutput, teacherOutput);

        // Assert
        Assert.True(loss < 0.01); // Should be very close to zero
    }

    [Fact]
    public void ComputeLoss_WithDifferentOutputs_ReturnsPositiveLoss()
    {
        // Arrange
        var strategy1 = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);
        var strategies = new[] { strategy1 };
        var weights = new[] { 1.0 };
        var hybridStrategy = new HybridDistillationStrategy<double>(strategies, weights);

        var studentOutput = new Vector<double>(new[] { 2.0, 1.0, 0.5 });
        var teacherOutput = new Vector<double>(new[] { 1.0, 2.0, 0.5 });

        // Act
        var loss = hybridStrategy.ComputeLoss(studentOutput, teacherOutput);

        // Assert
        Assert.True(loss > 0);
    }

    [Fact]
    public void ComputeLoss_CombinesMultipleStrategies_WeightsLossCorrectly()
    {
        // Arrange
        var strategy1 = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);
        var strategy2 = new DistillationLoss<double>(temperature: 1.0, alpha: 0.0);
        var strategies = new[] { strategy1, strategy2 };
        var weights = new[] { 0.5, 0.5 };
        var hybridStrategy = new HybridDistillationStrategy<double>(strategies, weights);

        var studentOutput = new Vector<double>(new[] { 2.0, 1.0, 0.5 });
        var teacherOutput = new Vector<double>(new[] { 1.5, 1.5, 0.5 });

        // Act
        var hybridLoss = hybridStrategy.ComputeLoss(studentOutput, teacherOutput);
        var loss1 = strategy1.ComputeLoss(studentOutput, teacherOutput);
        var loss2 = strategy2.ComputeLoss(studentOutput, teacherOutput);

        // Assert
        // Hybrid loss should be between the two component losses (weighted average)
        double expectedMin = Math.Min(loss1, loss2);
        double expectedMax = Math.Max(loss1, loss2);
        Assert.True(hybridLoss >= expectedMin * 0.9); // Allow small numerical error
        Assert.True(hybridLoss <= expectedMax * 1.1);
    }

    [Fact]
    public void ComputeGradient_WithIdenticalOutputs_ReturnsZeroGradient()
    {
        // Arrange
        var strategy1 = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);
        var strategies = new[] { strategy1 };
        var weights = new[] { 1.0 };
        var hybridStrategy = new HybridDistillationStrategy<double>(strategies, weights);

        var studentOutput = new Vector<double>(new[] { 2.0, 1.0, 0.5 });
        var teacherOutput = new Vector<double>(new[] { 2.0, 1.0, 0.5 });

        // Act
        var gradient = hybridStrategy.ComputeGradient(studentOutput, teacherOutput);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(studentOutput.Length, gradient.Length);
        for (int i = 0; i < gradient.Length; i++)
        {
            Assert.True(Math.Abs(gradient[i]) < 0.01); // Should be very close to zero
        }
    }

    [Fact]
    public void ComputeGradient_WithDifferentOutputs_ReturnsNonZeroGradient()
    {
        // Arrange
        var strategy1 = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);
        var strategies = new[] { strategy1 };
        var weights = new[] { 1.0 };
        var hybridStrategy = new HybridDistillationStrategy<double>(strategies, weights);

        var studentOutput = new Vector<double>(new[] { 2.0, 1.0, 0.5 });
        var teacherOutput = new Vector<double>(new[] { 1.0, 2.0, 0.5 });

        // Act
        var gradient = hybridStrategy.ComputeGradient(studentOutput, teacherOutput);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(studentOutput.Length, gradient.Length);
        bool hasNonZero = false;
        for (int i = 0; i < gradient.Length; i++)
        {
            if (Math.Abs(gradient[i]) > 0.01)
                hasNonZero = true;
        }
        Assert.True(hasNonZero);
    }

    [Fact]
    public void ComputeLoss_WithTrueLabels_IncorporatesHardLoss()
    {
        // Arrange
        var strategy1 = new DistillationLoss<double>(temperature: 3.0, alpha: 0.5);
        var strategies = new[] { strategy1 };
        var weights = new[] { 1.0 };
        var hybridStrategy = new HybridDistillationStrategy<double>(strategies, weights);

        var studentOutput = new Vector<double>(new[] { 2.0, 1.0, 0.5 });
        var teacherOutput = new Vector<double>(new[] { 1.5, 1.5, 0.5 });
        var trueLabels = new Vector<double>(new[] { 1.0, 0.0, 0.0 });

        // Act
        var lossWithLabels = hybridStrategy.ComputeLoss(studentOutput, teacherOutput, trueLabels);
        var lossWithoutLabels = hybridStrategy.ComputeLoss(studentOutput, teacherOutput);

        // Assert
        // Loss with labels should be different from loss without labels
        Assert.NotEqual(lossWithLabels, lossWithoutLabels);
    }
}
