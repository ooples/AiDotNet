using AiDotNet.Interfaces;
using AiDotNet.KnowledgeDistillation;
using AiDotNet.KnowledgeDistillation.Strategies;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.KnowledgeDistillation;

/// <summary>
/// Unit tests for the HybridDistillationStrategy class.
/// </summary>
public class HybridDistillationStrategyTests
{
    private Matrix<double> MatrixFromVector(Vector<double> vector)
    {
        var matrix = new Matrix<double>(1, vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            matrix[0, i] = vector[i];
        }
        return matrix;
    }

    [Fact]
    public void Constructor_WithValidStrategies_InitializesCorrectly()
    {
        // Arrange
        var strategy1 = new DistillationLoss<double>(temperature: 3.0, alpha: 0.3);
        var strategy2 = new DistillationLoss<double>(temperature: 2.0, alpha: 0.5);
        var strategies = new (IDistillationStrategy<double>, double)[]
        {
            (strategy1, 0.6),
            (strategy2, 0.4)
        };

        // Act
        var hybridStrategy = new HybridDistillationStrategy<double>(
            strategies,
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
        var strategy2 = new DistillationLoss<double>(temperature: 2.0, alpha: 0.5);
        var strategies = new (IDistillationStrategy<double>, double)[]
        {
            (strategy1, 0.5),
            (strategy2, 0.3)
        }; // Sum = 0.8, not 1.0

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new HybridDistillationStrategy<double>(strategies));
    }

    [Fact]
    public void Constructor_WithEmptyStrategies_ThrowsArgumentException()
    {
        // Arrange
        var strategies = Array.Empty<(IDistillationStrategy<double>, double)>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new HybridDistillationStrategy<double>(strategies));
    }

    [Fact]
    public void Constructor_WithNullStrategies_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new HybridDistillationStrategy<double>(null!));
    }

    [Fact]
    public void ComputeLoss_WithIdenticalOutputs_ReturnsZero()
    {
        // Arrange
        var strategy1 = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);
        var strategies = new (IDistillationStrategy<double>, double)[] { (strategy1, 1.0) };
        var hybridStrategy = new HybridDistillationStrategy<double>(strategies);

        var studentOutput = new Vector<double>(new[] { 2.0, 1.0, 0.5 });
        var teacherOutput = new Vector<double>(new[] { 2.0, 1.0, 0.5 });
        var studentBatch = MatrixFromVector(studentOutput);
        var teacherBatch = MatrixFromVector(teacherOutput);

        // Act
        var loss = hybridStrategy.ComputeLoss(studentBatch, teacherBatch);

        // Assert
        Assert.True(loss < 0.01); // Should be very close to zero
    }

    [Fact]
    public void ComputeLoss_WithDifferentOutputs_ReturnsPositiveLoss()
    {
        // Arrange
        var strategy1 = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);
        var strategies = new (IDistillationStrategy<double>, double)[] { (strategy1, 1.0) };
        var hybridStrategy = new HybridDistillationStrategy<double>(strategies);

        var studentOutput = new Vector<double>(new[] { 2.0, 1.0, 0.5 });
        var teacherOutput = new Vector<double>(new[] { 1.0, 2.0, 0.5 });
        var studentBatch = MatrixFromVector(studentOutput);
        var teacherBatch = MatrixFromVector(teacherOutput);

        // Act
        var loss = hybridStrategy.ComputeLoss(studentBatch, teacherBatch);

        // Assert
        Assert.True(loss > 0);
    }

    [Fact]
    public void ComputeLoss_CombinesMultipleStrategies_WeightsLossCorrectly()
    {
        // Arrange
        var strategy1 = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);
        var strategy2 = new DistillationLoss<double>(temperature: 1.0, alpha: 0.0);
        var strategies = new (IDistillationStrategy<double>, double)[]
        {
            (strategy1, 0.5),
            (strategy2, 0.5)
        };
        var hybridStrategy = new HybridDistillationStrategy<double>(strategies);

        var studentOutput = new Vector<double>(new[] { 2.0, 1.0, 0.5 });
        var teacherOutput = new Vector<double>(new[] { 1.5, 1.5, 0.5 });
        var studentBatch = MatrixFromVector(studentOutput);
        var teacherBatch = MatrixFromVector(teacherOutput);

        // Act
        var hybridLoss = hybridStrategy.ComputeLoss(studentBatch, teacherBatch);
        var loss1 = strategy1.ComputeLoss(studentBatch, teacherBatch);
        var loss2 = strategy2.ComputeLoss(studentBatch, teacherBatch);

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
        var strategies = new (IDistillationStrategy<double>, double)[] { (strategy1, 1.0) };
        var hybridStrategy = new HybridDistillationStrategy<double>(strategies);

        var studentOutput = new Vector<double>(new[] { 2.0, 1.0, 0.5 });
        var teacherOutput = new Vector<double>(new[] { 2.0, 1.0, 0.5 });
        var studentBatch = MatrixFromVector(studentOutput);
        var teacherBatch = MatrixFromVector(teacherOutput);

        // Act
        var gradient = hybridStrategy.ComputeGradient(studentBatch, teacherBatch);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(1, gradient.Rows);
        Assert.Equal(studentOutput.Length, gradient.Columns);
        for (int i = 0; i < gradient.Columns; i++)
        {
            Assert.True(Math.Abs(gradient[0, i]) < 0.01); // Should be very close to zero
        }
    }

    [Fact]
    public void ComputeGradient_WithDifferentOutputs_ReturnsNonZeroGradient()
    {
        // Arrange
        var strategy1 = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);
        var strategies = new (IDistillationStrategy<double>, double)[] { (strategy1, 1.0) };
        var hybridStrategy = new HybridDistillationStrategy<double>(strategies);

        var studentOutput = new Vector<double>(new[] { 2.0, 1.0, 0.5 });
        var teacherOutput = new Vector<double>(new[] { 1.0, 2.0, 0.5 });
        var studentBatch = MatrixFromVector(studentOutput);
        var teacherBatch = MatrixFromVector(teacherOutput);

        // Act
        var gradient = hybridStrategy.ComputeGradient(studentBatch, teacherBatch);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(1, gradient.Rows);
        Assert.Equal(studentOutput.Length, gradient.Columns);
        bool hasNonZero = false;
        for (int i = 0; i < gradient.Columns; i++)
        {
            if (Math.Abs(gradient[0, i]) > 0.01)
                hasNonZero = true;
        }
        Assert.True(hasNonZero);
    }

    [Fact]
    public void ComputeLoss_WithTrueLabels_IncorporatesHardLoss()
    {
        // Arrange
        var strategy1 = new DistillationLoss<double>(temperature: 3.0, alpha: 0.5);
        var strategies = new (IDistillationStrategy<double>, double)[] { (strategy1, 1.0) };
        var hybridStrategy = new HybridDistillationStrategy<double>(strategies);

        var studentOutput = new Vector<double>(new[] { 2.0, 1.0, 0.5 });
        var teacherOutput = new Vector<double>(new[] { 1.5, 1.5, 0.5 });
        var trueLabels = new Vector<double>(new[] { 1.0, 0.0, 0.0 });
        var studentBatch = MatrixFromVector(studentOutput);
        var teacherBatch = MatrixFromVector(teacherOutput);
        var labelsBatch = MatrixFromVector(trueLabels);

        // Act
        var lossWithLabels = hybridStrategy.ComputeLoss(studentBatch, teacherBatch, labelsBatch);
        var lossWithoutLabels = hybridStrategy.ComputeLoss(studentBatch, teacherBatch);

        // Assert
        // Loss with labels should be different from loss without labels
        Assert.NotEqual(lossWithLabels, lossWithoutLabels);
    }
}
