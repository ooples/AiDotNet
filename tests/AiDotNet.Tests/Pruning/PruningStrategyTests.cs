using AiDotNet.LinearAlgebra;
using AiDotNet.Pruning;
using Xunit;

namespace AiDotNet.Tests.Pruning;

/// <summary>
/// Tests for pruning strategies and masks.
/// </summary>
public class PruningStrategyTests
{
    [Fact]
    public void MagnitudePruning_50Percent_PrunesSmallestWeights()
    {
        // Arrange
        var weights = new Matrix<double>(2, 2);
        weights[0, 0] = 0.1;  // Small - should be pruned
        weights[0, 1] = 0.9;  // Large - should keep
        weights[1, 0] = 0.2;  // Small - should be pruned
        weights[1, 1] = 0.8;  // Large - should keep

        var strategy = new MagnitudePruningStrategy<double>();

        // Act
        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, targetSparsity: 0.5);

        // Assert
        Assert.Equal(0.5, mask.GetSparsity(), precision: 2);

        var pruned = mask.Apply(weights);
        Assert.Equal(0.0, pruned[0, 0]); // Pruned
        Assert.Equal(0.9, pruned[0, 1]); // Kept
        Assert.Equal(0.0, pruned[1, 0]); // Pruned
        Assert.Equal(0.8, pruned[1, 1]); // Kept
    }

    [Fact]
    public void MagnitudePruning_70Percent_PrunesMostWeights()
    {
        // Arrange
        var weights = new Matrix<double>(2, 5);
        weights[0, 0] = 0.1;
        weights[0, 1] = 0.2;
        weights[0, 2] = 0.3;
        weights[0, 3] = 0.4;
        weights[0, 4] = 0.5;
        weights[1, 0] = 0.6;
        weights[1, 1] = 0.7;
        weights[1, 2] = 0.8;
        weights[1, 3] = 0.9;
        weights[1, 4] = 1.0;

        var strategy = new MagnitudePruningStrategy<double>();

        // Act
        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, targetSparsity: 0.7);

        // Assert - 70% sparsity means 7 out of 10 weights should be zero
        Assert.Equal(0.7, mask.GetSparsity(), precision: 1);
    }

    [Fact]
    public void MagnitudePruning_RequiresGradients_ReturnsFalse()
    {
        // Arrange
        var strategy = new MagnitudePruningStrategy<double>();

        // Assert
        Assert.False(strategy.RequiresGradients);
    }

    [Fact]
    public void GradientPruning_RequiresGradients()
    {
        // Arrange
        var strategy = new GradientPruningStrategy<double>();

        // Assert
        Assert.True(strategy.RequiresGradients);
    }

    [Fact]
    public void GradientPruning_PrunesLowSensitivityWeights()
    {
        // Arrange
        var weights = new Matrix<double>(2, 2);
        weights[0, 0] = 0.5;
        weights[0, 1] = 0.5;
        weights[1, 0] = 0.5;
        weights[1, 1] = 0.5;

        var gradients = new Matrix<double>(2, 2);
        gradients[0, 0] = 0.01; // Low gradient - prune
        gradients[0, 1] = 1.0;  // High gradient - keep
        gradients[1, 0] = 0.02; // Low gradient - prune
        gradients[1, 1] = 0.9;  // High gradient - keep

        var strategy = new GradientPruningStrategy<double>();

        // Act
        var scores = strategy.ComputeImportanceScores(weights, gradients);
        var mask = strategy.CreateMask(scores, targetSparsity: 0.5);

        // Assert
        var pruned = mask.Apply(weights);
        Assert.Equal(0.0, pruned[0, 0]); // Low gradient → pruned
        Assert.Equal(0.5, pruned[0, 1]); // High gradient → kept
    }

    [Fact]
    public void GradientPruning_WithoutGradients_ThrowsException()
    {
        // Arrange
        var weights = new Matrix<double>(2, 2);
        var strategy = new GradientPruningStrategy<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            strategy.ComputeImportanceScores(weights, null));
    }

    [Fact]
    public void LotteryTicket_StoresAndRestoresInitialWeights()
    {
        // Arrange
        var initialWeights = new Matrix<double>(2, 2);
        initialWeights[0, 0] = 0.1;
        initialWeights[0, 1] = 0.2;
        initialWeights[1, 0] = 0.3;
        initialWeights[1, 1] = 0.4;

        var strategy = new LotteryTicketPruningStrategy<double>();
        strategy.StoreInitialWeights("layer1", initialWeights);

        // Simulate training - weights change
        var trainedWeights = new Matrix<double>(2, 2);
        trainedWeights[0, 0] = 0.5;
        trainedWeights[0, 1] = 0.6;
        trainedWeights[1, 0] = 0.7;
        trainedWeights[1, 1] = 0.8;

        // Act
        var scores = strategy.ComputeImportanceScores(trainedWeights);
        var mask = strategy.CreateMask(scores, targetSparsity: 0.5);

        // Reset to initial (key lottery ticket step)
        var resetWeights = trainedWeights.Clone();
        strategy.ResetToInitialWeights("layer1", resetWeights, mask);

        // Assert
        // Should have initial values where mask is 1, zero where mask is 0
        var maskedInitial = mask.Apply(initialWeights);
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                Assert.Equal(maskedInitial[i, j], resetWeights[i, j]);
            }
        }
    }

    [Fact]
    public void LotteryTicket_WithoutStoredWeights_ThrowsException()
    {
        // Arrange
        var strategy = new LotteryTicketPruningStrategy<double>();
        var weights = new Matrix<double>(2, 2);
        var mask = new PruningMask<double>(2, 2);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            strategy.ResetToInitialWeights("nonexistent", weights, mask));
    }

    [Fact]
    public void StructuredPruning_PrunesEntireNeurons()
    {
        // Arrange
        var weights = new Matrix<double>(3, 4); // 3 inputs, 4 neurons

        // Neuron 0: weak connections
        weights[0, 0] = 0.1;
        weights[1, 0] = 0.1;
        weights[2, 0] = 0.1;

        // Neuron 1: strong connections
        weights[0, 1] = 0.9;
        weights[1, 1] = 0.9;
        weights[2, 1] = 0.9;

        // Neuron 2: weak
        weights[0, 2] = 0.2;
        weights[1, 2] = 0.2;
        weights[2, 2] = 0.2;

        // Neuron 3: strong
        weights[0, 3] = 0.8;
        weights[1, 3] = 0.8;
        weights[2, 3] = 0.8;

        var strategy = new StructuredPruningStrategy<double>(
            StructuredPruningStrategy<double>.StructurePruningType.Neuron);

        // Act - prune 50% of neurons (2 out of 4)
        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, targetSparsity: 0.5);

        // Assert
        var pruned = mask.Apply(weights);

        // Neurons 0 and 2 (weakest) should be entirely pruned
        for (int row = 0; row < 3; row++)
        {
            Assert.Equal(0.0, pruned[row, 0]); // Neuron 0 pruned
            Assert.NotEqual(0.0, pruned[row, 1]); // Neuron 1 kept
            Assert.Equal(0.0, pruned[row, 2]); // Neuron 2 pruned
            Assert.NotEqual(0.0, pruned[row, 3]); // Neuron 3 kept
        }
    }

    [Fact]
    public void StructuredPruning_IsStructured_ReturnsTrue()
    {
        // Arrange
        var strategy = new StructuredPruningStrategy<double>(
            StructuredPruningStrategy<double>.StructurePruningType.Neuron);

        // Assert
        Assert.True(strategy.IsStructured);
    }

    [Fact]
    public void PruningMask_GetSparsity_CalculatesCorrectly()
    {
        // Arrange
        var mask = new PruningMask<double>(2, 2);
        var keepIndices = new bool[2, 2]
        {
            { true, false },
            { false, false }
        };
        mask.UpdateMask(keepIndices);

        // Act
        double sparsity = mask.GetSparsity();

        // Assert - 3 out of 4 are zero, so 75% sparse
        Assert.Equal(0.75, sparsity);
    }

    [Fact]
    public void PruningMask_CombineWith_LogicalAND()
    {
        // Arrange
        var mask1 = new PruningMask<double>(2, 2);
        var keep1 = new bool[2, 2] { { true, false }, { true, true } };
        mask1.UpdateMask(keep1);

        var mask2 = new PruningMask<double>(2, 2);
        var keep2 = new bool[2, 2] { { true, true }, { false, true } };
        mask2.UpdateMask(keep2);

        // Act
        var combined = mask1.CombineWith(mask2);

        // Assert - should be logical AND
        var weights = new Matrix<double>(2, 2);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                weights[i, j] = 1.0;

        var result = combined.Apply(weights);

        Assert.Equal(1.0, result[0, 0]); // true AND true
        Assert.Equal(0.0, result[0, 1]); // false AND true
        Assert.Equal(0.0, result[1, 0]); // true AND false
        Assert.Equal(1.0, result[1, 1]); // true AND true
    }

    [Fact]
    public void PruningMask_Apply_EnforcesShapeMatching()
    {
        // Arrange
        var mask = new PruningMask<double>(2, 2);
        var weights = new Matrix<double>(3, 3);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => mask.Apply(weights));
    }

    [Fact]
    public void MagnitudePruning_ApplyPruning_ModifiesWeightsInPlace()
    {
        // Arrange
        var weights = new Matrix<double>(2, 2);
        weights[0, 0] = 0.1;
        weights[0, 1] = 0.9;
        weights[1, 0] = 0.2;
        weights[1, 1] = 0.8;

        var strategy = new MagnitudePruningStrategy<double>();
        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, targetSparsity: 0.5);

        // Act
        strategy.ApplyPruning(weights, mask);

        // Assert - weights should be modified in place
        Assert.Equal(0.0, weights[0, 0]); // Pruned
        Assert.Equal(0.9, weights[0, 1]); // Kept
        Assert.Equal(0.0, weights[1, 0]); // Pruned
        Assert.Equal(0.8, weights[1, 1]); // Kept
    }

    [Fact]
    public void PruningMask_InitializedWithAllOnes()
    {
        // Arrange & Act
        var mask = new PruningMask<double>(3, 3);

        // Assert - initially no pruning (sparsity = 0)
        Assert.Equal(0.0, mask.GetSparsity());
    }

    [Fact]
    public void CreateMask_InvalidSparsity_ThrowsException()
    {
        // Arrange
        var strategy = new MagnitudePruningStrategy<double>();
        var scores = new Matrix<double>(2, 2);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => strategy.CreateMask(scores, -0.1));
        Assert.Throws<ArgumentException>(() => strategy.CreateMask(scores, 1.5));
    }

    [Fact]
    public void LotteryTicket_IterativePruning_AchievesTargetSparsity()
    {
        // Arrange
        var weights = new Matrix<double>(5, 5);
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                weights[i, j] = ((double)(i + 1)) * (j + 1) * 0.1; // Varying magnitudes

        var strategy = new LotteryTicketPruningStrategy<double>(iterativeRounds: 5);

        // Act
        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, targetSparsity: 0.8);

        // Assert - should achieve approximately 80% sparsity
        Assert.True(mask.GetSparsity() >= 0.75 && mask.GetSparsity() <= 0.85);
    }
}
