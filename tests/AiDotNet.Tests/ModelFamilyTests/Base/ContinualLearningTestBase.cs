using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for continual learning strategies implementing IContinualLearningStrategy.
/// Tests mathematical invariants: regularization non-negativity, finite outputs,
/// gradient preservation, and strategy lifecycle consistency.
/// </summary>
public abstract class ContinualLearningTestBase
{
    /// <summary>Factory method — subclasses return their concrete strategy instance.</summary>
    protected abstract IContinualLearningStrategy<double> CreateStrategy();

    /// <summary>Creates a mock neural network for testing.</summary>
    protected abstract INeuralNetwork<double> CreateMockNetwork();

    /// <summary>Number of model parameters for test data generation.</summary>
    protected virtual int NumParameters => 10;

    // =========================================================================
    // INVARIANT 1: Regularization loss is non-negative
    // =========================================================================

    [Fact]
    public void ComputeLoss_IsNonNegative()
    {
        var strategy = CreateStrategy();
        var network = CreateMockNetwork();

        // Initialize with task 0
        var taskData = CreateTestTaskData();
        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);

        double loss = strategy.ComputeLoss(network);

        Assert.True(loss >= -1e-10,
            $"Regularization loss should be non-negative but got {loss}. " +
            "Regularization penalizes deviation from reference parameters.");
    }

    // =========================================================================
    // INVARIANT 2: Regularization loss is finite
    // =========================================================================

    [Fact]
    public void ComputeLoss_IsFinite()
    {
        var strategy = CreateStrategy();
        var network = CreateMockNetwork();

        var taskData = CreateTestTaskData();
        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);

        double loss = strategy.ComputeLoss(network);

        Assert.False(double.IsNaN(loss), "Regularization loss is NaN.");
        Assert.False(double.IsInfinity(loss), "Regularization loss is Infinity.");
    }

    // =========================================================================
    // INVARIANT 3: Lambda is non-negative
    // =========================================================================

    [Fact]
    public void Lambda_IsNonNegative()
    {
        var strategy = CreateStrategy();
        Assert.True(strategy.Lambda >= 0,
            $"Lambda (regularization strength) should be non-negative but got {strategy.Lambda}.");
    }

    // =========================================================================
    // INVARIANT 4: Modified gradients have same length as input gradients
    // =========================================================================

    [Fact]
    public void ModifyGradients_PreservesLength()
    {
        var strategy = CreateStrategy();
        var network = CreateMockNetwork();

        var taskData = CreateTestTaskData();
        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);

        var gradients = new Vector<double>(NumParameters);
        for (int i = 0; i < NumParameters; i++)
            gradients[i] = 0.1 * (i + 1);

        var modified = strategy.ModifyGradients(network, gradients);

        Assert.Equal(gradients.Length, modified.Length);
    }

    // =========================================================================
    // INVARIANT 5: Modified gradients are finite
    // =========================================================================

    [Fact]
    public void ModifyGradients_AreFinite()
    {
        var strategy = CreateStrategy();
        var network = CreateMockNetwork();

        var taskData = CreateTestTaskData();
        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);

        var gradients = new Vector<double>(NumParameters);
        for (int i = 0; i < NumParameters; i++)
            gradients[i] = 0.1 * (i + 1);

        var modified = strategy.ModifyGradients(network, gradients);

        for (int i = 0; i < modified.Length; i++)
        {
            Assert.False(double.IsNaN(modified[i]),
                $"Modified gradient at index {i} is NaN.");
            Assert.False(double.IsInfinity(modified[i]),
                $"Modified gradient at index {i} is Infinity.");
        }
    }

    // =========================================================================
    // INVARIANT 6: Reset clears state without errors
    // =========================================================================

    [Fact]
    public void Reset_DoesNotThrow()
    {
        var strategy = CreateStrategy();
        var network = CreateMockNetwork();

        var taskData = CreateTestTaskData();
        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);

        // Reset should not throw
        strategy.Reset();
    }

    /// <summary>Creates synthetic task training data.</summary>
    protected virtual (Tensor<double> inputs, Tensor<double> targets) CreateTestTaskData()
    {
        var rng = new Random(42);
        int batchSize = 8;
        int dim = 4;
        var inputData = new double[batchSize * dim];
        var targetData = new double[batchSize * dim];

        for (int i = 0; i < inputData.Length; i++)
        {
            inputData[i] = rng.NextDouble() * 2.0 - 1.0;
            targetData[i] = rng.NextDouble() * 2.0 - 1.0;
        }

        return (
            new Tensor<double>(inputData, new[] { batchSize, dim }),
            new Tensor<double>(targetData, new[] { batchSize, dim })
        );
    }
}
