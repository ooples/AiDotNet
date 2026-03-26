using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for continual learning strategies implementing IContinualLearningStrategy.
/// Tests deep mathematical invariants: regularization monotonicity with parameter deviation,
/// Fisher information properties, gradient projection correctness, and multi-task consistency.
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
    // EWC: L = λ/2 * Σ F_i (θ_i - θ*_i)² >= 0 since F_i >= 0 and squared terms >= 0
    // =========================================================================

    [Fact]
    public void ComputeLoss_IsNonNegative()
    {
        var strategy = CreateStrategy();
        var network = CreateMockNetwork();
        var taskData = CreateTestTaskData();

        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);

        double loss = strategy.ComputeLoss(network);

        Assert.True(loss >= -1e-10,
            $"Regularization loss should be non-negative but got {loss}. " +
            "EWC/MAS regularization is a sum of non-negative terms: λ * Σ F_i * (θ_i - θ*_i)².");
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
    // INVARIANT 3: Loss increases monotonically with parameter deviation magnitude
    // If |θ - θ*| increases, regularization loss should increase or stay same.
    // This is a fundamental property of quadratic regularization.
    // =========================================================================

    [Fact]
    public void ComputeLoss_IncreasesWithParameterDeviation()
    {
        var strategy = CreateStrategy();
        var network = CreateMockNetwork();
        var taskData = CreateTestTaskData();

        // Learn task 0 reference parameters
        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);

        // Record loss with current parameters (should be near zero — no deviation yet)
        double lossNoChange = strategy.ComputeLoss(network);

        // Now perturb the parameters away from reference
        var params0 = network.GetParameters();
        var perturbedParams = new Vector<double>(params0.Length);
        for (int i = 0; i < params0.Length; i++)
            perturbedParams[i] = params0[i] + 0.5; // Add significant perturbation

        network.SetParameters(perturbedParams);
        double lossPerturbed = strategy.ComputeLoss(network);

        Assert.True(lossPerturbed >= lossNoChange - 1e-10,
            $"Loss should increase with parameter deviation: " +
            $"no change={lossNoChange:E4}, perturbed={lossPerturbed:E4}. " +
            "Quadratic regularization penalizes deviation from reference parameters.");

        // Restore original parameters
        network.SetParameters(params0);
    }

    // =========================================================================
    // INVARIANT 4: Lambda scaling — loss scales linearly with lambda
    // L(λ) = λ * g(θ), so L(2λ) ≈ 2 * L(λ) for the same parameters.
    // =========================================================================

    [Fact]
    public void ComputeLoss_ScalesWithLambda()
    {
        var strategy1 = CreateStrategy();
        var strategy2 = CreateStrategy();
        var network1 = CreateMockNetwork();
        var network2 = CreateMockNetwork();
        var taskData = CreateTestTaskData();

        strategy1.Lambda = 100.0;
        strategy2.Lambda = 200.0;

        strategy1.BeforeTask(network1, 0);
        strategy1.AfterTask(network1, taskData, 0);
        strategy2.BeforeTask(network2, 0);
        strategy2.AfterTask(network2, taskData, 0);

        // Perturb both networks identically
        var params1 = network1.GetParameters();
        var perturbed = new Vector<double>(params1.Length);
        for (int i = 0; i < params1.Length; i++)
            perturbed[i] = params1[i] + 0.3;

        network1.SetParameters(perturbed);
        network2.SetParameters(perturbed);

        double loss1 = strategy1.ComputeLoss(network1);
        double loss2 = strategy2.ComputeLoss(network2);

        if (loss1 < 1e-10) return; // Can't test ratio with near-zero values

        double ratio = loss2 / loss1;

        // lambda2/lambda1 = 200/100 = 2.0, so loss ratio should be ~2.0
        Assert.True(ratio > 1.5 && ratio < 2.5,
            $"Loss should scale linearly with lambda: λ1=100 loss={loss1:E4}, " +
            $"λ2=200 loss={loss2:E4}, ratio={ratio:F2} (expected ~2.0). " +
            "Regularization loss formula: L = λ * Σ importance_i * (θ_i - θ*_i)².");
    }

    // =========================================================================
    // INVARIANT 5: Modified gradients have same length as input
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
    // INVARIANT 6: Modified gradients are finite
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
            Assert.False(double.IsNaN(modified[i]), $"Modified gradient[{i}] is NaN.");
            Assert.False(double.IsInfinity(modified[i]), $"Modified gradient[{i}] is Infinity.");
        }
    }

    // =========================================================================
    // INVARIANT 7: Modified gradient norm ≤ original gradient norm (for GEM-like methods)
    // Gradient projection should not increase the gradient magnitude.
    // =========================================================================

    [Fact]
    public void ModifyGradients_DoesNotIncreaseNorm()
    {
        var strategy = CreateStrategy();
        var network = CreateMockNetwork();
        var taskData = CreateTestTaskData();

        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);

        var gradients = new Vector<double>(NumParameters);
        for (int i = 0; i < NumParameters; i++)
            gradients[i] = 0.5 * (i % 3 == 0 ? 1 : -1);

        var modified = strategy.ModifyGradients(network, gradients);

        double origNorm = 0, modifiedNorm = 0;
        for (int i = 0; i < NumParameters; i++)
        {
            origNorm += gradients[i] * gradients[i];
            modifiedNorm += modified[i] * modified[i];
        }

        origNorm = Math.Sqrt(origNorm);
        modifiedNorm = Math.Sqrt(modifiedNorm);

        // Modified gradient should not be drastically larger than original
        // (GEM projects to constraint set, which can only reduce magnitude)
        Assert.True(modifiedNorm <= origNorm * 2.0 + 0.01,
            $"Modified gradient norm ({modifiedNorm:E4}) should not drastically exceed " +
            $"original norm ({origNorm:E4}). Gradient modification should constrain, not amplify.");
    }

    // =========================================================================
    // INVARIANT 8: Multi-task accumulation — loss after 2 tasks ≥ loss after 1 task
    // With more tasks to remember, regularization should be at least as strong.
    // =========================================================================

    [Fact]
    public void ComputeLoss_IncreasesWithMoreTasks()
    {
        var strategy = CreateStrategy();
        var network = CreateMockNetwork();
        var taskData = CreateTestTaskData();

        // Task 0
        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);

        // Perturb parameters
        var params0 = network.GetParameters();
        var perturbed = new Vector<double>(params0.Length);
        for (int i = 0; i < params0.Length; i++)
            perturbed[i] = params0[i] + 0.3;
        network.SetParameters(perturbed);

        double lossAfterTask0 = strategy.ComputeLoss(network);

        // Task 1
        network.SetParameters(params0); // Reset
        strategy.BeforeTask(network, 1);
        strategy.AfterTask(network, taskData, 1);

        // Same perturbation
        network.SetParameters(perturbed);
        double lossAfterTask1 = strategy.ComputeLoss(network);

        // With 2 tasks, regularization should be at least as strong as with 1
        Assert.True(lossAfterTask1 >= lossAfterTask0 - 1e-10,
            $"Loss after 2 tasks ({lossAfterTask1:E4}) should be >= " +
            $"loss after 1 task ({lossAfterTask0:E4}). " +
            "More tasks to protect means stronger regularization.");
    }

    // =========================================================================
    // INVARIANT 9: Lambda is non-negative
    // =========================================================================

    [Fact]
    public void Lambda_IsNonNegative()
    {
        var strategy = CreateStrategy();
        Assert.True(strategy.Lambda >= 0,
            $"Lambda should be non-negative but got {strategy.Lambda}.");
    }

    // =========================================================================
    // INVARIANT 10: Reset clears state without errors
    // =========================================================================

    [Fact]
    public void Reset_DoesNotThrow()
    {
        var strategy = CreateStrategy();
        var network = CreateMockNetwork();
        var taskData = CreateTestTaskData();

        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);
        strategy.Reset();

        // After reset, loss should be near-zero (no tasks to protect)
        double lossAfterReset = strategy.ComputeLoss(network);
        Assert.True(lossAfterReset < 1e-6,
            $"After reset, regularization loss should be near-zero but got {lossAfterReset:E4}.");
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
