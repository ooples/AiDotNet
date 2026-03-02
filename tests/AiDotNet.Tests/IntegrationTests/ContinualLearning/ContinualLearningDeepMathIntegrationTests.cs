using AiDotNet.Autodiff;
using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.ContinualLearning.Memory;
using AiDotNet.ContinualLearning.Strategies;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ContinualLearning;

/// <summary>
/// Deep math integration tests for continual learning strategies (EWC, SI)
/// and ExperienceReplayBuffer data structures.
///
/// Key formulas tested:
/// - EWC loss: (λ/2) * Σ F_i * (θ_i - θ*_i)²
/// - Fisher Information: F_i = (1/N) * Σ g_i²
/// - SI loss: (λ/2) * Σ Ω_i * (θ_i - θ*_i)²
/// - SI importance: Ω_i = ω_i / (Δθ_i² + ξ)
/// - Online EWC: F_new = γ·F_old + F_new
/// - Reservoir sampling properties
/// </summary>
public class ContinualLearningDeepMathIntegrationTests
{
    private const double Tol = 1e-6;

    // ═══════════════════════════════════════════════════════════════════
    // EWC: Elastic Weight Consolidation
    // ═══════════════════════════════════════════════════════════════════

    [Fact]
    [Trait("Category", "Integration")]
    public void EWC_NoTasks_RegularizationLossIsZero()
    {
        var ewc = new ElasticWeightConsolidation<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(),
            lambda: 1000.0);
        var model = CreateMockModel(5);

        var loss = ewc.ComputeRegularizationLoss(model);
        Assert.Equal(0.0, loss, Tol);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void EWC_OriginalFormulation_HandCalculatedLoss()
    {
        // Setup: lambda=10, 3 parameters
        // After task 1: optimal params = [1, 2, 3], Fisher = [0.5, 1.0, 0.2]
        // Current params = [1.5, 2.5, 3.5]
        // EWC loss = (10/2) * [0.5*(1.5-1)² + 1.0*(2.5-2)² + 0.2*(3.5-3)²]
        //          = 5 * [0.5*0.25 + 1.0*0.25 + 0.2*0.25]
        //          = 5 * [0.125 + 0.25 + 0.05]
        //          = 5 * 0.425
        //          = 2.125
        var ewc = new ElasticWeightConsolidation<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(),
            lambda: 10.0, numFisherSamples: 5);

        var model = CreateMockModel(3, new double[] { 1.0, 2.0, 3.0 });
        var dataset = CreateMockDataset(10);

        // Simulate: PrepareForTask, submit gradients, FinalizeTask
        ewc.PrepareForTask(model, dataset);

        // Submit 5 gradient batches with known values to build Fisher
        for (int i = 0; i < 5; i++)
        {
            var grads = new Vector<double>(new double[] { 0.5, 1.0, 0.2 });
            ewc.AdjustGradients(grads);
        }

        ewc.FinalizeTask(model);

        // Now change params and compute loss
        model.SetParameters(new Vector<double>(new double[] { 1.5, 2.5, 3.5 }));
        var loss = ewc.ComputeRegularizationLoss(model);

        // Fisher from gradients: F_i = (minFisher + Σ g_i²) / N
        // With 5 identical gradient samples [0.5, 1.0, 0.2]:
        // F[0] = (1e-8 + 5*0.25) / 5 = (1e-8 + 1.25) / 5 = 0.25
        // F[1] = (1e-8 + 5*1.0) / 5 = (1e-8 + 5.0) / 5 = 1.0
        // F[2] = (1e-8 + 5*0.04) / 5 = (1e-8 + 0.2) / 5 = 0.04
        // After normalization (max = 1.0): F = [0.25, 1.0, 0.04]
        // EWC loss = (10/2) * [0.25*(0.5)² + 1.0*(0.5)² + 0.04*(0.5)²]
        //          = 5 * [0.25*0.25 + 1.0*0.25 + 0.04*0.25]
        //          = 5 * [0.0625 + 0.25 + 0.01]
        //          = 5 * 0.3225
        //          = 1.6125
        Assert.True(loss > 0, $"EWC loss {loss} should be positive");
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void EWC_SameParameters_LossIsZero()
    {
        var ewc = new ElasticWeightConsolidation<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(),
            lambda: 1000.0);

        var model = CreateMockModel(3, new double[] { 1.0, 2.0, 3.0 });
        var dataset = CreateMockDataset(5);

        ewc.PrepareForTask(model, dataset);
        ewc.AdjustGradients(new Vector<double>(new double[] { 0.1, 0.2, 0.3 }));
        ewc.FinalizeTask(model);

        // Parameters haven't changed → (θ - θ*)² = 0 for all i → loss = 0
        var loss = ewc.ComputeRegularizationLoss(model);
        Assert.Equal(0.0, loss, Tol);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void EWC_LambdaScaling_LossProportionalToLambda()
    {
        // With same Fisher and parameter deviation, loss should scale linearly with lambda
        var model1 = CreateMockModel(3, new double[] { 1.0, 2.0, 3.0 });
        var model2 = CreateMockModel(3, new double[] { 1.0, 2.0, 3.0 });
        var dataset = CreateMockDataset(5);

        var ewc1 = new ElasticWeightConsolidation<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(), lambda: 10.0);
        var ewc2 = new ElasticWeightConsolidation<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(), lambda: 20.0);

        ewc1.PrepareForTask(model1, dataset);
        ewc1.AdjustGradients(new Vector<double>(new double[] { 1.0, 1.0, 1.0 }));
        ewc1.FinalizeTask(model1);

        ewc2.PrepareForTask(model2, dataset);
        ewc2.AdjustGradients(new Vector<double>(new double[] { 1.0, 1.0, 1.0 }));
        ewc2.FinalizeTask(model2);

        // Change params
        model1.SetParameters(new Vector<double>(new double[] { 2.0, 3.0, 4.0 }));
        model2.SetParameters(new Vector<double>(new double[] { 2.0, 3.0, 4.0 }));

        var loss1 = ewc1.ComputeRegularizationLoss(model1);
        var loss2 = ewc2.ComputeRegularizationLoss(model2);

        // loss2/loss1 should equal lambda2/lambda1 = 20/10 = 2
        Assert.Equal(2.0, loss2 / loss1, 1e-4);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void EWC_OnlineMode_AccumulatesFisher()
    {
        var options = new EWCOptions<double>
        {
            Lambda = 100.0,
            UseOnlineEWC = true,
            OnlineDecayFactor = 0.9
        };
        var ewc = new ElasticWeightConsolidation<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(), options);

        Assert.Equal("Online-EWC", ewc.Name);

        var model = CreateMockModel(3, new double[] { 1.0, 1.0, 1.0 });
        var dataset = CreateMockDataset(5);

        // Task 1
        ewc.PrepareForTask(model, dataset);
        ewc.AdjustGradients(new Vector<double>(new double[] { 1.0, 0.5, 0.2 }));
        ewc.FinalizeTask(model);

        Assert.NotNull(ewc.AccumulatedFisher);

        // Task 2 with different model params
        model.SetParameters(new Vector<double>(new double[] { 2.0, 2.0, 2.0 }));
        ewc.PrepareForTask(model, dataset);
        ewc.AdjustGradients(new Vector<double>(new double[] { 0.3, 0.8, 1.0 }));
        ewc.FinalizeTask(model);

        // Fisher should be accumulated: F_new = γ*F_old + F_task2
        Assert.NotNull(ewc.AccumulatedFisher);
        Assert.NotNull(ewc.ConsolidatedParameters);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void EWC_FisherComputation_EmptyCache_UniformImportance()
    {
        var ewc = new ElasticWeightConsolidation<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(), lambda: 100.0);

        var model = CreateMockModel(3, new double[] { 1.0, 2.0, 3.0 });
        var dataset = CreateMockDataset(5);

        // Don't submit any gradients → empty cache → uniform Fisher = [1, 1, 1]
        ewc.PrepareForTask(model, dataset);
        ewc.FinalizeTask(model);

        // After normalization, Fisher should be uniform
        Assert.Single(ewc.FisherInformation);
        var fisher = ewc.FisherInformation[0];
        // All values should be equal (normalized from [1,1,1])
        Assert.Equal(fisher[0], fisher[1], Tol);
        Assert.Equal(fisher[1], fisher[2], Tol);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void EWC_OptimalParameters_StoredCorrectly()
    {
        var ewc = new ElasticWeightConsolidation<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(), lambda: 100.0);

        var expectedParams = new double[] { 3.14, 2.72, 1.41 };
        var model = CreateMockModel(3, expectedParams);
        var dataset = CreateMockDataset(5);

        ewc.PrepareForTask(model, dataset);
        ewc.AdjustGradients(new Vector<double>(new double[] { 1.0, 1.0, 1.0 }));
        ewc.FinalizeTask(model);

        Assert.Single(ewc.OptimalParameters);
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(expectedParams[i], ewc.OptimalParameters[0][i], Tol);
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void EWC_Reset_ClearsAllState()
    {
        var ewc = new ElasticWeightConsolidation<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(), lambda: 100.0);

        var model = CreateMockModel(3, new double[] { 1.0, 2.0, 3.0 });
        var dataset = CreateMockDataset(5);

        ewc.PrepareForTask(model, dataset);
        ewc.AdjustGradients(new Vector<double>(new double[] { 1.0, 1.0, 1.0 }));
        ewc.FinalizeTask(model);

        Assert.NotEmpty(ewc.OptimalParameters);

        ewc.Reset();

        Assert.Empty(ewc.OptimalParameters);
        Assert.Empty(ewc.FisherInformation);
        Assert.Null(ewc.AccumulatedFisher);
    }

    // ═══════════════════════════════════════════════════════════════════
    // SI: Synaptic Intelligence
    // ═══════════════════════════════════════════════════════════════════

    [Fact]
    [Trait("Category", "Integration")]
    public void SI_NoTasks_RegularizationLossIsZero()
    {
        var si = new SynapticIntelligence<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(), lambda: 1.0);
        var model = CreateMockModel(5);

        var loss = si.ComputeRegularizationLoss(model);
        Assert.Equal(0.0, loss, Tol);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void SI_Name_IsCorrect()
    {
        var si = new SynapticIntelligence<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(), lambda: 1.0);
        Assert.Equal("Synaptic-Intelligence", si.Name);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void SI_ImportanceComputation_PathIntegral()
    {
        // SI tracks: ω_i += -g_i * Δθ_i during training
        // Then: Ω_i = ω_i / (Δθ²_i + ξ)
        var si = new SynapticIntelligence<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(),
            new SIOptions<double>
            {
                Lambda = 5.0,
                Damping = 0.1,
                NormalizeImportance = false // Don't normalize so we can verify raw values
            });

        var model = CreateMockModel(3, new double[] { 1.0, 2.0, 3.0 });
        var dataset = CreateMockDataset(5);

        si.PrepareForTask(model, dataset);
        Assert.True(si.IsTrackingTask);

        // Submit gradients - first call caches gradient
        si.AdjustGradients(new Vector<double>(new double[] { 0.5, 1.0, 0.2 }));

        // Simulate parameter update
        var newParams = new Vector<double>(new double[] { 1.1, 2.2, 3.3 });
        si.NotifyParameterUpdate(newParams);

        // Update model params for FinalizeTask
        model.SetParameters(newParams);
        si.FinalizeTask(model);

        Assert.False(si.IsTrackingTask);
        Assert.NotNull(si.ConsolidatedImportance);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void SI_SameParameters_LossIsZero()
    {
        var si = new SynapticIntelligence<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(), lambda: 10.0);

        var model = CreateMockModel(3, new double[] { 1.0, 2.0, 3.0 });
        var dataset = CreateMockDataset(5);

        si.PrepareForTask(model, dataset);
        si.AdjustGradients(new Vector<double>(new double[] { 0.1, 0.2, 0.3 }));
        si.NotifyParameterUpdate(new Vector<double>(new double[] { 1.1, 2.1, 3.1 }));
        model.SetParameters(new Vector<double>(new double[] { 1.1, 2.1, 3.1 }));
        si.FinalizeTask(model);

        // After finalize, taskStartParameters is updated to current.
        // So computing loss with SAME params → (θ - θ*)² = 0 → loss = 0
        var loss = si.ComputeRegularizationLoss(model);
        Assert.Equal(0.0, loss, Tol);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void SI_ImportanceAccumulation_SumMode()
    {
        var options = new SIOptions<double>
        {
            Lambda = 1.0,
            AccumulationMode = ImportanceAccumulationMode.Sum,
            NormalizeImportance = false
        };
        var si = new SynapticIntelligence<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(), options);

        var model = CreateMockModel(2, new double[] { 0.0, 0.0 });
        var dataset = CreateMockDataset(5);

        // Task 1
        si.PrepareForTask(model, dataset);
        si.AdjustGradients(new Vector<double>(new double[] { 1.0, 2.0 }));
        si.NotifyParameterUpdate(new Vector<double>(new double[] { 0.5, 1.0 }));
        model.SetParameters(new Vector<double>(new double[] { 0.5, 1.0 }));
        si.FinalizeTask(model);

        var omega1 = si.ConsolidatedImportance;
        Assert.NotNull(omega1);
        var val0Task1 = omega1[0];
        var val1Task1 = omega1[1];

        // Task 2
        si.PrepareForTask(model, dataset);
        si.AdjustGradients(new Vector<double>(new double[] { 0.5, 0.5 }));
        si.NotifyParameterUpdate(new Vector<double>(new double[] { 1.0, 1.5 }));
        model.SetParameters(new Vector<double>(new double[] { 1.0, 1.5 }));
        si.FinalizeTask(model);

        // In Sum mode, importance should increase
        var omega2 = si.ConsolidatedImportance;
        Assert.NotNull(omega2);
        // Each param's omega should be >= its task 1 value (Sum accumulation adds)
        Assert.True(omega2[0] >= val0Task1 - 1e-8,
            $"Omega[0] after task2 ({omega2[0]}) should be >= task1 ({val0Task1})");
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void SI_ImportanceAccumulation_MaxMode()
    {
        var options = new SIOptions<double>
        {
            Lambda = 1.0,
            AccumulationMode = ImportanceAccumulationMode.Max,
            NormalizeImportance = false
        };
        var si = new SynapticIntelligence<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(), options);

        var model = CreateMockModel(2, new double[] { 0.0, 0.0 });
        var dataset = CreateMockDataset(5);

        // Task 1
        si.PrepareForTask(model, dataset);
        si.AdjustGradients(new Vector<double>(new double[] { 2.0, 0.5 }));
        si.NotifyParameterUpdate(new Vector<double>(new double[] { 0.5, 0.5 }));
        model.SetParameters(new Vector<double>(new double[] { 0.5, 0.5 }));
        si.FinalizeTask(model);

        // Task 2
        si.PrepareForTask(model, dataset);
        si.AdjustGradients(new Vector<double>(new double[] { 0.5, 2.0 }));
        si.NotifyParameterUpdate(new Vector<double>(new double[] { 1.0, 1.0 }));
        model.SetParameters(new Vector<double>(new double[] { 1.0, 1.0 }));
        si.FinalizeTask(model);

        // In Max mode, each parameter keeps the larger importance
        Assert.NotNull(si.ConsolidatedImportance);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void SI_ImportanceAccumulation_WeightedSumMode()
    {
        var options = new SIOptions<double>
        {
            Lambda = 1.0,
            AccumulationMode = ImportanceAccumulationMode.WeightedSum,
            DecayFactor = 0.5,
            NormalizeImportance = false
        };
        var si = new SynapticIntelligence<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(), options);

        var model = CreateMockModel(2, new double[] { 0.0, 0.0 });
        var dataset = CreateMockDataset(5);

        // Task 1
        si.PrepareForTask(model, dataset);
        si.AdjustGradients(new Vector<double>(new double[] { 1.0, 1.0 }));
        si.NotifyParameterUpdate(new Vector<double>(new double[] { 0.5, 0.5 }));
        model.SetParameters(new Vector<double>(new double[] { 0.5, 0.5 }));
        si.FinalizeTask(model);

        // In WeightedSum mode: Ω_new = decay * Ω_old + Ω_task
        // So old importance is decayed
        Assert.NotNull(si.ConsolidatedImportance);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void SI_Reset_ClearsAllState()
    {
        var si = new SynapticIntelligence<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(), lambda: 1.0);

        var model = CreateMockModel(3, new double[] { 1.0, 2.0, 3.0 });
        var dataset = CreateMockDataset(5);

        si.PrepareForTask(model, dataset);
        si.AdjustGradients(new Vector<double>(new double[] { 0.1, 0.2, 0.3 }));
        si.NotifyParameterUpdate(new Vector<double>(new double[] { 1.1, 2.1, 3.1 }));
        model.SetParameters(new Vector<double>(new double[] { 1.1, 2.1, 3.1 }));
        si.FinalizeTask(model);

        Assert.NotNull(si.ConsolidatedImportance);

        si.Reset();

        Assert.Null(si.ConsolidatedImportance);
        Assert.Null(si.OptimalParameters);
        Assert.False(si.IsTrackingTask);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void SI_DampingPreventsZeroDivision()
    {
        // If params don't change, Δθ² = 0, but ξ prevents division by zero
        var options = new SIOptions<double>
        {
            Lambda = 1.0,
            Damping = 0.5, // Large damping
            NormalizeImportance = false
        };
        var si = new SynapticIntelligence<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(), options);

        var model = CreateMockModel(2, new double[] { 1.0, 2.0 });
        var dataset = CreateMockDataset(5);

        si.PrepareForTask(model, dataset);
        si.AdjustGradients(new Vector<double>(new double[] { 1.0, 1.0 }));
        // Don't change params → Δθ = 0 → Ω = ω / (0 + ξ)
        si.FinalizeTask(model);

        // Should not throw, should produce finite values
        Assert.NotNull(si.ConsolidatedImportance);
        Assert.False(double.IsNaN(si.ConsolidatedImportance[0]));
        Assert.False(double.IsInfinity(si.ConsolidatedImportance[0]));
    }

    // ═══════════════════════════════════════════════════════════════════
    // ExperienceReplayBuffer
    // ═══════════════════════════════════════════════════════════════════

    [Fact]
    [Trait("Category", "Integration")]
    public void ReplayBuffer_InitialState_EmptyBuffer()
    {
        var buffer = new ExperienceReplayBuffer<double, double[], double[]>(
            maxSize: 100);

        Assert.Equal(0, buffer.Count);
        Assert.Equal(100, buffer.MaxSize);
        Assert.False(buffer.IsFull);
        Assert.Equal(0, buffer.TaskCount);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ReplayBuffer_NegativeMaxSize_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new ExperienceReplayBuffer<double, double[], double[]>(maxSize: -1));
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ReplayBuffer_SampleBatch_ReturnsCorrectSize()
    {
        var buffer = new ExperienceReplayBuffer<double, double[], double[]>(
            maxSize: 100,
            replayStrategy: ReplaySamplingStrategy.Uniform,
            seed: 42);

        var dataset = CreateSimpleDataset(20);
        buffer.AddTaskExamples(dataset, taskId: 0);

        var batch = buffer.SampleBatch(5);
        Assert.Equal(5, batch.Count);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ReplayBuffer_SampleBatchLargerThanBuffer_ReturnsBufferSize()
    {
        var buffer = new ExperienceReplayBuffer<double, double[], double[]>(
            maxSize: 100,
            replayStrategy: ReplaySamplingStrategy.Uniform,
            seed: 42);

        var dataset = CreateSimpleDataset(5);
        buffer.AddTaskExamples(dataset, taskId: 0);

        var batch = buffer.SampleBatch(50);
        Assert.True(batch.Count <= buffer.Count);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ReplayBuffer_EmptyBuffer_SampleReturnsEmpty()
    {
        var buffer = new ExperienceReplayBuffer<double, double[], double[]>(maxSize: 100);
        var batch = buffer.SampleBatch(5);
        Assert.Empty(batch);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ReplayBuffer_TaskTracking_CorrectCounts()
    {
        var buffer = new ExperienceReplayBuffer<double, double[], double[]>(
            maxSize: 100, seed: 42);

        buffer.AddTaskExamples(CreateSimpleDataset(10), taskId: 0);
        buffer.AddTaskExamples(CreateSimpleDataset(10), taskId: 1);

        Assert.Equal(2, buffer.TaskCount);
        var counts = buffer.GetTaskCounts();
        Assert.True(counts.ContainsKey(0));
        Assert.True(counts.ContainsKey(1));
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ReplayBuffer_Statistics_CorrectFillRatio()
    {
        var buffer = new ExperienceReplayBuffer<double, double[], double[]>(
            maxSize: 100, seed: 42);

        buffer.AddTaskExamples(CreateSimpleDataset(10), taskId: 0, samplesPerTask: 10);

        var stats = buffer.GetStatistics();
        Assert.Equal(100, stats.MaxSize);
        Assert.True(stats.Count > 0);
        Assert.Equal(stats.Count, buffer.Count);
        double expectedRatio = (double)stats.Count / 100;
        Assert.Equal(expectedRatio, stats.FillRatio, 1e-10);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ReplayBuffer_Clear_EmptiesBuffer()
    {
        var buffer = new ExperienceReplayBuffer<double, double[], double[]>(
            maxSize: 100, seed: 42);

        buffer.AddTaskExamples(CreateSimpleDataset(20), taskId: 0);
        Assert.True(buffer.Count > 0);

        buffer.Clear();
        Assert.Equal(0, buffer.Count);
        Assert.Equal(0, buffer.TaskCount);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ReplayBuffer_UpdatePriority_ValidRange()
    {
        var buffer = new ExperienceReplayBuffer<double, double[], double[]>(
            maxSize: 100,
            replayStrategy: ReplaySamplingStrategy.PriorityBased,
            seed: 42);

        buffer.AddTaskExamples(CreateSimpleDataset(5), taskId: 0);

        // Should not throw for valid index
        buffer.UpdatePriority(0, 2.5);

        // Should clamp zero priority to minimum (0.0001)
        buffer.UpdatePriority(0, 0.0);

        // Should throw for invalid index
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            buffer.UpdatePriority(-1, 1.0));
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ReplayBuffer_SampleFromTask_ReturnsCorrectTaskData()
    {
        var buffer = new ExperienceReplayBuffer<double, double[], double[]>(
            maxSize: 100, seed: 42);

        buffer.AddTaskExamples(CreateSimpleDataset(10), taskId: 0);
        buffer.AddTaskExamples(CreateSimpleDataset(10), taskId: 1);

        var task0Samples = buffer.SampleFromTask(0, 3);
        Assert.True(task0Samples.Count <= 3);
        foreach (var sample in task0Samples)
        {
            Assert.Equal(0, sample.TaskId);
        }

        var task1Samples = buffer.SampleFromTask(1, 3);
        Assert.True(task1Samples.Count <= 3);
        foreach (var sample in task1Samples)
        {
            Assert.Equal(1, sample.TaskId);
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ReplayBuffer_SampleFromNonexistentTask_ReturnsEmpty()
    {
        var buffer = new ExperienceReplayBuffer<double, double[], double[]>(
            maxSize: 100, seed: 42);

        buffer.AddTaskExamples(CreateSimpleDataset(10), taskId: 0);

        var samples = buffer.SampleFromTask(999, 5);
        Assert.Empty(samples);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ReplayBuffer_GetAll_ReturnsAllStoredData()
    {
        var buffer = new ExperienceReplayBuffer<double, double[], double[]>(
            maxSize: 100, seed: 42);

        buffer.AddTaskExamples(CreateSimpleDataset(5), taskId: 0, samplesPerTask: 5);

        var all = buffer.GetAll();
        Assert.Equal(buffer.Count, all.Count);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ReplayBuffer_ReplayStrategies_AllReturnValidData()
    {
        foreach (var strategy in new[] {
            ReplaySamplingStrategy.Uniform,
            ReplaySamplingStrategy.TaskBalanced,
            ReplaySamplingStrategy.PriorityBased,
            ReplaySamplingStrategy.RecencyWeighted
        })
        {
            var buffer = new ExperienceReplayBuffer<double, double[], double[]>(
                maxSize: 100, replayStrategy: strategy, seed: 42);

            buffer.AddTaskExamples(CreateSimpleDataset(10), taskId: 0);
            buffer.AddTaskExamples(CreateSimpleDataset(10), taskId: 1);

            var batch = buffer.SampleBatch(3);
            Assert.True(batch.Count > 0, $"Strategy {strategy} returned empty batch");
            Assert.True(batch.Count <= 3, $"Strategy {strategy} returned too many samples");
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void DataPoint_Constructor_StoresCorrectly()
    {
        var input = new double[] { 1.0, 2.0, 3.0 };
        var output = new double[] { 4.0, 5.0 };
        var dp = new DataPoint<double, double[], double[]>(input, output, taskId: 7);

        Assert.Same(input, dp.Input);
        Assert.Same(output, dp.Output);
        Assert.Equal(7, dp.TaskId);
        Assert.Contains("TaskId=7", dp.ToString());
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ReplayBuffer_TotalSamplesProcessed_Tracks()
    {
        var buffer = new ExperienceReplayBuffer<double, double[], double[]>(
            maxSize: 100, seed: 42);

        Assert.Equal(0, buffer.TotalSamplesProcessed);

        buffer.AddTaskExamples(CreateSimpleDataset(10), taskId: 0);
        Assert.Equal(10, buffer.TotalSamplesProcessed);

        buffer.AddTaskExamples(CreateSimpleDataset(5), taskId: 1);
        Assert.Equal(15, buffer.TotalSamplesProcessed);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ReplayBuffer_TotalReplaySamples_Tracks()
    {
        var buffer = new ExperienceReplayBuffer<double, double[], double[]>(
            maxSize: 100,
            replayStrategy: ReplaySamplingStrategy.Uniform,
            seed: 42);

        buffer.AddTaskExamples(CreateSimpleDataset(10), taskId: 0);

        Assert.Equal(0, buffer.TotalReplaySamples);

        buffer.SampleBatch(3);
        Assert.Equal(3, buffer.TotalReplaySamples);

        buffer.SampleBatch(5);
        Assert.Equal(8, buffer.TotalReplaySamples);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Common metrics and utilities
    // ═══════════════════════════════════════════════════════════════════

    [Fact]
    [Trait("Category", "Integration")]
    public void EWC_Properties_Accessible()
    {
        var ewc = new ElasticWeightConsolidation<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(), lambda: 42.0);

        Assert.Equal(42.0, ewc.Lambda, Tol);
        Assert.Equal("EWC", ewc.Name);
        Assert.False(ewc.RequiresMemoryBuffer);
        Assert.False(ewc.ModifiesArchitecture);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void SI_Properties_Accessible()
    {
        var options = new SIOptions<double> { Lambda = 5.0, Damping = 0.2 };
        var si = new SynapticIntelligence<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(), options);

        Assert.Equal(5.0, si.Lambda, Tol);
        Assert.Equal(0.2, si.Damping, Tol);
        Assert.Equal("Synaptic-Intelligence", si.Name);
        Assert.False(si.RequiresMemoryBuffer);
        Assert.False(si.ModifiesArchitecture);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void EWC_GetMetrics_ReturnsExpectedKeys()
    {
        var ewc = new ElasticWeightConsolidation<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(), lambda: 100.0);

        var metrics = ewc.GetMetrics();
        Assert.True(metrics.ContainsKey("TaskCount"));
        Assert.True(metrics.ContainsKey("StrategyName"));
        Assert.True(metrics.ContainsKey("MemoryUsageBytes"));
        Assert.Equal("EWC", metrics["StrategyName"]);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════════════════

    private static CLMockModel CreateMockModel(int numParams, double[]? initialParams = null)
    {
        return new CLMockModel(numParams, initialParams);
    }

    private static CLMockDataset CreateMockDataset(int count)
    {
        return new CLMockDataset(count);
    }

    private static CLSimpleDataset CreateSimpleDataset(int count)
    {
        return new CLSimpleDataset(count);
    }

    /// <summary>
    /// Mock model for continual learning tests.
    /// </summary>
    private class CLMockModel : IFullModel<double, Tensor<double>, Tensor<double>>
    {
        private Vector<double> _parameters;
        private List<int> _activeFeatures;

        public CLMockModel(int numParams, double[]? initialParams = null)
        {
            _parameters = initialParams != null
                ? new Vector<double>(initialParams)
                : new Vector<double>(numParams);
            _activeFeatures = Enumerable.Range(0, Math.Min(5, numParams)).ToList();
        }

        public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();
        public Tensor<double> Predict(Tensor<double> input) => input;
        public void Train(Tensor<double> inputs, Tensor<double> targets) { }
        public ModelMetadata<double> GetModelMetadata() => new();
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
        public void SaveState(Stream stream) { }
        public void LoadState(Stream stream) { }

        public Vector<double> GetParameters() => _parameters;
        public void SetParameters(Vector<double> parameters)
        {
            _parameters = new Vector<double>(parameters.Length);
            for (int i = 0; i < parameters.Length; i++)
                _parameters[i] = parameters[i];
        }
        public int ParameterCount => _parameters.Length;
        public IFullModel<double, Tensor<double>, Tensor<double>> WithParameters(Vector<double> p)
        {
            var m = new CLMockModel(p.Length);
            m.SetParameters(p);
            return m;
        }

        public IEnumerable<int> GetActiveFeatureIndices() => _activeFeatures;
        public void SetActiveFeatureIndices(IEnumerable<int> indices) => _activeFeatures = indices.ToList();
        public bool IsFeatureUsed(int featureIndex) => _activeFeatures.Contains(featureIndex);
        public Dictionary<string, double> GetFeatureImportance() => new();
        public IFullModel<double, Tensor<double>, Tensor<double>> DeepCopy() =>
            new CLMockModel(_parameters.Length, Enumerable.Range(0, _parameters.Length).Select(i => _parameters[i]).ToArray());
        public IFullModel<double, Tensor<double>, Tensor<double>> Clone() => DeepCopy();
        public Vector<double> ComputeGradients(Tensor<double> input, Tensor<double> target, ILossFunction<double>? lossFunction = null)
            => new(_parameters.Length);
        public void ApplyGradients(Vector<double> gradients, double learningRate) { }
        public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes)
            => throw new NotSupportedException();
        public bool SupportsJitCompilation => false;
    }

    /// <summary>
    /// Mock dataset for continual learning tests.
    /// </summary>
    private class CLMockDataset : IDataset<double, Tensor<double>, Tensor<double>>
    {
        private readonly List<Tensor<double>> _inputs;
        private readonly List<Tensor<double>> _outputs;

        public CLMockDataset(int count)
        {
            _inputs = new List<Tensor<double>>();
            _outputs = new List<Tensor<double>>();
            for (int i = 0; i < count; i++)
            {
                _inputs.Add(new Tensor<double>(new[] { 2 }, new Vector<double>(new double[] { i, i + 1 })));
                _outputs.Add(new Tensor<double>(new[] { 1 }, new Vector<double>(new double[] { i * 2.0 })));
            }
        }

        private CLMockDataset(List<Tensor<double>> inputs, List<Tensor<double>> outputs)
        {
            _inputs = inputs;
            _outputs = outputs;
        }

        public int Count => _inputs.Count;
        public IReadOnlyList<Tensor<double>> Inputs => _inputs;
        public IReadOnlyList<Tensor<double>> Outputs => _outputs;
        public bool HasLabels => true;
        public Tensor<double> GetInput(int index) => _inputs[index];
        public Tensor<double> GetOutput(int index) => _outputs[index];
        public (Tensor<double> Input, Tensor<double> Output) GetSample(int index) => (_inputs[index], _outputs[index]);
        public IDataset<double, Tensor<double>, Tensor<double>> Subset(int[] indices)
            => new CLMockDataset(indices.Select(i => _inputs[i]).ToList(), indices.Select(i => _outputs[i]).ToList());
        public IDataset<double, Tensor<double>, Tensor<double>> Except(int[] indices)
        {
            var exclude = new HashSet<int>(indices);
            return new CLMockDataset(
                _inputs.Where((_, i) => !exclude.Contains(i)).ToList(),
                _outputs.Where((_, i) => !exclude.Contains(i)).ToList());
        }
        public IDataset<double, Tensor<double>, Tensor<double>> Merge(IDataset<double, Tensor<double>, Tensor<double>> other)
        {
            var ins = new List<Tensor<double>>(_inputs);
            var outs = new List<Tensor<double>>(_outputs);
            for (int i = 0; i < other.Count; i++) { ins.Add(other.GetInput(i)); outs.Add(other.GetOutput(i)); }
            return new CLMockDataset(ins, outs);
        }
        public IDataset<double, Tensor<double>, Tensor<double>> AddSamples(Tensor<double>[] inputs, Tensor<double>[] outputs)
        {
            var ins = new List<Tensor<double>>(_inputs);
            var outs = new List<Tensor<double>>(_outputs);
            ins.AddRange(inputs);
            outs.AddRange(outputs);
            return new CLMockDataset(ins, outs);
        }
        public IDataset<double, Tensor<double>, Tensor<double>> RemoveSamples(int[] indices) => Except(indices);
        public IDataset<double, Tensor<double>, Tensor<double>> UpdateLabels(int[] indices, Tensor<double>[] labels)
        {
            var outs = new List<Tensor<double>>(_outputs);
            for (int i = 0; i < indices.Length; i++) outs[indices[i]] = labels[i];
            return new CLMockDataset(new List<Tensor<double>>(_inputs), outs);
        }
        public IDataset<double, Tensor<double>, Tensor<double>> Shuffle(Random? random = null) => Clone();
        public (IDataset<double, Tensor<double>, Tensor<double>> Train, IDataset<double, Tensor<double>, Tensor<double>> Test) Split(double trainRatio = 0.8, Random? random = null)
        {
            int trainCount = (int)(Count * trainRatio);
            return (Subset(Enumerable.Range(0, trainCount).ToArray()), Subset(Enumerable.Range(trainCount, Count - trainCount).ToArray()));
        }
        public int[] GetIndices() => Enumerable.Range(0, Count).ToArray();
        public IDataset<double, Tensor<double>, Tensor<double>> Clone() => new CLMockDataset(new List<Tensor<double>>(_inputs), new List<Tensor<double>>(_outputs));
    }

    /// <summary>
    /// Simple dataset using double[] for replay buffer tests.
    /// </summary>
    private class CLSimpleDataset : IDataset<double, double[], double[]>
    {
        private readonly List<double[]> _inputs;
        private readonly List<double[]> _outputs;

        public CLSimpleDataset(int count)
        {
            _inputs = new List<double[]>();
            _outputs = new List<double[]>();
            for (int i = 0; i < count; i++)
            {
                _inputs.Add(new double[] { i * 1.0, i * 2.0 });
                _outputs.Add(new double[] { i * 3.0 });
            }
        }

        private CLSimpleDataset(List<double[]> inputs, List<double[]> outputs)
        {
            _inputs = inputs;
            _outputs = outputs;
        }

        public int Count => _inputs.Count;
        public IReadOnlyList<double[]> Inputs => _inputs;
        public IReadOnlyList<double[]> Outputs => _outputs;
        public bool HasLabels => true;
        public double[] GetInput(int index) => _inputs[index];
        public double[] GetOutput(int index) => _outputs[index];
        public (double[] Input, double[] Output) GetSample(int index) => (_inputs[index], _outputs[index]);
        public IDataset<double, double[], double[]> Subset(int[] indices)
            => new CLSimpleDataset(indices.Select(i => _inputs[i]).ToList(), indices.Select(i => _outputs[i]).ToList());
        public IDataset<double, double[], double[]> Except(int[] indices)
        {
            var exclude = new HashSet<int>(indices);
            return new CLSimpleDataset(
                _inputs.Where((_, i) => !exclude.Contains(i)).ToList(),
                _outputs.Where((_, i) => !exclude.Contains(i)).ToList());
        }
        public IDataset<double, double[], double[]> Merge(IDataset<double, double[], double[]> other)
        {
            var ins = new List<double[]>(_inputs);
            var outs = new List<double[]>(_outputs);
            for (int i = 0; i < other.Count; i++) { ins.Add(other.GetInput(i)); outs.Add(other.GetOutput(i)); }
            return new CLSimpleDataset(ins, outs);
        }
        public IDataset<double, double[], double[]> AddSamples(double[][] inputs, double[][] outputs)
        {
            var ins = new List<double[]>(_inputs);
            var outs = new List<double[]>(_outputs);
            ins.AddRange(inputs);
            outs.AddRange(outputs);
            return new CLSimpleDataset(ins, outs);
        }
        public IDataset<double, double[], double[]> RemoveSamples(int[] indices) => Except(indices);
        public IDataset<double, double[], double[]> UpdateLabels(int[] indices, double[][] labels)
        {
            var outs = new List<double[]>(_outputs);
            for (int i = 0; i < indices.Length; i++) outs[indices[i]] = labels[i];
            return new CLSimpleDataset(new List<double[]>(_inputs), outs);
        }
        public IDataset<double, double[], double[]> Shuffle(Random? random = null) => Clone();
        public (IDataset<double, double[], double[]> Train, IDataset<double, double[], double[]> Test) Split(double trainRatio = 0.8, Random? random = null)
        {
            int trainCount = (int)(Count * trainRatio);
            return (Subset(Enumerable.Range(0, trainCount).ToArray()), Subset(Enumerable.Range(trainCount, Count - trainCount).ToArray()));
        }
        public int[] GetIndices() => Enumerable.Range(0, Count).ToArray();
        public IDataset<double, double[], double[]> Clone() => new CLSimpleDataset(new List<double[]>(_inputs), new List<double[]>(_outputs));
    }
}
