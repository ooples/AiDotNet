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
/// - EWC penalty gradient: λ * F_i * (θ_i - θ*_i)
/// - Fisher Information: F_i = (1/N) * Σ_n g_(n,i)², one gradient per example
/// - SI loss: (λ/2) * Σ Ω_i * (θ_i - θ*_i)²
/// - SI importance: Ω_i = ω_i / (Δθ_i² + ξ)
/// - Online EWC: F*_i = γ·F*_(i-1) + F_i (Schwarz et al. 2018)
/// - Reservoir sampling properties
/// </summary>
public class ContinualLearningDeepMathIntegrationTests
{
    private const double Tol = 1e-6;

    // ═══════════════════════════════════════════════════════════════════
    // EWC: Elastic Weight Consolidation (Kirkpatrick et al. 2017)
    // ═══════════════════════════════════════════════════════════════════

    private static ElasticWeightConsolidation<double, Tensor<double>, Tensor<double>> Ewc(double lambda, int samples = 5, EWCOptions<double>? options = null)
    {
        options ??= new EWCOptions<double>();
        options.Lambda = lambda;
        options.NumFisherSamples = samples;
        return new ElasticWeightConsolidation<double, Tensor<double>, Tensor<double>>(new MeanSquaredErrorLoss<double>(), options);
    }

    // One EWC task: the mock returns `gradient` for every example, so the empirical Fisher is gradient².
    private static void LearnTask(ElasticWeightConsolidation<double, Tensor<double>, Tensor<double>> ewc, CLMockModel model, double[] gradient)
    {
        model.GradientFunction = (_, _, _) => new Vector<double>(gradient);
        ewc.PrepareForTask(model, CreateMockDataset(10));
        ewc.FinalizeTask(model);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task EWC_NoTasks_RegularizationLossIsZero()
    {
        await Task.Yield();
        var ewc = Ewc(1000.0);
        var model = CreateMockModel(5);
        Assert.Equal(0.0, ewc.ComputeRegularizationLoss(model), Tol);
        var g = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        Assert.Equal(g.ToArray(), ewc.AdjustGradients(model, g).ToArray());
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task EWC_OriginalFormulation_HandCalculatedLoss()
    {
        await Task.Yield();
        // Per-example gradients [0.5, 1.0, 0.2] give F = [0.25, 1.0, 0.04] (no normalization: the paper has none).
        // Loss = (10/2) · Σ F_i (0.5)² = 5 · 0.25 · 1.29 = 1.6125.
        var ewc = Ewc(10.0);
        var model = CreateMockModel(3, new double[] { 1.0, 2.0, 3.0 });
        LearnTask(ewc, model, new[] { 0.5, 1.0, 0.2 });
        Assert.Equal(new[] { 0.25, 1.0, 0.04 }, ewc.FisherInformation[0].ToArray().Select(v => Math.Round(v, 12)).ToArray());
        model.SetParameters(new Vector<double>(new double[] { 1.5, 2.5, 3.5 }));
        Assert.Equal(1.6125, ewc.ComputeRegularizationLoss(model), Tol);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task EWC_AdjustGradients_AddsThePenaltyGradient()
    {
        await Task.Yield();
        // ∂/∂θ (λ/2) Σ F (θ − θ*)² = λ F (θ − θ*) = 10 · [0.25, 1, 0.04] · 0.5 = [1.25, 5, 0.2], added to the task gradient.
        // Before the fix AdjustGradients returned the task gradient unchanged, so the trainer never applied EWC.
        var ewc = Ewc(10.0);
        var model = CreateMockModel(3, new double[] { 1.0, 2.0, 3.0 });
        LearnTask(ewc, model, new[] { 0.5, 1.0, 0.2 });
        model.SetParameters(new Vector<double>(new double[] { 1.5, 2.5, 3.5 }));
        var adjusted = ewc.AdjustGradients(model, new Vector<double>(new double[] { 1.0, 1.0, 1.0 }));
        Assert.Equal(2.25, adjusted[0], Tol);
        Assert.Equal(6.0, adjusted[1], Tol);
        Assert.Equal(1.2, adjusted[2], Tol);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task EWC_Fisher_IsTheMeanOfPerExampleSquaredGradients()
    {
        await Task.Yield();
        // Example n has gradient n·[1, 1]: F = mean(n²) over n = 0..3 = 3.5, not the squared mean gradient (2.25).
        var ewc = Ewc(1.0, samples: 4);
        var model = CreateMockModel(2);
        model.GradientFunction = (input, _, _) => new Vector<double>(new[] { input[0], input[0] });
        ewc.PrepareForTask(model, CreateMockDataset(4));
        ewc.FinalizeTask(model);
        Assert.Equal(3.5, ewc.FisherInformation[0][0], Tol);
        Assert.Equal(3.5, ewc.FisherInformation[0][1], Tol);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task EWC_SameParameters_LossIsZero()
    {
        await Task.Yield();
        var ewc = Ewc(100.0);
        var model = CreateMockModel(3, new double[] { 1.0, 2.0, 3.0 });
        LearnTask(ewc, model, new[] { 0.1, 0.2, 0.3 });
        Assert.Equal(0.0, ewc.ComputeRegularizationLoss(model), Tol);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task EWC_LambdaScaling_LossProportionalToLambda()
    {
        await Task.Yield();
        var ewc1 = Ewc(10.0);
        var ewc2 = Ewc(20.0);
        var model1 = CreateMockModel(3, new double[] { 0, 0, 0 });
        var model2 = CreateMockModel(3, new double[] { 0, 0, 0 });
        LearnTask(ewc1, model1, new[] { 1.0, 1.0, 1.0 });
        LearnTask(ewc2, model2, new[] { 1.0, 1.0, 1.0 });
        model1.SetParameters(new Vector<double>(new double[] { 1, 1, 1 }));
        model2.SetParameters(new Vector<double>(new double[] { 1, 1, 1 }));
        Assert.Equal(2.0, ewc2.ComputeRegularizationLoss(model2) / ewc1.ComputeRegularizationLoss(model1), 1e-9);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task EWC_OnlineMode_FollowsProgressAndCompress()
    {
        await Task.Yield();
        // Schwarz et al. 2018, Eqs. 8–9: F*_i = γ F*_(i−1) + F_i, anchored at the latest task's parameters, and the
        // penalty weighted by γ F*.
        var ewc = Ewc(1.0, options: new EWCOptions<double> { UseOnlineEWC = true, OnlineDecayFactor = 0.9 });
        Assert.Equal("Online-EWC", ewc.Name);
        var model = CreateMockModel(3, new double[] { 1.0, 2.0, 3.0 });
        LearnTask(ewc, model, new[] { 1.0, 0.5, 0.2 });
        model.SetParameters(new Vector<double>(new double[] { 1.5, 2.5, 3.5 }));
        LearnTask(ewc, model, new[] { 0.3, 0.8, 1.0 });
        var f = ewc.AccumulatedFisher ?? throw new InvalidOperationException("no accumulated Fisher");
        Assert.Equal(0.9 * 1.0 + 0.09, f[0], Tol);
        Assert.Equal(0.9 * 0.25 + 0.64, f[1], Tol);
        Assert.Equal(0.9 * 0.04 + 1.0, f[2], Tol);
        Assert.Equal(new[] { 1.5, 2.5, 3.5 }, (ewc.ConsolidatedParameters ?? throw new InvalidOperationException()).ToArray());
        model.SetParameters(new Vector<double>(new double[] { 2.5, 2.5, 3.5 }));
        var adjusted = ewc.AdjustGradients(model, new Vector<double>(3));
        Assert.Equal(0.9 * f[0] * 1.0, adjusted[0], Tol);
        Assert.Equal(0.0, adjusted[1], Tol);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task EWC_FinalizeWithoutTaskData_Throws()
    {
        await Task.Yield();
        var ewc = Ewc(1.0);
        Assert.Throws<InvalidOperationException>(() => ewc.FinalizeTask(CreateMockModel(3)));
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task EWC_OptimalParameters_StoredCorrectly()
    {
        await Task.Yield();
        var expectedParams = new double[] { 1.5, -0.3, 2.7 };
        var ewc = Ewc(100.0);
        var model = CreateMockModel(3, expectedParams);
        LearnTask(ewc, model, new[] { 1.0, 1.0, 1.0 });
        Assert.Single(ewc.OptimalParameters);
        for (int i = 0; i < expectedParams.Length; i++)
            Assert.Equal(expectedParams[i], ewc.OptimalParameters[0][i], Tol);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task EWC_Reset_ClearsAllState()
    {
        await Task.Yield();
        var ewc = Ewc(100.0);
        var model = CreateMockModel(3, new double[] { 1, 2, 3 });
        LearnTask(ewc, model, new[] { 1.0, 1.0, 1.0 });
        Assert.NotEmpty(ewc.OptimalParameters);
        ewc.Reset();
        Assert.Empty(ewc.OptimalParameters);
        Assert.Empty(ewc.FisherInformation);
        Assert.Null(ewc.AccumulatedFisher);
    }

    // ═══════════════════════════════════════════════════════════════════
    // MAS: Memory Aware Synapses (Aljundi et al. 2018)
    // ═══════════════════════════════════════════════════════════════════

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task MAS_ImportanceAndPenaltyGradient_FollowEquations2And3()
    {
        await Task.Yield();
        // Ω = mean_k |∂ℓ2²(F(x_k))/∂θ| (Eq. 2): the mock reports ±[0.5, 1.0, 0.2] for the squared-output-norm objective,
        // so Ω = [0.5, 1.0, 0.2]. Penalty λ Σ Ω (θ − θ*)² (Eq. 3) has gradient 2 λ Ω (θ − θ*) = 2 · 3 · Ω · 0.5 = 3 Ω.
        var mas = new MemoryAwareSynapses<double, Tensor<double>, Tensor<double>>(new MeanSquaredErrorLoss<double>(),
            new MASOptions<double> { Lambda = 3.0, NumSamples = 4 });
        var model = CreateMockModel(3, new double[] { 1.0, 2.0, 3.0 });
        int calls = 0;
        model.GradientFunction = (input, target, loss) =>
        {
            calls++;
            // The objective MAS differentiates is the squared norm of the output, whatever the target.
            Assert.NotNull(loss);
            Assert.Equal(1.0 * 1.0 + 2.0 * 2.0, loss!.CalculateLoss(new Vector<double>(new[] { 1.0, 2.0 }), new Vector<double>(2)), Tol);
            double sign = calls % 2 == 0 ? 1 : -1;
            return new Vector<double>(new[] { 0.5 * sign, -1.0 * sign, 0.2 * sign });
        };
        mas.PrepareForTask(model, CreateMockDataset(4));
        mas.FinalizeTask(model);
        Assert.Equal(4, calls);
        model.SetParameters(new Vector<double>(new double[] { 1.5, 2.5, 3.5 }));
        var adjusted = mas.AdjustGradients(model, new Vector<double>(3));
        Assert.Equal(1.5, adjusted[0], Tol);
        Assert.Equal(3.0, adjusted[1], Tol);
        Assert.Equal(0.6, adjusted[2], Tol);
        // λ Σ Ω (θ − θ*)² = 3 · 0.25 · 1.7 = 1.275
        Assert.Equal(1.275, mas.ComputeRegularizationLoss(model), Tol);
    }

    // ═══════════════════════════════════════════════════════════════════
    // SI: Synaptic Intelligence
    // ═══════════════════════════════════════════════════════════════════

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task SI_NoTasks_RegularizationLossIsZero()
    {
        await Task.Yield();
        var si = new SynapticIntelligence<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(), lambda: 1.0);
        var model = CreateMockModel(5);

        var loss = si.ComputeRegularizationLoss(model);
        Assert.Equal(0.0, loss, Tol);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task SI_Name_IsCorrect()
    {
        await Task.Yield();
        var si = new SynapticIntelligence<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(), lambda: 1.0);
        Assert.Equal("Synaptic-Intelligence", si.Name);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task SI_ImportanceComputation_PathIntegral()
    {
        await Task.Yield();
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
        si.AdjustGradients(model, new Vector<double>(new double[] { 0.5, 1.0, 0.2 }));

        // Simulate parameter update
        var newParams = new Vector<double>(new double[] { 1.1, 2.2, 3.3 });
        si.NotifyParameterUpdate(newParams);

        // Update model params for FinalizeTask
        model.SetParameters(newParams);
        si.FinalizeTask(model);

        Assert.False(si.IsTrackingTask);
        Assert.NotNull(si.ConsolidatedImportance);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task SI_SameParameters_LossIsZero()
    {
        await Task.Yield();
        var si = new SynapticIntelligence<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(), lambda: 10.0);

        var model = CreateMockModel(3, new double[] { 1.0, 2.0, 3.0 });
        var dataset = CreateMockDataset(5);

        si.PrepareForTask(model, dataset);
        si.AdjustGradients(model, new Vector<double>(new double[] { 0.1, 0.2, 0.3 }));
        si.NotifyParameterUpdate(new Vector<double>(new double[] { 1.1, 2.1, 3.1 }));
        model.SetParameters(new Vector<double>(new double[] { 1.1, 2.1, 3.1 }));
        si.FinalizeTask(model);

        // After finalize, taskStartParameters is updated to current.
        // So computing loss with SAME params → (θ - θ*)² = 0 → loss = 0
        var loss = si.ComputeRegularizationLoss(model);
        Assert.Equal(0.0, loss, Tol);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task SI_ImportanceAccumulation_SumMode()
    {
        await Task.Yield();
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
        si.AdjustGradients(model, new Vector<double>(new double[] { 1.0, 2.0 }));
        si.NotifyParameterUpdate(new Vector<double>(new double[] { 0.5, 1.0 }));
        model.SetParameters(new Vector<double>(new double[] { 0.5, 1.0 }));
        si.FinalizeTask(model);

        var omega1 = si.ConsolidatedImportance;
        Assert.NotNull(omega1);
        var val0Task1 = omega1[0];
        var val1Task1 = omega1[1];

        // Task 2
        si.PrepareForTask(model, dataset);
        si.AdjustGradients(model, new Vector<double>(new double[] { 0.5, 0.5 }));
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

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task SI_ImportanceAccumulation_MaxMode()
    {
        await Task.Yield();
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
        si.AdjustGradients(model, new Vector<double>(new double[] { 2.0, 0.5 }));
        si.NotifyParameterUpdate(new Vector<double>(new double[] { 0.5, 0.5 }));
        model.SetParameters(new Vector<double>(new double[] { 0.5, 0.5 }));
        si.FinalizeTask(model);

        // Task 2
        si.PrepareForTask(model, dataset);
        si.AdjustGradients(model, new Vector<double>(new double[] { 0.5, 2.0 }));
        si.NotifyParameterUpdate(new Vector<double>(new double[] { 1.0, 1.0 }));
        model.SetParameters(new Vector<double>(new double[] { 1.0, 1.0 }));
        si.FinalizeTask(model);

        // In Max mode, each parameter keeps the larger importance
        Assert.NotNull(si.ConsolidatedImportance);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task SI_ImportanceAccumulation_WeightedSumMode()
    {
        await Task.Yield();
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
        si.AdjustGradients(model, new Vector<double>(new double[] { 1.0, 1.0 }));
        si.NotifyParameterUpdate(new Vector<double>(new double[] { 0.5, 0.5 }));
        model.SetParameters(new Vector<double>(new double[] { 0.5, 0.5 }));
        si.FinalizeTask(model);

        // In WeightedSum mode: Ω_new = decay * Ω_old + Ω_task
        // So old importance is decayed
        Assert.NotNull(si.ConsolidatedImportance);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task SI_Reset_ClearsAllState()
    {
        await Task.Yield();
        var si = new SynapticIntelligence<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(), lambda: 1.0);

        var model = CreateMockModel(3, new double[] { 1.0, 2.0, 3.0 });
        var dataset = CreateMockDataset(5);

        si.PrepareForTask(model, dataset);
        si.AdjustGradients(model, new Vector<double>(new double[] { 0.1, 0.2, 0.3 }));
        si.NotifyParameterUpdate(new Vector<double>(new double[] { 1.1, 2.1, 3.1 }));
        model.SetParameters(new Vector<double>(new double[] { 1.1, 2.1, 3.1 }));
        si.FinalizeTask(model);

        Assert.NotNull(si.ConsolidatedImportance);

        si.Reset();

        Assert.Null(si.ConsolidatedImportance);
        Assert.Null(si.OptimalParameters);
        Assert.False(si.IsTrackingTask);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task SI_DampingPreventsZeroDivision()
    {
        await Task.Yield();
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
        si.AdjustGradients(model, new Vector<double>(new double[] { 1.0, 1.0 }));
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

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task ReplayBuffer_InitialState_EmptyBuffer()
    {
        await Task.Yield();
        var buffer = new ExperienceReplayBuffer<double, double[], double[]>(
            maxSize: 100);

        Assert.Equal(0, buffer.Count);
        Assert.Equal(100, buffer.MaxSize);
        Assert.False(buffer.IsFull);
        Assert.Equal(0, buffer.TaskCount);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task ReplayBuffer_NegativeMaxSize_Throws()
    {
        await Task.Yield();
        Assert.Throws<ArgumentException>(() =>
            new ExperienceReplayBuffer<double, double[], double[]>(maxSize: -1));
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task ReplayBuffer_SampleBatch_ReturnsCorrectSize()
    {
        await Task.Yield();
        var buffer = new ExperienceReplayBuffer<double, double[], double[]>(
            maxSize: 100,
            replayStrategy: ReplaySamplingStrategy.Uniform,
            seed: 42);

        var dataset = CreateSimpleDataset(20);
        buffer.AddTaskExamples(dataset, taskId: 0);

        var batch = buffer.SampleBatch(5);
        Assert.Equal(5, batch.Count);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task ReplayBuffer_SampleBatchLargerThanBuffer_ReturnsBufferSize()
    {
        await Task.Yield();
        var buffer = new ExperienceReplayBuffer<double, double[], double[]>(
            maxSize: 100,
            replayStrategy: ReplaySamplingStrategy.Uniform,
            seed: 42);

        var dataset = CreateSimpleDataset(5);
        buffer.AddTaskExamples(dataset, taskId: 0);

        var batch = buffer.SampleBatch(50);
        Assert.True(batch.Count <= buffer.Count);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task ReplayBuffer_EmptyBuffer_SampleReturnsEmpty()
    {
        await Task.Yield();
        var buffer = new ExperienceReplayBuffer<double, double[], double[]>(maxSize: 100);
        var batch = buffer.SampleBatch(5);
        Assert.Empty(batch);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task ReplayBuffer_TaskTracking_CorrectCounts()
    {
        await Task.Yield();
        var buffer = new ExperienceReplayBuffer<double, double[], double[]>(
            maxSize: 100, seed: 42);

        buffer.AddTaskExamples(CreateSimpleDataset(10), taskId: 0);
        buffer.AddTaskExamples(CreateSimpleDataset(10), taskId: 1);

        Assert.Equal(2, buffer.TaskCount);
        var counts = buffer.GetTaskCounts();
        Assert.True(counts.ContainsKey(0));
        Assert.True(counts.ContainsKey(1));
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task ReplayBuffer_Statistics_CorrectFillRatio()
    {
        await Task.Yield();
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

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task ReplayBuffer_Clear_EmptiesBuffer()
    {
        await Task.Yield();
        var buffer = new ExperienceReplayBuffer<double, double[], double[]>(
            maxSize: 100, seed: 42);

        buffer.AddTaskExamples(CreateSimpleDataset(20), taskId: 0);
        Assert.True(buffer.Count > 0);

        buffer.Clear();
        Assert.Equal(0, buffer.Count);
        Assert.Equal(0, buffer.TaskCount);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task ReplayBuffer_UpdatePriority_ValidRange()
    {
        await Task.Yield();
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

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task ReplayBuffer_SampleFromTask_ReturnsCorrectTaskData()
    {
        await Task.Yield();
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

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task ReplayBuffer_SampleFromNonexistentTask_ReturnsEmpty()
    {
        await Task.Yield();
        var buffer = new ExperienceReplayBuffer<double, double[], double[]>(
            maxSize: 100, seed: 42);

        buffer.AddTaskExamples(CreateSimpleDataset(10), taskId: 0);

        var samples = buffer.SampleFromTask(999, 5);
        Assert.Empty(samples);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task ReplayBuffer_GetAll_ReturnsAllStoredData()
    {
        await Task.Yield();
        var buffer = new ExperienceReplayBuffer<double, double[], double[]>(
            maxSize: 100, seed: 42);

        buffer.AddTaskExamples(CreateSimpleDataset(5), taskId: 0, samplesPerTask: 5);

        var all = buffer.GetAll();
        Assert.Equal(buffer.Count, all.Count);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task ReplayBuffer_ReplayStrategies_AllReturnValidData()
    {
        await Task.Yield();
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

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task DataPoint_Constructor_StoresCorrectly()
    {
        await Task.Yield();
        var input = new double[] { 1.0, 2.0, 3.0 };
        var output = new double[] { 4.0, 5.0 };
        var dp = new DataPoint<double, double[], double[]>(input, output, taskId: 7);

        Assert.Same(input, dp.Input);
        Assert.Same(output, dp.Output);
        Assert.Equal(7, dp.TaskId);
        Assert.Contains("TaskId=7", dp.ToString());
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task ReplayBuffer_TotalSamplesProcessed_Tracks()
    {
        await Task.Yield();
        var buffer = new ExperienceReplayBuffer<double, double[], double[]>(
            maxSize: 100, seed: 42);

        Assert.Equal(0, buffer.TotalSamplesProcessed);

        buffer.AddTaskExamples(CreateSimpleDataset(10), taskId: 0);
        Assert.Equal(10, buffer.TotalSamplesProcessed);

        buffer.AddTaskExamples(CreateSimpleDataset(5), taskId: 1);
        Assert.Equal(15, buffer.TotalSamplesProcessed);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task ReplayBuffer_TotalReplaySamples_Tracks()
    {
        await Task.Yield();
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

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task EWC_Properties_Accessible()
    {
        await Task.Yield();
        var ewc = new ElasticWeightConsolidation<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(), lambda: 42.0);

        Assert.Equal(42.0, ewc.Lambda, Tol);
        Assert.Equal("EWC", ewc.Name);
        Assert.False(ewc.RequiresMemoryBuffer);
        Assert.False(ewc.ModifiesArchitecture);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task SI_Properties_Accessible()
    {
        await Task.Yield();
        var options = new SIOptions<double> { Lambda = 5.0, Damping = 0.2 };
        var si = new SynapticIntelligence<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>(), options);

        Assert.Equal(5.0, si.Lambda, Tol);
        Assert.Equal(0.2, si.Damping, Tol);
        Assert.Equal("Synaptic-Intelligence", si.Name);
        Assert.False(si.RequiresMemoryBuffer);
        Assert.False(si.ModifiesArchitecture);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task EWC_GetMetrics_ReturnsExpectedKeys()
    {
        await Task.Yield();
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
    /// Mock model for continual learning tests. IParameterizable and
    /// IGradientComputable are declared explicitly because IFullModel no
    /// longer extends them — the CL strategies resolve both capabilities
    /// at runtime via InterfaceGuard, which throws for models that only
    /// declare IFullModel.
    /// </summary>
    // Implements IParameterizable explicitly: the interface-segregation refactor
    // moved GetParameters/SetParameters/ParameterCount/WithParameters off IFullModel
    // and onto IParameterizable, and EWC/SI guard-require it via
    // InterfaceGuard.Parameterizable (EWC's Fisher matrix and SI's importance are
    // computed OVER the model's parameters). The mock already implements every
    // IParameterizable member below; it just needs to declare the interface so the
    // runtime `model is IParameterizable<...>` check succeeds. IGradientComputable
    // is declared for the same reason — the mock implements ComputeGradients/
    // ApplyGradients, and the gradient-based CL paths resolve it via
    // InterfaceGuard.GradientComputable.
    private class CLMockModel : IFullModel<double, Tensor<double>, Tensor<double>>,
        IParameterizable<double, Tensor<double>, Tensor<double>>,
        IGradientComputable<double, Tensor<double>, Tensor<double>>
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
        public long ParameterCount => _parameters.Length;
        public bool SupportsParameterInitialization => ParameterCount > 0;
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
        /// <summary>Per-example gradients the mock reports; zeros unless a test sets it.</summary>
        public Func<Tensor<double>, Tensor<double>, ILossFunction<double>?, Vector<double>>? GradientFunction { get; set; }

        public Vector<double> ComputeGradients(Tensor<double> input, Tensor<double> target, ILossFunction<double>? lossFunction = null)
            => GradientFunction?.Invoke(input, target, lossFunction) ?? new Vector<double>(_parameters.Length);
        public void ApplyGradients(Vector<double> gradients, double learningRate) { }

    public Vector<double> SanitizeParameters(Vector<double> parameters) => parameters;

        public void Dispose() { /* test mock holds no resources */ }
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
