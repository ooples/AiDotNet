using AiDotNet.Enums;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Optimizers.Fused;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.Optimizers;

/// <summary>
/// Parity guards for <c>GradientBasedOptimizerBase.TryGetFusedLrSchedule</c> —
/// the seam that lets an LR scheduler run on the fused-compiled training path.
/// The fused plan evaluates <c>LrSchedule.GetLr(step)</c> once per optimizer
/// step (the same model as PyTorch <c>fused=True</c>), so each mapped AiDotNet
/// scheduler MUST produce a per-step LR sequence bit-identical to its eager
/// counterpart — otherwise the fused fast path would train differently from the
/// eager reference. These tests construct an Adam carrying each scheduler, pull
/// the mapped fused <c>LrSchedule</c> out of the optimizer's fused config, and
/// assert step-for-step equality against the eager scheduler.
/// </summary>
public class FusedLrScheduleMappingTests
{
    private const double Tol = 1e-9;
    private readonly ITestOutputHelper _output;
    public FusedLrScheduleMappingTests(ITestOutputHelper output) => _output = output;

    private static AiDotNet.Tensors.Engines.Compilation.LrSchedule? MapToFused(ILearningRateScheduler scheduler)
    {
        var adam = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
            null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                LearningRateScheduler = scheduler,
                SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            });
        Assert.True(((IFusedOptimizerSpec)adam).TryGetFusedOptimizerConfig(out var cfg),
            $"{scheduler.GetType().Name} should map to a fused-optimizer config.");
        return cfg.Schedule;
    }

    /// <summary>
    /// Replays the eager scheduler's per-batch LR sequence (ctor value for batch
    /// 1, then one Step() per subsequent batch) and asserts it equals the fused
    /// schedule's GetLr(N) for every batch N in [1, steps].
    /// </summary>
    private void AssertStepForStepParity(
        ILearningRateScheduler eager,
        AiDotNet.Tensors.Engines.Compilation.LrSchedule fused,
        int steps,
        string label)
    {
        double eagerLr = eager.CurrentLearningRate;   // batch 1 (set in ctor)
        for (int n = 1; n <= steps; n++)
        {
            double fusedLr = fused.GetLr(n);
            Assert.True(System.Math.Abs(eagerLr - fusedLr) <= Tol,
                $"{label}: batch {n} eager={eagerLr:E6} vs fused={fusedLr:E6} (Δ={System.Math.Abs(eagerLr - fusedLr):E3})");
            eagerLr = eager.Step();                    // advance to batch n+1
        }
        _output.WriteLine($"{label}: step-for-step parity over {steps} batches ✓");
    }

    [Fact]
    public void StepLR_MapsToFusedStep_StepForStepParity()
    {
        var fused = MapToFused(new StepLRScheduler(baseLearningRate: 0.1, stepSize: 7, gamma: 0.5));
        Assert.NotNull(fused);
        AssertStepForStepParity(
            new StepLRScheduler(baseLearningRate: 0.1, stepSize: 7, gamma: 0.5),
            fused!, steps: 50, label: "StepLR");
    }

    [Fact]
    public void CyclicLR_TriangularSymmetric_MapsToFusedCyclic_StepForStepParity()
    {
        var fused = MapToFused(new CyclicLRScheduler(
            baseLearningRate: 0.001, maxLearningRate: 0.1, stepSizeUp: 10, stepSizeDown: 10,
            mode: CyclicLRScheduler.CyclicMode.Triangular));
        Assert.NotNull(fused);
        AssertStepForStepParity(
            new CyclicLRScheduler(
                baseLearningRate: 0.001, maxLearningRate: 0.1, stepSizeUp: 10, stepSizeDown: 10,
                mode: CyclicLRScheduler.CyclicMode.Triangular),
            fused!, steps: 60, label: "CyclicLR(Triangular)");
    }

    [Fact]
    public void CyclicLR_NonCanonical_FallsBackToEager_NotFused()
    {
        // Triangular2 has no fused shape (per-cycle amplitude decay) → must NOT map.
        var t2 = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
            null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                LearningRateScheduler = new CyclicLRScheduler(
                    baseLearningRate: 0.001, maxLearningRate: 0.1, stepSizeUp: 10, stepSizeDown: 10,
                    mode: CyclicLRScheduler.CyclicMode.Triangular2),
                SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            });
        Assert.False(((IFusedOptimizerSpec)t2).TryGetFusedOptimizerConfig(out _),
            "Triangular2 cyclic has no fused equivalent and must fall back to the eager path.");

        // Asymmetric up≠down also has no fused shape.
        var asym = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
            null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                LearningRateScheduler = new CyclicLRScheduler(
                    baseLearningRate: 0.001, maxLearningRate: 0.1, stepSizeUp: 5, stepSizeDown: 15,
                    mode: CyclicLRScheduler.CyclicMode.Triangular),
                SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            });
        Assert.False(((IFusedOptimizerSpec)asym).TryGetFusedOptimizerConfig(out _),
            "Asymmetric cyclic (up≠down) has no fused equivalent and must fall back to eager.");
    }

    [Fact]
    public void LinearWarmup_Constant_MapsToFused_StepForStepParity()
    {
        // Warmup-then-constant is the HRE cortex recipe; before the LinearWarmup
        // mapping it fell through to the eager tape (no CUDA-graph capture, 14-23x).
        var fused = MapToFused(new LinearWarmupScheduler(
            baseLearningRate: 0.002, warmupSteps: 10,
            decayMode: LinearWarmupScheduler.DecayMode.Constant));
        Assert.NotNull(fused);
        AssertStepForStepParity(
            new LinearWarmupScheduler(
                baseLearningRate: 0.002, warmupSteps: 10,
                decayMode: LinearWarmupScheduler.DecayMode.Constant),
            fused!, steps: 40, label: "LinearWarmup(Constant)");
    }

    [Fact]
    public void LinearWarmup_LinearDecay_MapsToFused_StepForStepParity()
    {
        var fused = MapToFused(new LinearWarmupScheduler(
            baseLearningRate: 0.1, warmupSteps: 8, totalSteps: 50,
            decayMode: LinearWarmupScheduler.DecayMode.Linear, endLr: 0.001));
        Assert.NotNull(fused);
        AssertStepForStepParity(
            new LinearWarmupScheduler(
                baseLearningRate: 0.1, warmupSteps: 8, totalSteps: 50,
                decayMode: LinearWarmupScheduler.DecayMode.Linear, endLr: 0.001),
            fused!, steps: 60, label: "LinearWarmup(Linear)");
    }

    [Fact]
    public void LinearWarmup_CosineDecay_MapsToFused_StepForStepParity()
    {
        var fused = MapToFused(new LinearWarmupScheduler(
            baseLearningRate: 0.05, warmupSteps: 12, totalSteps: 80,
            warmupInitLr: 1e-5,
            decayMode: LinearWarmupScheduler.DecayMode.Cosine, endLr: 1e-4));
        Assert.NotNull(fused);
        AssertStepForStepParity(
            new LinearWarmupScheduler(
                baseLearningRate: 0.05, warmupSteps: 12, totalSteps: 80,
                warmupInitLr: 1e-5,
                decayMode: LinearWarmupScheduler.DecayMode.Cosine, endLr: 1e-4),
            fused!, steps: 90, label: "LinearWarmup(Cosine)");
    }

    [Fact]
    public void ExistingMappings_CosineAndExponential_RemainStepForStepCorrect()
    {
        var cosFused = MapToFused(new CosineAnnealingLRScheduler(baseLearningRate: 0.1, tMax: 50));
        Assert.NotNull(cosFused);
        AssertStepForStepParity(
            new CosineAnnealingLRScheduler(baseLearningRate: 0.1, tMax: 50),
            cosFused!, steps: 50, label: "CosineAnnealing");

        var expFused = MapToFused(new ExponentialLRScheduler(baseLearningRate: 0.1, gamma: 0.95));
        Assert.NotNull(expFused);
        AssertStepForStepParity(
            new ExponentialLRScheduler(baseLearningRate: 0.1, gamma: 0.95),
            expFused!, steps: 40, label: "Exponential");
    }
}
