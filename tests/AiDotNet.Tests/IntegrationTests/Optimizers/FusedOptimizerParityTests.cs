using System.Collections.Generic;
using System;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Training;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.Optimizers;

/// <summary>
/// Fused-vs-eager numerical-parity gate for the optimizers wired onto the
/// fused-compiled training path (#1447). An optimizer is only safe to map to a
/// Tensors fused kernel if training a model on the fused path produces the same
/// parameters as the eager tape — otherwise users silently get different results
/// depending on whether compilation engaged.
///
/// <para><b>Methodology:</b> train two identically-initialised MLPs on the same
/// data — one with <c>TensorCodecOptions.EnableCompilation = true</c> (fused),
/// one <c>= false</c> (eager) — and compare final parameters. Fused and eager
/// differ slightly even for a known-correct optimizer because the fused plan
/// orders float ops differently from the eager tape; <b>Adam is the control</b>
/// (already wired and known-correct), so each newly-wired optimizer must diverge
/// no more than Adam does. A wrong kernel mapping diverges by orders of
/// magnitude more. Each test also asserts the fused path actually engaged
/// (<c>GetFusedStepCount &gt; 0</c>) so the comparison isn't vacuously
/// eager-vs-eager.</para>
/// </summary>
[Collection("FusedTrainingSerial")]
public class FusedOptimizerParityTests
{
    private const int Steps = 40;
    private readonly ITestOutputHelper _output;
    public FusedOptimizerParityTests(ITestOutputHelper output) => _output = output;

    private static NeuralNetworkArchitecture<float> MakeArch() =>
        new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 8,
            outputSize: 3);

    private static (Tensor<float> x, Tensor<float> y) MakeData()
    {
        var x = new Tensor<float>(new[] { 4, 8 });
        var y = new Tensor<float>(new[] { 4, 3 });
        // Deterministic synthetic batch (fixed values, no RNG).
        for (int b = 0; b < 4; b++)
        {
            for (int f = 0; f < 8; f++) x[b, f] = (float)(((b * 8 + f) % 7) - 3) * 0.1f;
            for (int o = 0; o < 3; o++) y[b, o] = (float)(((b * 3 + o) % 5) - 2) * 0.2f;
        }
        return (x, y);
    }

    /// <summary>
    /// Trains a fused model and an identically-initialised eager model for
    /// <see cref="Steps"/> steps and returns the max abs parameter divergence
    /// plus the number of fused steps that actually engaged.
    /// </summary>
    private (double maxAbsDiff, long fusedSteps, double trainDelta) Divergence(
        Func<IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>>> optFactory)
    {
        // Pin the global default init seed. The model-family test base sets it to 1234 process-wide and never resets it,
        // so without this the initial weights, and every result here, depended on whether such a test had already run
        // in the process (FTRL's L1 held all weights at zero under one init and not the other).
        NeuralNetworkArchitecture<float>.DefaultRandomSeedOverride = 1234;
        var fused = new FeedForwardNeuralNetwork<float>(MakeArch(), optFactory(), new MeanSquaredErrorLoss<float>());
        var eager = new FeedForwardNeuralNetwork<float>(MakeArch(), optFactory(), new MeanSquaredErrorLoss<float>());
        // Identical initial weights: copy the fused model's init into the eager one.
        var init = fused.GetParameters();
        eager.UpdateParameters(init);
        fused.SetTrainingMode(true);
        eager.SetTrainingMode(true);
        var (x, y) = MakeData();

        // Install the options with SetCurrent. TensorCodecOptions.Current returns a fresh copy of the defaults on a
        // thread that never called SetCurrent, so assigning Current.EnableCompilation there changed nothing: the
        // "eager" run trained fused as well and every optimizer reported a divergence of exactly zero.
        var saved = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current;
        long fusedSteps;
        long eagerFusedSteps;
        try
        {
            // Fused run.
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(
                new AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions { EnableCompilation = true });
            CompiledTapeTrainingStep<float>.Invalidate();
            CompiledTapeTrainingStep<float>.ResetFusedStepCount();
            for (int i = 0; i < Steps; i++) fused.Train(x, y);
            fusedSteps = CompiledTapeTrainingStep<float>.GetFusedStepCount();

            // Eager run (compilation disabled → pure tape path).
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(
                new AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions { EnableCompilation = false });
            CompiledTapeTrainingStep<float>.Invalidate();
            CompiledTapeTrainingStep<float>.ResetFusedStepCount();
            for (int i = 0; i < Steps; i++) eager.Train(x, y);
            eagerFusedSteps = CompiledTapeTrainingStep<float>.GetFusedStepCount();
        }
        finally
        {
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(saved);
            CompiledTapeTrainingStep<float>.Invalidate();
        }

        // The comparison is only meaningful when the reference really ran eagerly.
        Assert.Equal(0, eagerFusedSteps);

        var pf = fused.GetParameters();
        var pe = eager.GetParameters();
        Assert.Equal(pf.Length, pe.Length);
        double maxAbs = 0, trainDelta = 0;
        for (int i = 0; i < pf.Length; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs((double)pf[i] - (double)pe[i]));
            trainDelta = Math.Max(trainDelta, Math.Abs((double)pf[i] - (double)init[i]));
        }
        return (maxAbs, fusedSteps, trainDelta);
    }

    private static AdamOptimizer<float, Tensor<float>, Tensor<float>> Adam() =>
        new(null, new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 });

    /// <summary>
    /// Every optimizer that maps onto a fused kernel must actually engage the fused path when a network trains, and
    /// train it the way its eager step does. FusedKernelParityTests drives each kernel directly from a fixed gradient
    /// sequence, which cannot notice an optimizer that maps onto a kernel but whose training never reaches it
    /// (Momentum, TrustRegion, ProximalGradientDescent and L-BFGS all did); this runs the real training step.
    /// </summary>
    [Theory]
    [MemberData(nameof(FusedKernelParityTests.Cases), MemberType = typeof(FusedKernelParityTests))]
    public void Training_EngagesTheFusedPath_AndMatchesTheEagerStep(string name)
    {
        var (adamDiff, _, _) = Divergence(Adam);
        var (diff, fusedSteps, trainDelta) = Divergence(() => FusedKernelParityTests.Create(name));
        _output.WriteLine($"{name}: fusedSteps={fusedSteps}, maxAbsDiff={diff:E3}, trainDelta={trainDelta:E3} (Adam control {adamDiff:E3})");
        Assert.True(fusedSteps > 0, $"{name} maps onto a fused kernel but training never engaged it (fusedSteps == 0).");
        Assert.True(trainDelta > 1e-6, $"{name}: training barely moved the parameters ({trainDelta:E3}); the comparison would be vacuous.");
        Assert.True(diff <= Math.Max(adamDiff * 10.0, 1e-4),
            $"{name}: fused and eager training differ by {diff:E3}, against {adamDiff:E3} for the Adam control.");
    }
    /// <summary>
    /// A warmup that starts at learning rate 0 (the LinearWarmupScheduler default) makes the first step change nothing.
    /// The #1822 persistence probe read that as a plan decoupled from the live tensors and disabled fused training for
    /// the rest of the run. It must stay fused for every step, and match the eager warmup.
    /// </summary>
    [Fact]
    public void Adam_WithAWarmupFromZero_StaysOnTheFusedPath()
    {
        var (adamDiff, _, _) = Divergence(Adam);
        var (diff, fusedSteps, trainDelta) = Divergence(() =>
            new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 1e-2,
                LearningRateScheduler = new AiDotNet.LearningRateSchedulers.LinearWarmupScheduler(1e-2, warmupSteps: 5),
                // Per batch: the cadence the compiled plan's per-step schedule expresses.
                SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
            }));
        _output.WriteLine($"Adam + warmup from 0: fusedSteps={fusedSteps}, maxAbsDiff={diff:E3} (Adam control {adamDiff:E3})");
        Assert.Equal(Steps, fusedSteps);
        Assert.True(trainDelta > 1e-6, $"training barely moved the parameters ({trainDelta:E3}); the comparison would be vacuous.");
        Assert.True(diff <= Math.Max(adamDiff * 10.0, 1e-4), $"fused and eager warmup differ by {diff:E3}, against {adamDiff:E3}.");
    }

    /// <summary>
    /// A scheduler stepped per epoch (the default mode) cannot be expressed by the compiled plan's per-step schedule.
    /// Mapping it made the fused path ramp the learning rate every batch while the eager path, following the
    /// configuration, held it until the epoch ended. It must stay eager, and so train exactly like the eager model.
    /// </summary>
    [Fact]
    public void Adam_WithAPerEpochScheduler_StaysEager_AndMatchesTheEagerStep()
    {
        var (diff, fusedSteps, _) = Divergence(() =>
            new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 1e-2,
                LearningRateScheduler = new AiDotNet.LearningRateSchedulers.CosineAnnealingLRScheduler(1e-2, tMax: 40),
                SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerEpoch,
            }));
        _output.WriteLine($"Adam + per-epoch cosine: fusedSteps={fusedSteps}, maxAbsDiff={diff:E3}");
        Assert.Equal(0, fusedSteps);
        Assert.Equal(0.0, diff);
    }
    /// <summary>
    /// FTRL's L1 term holds a weight at exactly zero while its accumulator stays inside lambda1, so under this init a
    /// fused step can leave every parameter unchanged. That is FTRL working, not a decoupled plan: FTRL must keep the
    /// fused path for every step instead of being reset to eager.
    /// </summary>
    [Fact]
    public void FTRL_KeepsTheFusedPath_WhenItsL1TermHoldsEveryWeightAtZero()
    {
        var (_, fusedSteps, _) = Divergence(() => FusedKernelParityTests.Create("FTRL"));
        _output.WriteLine($"FTRL: fusedSteps={fusedSteps}");
        Assert.Equal(Steps, fusedSteps);
    }
    [Fact]
    public void Adam_Control_FusedMatchesEager()
    {
        var (diff, fusedSteps, trainDelta) = Divergence(Adam);
        _output.WriteLine($"Adam control: fusedSteps={fusedSteps}, maxAbsDiff={diff:E3}");
        Assert.True(fusedSteps > 0, "Adam must engage the fused path (control is meaningless otherwise).");
        Assert.True(trainDelta > 1e-6,
            $"Adam control: training did not move parameters (trainDelta={trainDelta:E3}); the fused-vs-eager parity comparison is vacuous.");
        Assert.True(diff < 1e-3, $"Adam fused-vs-eager divergence {diff:E3} unexpectedly large — forward/backward float-order issue?");
    }

    private void AssertOptimizerParity(
        string name, long fusedSteps, double diff, double trainDelta, double adamDiff)
    {
        _output.WriteLine($"{name}: fusedSteps={fusedSteps}, maxAbsDiff={diff:E3}, trainDelta={trainDelta:E3} (Adam control {adamDiff:E3})");
        Assert.True(fusedSteps > 0,
            $"{name} must engage the fused path — fusedSteps==0 means the mapping didn't take (allowlist/spec).");
        // Non-vacuous guard: training must actually move the parameters, else a
        // 0 divergence is meaningless (two un-trained models trivially match).
        Assert.True(trainDelta > 1e-6,
            $"{name}: training did not change parameters (trainDelta={trainDelta:E3}); the parity comparison is vacuous.");
        Assert.True(diff <= Math.Max(adamDiff * 10.0, 1e-4),
            $"{name} fused-vs-eager divergence {diff:E3} ≫ Adam control {adamDiff:E3} — the fused kernel does not " +
            $"match AiDotNet's eager {name} update. Do NOT wire this mapping until reconciled.");
    }

    [Fact]
    public void Lion_FusedMatchesEager_NoWorseThanAdam()
    {
        var (adamDiff, _, _) = Divergence(Adam);
        var (diff, fusedSteps, trainDelta) = Divergence(() =>
            new LionOptimizer<float, Tensor<float>, Tensor<float>>(
                null, new LionOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 }));
        AssertOptimizerParity("Lion", fusedSteps, diff, trainDelta, adamDiff);
    }

    [Fact]
    public void AdaDelta_FusedMatchesEager_NoWorseThanAdam()
    {
        var (adamDiff, _, _) = Divergence(Adam);
        var (diff, fusedSteps, trainDelta) = Divergence(() =>
            new AdaDeltaOptimizer<float, Tensor<float>, Tensor<float>>(
                null, new AdaDeltaOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 }));
        AssertOptimizerParity("AdaDelta", fusedSteps, diff, trainDelta, adamDiff);
    }

    // LAMB is deliberately NOT fused (see LAMBOptimizer's IFusedOptimizerSpec): the kernel does not clamp the trust
    // ratio, and even unclamped its divergence grows past the parity bound after ~20 steps. Both configurations must
    // decline and still train on the eager path.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void LAMB_DeclinesTheFusedPath_AndTrainsEager(bool clipTrustRatio)
    {
        LAMBOptimizer<float, Tensor<float>, Tensor<float>> Create() => new(
            null, new LAMBOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 1e-2,
                ClipTrustRatio = clipTrustRatio,
            });

        AiDotNet.Optimizers.Fused.IFusedOptimizerSpec spec = Create();
        Assert.False(spec.TryGetFusedOptimizerConfig(out _), "LAMB mapped to the fused kernel.");
        var (_, fusedSteps, trainDelta) = Divergence(Create);
        Assert.Equal(0, fusedSteps);
        Assert.True(trainDelta > 1e-6, "LAMB did not train on the eager path.");
    }
}
