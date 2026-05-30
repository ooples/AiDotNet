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
    private (double maxAbsDiff, long fusedSteps) Divergence(
        Func<IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>>> optFactory)
    {
        var fused = new FeedForwardNeuralNetwork<float>(MakeArch(), optFactory(), new MeanSquaredErrorLoss<float>());
        var eager = new FeedForwardNeuralNetwork<float>(MakeArch(), optFactory(), new MeanSquaredErrorLoss<float>());
        // Identical initial weights: copy the fused model's init into the eager one.
        eager.UpdateParameters(fused.GetParameters());
        fused.SetTrainingMode(true);
        eager.SetTrainingMode(true);
        var (x, y) = MakeData();

        bool saved = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation;
        long fusedSteps;
        try
        {
            // Fused run.
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation = true;
            CompiledTapeTrainingStep<float>.Invalidate();
            CompiledTapeTrainingStep<float>.ResetFusedStepCount();
            for (int i = 0; i < Steps; i++) fused.Train(x, y);
            fusedSteps = CompiledTapeTrainingStep<float>.GetFusedStepCount();

            // Eager run (compilation disabled → pure tape path).
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation = false;
            CompiledTapeTrainingStep<float>.Invalidate();
            for (int i = 0; i < Steps; i++) eager.Train(x, y);
        }
        finally
        {
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation = saved;
            CompiledTapeTrainingStep<float>.Invalidate();
        }

        var pf = fused.GetParameters();
        var pe = eager.GetParameters();
        Assert.Equal(pf.Length, pe.Length);
        double maxAbs = 0;
        for (int i = 0; i < pf.Length; i++)
            maxAbs = Math.Max(maxAbs, Math.Abs((double)pf[i] - (double)pe[i]));
        return (maxAbs, fusedSteps);
    }

    private static AdamOptimizer<float, Tensor<float>, Tensor<float>> Adam() =>
        new(null, new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 });

    [Fact]
    public void Adam_Control_FusedMatchesEager()
    {
        var (diff, fusedSteps) = Divergence(Adam);
        _output.WriteLine($"Adam control: fusedSteps={fusedSteps}, maxAbsDiff={diff:E3}");
        Assert.True(fusedSteps > 0, "Adam must engage the fused path (control is meaningless otherwise).");
        Assert.True(diff < 1e-3, $"Adam fused-vs-eager divergence {diff:E3} unexpectedly large — forward/backward float-order issue?");
    }

    [Fact]
    public void AdaMax_FusedMatchesEager_NoWorseThanAdam()
    {
        var (adamDiff, _) = Divergence(Adam);
        var (diff, fusedSteps) = Divergence(() =>
            new AdaMaxOptimizer<float, Tensor<float>, Tensor<float>>(
                null, new AdaMaxOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 }));
        _output.WriteLine($"AdaMax: fusedSteps={fusedSteps}, maxAbsDiff={diff:E3} (Adam control {adamDiff:E3})");
        Assert.True(fusedSteps > 0,
            "AdaMax must engage the fused path (OptimizerType.AdaMax) — fusedSteps==0 means the mapping didn't take.");
        Assert.True(diff <= Math.Max(adamDiff * 10.0, 1e-4),
            $"AdaMax fused-vs-eager divergence {diff:E3} ≫ Adam control {adamDiff:E3} — the fused AdaMax kernel does " +
            "not match AiDotNet's eager AdaMax update. Do NOT wire this mapping until reconciled.");
    }

    [Fact]
    public void Nadam_FusedMatchesEager_NoWorseThanAdam()
    {
        var (adamDiff, _) = Divergence(Adam);
        var (diff, fusedSteps) = Divergence(() =>
            new NadamOptimizer<float, Tensor<float>, Tensor<float>>(
                null, new NadamOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 }));
        _output.WriteLine($"Nadam: fusedSteps={fusedSteps}, maxAbsDiff={diff:E3} (Adam control {adamDiff:E3})");
        Assert.True(fusedSteps > 0,
            "Nadam must engage the fused path (OptimizerType.Nadam) — fusedSteps==0 means the mapping didn't take.");
        Assert.True(diff <= Math.Max(adamDiff * 10.0, 1e-4),
            $"Nadam fused-vs-eager divergence {diff:E3} ≫ Adam control {adamDiff:E3} — the fused Nadam kernel does " +
            "not match AiDotNet's eager Nadam update. Do NOT wire this mapping until reconciled.");
    }
}
