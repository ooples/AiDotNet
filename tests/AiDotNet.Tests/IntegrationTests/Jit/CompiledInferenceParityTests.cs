using System;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Jit;

/// <summary>
/// Compiled-inference parity + replay-cost regression tests, reproducing the three
/// defects the AIsEval compiled-mode benchmark surfaced (2026-06: compiled mode
/// scored 2/16 vs eager's 4/16 against same-rig PyTorch):
///
///  1. CNN compiled plan produced COMPLETELY WRONG output (relErr = 1.0 vs eager)
///     — caught only because the benchmark compared outputs before trusting it.
///  2. MLP bs=128 replay ran 11.9 ms vs 1.5 ms eager (8x) — a replayed plan must
///     never lose badly to the eager walk it traced. The suspected mechanism is the
///     BatchDynamic symbolic plan traced at one batch size replaying degenerately
///     at another, so parity is asserted at EVERY batch size, not just the traced one.
///  3. Plan buffer residency (~100 MB-class per model x shape) contaminating the
///     whole process (covered by CompiledPlanMemoryTests once the bound lands).
///
/// Parity tests compare <see cref="NeuralNetworkBase{T}.PredictCompiled"/> (replay)
/// against <see cref="NeuralNetworkBase{T}.Predict"/> (eager) on the SAME network and
/// SAME input — any divergence beyond float reduction-order noise is a plan bug.
/// </summary>
public class CompiledInferenceParityTests : IDisposable
{
    private readonly TensorCodecOptions _originalOptions;

    public CompiledInferenceParityTests()
    {
        AiDotNetEngine.ResetToCpu();
        _originalOptions = TensorCodecOptions.Current;
        TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
    }

    public void Dispose() => TensorCodecOptions.SetCurrent(_originalOptions);

    // ---------------------------------------------------------------------
    // Defect 1: CNN compiled plan correctness.
    // ---------------------------------------------------------------------

    [Fact]
    public void Cnn_CompiledReplay_MatchesEager()
    {
        var network = BuildAisEvalCnn();
        var input = MakeInput(new[] { 2, 1, 28, 28 }, seed: 7);

        var eager = network.Predict(input);
        Assert.True(network.CompileForward(input), "CompileForward failed for the CNN — trace did not produce a plan.");
        var compiled = network.PredictCompiled(input);

        AssertParity(eager, compiled, "CNN [2,1,28,28]");
    }

    // ---------------------------------------------------------------------
    // Defect 2a: BatchDynamic plan parity at batch sizes OTHER than the traced one.
    // The AIsEval benchmark compiles at bs=1 first; with SymbolicShapeMode.BatchDynamic
    // that plan then serves bs=8/32/128 via dynamic rebind — every one of those
    // replays must match eager.
    // ---------------------------------------------------------------------

    [Theory]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(32)]
    [InlineData(128)]
    public void Mlp_CompiledReplay_MatchesEager_AcrossBatchSizes(int batch)
    {
        var network = BuildAisEvalMlp();

        // Trace at bs=1 (the AIsEval warm order), then replay at the probed batch.
        var traceInput = MakeInput(new[] { 1, 784 }, seed: 11);
        Assert.True(network.CompileForward(traceInput), "CompileForward failed for the MLP.");

        var input = MakeInput(new[] { batch, 784 }, seed: 23 + batch);
        var eager = network.Predict(input);
        var compiled = network.PredictCompiled(input);

        AssertParity(eager, compiled, $"MLP [{batch},784] (plan traced at bs=1)");
    }

    [Theory]
    [InlineData(8)]
    [InlineData(128)]
    public void Cnn_CompiledReplay_MatchesEager_AcrossBatchSizes(int batch)
    {
        var network = BuildAisEvalCnn();
        var traceInput = MakeInput(new[] { 1, 1, 28, 28 }, seed: 13);
        Assert.True(network.CompileForward(traceInput), "CompileForward failed for the CNN.");

        var input = MakeInput(new[] { batch, 1, 28, 28 }, seed: 41 + batch);
        var eager = network.Predict(input);
        var compiled = network.PredictCompiled(input);

        AssertParity(eager, compiled, $"CNN [{batch},1,28,28] (plan traced at bs=1)");
    }

    // ---------------------------------------------------------------------
    // Defect 2b: replay must not be drastically slower than the eager walk it
    // traced. The AIsEval run measured 8x at mlp bs=128; the 3x ceiling here is
    // generous enough for rig noise while still catching that pathology.
    // ---------------------------------------------------------------------

    [Fact]
    public void Mlp_CompiledReplay_NotDrasticallySlowerThanEager_AtBs128()
    {
        var network = BuildAisEvalMlp();
        var input = MakeInput(new[] { 128, 784 }, seed: 5);

        Assert.True(network.CompileForward(input), "CompileForward failed for the MLP at bs=128.");

        // Parity first — a fast-but-wrong replay must fail here, not pass on time.
        var eager = network.Predict(input);
        var compiled = network.PredictCompiled(input);
        AssertParity(eager, compiled, "MLP [128,784]");

        double eagerUs = MinMicros(30, () => network.Predict(input));
        double replayUs = MinMicros(30, () => network.PredictCompiled(input));

        Assert.True(
            replayUs <= eagerUs * 3.0,
            $"Compiled replay ({replayUs:F0} us) is more than 3x the eager forward ({eagerUs:F0} us) at mlp [128,784] — " +
            "replay of a traced plan must not drastically lose to the eager walk it captured (AIsEval measured 8x).");
    }

    // ---------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------

    private static NeuralNetworkBase<float> BuildAisEvalMlp()
    {
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(512, (IActivationFunction<float>)new ReLUActivation<float>()),
            new DenseLayer<float>(128, (IActivationFunction<float>)new ReLUActivation<float>()),
            new DenseLayer<float>(10, activationFunction: (IActivationFunction<float>?)null),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 784,
            outputSize: 10,
            layers: layers);
        return new FeedForwardNeuralNetwork<float>(arch);
    }

    private static NeuralNetworkBase<float> BuildAisEvalCnn()
    {
        var layers = new List<ILayer<float>>
        {
            new ConvolutionalLayer<float>(outputDepth: 16, kernelSize: 3, stride: 1, padding: 1,
                                          activationFunction: new ReLUActivation<float>()),
            new MaxPoolingLayer<float>(poolSize: 2, stride: 2),
            new ConvolutionalLayer<float>(outputDepth: 32, kernelSize: 3, stride: 1, padding: 1,
                                          activationFunction: new ReLUActivation<float>()),
            new MaxPoolingLayer<float>(poolSize: 2, stride: 2),
            new FlattenLayer<float>(),
            new DenseLayer<float>(10, activationFunction: (IActivationFunction<float>?)null),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 28, inputWidth: 28, inputDepth: 1,
            outputSize: 10,
            layers: layers);
        return new ConvolutionalNeuralNetwork<float>(arch);
    }

    private static Tensor<float> MakeInput(int[] shape, int seed)
    {
        var rng = new Random(seed);
        int len = 1;
        foreach (var d in shape) len *= d;
        var data = new float[len];
        for (int i = 0; i < len; i++) data[i] = (float)rng.NextDouble();
        return new Tensor<float>(data, shape);
    }

    private static void AssertParity(Tensor<float> eager, Tensor<float> compiled, string label)
    {
        Assert.Equal(eager.Length, compiled.Length);
        var e = eager.AsSpan();
        var c = compiled.AsSpan();
        double maxAbs = 0, maxMag = 1e-6;
        for (int i = 0; i < e.Length; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(e[i] - c[i]));
            maxMag = Math.Max(maxMag, Math.Abs(e[i]));
        }
        double rel = maxAbs / maxMag;
        Assert.True(rel < 1e-3,
            $"{label}: compiled output diverges from eager (relErr={rel:E2}, maxAbs={maxAbs:E2}). " +
            "Replay of a traced plan must reproduce the eager forward.");
    }

    private static double MinMicros(int iters, Action act)
    {
        // Warm.
        for (int i = 0; i < 5; i++) act();
        double min = double.MaxValue;
        for (int i = 0; i < iters; i++)
        {
            long t0 = Stopwatch.GetTimestamp();
            act();
            double us = (Stopwatch.GetTimestamp() - t0) * 1e6 / Stopwatch.Frequency;
            if (us < min) min = us;
        }
        return min;
    }
}
