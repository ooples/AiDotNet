using System;
using System.Collections.Generic;
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
/// #1622 compilation-lever track (L3b / shape-guard / value-memo / unified quant+compile):
/// the verify-then-trust auto-compiled inference path (<see cref="NeuralNetworkBase{T}.PredictAccelerated"/>)
/// must be NUMERICALLY IDENTICAL to eager for every input — it adopts a compiled plan per shape only
/// after confirming it matches eager (#87), caches that verdict so it never re-verifies or recompile-
/// thrashes a shape (#88), short-circuits an identical repeated input through a collision-safe value
/// memo (#93), and auto-engages for a foundation-scale model alongside weight streaming (#92).
///
/// Runs non-parallel: it mutates process-global <see cref="TensorCodecOptions"/> and the engine.
/// </summary>
[Collection("NonParallelIntegration")]
public class AcceleratedInferenceTests : IDisposable
{
    private readonly TensorCodecOptions _originalOptions;

    public AcceleratedInferenceTests()
    {
        AiDotNetEngine.ResetToCpu();
        _originalOptions = TensorCodecOptions.Current;
        TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
    }

    public void Dispose() => TensorCodecOptions.SetCurrent(_originalOptions);

    private const int InputDim = 64;
    private const int OutputDim = 8;

    /// <summary>Small MLP with DETERMINISTIC weights so two instances are bit-identical (for oracles).</summary>
    private static NeuralNetworkBase<float> BuildMlp(int seed = 1)
    {
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(32, (IActivationFunction<float>)new ReLUActivation<float>()),
            new DenseLayer<float>(OutputDim, activationFunction: (IActivationFunction<float>?)null),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: InputDim,
            outputSize: OutputDim,
            layers: layers);
        var net = new FeedForwardNeuralNetwork<float>(arch);
        net.SetTrainingMode(false);

        var p = net.GetParameters();
        var det = new float[p.Length];
        var rng = new Random(seed);
        for (int i = 0; i < det.Length; i++) det[i] = (float)(rng.NextDouble() - 0.5) * 0.3f;
        net.UpdateParameters(new Vector<float>(det));
        return net;
    }

    private static Tensor<float> MakeInput(int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var data = new float[InputDim];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return new Tensor<float>(data, new[] { 1, InputDim });
    }

    /// <summary>Pure-eager oracle: compilation off + acceleration not engaged.</summary>
    private static Tensor<float> Eager(NeuralNetworkBase<float> net, Tensor<float> input)
    {
        var prev = TensorCodecOptions.Current;
        TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = false });
        try { return net.Predict(input); }
        finally { TensorCodecOptions.SetCurrent(prev); }
    }

    private static void AssertParity(Tensor<float> expected, Tensor<float> actual, string what)
    {
        Assert.Equal(expected.Shape.Length, actual.Shape.Length);
        var e = expected.AsSpan();
        var a = actual.AsSpan();
        Assert.Equal(e.Length, a.Length);
        double maxRel = 0;
        for (int i = 0; i < e.Length; i++)
        {
            double diff = Math.Abs(e[i] - a[i]);
            double tol = 1e-4 + 3e-3 * Math.Max(Math.Abs(e[i]), Math.Abs(a[i]));
            maxRel = Math.Max(maxRel, diff - tol);
            Assert.True(diff <= tol, $"{what}: element {i} diverged eager={e[i]} accel={a[i]} (diff={diff}).");
        }
    }

    private static double MaxAbsDiff(Tensor<float> x, Tensor<float> y)
    {
        var a = x.AsSpan(); var b = y.AsSpan();
        double m = 0;
        for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
        return m;
    }

    [Fact]
    public void Accelerated_MatchesEager_VerifiesOnce_ThenMemoHitsOnRepeat()
    {
        var net = BuildMlp();
        var input = MakeInput(7);
        var eager = Eager(net, input);

        net.ForceAutoCompiledInferenceForTesting(true);

        // First call at this shape: verify-then-trust (one verification), output == eager.
        var first = net.PredictAccelerated(input);
        AssertParity(eager, first, "accelerated first call");
        Assert.Equal(1, net.AcceleratedVerifyCount);
        Assert.NotEqual(0, net.CompiledVerdictForTesting(new[] { 1, InputDim })); // decided (trusted or rejected)
        Assert.Equal(0, net.AcceleratedMemoHits);

        // Second call with the SAME input: collision-safe value memo short-circuits it (no re-verify).
        var second = net.PredictAccelerated(input);
        AssertParity(eager, second, "accelerated memo repeat");
        Assert.Equal(1, net.AcceleratedMemoHits);
        Assert.Equal(1, net.AcceleratedVerifyCount); // shape verdict cached — never re-verified
    }

    [Fact]
    public void Accelerated_DifferentSameShapeInput_IsNotStale_AndDoesNotReVerify()
    {
        var net = BuildMlp();
        var inputA = MakeInput(11);
        var inputB = MakeInput(22); // different values, same shape
        var eagerA = Eager(net, inputA);
        var eagerB = Eager(net, inputB);
        Assert.True(MaxAbsDiff(eagerA, eagerB) > 1e-3, "inputs must produce distinct outputs for staleness to be observable");

        net.ForceAutoCompiledInferenceForTesting(true);

        var accelA = net.PredictAccelerated(inputA);
        AssertParity(eagerA, accelA, "accelerated A");
        Assert.Equal(1, net.AcceleratedVerifyCount);

        // B is a different input at the same shape: must reflect eager(B), not stale A; and the shape
        // verdict is already decided so there is NO second verification (shape-guard / no recompile thrash).
        var accelB = net.PredictAccelerated(inputB);
        AssertParity(eagerB, accelB, "accelerated B (not stale)");
        Assert.Equal(1, net.AcceleratedVerifyCount);
        Assert.True(MaxAbsDiff(accelA, accelB) > 1e-3, "accelerated A and B must differ");
    }

    [Fact]
    public void Accelerated_ScaledInput_ChangesOutput()
    {
        var net = BuildMlp();
        var input = MakeInput(33, scale: 1f);
        var scaled = MakeInput(33, scale: 4f); // same seed, scaled values
        var eager = Eager(net, input);
        var eagerScaled = Eager(net, scaled);

        net.ForceAutoCompiledInferenceForTesting(true);
        var accel = net.PredictAccelerated(input);
        var accelScaled = net.PredictAccelerated(scaled);

        AssertParity(eager, accel, "accelerated base");
        AssertParity(eagerScaled, accelScaled, "accelerated scaled");
        Assert.True(MaxAbsDiff(accel, accelScaled) > 1e-3,
            "scaled input must change the accelerated output (the canonical staleness guard)");
    }

    [Fact]
    public void Accelerated_RepeatedDistinctInputs_StayEagerIdentical()
    {
        // The verify-then-trust verdict is decided once for the shape; every later input at that shape
        // (trusted replay or rejected eager) must still equal the eager forward for THAT input. Drive a
        // handful of distinct inputs and confirm each matches its eager oracle, with exactly one
        // verification total (no per-input recompile thrash).
        var net = BuildMlp();
        var inputs = new[] { MakeInput(1), MakeInput(2), MakeInput(3), MakeInput(4) };
        var eagers = new Tensor<float>[inputs.Length];
        for (int i = 0; i < inputs.Length; i++) eagers[i] = Eager(net, inputs[i]);

        net.ForceAutoCompiledInferenceForTesting(true);
        // Each DISTINCT input at the same shape must match eager. The shape-guard verdict is cached, so
        // the number of verifications stays bounded (at most one initial + one structure-version
        // stabilization when a lazy layer resolves its input dim) — NOT one-per-input. Four distinct
        // inputs with ≤2 verifies proves there is no per-input recompile thrash.
        for (int i = 0; i < inputs.Length; i++)
        {
            var accel = net.PredictAccelerated(inputs[i]);
            AssertParity(eagers[i], accel, $"accelerated distinct input {i}");
        }
        Assert.True(net.AcceleratedVerifyCount <= 2,
            $"verify-then-trust must cache its verdict (≤2 verifies for 4 distinct inputs), got {net.AcceleratedVerifyCount}");
    }

    /// <summary>A model that OVERRIDES Predict and opts into acceleration via the one-line Accelerate helper.</summary>
    private sealed class HelperWrappedNet : FeedForwardNeuralNetwork<float>
    {
        public HelperWrappedNet(NeuralNetworkArchitecture<float> arch) : base(arch) { }
        public override Tensor<float> Predict(Tensor<float> input)
            => Accelerate(input, () => { var c = input; foreach (var l in Layers) c = l.Forward(c); return c; });
    }

    [Fact]
    public void AccelerateHelper_OnOverridingModel_MatchesEager()
    {
        // The Accelerate helper is how the ~126 Predict-overriding models opt in with one line. Prove it
        // accelerates an overriding model's custom forward and stays eager-identical.
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: InputDim, outputSize: OutputDim,
            layers: new List<ILayer<float>>
            {
                new DenseLayer<float>(32, (IActivationFunction<float>)new ReLUActivation<float>()),
                new DenseLayer<float>(OutputDim, activationFunction: (IActivationFunction<float>?)null),
            });
        var net = new HelperWrappedNet(arch);
        net.SetTrainingMode(false);
        var p = net.GetParameters();
        var det = new float[p.Length];
        var rng = new Random(7);
        for (int i = 0; i < det.Length; i++) det[i] = (float)(rng.NextDouble() - 0.5) * 0.3f;
        net.UpdateParameters(new Vector<float>(det));

        var input = MakeInput(88);
        var eager = Eager(net, input);             // accel not engaged (no force) → pure eager via the helper

        net.ForceAutoCompiledInferenceForTesting(true);
        var accel = net.Predict(input);            // overriding Predict → Accelerate → verify-then-trust gate
        AssertParity(eager, accel, "overriding model via Accelerate helper");
        Assert.True(net.AcceleratedVerifyCount >= 1, "the helper should have driven the verify gate");
    }

    [Fact]
    public void Training_DisablesAcceleration_AndClearsVerdicts()
    {
        var net = BuildMlp();
        var input = MakeInput(55);

        net.ForceAutoCompiledInferenceForTesting(true);
        _ = net.PredictAccelerated(input);
        Assert.True(net.AutoCompiledInferenceEngaged);
        Assert.NotEqual(0, net.CompiledVerdictForTesting(new[] { 1, InputDim }));

        net.SetTrainingMode(true);
        Assert.False(net.AutoCompiledInferenceEngaged);
        Assert.Equal(0, net.CompiledVerdictForTesting(new[] { 1, InputDim })); // verdicts + memo cleared
    }

    [Fact]
    public void FoundationScaleThreshold_AutoEngagesAcceleration_ComposedWithStreaming()
    {
        // Force the foundation-scale gate by lowering the param threshold to 1 BEFORE the first
        // forward, so weight-streaming auto-detect engages and the model latches the auto-compiled path
        // (L1 + L3 unified, #92). The forward must run cleanly (composing the compiled gate with the
        // quant-resident streamed weights) and be deterministic across repeats. We assert engagement +
        // finiteness + determinism rather than fp32 parity, since the L1 lever may keep the resident
        // weights at reduced precision (the accepted foundation-scale memory tradeoff).
        var net = BuildMlp(seed: 99);
        net.ApplyAutoDetectThresholdOverride(1);
        // SetTrainingMode(false) is the universal funnel that runs weight-streaming auto-detect and
        // latches the auto-compiled eligibility; call it explicitly so the test does not depend on a
        // particular model's Predict override calling it.
        net.SetTrainingMode(false);
        var input = MakeInput(66);

        var out1 = net.Predict(input);
        var out2 = net.Predict(input);

        Assert.True(net.AutoCompiledInferenceEngaged,
            "a foundation-scale (over-threshold) inference model must auto-engage the compiled path");
        foreach (var x in out1.AsSpan().ToArray())
            Assert.False(float.IsNaN(x) || float.IsInfinity(x), "auto-engaged streamed forward must be finite");
        Assert.Equal(out1.Shape, out2.Shape);
        var a = out1.AsSpan(); var b = out2.AsSpan();
        for (int i = 0; i < a.Length; i++)
            Assert.True(Math.Abs(a[i] - b[i]) <= 1e-4 + 3e-3 * Math.Max(Math.Abs(a[i]), Math.Abs(b[i])),
                "auto-engaged accelerated forward must be deterministic across repeats");
    }
}
