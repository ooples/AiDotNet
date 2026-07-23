using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>Serializes the inference-arena tests — they flip the process-global
/// <see cref="InferenceArenaSettings.Enabled"/> and the alloc test reads a process-wide GC counter.</summary>
[CollectionDefinition("InferenceArena", DisableParallelization = true)]
public sealed class InferenceArenaCollection { }

/// <summary>
/// Foundation tests for the inference forward-caching allocator (#1661 / Tensors #661):
/// the <see cref="NeuralNetworkBase{T}.Predict"/> arena funnel + <c>PredictCore</c> hook.
///
/// <para>These mirror Tensors' <c>InferenceArenaForwardTests</c> at the consumer layer:</para>
/// <list type="bullet">
///   <item><b>Bit-identity</b> — a migrated model's Predict is bit-exact with the arena ON vs OFF,
///   across repeated calls (the arena hands back recycled <c>RentUninitialized</c> buffers holding
///   the prior forward's bytes, so this catches any op relying on zero-initialised scratch and any
///   missing output detach).</item>
///   <item><b>Detach</b> — the returned tensor is unaffected by a subsequent Predict (no escaping
///   arena buffer).</item>
///   <item><b>Alloc collapse</b> — repeated Predict under the arena allocates far less than the
///   non-arena path (net5.0+; the GC API is absent on net471).</item>
///   <item><b>Bypass</b> — a model still on <c>override Predict</c> (not yet migrated) is unaffected
///   by the flag, proving the migration is incremental.</item>
/// </list>
///
/// <para>The class flips the process-global <see cref="InferenceArenaSettings.Enabled"/>, so it runs
/// in a non-parallel collection and always restores the flag in <c>finally</c>.</para>
/// </summary>
[Collection("InferenceArena")]
public class InferenceArenaPredictTests
{
    private readonly ITestOutputHelper _output;

    public InferenceArenaPredictTests(ITestOutputHelper output) => _output = output;

    // ── model builders ──────────────────────────────────────────────────────────

    // ConvolutionalNeuralNetwork<double>: double dtype intentionally skips the float-only fused
    // conv-stem fast path, so the forward runs the eager Layers path that Rents intermediates —
    // i.e. exactly the allocation the arena recycles. It is a MIGRATED model (override PredictCore),
    // so it routes through the base Predict funnel.
    private static ConvolutionalNeuralNetwork<double> BuildEagerCnn()
    {
        // Sized so the per-forward intermediate tensors (e.g. [2,32,32,32] ≈ 0.5 MB) dominate the
        // small per-Create arena bookkeeping + the detach copy — only then does the arena's
        // cross-call buffer reuse (via the persistent pool) show a net allocation win (#661).
        var layers = new List<ILayer<double>>
        {
            new ConvolutionalLayer<double>(outputDepth: 32, kernelSize: 3, stride: 1, padding: 1,
                                           activationFunction: new ReLUActivation<double>()),
            new MaxPoolingLayer<double>(poolSize: 2, stride: 2),
            new ConvolutionalLayer<double>(outputDepth: 64, kernelSize: 3, stride: 1, padding: 1,
                                           activationFunction: new ReLUActivation<double>()),
            new FlattenLayer<double>(),
            new DenseLayer<double>(10, activationFunction: (IActivationFunction<double>?)null),
        };
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 32, inputWidth: 32, inputDepth: 1,
            outputSize: 10,
            layers: layers);
        var net = new ConvolutionalNeuralNetwork<double>(arch);
        net.SetTrainingMode(false);
        return net;
    }

    // NeuralNetwork<double> overrides Predict directly (NOT migrated) → it bypasses the arena funnel.
    private static NeuralNetwork<double> BuildUnmigratedDenseNet()
    {
        var layers = new List<ILayer<double>>
        {
            new DenseLayer<double>(8, activationFunction: new ReLUActivation<double>()),
            new DenseLayer<double>(3, activationFunction: (IActivationFunction<double>?)null),
        };
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 5,
            outputSize: 3,
            layers: layers);
        var net = new NeuralNetwork<double>(arch);
        net.SetTrainingMode(false);
        return net;
    }

    private static Tensor<double> Image(double phase)
    {
        var t = new Tensor<double>(new[] { 2, 1, 32, 32 });
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = Math.Sin(i * 0.17 + phase);
        return t;
    }

    private static Tensor<double> Vec(double phase)
    {
        var t = new Tensor<double>(new[] { 1, 5 });
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = Math.Cos(i * 0.31 + phase);
        return t;
    }

    private static T WithArena<T>(bool enabled, Func<T> body)
    {
        bool prev = InferenceArenaSettings.Enabled;
        InferenceArenaSettings.Enabled = enabled;
        try { return body(); }
        finally { InferenceArenaSettings.Enabled = prev; }
    }

    // ── tests ───────────────────────────────────────────────────────────────────

    [Fact]
    public void MigratedModel_Predict_IsBitIdentical_ArenaOnVsOff_AcrossRepeats()
    {
        var model = BuildEagerCnn();
        var x = Image(0.0);

        double[] baseline = WithArena(false, () => model.Predict(x).ToArray());

        // 5 repeats under the arena: each Reset recycles the prior forward's buffers, so a stale-
        // scratch dependency or a missing detach would diverge on repeat 2+.
        WithArena(true, () =>
        {
            for (int r = 0; r < 5; r++)
            {
                double[] got = model.Predict(x).ToArray();
                Assert.Equal(baseline.Length, got.Length);
                for (int i = 0; i < baseline.Length; i++)
                    Assert.Equal(baseline[i], got[i]); // bit-exact (double, no tolerance)
            }
            return 0;
        });
    }

    [Fact]
    public void MigratedModel_Output_IsDetached_AndSurvivesNextPredict()
    {
        var model = BuildEagerCnn();

        WithArena(true, () =>
        {
            var firstOutput = model.Predict(Image(0.0));
            double[] snapshot = firstOutput.ToArray();

            // A different input drives another full forward; the arena recycles its buffers on the
            // next Create/Reset. If the first output still aliased an arena buffer it would change.
            model.Predict(Image(1.5));

            double[] after = firstOutput.ToArray();
            Assert.Equal(snapshot, after);
            return 0;
        });
    }

    [Fact]
    public void UnmigratedModel_Predict_IsUnaffectedByFlag()
    {
        var model = BuildUnmigratedDenseNet();
        var x = Vec(0.0);

        double[] off = WithArena(false, () => model.Predict(x).ToArray());
        double[] on = WithArena(true, () => model.Predict(x).ToArray());

        Assert.Equal(off.Length, on.Length);
        for (int i = 0; i < off.Length; i++)
            Assert.Equal(off[i], on[i]); // bit-exact — the override bypasses the arena entirely
    }

#if NET5_0_OR_GREATER
    [Fact]
    public void MigratedModel_Arena_ReducesPerForwardAllocation()
    {
        var model = BuildEagerCnn();
        var x = Image(0.0);
        const int reps = 40;

        // Warm both paths fully (first arena forward allocates; subsequent ones recycle).
        WithArena(false, () => { model.Predict(x); return 0; });
        long noArena = WithArena(false, () => MeasurePerForward(() => model.Predict(x), reps));

        WithArena(true, () => { model.Predict(x); model.Predict(x); return 0; });
        long arena = WithArena(true, () => MeasurePerForward(() => model.Predict(x), reps));

        // #661 measured ~95% reduction at steady state; a generous 0.75 bound is well clear of that
        // while tolerating the process-wide GC counter picking up some concurrent-test noise.
        double reduction = noArena == 0 ? 0 : 100.0 * (noArena - arena) / noArena;
        _output.WriteLine($"per-forward alloc: noArena={noArena:N0} B  arena={arena:N0} B  reduction={reduction:F1}%");
        Assert.True(arena < noArena * 3 / 4,
            $"arena per-forward alloc ({arena} B) should be well under the non-arena path ({noArena} B).");
    }

    private static long MeasurePerForward(Action forward, int reps)
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        long before = GC.GetTotalAllocatedBytes(precise: true);
        for (int i = 0; i < reps; i++) forward();
        long after = GC.GetTotalAllocatedBytes(precise: true);
        return (after - before) / reps;
    }
#endif
}
