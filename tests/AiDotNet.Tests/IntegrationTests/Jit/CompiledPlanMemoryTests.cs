using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Jit;

/// <summary>
/// Compiled-plan memory-residency regression tests (defect 3 of the AIsEval
/// compiled-mode findings): each plan pre-allocates per-step buffers (tens of MB
/// for conv nets at batch), and with many plans warm the residency degraded even
/// non-compiled models in the same process. The bound is the
/// <see cref="AiDotNet.NeuralNetworks.NeuralNetworkBase{T}.ReleaseCompiledPlans"/>
/// valve: releasing must actually return the buffers to the GC, and the model
/// must keep predicting correctly (recompiling on demand) afterwards.
///
/// Runs in the NonParallelIntegration collection: GC.GetTotalMemory measures
/// the WHOLE managed heap, so concurrent test allocations would shift both
/// the &gt;5 MB warmup gate and the post-release ceiling arbitrarily — and the
/// constructor mutates process-global state (ResetToCpu /
/// TensorCodecOptions.SetCurrent) that parallel classes would race.
/// </summary>
[Collection("NonParallelIntegration")]
public class CompiledPlanMemoryTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly TensorCodecOptions _originalOptions;

    public CompiledPlanMemoryTests(ITestOutputHelper o)
    {
        _out = o;
        AiDotNetEngine.ResetToCpu();
        _originalOptions = TensorCodecOptions.Current;
        TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
    }

    public void Dispose() => TensorCodecOptions.SetCurrent(_originalOptions);

    [Fact]
    public void ReleaseCompiledPlans_ReturnsPlanBuffersToGc_AndModelStillPredicts()
    {
        long Settled() { GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect(); return GC.GetTotalMemory(forceFullCollection: true); }

        // Build the conv net FIRST so its own weights/lazy-init memory is part of
        // the baseline — the deltas below then isolate plan-buffer residency.
        var net = (AiDotNet.NeuralNetworks.NeuralNetworkBase<float>)typeof(CompiledInferenceParityTests)
            .GetMethod("BuildAisEvalCnn", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!
            .Invoke(null, null)!;
        var input = MakeInput(new[] { 64, 1, 28, 28 }, 3);
        // Warm through a TRUE eager forward: the class fixture leaves
        // EnableCompilation = true, so a plain Predict here could compile and
        // allocate plan buffers BEFORE the baseline snapshot — silently
        // shrinking the warmed-residency delta this test exists to measure.
        Tensor<float> eagerWarm;
        var optsBeforeWarm = TensorCodecOptions.Current;
        TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = false });
        try
        {
            eagerWarm = net.Predict(input); // materialize weights + lazy shapes
        }
        finally
        {
            TensorCodecOptions.SetCurrent(optsBeforeWarm);
        }
        long baseline = Settled();

        Assert.True(net.CompileForward(input), "CompileForward failed for the CNN.");
        var compiled1 = net.PredictCompiled(input);
        long warmed = Settled();
        double warmedMb = (warmed - baseline) / 1024.0 / 1024.0;
        _out.WriteLine($"plan residency after warm: {warmedMb:F1} MB");

        // The conv plan at bs=64 holds multiple [64,16,28,28]-class buffers —
        // require that warming visibly cost real memory so the release assertion
        // below is actually exercising something (guards against the test silently
        // measuring nothing if compilation fell back to eager).
        Assert.True(warmedMb > 5.0, $"expected the warmed CNN plan to hold >5 MB of buffers, measured {warmedMb:F1} MB — did compilation fall back to eager?");

        net.ReleaseCompiledPlans();
        long released = Settled();
        double residualMb = (released - baseline) / 1024.0 / 1024.0;
        _out.WriteLine($"plan residency after ReleaseCompiledPlans: {residualMb:F1} MB");

        // The valve must return the dominant share of the plan buffers. Allow a
        // small residual (pool growth, trace bookkeeping) but fail if most of the
        // plan memory survives the release.
        Assert.True(residualMb < warmedMb * 0.35 + 1.0,
            $"ReleaseCompiledPlans left {residualMb:F1} MB of {warmedMb:F1} MB resident — the release valve is not releasing.");

        // And the model must still work: next compiled call re-traces transparently.
        Assert.True(net.CompileForward(input), "re-CompileForward after release failed.");
        var compiled2 = net.PredictCompiled(input);
        AssertClose(compiled1, compiled2);
        AssertClose(eagerWarm, compiled2);
    }

    private static void AssertClose(Tensor<float> a, Tensor<float> b)
    {
        Assert.Equal(a.Length, b.Length);
        var x = a.AsSpan(); var y = b.AsSpan();
        double maxAbs = 0, maxMag = 1e-6;
        for (int i = 0; i < x.Length; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(x[i] - y[i]));
            maxMag = Math.Max(maxMag, Math.Abs(x[i]));
        }
        Assert.True(maxAbs / maxMag < 1e-4, $"outputs diverge: rel={maxAbs / maxMag:E2}");
    }

    private static Tensor<float> MakeInput(int[] shape, int seed)
    {
        var rng = new Random(seed);
        int len = 1; foreach (var d in shape) len *= d;
        var data = new float[len];
        for (int i = 0; i < len; i++) data[i] = (float)rng.NextDouble();
        return new Tensor<float>(data, shape);
    }
}
