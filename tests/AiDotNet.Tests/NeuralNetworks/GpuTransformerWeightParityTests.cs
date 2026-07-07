#if !NETFRAMEWORK
#nullable disable
using System;
using AiDotNet;
using AiDotNet.Enums;
using AiDotNet.Data.Loaders;
using AiDotNet.FitnessCalculators;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.NeuralNetworks;

/// <summary>
/// Regression gate for the GPU weight-coherence fix (facade side).
///
/// The optimizer updates each weight tensor's CPU backing IN PLACE. On a DirectGpu engine the
/// forward reads a cached DEVICE copy; unless every device weight cache is invalidated after the
/// update, the next forward serves STALE weights and the GPU model trains against frozen weights
/// (held-out accuracy pinned at chance while the CPU engine learns). The fix calls
/// <c>DirectGpuTensorEngine.InvalidateResidentWeightBuffer</c> per updated parameter in
/// <c>GradientBasedOptimizerBase.UpdateSolution</c> and
/// <c>NeuralNetworkBase.InvalidateWeightCachesAfterSuccessfulWeightUpdate</c>.
///
/// This test trains the SAME transformer + seed for exactly 2 optimizer steps on the GPU engine
/// and on the CPU engine, and asserts the resulting weights match to FP tolerance. BEFORE the fix
/// the GPU trajectory diverged from CPU at step 2 (the forward froze on step-0 weights); after it,
/// the per-step trajectory matches to FP precision. Skips when no DirectGpu backend is available.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public sealed class GpuTransformerWeightParityTests
{
    private readonly ITestOutputHelper _out;
    public GpuTransformerWeightParityTests(ITestOutputHelper o) => _out = o;

    private const int V = 24, Ctx = 8;

    private static (Tensor<float> x, Tensor<float> y) MakeData(int n, int seed)
    {
        var rng = new Random(seed);
        var x = new Tensor<float>(new[] { n, Ctx });
        var y = new Tensor<float>(new[] { n, V });
        for (int i = 0; i < n; i++)
        {
            var cnt = new int[V];
            for (int c = 0; c < Ctx; c++) { int t = rng.Next(V); x[i, c] = t; cnt[t]++; }
            int mf = 0; for (int v = 1; v < V; v++) if (cnt[v] > cnt[mf]) mf = v;
            y[i, mf] = 1f;
        }
        return (x, y);
    }

    private static float[] Train2Steps()
    {
        var (tx, ty) = MakeData(64, 1);
        NeuralNetworkArchitecture<float>.DefaultRandomSeedOverride = 42;
        var arch = new TransformerArchitecture<float>(
            InputType.TwoDimensional, NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 2, numDecoderLayers: 0, numHeads: 4, modelDimension: 48, feedForwardDimension: 96,
            inputSize: Ctx, outputSize: V, maxSequenceLength: Ctx, vocabularySize: V, randomSeed: 42, dropoutRate: 0.0);
        var opt = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.01, MaxIterations = 2, UseAdaptiveLearningRate = false,
                UseEarlyStopping = false, Tolerance = 0.0, BatchSize = 64,
                FitnessCalculator = new MeanSquaredErrorFitnessCalculator<float, Tensor<float>, Tensor<float>>()
            });
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>(), optimizer: opt);
        var result = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model).ConfigureOptimizer(opt)
            .ConfigureDataLoader(new InMemoryDataLoader<float, Tensor<float>, Tensor<float>>(tx, ty))
            .BuildAsync().GetAwaiter().GetResult();
        var p = result.GetParameters();
        var arr = new float[p.Length];
        for (int i = 0; i < p.Length; i++) arr[i] = p[i];
        return arr;
    }

    [Fact]
    public void GpuTrainedWeights_MatchCpu_After2Steps()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); if (!gpu.IsGpuAvailable) return; }
        catch { return; } // no DirectGpu backend on this host — skip

        var savedEngine = AiDotNetEngine.Current;
        float[] gpuW, cpuW;
        try
        {
            AiDotNetEngine.Current = gpu;
            gpuW = Train2Steps();
            AiDotNetEngine.Current = new CpuEngine();
            cpuW = Train2Steps();
        }
        finally
        {
            AiDotNetEngine.Current = savedEngine;
            gpu.Dispose();
        }

        Assert.Equal(cpuW.Length, gpuW.Length);
        double maxAbs = 0;
        for (int i = 0; i < cpuW.Length; i++)
        {
            double d = Math.Abs((double)gpuW[i] - cpuW[i]);
            if (d > maxAbs) maxAbs = d;
        }
        _out.WriteLine($"2-step GPU-vs-CPU weight max_abs_diff = {maxAbs:E4} over {cpuW.Length} params");
        // Per-step trajectory matches to FP precision when the resident weight buffers are invalidated
        // after each in-place update. Before the fix this was ~1e-1 (GPU forward frozen on step-0 weights).
        Assert.True(maxAbs < 1e-3,
            $"GPU-trained weights diverged from CPU after 2 steps (max_abs_diff={maxAbs:E4}); " +
            $"resident GPU weight buffers are not being invalidated after the in-place optimizer update.");
    }
}
#endif
