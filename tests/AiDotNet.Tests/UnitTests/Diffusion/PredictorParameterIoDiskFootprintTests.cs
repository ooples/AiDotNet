using System;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Enums;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// #2447: the on-disk footprint of parameter IO on a predictor whose weights stream.
/// </summary>
/// <remarks>
/// <para>The nightly's FLUX GetParameterChunks/SetParameterChunks round-trip wrote ~76 GB to /tmp for a
/// 48 GB weight set before the disk ran out (and on 2026-11-24 took the runner down with it). The pool's
/// backing file is append-only and every parameter walk wrote each weight back as dirty, so the
/// round-trip wrote about 5.9 times the model. Parameter IO now maps each weight once, writably, into
/// the registry's parameter file and drops the pool copy, so the model reaches the disk exactly once.</para>
/// <para>These are the same round-trip shapes at a reduced width with streaming forced on and a
/// resident cap of a quarter of the model, so every walk has to page. The bound is the model's own
/// byte size plus slice alignment: any second copy of any weight breaks it.</para>
/// </remarks>
[Collection("WeightStreaming-Singleton")]
public sealed class PredictorParameterIoDiskFootprintTests : DiffusionUnitTestBase
{
    [Fact]
    public void ChunkRoundTrip_OfAStreamedFluxPredictor_WritesTheModelToDiskOnce()
    {
        using var scope = ForceStreaming<float>();
        var predictor = new FluxDoubleStreamPredictor<float>(
            inputChannels: 4, hiddenSize: 64, numJointLayers: 2, numSingleLayers: 2,
            numHeads: 4, contextDim: 32, seed: 1);
        long modelBytes = predictor.ParameterCount * sizeof(float);
        scope.CapResidentBytes(modelBytes / 4);

        AssertDetachedChunksRestore(predictor.GetParameterChunks, predictor.SetParameterChunks);

        AssertWrittenOnce(modelBytes, TensorCount(predictor.GetParameterChunks()));
    }

    [Fact]
    public void ChunkRoundTrip_OfAStreamedMMDiTXPredictor_WritesTheModelToDiskOnce()
    {
        using var scope = ForceStreaming<float>();
        var predictor = new MMDiTXNoisePredictor<float>(
            MMDiTXVariant.Medium, inputChannels: 4, patchSize: 2, contextDim: 32, seed: 1,
            hiddenSizeOverride: 64, numLayersOverride: 3, numHeadsOverride: 4);
        long modelBytes = predictor.ParameterCount * sizeof(float);
        scope.CapResidentBytes(modelBytes / 4);

        AssertDetachedChunksRestore(predictor.GetParameterChunks, predictor.SetParameterChunks);

        AssertWrittenOnce(modelBytes, TensorCount(predictor.GetParameterChunks()));
    }

    [Fact]
    public void FlatRoundTrip_OfAStreamedSiTPredictor_WritesTheModelToDiskOnce_AndRestoresEveryValue()
    {
        using var scope = ForceStreaming<double>();
        var predictor = new SiTPredictor<double>(
            inputChannels: 4, hiddenSize: 64, numLayers: 3, numHeads: 4, seed: 1);
        long modelBytes = predictor.ParameterCount * sizeof(double);
        scope.CapResidentBytes(modelBytes / 4);

        var parameters = predictor.GetParameters();
        var written = new Vector<double>(parameters.Length);
        for (int i = 0; i < written.Length; i++) written[i] = (i % 997) * 0.001 - 0.5;
        predictor.SetParameters(written);
        var read = predictor.GetParameters();

        Assert.Equal(written.Length, read.Length);
        for (int i = 0; i < read.Length; i++)
        {
            if (read[i] != written[i]) Assert.Equal(written[i], read[i]);
        }
        AssertWrittenOnce(modelBytes, TensorCount(predictor.GetParameterChunks()));
    }

    /// <summary>
    /// The MMDiT-X-after-FLUX shape from the nightly: a streamed predictor that is dropped without
    /// Dispose leaves its entries in the process-wide registry. Counting those kept the next
    /// foundation-scale predictor from streaming at all, so it materialized fully resident and OOMed.
    /// </summary>
    [Fact]
    public void APredictorDroppedWithRegisteredWeights_DoesNotStopTheNextOneFromStreaming()
    {
        using var scope = ForceStreaming<float>();
        scope.CapResidentBytes(16 * 1024 * 1024);
        RunAForwardAndDropThePredictor();
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var next = new MMDiTXNoisePredictor<float>(
            MMDiTXVariant.Medium, inputChannels: 4, patchSize: 2, contextDim: 32, seed: 2,
            hiddenSizeOverride: 64, numLayersOverride: 2, numHeadsOverride: 4);
        var streamed = next.GetParameterChunks().Count(chunk => chunk.IsWritableMmapAliased);

        Assert.True(streamed > 0,
            "The next predictor's weights stayed on the heap: streaming never engaged because the dropped " +
            "predictor's registry entries were still counted as a live session.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void RunAForwardAndDropThePredictor()
    {
        var dropped = new MMDiTNoisePredictor<float>(
            inputChannels: 4, hiddenSize: 32, numJointLayers: 1, numSingleLayers: 0,
            numHeads: 4, patchSize: 2, contextDim: 16, seed: 42);
        dropped.PredictNoise(new Tensor<float>(new[] { 1, 4, 8, 8 }), timestep: 25,
            new Tensor<float>(new[] { 1, 1, 16 }));
        Assert.True(WeightRegistry.GetStreamingReport().RegisteredEntryCount > 0,
            "Precondition: a streamed forward leaves its weights registered.");
    }

    /// <summary>
    /// Feeds the setter detached chunks holding values the model does not have, then reads them back. Writing the
    /// values into the live tensors and handing those same tensors to the setter would let a setter that does nothing
    /// pass; these reduced-size models can afford the copy the foundation-scale helper avoids.
    /// </summary>
    private static void AssertDetachedChunksRestore(
        Func<System.Collections.Generic.IEnumerable<Tensor<float>>> getChunks,
        Action<System.Collections.Generic.IEnumerable<Tensor<float>>> setChunks)
    {
        static float Expected(long globalIndex) => (float)((globalIndex % 991) * 0.001 - 0.25);

        long gi = 0;
        var detached = getChunks().Select(chunk =>
        {
            var values = new float[chunk.Length];
            for (int i = 0; i < values.Length; i++) values[i] = Expected(gi + i);
            gi += values.Length;
            return new Tensor<float>(chunk.Shape.ToArray(), new Vector<float>(values));
        }).ToArray();
        Assert.True(gi > 0, "Model exposed no parameter chunks.");
        Assert.Contains(getChunks().SelectMany(chunk => chunk.ToArray()).Select((v, i) => v != Expected(i)), changed => changed);

        setChunks(detached);

        long index = 0;
        foreach (var chunk in getChunks())
        {
            var values = chunk.ToArray();
            for (int i = 0; i < values.Length; i++, index++)
            {
                if (values[i] != Expected(index)) Assert.Equal(Expected(index), values[i]);
            }
        }
        Assert.Equal(gi, index);
    }

    private static int TensorCount(System.Collections.Generic.IEnumerable<Tensor<float>> chunks) => chunks.Count();

    private static int TensorCount(System.Collections.Generic.IEnumerable<Tensor<double>> chunks) => chunks.Count();

    private static void AssertWrittenOnce(long modelBytes, int tensorCount)
    {
        long poolBytes = WeightRegistry.GetStreamingReport().DiskWriteBytes;
        long parameterFileBytes = ParameterFileUsedBytes();
        long written = poolBytes + parameterFileBytes;
        // Each mapped slice is 64-byte aligned.
        long bound = modelBytes + 64L * tensorCount;

        Assert.True(written <= bound,
            $"Parameter IO wrote {written:N0} bytes to disk ({poolBytes:N0} to the pool's backing file, " +
            $"{parameterFileBytes:N0} to the parameter file) for a {modelBytes:N0}-byte model; at most " +
            $"{bound:N0} (one copy) is expected. Before #2447 this round-trip wrote ~5.9x the model.");
        Assert.True(parameterFileBytes > 0,
            "Parameter IO never mapped a weight, so this test is not measuring the streamed path.");
    }

    // The registry's parameter file is internal to AiDotNet.Tensors and has no public size surface.
    private static long ParameterFileUsedBytes()
    {
        var field = typeof(WeightRegistry).GetField("_paramIoStore", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException(
                "WeightRegistry._paramIoStore is gone; update this test to the package's parameter-file surface.");
        object? store = field.GetValue(null);
        if (store is null) return 0;
        return (long)(store.GetType().GetProperty("UsedBytes")?.GetValue(store)
            ?? throw new InvalidOperationException("The parameter file no longer reports UsedBytes."));
    }

    private static StreamingScope<T> ForceStreaming<T>()
    {
        WeightRegistry.Reset();
        NoisePredictorBase<T>.StreamingThresholdOverride = 1;
        return new StreamingScope<T>();
    }

    private sealed class StreamingScope<T> : IDisposable
    {
        public void CapResidentBytes(long bytes) =>
            NoisePredictorBase<T>.StreamingResidentCapOverride = Math.Max(bytes, 64 * 1024);

        public void Dispose()
        {
            NoisePredictorBase<T>.StreamingThresholdOverride = null;
            NoisePredictorBase<T>.StreamingResidentCapOverride = null;
            WeightRegistry.Reset();
        }
    }
}
