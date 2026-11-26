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

        AssertParameterChunksRoundTrip(predictor.GetParameterChunks, predictor.SetParameterChunks);

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

        AssertParameterChunksRoundTrip(predictor.GetParameterChunks, predictor.SetParameterChunks);

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
    /// Parameter IO unregisters each weight once it is mapped, so a predictor half-way through a parameter
    /// walk has no registered entry. A second predictor engaging on another thread took that for an idle
    /// registry and reset it, disposing the pool and the parameter file under the first.
    /// </summary>
    [Fact]
    public void ASecondPredictorEngaging_DoesNotResetTheRegistryUnderOneInTheMiddleOfAParameterWalk()
    {
        using var scope = ForceStreaming<float>();
        var first = NewMMDiTX(seed: 1);
        long modelBytes = first.ParameterCount * sizeof(float);
        scope.CapResidentBytes(modelBytes / 4);

        static float Expected(long index) => (float)((index % 997) * 0.001 - 0.5);

        // The start of a walk: the weights seen so far are mapped and hold the ramp, the rest are untouched.
        const int WalkedFirst = 4;
        int tensors;
        long written = 0;
        using (var walk = first.GetParameterChunks().GetEnumerator())
        {
            for (int t = 0; t < WalkedFirst; t++)
            {
                Assert.True(walk.MoveNext());
                var span = walk.Current.Data.Span;
                for (int i = 0; i < span.Length; i++) span[i] = Expected(written + i);
                written += span.Length;
            }

            var poolBefore = WeightRegistry.CurrentStreamingPool;
            long fileBytesBefore = ParameterFileUsedBytes();
            Assert.NotNull(poolBefore);
            Assert.True(fileBytesBefore > 0, "Precondition: the first predictor's weights are in the parameter file.");
            Assert.Equal(0, WeightRegistry.GetStreamingReport().RegisteredEntryCount);

            // The second predictor walks its parameters on another thread while the first is mid-walk.
            MMDiTXNoisePredictor<float>? second = null;
            int secondStreamed = -1;
            Exception? failure = null;
            var thread = new System.Threading.Thread(() =>
            {
                try
                {
                    second = NewMMDiTX(seed: 2);
                    secondStreamed = second.GetParameterChunks().Count(chunk => chunk.IsWritableMmapAliased);
                }
                catch (Exception ex)
                {
                    failure = ex;
                }
            });
            thread.Start();
            thread.Join();
            Assert.Null(failure);

            Assert.Same(poolBefore, WeightRegistry.CurrentStreamingPool);
            Assert.True(ParameterFileUsedBytes() >= fileBytesBefore,
                "The parameter file the first predictor's weights live in was released under it.");
            Assert.Equal(0, secondStreamed);   // it ran resident: the session belongs to the first
            GC.KeepAlive(second);

            // The first finishes its walk against the registry it started on.
            long index = written;
            tensors = WalkedFirst;
            while (walk.MoveNext())
            {
                var span = walk.Current.Data.Span;
                for (int i = 0; i < span.Length; i++) span[i] = Expected(index + i);
                index += span.Length;
                tensors++;
            }
            Assert.True(tensors > WalkedFirst, "The walk was already over when the second predictor engaged.");
        }

        long verified = 0;
        foreach (var chunk in first.GetParameterChunks())
        {
            var span = chunk.Data.Span;
            for (int i = 0; i < span.Length; i++)
            {
                if (span[i] != Expected(verified + i)) Assert.Equal(Expected(verified + i), span[i]);
            }
            verified += span.Length;
        }
        Assert.Equal(first.ParameterCount, verified);
        AssertWrittenOnce(modelBytes, tensors);

        // Disposing the owner gives the session up: the next predictor streams.
        first.Dispose();
        var third = NewMMDiTX(seed: 3);
        Assert.True(third.GetParameterChunks().Count(chunk => chunk.IsWritableMmapAliased) > 0,
            "The session was not released when its owner was disposed.");
    }

    /// <summary>
    /// The same race without choreography: predictors created, round-tripped and disposed on several
    /// threads at once. Whichever holds the session streams and the others run resident; every one of them
    /// must read back exactly what it wrote.
    /// </summary>
    [Fact]
    public void PredictorsRoundTrippingConcurrently_EachReadBackWhatTheyWrote()
    {
        using var scope = ForceStreaming<float>();
        scope.CapResidentBytes(256 * 1024);
        const int Threads = 4, Rounds = 6;
        var failures = new System.Collections.Concurrent.ConcurrentQueue<Exception>();
        int streamedRounds = 0;
        using var start = new System.Threading.Barrier(Threads);

        var threads = Enumerable.Range(0, Threads).Select(worker => new System.Threading.Thread(() =>
        {
            try
            {
                start.SignalAndWait();
                for (int round = 0; round < Rounds; round++)
                {
                    float bias = worker * 10 + round;
                    using var predictor = NewMMDiTX(seed: worker * 100 + round);
                    long index = 0;
                    bool streamed = false;
                    foreach (var chunk in predictor.GetParameterChunks())
                    {
                        streamed |= chunk.IsWritableMmapAliased;
                        var span = chunk.Data.Span;
                        for (int i = 0; i < span.Length; i++) span[i] = bias + ((index + i) % 997) * 0.001f;
                        index += span.Length;
                    }
                    if (streamed) System.Threading.Interlocked.Increment(ref streamedRounds);

                    predictor.SetParameterChunks(predictor.GetParameterChunks());

                    index = 0;
                    foreach (var chunk in predictor.GetParameterChunks())
                    {
                        var span = chunk.Data.Span;
                        for (int i = 0; i < span.Length; i++)
                        {
                            float expected = bias + ((index + i) % 997) * 0.001f;
                            if (span[i] != expected)
                                throw new Xunit.Sdk.XunitException(
                                    $"Worker {worker} round {round}: parameter {index + i} read {span[i]}, wrote {expected}.");
                        }
                        index += span.Length;
                    }
                    if (index != predictor.ParameterCount)
                        throw new Xunit.Sdk.XunitException(
                            $"Worker {worker} round {round}: read {index} of {predictor.ParameterCount} parameters.");
                }
            }
            catch (Exception ex)
            {
                failures.Enqueue(ex);
            }
        })).ToArray();
        foreach (var thread in threads) thread.Start();
        foreach (var thread in threads) thread.Join();

        Assert.True(failures.IsEmpty, string.Join(Environment.NewLine, failures.Select(f => f.ToString())));
        Assert.True(streamedRounds > 0, "No round streamed, so the registry was never contended.");
    }

    /// <summary>
    /// A write through the sparse parameter mapping on a volume with no blocks left is SIGBUS on Linux, not
    /// an exception. The walk is refused up front, with the path and both sizes, and nothing is mapped.
    /// </summary>
    [Fact]
    public void AParameterWalkTheDiskCannotHold_ThrowsBeforeMappingAnything()
    {
        using var scope = ForceStreaming<float>();
        var predictor = NewMMDiTX(seed: 1);
        long modelBytes = predictor.ParameterCount * sizeof(float);
        scope.CapResidentBytes(modelBytes / 4);
        string? probed = null;
        NoisePredictorBase<float>.AvailableDiskBytesOverride = directory =>
        {
            probed = directory;
            return modelBytes / 2;
        };

        var refused = Assert.Throws<System.IO.IOException>(() => predictor.GetParameterChunks().Count());

        Assert.NotNull(probed);
        Assert.Equal(NoisePredictorBase<float>.ParameterFileDirectory, probed);
        Assert.Contains(probed, refused.Message);
        Assert.Contains(modelBytes.ToString("N0"), refused.Message);
        Assert.Contains((modelBytes / 2).ToString("N0"), refused.Message);
        Assert.Equal(0, ParameterFileUsedBytes());
        // The session the refused walk engaged is given back: nothing was mapped into it.
        Assert.Equal(0, WeightStreamingSessions.LiveOwnerCount);
        Assert.Throws<System.IO.IOException>(() => predictor.GetParameters());
        Assert.Throws<System.IO.IOException>(() => predictor.SetParameterChunks(Array.Empty<Tensor<float>>()));

        // Room for the model is enough, and once every weight is mapped the volume is not asked again.
        NoisePredictorBase<float>.AvailableDiskBytesOverride = _ => modelBytes;
        AssertParameterChunksRoundTrip(predictor.GetParameterChunks, predictor.SetParameterChunks);
        NoisePredictorBase<float>.AvailableDiskBytesOverride = _ => 0;
        Assert.Equal(predictor.ParameterCount, predictor.GetParameterChunks().Sum(chunk => (long)chunk.Length));
        AssertWrittenOnce(modelBytes, TensorCount(predictor.GetParameterChunks()));
    }

    private static MMDiTXNoisePredictor<float> NewMMDiTX(int seed) => new(
        MMDiTXVariant.Medium, inputChannels: 4, patchSize: 2, contextDim: 32, seed: seed,
        hiddenSizeOverride: 64, numLayersOverride: 2, numHeadsOverride: 4);

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
        // The registry is reset by hand here, so the sessions over it are forgotten with it.
        WeightStreamingSessions.Clear();
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
            NoisePredictorBase<T>.AvailableDiskBytesOverride = null;
            WeightStreamingSessions.Clear();
            WeightRegistry.Reset();
        }
    }
}
