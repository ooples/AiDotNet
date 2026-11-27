using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.DistributedTraining;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.DistributedTraining;

/// <summary>
/// The shared-memory float all-reduce sums in rank order (((r0 + r1) + r2) ...), bitwise, on every rank, for every
/// chunk of a buffer longer than a slot. Data-parallel runs rely on this to be reproducible and identical across
/// ranks; the reduce's one-pass first add (slot 0 + slot 1) must keep exactly those additions.
/// </summary>
[Collection("SequentialMultiProcessGpuDistributed")]
public sealed class SharedMemoryAllReduceExactnessTests
{
    [Theory]
    [InlineData(2, ReductionOperation.Sum)]
    [InlineData(3, ReductionOperation.Sum)]
    [InlineData(2, ReductionOperation.Average)]
    [InlineData(3, ReductionOperation.Average)]
    [InlineData(2, ReductionOperation.Max)]
    public void FloatAllReduce_IsTheRankOrderedReduction_Bitwise_OnEveryRankAndChunk(int world, ReductionOperation op)
    {
        // 4 KB slots: 1024 floats per chunk, so 2500 + 3 elements span three chunks with a ragged tail.
        const int length = 2503;
        string session = "aidn_exact_" + Guid.NewGuid().ToString("N");
        var backends = Enumerable.Range(0, world)
            .Select(r => new SharedMemoryCommunicationBackend<float>(r, world, session, 1L << 12, 30_000)).ToArray();
        var inputs = Enumerable.Range(0, world).Select(r =>
        {
            var rng = new Random(100 + r);
            return Enumerable.Range(0, length).Select(_ => (float)((rng.NextDouble() - 0.5) * Math.Pow(10, rng.Next(-6, 6)))).ToArray();
        }).ToArray();
        var expected = new float[length];
        for (int i = 0; i < length; i++)
        {
            float acc = inputs[0][i];
            for (int r = 1; r < world; r++)
                acc = op == ReductionOperation.Max ? Math.Max(acc, inputs[r][i]) : acc + inputs[r][i];
            if (op == ReductionOperation.Average) acc *= 1.0f / world;
            expected[i] = acc;
        }
        try
        {
            Task.WaitAll(backends.Select(b => Task.Run(() => b.Initialize())).ToArray());
            var results = new float[world][];
            Task.WaitAll(Enumerable.Range(0, world).Select(r => Task.Run(() =>
            {
                var data = new Vector<float>(inputs[r].ToArray());
                backends[r].AllReduce(data, op);
                results[r] = data.ToArray();
            })).ToArray());
            for (int r = 0; r < world; r++)
                Assert.Equal(expected, results[r]);
        }
        finally
        {
            foreach (var b in backends)
                try { b.Shutdown(); } catch (Exception) { /* a failed init leaves nothing to shut down */ }
        }
    }

    // The fused scaled sum (each rank scales into its own slot, the slots sum into the destination) is bitwise the
    // scale-then-AllReduce(Sum) it replaces, in place, on every rank and chunk.
    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    public void ScaledSumAllReduce_InPlace_IsBitwiseTheScaleThenSum(int world)
    {
        const int length = 2503;
        string session = "aidn_scaled_" + Guid.NewGuid().ToString("N");
        var backends = Enumerable.Range(0, world)
            .Select(r => new SharedMemoryCommunicationBackend<float>(r, world, session, 1L << 12, 30_000)).ToArray();
        var inputs = Enumerable.Range(0, world).Select(r =>
        {
            var rng = new Random(200 + r);
            return Enumerable.Range(0, length).Select(_ => (float)((rng.NextDouble() - 0.5) * Math.Pow(10, rng.Next(-6, 6)))).ToArray();
        }).ToArray();
        var scales = Enumerable.Range(0, world).Select(r => (float)(r + 1) / (world * (world + 1) / 2)).ToArray();
        try
        {
            Task.WaitAll(backends.Select(b => Task.Run(() => b.Initialize())).ToArray());
            var reference = new float[world][];
            Task.WaitAll(Enumerable.Range(0, world).Select(r => Task.Run(() =>
            {
                var data = new Vector<float>(inputs[r].Select(v => v * scales[r]).ToArray());
                backends[r].AllReduce(data, ReductionOperation.Sum);
                reference[r] = data.ToArray();
            })).ToArray());
            var fused = new float[world][];
            Task.WaitAll(Enumerable.Range(0, world).Select(r => Task.Run(() =>
            {
                var buffer = inputs[r].ToArray();
                Assert.True(backends[r].TryAllReduceScaledSum(buffer, scales[r], buffer));
                fused[r] = buffer;
            })).ToArray());
            for (int r = 0; r < world; r++)
            {
                Assert.Equal(reference[0], reference[r]);
                Assert.Equal(reference[r], fused[r]);
            }
        }
        finally
        {
            foreach (var b in backends)
                try { b.Shutdown(); } catch (Exception) { /* a failed init leaves nothing to shut down */ }
        }
    }

    // The slot exchange (each rank's unweighted gradients in its own slot, each rank reducing one partition in place,
    // every partition read from its owner's slot) gives every rank, bitwise, what scaling by the rank weight and an
    // all-reduce sum gives. A length that is not a multiple of the vector width exercises the scalar tail.
    [Theory]
    [InlineData(2, 2503)]
    [InlineData(3, 2503)]
    [InlineData(2, 7)]
    public void SlotPartitionReduce_IsBitwiseTheScaledAllReduce(int world, int length)
    {
        string session = "aidn_slots_" + Guid.NewGuid().ToString("N");
        // 16 KB slots: page multiples, as the region's registration needs.
        var backends = Enumerable.Range(0, world)
            .Select(r => new SharedMemoryCommunicationBackend<float>(r, world, session, 1L << 14, 30_000)).ToArray();
        var inputs = Enumerable.Range(0, world).Select(r =>
        {
            var rng = new Random(300 + r);
            return Enumerable.Range(0, length).Select(_ => (float)((rng.NextDouble() - 0.5) * Math.Pow(10, rng.Next(-6, 6)))).ToArray();
        }).ToArray();
        var weights = Enumerable.Range(0, world).Select(r => (float)(r + 1) / (world * (world + 1) / 2)).ToArray();
        try
        {
            Task.WaitAll(backends.Select(b => Task.Run(() => b.Initialize())).ToArray());
            var reference = new float[world][];
            Task.WaitAll(Enumerable.Range(0, world).Select(r => Task.Run(() =>
            {
                var data = new Vector<float>(inputs[r].Select(v => v * weights[r]).ToArray());
                backends[r].AllReduce(data, ReductionOperation.Sum);
                reference[r] = data.ToArray();
            })).ToArray());
            var viaSlots = new float[world][];
            Task.WaitAll(Enumerable.Range(0, world).Select(r => Task.Run(() =>
            {
                var b = backends[r];
                Assert.True(b.TryGetSlotRegion(length, out IntPtr start, out _, out int stride, out string? why), why);
                var gathered = b.AllGatherFloat(weights[r]);
                Assert.Equal(weights, gathered);
                System.Runtime.InteropServices.Marshal.Copy(inputs[r], 0, start + r * stride * sizeof(float), length);
                b.ReduceSlotPartitions(length, gathered);
                var result = new float[length];
                for (int q = 0; q < world; q++)
                {
                    var (offset, count) = b.SlotPartition(q, length);
                    System.Runtime.InteropServices.Marshal.Copy(start + (q * stride + offset) * sizeof(float), result, offset, count);
                }
                b.FinishSlotReads(length);
                viaSlots[r] = result;
            })).ToArray());
            for (int r = 0; r < world; r++) Assert.Equal(reference[r], viaSlots[r]);
            // Too long for a slot: refused with a reason, nothing touched.
            Assert.False(backends[0].TryGetSlotRegion((1 << 14) / sizeof(float) + 1, out _, out _, out _, out string? tooLong));
            Assert.Contains("exceed", tooLong);
        }
        finally
        {
            foreach (var b in backends)
                try { b.Shutdown(); } catch (Exception) { /* a failed init leaves nothing to shut down */ }
        }
    }
}
