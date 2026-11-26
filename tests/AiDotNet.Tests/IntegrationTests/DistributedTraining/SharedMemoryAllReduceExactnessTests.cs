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
}
