using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.DistributedTraining;
using AiDotNet.DistributedTraining.Layers;
using AiDotNet.Inference.PagedAttention;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.DistributedTraining;

/// <summary>
/// Proves tensor-parallel PAGED attention is transparent: running the paged prefill+decode over a sharded
/// per-rank KV cache (world size 2 and 4) produces the SAME per-step output as the un-sharded (world size 1)
/// paged attention, when seeded from the same full Q/K/V/O weights. Shard-count invariance == correctness
/// (no N-GPU box needed). Ranks run concurrently so the output-projection all-reduce actually rendezvous.
/// </summary>
public sealed class TensorParallelPagedEquivalenceTests
{
    private const int EmbedDim = 16;
    private const int NumHeads = 4;   // divisible by 2 and 4
    private const int HeadDim = EmbedDim / NumHeads;
    private const int Prefill = 4;
    private const int Decode = 3;
    private const int Total = Prefill + Decode;
    private const double Tol = 1e-9;

    [Theory(Timeout = 120000)]
    [InlineData(2)]
    [InlineData(4)]
    public async Task TensorParallelPagedAttention_ShardedDecode_MatchesUnsharded(int worldSize)
    {
        await Task.Yield();

        var rng = RandomHelper.CreateSeededRandom(20260718);
        var qW = RandomTensor(rng, EmbedDim, EmbedDim); var qB = RandomTensor(rng, EmbedDim);
        var kW = RandomTensor(rng, EmbedDim, EmbedDim); var kB = RandomTensor(rng, EmbedDim);
        var vW = RandomTensor(rng, EmbedDim, EmbedDim); var vB = RandomTensor(rng, EmbedDim);
        var oW = RandomTensor(rng, EmbedDim, EmbedDim); var oB = RandomTensor(rng, EmbedDim);

        // One sequence of Total tokens: a prefill chunk followed by single-token decode steps.
        var tokens = new Tensor<double>[Total];
        for (int i = 0; i < Total; i++) tokens[i] = RandomTensor(rng, 1, 1, EmbedDim);

        var reference = RunPaged(1, qW, qB, kW, kB, vW, vB, oW, oB, tokens);
        var sharded = RunPaged(worldSize, qW, qB, kW, kB, vW, vB, oW, oB, tokens);

        Assert.Equal(reference.Length, sharded.Length);
        for (int i = 0; i < reference.Length; i++)
            Assert.Equal(reference[i], sharded[i], Tol);
    }

    // Runs the paged TP attention over the token sequence across `worldSize` ranks concurrently (one per rank,
    // each with its own per-rank cache), and returns rank 0's flattened per-step outputs.
    private static double[] RunPaged(
        int worldSize,
        Tensor<double> qW, Tensor<double> qB, Tensor<double> kW, Tensor<double> kB,
        Tensor<double> vW, Tensor<double> vB, Tensor<double> oW, Tensor<double> oB,
        Tensor<double>[] tokens)
    {
        string envId = System.Guid.NewGuid().ToString();
        const long seqId = 1;
        var outputs = new double[worldSize][];
        var tasks = new Task[worldSize];

        for (int r = 0; r < worldSize; r++)
        {
            int rank = r;
            tasks[rank] = Task.Run(() =>
            {
                var backend = new InMemoryCommunicationBackend<double>(rank, worldSize, envId);
                backend.Initialize();
                var cache = new PagedKVCache<double>(new PagedKVCacheConfig
                {
                    BlockSize = 16,
                    NumBlocks = 64,
                    NumLayers = 1,
                    NumHeads = NumHeads / worldSize, // this rank's local heads
                    HeadDimension = HeadDim
                });
                try
                {
                    cache.AllocateSequence(seqId, Total); // all positions writable up-front
                    var attn = new TensorParallelPagedAttention<double>(backend, EmbedDim, NumHeads, cache, layerIndex: 0);
                    attn.SetFromFullWeights(qW, qB, kW, kB, vW, vB, oW, oB);

                    var collected = new List<double>();

                    // Prefill the first Prefill tokens in one call.
                    var prefillInput = StackTokens(tokens, 0, Prefill);
                    var prefillOut = attn.Forward(prefillInput, seqId, basePosition: 0);
                    AppendSpan(collected, prefillOut.AsSpan());

                    // Decode the remaining tokens one at a time.
                    for (int t = Prefill; t < Total; t++)
                    {
                        var stepOut = attn.Forward(tokens[t], seqId, basePosition: t);
                        AppendSpan(collected, stepOut.AsSpan());
                    }

                    outputs[rank] = collected.ToArray();
                }
                finally
                {
                    cache.Dispose();
                    backend.Shutdown();
                }
            });
        }

        Task.WaitAll(tasks);
        return outputs[0];
    }

    private static Tensor<double> StackTokens(Tensor<double>[] tokens, int start, int count)
    {
        var t = new Tensor<double>(new[] { 1, count, EmbedDim });
        var span = t.AsWritableSpan();
        int idx = 0;
        for (int i = 0; i < count; i++)
        {
            var src = tokens[start + i].AsSpan(); // [1,1,EmbedDim]
            for (int d = 0; d < EmbedDim; d++) span[idx++] = src[d];
        }
        return t;
    }

    private static void AppendSpan(List<double> list, System.ReadOnlySpan<double> span)
    {
        for (int i = 0; i < span.Length; i++) list.Add(span[i]);
    }

    private static Tensor<double> RandomTensor(System.Random rng, params int[] shape)
    {
        var t = new Tensor<double>(shape);
        var span = t.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = rng.NextDouble() * 2.0 - 1.0;
        return t;
    }
}
