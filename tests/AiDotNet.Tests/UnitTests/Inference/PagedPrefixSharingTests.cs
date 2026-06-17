// Copyright (c) AiDotNet. All rights reserved.
// #99 Stage 2 (exceed-industry, RadixAttention-style): automatic prompt-prefix KV sharing via
// PagedKVCache.ForkSequence (copy-on-write blocks). A sequence forked after a shared prefix reuses
// the prefix's KV blocks (no re-allocation, no recompute) and still produces the SAME attention as
// a fresh sequence that processed the full prefix+suffix — the correctness + savings guarantee that
// lets concurrent requests with a common system prompt share KV.

using System;
using System.Threading.Tasks;
using AiDotNet.Inference;
using AiDotNet.Inference.PagedAttention;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Inference;

public class PagedPrefixSharingTests
{
    private const int EmbDim = 16;
    private const int Heads = 4;
    private const int HeadDim = EmbDim / Heads;
    private const int BlockSize = 4;
    private const int Prefix = 4;   // exactly one block
    private const int Suffix = 2;
    private const int Total = Prefix + Suffix;

    private static PagedCachedMultiHeadAttention<float> Build(out PagedKVCache<float> cache)
    {
        var layer = new PagedCachedMultiHeadAttention<float>(Total, EmbDim, Heads, useCausalMask: true)
        { InferenceMode = true, LayerIndex = 0 };
        var rng = new Random(20260616);
        var p = layer.GetParameters();
        var pv = new float[p.Length];
        for (int i = 0; i < pv.Length; i++) pv[i] = (float)(rng.NextDouble() - 0.5) * 0.2f;
        layer.SetParameters(new Vector<float>(pv));
        cache = PagedKVCache<float>.FromMemorySize(64L * 1024 * 1024, 1, Heads, HeadDim, BlockSize);
        layer.Kernel = new PagedAttentionKernel<float>(cache, new PagedAttentionConfig
        { NumHeads = Heads, HeadDimension = HeadDim, BlockSize = BlockSize, MaxBatchSize = 8 });
        return layer;
    }

    private static float[] Tokens(int seed)
    {
        var rng = new Random(seed);
        var d = new float[Total * EmbDim];
        for (int i = 0; i < d.Length; i++) d[i] = (float)(rng.NextDouble() - 0.5);
        return d;
    }

    private static float[] Step(PagedCachedMultiHeadAttention<float> layer, long seq, float[] data, int t)
    {
        var tok = new float[EmbDim];
        Array.Copy(data, t * EmbDim, tok, 0, EmbDim);
        var step = layer.ForwardWithContext(new Tensor<float>(tok, new[] { 1, 1, EmbDim }), new InferenceForwardContext(seq, t));
        return step.AsSpan().ToArray();
    }

    [Fact(Timeout = 120000)]
    public async Task ForkedPrefix_ReusesBlocks_AndMatchesFreshSequence()
    {
        await Task.Yield();
        var layer = Build(out var cache);
        var data = Tokens(31337);

        // Reference: a fresh sequence processes the whole prefix+suffix.
        long full = 1;
        Assert.True(cache.AllocateSequence(full, 0));
        var refSuffix = new float[Suffix][];
        for (int t = 0; t < Total; t++)
        {
            var o = Step(layer, full, data, t);
            if (t >= Prefix) refSuffix[t - Prefix] = o;
        }

        // Prefix-shared: write the prefix once on a base sequence, fork it (COW), then continue the
        // fork with the suffix. The fork must NOT allocate new blocks for the shared prefix.
        long baseSeq = 2;
        Assert.True(cache.AllocateSequence(baseSeq, 0));
        for (int t = 0; t < Prefix; t++) Step(layer, baseSeq, data, t);

        int freeBeforeFork = cache.BlockManager.FreeBlockCount;
        long forkSeq = 3;
        Assert.True(cache.ForkSequence(baseSeq, forkSeq));
        int freeAfterFork = cache.BlockManager.FreeBlockCount;
        Assert.Equal(freeBeforeFork, freeAfterFork); // fork shares the prefix block(s), no new allocation

        var forkSuffix = new float[Suffix][];
        for (int t = Prefix; t < Total; t++)
        {
            forkSuffix[t - Prefix] = Step(layer, forkSeq, data, t);
        }

        // The forked sequence's suffix attention (over the SHARED prefix KV + its own suffix KV)
        // matches the fresh full sequence's suffix exactly.
        for (int i = 0; i < Suffix; i++)
        {
            double num = 0, den = 1e-12;
            for (int e = 0; e < EmbDim; e++)
            {
                double diff = forkSuffix[i][e] - refSuffix[i][e];
                num += diff * diff;
                den += (double)refSuffix[i][e] * refSuffix[i][e];
            }
            Assert.True(Math.Sqrt(num / den) < 1e-4,
                $"forked-prefix suffix token {i} diverges from the fresh full sequence");
        }
    }
}
