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
    public async Task ForkAfterSourceMutation_PreservesPrefixKv()
    {
        await Task.Yield();
        // blockSize 16 so prefix(4) AND suffix share block 0 — exercises COW on the shared block,
        // which the block-granular test below does not (its suffix lands in a new block).
        const int E = EmbDim, H = Heads, HD = E / H;
        var layer = new PagedCachedMultiHeadAttention<float>(8, E, H, useCausalMask: true)
        { InferenceMode = true, LayerIndex = 0 };
        var rng = new Random(20260616);
        var pv = new float[layer.GetParameters().Length];
        for (int i = 0; i < pv.Length; i++) pv[i] = (float)(rng.NextDouble() - 0.5) * 0.2f;
        layer.SetParameters(new Vector<float>(pv));
        var cache = PagedKVCache<float>.FromMemorySize(64L * 1024 * 1024, 1, H, HD, blockSize: 16);
        layer.Kernel = new PagedAttentionKernel<float>(cache, new PagedAttentionConfig
        { NumHeads = H, HeadDimension = HD, BlockSize = 16, MaxBatchSize = 8 });

        var data = new float[6 * E];
        var rd = new Random(55);
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rd.NextDouble() - 0.5);

        float[] StepOn(long seq, int t)
        {
            var tok = new float[E];
            Array.Copy(data, t * E, tok, 0, E);
            return layer.ForwardWithContext(new Tensor<float>(tok, new[] { 1, 1, E }), new InferenceForwardContext(seq, t)).AsSpan().ToArray();
        }

        // Reference: fresh sequence, full 6 tokens.
        Assert.True(cache.AllocateSequence(1, 0));
        float[] refOut5 = new float[E];
        for (int t = 0; t < 6; t++) { var o = StepOn(1, t); if (t == 5) refOut5 = o; }

        // Source: prefill 4 tokens, fork to base, then MUTATE source (write 2 more into shared block 0).
        Assert.True(cache.AllocateSequence(10, 0));
        for (int t = 0; t < 4; t++) StepOn(10, t);
        Assert.True(cache.ForkSequence(10, 11)); // base = prefix snapshot
        for (int t = 4; t < 6; t++) StepOn(10, t); // source decode -> COW must protect base
        cache.FreeSequence(10);

        // Fork base and continue with the real suffix; position 5 must match the fresh reference.
        Assert.True(cache.ForkSequence(11, 12));
        float[] forkOut5 = new float[E];
        for (int t = 4; t < 6; t++) forkOut5 = StepOn(12, t);

        Assert.True(RelErrV(forkOut5, refOut5) < 1e-4,
            "fork-after-source-mutation: COW failed to preserve the prefix KV (output diverges from fresh).");
    }

    private static double RelErrV(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double num = 0, den = 1e-12;
        for (int i = 0; i < a.Length; i++) { double d = a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
        return Math.Sqrt(num / den);
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
