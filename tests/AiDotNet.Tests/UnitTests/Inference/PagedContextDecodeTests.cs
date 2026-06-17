// Copyright (c) AiDotNet. All rights reserved.
// #99: per-sequence paged-KV decode. PagedCachedMultiHeadAttention.ForwardWithContext routes by a
// per-call InferenceForwardContext (sequence id + position) instead of mutable layer state, so one
// shared layer + one shared PagedKVCache can decode many sequences concurrently, isolated by id.
// These tests prove (a) incremental ctx decode == full-sequence forward (causal correctness) and
// (b) interleaving two sequences does not corrupt either (isolation) — the foundation for true
// concurrent serving.

using System;
using System.Threading.Tasks;
using AiDotNet.Inference;
using AiDotNet.Inference.PagedAttention;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Inference;

public class PagedContextDecodeTests
{
    private const int SeqLen = 6;
    private const int EmbDim = 16;
    private const int Heads = 4;
    private const int HeadDim = EmbDim / Heads;

    private static PagedCachedMultiHeadAttention<float> BuildLayer(out PagedKVCache<float> cache)
    {
        var layer = new PagedCachedMultiHeadAttention<float>(
            sequenceLength: SeqLen, embeddingDimension: EmbDim, headCount: Heads, useCausalMask: true)
        {
            InferenceMode = true,
            LayerIndex = 0
        };

        // Deterministic non-degenerate weights so different inputs yield different outputs.
        var p = layer.GetParameters();
        var rng = new Random(20260616);
        var pv = new float[p.Length];
        for (int i = 0; i < pv.Length; i++) pv[i] = (float)(rng.NextDouble() - 0.5) * 0.2f;
        layer.SetParameters(new Vector<float>(pv));

        cache = PagedKVCache<float>.FromMemorySize(
            availableBytes: 64L * 1024 * 1024, numLayers: 1, numHeads: Heads, headDim: HeadDim, blockSize: 16);
        layer.Kernel = new PagedAttentionKernel<float>(cache, new PagedAttentionConfig
        {
            NumHeads = Heads,
            HeadDimension = HeadDim,
            BlockSize = 16,
            MaxBatchSize = 8
        });
        return layer;
    }

    private static float[] RandomSequence(int seed)
    {
        var rng = new Random(seed);
        var data = new float[SeqLen * EmbDim];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() - 0.5);
        return data;
    }

    private static double RelErr(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double num = 0, den = 1e-12;
        for (int i = 0; i < a.Length; i++) { double d = a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
        return Math.Sqrt(num / den);
    }

    [Fact(Timeout = 120000)]
    public async Task IncrementalContextDecode_MatchesFullSequenceForward()
    {
        await Task.Yield();
        var layer = BuildLayer(out var cache);
        var data = RandomSequence(20260616);

        long seqFull = 9001;
        Assert.True(cache.AllocateSequence(seqFull, 0));
        var outFull = layer.ForwardWithContext(
            new Tensor<float>(data, new[] { 1, SeqLen, EmbDim }), new InferenceForwardContext(seqFull, 0));

        long seqIncr = 9002;
        Assert.True(cache.AllocateSequence(seqIncr, 0));
        var outIncr = new float[SeqLen * EmbDim];
        for (int t = 0; t < SeqLen; t++)
        {
            var tok = new float[EmbDim];
            Array.Copy(data, t * EmbDim, tok, 0, EmbDim);
            var step = layer.ForwardWithContext(
                new Tensor<float>(tok, new[] { 1, 1, EmbDim }), new InferenceForwardContext(seqIncr, t));
            var s = step.AsSpan();
            for (int e = 0; e < EmbDim; e++) outIncr[t * EmbDim + e] = s[e];
        }

        double relErr = RelErr(outIncr, outFull.AsSpan());
        Assert.True(relErr < 1e-4,
            $"per-sequence incremental ctx decode diverges from the full forward (relErr={relErr:E3}).");
    }

    [Fact(Timeout = 120000)]
    public async Task InterleavedSequences_DoNotCorruptEachOther()
    {
        await Task.Yield();
        var layer = BuildLayer(out var cache);
        var dataA = RandomSequence(111);
        var dataB = RandomSequence(222);

        // Baseline: decode A fully, then B fully (sequential, non-interleaved).
        var refA = DecodeIncremental(layer, cache, 1001, dataA);
        var refB = DecodeIncremental(layer, cache, 1002, dataB);

        // Interleaved: step A0, B0, A1, B1, ... on fresh sequence ids over the SAME shared cache.
        long seqA = 2001, seqB = 2002;
        Assert.True(cache.AllocateSequence(seqA, 0));
        Assert.True(cache.AllocateSequence(seqB, 0));
        var interA = new float[SeqLen * EmbDim];
        var interB = new float[SeqLen * EmbDim];
        for (int t = 0; t < SeqLen; t++)
        {
            CopyStep(layer, seqA, dataA, t, interA);
            CopyStep(layer, seqB, dataB, t, interB);
        }

        // Interleaving must not change either sequence's outputs.
        Assert.True(RelErr(interA, refA) < 1e-4, "sequence A corrupted by interleaving with B");
        Assert.True(RelErr(interB, refB) < 1e-4, "sequence B corrupted by interleaving with A");
        // And the two sequences genuinely differ (the test isn't trivially passing on zeros).
        Assert.True(RelErr(refA, refB) > 1e-2, "sequences A and B should produce different outputs");
    }

    private static float[] DecodeIncremental(PagedCachedMultiHeadAttention<float> layer, PagedKVCache<float> cache, long seqId, float[] data)
    {
        Assert.True(cache.AllocateSequence(seqId, 0));
        var outBuf = new float[SeqLen * EmbDim];
        for (int t = 0; t < SeqLen; t++) CopyStep(layer, seqId, data, t, outBuf);
        return outBuf;
    }

    private static void CopyStep(PagedCachedMultiHeadAttention<float> layer, long seqId, float[] data, int t, float[] dest)
    {
        var tok = new float[EmbDim];
        Array.Copy(data, t * EmbDim, tok, 0, EmbDim);
        var step = layer.ForwardWithContext(
            new Tensor<float>(tok, new[] { 1, 1, EmbDim }), new InferenceForwardContext(seqId, t));
        var s = step.AsSpan();
        for (int e = 0; e < EmbDim; e++) dest[t * EmbDim + e] = s[e];
    }
}
