// Copyright (c) AiDotNet. All rights reserved.
// #99 Stage 3 (exceed-industry combo): quantized paged KV decode. With int8 weight-only quantization
// engaged, the per-sequence paged decode must still be correct — incremental (token-by-token) decode
// equals the full forward — and stay close to the fp32 result. This lets more sequences stay resident
// (smaller weights) without breaking the paged per-sequence decode proven in Stages 1-2.

using System;
using System.Threading.Tasks;
using AiDotNet.Inference;
using AiDotNet.Inference.PagedAttention;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Inference;

public class PagedQuantizedDecodeTests
{
    private const int SeqLen = 6;
    private const int EmbDim = 16;
    private const int Heads = 4;
    private const int HeadDim = EmbDim / Heads;

    private static PagedCachedMultiHeadAttention<float> Build(bool quantize, out PagedKVCache<float> cache)
    {
        var layer = new PagedCachedMultiHeadAttention<float>(SeqLen, EmbDim, Heads, useCausalMask: true)
        {
            InferenceMode = true,
            LayerIndex = 0,
            EnableWeightOnlyQuantization = quantize
        };
        var rng = new Random(20260616);
        var p = layer.GetParameters();
        var pv = new float[p.Length];
        for (int i = 0; i < pv.Length; i++) pv[i] = (float)(rng.NextDouble() - 0.5) * 0.2f;
        layer.SetParameters(new Vector<float>(pv));
        cache = PagedKVCache<float>.FromMemorySize(64L * 1024 * 1024, 1, Heads, HeadDim, 16);
        layer.Kernel = new PagedAttentionKernel<float>(cache, new PagedAttentionConfig
        { NumHeads = Heads, HeadDimension = HeadDim, BlockSize = 16, MaxBatchSize = 8 });
        return layer;
    }

    private static float[] Data(int seed)
    {
        var rng = new Random(seed);
        var d = new float[SeqLen * EmbDim];
        for (int i = 0; i < d.Length; i++) d[i] = (float)(rng.NextDouble() - 0.5);
        return d;
    }

    private static float[] Full(PagedCachedMultiHeadAttention<float> layer, PagedKVCache<float> cache, long id, float[] data)
    {
        Assert.True(cache.AllocateSequence(id, 0));
        return layer.ForwardWithContext(new Tensor<float>(data, new[] { 1, SeqLen, EmbDim }), new InferenceForwardContext(id, 0)).AsSpan().ToArray();
    }

    private static float[] Incremental(PagedCachedMultiHeadAttention<float> layer, PagedKVCache<float> cache, long id, float[] data)
    {
        Assert.True(cache.AllocateSequence(id, 0));
        var outBuf = new float[SeqLen * EmbDim];
        for (int t = 0; t < SeqLen; t++)
        {
            var tok = new float[EmbDim];
            Array.Copy(data, t * EmbDim, tok, 0, EmbDim);
            var step = layer.ForwardWithContext(new Tensor<float>(tok, new[] { 1, 1, EmbDim }), new InferenceForwardContext(id, t));
            var s = step.AsSpan();
            for (int e = 0; e < EmbDim; e++) outBuf[t * EmbDim + e] = s[e];
        }
        return outBuf;
    }

    private static double RelErr(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double num = 0, den = 1e-12;
        for (int i = 0; i < a.Length; i++) { double d = a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
        return Math.Sqrt(num / den);
    }

    [Fact(Timeout = 120000)]
    public async Task QuantizedPagedDecode_IncrementalMatchesFull_AndApproximatesFp32()
    {
        await Task.Yield();
        var data = Data(909);

        var fp32 = Build(quantize: false, out var cacheF);
        var outFp32 = Full(fp32, cacheF, 1, data);

        var q = Build(quantize: true, out var cacheQ);
        var qFull = Full(q, cacheQ, 1, data);
        var qIncr = Incremental(q, cacheQ, 2, data);

        // All outputs finite. (Use !IsNaN && !IsInfinity instead of float.IsFinite — the latter
        // is net5+ only and this test project also targets net471.)
        foreach (var v in qIncr) Assert.True(!float.IsNaN(v) && !float.IsInfinity(v));

        // Core: quantized incremental decode == quantized full forward (per-sequence paged decode
        // stays correct under int8 weight-only quantization).
        Assert.True(RelErr(qIncr, qFull) < 1e-4,
            "quantized incremental paged decode must equal the quantized full forward");

        // Quantized output stays close to fp32 (int8 weight-only error is bounded).
        Assert.True(RelErr(qFull, outFp32) < 0.3,
            "quantized paged output should approximate the fp32 output");
    }
}
