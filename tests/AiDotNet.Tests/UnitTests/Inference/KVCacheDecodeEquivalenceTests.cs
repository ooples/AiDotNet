// Copyright (c) AiDotNet. All rights reserved.
// #1632 / #95 item: wire + verify KV-cache autoregressive decode. The KV cache only saves
// compute if a decode loop feeds ONE token at a time and relies on the cache instead of
// recomputing the whole prefix. This proves that incremental cached decode (token-by-token)
// produces the SAME output as a single full-sequence forward — i.e. the cache path is causally
// correct and the wired KV cache actually works for decode.

using System;
using System.Threading.Tasks;
using AiDotNet.Inference;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Inference;

public class KVCacheDecodeEquivalenceTests
{
    [Fact(Timeout = 120000)]
    public async Task IncrementalCachedDecode_MatchesFullSequenceForward()
    {
        await Task.Yield();
        const int seqLen = 6, embDim = 16, heads = 4, headDim = embDim / heads;

        var mha = new CachedMultiHeadAttention<float>(
            sequenceLength: seqLen, embeddingDimension: embDim, headCount: heads,
            useFlashAttention: true, layerIndex: 0, useCausalMask: true)
        {
            InferenceMode = true,
        };
        var cache = new KVCache<float>(numLayers: 1, numHeads: heads, headDim: headDim,
            maxSeqLen: seqLen, maxBatchSize: 1);
        mha.Cache = cache;

        // Deterministic input [1, seqLen, embDim].
        var rng = new Random(20260616);
        var data = new float[seqLen * embDim];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() - 0.5);
        var full = new Tensor<float>(data, new[] { 1, seqLen, embDim });

        // Run A — one full-sequence forward (token t attends 0..t via the causal mask).
        cache.Clear();
        var outFull = mha.Forward(full);
        Assert.Equal(new[] { 1, seqLen, embDim }, outFull.Shape.ToArray());

        // Run B — autoregressive decode: feed one token at a time; the cache accumulates K/V so
        // token t attends over the cached 0..t. This is the real decode path.
        cache.Clear();
        var outIncr = new float[seqLen * embDim];
        for (int t = 0; t < seqLen; t++)
        {
            var tok = new float[embDim];
            Array.Copy(data, t * embDim, tok, 0, embDim);
            var stepOut = mha.Forward(new Tensor<float>(tok, new[] { 1, 1, embDim }));
            Assert.Equal(new[] { 1, 1, embDim }, stepOut.Shape.ToArray());
            var s = stepOut.AsSpan();
            for (int e = 0; e < embDim; e++) outIncr[t * embDim + e] = s[e];
        }
        // The cache must have grown to the full sequence — proves decode actually used it.
        Assert.Equal(seqLen, cache.CurrentLength);

        // Causal equivalence: per-token incremental decode == the full forward's per-token output.
        var f = outFull.AsSpan();
        double num = 0, den = 1e-12;
        for (int i = 0; i < outIncr.Length; i++) { double d = outIncr[i] - f[i]; num += d * d; den += (double)f[i] * f[i]; }
        double relErr = Math.Sqrt(num / den);
        Assert.True(relErr < 1e-4,
            $"incremental cached decode diverges from the full-sequence forward (relErr={relErr:E3}) — "
          + "the KV-cache decode path is not causally equivalent to a single forward.");
    }
}
