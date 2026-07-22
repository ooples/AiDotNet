using System;
using AiDotNet.Inference.PagedAttention;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.DistributedTraining.Layers;

/// <summary>
/// Tensor-parallel multi-head self-attention over a PAGED KV cache, for tensor-parallel serving. Each rank owns
/// a contiguous group of heads: it projects Q/K/V for its local heads (<see cref="ColumnParallelLinear{T}"/>),
/// writes this step's K/V into its own <see cref="PagedKVCache{T}"/>, attends its local-head queries against its
/// cached K/V, and the output projection (<see cref="RowParallelLinear{T}"/>) all-reduces the per-rank head
/// contributions into the full output. Exactly one collective (the output all-reduce) runs per attention call.
/// </summary>
/// <remarks>
/// <para>
/// This is the paged counterpart of <see cref="TensorParallelAttention{T}"/>: instead of recomputing attention
/// over the whole input each step, each rank reads its head-group's keys/values from its paged cache (O(1) new
/// KV per token). With head count divisible by the world size the sharded result equals the un-sharded model's.
/// Plain scaled-dot-product attention (no rotary/ALiBi/quantization) so it is numerically transparent under
/// sharding — those extensions can be layered on later.
/// </para>
/// <para><b>For Beginners:</b> This is the "remembering" version of tensor-parallel attention used when serving.
/// Each GPU keeps the key/value memory for its own heads and only computes those, then the pieces are summed —
/// giving the same answer as one big GPU while splitting the memory and work.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
internal sealed class TensorParallelPagedAttention<T>
{
    private static readonly INumericOperations<T> NumOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();

    private readonly ColumnParallelLinear<T> _q;
    private readonly ColumnParallelLinear<T> _k;
    private readonly ColumnParallelLinear<T> _v;
    private readonly RowParallelLinear<T> _o;

    private readonly PagedKVCache<T> _cache;
    private readonly int _layerIndex;
    private readonly int _embedDim;
    private readonly int _headDim;
    private readonly int _localHeads;
    private readonly int _localDim; // localHeads * headDim
    private readonly double _scale;

    /// <summary>Creates a paged tensor-parallel attention block for one rank.</summary>
    /// <param name="backend">The tensor-parallel communication backend (defines this rank / world size).</param>
    /// <param name="embedDim">Model embedding dimension (= numHeads * headDim).</param>
    /// <param name="numHeads">Total heads. Must be divisible by the world size so each rank owns whole heads.</param>
    /// <param name="cache">This rank's paged KV cache (sized for the rank's local heads).</param>
    /// <param name="layerIndex">The transformer layer index this attention writes/reads in the cache.</param>
    public TensorParallelPagedAttention(
        ICommunicationBackend<T> backend, int embedDim, int numHeads, PagedKVCache<T> cache, int layerIndex)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (cache is null) throw new ArgumentNullException(nameof(cache));
        if (embedDim <= 0 || numHeads <= 0 || embedDim % numHeads != 0)
            throw new ArgumentException($"embedDim ({embedDim}) must be a positive multiple of numHeads ({numHeads}).");
        if (numHeads % backend.WorldSize != 0)
            throw new ArgumentException($"numHeads ({numHeads}) must be divisible by world size ({backend.WorldSize}).");

        _embedDim = embedDim;
        _headDim = embedDim / numHeads;
        _localHeads = numHeads / backend.WorldSize;
        _localDim = _localHeads * _headDim;
        _scale = 1.0 / Math.Sqrt(_headDim);
        _cache = cache;
        _layerIndex = layerIndex;

        _q = new ColumnParallelLinear<T>(backend, embedDim, embedDim, gatherOutput: false);
        _k = new ColumnParallelLinear<T>(backend, embedDim, embedDim, gatherOutput: false);
        _v = new ColumnParallelLinear<T>(backend, embedDim, embedDim, gatherOutput: false);
        _o = new RowParallelLinear<T>(backend, embedDim, embedDim);
    }

    /// <summary>Seeds Q/K/V/O from full (un-sharded) weights, slicing this rank's shard.</summary>
    public void SetFromFullWeights(
        Tensor<T> qW, Tensor<T> qB, Tensor<T> kW, Tensor<T> kB, Tensor<T> vW, Tensor<T> vB, Tensor<T> oW, Tensor<T> oB)
    {
        _q.SetFromFullWeights(qW, qB);
        _k.SetFromFullWeights(kW, kB);
        _v.SetFromFullWeights(vW, vB);
        _o.SetFromFullWeights(oW, oB);
    }

    /// <summary>
    /// Runs paged attention for <paramref name="seqId"/> at <paramref name="basePosition"/>. Input is
    /// <c>[1, seqLen, embedDim]</c> (prefill seqLen&gt;1 or decode seqLen==1); the sequence must already be
    /// allocated in the cache to at least <c>basePosition + seqLen</c>. Returns <c>[1, seqLen, embedDim]</c>
    /// (identical across ranks after the output all-reduce).
    /// </summary>
    public Tensor<T> Forward(Tensor<T> input, long seqId, int basePosition)
    {
        if (input.Rank != 3 || input.Shape[0] != 1 || input.Shape[2] != _embedDim)
            throw new ArgumentException($"Expected input [1, seqLen, {_embedDim}]; got [{string.Join(",", input.Shape)}].", nameof(input));

        int seqLen = input.Shape[1];
        var flat = input.Reshape(seqLen, _embedDim);

        // Column-parallel Q/K/V for this rank's local heads.
        var q = _q.Forward(flat); // [seqLen, localDim]
        var k = _k.Forward(flat);
        var v = _v.Forward(flat);
        var qs = q.AsSpan();
        var ks = k.AsSpan();
        var vs = v.AsSpan();

        // Write this step's K/V into the rank's paged cache at the absolute positions.
        var keyRow = new T[_localDim];
        var valRow = new T[_localDim];
        for (int t = 0; t < seqLen; t++)
        {
            int src = t * _localDim;
            for (int d = 0; d < _localDim; d++) { keyRow[d] = ks[src + d]; valRow[d] = vs[src + d]; }
            _cache.WriteKey(seqId, basePosition + t, _layerIndex, keyRow);
            _cache.WriteValue(seqId, basePosition + t, _layerIndex, valRow);
        }

        // Attend each query against the cached K/V for its head group (causal: positions 0..basePosition+t).
        var context = new Tensor<T>(new[] { seqLen, _localDim });
        var ctx = context.AsWritableSpan();
        var cachedKey = new T[_localDim];
        var cachedVal = new T[_localDim];
        // Per-head score rows for one query position (flattened [head, position]). Each cached K/V row is read
        // ONCE per position and reused across every head, rather than re-reading the whole _localDim-wide row
        // (under the cache lock) _localHeads times — the read traffic and lock acquisitions drop by _localHeads.
        int scoreStride = basePosition + seqLen;
        var scores = new double[_localHeads * scoreStride];
        var maxScore = new double[_localHeads];
        var sumExp = new double[_localHeads];
        var acc = new double[_localDim];

        for (int t = 0; t < seqLen; t++)
        {
            int lastPos = basePosition + t;
            int qBase = t * _localDim;

            // Pass 1 — read each cached key row once, score every head against it.
            for (int h = 0; h < _localHeads; h++) maxScore[h] = double.NegativeInfinity;
            for (int j = 0; j <= lastPos; j++)
            {
                _cache.ReadKey(seqId, j, _layerIndex, cachedKey);
                for (int h = 0; h < _localHeads; h++)
                {
                    int hOff = h * _headDim;
                    double dot = 0.0;
                    for (int d = 0; d < _headDim; d++)
                        dot += Convert.ToDouble(qs[qBase + hOff + d]) * Convert.ToDouble(cachedKey[hOff + d]);
                    double s = dot * _scale;
                    scores[h * scoreStride + j] = s;
                    if (s > maxScore[h]) maxScore[h] = s;
                }
            }

            // Softmax per head (over positions 0..lastPos).
            for (int h = 0; h < _localHeads; h++)
            {
                double se = 0.0;
                int rowBase = h * scoreStride;
                for (int j = 0; j <= lastPos; j++) { double e = Math.Exp(scores[rowBase + j] - maxScore[h]); scores[rowBase + j] = e; se += e; }
                sumExp[h] = se;
            }

            // Pass 2 — read each cached value row once, accumulate the weighted context for every head.
            Array.Clear(acc, 0, _localDim);
            for (int j = 0; j <= lastPos; j++)
            {
                _cache.ReadValue(seqId, j, _layerIndex, cachedVal);
                for (int h = 0; h < _localHeads; h++)
                {
                    int hOff = h * _headDim;
                    double inv = sumExp[h] > 0 ? 1.0 / sumExp[h] : 0.0;
                    double w = scores[h * scoreStride + j] * inv;
                    for (int d = 0; d < _headDim; d++) acc[hOff + d] += w * Convert.ToDouble(cachedVal[hOff + d]);
                }
            }

            int ctxBase = t * _localDim;
            for (int d = 0; d < _localDim; d++) ctx[ctxBase + d] = NumOps.FromDouble(acc[d]);
        }

        // Row-parallel output projection: all-reduce the per-rank head contributions into the full output.
        var outFlat = _o.Forward(context); // [seqLen, embedDim]
        return outFlat.Reshape(1, seqLen, _embedDim);
    }
}
