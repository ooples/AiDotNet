using System;
using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Inference;
using AiDotNet.Inference.PagedAttention;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Full trained weights for one tensor-parallel transformer layer (un-sharded); the model slices each rank's
/// shard from these at compute time.
/// </summary>
internal sealed class TensorParallelLayerWeights<T>
{
    public required Tensor<T> QWeight { get; init; }
    public required Tensor<T> QBias { get; init; }
    public required Tensor<T> KWeight { get; init; }
    public required Tensor<T> KBias { get; init; }
    public required Tensor<T> VWeight { get; init; }
    public required Tensor<T> VBias { get; init; }
    public required Tensor<T> OWeight { get; init; }
    public required Tensor<T> OBias { get; init; }
    public required Tensor<T> UpWeight { get; init; }
    public required Tensor<T> UpBias { get; init; }
    public required Tensor<T> DownWeight { get; init; }
    public required Tensor<T> DownBias { get; init; }

    // Optional gated-SwiGLU gate projection (LLaMA/Mistral/Qwen2). When set, the FFN is
    // Down(act(Gate(x)) * Up(x)) with the activation on the gate path and a linear up path;
    // null => classic Down(act(Up(x))). Column-partitioned identically to UpWeight.
    public Tensor<T>? GateWeight { get; init; }
    public Tensor<T>? GateBias { get; init; }

    // Optional per-layer RMSNorm scale (γ) for the pre-attention (Norm1) and pre-FFN (Norm2) norms, used only in
    // the faithful path where the model reproduces a real trained transformer's RMSNorm. Null => unit scale.
    public Tensor<T>? Norm1Gamma { get; init; }
    public Tensor<T>? Norm2Gamma { get; init; }
}

/// <summary>
/// A tensor-parallel transformer language model served over PAGED KV caches. Attention heads and the
/// feed-forward hidden dimension are partitioned across <c>worldSize</c> ranks; each rank keeps its own paged KV
/// cache for its head-group. The output/down projections sum each rank's partial contribution — the "all-reduce"
/// — producing exactly the logits an un-sharded model would, while dividing the KV memory and per-token work
/// across the ranks.
/// </summary>
/// <remarks>
/// <para>
/// The forward runs each rank's compute independently — on its own GPU device (rank r -&gt; device r % deviceCount)
/// or CPU thread, with its own paged cache and scratch — then reduces the per-rank partials in a FIXED rank order,
/// so the parallel result is bit-identical to a sequential one (deterministic — no dependence on the process-global
/// compute engine or autodiff tape, which are not thread-safe). Falls back to one device / the CPU double path when
/// fewer GPUs (or none) are present. Supports plain multi-head and grouped-query attention (no rotary/ALiBi/quant).
/// </para>
/// <para><b>For Beginners:</b> This is a language model whose layers are split across several GPUs so a model
/// too big for one GPU still runs — and it remembers past tokens efficiently (paged KV) while generating.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.TextGeneration)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Megatron-LM: Training Multi-Billion Parameter Language Models", "https://arxiv.org/abs/1909.08053")]
internal sealed class TensorParallelPagedModel<T> : NeuralNetworkBase<T>
{
    private const double LayerNormEpsilon = 1e-5;

    private readonly int _worldSize;
    private readonly int _embedDim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _numLayers;
    private readonly int _ffnDim;
    private readonly int _vocabSize;
    private readonly int _localHeads;
    private readonly int _localDim;   // localHeads * headDim (query shard width)
    private readonly int _numKVHeads; // grouped-query attention: KV heads (== numHeads for plain MHA)
    private readonly int _localKVHeads; // numKVHeads / worldSize
    private readonly int _localKVDim; // localKVHeads * headDim (K/V shard width; == localDim for MHA)
    private readonly int _groupsPerLocal; // localHeads / localKVHeads (query heads sharing one local KV head)
    private readonly int _localFfn;   // ffnDim / worldSize
    private readonly double _scale;

    // Full (un-sharded) weights.
    private readonly Tensor<T> _embedding; // [vocab, embedDim]
    private readonly Tensor<T> _lmHead;    // [vocab, embedDim]
    private readonly TensorParallelLayerWeights<T>[] _layers;

    // Per-rank paged KV caches (each holds this rank's head-group KV).
    private readonly PagedKVCache<T>[] _caches;

    // Faithful path (constructed from a real trained transformer): when set, the pre-attention/pre-FFN/final
    // norms are RMSNorm with the trained γ (else the parameter-free LayerNorm reference path used by the
    // primitive equivalence tests), and the FFN uses the trained activation (else ReLU). This lets the sharded
    // model reproduce a real model's output token-for-token rather than a reference block.
    private readonly bool _useRmsNorm;
    private readonly Tensor<T>? _finalNormGamma;
    private readonly Func<double, double> _ffnActivation;
    private readonly double _rmsNormEpsilon;
    private readonly double[]? _lmHeadBias; // optional [vocab] bias added after the LM-head projection

    // GPU execution: per-(rank, layer) device paged attention. When set, each rank's head-group attention runs
    // on the GPU (FP32) instead of the manual CPU double loop; the rest of the forward stays CPU/double.
    private readonly GpuPagedAttention?[][]? _gpuAttn;
    private readonly IDirectGpuBackend?[]? _deviceBackends; // per-rank GPU device backend (rank r -> device r%N)
    private bool GpuEnabled => _gpuAttn is not null;

    // The device caches key sequences by int, but the batcher's sequence ids are long. Map each long id to a
    // distinct int (allocated on first use, released on free) so different sequences never collide on the device.
    private readonly System.Collections.Generic.Dictionary<long, int> _gpuSeqIds = new();
    private int _nextGpuSid;

    /// <summary>Whether this model's attention runs on the GPU (device paged attention) vs the CPU double path.</summary>
    public bool GpuActive => _gpuAttn is not null;

    /// <summary>The per-rank paged caches (for the composite cache the scheduler drives).</summary>
    public PagedKVCache<T>[] RankCaches => _caches;

    public TensorParallelPagedModel(
        int worldSize, int embedDim, int numHeads, int numLayers, int ffnDim, int vocabSize,
        int blockSize = 16, int numBlocks = 256,
        bool useRmsNorm = false, Tensor<T>? finalNormGamma = null,
        Func<double, double>? ffnActivation = null, double rmsNormEpsilon = 1e-6,
        Tensor<T>? lmHeadBias = null, bool useGpu = false, int? numKVHeads = null, int? headDim = null)
        : base(new MeanSquaredErrorLoss<T>())
    {
        int kvHeads = numKVHeads ?? numHeads;
        if (worldSize < 1) throw new ArgumentOutOfRangeException(nameof(worldSize));
        // An explicit headDim allows numHeads*headDim != embedDim (Gemma-style); only the default
        // (headDim = embedDim/numHeads) requires embedDim to be a multiple of numHeads.
        if (embedDim <= 0 || numHeads <= 0 || (headDim is null && embedDim % numHeads != 0))
            throw new ArgumentException($"embedDim ({embedDim}) must be a positive multiple of numHeads ({numHeads}).");
        if (headDim is { } hdv && hdv <= 0)
            throw new ArgumentException($"headDim ({hdv}) must be positive.", nameof(headDim));
        if (numHeads % worldSize != 0)
            throw new ArgumentException($"numHeads ({numHeads}) must be divisible by worldSize ({worldSize}).");
        if (kvHeads <= 0 || numHeads % kvHeads != 0)
            throw new ArgumentException($"numHeads ({numHeads}) must be a positive multiple of numKVHeads ({kvHeads}).");
        if (kvHeads % worldSize != 0)
            throw new ArgumentException($"numKVHeads ({kvHeads}) must be divisible by worldSize ({worldSize}).");
        if (ffnDim % worldSize != 0)
            throw new ArgumentException($"ffnDim ({ffnDim}) must be divisible by worldSize ({worldSize}).");

        _worldSize = worldSize;
        _embedDim = embedDim;
        _numHeads = numHeads;
        _headDim = headDim ?? (embedDim / numHeads);
        _numLayers = numLayers;
        _ffnDim = ffnDim;
        _vocabSize = vocabSize;
        _localHeads = numHeads / worldSize;
        _localDim = _localHeads * _headDim;
        _numKVHeads = kvHeads;
        _localKVHeads = kvHeads / worldSize;
        _localKVDim = _localKVHeads * _headDim;
        _groupsPerLocal = _localHeads / _localKVHeads; // query heads per local KV head
        _localFfn = ffnDim / worldSize;
        _scale = 1.0 / Math.Sqrt(_headDim);
        _useRmsNorm = useRmsNorm;
        _finalNormGamma = finalNormGamma;
        _ffnActivation = ffnActivation ?? (v => v < 0 ? 0.0 : v); // default ReLU (reference path)
        _rmsNormEpsilon = rmsNormEpsilon;
        if (lmHeadBias is not null)
        {
            var lb = lmHeadBias.AsSpan();
            _lmHeadBias = new double[vocabSize];
            for (int v = 0; v < vocabSize && v < lb.Length; v++) _lmHeadBias[v] = Convert.ToDouble(lb[v]);
        }

        _embedding = new Tensor<T>(new[] { vocabSize, embedDim });
        _lmHead = new Tensor<T>(new[] { vocabSize, embedDim });
        _layers = new TensorParallelLayerWeights<T>[numLayers];

        _caches = new PagedKVCache<T>[worldSize];
        for (int r = 0; r < worldSize; r++)
        {
            _caches[r] = new PagedKVCache<T>(new PagedKVCacheConfig
            {
                BlockSize = blockSize,
                NumBlocks = numBlocks,
                NumLayers = numLayers,
                NumHeads = _localKVHeads, // the cache stores KV heads (== query heads for plain MHA)
                HeadDimension = _headDim
            });
        }

        // GPU mode: one device paged-attention head-group per (rank, layer). Each rank is placed on a distinct
        // GPU device when several are present (rank r -> device r % deviceCount), so its KV cache + attention
        // live on that device (Megatron multi-GPU memory distribution). Falls back to CPU (leaves _gpuAttn null)
        // when no compatible GPU backend is active, so useGpu is a safe request, not a demand.
        if (useGpu && GpuPagedAttention.IsAvailable)
        {
            int deviceCount = Math.Max(1, GpuPagedAttention.DeviceCount);
            var backends = new IDirectGpuBackend?[worldSize];
            var gpu = new GpuPagedAttention?[worldSize][];
            bool ok = true;
            for (int r = 0; r < worldSize && ok; r++)
            {
                // Rank r runs on device r % deviceCount; each rank gets its own backend/context on that device.
                var backend = GpuPagedAttention.CreateDeviceBackend(r % deviceCount);
                if (backend is null) { ok = false; break; }
                backends[r] = backend;
                gpu[r] = new GpuPagedAttention?[numLayers];
                for (int l = 0; l < numLayers; l++)
                {
                    var g = GpuPagedAttention.Create(backend, _localHeads, _localKVHeads, _headDim, blockSize, numBlocks);
                    if (g is null) { ok = false; break; }
                    gpu[r][l] = g;
                }
            }
            if (ok)
            {
                _gpuAttn = gpu;
                _deviceBackends = backends;
            }
            else
            {
                // Roll back any backends created before the failure so a partial GPU init doesn't leak devices.
                for (int r = 0; r < worldSize; r++) backends[r]?.Dispose();
            }
        }
    }

    /// <summary>Seeds the full embedding/LM head and per-layer attention/FFN weights.</summary>
    public void SetFromFullWeights(Tensor<T> embedding, Tensor<T> lmHead, TensorParallelLayerWeights<T>[] layers)
    {
        if (layers is null || layers.Length != _numLayers)
            throw new ArgumentException($"Expected {_numLayers} layer-weight sets.", nameof(layers));
        Copy(_embedding, embedding);
        Copy(_lmHead, lmHead);
        for (int l = 0; l < _numLayers; l++) _layers[l] = layers[l];
    }

    private static void Copy(Tensor<T> dst, Tensor<T> src)
    {
        var d = dst.AsWritableSpan();
        var s = src.AsSpan();
        if (d.Length != s.Length) throw new ArgumentException("Weight shape mismatch.");
        for (int i = 0; i < d.Length; i++) d[i] = s[i];
    }

    /// <summary>
    /// Tensor-parallel paged forward for <paramref name="context"/>'s sequence at its position. Input is token
    /// ids <c>[1, seqLen]</c>; returns next-token logits <c>[1, seqLen, vocab]</c>. Runs ranks sequentially and
    /// reduces partials in fixed order, so the result is bit-identical to the un-sharded model.
    /// </summary>
    internal override Tensor<T> PredictWithContext(Tensor<T> input, InferenceForwardContext context)
    {
        if (context is null) throw new ArgumentNullException(nameof(context));
        if (input.Rank != 2 || input.Shape[0] != 1)
            throw new ArgumentException($"Expected token-id input [1, seqLen]; got [{string.Join(",", input.Shape)}].", nameof(input));

        int seqLen = input.Shape[1];
        long seqId = context.SequenceId;
        int basePos = context.Position;

        // Hidden state [seqLen, embedDim] as a flat double buffer; work in double throughout for determinism.
        var x = new double[seqLen * _embedDim];
        Embed(input, seqLen, x);

        // Resolve the device sequence id ONCE (single-threaded) before the parallel rank regions, since GpuSid
        // mutates the id map. -1 when running on the CPU path.
        int gpuSid = GpuEnabled ? GpuSid(seqId) : -1;

        for (int l = 0; l < _numLayers; l++)
        {
            var w = _layers[l];

            // ---- Attention sub-block: x = x + O( Attn(Norm1(x)) ). Each rank computes its O-projection PARTIAL
            // independently (on its own GPU device / CPU thread, its own paged cache + scratch); the partials are
            // then summed in fixed rank order, so the parallel result is bit-identical to the sequential one. ----
            var ln1 = Normalize(x, seqLen, w.Norm1Gamma);
            var attnPartials = new double[_worldSize][];
            int layer = l;
            RunRanks(r =>
            {
                int qShardStart = r * _localDim;   // this rank's query-head slice
                int kvShardStart = r * _localKVDim; // this rank's KV-head slice (== query slice for plain MHA)
                // Project this rank's Q (query heads) and K/V (KV heads, narrower under grouped-query attention).
                var q = ProjectRows(ln1, seqLen, w.QWeight, w.QBias, qShardStart, _localDim);
                var k = ProjectRows(ln1, seqLen, w.KWeight, w.KBias, kvShardStart, _localKVDim);
                var v = ProjectRows(ln1, seqLen, w.VWeight, w.VBias, kvShardStart, _localKVDim);

                // Attention over this rank's cached heads -> context [seqLen, localDim].
                var context_r = new double[seqLen * _localDim];
                if (GpuEnabled)
                {
                    GpuAttention(r, layer, gpuSid, q, k, v, seqLen, basePos, context_r);
                }
                else
                {
                    var keyRow = new T[_localKVDim];
                    var valRow = new T[_localKVDim];
                    var cachedKey = new T[_localKVDim];
                    var cachedVal = new T[_localKVDim];
                    // Write this rank's KV-head K/V for each step into its (rank-private) paged cache.
                    for (int t = 0; t < seqLen; t++)
                    {
                        int src = t * _localKVDim;
                        for (int d = 0; d < _localKVDim; d++) { keyRow[d] = NumOps.FromDouble(k[src + d]); valRow[d] = NumOps.FromDouble(v[src + d]); }
                        _caches[r].WriteKey(seqId, basePos + t, layer, keyRow);
                        _caches[r].WriteValue(seqId, basePos + t, layer, valRow);
                    }

                    var scores = new double[basePos + seqLen];
                    for (int t = 0; t < seqLen; t++)
                    {
                        int lastPos = basePos + t;
                        int qBase = t * _localDim;
                        for (int h = 0; h < _localHeads; h++)
                        {
                            int hOff = h * _headDim;                          // query head offset in q/context
                            int kvOff = (h / _groupsPerLocal) * _headDim;     // grouped-query: this head's KV head
                            double maxScore = double.NegativeInfinity;
                            for (int j = 0; j <= lastPos; j++)
                            {
                                _caches[r].ReadKey(seqId, j, layer, cachedKey);
                                double dot = 0.0;
                                for (int d = 0; d < _headDim; d++)
                                    dot += q[qBase + hOff + d] * Convert.ToDouble(cachedKey[kvOff + d]);
                                double s = dot * _scale;
                                scores[j] = s;
                                if (s > maxScore) maxScore = s;
                            }
                            double sumExp = 0.0;
                            for (int j = 0; j <= lastPos; j++) { double e = Math.Exp(scores[j] - maxScore); scores[j] = e; sumExp += e; }
                            double inv = sumExp > 0 ? 1.0 / sumExp : 0.0;
                            for (int d = 0; d < _headDim; d++)
                            {
                                double acc = 0.0;
                                for (int j = 0; j <= lastPos; j++)
                                {
                                    _caches[r].ReadValue(seqId, j, layer, cachedVal);
                                    acc += scores[j] * inv * Convert.ToDouble(cachedVal[kvOff + d]);
                                }
                                context_r[qBase + hOff + d] = acc;
                            }
                        }
                    }
                }

                // This rank's O-projection PARTIAL into its own buffer: partial[t,o] = Σ_j context_r[t,j] * O[o, qShardStart+j].
                var partial = new double[seqLen * _embedDim];
                AccumulateColumnProjection(partial, context_r, seqLen, w.OWeight, qShardStart, _localDim);
                attnPartials[r] = partial;
            });
            var attnOut = new double[seqLen * _embedDim];
            for (int r = 0; r < _worldSize; r++) { var p = attnPartials[r]; for (int i = 0; i < p.Length; i++) attnOut[i] += p[i]; }
            AddBias(attnOut, seqLen, w.OBias); // O bias added once after the reduce
            for (int i = 0; i < x.Length; i++) x[i] += attnOut[i]; // residual

            // ---- MLP sub-block: x = x + Down(act(Up(Norm2(x)))). Rank Down-projection partials computed in
            // parallel, then summed in fixed rank order (bit-identical to the sequential reduction). ----
            var ln2 = Normalize(x, seqLen, w.Norm2Gamma);
            var ffnPartials = new double[_worldSize][];
            RunRanks(r =>
            {
                int ffnStart = r * _localFfn;
                // Up-projection for this rank's FFN slice -> h_r [seqLen, localFfn].
                var h = ProjectRows(ln2, seqLen, w.UpWeight, w.UpBias, ffnStart, _localFfn);
                if (w.GateWeight is not null && w.GateBias is not null)
                {
                    // Gated SwiGLU: activation on the gate path, linear up path. Gate and up share this
                    // rank's ffn slice, so act(gate_r) * up_r is exactly the slice of act(gate) * up.
                    var g = ProjectRows(ln2, seqLen, w.GateWeight, w.GateBias, ffnStart, _localFfn);
                    for (int i = 0; i < h.Length; i++) h[i] = _ffnActivation(g[i]) * h[i];
                }
                else
                {
                    // Classic two-matrix FFN: activation on the up path.
                    for (int i = 0; i < h.Length; i++) h[i] = _ffnActivation(h[i]);
                }
                var partial = new double[seqLen * _embedDim];
                AccumulateColumnProjection(partial, h, seqLen, w.DownWeight, ffnStart, _localFfn);
                ffnPartials[r] = partial;
            });
            var ffnOut = new double[seqLen * _embedDim];
            for (int r = 0; r < _worldSize; r++) { var p = ffnPartials[r]; for (int i = 0; i < p.Length; i++) ffnOut[i] += p[i]; }
            AddBias(ffnOut, seqLen, w.DownBias);
            for (int i = 0; i < x.Length; i++) x[i] += ffnOut[i]; // residual
        }

        var finalNorm = Normalize(x, seqLen, _finalNormGamma);
        return Project(finalNorm, seqLen); // [1, seqLen, vocab]
    }

    // out[t, j] = Σ_i input[t, i] * W[rowStart + j, i] + bias[rowStart + j], for j in [0, rowCount).
    private double[] ProjectRows(double[] input, int seqLen, Tensor<T> weight, Tensor<T> bias, int rowStart, int rowCount)
    {
        var w = weight.AsSpan();   // [rows, embedDim] row-major
        var b = bias.AsSpan();
        var outp = new double[seqLen * rowCount];
        for (int t = 0; t < seqLen; t++)
        {
            int inBase = t * _embedDim;
            int outBase = t * rowCount;
            for (int j = 0; j < rowCount; j++)
            {
                int wBase = (rowStart + j) * _embedDim;
                double acc = Convert.ToDouble(b[rowStart + j]);
                for (int i = 0; i < _embedDim; i++)
                    acc += input[inBase + i] * Convert.ToDouble(w[wBase + i]);
                outp[outBase + j] = acc;
            }
        }
        return outp;
    }

    // output[t, o] += Σ_j input[t, j] * W[o, colStart + j], for all o (W is [outDim, fullInDim]).
    private void AccumulateColumnProjection(double[] output, double[] input, int seqLen, Tensor<T> weight, int colStart, int inCount)
    {
        var w = weight.AsSpan();
        int outDim = _embedDim;
        int fullIn = weight.Shape[1];
        for (int t = 0; t < seqLen; t++)
        {
            int inBase = t * inCount;
            int outBase = t * outDim;
            for (int o = 0; o < outDim; o++)
            {
                int wBase = o * fullIn + colStart;
                double acc = 0.0;
                for (int j = 0; j < inCount; j++)
                    acc += input[inBase + j] * Convert.ToDouble(w[wBase + j]);
                output[outBase + o] += acc;
            }
        }
    }

    private void AddBias(double[] buffer, int seqLen, Tensor<T> bias)
    {
        var b = bias.AsSpan();
        for (int t = 0; t < seqLen; t++)
        {
            int baseIdx = t * _embedDim;
            for (int o = 0; o < _embedDim; o++) buffer[baseIdx + o] += Convert.ToDouble(b[o]);
        }
    }

    // Normalizes each row. Faithful path (_useRmsNorm): RMSNorm y_d = x_d / sqrt(mean(x^2)+eps) * γ_d (γ=1 when
    // gamma is null) — matching a real trained transformer's RMSNormalizationLayer. Reference path: parameter-free
    // LayerNorm (mean-center + unit variance) used by the primitive ws-equivalence tests.
    private double[] Normalize(double[] x, int seqLen, Tensor<T>? gamma)
    {
        var result = new double[seqLen * _embedDim];
        if (_useRmsNorm)
        {
            double[]? g = null;
            if (gamma is not null)
            {
                var gs = gamma.AsSpan();
                g = new double[_embedDim];
                for (int d = 0; d < _embedDim; d++) g[d] = Convert.ToDouble(gs[d]);
            }
            for (int t = 0; t < seqLen; t++)
            {
                int b = t * _embedDim;
                double ms = 0.0;
                for (int d = 0; d < _embedDim; d++) { double v = x[b + d]; ms += v * v; }
                ms /= _embedDim;
                double inv = 1.0 / Math.Sqrt(ms + _rmsNormEpsilon);
                for (int d = 0; d < _embedDim; d++)
                    result[b + d] = x[b + d] * inv * (g is null ? 1.0 : g[d]);
            }
            return result;
        }

        for (int t = 0; t < seqLen; t++)
        {
            int b = t * _embedDim;
            double mean = 0.0;
            for (int d = 0; d < _embedDim; d++) mean += x[b + d];
            mean /= _embedDim;
            double variance = 0.0;
            for (int d = 0; d < _embedDim; d++) { double c = x[b + d] - mean; variance += c * c; }
            variance /= _embedDim;
            double inv = 1.0 / Math.Sqrt(variance + LayerNormEpsilon);
            for (int d = 0; d < _embedDim; d++) result[b + d] = (x[b + d] - mean) * inv;
        }
        return result;
    }

    private void Embed(Tensor<T> tokenIds, int seqLen, double[] outBuf)
    {
        var emb = _embedding.AsSpan();
        var ids = tokenIds.AsSpan();
        for (int t = 0; t < seqLen; t++)
        {
            int id = (int)Math.Round(Convert.ToDouble(ids[t]));
            id = ((id % _vocabSize) + _vocabSize) % _vocabSize;
            int src = id * _embedDim;
            int dst = t * _embedDim;
            for (int d = 0; d < _embedDim; d++) outBuf[dst + d] = Convert.ToDouble(emb[src + d]);
        }
    }

    private Tensor<T> Project(double[] hidden, int seqLen)
    {
        var logits = new Tensor<T>(new[] { 1, seqLen, _vocabSize });
        var outSpan = logits.AsWritableSpan();
        var head = _lmHead.AsSpan();
        for (int t = 0; t < seqLen; t++)
        {
            int hBase = t * _embedDim;
            int oBase = t * _vocabSize;
            for (int vch = 0; vch < _vocabSize; vch++)
            {
                int wBase = vch * _embedDim;
                double acc = _lmHeadBias is null ? 0.0 : _lmHeadBias[vch];
                for (int d = 0; d < _embedDim; d++) acc += hidden[hBase + d] * Convert.ToDouble(head[wBase + d]);
                outSpan[oBase + vch] = NumOps.FromDouble(acc);
            }
        }
        return logits;
    }

    // GPU attention for one (rank, layer): append this step's K/V (FP32) to the device cache and run the paged
    // decode (single query) or prefill (causal, multiple queries) kernel, writing the context into context_r.
    private void GpuAttention(int r, int l, int sid, double[] q, double[] k, double[] v, int seqLen, int basePos, double[] context_r)
    {
        var g = _gpuAttn![r]![l]!;
        // K/V are the (possibly narrower) KV-head slice; the query is the full query-head slice.
        for (int t = 0; t < seqLen; t++)
        {
            int src = t * _localKVDim;
            var kf = new float[_localKVDim];
            var vf = new float[_localKVDim];
            for (int d = 0; d < _localKVDim; d++) { kf[d] = (float)k[src + d]; vf[d] = (float)v[src + d]; }
            g.Append(sid, kf, vf);
        }

        if (seqLen == 1)
        {
            var qf = new float[_localDim];
            for (int d = 0; d < _localDim; d++) qf[d] = (float)q[d];
            var ctx = g.Decode(sid, qf, basePos + 1);
            for (int d = 0; d < _localDim && d < ctx.Length; d++) context_r[d] = ctx[d];
        }
        else
        {
            var qf = new float[seqLen * _localDim];
            for (int i = 0; i < qf.Length; i++) qf[i] = (float)q[i];
            var ctx = g.Prefill(sid, qf, seqLen, basePos);
            for (int i = 0; i < seqLen * _localDim && i < ctx.Length; i++) context_r[i] = ctx[i];
        }
    }

    // Runs each rank's independent per-rank work (own device/thread, own paged cache + scratch, own partial
    // buffer). Ranks are parallelized across GPU devices / CPU cores when worldSize > 1; the callers sum the
    // per-rank partials in fixed rank order afterwards, so the result stays deterministic (bit-identical to
    // sequential). A single rank runs inline to avoid thread-pool overhead.
    private void RunRanks(Action<int> perRank)
    {
        if (_worldSize <= 1) { perRank(0); return; }
        System.Threading.Tasks.Parallel.For(0, _worldSize, perRank);
    }

    // Maps a batcher long sequence id to the distinct int the device caches use (stable for the sequence's life).
    private int GpuSid(long seqId)
    {
        if (!_gpuSeqIds.TryGetValue(seqId, out int sid))
        {
            sid = _nextGpuSid++;
            _gpuSeqIds[seqId] = sid;
        }
        return sid;
    }

    /// <summary>Frees this sequence's device KV across all ranks/layers (GPU mode); no-op on the CPU path.</summary>
    internal void FreeGpuSequence(long seqId)
    {
        if (_gpuAttn is null) return;
        if (!_gpuSeqIds.TryGetValue(seqId, out int sid)) return;
        for (int r = 0; r < _worldSize; r++)
            for (int l = 0; l < _numLayers; l++)
                _gpuAttn[r]?[l]?.Free(sid);
        _gpuSeqIds.Remove(seqId);
    }

    /// <summary>Frees the per-rank caches.</summary>
    public void ShutdownRanks()
    {
        for (int r = 0; r < _worldSize; r++) _caches[r].Dispose();
        if (_gpuAttn is not null)
            for (int r = 0; r < _worldSize; r++)
                for (int l = 0; l < _numLayers; l++)
                    _gpuAttn[r]?[l]?.Dispose();
        if (_deviceBackends is not null)
            for (int r = 0; r < _worldSize; r++)
                _deviceBackends[r]?.Dispose();
    }

    // ---- NeuralNetworkBase inference-only overrides ----

    protected override void InitializeLayers() { /* manual tensor-parallel model; no ILayer stack */ }

    public override void UpdateParameters(Vector<T> parameters)
        => throw new NotSupportedException("TensorParallelPagedModel is an inference-only serving model.");

    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        Name = "TensorParallelPagedModel",
        Description = $"Tensor-parallel paged serving model (worldSize={_worldSize}, layers={_numLayers}).",
        AdditionalInfo = new System.Collections.Generic.Dictionary<string, object>
        {
            ["WorldSize"] = _worldSize,
            ["EmbedDim"] = _embedDim,
            ["NumHeads"] = _numHeads,
            ["NumLayers"] = _numLayers,
            ["VocabSize"] = _vocabSize
        }
    };

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        => throw new NotSupportedException("TensorParallelPagedModel is a live serving model and is not serialized.");

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        => throw new NotSupportedException("TensorParallelPagedModel is a live serving model and is not deserialized.");

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => throw new NotSupportedException("TensorParallelPagedModel cannot be cloned.");
}
