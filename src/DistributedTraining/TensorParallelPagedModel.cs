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
/// The forward runs the ranks SEQUENTIALLY and reduces the partials in a fixed rank order, so the sharded result
/// is bit-identical to the un-sharded one (deterministic — no dependence on the process-global compute engine or
/// autodiff tape, which are not thread-safe). A real multi-GPU deployment runs the ranks on separate devices in
/// parallel; the math (and result) is the same. Plain pre-LN GPT blocks (no rotary/ALiBi/quantization).
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
    private readonly int _localDim;   // localHeads * headDim (attention shard width)
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
    private bool GpuEnabled => _gpuAttn is not null;

    /// <summary>Whether this model's attention runs on the GPU (device paged attention) vs the CPU double path.</summary>
    public bool GpuActive => _gpuAttn is not null;

    /// <summary>The per-rank paged caches (for the composite cache the scheduler drives).</summary>
    public PagedKVCache<T>[] RankCaches => _caches;

    public TensorParallelPagedModel(
        int worldSize, int embedDim, int numHeads, int numLayers, int ffnDim, int vocabSize,
        int blockSize = 16, int numBlocks = 256,
        bool useRmsNorm = false, Tensor<T>? finalNormGamma = null,
        Func<double, double>? ffnActivation = null, double rmsNormEpsilon = 1e-6,
        Tensor<T>? lmHeadBias = null, bool useGpu = false)
        : base(new MeanSquaredErrorLoss<T>())
    {
        if (worldSize < 1) throw new ArgumentOutOfRangeException(nameof(worldSize));
        if (embedDim <= 0 || numHeads <= 0 || embedDim % numHeads != 0)
            throw new ArgumentException($"embedDim ({embedDim}) must be a positive multiple of numHeads ({numHeads}).");
        if (numHeads % worldSize != 0)
            throw new ArgumentException($"numHeads ({numHeads}) must be divisible by worldSize ({worldSize}).");
        if (ffnDim % worldSize != 0)
            throw new ArgumentException($"ffnDim ({ffnDim}) must be divisible by worldSize ({worldSize}).");

        _worldSize = worldSize;
        _embedDim = embedDim;
        _numHeads = numHeads;
        _headDim = embedDim / numHeads;
        _numLayers = numLayers;
        _ffnDim = ffnDim;
        _vocabSize = vocabSize;
        _localHeads = numHeads / worldSize;
        _localDim = _localHeads * _headDim;
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
                NumHeads = _localHeads,
                HeadDimension = _headDim
            });
        }

        // GPU mode: one device paged-attention head-group per (rank, layer). Falls back to CPU (leaves
        // _gpuAttn null) when no compatible GPU backend is active, so useGpu is a safe request, not a demand.
        if (useGpu && GpuPagedAttention.IsAvailable)
        {
            var gpu = new GpuPagedAttention?[worldSize][];
            bool ok = true;
            for (int r = 0; r < worldSize && ok; r++)
            {
                gpu[r] = new GpuPagedAttention?[numLayers];
                for (int l = 0; l < numLayers; l++)
                {
                    var g = GpuPagedAttention.TryCreate(_localHeads, _headDim, blockSize, numBlocks);
                    if (g is null) { ok = false; break; }
                    gpu[r][l] = g;
                }
            }
            _gpuAttn = ok ? gpu : null;
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

        var keyRow = new T[_localDim];
        var valRow = new T[_localDim];
        var cachedKey = new T[_localDim];
        var cachedVal = new T[_localDim];

        for (int l = 0; l < _numLayers; l++)
        {
            var w = _layers[l];

            // ---- Attention sub-block: x = x + O( Attn(Norm1(x)) ), partials summed over ranks ----
            var ln1 = Normalize(x, seqLen, w.Norm1Gamma);
            var attnOut = new double[seqLen * _embedDim]; // accumulates O-projection partials + bias
            for (int r = 0; r < _worldSize; r++)
            {
                int shardStart = r * _localDim; // this rank's contiguous slice of the embedding/head dims
                // Project this rank's Q/K/V for its head group (rows [shardStart, shardStart+localDim)).
                var q = ProjectRows(ln1, seqLen, w.QWeight, w.QBias, shardStart, _localDim);
                var k = ProjectRows(ln1, seqLen, w.KWeight, w.KBias, shardStart, _localDim);
                var v = ProjectRows(ln1, seqLen, w.VWeight, w.VBias, shardStart, _localDim);

                // Attention over this rank's cached heads -> context [seqLen, localDim].
                var context_r = new double[seqLen * _localDim];
                if (GpuEnabled)
                {
                    GpuAttention(r, l, seqId, q, k, v, seqLen, basePos, context_r);
                }
                else
                {
                    // Write K/V for this step into the rank's paged cache.
                    for (int t = 0; t < seqLen; t++)
                    {
                        int src = t * _localDim;
                        for (int d = 0; d < _localDim; d++) { keyRow[d] = NumOps.FromDouble(k[src + d]); valRow[d] = NumOps.FromDouble(v[src + d]); }
                        _caches[r].WriteKey(seqId, basePos + t, l, keyRow);
                        _caches[r].WriteValue(seqId, basePos + t, l, valRow);
                    }

                    var scores = new double[basePos + seqLen];
                    for (int t = 0; t < seqLen; t++)
                    {
                        int lastPos = basePos + t;
                        int qBase = t * _localDim;
                        for (int h = 0; h < _localHeads; h++)
                        {
                            int hOff = h * _headDim;
                            double maxScore = double.NegativeInfinity;
                            for (int j = 0; j <= lastPos; j++)
                            {
                                _caches[r].ReadKey(seqId, j, l, cachedKey);
                                double dot = 0.0;
                                for (int d = 0; d < _headDim; d++)
                                    dot += q[qBase + hOff + d] * Convert.ToDouble(cachedKey[hOff + d]);
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
                                    _caches[r].ReadValue(seqId, j, l, cachedVal);
                                    acc += scores[j] * inv * Convert.ToDouble(cachedVal[hOff + d]);
                                }
                                context_r[qBase + hOff + d] = acc;
                            }
                        }
                    }
                }

                // Output projection PARTIAL: attnOut[t,o] += Σ_j context_r[t,j] * O[o, shardStart+j].
                AccumulateColumnProjection(attnOut, context_r, seqLen, w.OWeight, shardStart, _localDim);
            }
            AddBias(attnOut, seqLen, w.OBias); // O bias added once after the reduce
            for (int i = 0; i < x.Length; i++) x[i] += attnOut[i]; // residual

            // ---- MLP sub-block: x = x + Down(act(Up(Norm2(x)))), partials summed over ranks ----
            var ln2 = Normalize(x, seqLen, w.Norm2Gamma);
            var ffnOut = new double[seqLen * _embedDim];
            for (int r = 0; r < _worldSize; r++)
            {
                int ffnStart = r * _localFfn;
                // Up-projection for this rank's FFN slice + activation -> h_r [seqLen, localFfn].
                var h = ProjectRows(ln2, seqLen, w.UpWeight, w.UpBias, ffnStart, _localFfn);
                for (int i = 0; i < h.Length; i++) h[i] = _ffnActivation(h[i]);
                // Down-projection PARTIAL: ffnOut[t,o] += Σ_f h[t,f] * Down[o, ffnStart+f].
                AccumulateColumnProjection(ffnOut, h, seqLen, w.DownWeight, ffnStart, _localFfn);
            }
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
    private void GpuAttention(int r, int l, long seqId, double[] q, double[] k, double[] v, int seqLen, int basePos, double[] context_r)
    {
        var g = _gpuAttn![r]![l]!;
        int sid = (int)(seqId & 0x7FFFFFFF);
        for (int t = 0; t < seqLen; t++)
        {
            int src = t * _localDim;
            var kf = new float[_localDim];
            var vf = new float[_localDim];
            for (int d = 0; d < _localDim; d++) { kf[d] = (float)k[src + d]; vf[d] = (float)v[src + d]; }
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

    /// <summary>Frees this sequence's device KV across all ranks/layers (GPU mode); no-op on the CPU path.</summary>
    internal void FreeGpuSequence(long seqId)
    {
        if (_gpuAttn is null) return;
        int sid = (int)(seqId & 0x7FFFFFFF);
        for (int r = 0; r < _worldSize; r++)
            for (int l = 0; l < _numLayers; l++)
                _gpuAttn[r]?[l]?.Free(sid);
    }

    /// <summary>Frees the per-rank caches.</summary>
    public void ShutdownRanks()
    {
        for (int r = 0; r < _worldSize; r++) _caches[r].Dispose();
        if (_gpuAttn is not null)
            for (int r = 0; r < _worldSize; r++)
                for (int l = 0; l < _numLayers; l++)
                    _gpuAttn[r]?[l]?.Dispose();
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
