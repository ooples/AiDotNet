using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Attention;

namespace AiDotNet.Diffusion.Conditioning;

/// <summary>
/// T5 text encoder conditioning module for diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// T5 (Text-To-Text Transfer Transformer) text encoder provides high-dimensional sequence
/// embeddings for advanced diffusion models. T5-XXL is used in Imagen, SD3, and FLUX.1
/// for its superior text understanding compared to CLIP.
/// </para>
/// <para>
/// <b>For Beginners:</b> T5 is a more powerful text encoder than CLIP.
///
/// Why T5 in addition to CLIP:
/// - CLIP understands image-text relationships (good for visual concepts)
/// - T5 understands language deeply (good for complex prompts, text rendering)
/// - Together they give the best of both worlds
///
/// Key differences from CLIP:
/// - Much larger: T5-XXL has 4.7B parameters (vs CLIP's 123M-354M)
/// - Longer sequences: 256-512 tokens (vs CLIP's 77)
/// - Higher dimensional: 4096-dim (vs CLIP's 768-1280)
/// - No pooled output: Only produces sequence embeddings
/// - Better at: Complex prompts, counting, spatial relationships, text in images
///
/// T5 variants used in diffusion:
/// - T5-XXL: 4096-dim, 24 layers, used in Imagen, SD3, FLUX.1
/// - T5-XL: 2048-dim, 24 layers, sometimes used for memory-constrained setups
/// - T5-Large: 1024-dim, 24 layers
///
/// Important: T5 uses relative position encodings (not absolute like CLIP),
/// which helps with varying-length inputs.
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
/// JMLR 2020 — defines the T5 architecture: pre-norm RMSNorm, no attention scaling, ReLU FFN,
/// shared learned relative position bias with logarithmic bucketing.</item>
/// <item>Shazeer, "GLU Variants Improve Transformer", arXiv:2002.05202, 2020 — introduces the GeGLU
/// activation that replaces the original ReLU FFN block in T5.1.1 (Google's later T5 codebase
/// improvement). This implementation uses GeGLU because the empirical gains over ReLU are reported
/// across multiple downstream tasks while changing only the FFN block.</item>
/// </list>
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class T5TextConditioner<T> : TextConditioningBase<T>
{
    /// <summary>
    /// Relative position bias weights for self-attention.
    /// </summary>
    private readonly Vector<T> _relativePositionBias;

    /// <summary>
    /// Number of relative position buckets.
    /// </summary>
    private readonly int _numBuckets;

    /// <summary>
    /// The T5 variant name.
    /// </summary>
    private readonly string _variant;

    /// <summary>
    /// FFN inner dimension per T5.1.1 variant configs (d_ff in the Raffel/Shazeer notation).
    /// </summary>
    private readonly int _ffnDim;

    // ---- Cached per-layer weight tensors. Lazily built from TransformerWeights on first call. ----
    // Layout per layer (offset into TransformerWeights, sizes in elements):
    //   0                                    : norm1_gamma  [H]
    //   H                                    : W_q          [H, H]
    //   H + H^2                              : W_k          [H, H]
    //   H + 2*H^2                            : W_v          [H, H]
    //   H + 3*H^2                            : W_o          [H, H]
    //   H + 4*H^2                            : norm2_gamma  [H]
    //   2*H + 4*H^2                          : W_i_0        [H, FFN]   (GeGLU gate path, GELU activated)
    //   2*H + 4*H^2 + H*FFN                  : W_i_1        [H, FFN]   (GeGLU value path, linear)
    //   2*H + 4*H^2 + 2*H*FFN                : W_o_ffn      [FFN, H]
    // Total per layer: 4*H^2 + 3*H*FFN + 2*H elements.
    private Tensor<T>[]? _qWeightTensors;
    private Tensor<T>[]? _kWeightTensors;
    private Tensor<T>[]? _vWeightTensors;
    private Tensor<T>[]? _attnOutWeightTensors;
    private Tensor<T>[]? _ffnGateWeightTensors;   // W_i_0 in GeGLU
    private Tensor<T>[]? _ffnValueWeightTensors;  // W_i_1 in GeGLU
    private Tensor<T>[]? _ffnOutWeightTensors;    // W_o (FFN down-projection)

    /// <summary>
    /// Cached relative-position-bias broadcast for a specific seqLen, shape [NumHeads, seqLen, seqLen].
    /// Computed once per unique seqLen via T5 bucketing and reused on subsequent calls.
    /// Invalidated together with the weight tensors when <see cref="InvalidateCachedWeightTensors"/>
    /// is called (so a mutated <see cref="_relativePositionBias"/> is picked up correctly).
    /// Capacity is bounded by <see cref="MaxRpbCacheEntries"/> with simple LRU eviction —
    /// each tensor is O(NumHeads * seqLen^2) which can be very large for T5-XXL (64 heads,
    /// seqLen=256 → ~4M elements ≈ 16MB per entry in float32). An unbounded cache would
    /// retain hundreds of MB across workloads with variable sequence lengths.
    /// </summary>
    private readonly Dictionary<int, Tensor<T>> _rpbBroadcastCache = new();

    /// <summary>
    /// LRU eviction order for <see cref="_rpbBroadcastCache"/>: tracks most-recently-used
    /// seqLen at the tail. Sized small to bound peak memory; in practice 4 distinct
    /// sequence lengths covers most workloads (one common train length, one inference
    /// length, plus a couple of edge cases) without thrashing.
    /// </summary>
    private readonly LinkedList<int> _rpbCacheLruOrder = new();

    /// <summary>
    /// Maximum number of distinct seqLen entries to cache. With T5-XXL each entry can be
    /// tens of MB; this cap keeps the worst-case memory footprint bounded regardless of
    /// how many distinct sequence lengths a workload feeds the encoder.
    /// </summary>
    private const int MaxRpbCacheEntries = 4;

    /// <summary>
    /// Cached ones tensor [HiddenSize, 1] for RMS norm sum reduction.
    /// </summary>
    private Tensor<T>? _rmsOnesTensor;

    /// <summary>
    /// Cached epsilon tensor [1, 1] for RMS norm numerical stability.
    /// </summary>
    private Tensor<T>? _rmsEpsTensor;

    /// <summary>
    /// Gets whether this module produces pooled output (T5 does not).
    /// </summary>
    public override bool ProducesPooledOutput => false;

    /// <summary>
    /// Initializes a new T5 text encoder conditioning module.
    /// </summary>
    /// <param name="variant">
    /// T5 variant to use:
    /// - "T5-XXL": 4096-dim, 24 layers (used in Imagen, SD3, FLUX.1)
    /// - "T5-XL": 2048-dim, 24 layers
    /// - "T5-Large": 1024-dim, 24 layers
    /// Default: "T5-XXL"
    /// </param>
    /// <param name="maxSequenceLength">Maximum sequence length. Default: 256.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <example>
    /// <code>
    /// // Create T5-XXL for SD3/FLUX.1
    /// var t5 = new T5TextConditioner&lt;float&gt;();
    ///
    /// // Create T5-XL for memory-constrained setups
    /// var t5xl = new T5TextConditioner&lt;float&gt;(variant: "T5-XL");
    /// </code>
    /// </example>
    public T5TextConditioner(string variant = "T5-XXL", int maxSequenceLength = 256, int? seed = null)
        : base(
            vocabSize: 32128, // T5 SentencePiece vocabulary size
            embeddingDimension: GetEmbeddingDim(variant),
            hiddenSize: GetHiddenSize(variant),
            numLayers: GetNumLayers(variant),
            numHeads: GetNumHeads(variant),
            maxSequenceLength: maxSequenceLength,
            seed: seed)
    {
        _variant = variant;
        _numBuckets = 32;
        _ffnDim = GetFfnDim(variant);

        // Relative position bias: [numHeads, numBuckets] — shared across all encoder layers
        // per the original T5 reference implementation.
        _relativePositionBias = InitializeWeights(NumHeads * _numBuckets);

        // The base class allocated TransformerWeights using its generic 12*H^2 + 4*H per-layer
        // budget. Reallocate using the actual T5.1.1 budget so each layer gets the right
        // amount of storage for Q/K/V/O attention plus the GeGLU FFN's three matrices and two
        // RMS norm gammas. With T5-XXL's d_ff=10240, this is roughly the same total size as
        // the old generic allocation (~190M params/layer for T5-XXL), but with a layout that
        // lets us slice each weight matrix to its correct shape rather than reading a single
        // truncated H^2 block.
        TransformerWeights = InitializeWeights(NumLayers * GetWeightsPerLayer(HiddenSize, _ffnDim));
    }

    /// <summary>
    /// Per-layer storage budget for the T5.1.1 weight blob:
    /// 4*H^2 attention (Q, K, V, O) + 3*H*FFN GeGLU FFN (gate, value, down) + 2*H norm gammas.
    /// </summary>
    private static int GetWeightsPerLayer(int hiddenSize, int ffnDim)
        => 4 * hiddenSize * hiddenSize + 3 * hiddenSize * ffnDim + 2 * hiddenSize;

    /// <summary>
    /// Pre-builds per-layer weight matrix tensors from the flat TransformerWeights vector.
    /// Called lazily on first EncodeText call (after which the cache is reused) and
    /// after <see cref="InvalidateCachedWeightTensors"/> is invoked to drop stale views.
    /// </summary>
    private void EnsureWeightTensorsBuilt()
    {
        if (_qWeightTensors != null) return;

        int H = HiddenSize;
        int F = _ffnDim;
        int hSq = H * H;
        int hF = H * F;
        int weightsPerLayer = GetWeightsPerLayer(H, F);

        _qWeightTensors = new Tensor<T>[NumLayers];
        _kWeightTensors = new Tensor<T>[NumLayers];
        _vWeightTensors = new Tensor<T>[NumLayers];
        _attnOutWeightTensors = new Tensor<T>[NumLayers];
        _ffnGateWeightTensors = new Tensor<T>[NumLayers];
        _ffnValueWeightTensors = new Tensor<T>[NumLayers];
        _ffnOutWeightTensors = new Tensor<T>[NumLayers];

        for (int layer = 0; layer < NumLayers; layer++)
        {
            int layerOffset = layer * weightsPerLayer;

            // Skip norm1_gamma (read directly by RMSNormEngine in the forward pass).
            int qOffset    = layerOffset + H;
            int kOffset    = qOffset + hSq;
            int vOffset    = kOffset + hSq;
            int oOffset    = vOffset + hSq;
            // Skip norm2_gamma.
            int wi0Offset  = oOffset + hSq + H;
            int wi1Offset  = wi0Offset + hF;
            int woOffset   = wi1Offset + hF;

            _qWeightTensors[layer]        = SliceMatrixTensor(qOffset,   H, H);
            _kWeightTensors[layer]        = SliceMatrixTensor(kOffset,   H, H);
            _vWeightTensors[layer]        = SliceMatrixTensor(vOffset,   H, H);
            _attnOutWeightTensors[layer]  = SliceMatrixTensor(oOffset,   H, H);
            _ffnGateWeightTensors[layer]  = SliceMatrixTensor(wi0Offset, H, F);
            _ffnValueWeightTensors[layer] = SliceMatrixTensor(wi1Offset, H, F);
            _ffnOutWeightTensors[layer]   = SliceMatrixTensor(woOffset,  F, H);
        }
    }

    /// <summary>
    /// Slices a <c>[rows, cols]</c> matrix tensor out of <see cref="TransformerWeights"/>
    /// at the given offset, padding with zeros if the source ran short (defensive against
    /// any caller that under-allocated TransformerWeights).
    /// </summary>
    private Tensor<T> SliceMatrixTensor(int offset, int rows, int cols)
    {
        int needed = rows * cols;
        int safeSize = Math.Max(0, Math.Min(needed, TransformerWeights.Length - offset));
        var data = new Vector<T>(needed);
        if (safeSize > 0)
        {
            var src = TransformerWeights.AsSpan().Slice(offset, safeSize);
            for (int i = 0; i < safeSize; i++) data[i] = src[i];
        }
        return new Tensor<T>(new[] { rows, cols }, data);
    }

    /// <summary>
    /// Drops every cached tensor view of <see cref="TransformerWeights"/> and
    /// <see cref="_relativePositionBias"/>. Call this from any code path that mutates the
    /// underlying parameter vectors (e.g., custom <c>SetParameters</c> overrides, checkpoint
    /// loading, or quantization-aware training updates) to prevent the cached tensors from
    /// silently serving stale weights.
    /// </summary>
    /// <remarks>
    /// The cached weight tensors are rebuilt by copying slices of <see cref="TransformerWeights"/>
    /// (because <c>Vector&lt;T&gt;</c> does not currently expose a view constructor). Without
    /// invalidation, parameter updates would leave one half of each layer (the cached attn/FFN
    /// matmul) reading the old weights while the RMS norm path reads the live vector — a subtle
    /// correctness bug noted in PR review.
    /// </remarks>
    public void InvalidateCachedWeightTensors()
    {
        _qWeightTensors = null;
        _kWeightTensors = null;
        _vWeightTensors = null;
        _attnOutWeightTensors = null;
        _ffnGateWeightTensors = null;
        _ffnValueWeightTensors = null;
        _ffnOutWeightTensors = null;
        _rpbBroadcastCache.Clear();
        _rpbCacheLruOrder.Clear();
    }

    /// <inheritdoc />
    public override Tensor<T> Encode(Tensor<T> input)
    {
        return EncodeText(input);
    }

    /// <inheritdoc />
    public override Tensor<T> EncodeText(Tensor<T> tokenIds, Tensor<T>? attentionMask = null)
    {
        EnsureWeightTensorsBuilt();

        var shape = tokenIds._shape;
        int batchSize = shape[0];
        int seqLen = shape.Length > 1 ? shape[1] : MaxSequenceLength;
        int totalTokens = batchSize * (shape.Length > 1 ? shape[1] : 1);

        // Token embedding lookup: build hidden tensor [batchSize, seqLen, HiddenSize].
        // (Per-token gather is inherently scalar; this is the T5 input embedding layer.)
        var hiddenData = new Vector<T>(batchSize * seqLen * HiddenSize);
        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int flatIdx = b * seqLen + s;
                int tokenId = flatIdx < totalTokens
                    ? Math.Max(0, Math.Min((int)NumOps.ToDouble(tokenIds[flatIdx]), VocabSize - 1))
                    : 0;
                int srcOff = tokenId * HiddenSize;
                int dstOff = (b * seqLen + s) * HiddenSize;
                for (int d = 0; d < HiddenSize; d++)
                    hiddenData[dstOff + d] = TokenEmbeddings[srcOff + d];
            }
        }

        var hidden = new Tensor<T>(new[] { batchSize, seqLen, HiddenSize }, hiddenData);

        // Run all encoder layers fully batched over [B, S, H] — no per-batch Python-style loop.
        hidden = ApplyT5EncoderLayersBatched(hidden, batchSize, seqLen);

        // T5 final RMS norm applied across the trailing HiddenSize axis.
        hidden = RMSNormEngine(hidden, FinalLayerNormWeights, HiddenSize);

        // For all current variants EmbeddingDimension == HiddenSize, so the encoder output
        // is already the desired shape. Reshape (no-copy) only if a future variant decouples them.
        if (EmbeddingDimension == HiddenSize)
            return hidden;

        // Defensive path: if a variant ever sets EmbeddingDimension != HiddenSize, project / pad
        // by simple element copy. Avoids hiding a configuration mismatch at runtime.
        var outputData = new Vector<T>(batchSize * seqLen * EmbeddingDimension);
        int copyDim = Math.Min(HiddenSize, EmbeddingDimension);
        for (int b = 0; b < batchSize; b++)
        for (int s = 0; s < seqLen; s++)
        for (int d = 0; d < copyDim; d++)
            outputData[(b * seqLen + s) * EmbeddingDimension + d] = hidden[(b * seqLen + s) * HiddenSize + d];
        return new Tensor<T>(new[] { batchSize, seqLen, EmbeddingDimension }, outputData);
    }

    /// <inheritdoc />
    public override Tensor<T> GetPooledEmbedding(Tensor<T> sequenceEmbeddings)
    {
        // T5 doesn't produce pooled output - use mean pooling via Engine
        var shape = sequenceEmbeddings._shape;
        int batchSize = shape[0];
        int seqLen = shape[1];

        // Mean pool: sum across sequence dimension, then divide
        var pooledData = new Vector<T>(batchSize * EmbeddingDimension);

        // Hoist constant ones tensor outside the batch loop
        var onesForMean = new Tensor<T>(new[] { 1, seqLen });
        var onesSpan = onesForMean.AsWritableSpan();
        for (int i = 0; i < seqLen; i++) onesSpan[i] = NumOps.One;

        for (int b = 0; b < batchSize; b++)
        {
            // Build a [seqLen, embDim] tensor for this batch
            int batchOff = b * seqLen * EmbeddingDimension;
            var batchVec = new Vector<T>(sequenceEmbeddings.AsSpan().Slice(batchOff, seqLen * EmbeddingDimension).ToArray());
            var batchTensor = new Tensor<T>(new[] { seqLen, EmbeddingDimension }, batchVec);

            // [1, seqLen] @ [seqLen, embDim] -> [1, embDim]
            var summed = Engine.TensorMatMul(onesForMean, batchTensor);

            // Divide by seqLen
            var meanTensor = Engine.TensorDivideScalar(summed, NumOps.FromDouble(seqLen));

            // Copy result
            for (int d = 0; d < EmbeddingDimension; d++)
                pooledData[b * EmbeddingDimension + d] = meanTensor[d];
        }

        return new Tensor<T>(new[] { batchSize, EmbeddingDimension }, pooledData);
    }

    /// <inheritdoc />
    public override Tensor<T> GetUnconditionalEmbedding(int batchSize)
    {
        // Empty input: just the padding/EOS token
        var tokenIds = new Vector<T>(batchSize * MaxSequenceLength);
        for (int b = 0; b < batchSize; b++)
        {
            tokenIds[b * MaxSequenceLength] = NumOps.FromDouble(1); // EOS/pad token
        }

        var input = new Tensor<T>(new[] { batchSize, MaxSequenceLength }, tokenIds);
        return EncodeText(input);
    }

    /// <inheritdoc />
    public override Tensor<T> Tokenize(string text)
    {
        var tokens = SimpleTokenize(text, MaxSequenceLength);
        var tokenData = new Vector<T>(MaxSequenceLength);
        for (int i = 0; i < MaxSequenceLength; i++)
            tokenData[i] = NumOps.FromDouble(tokens[i]);

        return new Tensor<T>(new[] { 1, MaxSequenceLength }, tokenData);
    }

    /// <inheritdoc />
    public override Tensor<T> TokenizeBatch(string[] texts)
    {
        var tokenData = new Vector<T>(texts.Length * MaxSequenceLength);
        for (int b = 0; b < texts.Length; b++)
        {
            var tokens = SimpleTokenize(texts[b], MaxSequenceLength);
            for (int i = 0; i < MaxSequenceLength; i++)
                tokenData[b * MaxSequenceLength + i] = NumOps.FromDouble(tokens[i]);
        }

        return new Tensor<T>(new[] { texts.Length, MaxSequenceLength }, tokenData);
    }

    /// <summary>
    /// Paper-faithful T5 encoder forward pass, fully batched over <c>[B, S, H]</c>.
    /// </summary>
    /// <remarks>
    /// <para>Per-layer structure (Raffel et al. 2020, §3.1.1; Shazeer 2020 for GeGLU):</para>
    /// <list type="number">
    /// <item>Pre-norm self-attention: <c>RMSNorm → Q/K/V projections → multi-head attention with
    /// shared learned relative position bias added to scores → output projection → residual</c>.
    /// Attention is unscaled (T5 explicitly omits 1/√d_k) and bidirectional (encoder).</item>
    /// <item>Pre-norm GeGLU FFN: <c>RMSNorm → (GELU(x · W_i_0) ⊙ (x · W_i_1)) · W_o → residual</c>.
    /// GeGLU replaces the original ReLU FFN per Shazeer 2020.</item>
    /// </list>
    /// <para>Attention uses <see cref="FlashAttention{T}"/> with <c>ScaleFactor=1.0</c> and the T5
    /// relative position bias passed via <c>attentionBias</c>; FlashAttention adds it to scores
    /// before the (online) softmax — exactly where T5 places it.</para>
    /// </remarks>
    private Tensor<T> ApplyT5EncoderLayersBatched(Tensor<T> hidden, int batchSize, int seqLen)
    {
        EnsureWeightTensorsBuilt();

        int H = HiddenSize;
        int F = _ffnDim;
        if (H % NumHeads != 0)
        {
            throw new InvalidOperationException(
                $"HiddenSize ({H}) must be divisible by NumHeads ({NumHeads}). Variant '{_variant}' has an invalid configuration.");
        }
        int headDim = H / NumHeads;
        int weightsPerLayer = GetWeightsPerLayer(H, F);
        int totalTokens = batchSize * seqLen;

        // Build the relative-position-bias tensor once for this seqLen and reuse for every
        // layer — T5 shares one learned bias matrix [NumHeads, NumBuckets] across all layers.
        // Cached per seqLen; invalidated by InvalidateCachedWeightTensors when bias mutates.
        var rpb = GetOrBuildRelativePositionBias(seqLen);  // [NumHeads, S, S]

        // Configure FlashAttention for T5 encoder: no scaling, bidirectional, no dropout.
        var flashConfig = new FlashAttentionConfig
        {
            ScaleFactor = 1.0f,
            UseCausalMask = false,
            ReturnAttentionWeights = false,
            DropoutProbability = 0.0f,
        };

        for (int layer = 0; layer < NumLayers; layer++)
        {
            int layerOffset = layer * weightsPerLayer;

            // ===== Self-attention block (pre-norm + residual) =====
            var residual = hidden;
            var norm1Gamma = ExtractSubVectorFast(TransformerWeights, layerOffset, H);
            var normed = RMSNormEngine(hidden, norm1Gamma, H);

            // Project Q, K, V via 2D matmul (collapse batch and seq into one axis).
            var normed2D = Engine.Reshape(normed, new[] { totalTokens, H });
            var qFlat = Engine.TensorMatMul(normed2D, _qWeightTensors![layer]);  // [B*S, H]
            var kFlat = Engine.TensorMatMul(normed2D, _kWeightTensors![layer]);
            var vFlat = Engine.TensorMatMul(normed2D, _vWeightTensors![layer]);

            // Split heads: [B*S, H] → [B, S, NumHeads, HeadDim] → [B, NumHeads, S, HeadDim]
            var qSplit = Engine.Reshape(qFlat, new[] { batchSize, seqLen, NumHeads, headDim });
            var kSplit = Engine.Reshape(kFlat, new[] { batchSize, seqLen, NumHeads, headDim });
            var vSplit = Engine.Reshape(vFlat, new[] { batchSize, seqLen, NumHeads, headDim });
            var q4D = Engine.TensorPermute(qSplit, new[] { 0, 2, 1, 3 });
            var k4D = Engine.TensorPermute(kSplit, new[] { 0, 2, 1, 3 });
            var v4D = Engine.TensorPermute(vSplit, new[] { 0, 2, 1, 3 });

            // FlashAttention: softmax(Q @ K^T + RPB) @ V, with our scale=1.0 override (T5).
            var (attnOut4D, _) = FlashAttention<T>.Forward(
                q4D, k4D, v4D, flashConfig, queryOffset: 0, attentionBias: rpb);
            // attnOut4D: [B, NumHeads, S, HeadDim]

            // Combine heads: [B, NumHeads, S, HeadDim] → [B, S, NumHeads, HeadDim] → [B*S, H]
            var attnPerm = Engine.TensorPermute(attnOut4D, new[] { 0, 2, 1, 3 });
            var attnFlat = Engine.Reshape(attnPerm, new[] { totalTokens, H });

            // Output projection W_o.
            var attnProjFlat = Engine.TensorMatMul(attnFlat, _attnOutWeightTensors![layer]);
            var attnProj = Engine.Reshape(attnProjFlat, new[] { batchSize, seqLen, H });

            hidden = Engine.TensorAdd(residual, attnProj);

            // ===== Feed-forward block (pre-norm + residual + GeGLU) =====
            residual = hidden;
            int norm2Offset = layerOffset + H + 4 * H * H;
            var norm2Gamma = ExtractSubVectorFast(TransformerWeights, norm2Offset, H);
            normed = RMSNormEngine(hidden, norm2Gamma, H);
            normed2D = Engine.Reshape(normed, new[] { totalTokens, H });

            // GeGLU: gate = GELU(x @ W_i_0), value = x @ W_i_1, fused = gate ⊙ value, out = fused @ W_o
            var gateProj = Engine.TensorMatMul(normed2D, _ffnGateWeightTensors![layer]);   // [B*S, F]
            var valueProj = Engine.TensorMatMul(normed2D, _ffnValueWeightTensors![layer]); // [B*S, F]
            var gateActivated = Engine.GELU(gateProj);                                      // GELU per Shazeer
            var fused = Engine.TensorMultiply(gateActivated, valueProj);                    // elementwise gating
            var ffnOutFlat = Engine.TensorMatMul(fused, _ffnOutWeightTensors![layer]);      // [B*S, H]
            var ffnOut = Engine.Reshape(ffnOutFlat, new[] { batchSize, seqLen, H });

            hidden = Engine.TensorAdd(residual, ffnOut);
        }

        return hidden;
    }

    /// <summary>
    /// Returns (and caches per <paramref name="seqLen"/>) the T5 relative-position-bias tensor
    /// of shape <c>[NumHeads, seqLen, seqLen]</c>, ready to be added to attention scores.
    /// The underlying <see cref="_relativePositionBias"/> is shared across all layers per the
    /// T5 reference implementation.
    /// </summary>
    private Tensor<T> GetOrBuildRelativePositionBias(int seqLen)
    {
        if (_rpbBroadcastCache.TryGetValue(seqLen, out var cached))
        {
            // Hit — bump to MRU position so it survives the next eviction.
            _rpbCacheLruOrder.Remove(seqLen);
            _rpbCacheLruOrder.AddLast(seqLen);
            return cached;
        }

        const int maxDistance = 128; // T5 default
        var data = new Vector<T>(NumHeads * seqLen * seqLen);

        // Loop order: q outermost, k inner — same as the reference implementation.
        // For each (q, k) pair, the bucket is the same across heads, so compute once and reuse.
        for (int q = 0; q < seqLen; q++)
        {
            for (int k = 0; k < seqLen; k++)
            {
                int bucket = RelativePositionBucket(
                    relativePosition: k - q,
                    bidirectional: true,
                    numBuckets: _numBuckets,
                    maxDistance: maxDistance);

                for (int h = 0; h < NumHeads; h++)
                {
                    int rpbIdx = h * _numBuckets + bucket;
                    int outIdx = h * seqLen * seqLen + q * seqLen + k;
                    data[outIdx] = _relativePositionBias[rpbIdx];
                }
            }
        }

        var tensor = new Tensor<T>(new[] { NumHeads, seqLen, seqLen }, data);

        // LRU eviction before insert — evict the LEAST-recently-used seqLen entry when at
        // capacity. Without this, workloads that feed varying sequence lengths (e.g.,
        // dynamic-batch inference) would grow the cache unboundedly and retain potentially
        // hundreds of MB worth of RPB tensors.
        if (_rpbBroadcastCache.Count >= MaxRpbCacheEntries && _rpbCacheLruOrder.First is { } victim)
        {
            _rpbBroadcastCache.Remove(victim.Value);
            _rpbCacheLruOrder.RemoveFirst();
        }
        _rpbBroadcastCache[seqLen] = tensor;
        _rpbCacheLruOrder.AddLast(seqLen);
        return tensor;
    }

    /// <summary>
    /// T5 bidirectional relative-position bucketing function (Raffel et al. 2020).
    /// Half of the buckets handle exact positions in <c>[0, numBuckets/4)</c>, the other half
    /// log-spaced positions out to <paramref name="maxDistance"/>; for bidirectional encoders
    /// the full bucket range is split in two — one half for forward, one half for backward.
    /// </summary>
    /// <remarks>
    /// Mirrors the canonical implementation in the T5/HuggingFace codebase. Both directions
    /// are necessary because T5's encoder attention is bidirectional, so the model must
    /// distinguish "key is N tokens ahead" from "key is N tokens behind".
    /// </remarks>
    private static int RelativePositionBucket(int relativePosition, bool bidirectional, int numBuckets, int maxDistance)
    {
        int bucket = 0;
        int n = numBuckets;
        int relPos;
        if (bidirectional)
        {
            n = numBuckets / 2;
            if (relativePosition > 0) bucket += n; // direction bit: forward = upper half
            relPos = Math.Abs(relativePosition);
        }
        else
        {
            relPos = -Math.Min(relativePosition, 0); // decoder: only attend to past
        }

        int maxExact = n / 2;
        if (relPos < maxExact)
        {
            bucket += relPos;
        }
        else
        {
            // Logarithmic bucketing for distant positions:
            //   bucket = maxExact + log(relPos / maxExact) / log(maxDistance / maxExact) * (n - maxExact)
            double ratio = Math.Log((double)relPos / maxExact) / Math.Log((double)maxDistance / maxExact);
            int relPosIfLarge = maxExact + (int)(ratio * (n - maxExact));
            if (relPosIfLarge > n - 1) relPosIfLarge = n - 1;
            bucket += relPosIfLarge;
        }
        return bucket;
    }

    /// <summary>
    /// Engine-accelerated RMS normalization.
    /// Uses Engine.TensorMultiply for element-wise square, Engine.TensorMatMul for sum reduction,
    /// and Engine.TensorBroadcastDivide/Multiply for normalize+scale.
    /// </summary>
    private Tensor<T> RMSNormEngine(Tensor<T> input, Vector<T> gamma, int dim)
    {
        var shape = input._shape;
        int numVectors = input.Length / dim;

        // Element-wise square via Engine
        var squared = Engine.TensorMultiply(input, input);

        // Sum across last dimension: [numVectors, dim] @ [dim, 1] -> [numVectors, 1]
        // Cache the ones tensor to avoid reallocating on every call
        if (_rmsOnesTensor == null || _rmsOnesTensor.Length != dim)
        {
            var onesVec = new Vector<T>(dim);
            for (int i = 0; i < dim; i++) onesVec[i] = NumOps.One;
            _rmsOnesTensor = new Tensor<T>(new[] { dim, 1 }, onesVec);
        }
        var onesTensor = _rmsOnesTensor;

        var reshapedSquared = squared.Reshape(numVectors, dim);
        var sumSq = Engine.TensorMatMul(reshapedSquared, onesTensor); // [numVectors, 1]

        // mean = sumSq / dim
        var mean = Engine.TensorDivideScalar(sumSq, NumOps.FromDouble(dim));

        // rms = sqrt(mean + eps)
        _rmsEpsTensor ??= new Tensor<T>(new[] { 1, 1 }, new Vector<T>(new[] { NumOps.FromDouble(1e-6) }));
        var meanPlusEps = Engine.TensorBroadcastAdd(mean, _rmsEpsTensor);
        var rms = Engine.TensorSqrt(meanPlusEps); // [numVectors, 1]

        // normalized = input / rms (broadcast along last dim)
        var inputReshaped = input.Reshape(numVectors, dim);
        var normalized = Engine.TensorBroadcastDivide(inputReshaped, rms);

        // scaled = normalized * gamma (broadcast along first dim)
        var gammaTensor = new Tensor<T>(new[] { 1, dim }, gamma);
        var scaled = Engine.TensorBroadcastMultiply(normalized, gammaTensor);

        return scaled.Reshape(shape);
    }

    /// <summary>
    /// Fast sub-vector extraction using Span slice (avoids scalar element copy loop).
    /// </summary>
    private static Vector<T> ExtractSubVectorFast(Vector<T> source, int offset, int length)
    {
        if (offset < 0)
            throw new ArgumentOutOfRangeException(nameof(offset), "Offset must be non-negative.");
        int safeLength = Math.Min(length, source.Length - offset);
        if (safeLength <= 0)
            return new Vector<T>(length);
        return new Vector<T>(source.AsSpan().Slice(offset, safeLength).ToArray());
    }

    #region Variant Configuration

    private static int GetEmbeddingDim(string variant) => variant switch
    {
        "T5-XL" => 2048,
        "T5-Large" => 1024,
        _ => 4096 // T5-XXL default
    };

    private static int GetHiddenSize(string variant) => variant switch
    {
        "T5-XL" => 2048,
        "T5-Large" => 1024,
        _ => 4096 // T5-XXL
    };

    private static int GetNumLayers(string variant) => variant switch
    {
        "T5-Large" => 24,
        _ => 24 // T5-XXL and T5-XL both have 24 layers
    };

    private static int GetNumHeads(string variant) => variant switch
    {
        "T5-XL" => 32,
        "T5-Large" => 16,
        _ => 64 // T5-XXL
    };

    /// <summary>
    /// Returns the FFN inner dimension (d_ff) per T5.1.1 official configs.
    /// These ratios are not a clean multiple of HiddenSize because Google tuned them
    /// per-variant rather than as a fixed expansion factor.
    /// </summary>
    private static int GetFfnDim(string variant) => variant switch
    {
        "T5-XL" => 5120,    // T5.1.1-XL  (d_model=2048, d_ff=5120)
        "T5-Large" => 2816, // T5.1.1-Large (d_model=1024, d_ff=2816)
        _ => 10240          // T5.1.1-XXL (d_model=4096, d_ff=10240)
    };

    #endregion
}
