using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

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
/// <b>Reference:</b> Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", JMLR 2020
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
    /// Pre-built weight matrix tensors for each layer (avoids rebuilding per call).
    /// Lazily initialized on first EncodeText call.
    /// </summary>
    private Tensor<T>[]? _attnWeightTensors;
    private Tensor<T>[]? _ffnWeightTensors;

    /// <summary>
    /// Cached ones tensor [HiddenSize, 1] for RMS norm sum reduction.
    /// </summary>
    private Tensor<T>? _rmsOnesTensor;

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

        // Relative position bias: [numHeads, numBuckets]
        _relativePositionBias = InitializeWeights(NumHeads * _numBuckets);
    }

    /// <summary>
    /// Pre-builds weight matrix tensors from the flat TransformerWeights vector.
    /// Called once, then reused across all EncodeText calls.
    /// </summary>
    private void EnsureWeightTensorsBuilt()
    {
        if (_attnWeightTensors != null) return;

        int weightsPerLayer = 12 * HiddenSize * HiddenSize + 4 * HiddenSize;
        _attnWeightTensors = new Tensor<T>[NumLayers];
        _ffnWeightTensors = new Tensor<T>[NumLayers];

        for (int layer = 0; layer < NumLayers; layer++)
        {
            int layerOffset = layer * weightsPerLayer;

            // Attention weight matrix [HiddenSize, HiddenSize]
            int attnOffset = layerOffset + HiddenSize;
            int weightSize = HiddenSize * HiddenSize;
            int safeSize = Math.Min(weightSize, TransformerWeights.Length - attnOffset);
            var attnData = safeSize > 0
                ? new Vector<T>(TransformerWeights.AsSpan().Slice(attnOffset, safeSize).ToArray())
                : new Vector<T>(weightSize);
            if (safeSize < weightSize)
            {
                var padded = new Vector<T>(weightSize);
                for (int i = 0; i < safeSize; i++) padded[i] = attnData[i];
                attnData = padded;
            }
            _attnWeightTensors[layer] = new Tensor<T>(new[] { HiddenSize, HiddenSize }, attnData);

            // FFN weight matrix [HiddenSize, HiddenSize]
            int ffnNormOffset = layerOffset + HiddenSize + HiddenSize * HiddenSize;
            int ffnOffset = ffnNormOffset + HiddenSize;
            safeSize = Math.Min(weightSize, TransformerWeights.Length - ffnOffset);
            var ffnData = safeSize > 0
                ? new Vector<T>(TransformerWeights.AsSpan().Slice(ffnOffset, safeSize).ToArray())
                : new Vector<T>(weightSize);
            if (safeSize < weightSize)
            {
                var padded = new Vector<T>(weightSize);
                for (int i = 0; i < safeSize; i++) padded[i] = ffnData[i];
                ffnData = padded;
            }
            _ffnWeightTensors[layer] = new Tensor<T>(new[] { HiddenSize, HiddenSize }, ffnData);
        }
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

        // Token embedding lookup: build hidden tensor [batchSize, seqLen, HiddenSize]
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

        // Process each batch through transformer layers using Engine-accelerated matmul
        var outputData = new Vector<T>(batchSize * seqLen * EmbeddingDimension);

        for (int b = 0; b < batchSize; b++)
        {
            // Extract batch slice as 2D tensor [seqLen, HiddenSize]
            var batchVec = new Vector<T>(hiddenData.AsSpan().Slice(b * seqLen * HiddenSize, seqLen * HiddenSize).ToArray());
            var hidden = new Tensor<T>(new[] { seqLen, HiddenSize }, batchVec);

            // Apply transformer encoder layers (Engine-accelerated matmul + GELU)
            hidden = ApplyT5EncoderLayersEngine(hidden);

            // Apply final RMS norm
            hidden = RMSNormEngine(hidden, FinalLayerNormWeights, HiddenSize);

            // Copy output
            int dstOff = b * seqLen * EmbeddingDimension;
            for (int s = 0; s < seqLen; s++)
            {
                int srcBase = s * HiddenSize;
                int dstBase = dstOff + s * EmbeddingDimension;
                for (int d = 0; d < EmbeddingDimension; d++)
                    outputData[dstBase + d] = hidden[srcBase + d];
            }
        }

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

        for (int b = 0; b < batchSize; b++)
        {
            // Build a [seqLen, embDim] tensor for this batch
            int batchOff = b * seqLen * EmbeddingDimension;
            var batchVec = new Vector<T>(sequenceEmbeddings.AsSpan().Slice(batchOff, seqLen * EmbeddingDimension).ToArray());
            var batchTensor = new Tensor<T>(new[] { seqLen, EmbeddingDimension }, batchVec);

            // Sum across seqLen using Engine: create ones vector [1, seqLen]
            var ones = new Tensor<T>(new[] { 1, seqLen });
            var onesSpan = ones.AsWritableSpan();
            for (int i = 0; i < seqLen; i++) onesSpan[i] = NumOps.One;

            // [1, seqLen] @ [seqLen, embDim] -> [1, embDim]
            var summed = Engine.TensorMatMul(ones, batchTensor);

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
    /// Engine-accelerated T5 encoder transformer layers.
    /// Uses Engine.TensorMatMul for linear projections (replaces O(seq*dim*dim) scalar loops)
    /// and Engine.GELU for activation (replaces O(seq*dim) scalar loops).
    /// </summary>
    private Tensor<T> ApplyT5EncoderLayersEngine(Tensor<T> hidden)
    {
        int weightsPerLayer = 12 * HiddenSize * HiddenSize + 4 * HiddenSize;

        for (int layer = 0; layer < NumLayers; layer++)
        {
            int layerOffset = layer * weightsPerLayer;

            // T5 uses pre-norm: RMSNorm -> attention -> residual
            // Save residual via Engine.TensorAdd with a zero tensor (creates a copy)
            var residual = Engine.TensorBroadcastAdd(hidden,
                new Tensor<T>(new[] { 1, HiddenSize })); // broadcast add zero = copy

            // RMS Norm 1
            var rmsGamma = ExtractSubVectorFast(TransformerWeights, layerOffset, HiddenSize);
            hidden = RMSNormEngine(hidden, rmsGamma, HiddenSize);

            // Self-attention via Engine.TensorMatMul: [seqLen, H] @ [H, H] -> [seqLen, H]
            hidden = Engine.TensorMatMul(hidden, _attnWeightTensors![layer]);

            // Residual connection
            hidden = Engine.TensorAdd(hidden, residual);

            // Save residual for FFN
            residual = Engine.TensorBroadcastAdd(hidden,
                new Tensor<T>(new[] { 1, HiddenSize }));

            // RMS Norm 2
            int ffnNormOffset = layerOffset + HiddenSize + HiddenSize * HiddenSize;
            var ffnGamma = ExtractSubVectorFast(TransformerWeights, ffnNormOffset, HiddenSize);
            hidden = RMSNormEngine(hidden, ffnGamma, HiddenSize);

            // FFN via Engine.TensorMatMul: [seqLen, H] @ [H, H] -> [seqLen, H]
            hidden = Engine.TensorMatMul(hidden, _ffnWeightTensors![layer]);

            // GELU activation via Engine (hardware-accelerated SIMD)
            hidden = Engine.GELU(hidden);

            // Residual connection
            hidden = Engine.TensorAdd(hidden, residual);
        }

        return hidden;
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
        var epsTensor = new Tensor<T>(new[] { 1, 1 }, new Vector<T>(new[] { NumOps.FromDouble(1e-6) }));
        var meanPlusEps = Engine.TensorBroadcastAdd(mean, epsTensor);
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

    #endregion
}
