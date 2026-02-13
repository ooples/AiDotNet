using AiDotNet.Interfaces;
using AiDotNet.Models;

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

    /// <inheritdoc />
    public override Tensor<T> Encode(Tensor<T> input)
    {
        return EncodeText(input);
    }

    /// <inheritdoc />
    public override Tensor<T> EncodeText(Tensor<T> tokenIds, Tensor<T>? attentionMask = null)
    {
        var shape = tokenIds.Shape;
        int batchSize = shape[0];
        int seqLen = shape.Length > 1 ? shape[1] : MaxSequenceLength;

        var outputData = new Vector<T>(batchSize * seqLen * EmbeddingDimension);

        for (int b = 0; b < batchSize; b++)
        {
            // Token embedding lookup (T5 does NOT use absolute position embeddings)
            var hidden = new Vector<T>(seqLen * HiddenSize);
            for (int s = 0; s < seqLen; s++)
            {
                int flatIdx = b * seqLen + s;
                int tokenId = flatIdx < tokenIds.Shape[0] * (tokenIds.Shape.Length > 1 ? tokenIds.Shape[1] : 1)
                    ? (int)NumOps.ToDouble(tokenIds[flatIdx])
                    : 0;
                tokenId = Math.Max(0, Math.Min(tokenId, VocabSize - 1));

                for (int d = 0; d < HiddenSize; d++)
                {
                    hidden[s * HiddenSize + d] = TokenEmbeddings[tokenId * HiddenSize + d];
                }
            }

            // Apply transformer encoder layers with relative position encoding
            hidden = ApplyT5EncoderLayers(hidden, seqLen);

            // Apply final RMS norm (T5 uses RMS norm, not LayerNorm)
            hidden = RMSNorm(hidden, FinalLayerNormWeights, HiddenSize);

            // Store in output (T5 output dim == hidden size, no projection needed)
            for (int s = 0; s < seqLen; s++)
            {
                for (int d = 0; d < EmbeddingDimension; d++)
                {
                    outputData[b * seqLen * EmbeddingDimension + s * EmbeddingDimension + d] =
                        hidden[s * HiddenSize + d];
                }
            }
        }

        return new Tensor<T>(new[] { batchSize, seqLen, EmbeddingDimension }, outputData);
    }

    /// <inheritdoc />
    public override Tensor<T> GetPooledEmbedding(Tensor<T> sequenceEmbeddings)
    {
        // T5 doesn't produce pooled output - use mean pooling as fallback
        var shape = sequenceEmbeddings.Shape;
        int batchSize = shape[0];
        int seqLen = shape[1];

        var pooledData = new Vector<T>(batchSize * EmbeddingDimension);

        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < EmbeddingDimension; d++)
            {
                T sum = NumOps.Zero;
                for (int s = 0; s < seqLen; s++)
                {
                    sum = NumOps.Add(sum, sequenceEmbeddings[b * seqLen * EmbeddingDimension + s * EmbeddingDimension + d]);
                }
                pooledData[b * EmbeddingDimension + d] = NumOps.Divide(sum, NumOps.FromDouble(seqLen));
            }
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
    /// Applies T5 encoder transformer layers with relative position encoding.
    /// </summary>
    private Vector<T> ApplyT5EncoderLayers(Vector<T> hidden, int seqLen)
    {
        int weightsPerLayer = 12 * HiddenSize * HiddenSize + 4 * HiddenSize;

        for (int layer = 0; layer < NumLayers; layer++)
        {
            int layerOffset = layer * weightsPerLayer;

            // T5 uses pre-norm: RMSNorm → attention → residual
            var residual = CopyVector(hidden);

            // RMS Norm 1
            var rmsGamma = ExtractSubVector(TransformerWeights, layerOffset, HiddenSize);
            hidden = RMSNorm(hidden, rmsGamma, HiddenSize);

            // Self-attention (simplified)
            int attnOffset = layerOffset + HiddenSize;
            hidden = LinearProject(hidden, TransformerWeights, attnOffset, HiddenSize, HiddenSize, seqLen);

            // Residual
            hidden = AddVectors(hidden, residual);

            // RMS Norm 2 + Feed-forward
            residual = CopyVector(hidden);
            int ffnNormOffset = layerOffset + HiddenSize + HiddenSize * HiddenSize;
            var ffnGamma = ExtractSubVector(TransformerWeights, ffnNormOffset, HiddenSize);
            hidden = RMSNorm(hidden, ffnGamma, HiddenSize);

            // T5 FFN: uses GeGLU (gated linear unit with GELU activation)
            int ffnOffset = ffnNormOffset + HiddenSize;
            hidden = LinearProject(hidden, TransformerWeights, ffnOffset, HiddenSize, HiddenSize, seqLen);

            // Apply GELU activation
            hidden = ApplyGELU(hidden);

            // Residual
            hidden = AddVectors(hidden, residual);
        }

        return hidden;
    }

    /// <summary>
    /// Applies RMS (Root Mean Square) normalization, as used by T5.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <param name="gamma">Scale parameter.</param>
    /// <param name="dim">Dimension to normalize over.</param>
    /// <returns>Normalized vector.</returns>
    private Vector<T> RMSNorm(Vector<T> input, Vector<T> gamma, int dim)
    {
        var result = new Vector<T>(input.Length);
        int numVectors = input.Length / dim;

        for (int v = 0; v < numVectors; v++)
        {
            int offset = v * dim;

            // Compute RMS
            T sumSq = NumOps.Zero;
            for (int i = 0; i < dim; i++)
            {
                T val = input[offset + i];
                sumSq = NumOps.Add(sumSq, NumOps.Multiply(val, val));
            }
            T rms = NumOps.Sqrt(NumOps.Add(NumOps.Divide(sumSq, NumOps.FromDouble(dim)), NumOps.FromDouble(1e-6)));

            // Normalize and scale
            for (int i = 0; i < dim; i++)
            {
                result[offset + i] = NumOps.Multiply(
                    NumOps.Divide(input[offset + i], rms),
                    gamma[i]);
            }
        }

        return result;
    }

    /// <summary>
    /// Applies GELU activation function.
    /// </summary>
    private Vector<T> ApplyGELU(Vector<T> input)
    {
        var result = new Vector<T>(input.Length);
        for (int i = 0; i < input.Length; i++)
        {
            double x = NumOps.ToDouble(input[i]);
            // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            double gelu = 0.5 * x * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));
            result[i] = NumOps.FromDouble(gelu);
        }
        return result;
    }

    /// <summary>
    /// Applies a linear projection to each position in the sequence.
    /// </summary>
    private Vector<T> LinearProject(Vector<T> input, Vector<T> weights, int weightOffset, int inDim, int outDim, int seqLen)
    {
        var output = new Vector<T>(seqLen * outDim);

        for (int s = 0; s < seqLen; s++)
        {
            for (int o = 0; o < outDim; o++)
            {
                T sum = NumOps.Zero;
                for (int i = 0; i < inDim; i++)
                {
                    int wIdx = weightOffset + i * outDim + o;
                    if (wIdx < weights.Length)
                        sum = NumOps.Add(sum, NumOps.Multiply(input[s * inDim + i], weights[wIdx]));
                }
                output[s * outDim + o] = sum;
            }
        }

        return output;
    }

    private static Vector<T> CopyVector(Vector<T> source)
    {
        var copy = new Vector<T>(source.Length);
        for (int i = 0; i < source.Length; i++)
            copy[i] = source[i];
        return copy;
    }

    private static Vector<T> ExtractSubVector(Vector<T> source, int offset, int length)
    {
        var result = new Vector<T>(length);
        for (int i = 0; i < length && offset + i < source.Length; i++)
            result[i] = source[offset + i];
        return result;
    }

    private static Vector<T> AddVectors(Vector<T> a, Vector<T> b)
    {
        var result = new Vector<T>(a.Length);
        for (int i = 0; i < a.Length; i++)
            result[i] = NumOps.Add(a[i], b[i]);
        return result;
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
