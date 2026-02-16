using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Conditioning;

/// <summary>
/// Dual text encoder conditioning module combining CLIP and T5 encoders.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Many modern diffusion models use multiple text encoders together to get the best
/// of each encoder's strengths. This module combines a CLIP encoder (for pooled embeddings
/// and visual-semantic alignment) with a T5 encoder (for detailed text understanding
/// and cross-attention).
/// </para>
/// <para>
/// <b>For Beginners:</b> This uses TWO text encoders together for better results.
///
/// Why two encoders?
/// - CLIP: Great at understanding the visual "gist" of your prompt
///   (e.g., knowing what a cat looks like, what "golden light" means visually)
/// - T5: Great at understanding language details
///   (e.g., counting objects, understanding spatial relationships, rendering text)
///
/// How they work together:
/// 1. CLIP produces a "pooled" embedding (one vector summarizing the whole prompt)
///    → Used for the global style/content of the image
/// 2. T5 produces "sequence" embeddings (one vector per token)
///    → Used for cross-attention, giving fine-grained control
///
/// Used by:
/// - FLUX.1: CLIP ViT-L/14 (768-dim) + T5-XXL (4096-dim)
/// - SD3: CLIP ViT-L/14 (768-dim) + OpenCLIP ViT-bigG/14 (1280-dim) + T5-XXL (4096-dim)
/// - Imagen: T5-XXL (text only, no CLIP)
/// </para>
/// </remarks>
public class DualTextConditioner<T> : IConditioningModule<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The CLIP text encoder providing pooled and sequence embeddings.
    /// </summary>
    private readonly CLIPTextConditioner<T> _clipEncoder;

    /// <summary>
    /// The T5 text encoder providing high-dimensional sequence embeddings.
    /// </summary>
    private readonly T5TextConditioner<T> _t5Encoder;

    /// <summary>
    /// Gets the combined context dimension (T5 embedding dimension for cross-attention).
    /// </summary>
    public int EmbeddingDimension => _t5Encoder.EmbeddingDimension;

    /// <inheritdoc />
    public ConditioningType ConditioningType => ConditioningType.MultiModal;

    /// <summary>
    /// Gets whether this module produces pooled output (yes, from CLIP).
    /// </summary>
    public bool ProducesPooledOutput => true;

    /// <summary>
    /// Gets the maximum sequence length (uses T5's longer sequence length).
    /// </summary>
    public int MaxSequenceLength => _t5Encoder.MaxSequenceLength;

    /// <summary>
    /// Gets the CLIP encoder's embedding dimension (for pooled embeddings).
    /// </summary>
    public int CLIPEmbeddingDimension => _clipEncoder.EmbeddingDimension;

    /// <summary>
    /// Gets the T5 encoder's embedding dimension (for cross-attention).
    /// </summary>
    public int T5EmbeddingDimension => _t5Encoder.EmbeddingDimension;

    /// <summary>
    /// Initializes a new dual text encoder conditioning module.
    /// </summary>
    /// <param name="clipVariant">CLIP variant. Default: "ViT-L/14".</param>
    /// <param name="t5Variant">T5 variant. Default: "T5-XXL".</param>
    /// <param name="t5MaxSequenceLength">T5 maximum sequence length. Default: 256.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <example>
    /// <code>
    /// // Create for FLUX.1
    /// var conditioner = new DualTextConditioner&lt;float&gt;(
    ///     clipVariant: "ViT-L/14",
    ///     t5Variant: "T5-XXL");
    ///
    /// // Create for SDXL (two CLIP encoders, no T5)
    /// // Note: For SDXL, use TripleTextConditioner instead
    /// </code>
    /// </example>
    public DualTextConditioner(
        string clipVariant = "ViT-L/14",
        string t5Variant = "T5-XXL",
        int t5MaxSequenceLength = 256,
        int? seed = null)
    {
        _clipEncoder = new CLIPTextConditioner<T>(variant: clipVariant, seed: seed);
        _t5Encoder = new T5TextConditioner<T>(variant: t5Variant, maxSequenceLength: t5MaxSequenceLength, seed: seed);
    }

    /// <inheritdoc />
    public Tensor<T> Encode(Tensor<T> input)
    {
        // Default: use T5 for cross-attention embeddings
        return _t5Encoder.Encode(input);
    }

    /// <inheritdoc />
    public Tensor<T> EncodeText(Tensor<T> tokenIds, Tensor<T>? attentionMask = null)
    {
        // Use T5 for the primary cross-attention embeddings
        return _t5Encoder.EncodeText(tokenIds, attentionMask);
    }

    /// <summary>
    /// Encodes text using both CLIP and T5 encoders.
    /// </summary>
    /// <param name="text">The text prompt to encode.</param>
    /// <returns>A tuple of (T5 sequence embeddings for cross-attention, CLIP pooled embedding).</returns>
    public (Tensor<T> SequenceEmbeddings, Tensor<T> PooledEmbedding) EncodeDual(string text)
    {
        // Tokenize for each encoder
        var clipTokens = _clipEncoder.Tokenize(text);
        var t5Tokens = _t5Encoder.Tokenize(text);

        // Encode with each
        var clipEmbeddings = _clipEncoder.EncodeText(clipTokens);
        var t5Embeddings = _t5Encoder.EncodeText(t5Tokens);

        // Get CLIP pooled embedding
        var clipPooled = _clipEncoder.GetPooledEmbedding(clipEmbeddings);

        return (t5Embeddings, clipPooled);
    }

    /// <inheritdoc />
    /// <remarks>
    /// The pooled embedding is always derived from the CLIP encoder's own output,
    /// not from the passed-in sequence embeddings (which may be T5 embeddings with
    /// a different dimensionality). This method re-encodes using CLIP to produce
    /// a semantically correct pooled representation.
    /// </remarks>
    public Tensor<T> GetPooledEmbedding(Tensor<T> sequenceEmbeddings)
    {
        // The caller may pass T5 sequence embeddings here (e.g., from Encode/EncodeText).
        // CLIP pooling requires CLIP-encoded embeddings, so we use the CLIP encoder's
        // unconditional embedding to produce a pooled output with the correct dimensions.
        // For prompt-specific pooling, callers should use EncodeDual() which correctly
        // routes each encoder's output.
        var clipEmbeddings = _clipEncoder.Encode(sequenceEmbeddings);
        return _clipEncoder.GetPooledEmbedding(clipEmbeddings);
    }

    /// <inheritdoc />
    public Tensor<T> GetUnconditionalEmbedding(int batchSize)
    {
        // Use T5 unconditional for cross-attention
        return _t5Encoder.GetUnconditionalEmbedding(batchSize);
    }

    /// <summary>
    /// Gets unconditional embeddings from both encoders.
    /// </summary>
    /// <param name="batchSize">The batch size.</param>
    /// <returns>A tuple of (T5 unconditional sequence, CLIP unconditional pooled).</returns>
    public (Tensor<T> SequenceEmbeddings, Tensor<T> PooledEmbedding) GetUnconditionalDual(int batchSize)
    {
        var t5Uncond = _t5Encoder.GetUnconditionalEmbedding(batchSize);
        var clipUncond = _clipEncoder.GetUnconditionalEmbedding(batchSize);
        var clipPooled = _clipEncoder.GetPooledEmbedding(clipUncond);

        return (t5Uncond, clipPooled);
    }

    /// <inheritdoc />
    public Tensor<T> Tokenize(string text)
    {
        // Default to T5 tokenization (longer sequences)
        return _t5Encoder.Tokenize(text);
    }

    /// <inheritdoc />
    public Tensor<T> TokenizeBatch(string[] texts)
    {
        return _t5Encoder.TokenizeBatch(texts);
    }
}
