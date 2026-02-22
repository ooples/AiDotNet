using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion;

/// <summary>
/// Triple text encoder conditioning module combining two CLIP encoders and a T5 encoder.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Stable Diffusion 3 uses three text encoders to achieve the highest quality text understanding:
/// two CLIP encoders for visual-semantic alignment (with pooled embeddings) and a T5 encoder
/// for detailed language understanding (with cross-attention sequence embeddings).
/// </para>
/// <para>
/// <b>For Beginners:</b> This uses THREE text encoders together for the best possible results.
///
/// Why three encoders?
/// - CLIP ViT-L/14 (768-dim): Fast, general visual understanding of your prompt
/// - OpenCLIP ViT-bigG/14 (1280-dim): Larger, more detailed visual understanding
/// - T5-XXL (4096-dim): Deep language understanding for complex prompts
///
/// How they work together:
/// 1. Both CLIP encoders produce "pooled" embeddings (one vector each summarizing the prompt)
///    → These are concatenated into a combined 2048-dim (768+1280) vector
///    → Used for the global conditioning vector fed to the MMDiT timestep embedder
/// 2. T5 produces "sequence" embeddings (one vector per token, 4096-dim each)
///    → Used for cross-attention, giving fine-grained control over details
///
/// Used by:
/// - Stable Diffusion 3 (SD3): All three encoders
/// - SD3 Turbo: All three encoders with fewer steps
/// </para>
/// <para>
/// <b>Reference:</b> Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", ICML 2024
/// </para>
/// </remarks>
public class TripleTextConditioner<T> : IConditioningModule<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The first CLIP text encoder (ViT-L/14, 768-dim).
    /// </summary>
    private readonly CLIPTextConditioner<T> _clipLEncoder;

    /// <summary>
    /// The second CLIP/OpenCLIP text encoder (ViT-bigG/14, 1280-dim).
    /// </summary>
    private readonly CLIPTextConditioner<T> _clipGEncoder;

    /// <summary>
    /// The T5 text encoder providing high-dimensional sequence embeddings.
    /// </summary>
    private readonly T5TextConditioner<T> _t5Encoder;

    /// <summary>
    /// Gets the T5 context dimension for cross-attention (4096 for T5-XXL).
    /// </summary>
    public int EmbeddingDimension => _t5Encoder.EmbeddingDimension;

    /// <inheritdoc />
    public ConditioningType ConditioningType => ConditioningType.MultiModal;

    /// <summary>
    /// Gets whether this module produces pooled output (yes, from both CLIP encoders).
    /// </summary>
    public bool ProducesPooledOutput => true;

    /// <summary>
    /// Gets the maximum sequence length (uses T5's longer sequence length).
    /// </summary>
    public int MaxSequenceLength => _t5Encoder.MaxSequenceLength;

    /// <summary>
    /// Gets the first CLIP encoder's embedding dimension (768 for ViT-L/14).
    /// </summary>
    public int CLIPLEmbeddingDimension => _clipLEncoder.EmbeddingDimension;

    /// <summary>
    /// Gets the second CLIP/OpenCLIP encoder's embedding dimension (1280 for ViT-bigG/14).
    /// </summary>
    public int CLIPGEmbeddingDimension => _clipGEncoder.EmbeddingDimension;

    /// <summary>
    /// Gets the T5 encoder's embedding dimension (4096 for T5-XXL).
    /// </summary>
    public int T5EmbeddingDimension => _t5Encoder.EmbeddingDimension;

    /// <summary>
    /// Gets the combined pooled embedding dimension (CLIP-L + CLIP-G = 768 + 1280 = 2048).
    /// </summary>
    public int CombinedPooledDimension => _clipLEncoder.EmbeddingDimension + _clipGEncoder.EmbeddingDimension;

    /// <summary>
    /// Initializes a new triple text encoder conditioning module for SD3.
    /// </summary>
    /// <param name="clipLVariant">First CLIP variant. Default: "ViT-L/14" (768-dim).</param>
    /// <param name="clipGVariant">Second CLIP/OpenCLIP variant. Default: "ViT-bigG/14" (1280-dim).</param>
    /// <param name="t5Variant">T5 variant. Default: "T5-XXL" (4096-dim).</param>
    /// <param name="t5MaxSequenceLength">T5 maximum sequence length. Default: 256.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <example>
    /// <code>
    /// // Create for Stable Diffusion 3
    /// var conditioner = new TripleTextConditioner&lt;float&gt;();
    ///
    /// // Encode a prompt with all three encoders
    /// var (sequenceEmbeddings, combinedPooled) = conditioner.EncodeTriple("a serene mountain lake");
    ///
    /// // sequenceEmbeddings: [1, 256, 4096] - from T5, used for cross-attention
    /// // combinedPooled: [1, 2048] - from CLIP-L + CLIP-G, used for timestep conditioning
    /// </code>
    /// </example>
    public TripleTextConditioner(
        string clipLVariant = "ViT-L/14",
        string clipGVariant = "ViT-bigG/14",
        string t5Variant = "T5-XXL",
        int t5MaxSequenceLength = 256,
        int? seed = null)
    {
        _clipLEncoder = new CLIPTextConditioner<T>(variant: clipLVariant, seed: seed);
        _clipGEncoder = new CLIPTextConditioner<T>(variant: clipGVariant, seed: seed);
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
    /// Encodes text using all three encoders (CLIP-L, CLIP-G/OpenCLIP, and T5).
    /// </summary>
    /// <param name="text">The text prompt to encode.</param>
    /// <returns>
    /// A tuple of:
    /// - SequenceEmbeddings: T5 sequence embeddings [batchSize, seqLen, 4096] for cross-attention
    /// - CombinedPooledEmbedding: Concatenated CLIP-L + CLIP-G pooled [batchSize, 2048] for timestep conditioning
    /// </returns>
    /// <remarks>
    /// <para>
    /// In SD3's MMDiT architecture:
    /// - The combined pooled embedding (2048-dim) is added to the timestep embedding
    ///   and used as the global conditioning signal
    /// - The T5 sequence embeddings (4096-dim per token) are used in cross-attention
    ///   for fine-grained text-to-image alignment
    /// </para>
    /// </remarks>
    public (Tensor<T> SequenceEmbeddings, Tensor<T> CombinedPooledEmbedding) EncodeTriple(string text)
    {
        // Tokenize for each encoder
        var clipLTokens = _clipLEncoder.Tokenize(text);
        var clipGTokens = _clipGEncoder.Tokenize(text);
        var t5Tokens = _t5Encoder.Tokenize(text);

        // Encode with each
        var clipLEmbeddings = _clipLEncoder.EncodeText(clipLTokens);
        var clipGEmbeddings = _clipGEncoder.EncodeText(clipGTokens);
        var t5Embeddings = _t5Encoder.EncodeText(t5Tokens);

        // Get pooled embeddings from both CLIP encoders
        var clipLPooled = _clipLEncoder.GetPooledEmbedding(clipLEmbeddings);
        var clipGPooled = _clipGEncoder.GetPooledEmbedding(clipGEmbeddings);

        // Concatenate pooled embeddings: [batchSize, clipL_dim + clipG_dim]
        var combinedPooled = ConcatenatePooledEmbeddings(clipLPooled, clipGPooled);

        return (t5Embeddings, combinedPooled);
    }

    /// <inheritdoc />
    public Tensor<T> GetPooledEmbedding(Tensor<T> sequenceEmbeddings)
    {
        // For the standard interface, get pooled from the first CLIP encoder
        // For the full combined pooled, use EncodeTriple() instead
        return _clipLEncoder.GetPooledEmbedding(sequenceEmbeddings);
    }

    /// <summary>
    /// Gets the combined pooled embedding from both CLIP encoders for the given text.
    /// </summary>
    /// <param name="text">The text prompt.</param>
    /// <returns>Combined pooled embedding [batchSize, 2048].</returns>
    public Tensor<T> GetCombinedPooledEmbedding(string text)
    {
        var clipLTokens = _clipLEncoder.Tokenize(text);
        var clipGTokens = _clipGEncoder.Tokenize(text);

        var clipLEmbeddings = _clipLEncoder.EncodeText(clipLTokens);
        var clipGEmbeddings = _clipGEncoder.EncodeText(clipGTokens);

        var clipLPooled = _clipLEncoder.GetPooledEmbedding(clipLEmbeddings);
        var clipGPooled = _clipGEncoder.GetPooledEmbedding(clipGEmbeddings);

        return ConcatenatePooledEmbeddings(clipLPooled, clipGPooled);
    }

    /// <inheritdoc />
    public Tensor<T> GetUnconditionalEmbedding(int batchSize)
    {
        // Use T5 unconditional for cross-attention
        return _t5Encoder.GetUnconditionalEmbedding(batchSize);
    }

    /// <summary>
    /// Gets unconditional embeddings from all three encoders.
    /// </summary>
    /// <param name="batchSize">The batch size.</param>
    /// <returns>
    /// A tuple of:
    /// - SequenceEmbeddings: T5 unconditional sequence [batchSize, seqLen, 4096]
    /// - CombinedPooledEmbedding: Concatenated CLIP-L + CLIP-G unconditional pooled [batchSize, 2048]
    /// </returns>
    public (Tensor<T> SequenceEmbeddings, Tensor<T> CombinedPooledEmbedding) GetUnconditionalTriple(int batchSize)
    {
        // Get unconditional from each encoder
        var clipLUncond = _clipLEncoder.GetUnconditionalEmbedding(batchSize);
        var clipGUncond = _clipGEncoder.GetUnconditionalEmbedding(batchSize);
        var t5Uncond = _t5Encoder.GetUnconditionalEmbedding(batchSize);

        // Get pooled from both CLIP encoders
        var clipLPooled = _clipLEncoder.GetPooledEmbedding(clipLUncond);
        var clipGPooled = _clipGEncoder.GetPooledEmbedding(clipGUncond);

        var combinedPooled = ConcatenatePooledEmbeddings(clipLPooled, clipGPooled);

        return (t5Uncond, combinedPooled);
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

    /// <summary>
    /// Concatenates pooled embeddings from CLIP-L and CLIP-G along the feature dimension.
    /// </summary>
    /// <param name="clipLPooled">CLIP ViT-L/14 pooled embedding [batchSize, 768].</param>
    /// <param name="clipGPooled">CLIP ViT-bigG/14 pooled embedding [batchSize, 1280].</param>
    /// <returns>Combined pooled embedding [batchSize, 2048].</returns>
    private Tensor<T> ConcatenatePooledEmbeddings(Tensor<T> clipLPooled, Tensor<T> clipGPooled)
    {
        int batchSize = clipLPooled.Shape[0];
        int clipLDim = _clipLEncoder.EmbeddingDimension;
        int clipGDim = _clipGEncoder.EmbeddingDimension;
        int combinedDim = clipLDim + clipGDim;

        var combinedData = new Vector<T>(batchSize * combinedDim);

        for (int b = 0; b < batchSize; b++)
        {
            // Copy CLIP-L pooled embedding
            for (int d = 0; d < clipLDim; d++)
            {
                combinedData[b * combinedDim + d] = clipLPooled[b * clipLDim + d];
            }

            // Copy CLIP-G pooled embedding
            for (int d = 0; d < clipGDim; d++)
            {
                combinedData[b * combinedDim + clipLDim + d] = clipGPooled[b * clipGDim + d];
            }
        }

        return new Tensor<T>(new[] { batchSize, combinedDim }, combinedData);
    }
}
