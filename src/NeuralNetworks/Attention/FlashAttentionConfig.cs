namespace AiDotNet.NeuralNetworks.Attention;

/// <summary>
/// Configuration options for Flash Attention algorithm.
/// </summary>
/// <remarks>
/// <para>
/// Flash Attention is a memory-efficient attention algorithm that avoids materializing
/// the full N x N attention matrix. Instead, it processes attention in tiles/blocks,
/// computing online softmax incrementally.
/// </para>
/// <para><b>For Beginners:</b> Flash Attention is a faster way to compute attention.
///
/// Standard attention creates a huge matrix comparing every position to every other position.
/// For long sequences (like 4096 tokens), this matrix has 16 million entries!
///
/// Flash Attention avoids creating this huge matrix by:
/// - Processing in small blocks that fit in fast GPU memory (SRAM)
/// - Computing softmax incrementally as it processes each block
/// - Never storing the full attention matrix
///
/// Benefits:
/// - 2-4x faster than standard attention
/// - Uses much less memory (O(N) instead of O(N^2))
/// - Enables training with longer sequences
/// </para>
/// </remarks>
public class FlashAttentionConfig
{
    /// <summary>
    /// Block size for query processing (Br in the paper).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Controls how many query positions are processed together.
    /// Larger values may be faster but use more memory.
    /// Must divide sequence length evenly for best performance.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many "questions" we process at once.
    ///
    /// Default of 64 works well for most GPUs:
    /// - RTX 3090/4090: Can use 128
    /// - Older GPUs: May need 32
    /// </para>
    /// </remarks>
    public int BlockSizeQ { get; set; } = 64;

    /// <summary>
    /// Block size for key/value processing (Bc in the paper).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Controls how many key/value positions are processed together.
    /// Should typically match BlockSizeQ for square blocks.
    /// </para>
    /// </remarks>
    public int BlockSizeKV { get; set; } = 64;

    /// <summary>
    /// Whether to apply causal masking (for autoregressive models).
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true, position i can only attend to positions j where j &lt;= i.
    /// This is essential for language models like GPT where future tokens should not influence current predictions.
    /// </para>
    /// <para><b>For Beginners:</b> Causal masking prevents "cheating" in text generation.
    ///
    /// When generating text word by word:
    /// - The model shouldn't see future words when predicting the next word
    /// - Causal masking hides future positions
    /// - Set to true for GPT-style models
    /// - Set to false for BERT-style models (bidirectional)
    /// </para>
    /// </remarks>
    public bool UseCausalMask { get; set; } = false;

    /// <summary>
    /// Dropout probability to apply to attention weights during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Randomly zeros out attention weights to prevent overfitting.
    /// Only applied during training, not inference.
    /// </para>
    /// </remarks>
    public float DropoutProbability { get; set; } = 0.0f;

    /// <summary>
    /// Scale factor for attention scores. If null, uses 1/sqrt(head_dim).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The standard scale factor of 1/sqrt(d_k) prevents attention scores from
    /// becoming too large, which would cause softmax to produce very peaked distributions.
    /// </para>
    /// </remarks>
    public float? ScaleFactor { get; set; } = null;

    /// <summary>
    /// Whether to use the optimized GPU kernel (when available).
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true and GPU is available, uses optimized DirectGpu kernels for Flash Attention.
    /// Falls back to CPU implementation if GPU is not available.
    /// </para>
    /// </remarks>
    public bool UseGpuKernel { get; set; } = true;

    /// <summary>
    /// Whether to enable memory-efficient backward pass with recomputation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true, the backward pass recomputes attention weights instead of storing them.
    /// This significantly reduces memory usage at the cost of some additional computation.
    /// </para>
    /// <para><b>For Beginners:</b> This trades speed for memory during training.
    ///
    /// Standard approach: Store attention weights, use them in backward pass
    /// Recomputation: Recompute attention weights during backward pass
    ///
    /// Enable this when:
    /// - Training with limited GPU memory
    /// - Using very long sequences
    /// - Training large models
    /// </para>
    /// </remarks>
    public bool RecomputeInBackward { get; set; } = true;

    /// <summary>
    /// Numerical precision mode for attention computation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Controls the precision used for intermediate computations.
    /// Higher precision is more accurate but slower and uses more memory.
    /// </para>
    /// </remarks>
    public FlashAttentionPrecision Precision { get; set; } = FlashAttentionPrecision.Float32;

    /// <summary>
    /// Whether to return attention weights (for visualization/debugging).
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true, materializes and returns the attention weights.
    /// This negates some memory benefits of Flash Attention but is useful for debugging.
    /// Should typically be false in production.
    /// </para>
    /// </remarks>
    public bool ReturnAttentionWeights { get; set; } = false;

    /// <summary>
    /// Creates a default configuration suitable for most use cases.
    /// </summary>
    public static FlashAttentionConfig Default => new();

    /// <summary>
    /// Creates a configuration optimized for causal/autoregressive models.
    /// </summary>
    public static FlashAttentionConfig Causal => new()
    {
        UseCausalMask = true,
        RecomputeInBackward = true
    };

    /// <summary>
    /// Creates a configuration optimized for memory efficiency.
    /// </summary>
    public static FlashAttentionConfig MemoryEfficient => new()
    {
        BlockSizeQ = 32,
        BlockSizeKV = 32,
        RecomputeInBackward = true,
        ReturnAttentionWeights = false
    };

    /// <summary>
    /// Creates a configuration optimized for speed (uses more memory).
    /// </summary>
    public static FlashAttentionConfig HighPerformance => new()
    {
        BlockSizeQ = 128,
        BlockSizeKV = 128,
        RecomputeInBackward = false,
        UseGpuKernel = true
    };
}

/// <summary>
/// Precision modes for Flash Attention computation.
/// </summary>
public enum FlashAttentionPrecision
{
    /// <summary>
    /// Use 16-bit floating point (half precision).
    /// Fastest but may have numerical issues with very long sequences.
    /// </summary>
    Float16,

    /// <summary>
    /// Use 32-bit floating point (single precision).
    /// Good balance of speed and accuracy.
    /// </summary>
    Float32,

    /// <summary>
    /// Use mixed precision (FP16 for matmul, FP32 for softmax).
    /// Best combination of speed and numerical stability.
    /// </summary>
    Mixed
}
