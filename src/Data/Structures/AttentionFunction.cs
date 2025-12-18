namespace AiDotNet.Data.Structures;

/// <summary>
/// Enumeration of attention functions used in meta-learning and neural networks.
/// </summary>
/// <remarks>
/// <para>
/// Attention functions determine how different elements in a sequence or set
/// are weighted when computing representations. They are crucial for few-shot
/// learning where relationships between examples must be captured effectively.
/// </para>
/// <para><b>For Beginners:</b> Attention is like focusing on what's important:</para>
///
/// When comparing a query image to support images in few-shot learning:
/// - <b>Additive:</b> "Add up how similar each part is"
/// - <b>Multiplicative:</b> "Multiply similarities together"
/// - <b>Dot Product:</b> "Use direct similarity scores"
/// - <b>Cosine:</b> "Focus on angle similarity, not magnitude"
/// </para>
/// </remarks>
public enum AttentionFunction
{
    /// <summary>
    /// Additive attention mechanism.
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// - Computes compatibility through learned linear transformations
    /// - Adds query and key transformations
    /// - Used in the original Transformer paper
    ///
    /// <para><b>Formula:</b></para>
    /// attention(Q, K) = softmax(W * Q + W' * K)
    ///
    /// <para><b>When to use:</b></para>
    /// - When you need to model complex relationships
    /// - With larger models that can learn the transformation
    /// - When additive interactions are meaningful
    /// </remarks>
    Additive,

    /// <summary>
    /// Multiplicative (dot product) attention.
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// - Computes dot product between query and key
    /// - Scales by dimension to prevent small gradients
    /// - Faster and simpler than additive attention
    ///
    /// <para><b>Formula:</b></para>
    /// attention(Q, K) = softmax((Q · K^T) / √d_k)
    ///
    /// <para><b>When to use:</b></para>
    /// - Most common choice in modern architectures
    /// - When computational efficiency matters
    /// - Works well with properly normalized inputs
    /// </remarks>
    Multiplicative,

    /// <summary>
    /// Cosine similarity attention.
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// - Computes cosine similarity between vectors
    /// - Focuses on angle, not magnitude
    /// - Robust to scale differences
    ///
    /// <para><b>Formula:</b></para>
    /// attention(Q, K) = softmax(cosine_similarity(Q, K))
    ///
    /// <para><b>When to use:</b></para>
    /// - When feature magnitudes vary significantly
    /// - For prototype-based methods
    /// - When angular similarity is more meaningful
    /// </remarks>
    Cosine,

    /// <summary>
    /// Scaled dot product attention (Transformer default).
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// - Same as multiplicative but with dimension scaling
    /// - Prevents softmax saturation in high dimensions
    /// - Most widely used in practice
    ///
    /// <para><b>Formula:</b></para>
    /// attention(Q, K, V) = softmax((QK^T) / √d_k) V
    ///
    /// <para><b>When to use:</b></para>
    /// - Default choice for most applications
    /// - High-dimensional embeddings
    /// - When following Transformer architecture
    /// </remarks>
    ScaledDotProduct,

    /// <summary>
    /// Bilinear attention.
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// - Uses learned bilinear transformation
    /// - More expressive than dot product
    /// - Can model complex interactions
    ///
    /// <para><b>Formula:</b></para>
    /// attention(Q, K) = softmax(Q^T W K)
    ///
    /// <para><b>When to use:</b></para>
    /// - When quadratic relationships matter
    /// - For modeling interactions between different dimensions
    /// - With sufficient training data
    /// </remarks>
    Bilinear,

    /// <summary>
    /// Learned attention with trainable parameters.
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// - Attention mechanism is learned end-to-end
    /// - Can discover optimal attention patterns
    /// - Most flexible but requires more data
    ///
    /// <para><b>When to use:</b></para>
    /// - When domain knowledge is limited
    /// - With large datasets
    /// - For research and exploration
    /// </remarks>
    Learned,

    /// <summary>
    /// Gaussian attention (soft attention).
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// - Uses Gaussian kernel for attention weights
    /// - Smooth, differentiable attention
    /// - Naturally normalized
    ///
    /// <para><b>Formula:</b></para>
    /// attention(Q, K) = exp(-||Q - K||² / 2σ²)
    ///
    /// <para><b>When to use:</b></para>
    /// - For continuous attention distributions
    /// - When smoothness is important
    /// - In kernel-based methods
    /// </remarks>
    Gaussian,

    /// <summary>
    /// Hybrid attention combining multiple mechanisms.
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// - Combines multiple attention types
    /// - Can use the best of each approach
    /// - More complex but potentially more powerful
    ///
    /// <para><b>When to use:</b></para>
    /// - When single mechanism isn't sufficient
    /// - For challenging tasks
    /// - In research settings
    /// </remarks>
    Hybrid
}