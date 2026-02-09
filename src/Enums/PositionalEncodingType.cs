namespace AiDotNet.Enums;

/// <summary>
/// Represents the type of positional encoding used in transformer attention layers.
/// </summary>
/// <remarks>
/// <para>
/// Positional encodings provide sequence position information to attention mechanisms,
/// which have no inherent notion of token order. Different encoding strategies trade off
/// between extrapolation to unseen lengths, computational cost, and compatibility with
/// KV-caching during inference.
/// </para>
/// <para><b>For Beginners:</b> Transformers process all tokens in parallel, so they need
/// a way to know which position each token is at (first word, second word, etc.).
///
/// Different approaches:
/// - <b>Sinusoidal:</b> The original approach from "Attention Is All You Need" (2017)
/// - <b>Rotary (RoPE):</b> Used by Llama, Mistral, Phi, Gemma - encodes relative positions
/// - <b>ALiBi:</b> Used by BLOOM, MPT - adds a simple distance-based bias to attention scores
/// - <b>LearnedAbsolute:</b> Used by BERT, GPT-2 - learns position embeddings during training
/// - <b>None:</b> No positional encoding (for architectures that don't need it)
///
/// For modern LLMs, <see cref="Rotary"/> is the most common choice as of 2025-2026.
/// </para>
/// </remarks>
public enum PositionalEncodingType
{
    /// <summary>
    /// Sinusoidal positional encoding from the original Transformer paper (Vaswani et al., 2017).
    /// </summary>
    /// <remarks>
    /// Uses fixed sine and cosine functions at different frequencies.
    /// PE(pos, 2i) = sin(pos / 10000^(2i/d_model)),
    /// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)).
    /// </remarks>
    Sinusoidal,

    /// <summary>
    /// Rotary Position Embedding (RoPE) from Su et al., 2021.
    /// </summary>
    /// <remarks>
    /// <para>
    /// RoPE encodes position by rotating query and key vectors in 2D subspaces.
    /// This naturally encodes relative positions through the dot product, making it
    /// ideal for attention-based models.
    /// </para>
    /// <para>
    /// Used by: Llama 2/3, Mistral, Phi-3, Gemma, GPT-NeoX, CodeLlama.
    /// </para>
    /// </remarks>
    Rotary,

    /// <summary>
    /// Attention with Linear Biases (ALiBi) from Press et al., 2022.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ALiBi adds a linear distance-based bias to attention scores instead of modifying embeddings.
    /// Each head uses a different slope: bias[h, i, j] = -slope_h * |i - j|.
    /// This enables strong length extrapolation without any learned parameters.
    /// </para>
    /// <para>
    /// Used by: BLOOM, MPT, Falcon (some variants).
    /// </para>
    /// </remarks>
    ALiBi,

    /// <summary>
    /// Learned absolute positional embeddings (BERT, GPT-2 style).
    /// </summary>
    /// <remarks>
    /// Position embeddings are learned during training as a parameter matrix.
    /// Simple but limited to the maximum sequence length seen during training.
    /// </remarks>
    LearnedAbsolute,

    /// <summary>
    /// No positional encoding applied.
    /// </summary>
    /// <remarks>
    /// Use this when the architecture does not require positional information
    /// or when position encoding is handled externally.
    /// </remarks>
    None
}
