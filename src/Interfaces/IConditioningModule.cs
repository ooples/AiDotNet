namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for conditioning modules that encode various inputs into embeddings for diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Conditioning modules convert various types of input (text, images, audio, etc.) into
/// embedding tensors that guide the diffusion process. They are essential for controlled
/// generation like text-to-image, image-to-image, or style transfer.
/// </para>
/// <para>
/// <b>For Beginners:</b> A conditioning module is like a "translator" that converts your input
/// (like a text prompt) into a format the diffusion model can understand.
///
/// Common types of conditioning:
/// 1. Text conditioning (CLIP, T5): "A cat sitting on a couch" → embedding vectors
/// 2. Image conditioning (IP-Adapter): An image → embedding vectors for style/content
/// 3. Control conditioning (ControlNet): Depth maps, edges, poses → spatial guidance
///
/// Why conditioning matters:
/// - Without conditioning: Model generates random images
/// - With text conditioning: Model generates images matching your description
/// - With image conditioning: Model preserves style or content from reference images
/// - With control conditioning: Model follows spatial structure (poses, edges, depth)
///
/// Different conditioning methods:
/// - Cross-attention: Text embeddings attend to image features (most common for text)
/// - Addition/Concatenation: Add or concat embeddings to time embedding
/// - Spatial: Add control signals directly to features at each resolution
/// </para>
/// </remarks>
public interface IConditioningModule<T>
{
    /// <summary>
    /// Gets the dimension of the output embeddings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For CLIP text encoders, this is typically 768 or 1024.
    /// For T5, this is typically 1024 or 2048.
    /// For image encoders, it varies by architecture.
    /// </para>
    /// </remarks>
    int EmbeddingDimension { get; }

    /// <summary>
    /// Gets the type of conditioning this module provides.
    /// </summary>
    ConditioningType ConditioningType { get; }

    /// <summary>
    /// Gets whether this module produces pooled (global) or sequence embeddings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// - Pooled: Single vector representing the entire input (e.g., CLIP pooled output)
    /// - Sequence: Multiple vectors, one per token/patch (e.g., for cross-attention)
    /// </para>
    /// </remarks>
    bool ProducesPooledOutput { get; }

    /// <summary>
    /// Gets the maximum sequence length for text input.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For CLIP, this is typically 77 tokens.
    /// For T5, this can be much longer (512 or more).
    /// Returns 0 for non-text conditioning modules.
    /// </para>
    /// </remarks>
    int MaxSequenceLength { get; }

    /// <summary>
    /// Encodes the input into conditioning embeddings.
    /// </summary>
    /// <param name="input">The input tensor (format depends on conditioning type).</param>
    /// <returns>The conditioning embeddings.</returns>
    /// <remarks>
    /// <para>
    /// Input format by type:
    /// - Text: Tokenized text [batch, seqLength]
    /// - Image: Image tensor [batch, channels, height, width]
    /// - Audio: Audio tensor [batch, channels, samples]
    /// - Control: Control signal [batch, channels, height, width]
    /// </para>
    /// <para>
    /// Output format:
    /// - Sequence: [batch, seqLength, embeddingDim]
    /// - Pooled: [batch, embeddingDim]
    /// </para>
    /// </remarks>
    Tensor<T> Encode(Tensor<T> input);

    /// <summary>
    /// Encodes text input (convenience method for text conditioning).
    /// </summary>
    /// <param name="tokenIds">Tokenized text [batch, seqLength].</param>
    /// <param name="attentionMask">Optional attention mask [batch, seqLength].</param>
    /// <returns>Text embeddings for cross-attention.</returns>
    /// <remarks>
    /// <para>
    /// This is the primary method for text conditioning. The attention mask indicates
    /// which tokens are real (1) vs padding (0).
    /// </para>
    /// </remarks>
    Tensor<T> EncodeText(Tensor<T> tokenIds, Tensor<T>? attentionMask = null);

    /// <summary>
    /// Gets the pooled (global) embedding from sequence embeddings.
    /// </summary>
    /// <param name="sequenceEmbeddings">The sequence embeddings [batch, seqLength, dim].</param>
    /// <returns>Pooled embeddings [batch, dim].</returns>
    /// <remarks>
    /// <para>
    /// For CLIP, this is typically the EOS token embedding.
    /// For other models, it might be mean pooling or a learned pooler.
    /// </para>
    /// </remarks>
    Tensor<T> GetPooledEmbedding(Tensor<T> sequenceEmbeddings);

    /// <summary>
    /// Gets unconditioned (null) embeddings for classifier-free guidance.
    /// </summary>
    /// <param name="batchSize">The batch size.</param>
    /// <returns>Unconditional embeddings matching the output format.</returns>
    /// <remarks>
    /// <para>
    /// Classifier-free guidance requires running the model with both conditional
    /// and unconditional embeddings. This returns the "empty" or "null" conditioning
    /// used for the unconditional pass.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This creates a "blank" conditioning that says "generate anything".
    /// By comparing model outputs with and without the prompt, we can steer generation
    /// more strongly toward the prompt.
    /// </para>
    /// </remarks>
    Tensor<T> GetUnconditionalEmbedding(int batchSize);

    /// <summary>
    /// Tokenizes text input (for text conditioning modules).
    /// </summary>
    /// <param name="text">The text to tokenize.</param>
    /// <returns>Token IDs as a tensor [1, seqLength].</returns>
    /// <remarks>
    /// <para>
    /// Converts text to a sequence of integer token IDs that the encoder understands.
    /// Handles padding and truncation to MaxSequenceLength.
    /// </para>
    /// </remarks>
    Tensor<T> Tokenize(string text);

    /// <summary>
    /// Tokenizes a batch of text inputs.
    /// </summary>
    /// <param name="texts">The texts to tokenize.</param>
    /// <returns>Token IDs as a tensor [batch, seqLength].</returns>
    Tensor<T> TokenizeBatch(string[] texts);
}

/// <summary>
/// Types of conditioning supported by diffusion models.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This describes what kind of input you're using to guide generation:
/// - Text: Written descriptions like "a beautiful sunset"
/// - Image: Reference images for style or content
/// - Audio: Sound clips for audio generation
/// - Control: Structural guidance like edges, poses, or depth maps
/// - Class: Simple class labels like "dog" or "car"
/// </para>
/// </remarks>
public enum ConditioningType
{
    /// <summary>
    /// Text conditioning (e.g., CLIP, T5 text encoders).
    /// </summary>
    Text,

    /// <summary>
    /// Image conditioning (e.g., CLIP vision encoder, IP-Adapter).
    /// </summary>
    Image,

    /// <summary>
    /// Audio conditioning (e.g., audio spectrograms).
    /// </summary>
    Audio,

    /// <summary>
    /// Spatial control conditioning (e.g., ControlNet, T2I-Adapter).
    /// </summary>
    Control,

    /// <summary>
    /// Class label conditioning (e.g., ImageNet class embeddings).
    /// </summary>
    Class,

    /// <summary>
    /// Multi-modal conditioning (combines multiple types).
    /// </summary>
    MultiModal
}
