namespace AiDotNet.VisionLanguage.Interfaces;

/// <summary>
/// Interface for text encoders that extract feature representations from text.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Text encoders transform text strings into compact feature representations (embeddings)
/// that capture semantic meaning. In vision-language models, text embeddings are aligned with
/// visual embeddings in a shared space for cross-modal tasks.
/// </para>
/// <para>
/// <b>For Beginners:</b> A text encoder converts words and sentences into numbers that represent
/// their meaning. Similar sentences get similar numbers, which lets the model compare text with
/// images in the same mathematical space.
/// </para>
/// </remarks>
public interface ITextEncoder<T>
{
    /// <summary>
    /// Encodes a text string into an embedding vector.
    /// </summary>
    /// <param name="text">The text to encode.</param>
    /// <returns>Text embedding tensor in the model's feature space.</returns>
    Tensor<T> EncodeText(string text);

    /// <summary>
    /// Encodes multiple text strings into embedding vectors.
    /// </summary>
    /// <param name="texts">The texts to encode.</param>
    /// <returns>Batch of text embedding tensors.</returns>
    Tensor<T>[] EncodeTexts(string[] texts);

    /// <summary>
    /// Gets the maximum token sequence length supported by this encoder.
    /// </summary>
    int MaxSequenceLength { get; }

    /// <summary>
    /// Gets the dimensionality of the output text embedding space.
    /// </summary>
    int TextEmbeddingDimension { get; }
}
