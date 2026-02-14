using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for multimodal embedding models that can encode multiple modalities
/// (text, images, audio) into a shared embedding space.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para>
/// Multimodal embedding models like CLIP (Contrastive Language-Image Pre-training)
/// learn to project different types of data into the same vector space, enabling
/// cross-modal similarity search and zero-shot classification.
/// </para>
/// <para><b>For Beginners:</b> Imagine you want to search for images using text queries.
/// A multimodal model learns to convert both "a photo of a cat" and an actual cat image
/// into similar vectors, allowing direct comparison between text and images.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("MultimodalEmbedding")]
public interface IMultimodalEmbedding<T>
{
    /// <summary>
    /// Encodes text into an embedding vector.
    /// </summary>
    /// <param name="text">The text to encode.</param>
    /// <returns>A normalized embedding vector.</returns>
    Vector<T> EncodeText(string text);

    /// <summary>
    /// Encodes multiple texts into embedding vectors in a batch.
    /// </summary>
    /// <param name="texts">The texts to encode.</param>
    /// <returns>A matrix where each row is an embedding for the corresponding text.</returns>
    Matrix<T> EncodeTextBatch(IEnumerable<string> texts);

    /// <summary>
    /// Encodes an image into an embedding vector.
    /// </summary>
    /// <param name="imageData">The preprocessed image data as a flattened array in CHW format.</param>
    /// <returns>A normalized embedding vector.</returns>
    Vector<T> EncodeImage(double[] imageData);

    /// <summary>
    /// Encodes multiple images into embedding vectors in a batch.
    /// </summary>
    /// <param name="imageDataBatch">The preprocessed images as flattened arrays.</param>
    /// <returns>A matrix where each row is an embedding for the corresponding image.</returns>
    Matrix<T> EncodeImageBatch(IEnumerable<double[]> imageDataBatch);

    /// <summary>
    /// Computes similarity between two embeddings.
    /// </summary>
    /// <param name="embedding1">The first embedding.</param>
    /// <param name="embedding2">The second embedding.</param>
    /// <returns>Similarity score (cosine similarity for normalized embeddings).</returns>
    T ComputeSimilarity(Vector<T> embedding1, Vector<T> embedding2);

    /// <summary>
    /// Performs zero-shot classification of an image against text labels.
    /// </summary>
    /// <param name="imageData">The preprocessed image data.</param>
    /// <param name="labels">The candidate class labels.</param>
    /// <returns>A dictionary mapping each label to its probability score.</returns>
    Dictionary<string, T> ZeroShotClassify(double[] imageData, IEnumerable<string> labels);

    /// <summary>
    /// Gets the dimensionality of the embedding space.
    /// </summary>
    int EmbeddingDimension { get; }

    /// <summary>
    /// Gets the maximum sequence length for text input.
    /// </summary>
    int MaxSequenceLength { get; }

    /// <summary>
    /// Gets the expected image size (square images: ImageSize x ImageSize pixels).
    /// </summary>
    int ImageSize { get; }
}
