using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Models;

/// <summary>
/// Interface for multimodal models that can be served via the REST API.
/// Multimodal models handle multiple input types (text, images, audio) and produce
/// embeddings in a shared vector space.
/// </summary>
/// <typeparam name="T">The numeric type used by the model (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This interface extends the standard <see cref="IServableModel{T}"/> with capabilities
/// for handling multiple modalities, particularly useful for models like CLIP that can
/// encode both text and images into comparable embeddings.
/// </para>
/// <para><b>For Beginners:</b> Multimodal models understand multiple types of data:
///
/// - **Text**: "A photo of a cat" → [0.2, 0.5, 0.1, ...]
/// - **Image**: (cat.jpg) → [0.21, 0.48, 0.12, ...]
///
/// Because both outputs are in the same "space", you can compare them directly:
/// - High similarity = text describes the image well
/// - Low similarity = text doesn't match the image
///
/// This enables powerful applications like image search using text queries.
/// </para>
/// </remarks>
public interface IServableMultimodalModel<T> : IServableModel<T>
{
    /// <summary>
    /// Encodes text into an embedding vector.
    /// </summary>
    /// <param name="text">The text to encode.</param>
    /// <returns>A normalized embedding vector representing the text.</returns>
    /// <remarks>
    /// <para>
    /// The returned embedding is L2-normalized, meaning it has unit length.
    /// This allows similarity to be computed using simple dot product.
    /// </para>
    /// </remarks>
    Vector<T> EncodeText(string text);

    /// <summary>
    /// Encodes multiple texts into embedding vectors in a batch operation.
    /// </summary>
    /// <param name="texts">The texts to encode.</param>
    /// <returns>A matrix where each row is an embedding for the corresponding text.</returns>
    Matrix<T> EncodeTextBatch(IEnumerable<string> texts);

    /// <summary>
    /// Encodes an image into an embedding vector.
    /// </summary>
    /// <param name="imageData">The preprocessed image data as a flattened array.</param>
    /// <returns>A normalized embedding vector representing the image.</returns>
    /// <remarks>
    /// <para>
    /// The image should be preprocessed (resized, normalized) before encoding.
    /// The expected format is typically [channels, height, width] flattened.
    /// </para>
    /// </remarks>
    Vector<T> EncodeImage(double[] imageData);

    /// <summary>
    /// Encodes multiple images into embedding vectors in a batch operation.
    /// </summary>
    /// <param name="imageDataBatch">The preprocessed images as flattened arrays.</param>
    /// <returns>A matrix where each row is an embedding for the corresponding image.</returns>
    Matrix<T> EncodeImageBatch(IEnumerable<double[]> imageDataBatch);

    /// <summary>
    /// Computes the similarity score between a text embedding and an image embedding.
    /// </summary>
    /// <param name="textEmbedding">The text embedding vector.</param>
    /// <param name="imageEmbedding">The image embedding vector.</param>
    /// <returns>A similarity score (cosine similarity for normalized vectors).</returns>
    /// <remarks>
    /// <para>
    /// For L2-normalized embeddings, the dot product equals the cosine similarity.
    /// Values range from -1 (completely opposite) to 1 (identical).
    /// </para>
    /// </remarks>
    T ComputeSimilarity(Vector<T> textEmbedding, Vector<T> imageEmbedding);

    /// <summary>
    /// Performs zero-shot classification of an image against a set of class labels.
    /// </summary>
    /// <param name="imageData">The preprocessed image data.</param>
    /// <param name="classLabels">The candidate class labels.</param>
    /// <returns>A dictionary mapping each label to its probability score.</returns>
    /// <remarks>
    /// <para>
    /// Zero-shot classification means the model can classify images into categories
    /// it has never explicitly been trained on, using natural language descriptions.
    /// </para>
    /// </remarks>
    Dictionary<string, T> ZeroShotClassify(double[] imageData, IEnumerable<string> classLabels);

    /// <summary>
    /// Gets the dimensionality of the embedding space.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Both text and image embeddings will have this same dimension.
    /// Common values are 512 (CLIP ViT-B/32) or 768 (CLIP ViT-L/14).
    /// </para>
    /// </remarks>
    int EmbeddingDimension { get; }

    /// <summary>
    /// Gets the maximum number of tokens for text input.
    /// </summary>
    /// <remarks>
    /// <para>
    /// CLIP models typically have a maximum sequence length of 77 tokens.
    /// Longer texts will be truncated.
    /// </para>
    /// </remarks>
    int MaxSequenceLength { get; }

    /// <summary>
    /// Gets the expected image size (height and width in pixels).
    /// </summary>
    /// <remarks>
    /// <para>
    /// CLIP models expect square images of a specific size (e.g., 224x224).
    /// Images should be preprocessed to this size before encoding.
    /// </para>
    /// </remarks>
    int ImageSize { get; }

    /// <summary>
    /// Gets the supported modalities for this model.
    /// </summary>
    IReadOnlyList<Modality> SupportedModalities { get; }
}

/// <summary>
/// Represents the types of input modalities supported by multimodal models.
/// </summary>
public enum Modality
{
    /// <summary>
    /// Text input (natural language).
    /// </summary>
    Text,

    /// <summary>
    /// Image input (RGB pixels).
    /// </summary>
    Image,

    /// <summary>
    /// Audio input (waveform or spectrogram).
    /// </summary>
    Audio,

    /// <summary>
    /// Video input (sequence of frames).
    /// </summary>
    Video
}
