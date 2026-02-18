namespace AiDotNet.VisionLanguage.Interfaces;

/// <summary>
/// Interface for vision encoders that extract feature representations from images.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Visual encoders transform raw image data (pixel tensors) into compact feature representations
/// (embeddings) that capture semantic content. These embeddings can be used for:
/// <list type="bullet">
/// <item>Image classification (zero-shot or fine-tuned)</item>
/// <item>Image-text similarity search and retrieval</item>
/// <item>Visual question answering (as input to downstream VLMs)</item>
/// <item>Object detection and segmentation (as backbone features)</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> A visual encoder is like a "feature extractor" for images. It reads an image
/// and produces a list of numbers (an embedding) that summarizes what the image contains. Two similar
/// images will have similar embeddings, making it easy for other models to compare and reason about images.
/// </para>
/// </remarks>
public interface IVisualEncoder<T>
{
    /// <summary>
    /// Extracts a visual embedding from an image tensor.
    /// </summary>
    /// <param name="image">Image tensor in [channels, height, width] or [batch, channels, height, width] format.</param>
    /// <returns>Visual embedding tensor in the model's feature space.</returns>
    Tensor<T> EncodeImage(Tensor<T> image);

    /// <summary>
    /// Gets the dimensionality of the output embedding space.
    /// </summary>
    int EmbeddingDimension { get; }

    /// <summary>
    /// Gets the expected input image size (height and width in pixels).
    /// </summary>
    int ImageSize { get; }

    /// <summary>
    /// Gets the number of image channels expected (typically 3 for RGB).
    /// </summary>
    int ImageChannels { get; }
}
