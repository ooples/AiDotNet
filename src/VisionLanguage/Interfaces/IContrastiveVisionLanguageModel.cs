namespace AiDotNet.VisionLanguage.Interfaces;

/// <summary>
/// Interface for contrastive vision-language models that align image and text embeddings
/// in a shared space for zero-shot classification and cross-modal retrieval.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Contrastive VLMs (like CLIP, SigLIP, ALIGN) learn to place matching image-text pairs close
/// together in a shared embedding space while pushing non-matching pairs apart. This enables:
/// <list type="bullet">
/// <item><b>Zero-shot classification</b>: Classify images using text descriptions without training</item>
/// <item><b>Image-text retrieval</b>: Find images matching a text query, or texts describing an image</item>
/// <item><b>Similarity scoring</b>: Measure how well an image matches a text description</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this like a universal translator between images and text.
/// The model learns that a photo of a dog and the text "a photo of a dog" should have similar
/// representations. This means you can search for images using text, or classify images by
/// comparing them with text descriptions - without training on specific categories.
/// </para>
/// </remarks>
public interface IContrastiveVisionLanguageModel<T> : IVisualEncoder<T>, ITextEncoder<T>
{
    /// <summary>
    /// Computes the similarity score between an image and a text description.
    /// </summary>
    /// <param name="image">Image tensor in [channels, height, width] format.</param>
    /// <param name="text">Text description to compare with.</param>
    /// <returns>Similarity score (higher = more similar).</returns>
    T ComputeSimilarity(Tensor<T> image, string text);

    /// <summary>
    /// Performs zero-shot image classification by comparing the image with text labels.
    /// </summary>
    /// <param name="image">Image tensor to classify.</param>
    /// <param name="labels">Text descriptions of possible categories.</param>
    /// <returns>Dictionary mapping each label to its probability.</returns>
    Dictionary<string, T> ZeroShotClassify(Tensor<T> image, string[] labels);

    /// <summary>
    /// Gets the dimensionality of the shared projection space.
    /// </summary>
    int ProjectionDimension { get; }

    /// <summary>
    /// Gets the temperature parameter used for similarity scaling.
    /// </summary>
    T Temperature { get; }
}
