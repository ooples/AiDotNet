namespace AiDotNet.VisionLanguage.Interfaces;

/// <summary>
/// Interface for vision-language fusion models that combine image and text features
/// for tasks like VQA, image-text matching, and cross-modal retrieval.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Unlike contrastive models that keep vision and text separate, fusion models combine
/// modalities internally via co-attention, cross-attention, or single-stream architectures.
/// They support tasks including:
/// <list type="bullet">
/// <item>Visual Question Answering (VQA) - answering questions about images</item>
/// <item>Image-Text Matching (ITM) - determining if image and text match</item>
/// <item>Cross-Modal Retrieval - finding relevant images for text or vice versa</item>
/// <item>Visual Entailment - reasoning about image-text relationships</item>
/// </list>
/// </para>
/// </remarks>
public interface IVisionLanguageFusionModel<T> : IVisualEncoder<T>
{
    /// <summary>
    /// Computes a fused representation of an image-text pair.
    /// </summary>
    /// <param name="image">Image tensor.</param>
    /// <param name="text">Text input string.</param>
    /// <returns>Fused multimodal embedding tensor.</returns>
    Tensor<T> FuseImageText(Tensor<T> image, string text);

    /// <summary>
    /// Computes an image-text matching score indicating how well the pair matches.
    /// </summary>
    /// <param name="image">Image tensor.</param>
    /// <param name="text">Text input string.</param>
    /// <returns>Matching score (higher = better match).</returns>
    T ComputeMatchingScore(Tensor<T> image, string text);

    /// <summary>
    /// Gets the fusion embedding dimension.
    /// </summary>
    int FusionEmbeddingDim { get; }

    /// <summary>
    /// Gets the maximum text sequence length.
    /// </summary>
    int MaxSequenceLength { get; }
}
