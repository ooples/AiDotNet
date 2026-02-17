using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Safety.Image;

/// <summary>
/// Interface for image safety classifiers that detect NSFW, violent, or otherwise harmful images.
/// </summary>
/// <remarks>
/// <para>
/// Image safety classifiers analyze image tensors and assign per-category safety scores
/// for categories including sexual content, violence, self-harm, hate symbols, drugs,
/// child exploitation, shocking content, and dangerous activities.
/// </para>
/// <para>
/// <b>For Beginners:</b> An image safety classifier looks at an image and determines
/// whether it contains harmful content. Different implementations use different approaches —
/// CLIP embeddings, Vision Transformers, scene graph analysis, or an ensemble of all three.
/// </para>
/// <para>
/// <b>References:</b>
/// - UnsafeBench: 11 categories, GPT-4V achieves top F1 (2024, arxiv:2405.03486)
/// - USD: Scene-graph-based NSFW detection (USENIX Security 2025)
/// - Sensitive image classification via Vision Transformers (2024, arxiv:2412.16446)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IImageSafetyClassifier<T> : IImageSafetyModule<T>
{
    /// <summary>
    /// Gets per-category safety scores for the given image.
    /// </summary>
    /// <param name="image">The image tensor to classify.</param>
    /// <returns>A dictionary mapping safety category names to scores (0.0 = safe, 1.0 = maximum risk).</returns>
    IReadOnlyDictionary<string, double> GetCategoryScores(Tensor<T> image);
}
