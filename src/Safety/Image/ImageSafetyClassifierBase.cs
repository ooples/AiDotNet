using AiDotNet.Models;
using AiDotNet.Safety.Image;

namespace AiDotNet.Safety.Image;

/// <summary>
/// Abstract base class for image safety classifiers.
/// </summary>
/// <remarks>
/// <para>
/// Provides shared infrastructure for image safety classifiers including threshold
/// configuration and common image feature extraction utilities. Concrete implementations
/// provide the actual classification algorithm (CLIP, ViT, scene graph, ensemble).
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class provides common code for all image safety
/// classifiers. Each classifier type extends this and adds its own way of detecting
/// harmful content in images.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class ImageSafetyClassifierBase<T> : ImageSafetyModuleBase<T>, IImageSafetyClassifier<T>
{
    /// <summary>
    /// The safety threshold above which content is flagged.
    /// </summary>
    protected readonly double Threshold;

    /// <summary>
    /// Initializes the image safety classifier base with a threshold.
    /// </summary>
    /// <param name="threshold">The detection threshold (0.0 to 1.0). Default: 0.5.</param>
    protected ImageSafetyClassifierBase(double threshold = 0.5)
    {
        Threshold = threshold;
    }

    /// <inheritdoc />
    public abstract IReadOnlyDictionary<string, double> GetCategoryScores(Tensor<T> image);
}
