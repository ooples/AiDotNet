using AiDotNet.Models;

namespace AiDotNet.Safety.Image;

/// <summary>
/// Abstract base class for image deepfake detection modules.
/// </summary>
/// <remarks>
/// <para>
/// Provides shared infrastructure for deepfake detectors including threshold
/// configuration and common image analysis utilities. Concrete implementations
/// provide the actual detection algorithm (frequency, consistency, provenance).
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class provides common code for all deepfake detectors.
/// Each detector type extends this and adds its own way of detecting AI-generated
/// or manipulated images.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class DeepfakeDetectorBase<T> : ImageSafetyModuleBase<T>, IDeepfakeDetector<T>
{
    /// <summary>
    /// The detection threshold above which images are flagged as deepfakes.
    /// </summary>
    protected readonly double Threshold;

    /// <summary>
    /// Initializes the deepfake detector base with a threshold.
    /// </summary>
    /// <param name="threshold">The detection threshold (0.0 to 1.0). Default: 0.5.</param>
    protected DeepfakeDetectorBase(double threshold = 0.5)
    {
        if (threshold < 0 || threshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold),
                "Threshold must be between 0 and 1.");
        }

        Threshold = threshold;
    }

    /// <inheritdoc />
    public abstract double GetDeepfakeScore(Tensor<T> image);
}
