using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Safety.Image;

/// <summary>
/// Interface for deepfake and AI-generated image detection modules.
/// </summary>
/// <remarks>
/// <para>
/// Deepfake detectors analyze images for signs of AI generation or manipulation,
/// including frequency domain artifacts, facial/spatial inconsistencies, and
/// metadata/watermark provenance analysis.
/// </para>
/// <para>
/// <b>For Beginners:</b> A deepfake detector checks if an image is real or fake.
/// It looks for invisible signs of AI manipulation — patterns in the image frequencies,
/// inconsistencies in faces or backgrounds, and metadata clues that indicate the image
/// was generated or altered by AI.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IDeepfakeDetector<T> : IImageSafetyModule<T>
{
    /// <summary>
    /// Gets the deepfake probability score for the given image (0.0 = authentic, 1.0 = fake).
    /// </summary>
    /// <param name="image">The image tensor to evaluate.</param>
    /// <returns>A deepfake probability score between 0.0 and 1.0.</returns>
    double GetDeepfakeScore(Tensor<T> image);
}
