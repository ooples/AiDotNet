using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety.Image;
using AiDotNet.Safety.Watermarking;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Abstract base class for image watermarking modules.
/// </summary>
/// <remarks>
/// <para>
/// Provides shared infrastructure for image watermarkers including strength
/// configuration and frequency domain utilities. Concrete implementations provide
/// the actual watermarking technique (frequency, neural, invisible spatial).
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class provides common code for all image watermarkers.
/// Each watermarker type extends this and adds its own way of embedding invisible
/// signatures in images.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class ImageWatermarkerBase<T> : ImageSafetyModuleBase<T>, IImageWatermarker<T>
{
    /// <summary>
    /// The watermark strength factor (0.0 to 1.0).
    /// </summary>
    protected readonly double WatermarkStrength;

    /// <summary>
    /// The detection score at or above which <c>EvaluateImage</c> reports a watermark finding (0.0 to 1.0).
    /// </summary>
    protected readonly double DetectionThreshold;

    /// <summary>
    /// Initializes the image watermarker base.
    /// </summary>
    /// <param name="watermarkStrength">Watermark embedding strength. Default: 0.5.</param>
    /// <param name="detectionThreshold">
    /// Score at or above which a watermark finding is reported. Default: 0.3. ImageWatermarker takes the same setting;
    /// these detectors used to hard-code it.
    /// </param>
    protected ImageWatermarkerBase(double watermarkStrength = 0.5, double detectionThreshold = 0.3)
    {
        if (double.IsNaN(detectionThreshold) || detectionThreshold < 0 || detectionThreshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(detectionThreshold),
                "Detection threshold must be between 0 and 1.");
        }

        WatermarkStrength = watermarkStrength;
        DetectionThreshold = detectionThreshold;
    }
    /// <inheritdoc />
    public abstract double DetectWatermark(Tensor<T> image);
}
