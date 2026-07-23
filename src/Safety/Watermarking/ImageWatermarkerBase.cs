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
    /// Initializes the image watermarker base.
    /// </summary>
    /// <param name="watermarkStrength">Watermark embedding strength. Default: 0.5.</param>
    protected ImageWatermarkerBase(double watermarkStrength = 0.5)
    {
        WatermarkStrength = watermarkStrength;
    }

    /// <inheritdoc />
    public abstract double DetectWatermark(Tensor<T> image);
}
