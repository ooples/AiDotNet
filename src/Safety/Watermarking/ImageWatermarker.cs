using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// The safety pipeline's image watermarker: the keyed spread-spectrum DCT watermark of
/// <see cref="InvisibleImageWatermarker{T}"/> with a configurable reporting threshold.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="ImageWatermarkerBase{T}.EmbedWatermark"/> marks images the application produces;
/// <see cref="ImageWatermarkerBase{T}.EvaluateImage"/> reports a finding when an image carries the mark of this
/// watermarker's key with at least the configured confidence. A watermarker only recognises marks made with its own
/// key, so the pipeline detects the application's own watermark, not other vendors' (no single detector can).
/// </para>
/// <para>
/// <b>For Beginners:</b> image watermarking hides an invisible, password-protected "stamp" inside a picture, so you
/// can later prove the picture came from your system. You cannot see the stamp, but the watermarker that made it
/// can find it again, even after mild noise or re-saving with 8-bit precision.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Secure Spread Spectrum Watermarking for Multimedia",
    "https://doi.org/10.1109/83.650120",
    Year = 1997,
    Authors = "Ingemar J. Cox, Joe Kilian, F. Thomson Leighton, Talal Shamoon")]
public class ImageWatermarker<T> : InvisibleImageWatermarker<T>
{
    private readonly double _detectionThreshold;

    /// <inheritdoc />
    public override string ModuleName => "ImageWatermarker";

    /// <summary>
    /// Initializes a new image watermarker.
    /// </summary>
    /// <param name="detectionThreshold">
    /// The detection confidence (0-1) from which <see cref="ImageWatermarkerBase{T}.EvaluateImage"/> reports a
    /// finding; 0 reports any evidence (confidence above 0). Default: 0.5 (a correlation z-score of 4.5).
    /// </param>
    /// <param name="watermarkStrength">
    /// Embedding amplitude relative to the spread of the DCT coefficients. Higher values are more robust but less
    /// invisible. Default: 0.5.
    /// </param>
    /// <param name="key">
    /// The secret key the watermark is derived from. <c>null</c> uses the public
    /// <see cref="ImageWatermarkerBase{T}.PublicKey"/>, whose marks anyone can forge.
    /// </param>
    public ImageWatermarker(double detectionThreshold = 0.5, double watermarkStrength = 0.5, string? key = null)
        : base(watermarkStrength, key)
    {
        if (double.IsNaN(detectionThreshold) || detectionThreshold < 0 || detectionThreshold > 1)
            throw new ArgumentOutOfRangeException(nameof(detectionThreshold), "Detection threshold must be between 0 and 1.");
        _detectionThreshold = detectionThreshold;
    }

    /// <inheritdoc />
    protected override double ReportThreshold => _detectionThreshold;
}
