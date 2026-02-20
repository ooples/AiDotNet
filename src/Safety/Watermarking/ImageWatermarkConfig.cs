namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Configuration for image watermarking modules.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Use this to configure image watermarking including the
/// embedding strength and technique.
/// </para>
/// </remarks>
public class ImageWatermarkConfig
{
    /// <summary>Watermark embedding strength (0.0-1.0). Default: 0.5.</summary>
    public double? Strength { get; set; }

    /// <summary>Watermarking technique to use. Default: Frequency.</summary>
    public ImageWatermarkType? Technique { get; set; }

    internal double EffectiveStrength => Strength ?? 0.5;
    internal ImageWatermarkType EffectiveTechnique => Technique ?? ImageWatermarkType.Frequency;
}

/// <summary>
/// The type of image watermarking technique to use.
/// </summary>
public enum ImageWatermarkType
{
    /// <summary>DCT/DWT frequency domain embedding.</summary>
    Frequency,
    /// <summary>Encoder-decoder neural watermark.</summary>
    Neural,
    /// <summary>Imperceptible spatial domain watermark.</summary>
    Invisible
}
