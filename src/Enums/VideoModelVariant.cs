namespace AiDotNet.Enums;

/// <summary>
/// Defines common model size variants for video processing models (SR, interpolation, flow, etc.).
/// </summary>
/// <remarks>
/// <para>
/// Video models typically come in multiple sizes trading off speed vs quality:
/// - Smaller variants (Tiny, Small) run faster with lower memory, suitable for real-time applications
/// - Larger variants (Large, XLarge) produce higher quality but require more compute
/// - Base is the default recommended configuration balancing speed and quality
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of these like clothing sizes. "Tiny" is the fastest but least
/// accurate, "Base" is the recommended starting point, and "Large"/"XLarge" are for when
/// you need maximum quality and have powerful hardware.
/// </para>
/// </remarks>
public enum VideoModelVariant
{
    /// <summary>
    /// Tiny variant: minimal size for maximum speed (~30+ FPS). Lowest quality.
    /// </summary>
    Tiny,

    /// <summary>
    /// Small variant: reduced size for faster inference (~20 FPS). Good speed/quality tradeoff.
    /// </summary>
    Small,

    /// <summary>
    /// Base variant: default recommended configuration. Best balance of speed and quality.
    /// </summary>
    Base,

    /// <summary>
    /// Large variant: increased capacity for higher quality. Requires more compute.
    /// </summary>
    Large,

    /// <summary>
    /// Extra-large variant: maximum capacity. Best quality, slowest inference.
    /// </summary>
    XLarge,

    /// <summary>
    /// Pro variant: production-optimized variant with enhanced features.
    /// </summary>
    Pro
}
