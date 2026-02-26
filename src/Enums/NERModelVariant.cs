namespace AiDotNet.Enums;

/// <summary>
/// Defines common model size variants for Named Entity Recognition (NER) models.
/// </summary>
/// <remarks>
/// <para>
/// NER models typically come in multiple sizes trading off speed vs accuracy:
/// - Smaller variants (Tiny, Small) run faster with lower memory, suitable for real-time applications
/// - Larger variants (Large, XLarge) produce higher accuracy but require more compute
/// - Base is the default recommended configuration balancing speed and accuracy
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of these like clothing sizes. "Tiny" is the fastest but least
/// accurate, "Base" is the recommended starting point, and "Large"/"XLarge" are for when
/// you need maximum accuracy and have powerful hardware.
/// </para>
/// </remarks>
public enum NERModelVariant
{
    /// <summary>
    /// Tiny variant: minimal size for maximum speed. Lowest accuracy.
    /// </summary>
    Tiny,

    /// <summary>
    /// Small variant: reduced size for faster inference. Good speed/accuracy tradeoff.
    /// </summary>
    Small,

    /// <summary>
    /// Base variant: default recommended configuration. Best balance of speed and accuracy.
    /// </summary>
    Base,

    /// <summary>
    /// Large variant: increased capacity for higher accuracy. Requires more compute.
    /// </summary>
    Large,

    /// <summary>
    /// Extra-large variant: maximum capacity. Best accuracy, slowest inference.
    /// </summary>
    XLarge
}
