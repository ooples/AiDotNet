namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for X-Decoder.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> X-Decoder is a generalist vision decoder that handles referring segmentation,
/// open-vocabulary segmentation, and image captioning in a single unified model. It decodes both
/// pixel-level masks and text tokens using one shared architecture.
/// </para>
/// <para>
/// <b>Technical Details:</b> Uses a two-path decoder: one for pixel-level predictions (masks) and
/// one for token-level predictions (text). Both paths share the same attention mechanism. Supports
/// various segmentation and vision-language tasks without task-specific heads.
/// </para>
/// <para>
/// <b>Reference:</b> Zou et al., "Generalized Decoding for Pixel, Image, and Language", CVPR 2023.
/// </para>
/// </remarks>
public enum XDecoderModelSize
{
    /// <summary>
    /// Tiny backbone (~30M params). Fast inference for deployment.
    /// </summary>
    Tiny,

    /// <summary>
    /// Base backbone (~86M params). Standard configuration.
    /// </summary>
    Base,

    /// <summary>
    /// Large backbone (~307M params). Maximum accuracy for complex queries.
    /// </summary>
    Large
}
