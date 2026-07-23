namespace AiDotNet.Preprocessing.Document;

/// <summary>
/// Layout normalization strategies.
/// </summary>
public enum NormalizationStrategy
{
    /// <summary>
    /// Stretch to fit target dimensions (may distort aspect ratio).
    /// </summary>
    Stretch,

    /// <summary>
    /// Resize preserving aspect ratio and add padding.
    /// </summary>
    ResizeWithPadding,

    /// <summary>
    /// Crop the center of the image.
    /// </summary>
    CenterCrop,

    /// <summary>
    /// Resize to cover target and crop to exact size.
    /// </summary>
    ResizeAndCrop
}
