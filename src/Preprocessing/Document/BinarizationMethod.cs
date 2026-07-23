namespace AiDotNet.Preprocessing.Document;

/// <summary>
/// Binarization methods for document images.
/// </summary>
public enum BinarizationMethod
{
    /// <summary>
    /// Global Otsu thresholding.
    /// </summary>
    Otsu,

    /// <summary>
    /// Sauvola local adaptive thresholding.
    /// </summary>
    Sauvola,

    /// <summary>
    /// Niblack local adaptive thresholding.
    /// </summary>
    Niblack,

    /// <summary>
    /// Simple fixed threshold.
    /// </summary>
    Fixed
}
