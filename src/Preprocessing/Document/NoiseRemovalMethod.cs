namespace AiDotNet.Preprocessing.Document;

/// <summary>
/// Noise removal methods for document images.
/// </summary>
public enum NoiseRemovalMethod
{
    /// <summary>
    /// Median filtering.
    /// </summary>
    Median,

    /// <summary>
    /// Gaussian blurring.
    /// </summary>
    Gaussian,

    /// <summary>
    /// Bilateral filtering.
    /// </summary>
    Bilateral,

    /// <summary>
    /// Morphological opening.
    /// </summary>
    Morphological
}
